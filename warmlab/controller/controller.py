""" The simulation master controller. """

import json
import logging
import asyncio
import contextlib
import socket
from collections import defaultdict
from typing import Optional, NamedTuple
import igraph as ig

import aiosqlite
import aiohttp

from config import config
from warmlab.warm import warm

logger = logging.getLogger(__name__)

# Data type and constraints for magn and sound.
_dtype_graph_info: defaultdict[str, str] = defaultdict(lambda: "REAL", {
    "graphID": "INTEGER PRIMARY KEY",
})

# Data type and constraints for eqlst.
_dtype_sim = {
    "trial": "INTEGER",
    "t": "INTEGER",
    "root": "STRING",
    "data": "BLOB"
}


class DataManager(object):
    """ Automating data requests. """
    _is_open: bool
    _is_readonly: bool
    _use_fast: bool
    _database_path: str

    # The underlying connection
    _conn: aiosqlite.Connection

    def __init__(self,
                 db_path: str,
                 is_readonly: bool = True,
                 use_fast: bool = False) -> None:
        """ Constructs the data manager.

        :param is_readonly: Whether the database is read-only.
        :param use_fast: Whether to use fast mode. Warning: using this makes it
        thread-unsafe and may result in a corrupted database.
        """
        self._is_open = False
        self._is_readonly = is_readonly
        self._use_fast = use_fast
        self._database_path = db_path

    @property
    def is_open(self) -> bool:
        """ Whether the database connection is opened.

        :return: Whether the database connection is opened.
        """
        return self._is_open

    @property
    def is_readonly(self) -> bool:
        """ Whether the database connection is read-only.

        :return: Whether the database connection is read-only.
        """
        return self._is_readonly

    async def setup(self) -> None:
        """ Sets up the database for the first time. """
        await self.open()
        await self.close()

    async def enable_edit(self) -> None:
        """ Enables database editing. """
        await self._conn.execute("PRAGMA QUERY_ONLY = FALSE;")

    async def disable_edit(self) -> None:
        """ Disables database editing, making it read-only. """
        await self._conn.execute("PRAGMA QUERY_ONLY = TRUE;")

    async def open(self) -> aiosqlite.Connection:
        """ Opens the database connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if self._is_open:
            raise ConnectionError("The connection has already opened.")

        self._conn = aiosqlite.connect(self._database_path)
        self._is_open = True
        await self._conn.execute("PRAGMA foreign_keys = ON")

        if self._use_fast:
            await self._conn.execute("PRAGMA synchronous = OFF")
            await self._conn.execute("PRAGMA journal_mode = MEMORY")

        logger.info("Database connection established.")
        return self._conn

    async def close(self) -> None:
        """ Closes the database connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if not self._is_open:
            raise ConnectionError("The connection has already closed.")

        await self._conn.close()
        self._is_open = False
        logger.info("Database connection closed.")

    async def __aenter__(self) -> aiosqlite.Connection:
        """ Executes when the context manager enters. """
        return await self.open()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ Executes when the context manager exits. """
        await self.close()

        if exc_val is not None:
            logger.error(f"Upon DataManger exit: {exc_val}")


async def cloud_worker(session: aiohttp.ClientSession, data: str):
    """ A remote worker communicating through HTTP.

    :param data: Serialised data.
    """
    async with session.post('/post', data=data):
        pass


def client():
    host = '10.0.0.2'  # client ip
    port = 4005

    server = ('192.168.0.12', 4000)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))

    message = input("-> ")
    while message != 'q':
        s.sendto(message.encode('utf-8'), server)
        data, addr = s.recvfrom(1024)
        data = data.decode('utf-8')
        print("Received from server: " + data)
        message = input("-> ")
    s.close()


class WorkerData(NamedTuple):
    url: str


class Target(NamedTuple):
    t: int
    n: int
    simId: str
    model: warm.WarmModel
    workersData: list[WorkerData]
    db_path: str = config.DB_LOCATION


async def worker(
        session: aiohttp.ClientSession,
        pending_simulation: asyncio.Queue[warm.WarmSimulationData],
        pending_storage: asyncio.Queue[warm.WarmSimulationData]
):
    """
    Creates a simulation worker coroutine.

    Simulation is done remotely on the server.

    :param session: The HTTP session the worker is bind to.
    :param pending_simulation: The queue used for inputting simulation jobs.
    :param pending_storage: The queue used for storing completed jobs.
    """
    sim = await pending_simulation.get()

    # If the termination signal is received, stops working.
    if sim is None:
        return

    response = await session.post('/post', data=sim)
    res = await response.json(loads=warm.WarmSimulationData.from_json)
    pending_simulation.task_done()
    await pending_storage.put(res)


async def db_writer(conn: aiosqlite.Connection,
                    pending_simulation: asyncio.Queue[warm.WarmSimulationData],
                    pending_storage: asyncio.Queue[warm.WarmSimulationData],
                    target: Target):
    """ Writes the results into the database.

    :param conn: Database connection.
    :param pending_simulation: Queue for tracking items to be simulated.
    :param pending_storage: Results pending storage.
    :param target: The simulation target.
    """
    res = await pending_storage.get()

    cursor: aiosqlite.Cursor
    async with conn.cursor() as cursor:
        # Creates a new entry
        await cursor.execute("""
            INSERT INTO SimData (trialId, time, simId, data)
            VALUES (?, ?, ?, ?);
        """, [res.targetTime, res.t, target.simId, res.counts])

        # Adds the next simulation item into the queue.
        if res.t < target.t:
            await pending_simulation.put(warm.WarmSimulationData(
                model=res.model,
                root=res.root,
                targetTime=res.targetTime,
                t=res.t + config.TIME_STEP,
                counts=res.counts
            ))

    pending_storage.task_done()


async def simulation_controller(target: Optional[Target]):
    """ Controls the simulation.

    :param model: The model.
    :param simId: The unique id associated with the simulation. The convention
    to follow is '[graphNickname]-[nodeProbabilityPattern]'.
    :param target: The simulation target.
    :param continuous: Whether to continuously simulate.
    :param db: Location of the database.
    """
    pending_simulation = asyncio.PriorityQueue()
    pending_storage = asyncio.Queue()

    async with contextlib.AsyncExitStack() as stack:
        cursor: aiosqlite.Cursor
        db_sim_status: list[int]    # The latest time of simulation i.

        # Establish db connection and check the current simulation status.
        db = await stack.enter_async_context(DataManager(target.db_path, is_readonly=False))
        async with db.cursor() as cursor:
            # Creates the metadata table if it is missing.
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS SimInfo (
                    simID      TEXT PRIMARY KEY NOT NULL,
                    model      TEXT NOT NULL,
                    solution   TEXT DEFAULT NULL,
                    simCounts  INT NOT NULL DEFAULT 0,
                    simTime    INT NOT NULL DEFAULT 0,
                    isComplete INT NOT NULL DEFAULT 0
                )
            """)

            # Obtains relevant information regarding the simulation of interest.
            await cursor.execute("""
                SELECT * FROM SimInfo WHERE simID = ?
            """, [target.simId])
            info_row = await cursor.fetchone()

            if info_row is None:
                # No data available. Creates a new entry for it.
                await cursor.execute("""
                    INSERT INTO SimInfo (simID, model, isComplete) VALUES ?, ?, 1
                """, [target.simId, target.model])

                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS SimData (
                        trialId    INT  NOT NULL,
                        endTime    INT  NOT NULL,
                        simID      TEXT NOT NULL,
                        root       TEXT DEFAULT ('0-0'),
                        counts     TEXT NOT NULL,
                        FOREIGN KEY (simId) REFERENCES SimInfo (simId)
                    )
                """, [target.simId])

                db_sim_status = [0 for _ in range(target.n)]
            elif info_row["isComplete"] == 0:
                # Adjusts the target to complete the previously unfinished job.
                # Fetches the current status.
                target.n = info_row["simCounts"]
                target.t = info_row["simTime"]
                logger.warning(f"A previously incomplete target is found and used instead: "
                               f"n = {target.n}, t = {target.t}")

                # Fetches the previous progress.
                await cursor.execute("""
                    SELECT
                        max(endTime), counts
                    FROM SimData WHERE SimID = ?
                    GROUP BY trialId
                    ORDER BY trialId
                """, [target.simId])

                data_rows = list(await cursor.fetchall())
                db_sim_status = [data_row["endTime"] for data_row in data_rows]
                db_sim_counts = [json.loads(data_row["counts"]) for data_row in data_rows]
            else:
                # Starts the simulation at the previous time, incremented.
                db_sim_status = [info_row["simTime"] for _ in range(info_row["simCounts"])]

        # Creates sessions to communicate with workers.
        session_pool: list[aiohttp.ClientSession] = [
            stack.enter_context(aiohttp.ClientSession(x.url))
            for x in target.workersData]

        # Places a few initial simulations
        for t, counts in zip(db_sim_status, db_sim_counts):
            if t < target.t:
                await pending_simulation.put(warm.WarmSimulationData(
                    model=target.model,
                    root="0.0",
                    counts=counts,
                    targetTime=target.n,
                    t=t + config.TIME_STEP
                ))

        # Spawns the worker tasks and runs in concurrency.
        workers = [worker(session, pending_simulation, pending_storage)
                   for session in session_pool]
        await asyncio.gather(*workers, db_writer(
            db, pending_simulation, pending_storage, target))

if __name__ == '__main__':
    graph = ring_2d_graph(2)
    coroutine = simulation_controller(Target(
        t=100_000_000,
        n=2,
        simId=f"ring_2d_2",
        model=warm.WarmModel(is_graph=True, graph=graph),
        workersData=[WorkerData("http://localhost:7071/api/hello")]
    ))
    asyncio.run(coroutine)
