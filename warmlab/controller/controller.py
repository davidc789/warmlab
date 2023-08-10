""" The simulation master controller. """

import json
import logging
import asyncio
import contextlib
import socket
import sys
from collections import defaultdict
from typing import Optional, NamedTuple
from tqdm.asyncio import tqdm

import aiosqlite
import aiohttp
import pandas as pd

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


def dict_factory(cursor: aiosqlite.Cursor, row: aiosqlite.Row):
    """ Generates a dictionary for query results.

    :param cursor: The sqlite cursor.
    :param row: The native row object.
    :return: Dictionary representation of the row object.
    """
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


class DataManager(object):
    """ Automating data requests. """
    _is_open: bool
    _is_readonly: bool
    _use_fast: bool
    _database_path: str
    _use_dict_factory: bool
    _commit_on_exit: bool

    # The underlying connection
    _conn: aiosqlite.Connection

    def __init__(self,
                 db_path: str,
                 is_readonly: bool = True,
                 use_fast: bool = False,
                 use_dict_factory: bool = False,
                 commit_on_exit: bool = False) -> None:
        """ Constructs the data manager.

        :param is_readonly: Whether the database is read-only.
        :param use_fast: Whether to use fast mode. Warning: using this makes it
        thread-unsafe and may result in a corrupted database.
        """
        self._is_open = False
        self._is_readonly = is_readonly
        self._use_fast = use_fast
        self._database_path = db_path
        self._use_dict_factory = use_dict_factory
        self._commit_on_exit = commit_on_exit

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

        self._conn = await aiosqlite.connect(self._database_path)
        self._is_open = True

        if self._use_dict_factory:
            self._conn.row_factory = dict_factory

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

        if self._commit_on_exit:
            await self._conn.commit()

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
    t: int                             # Target end time.
    n: int                             # Target number of simulations.
    simId: str                         # The simulation ID.
    model: warm.WarmModel              # The underlying WARM model.
    workersData: list[WorkerData]      # The list of workers available.
    db_path: str = config.DB_LOCATION  # Path to the database.


def simulation_order(sim: warm.WarmSimulationData):
    """ Specifies how simulations should be ordered.

    :param sim: The simulation data.
    :return: The (simulationTime, trialId) pair.
    """
    return sim.t, sim.trialId


async def create_siminfo_if_missing(db: aiosqlite.Connection):
    """ Creates the metadata table if it is missing.

    :param db: The database connection.
    """
    async with db.cursor() as cursor:
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS SimInfo (
                simId            TEXT PRIMARY KEY NOT NULL,
                description      TEXT DEFAULT NULL,
                model            TEXT NOT NULL,
                solution         TEXT DEFAULT NULL,
                completedTrials  INT NOT NULL DEFAULT 0,
                maxTime          INT NOT NULL DEFAULT 0
            )
        """)
        await db.commit()


async def create_simdata_if_missing(db: aiosqlite.Connection):
    """ Creates the simulation data table if it is missing.

    :param db: The database connection.
    """
    async with db.cursor() as cursor:
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS SimData (
                trialId     INT  NOT NULL,
                endTime     INT  NOT NULL,
                simId       TEXT NOT NULL,
                root        TEXT DEFAULT '0-0',
                counts      TEXT NOT NULL,
                PRIMARY KEY (trialId, endTime),
                FOREIGN KEY (simId) REFERENCES SimInfo(simId)
            )
        """)
        await db.commit()


async def worker(
        session: aiohttp.ClientSession,
        pending_simulation: asyncio.Queue[Optional[tuple[int, int, warm.WarmSimulationData]]],
        pending_storage: asyncio.Queue[warm.WarmSimulationData]
):
    """
    Creates a simulation worker coroutine.

    Simulation is done remotely on the server.

    :param session: The HTTP session the worker is bind to.
    :param pending_simulation: The queue used for inputting simulation jobs.
    :param pending_storage: The queue used for storing completed jobs.
    """
    while True:
        sim_tuple = await pending_simulation.get()

        # If the termination signal is received, stops working.
        if sim_tuple is None:
            return

        _, _, sim = sim_tuple

        response = await session.post(f"/api/simulate", data=sim.to_json())
        response.raise_for_status()
        text = await response.text()
        res = warm.WarmSimulationData.from_json(text)
        pending_simulation.task_done()
        await pending_storage.put(res)


async def prepare_simulation(
        db: aiosqlite.Connection,
        pending_simulation: asyncio.Queue[Optional[tuple[int, int, warm.WarmSimulationData]]],
        target: Target
):
    """ Prepares the simulation.

    :param db: The database connection.
    :param pending_simulation: The queue of entries pending simulation.
    :param target: The simulation target.
    """
    # Ensures the presence of data tables required.
    await create_siminfo_if_missing(db)
    await create_simdata_if_missing(db)

    async with db.cursor() as cursor:
        # Obtains relevant information regarding the simulation of interest.
        # First query the master data.
        await cursor.execute(f"""
                SELECT simId, completedTrials, maxTime FROM SimInfo 
                WHERE simId = '{target.simId}'
            """)
        info_row = await cursor.fetchone()

        # Then query detailed sim data.
        await cursor.execute(f"""
                SELECT trialId, max(endTime) AS endTime, counts FROM SimData
                WHERE SimID = '{target.simId}'
                GROUP BY trialId
                ORDER BY trialId
            """)
        data_rows = list(await cursor.fetchall())

    completed_count = 0  # Tracks the number of completed jobs.
    latest_sims: list[warm.WarmSimulationData]  # Tracks the latest simulation trials.

    if data_rows is None:
        # No data available about this simulation. Start from scratch.
        latest_sims = [warm.WarmSimulationData(
            model=target.model,
            root="0.0",
            counts=None,
            targetTime=config.TIME_STEP,
            t=0,
            trialId=i
        ) for i in range(target.n)]
        print(" - No prior simulation found. Starting from scratch.")
    else:
        # Use the latest entries.
        existing_latest_sims = [warm.WarmSimulationData(
            model=target.model,
            root="0.0",
            counts=json.loads(data_row["counts"]),
            targetTime=data_row["endTime"] + config.TIME_STEP,
            t=data_row["endTime"],
            trialId=data_row["trialId"]
        ) for data_row in data_rows]
        latest_sims = existing_latest_sims + [warm.WarmSimulationData(
            model=target.model,
            root="0.0",
            counts=None,
            targetTime=config.TIME_STEP,
            t=0,
            trialId=i
        ) for i in range(len(data_rows), target.n)]
        print(f" - {len(data_rows)} previous simulations found. Using those as well")

    # Ensure the database know about this simulation.
    if info_row is None:
        async with db.cursor() as cursor:
            await cursor.execute(f"""
                INSERT INTO SimInfo (simID, model) 
                VALUES ('{target.simId}', '{target.model.to_json()}')
            """)
            await db.commit()

    # Place a few initial simulations and count completed simulations.
    for sim in latest_sims:
        if sim.t < target.t:
            await pending_simulation.put((sim.t, sim.trialId, sim))
        else:
            completed_count += 1

    return completed_count


async def simulation_manager(
        db: aiosqlite.Connection,
        pending_simulation: asyncio.Queue[Optional[tuple[int, int, warm.WarmSimulationData]]],
        pending_storage: asyncio.Queue[warm.WarmSimulationData],
        target: Target, completed_count: int, pbar: Optional[tqdm] = None):
    """ Writes the results into the database.

    :param db: Database connection.
    :param pending_simulation: Queue for tracking items to be simulated.
    :param pending_storage: Results pending storage.
    :param target: The simulation target.
    """
    while completed_count < target.n:
        # Fetches a simulation result and stores it in the database.
        res = await pending_storage.get()

        # Creates a new entry in the data table.
        cursor: aiosqlite.Cursor
        async with db.cursor() as cursor:
            await cursor.execute(f"""
                INSERT INTO SimData (trialId, endTime, simId, root, counts)
                VALUES ({res.trialId}, {res.t}, '{target.simId}', '{res.root}', '{json.dumps(res.counts)}')
            """)
            await db.commit()
        print(f" - Trial {res.trialId} time {res.t} is written to the database")

        # Adds the next simulation item into the queue.
        if res.t < target.t:
            await pending_simulation.put((res.t, res.trialId, warm.WarmSimulationData(
                model=res.model,
                root=res.root,
                counts=res.counts,
                targetTime=res.t + config.TIME_STEP,
                t=res.t,
                trialId=res.trialId
            )))
        else:
            completed_count += 1
            print(f" - Trial {res.trialId} is completed")

        pending_storage.task_done()

    # When all the trials are completed, inform the workers and finalise
    # the database entry.
    for _ in target.workersData:
        await pending_simulation.put(None)

    async with db.cursor() as cursor:
        await cursor.execute(f"""
            REPLACE INTO SimInfo (simId, model, completedTrials, maxTime)
            VALUES ('{target.simId}', '{target.model.to_json()}', {target.n}, {target.t})
        """)
        await db.commit()


async def simulation_controller(
        target: Target, use_progress_bar: bool = True):
    """ Controls the simulation.

    :param target: The simulation target.
    """
    pending_simulation = asyncio.PriorityQueue()  # Items waiting to be simulated
    pending_storage = asyncio.Queue()             # Items pending storage

    async with contextlib.AsyncExitStack() as stack:
        cursor: aiosqlite.Cursor

        # pbar = None
        # if use_progress_bar:
        #     pbar = stack.enter_async_context(tqdm(total=100))

        # Establish db connection and check the current simulation status.
        db = await stack.enter_async_context(DataManager(
            target.db_path, is_readonly=False, use_dict_factory=True))

        # Creates sessions to communicate with workers.
        session_pool = await asyncio.gather(*[stack.enter_async_context(
            aiohttp.ClientSession(x.url)) for x in target.workersData])

        # Prepare the simulation up.
        completed_count = await prepare_simulation(db, pending_simulation, target)

        # Keeps track of the tasks created.
        worker_tasks: list[asyncio.Task[None]] = []

        try:
            # Spawns the worker tasks and runs in concurrency.
            for session in session_pool:
                task = asyncio.create_task(
                    worker(session, pending_simulation, pending_storage))
                worker_tasks.append(task)
            manager_task = asyncio.create_task(simulation_manager(
                db, pending_simulation, pending_storage, target, completed_count))

            # Await for task completion.
            for task in worker_tasks:
                await task
            await manager_task
        except asyncio.CancelledError:
            # Propagates the kill switch
            for task in worker_tasks:
                task.cancel()
            manager_task.cancel()


async def main(args: Optional[list[str]] = None):
    """ The main coroutine serving as the entry point of the program.
    """
    if args is None:
        args = sys.argv

    # Creates the main task.
    graph = warm.ring_2d_graph(3)
    task = asyncio.create_task(simulation_controller(Target(
        t=1_000_000,
        n=10,
        simId=f"ring_2d_3",
        model=warm.WarmModel(is_graph=True, graph=graph),
        workersData=[WorkerData("http://localhost:7071")]
    )))

    try:
        await task
    except KeyboardInterrupt:
        task.cancel()


if __name__ == '__main__':
    asyncio.run(main(), debug=True)
