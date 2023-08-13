""" The simulation master controller. """

import json
import logging
import asyncio
import contextlib
import sqlite3

from typing import Optional, NamedTuple, Literal

import aiosqlite
import aiohttp

from tqdm import tqdm

from config import config
from warmlab.warm import warm

logger = logging.getLogger(__name__)


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
    """ Manages database connection. """
    _is_open: bool
    _is_readonly: bool
    _use_fast: bool
    _database_path: str
    _use_dict_factory: bool
    _commit_on_exit: bool

    # The underlying connection
    _async_conn: aiosqlite.Connection
    _conn: sqlite3.Connection

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
        await self.async_open()
        await self.async_close()

    async def enable_edit(self) -> None:
        """ Enables database editing. """
        await self._async_conn.execute("PRAGMA QUERY_ONLY = FALSE;")

    async def disable_edit(self) -> None:
        """ Disables database editing, making it read-only. """
        await self._async_conn.execute("PRAGMA QUERY_ONLY = TRUE;")

    def open(self) -> sqlite3.Connection:
        """ Opens the database connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if self._is_open:
            raise ConnectionError("The connection has already opened.")

        self._conn = sqlite3.connect(self._database_path)
        self._is_open = True

        if self._use_dict_factory:
            self._conn.row_factory = dict_factory

        self._conn.execute("PRAGMA foreign_keys = ON")

        if self._use_fast:
            self._conn.execute("PRAGMA synchronous = OFF")
            self._conn.execute("PRAGMA journal_mode = MEMORY")

        logger.info("Database connection established.")
        return self._conn

    async def async_open(self) -> aiosqlite.Connection:
        """ Opens the database connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if self._is_open:
            raise ConnectionError("The connection has already opened.")

        self._async_conn = await aiosqlite.connect(self._database_path)
        self._is_open = True

        if self._use_dict_factory:
            self._async_conn.row_factory = dict_factory

        await self._async_conn.execute("PRAGMA foreign_keys = ON")

        if self._use_fast:
            await self._async_conn.execute("PRAGMA synchronous = OFF")
            await self._async_conn.execute("PRAGMA journal_mode = MEMORY")

        logger.info("Database connection established.")
        return self._async_conn

    def close(self):
        """ Closes the database connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if not self._is_open:
            raise ConnectionError("The connection has already closed.")

        if self._commit_on_exit:
            self._conn.commit()

        self._conn.close()
        self._is_open = False
        logger.info("Database connection closed.")

    async def async_close(self) -> None:
        """ Closes the database connection.

        The context management syntax is recommended:
        `with DataManager() as data_manger: ...`
        """
        if not self._is_open:
            raise ConnectionError("The connection has already closed.")

        if self._commit_on_exit:
            await self._async_conn.commit()

        await self._async_conn.close()
        self._is_open = False
        logger.info("Database connection closed.")

    def __enter__(self) -> sqlite3.Connection:
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self) -> aiosqlite.Connection:
        """ Executes when the context manager enters. """
        return await self.async_open()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ Executes when the context manager exits. """
        await self.async_close()

        if exc_val is not None:
            logger.error(f"Upon DataManger exit: {exc_val}")


class WorkerData(NamedTuple):
    url: str


class Target(NamedTuple):
    t: int                             # Target end time.
    n: int                             # Target number of simulations.
    simId: str                         # The simulation ID.
    model: warm.WarmModel              # The underlying WARM model.


class SimulationContext(NamedTuple):
    """ Global context for the simulation controller. """
    target: Target
    pending_simulation: asyncio.Queue[Optional[tuple[int, int, warm.WarmSimulationData]]]
    pending_storage: asyncio.Queue[warm.WarmSimulationData]
    db: aiosqlite.Connection
    sessions: list[aiohttp.ClientSession]
    pbar_completed: Optional[tqdm] = None
    pbar_full: Optional[tqdm] = None


def simulation_order(sim: dict[str, any]):
    """ Specifies how simulations should be ordered.

    :param sim: The simulation data.
    :return: The (simulationTime, trialId) pair.
    """
    return sim["endTime"], sim["trialId"]


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


async def insert_simulation(context: SimulationContext, sim: warm.WarmSimulationData,
                            on_failure: Literal["raise", "ignore", "update"] = "replace"):
    """ Inserts the given simulation data into database.

    :param context: The simulation context.
    :param sim: Simulation data to be inserted.
    :param on_failure: Action to perform on insertion failure due to duplicates.
    """
    cursor: aiosqlite.Cursor
    async with context.db.cursor() as cursor:
        if on_failure == "raise":
            await cursor.execute(f"""
                INSERT INTO SimData (trialId, endTime, simId, root, counts)
                VALUES ({sim.trialId}, {sim.t}, '{context.target.simId}', '{sim.root}', '{json.dumps(sim.counts)}')
            """)
        elif on_failure == "ignore":
            await cursor.execute(f"""
                INSERT OR IGNORE INTO SimData (trialId, endTime, simId, root, counts)
                VALUES ({sim.trialId}, {sim.t}, '{context.target.simId}', '{sim.root}', '{json.dumps(sim.counts)}')
            """)
        elif on_failure == "replace":
            await cursor.execute(f"""
                INSERT OR REPLACE INTO SimData (trialId, endTime, simId, root, counts)
                VALUES ({sim.trialId}, {sim.t}, '{context.target.simId}', '{sim.root}', '{json.dumps(sim.counts)}')
            """)
        else:
            raise ValueError(f"Unknown failure action {on_failure}")


async def worker(
        context: SimulationContext,
        session: aiohttp.ClientSession
):
    """
    Creates a simulation worker coroutine.

    Simulation is done remotely on the server.

    :param context: The simulation context.
    :param session: The HTTP session the worker is bind to.
    """
    while True:
        # Fetches a new job and start working on it.
        # If the termination signal is received, stops working.
        sim_tuple = await context.pending_simulation.get()

        if sim_tuple is None:
            return

        _, _, sim = sim_tuple

        try:
            response = await session.post(f"/api/simulate", json=sim.to_dict())
            response.raise_for_status()
            text = await response.text()
            res = warm.WarmSimulationData.from_json(text)
            context.pending_simulation.task_done()
            await context.pending_storage.put(res)
        except ValueError:
            pass


async def prepare_simulation(
        context: SimulationContext,
        use_solver: bool = True
):
    """ Prepares the simulation.

    :param context: The simulation context.
    :param use_solver: Whether to obtain solution to the model.
    """
    # Ensures the presence of data tables required.
    await create_siminfo_if_missing(context.db)
    await create_simdata_if_missing(context.db)

    cursor: aiosqlite.Cursor
    async with context.db.cursor() as cursor:
        # Obtains relevant information regarding the simulation of interest.
        # First query the master data.
        await cursor.execute(f"""
            SELECT simId, completedTrials, maxTime FROM SimInfo 
            WHERE simId = '{context.target.simId}'
        """)
        info_row = await cursor.fetchone()

        # Then query detailed sim data.
        await cursor.execute(f"""
            SELECT trialId, max(endTime) AS endTime, counts FROM SimData
            WHERE SimID = '{context.target.simId}'
            GROUP BY trialId
            ORDER BY trialId
        """)
        data_rows = sorted(await cursor.fetchall(), key=simulation_order)

    completed_count = 0  # Tracks the number of completed jobs.
    latest_sims: list[warm.WarmSimulationData]  # Tracks the latest simulation trials.

    if data_rows is None:
        # No data available about this simulation. Start from scratch.
        # Also need to update the database to include a zero entry.
        existing_latest_sims = []
        latest_sims = [warm.WarmSimulationData(
            model=context.target.model,
            root="0.0",
            counts=None,
            targetTime=config.TIME_STEP,
            t=0,
            trialId=i
        ) for i in range(context.target.n)]
        print(" - No prior simulation found. Starting from scratch.")
    else:
        # Use the latest entries.
        existing_latest_sims = [warm.WarmSimulationData(
            model=context.target.model,
            root="0.0",
            counts=json.loads(data_row["counts"]),
            targetTime=data_row["endTime"] + config.TIME_STEP,
            t=data_row["endTime"],
            trialId=data_row["trialId"]
        ) for data_row in data_rows]
        latest_sims = existing_latest_sims + [warm.WarmSimulationData(
            model=context.target.model,
            root="0.0",
            counts=None,
            targetTime=config.TIME_STEP,
            t=0,
            trialId=i
        ) for i in range(len(data_rows), context.target.n)]
        print(f" - {len(data_rows)} previous simulations found. Using those as well")

    # TODO: solving support
    if use_solver:
        pass

    # Ensure the database know about this simulation.
    if info_row is None:
        async with context.db.cursor() as cursor:
            await cursor.execute(f"""
                INSERT INTO SimInfo (simID, model) 
                VALUES ('{context.target.simId}', '{context.target.model.to_json()}')
            """)
            await context.db.commit()

    # Place a few initial simulations and count completed simulations.
    for sim in latest_sims:
        if sim.t < context.target.t:
            await context.pending_simulation.put((sim.t, sim.trialId, sim))
        else:
            completed_count += 1

    return completed_count


async def simulation_manager(context: SimulationContext, target: Target, completed_count: int):
    """ Writes the results into the database.

    :param context: The simulation context.
    :param target: The simulation target.
    :param completed_count: Number of trials completed.
    """
    if context.pbar_completed is not None:
        context.pbar_completed.update(completed_count)

    while completed_count < target.n:
        # Fetches a simulation result and stores it in the database.
        res = await context.pending_storage.get()

        # Creates a new entry in the data table.
        cursor: aiosqlite.Cursor
        async with context.db.cursor() as cursor:
            await cursor.execute(f"""
                INSERT INTO SimData (trialId, endTime, simId, root, counts)
                VALUES ({res.trialId}, {res.t}, '{target.simId}', '{res.root}', '{json.dumps(res.counts)}')
            """)
            await context.db.commit()
        print(f" - Trial {res.trialId} time {res.t} is written to the database")

        # Adds the next simulation item into the queue.
        if res.t < target.t:
            await context.pending_simulation.put((res.t, res.trialId, warm.WarmSimulationData(
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
            if context.pbar_completed is not None:
                context.pbar_completed.update(1)

        context.pending_storage.task_done()

    # When all the trials are completed, inform the workers and finalise
    # the database entry.
    for _ in context.sessions:
        await context.pending_simulation.put(None)

    async with context.db.cursor() as cursor:
        await cursor.execute(f"""
            REPLACE INTO SimInfo (simId, model, completedTrials, maxTime)
            VALUES ('{target.simId}', '{target.model.to_json()}', {target.n}, {target.t})
        """)
        await context.db.commit()

# TODO: High priority. More robust handling of database entries.
# TODO: High priority. A crash-preventing caching system for restarting
# TODO: Low priority. Better progress bar.


async def simulate_one_target(
        target: Target, workersData: list[WorkerData], use_progress_bar: bool = True, use_solver: bool = True):
    """ Controls the simulation.

    :param target: The simulation target.
    :param workersData: The list of workers available.
    :param use_progress_bar: Whether to use a progress bar.
    :param use_solver: Whether to use a solver.
    """
    pending_simulation = asyncio.PriorityQueue()  # Items waiting to be simulated
    pending_storage = asyncio.Queue()             # Items pending storage

    async with contextlib.AsyncExitStack() as stack:
        cursor: aiosqlite.Cursor

        # Establish db connection and check the current simulation status.
        db = await stack.enter_async_context(DataManager(
            config.DB_LOCATION, is_readonly=False, use_dict_factory=True))

        # Creates sessions to communicate with workers.
        session_pool = await asyncio.gather(*[stack.enter_async_context(
            aiohttp.ClientSession(x.url)) for x in workersData])

        # Initialises the progress bars if they are demanded.
        pbar_completed = None
        pbar_full = None
        if use_progress_bar:
            pbar_completed = stack.enter_context(tqdm(
                desc="Fully completed trials: ",
                total=target.n
            ))
            pbar_full = stack.enter_context(tqdm(
                desc="Overall: ",
                total=target.n * target.t
            ))

        # Creates the global context and state store.
        context = SimulationContext(
            target=target,
            pending_simulation=pending_simulation,
            pending_storage=pending_storage,
            sessions=session_pool,
            db=db,
            pbar_completed=pbar_completed,
            pbar_full=pbar_full
        )

        # Launch the simulation manager to prepare simulations.
        completed_count = await prepare_simulation(context, use_solver=use_solver)

        # Keeps track of the tasks created.
        worker_tasks: list[asyncio.Task[None]] = []

        try:
            # Spawns the worker tasks and runs in concurrency.
            for session in session_pool:
                task = asyncio.create_task(
                    worker(context, session))
                worker_tasks.append(task)
            manager_task = asyncio.create_task(simulation_manager(context, target, completed_count))

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

    :param args: The command line arguments.
    """
    # Creates the main task.
    try:
        # targets = [Target(
        #     t=10_000_000,
        #     n=50,
        #     simId=f"ring_2d_{i}",
        #     model=warm.WarmModel(is_graph=True, graph=warm.ring_2d_graph(i)),
        # ) for i in range(3, 10)]

        targets = [Target(
            t=10_000_000,
            n=50,
            simId=f"ring_2d_3",
            model=warm.WarmModel(is_graph=True, graph=warm.ring_2d_graph(3)),
        )]

        for target in targets:
            # Executes the task until completion.
            task = asyncio.create_task(simulate_one_target(target, workersData=[
                # WorkerData("http://localhost:7071"),
                WorkerData("http://127.0.0.1:8080"),
                WorkerData("http://10.0.0.1:8079"),
                WorkerData("http://10.0.0.1:8080")
            ]))

        await task
    except KeyboardInterrupt:
        task.cancel()


if __name__ == '__main__':
    asyncio.run(main(), debug=True)
