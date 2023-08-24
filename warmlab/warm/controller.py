""" The simulation master controller. """

import json
import logging
import asyncio
import contextlib
import math

from typing import Optional, NamedTuple, Literal

import aiosqlite
import aiohttp

from tqdm import tqdm

from config import config
from warmlab.warm import warm
from DataManager import DataManager

logger = logging.getLogger(__name__)


class WorkerData(NamedTuple):
    url: str


class Target(NamedTuple):
    """ Specifying a simulation target. """
    t: int                             # Target end time.
    n: int                             # Target number of simulations.
    simId: str                         # The simulation ID.
    model: warm.WarmModel              # The underlying WARM model.


class SimulationContext(NamedTuple):
    """ Global context for the simulation controller. """
    target: Target
    pending_simulation: asyncio.Queue[tuple[int, int, warm.WarmSimData] | tuple[float, None, None]]
    pending_storage: asyncio.Queue[warm.WarmSimData]
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
                PRIMARY KEY (trialId, endTime, simId),
                FOREIGN KEY (simId) REFERENCES SimInfo(simId)
            )
        """)
        await db.commit()


async def insert_simulation(context: SimulationContext, sim: warm.WarmSimData,
                            on_failure: Literal["raise", "ignore", "update"] = "raise"):
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

        await context.db.commit()


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
        _, _, sim = sim_tuple

        if sim is None:
            return

        try:
            response = await session.post(f"/api/simulate", json=sim.to_dict())
            response.raise_for_status()
            text = await response.text()
            res = warm.WarmSimData.from_json(text)
            await context.pending_storage.put(res)
            context.pending_simulation.task_done()
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
            SELECT simId, solution, completedTrials, maxTime FROM SimInfo 
            WHERE simId = '{context.target.simId}'
        """)
        info_row = await cursor.fetchone()

        # Then query detailed sim data.
        await cursor.execute(f"""
            SELECT trialId, max(endTime) AS endTime, counts FROM SimData
            WHERE simId = '{context.target.simId}'
            GROUP BY trialId
            ORDER BY trialId
        """)
        data_rows = sorted(await cursor.fetchall(), key=simulation_order)

    completed_count = 0  # Tracks the number of completed jobs.
    latest_sims: list[warm.WarmSimData]  # Tracks the latest simulation trials.

    if data_rows is None:
        # No data available about this simulation. Start from scratch.
        # Also need to update the database to include a zero entry.
        existing_latest_sims = []
        latest_sims = [warm.WarmSimData(
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
        existing_latest_sims = [warm.WarmSimData(
            model=context.target.model,
            root="0.0",
            counts=json.loads(data_row["counts"]),
            targetTime=data_row["endTime"] + config.TIME_STEP,
            t=data_row["endTime"],
            trialId=data_row["trialId"]
        ) for data_row in data_rows]
        latest_sims = existing_latest_sims + [warm.WarmSimData(
            model=context.target.model,
            root="0.0",
            counts=None,
            targetTime=config.TIME_STEP,
            t=0,
            trialId=i
        ) for i in range(len(data_rows), context.target.n)]
        print(f" - {len(data_rows)} previous simulations found. Using those as well")

    # If solving is desired, solve it quickly.
    solution = None
    if use_solver:
        solution = warm.solve(model=context.target.model).to_json()
        print(" - Solver completed successfully")

    # Ensure the database know about this simulation and have the solution available.
    async with context.db.cursor() as cursor:
        await cursor.execute(f"""
            INSERT OR IGNORE INTO SimInfo (simId, model, solution)
            VALUES ('{context.target.simId}', '{context.target.model.to_json()}', '{solution}')
        """)
        await cursor.execute(f"""
            UPDATE SimInfo
            SET model = '{context.target.model.to_json()}',
                solution = '{solution}'
            WHERE simId = '{context.target.simId}'
        """)
        await context.db.commit()

    # Place a few initial simulations and count completed simulations.
    for sim in latest_sims:
        if sim.t < context.target.t:
            await context.pending_simulation.put((sim.t, sim.trialId, sim))
        else:
            completed_count += 1

    return completed_count


async def simulation_manager(context: SimulationContext, completed_count: int):
    """ Writes the results into the database.

    :param context: The simulation context.
    :param completed_count: Number of trials completed.
    """
    if context.pbar_completed is not None:
        context.pbar_completed.update(completed_count)

    while completed_count < context.target.n:
        # Fetches a simulation result and stores it in the database.
        res = await context.pending_storage.get()

        # Creates a new entry in the data table.
        cursor: aiosqlite.Cursor
        await insert_simulation(context, res, "raise")
        print(f" - Trial {res.trialId} time {res.t} is written to the database")

        # Adds the next simulation item into the queue.
        if res.t < context.target.t:
            await context.pending_simulation.put((res.t, res.trialId, warm.WarmSimData(
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
    await context.pending_simulation.put((math.inf, None, None))
    for _ in context.sessions:
        await context.pending_simulation.put((math.inf, None, None))

    # Overwrite the completed trials and maxtime.
    async with context.db.cursor() as cursor:
        await cursor.execute(f"""
            UPDATE SimInfo
            SET completedTrials = {context.target.n},
                maxTime = {context.target.t}
            WHERE simId = '{context.target.simId}'
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

        # Spawns the worker tasks and runs in concurrency.
        manager_task = asyncio.create_task(simulation_manager(context, completed_count))
        for session in session_pool:
            task = asyncio.create_task(
                worker(context, session))
            worker_tasks.append(task)

        # Await for task completion.
        await manager_task
        for task in worker_tasks:
            await task

        # try:
        #     # Spawns the worker tasks and runs in concurrency.
        #     for session in session_pool:
        #         task = asyncio.create_task(
        #             worker(context, session))
        #         worker_tasks.append(task)
        #     manager_task = asyncio.create_task(simulation_manager(context, completed_count))
        #
        #     # Await for task completion.
        #     for task in worker_tasks:
        #         await task
        #     await manager_task
        # except asyncio.CancelledError:
        #     # Propagates the kill switch
        #     for task in worker_tasks:
        #         task.cancel()
        #     manager_task.cancel()


async def main(args: Optional[list[str]] = None):
    """ The main coroutine serving as the entry point of the program.

    :param args: The command line arguments.
    """
    targets = ([
        Target(
            t=10_000_000,
            n=360,
            simId=f"ring_2d_{i}",
            model=warm.WarmModel(
                is_graph=True,
                graph=warm.ring_2d_graph(i),
                model_id=f"ring_2d_{i}"
            ),
        ) for i in range(3, 11)
    ])
    workers = ([]
        # + [WorkerData(f"http://127.0.0.1:{port}") for port in range(8080, 8081)]
        + [WorkerData(f"http://127.0.0.1:{port}") for port in range(8081 - 4, 8081)]
        # + [WorkerData(f"http://10.0.0.1:{port}") for port in range(8081 - 8, 8081)]
        # + [WorkerData(f"http://10.0.0.3:{port}") for port in range(8081 - 8, 8081)]
        # + [WorkerData(f"http://10.0.0.4:{port}") for port in range(8081 - 8, 8081)]
        # + [WorkerData("http://warmlab.azurewebsites.net")]
    )

    # Creates the main task.
    # Executes the task until completion.
    for target in targets:
        task = asyncio.create_task(simulate_one_target(target, workersData=workers))

        # This line is here so the tasks are not created simultaneously. If the
        # number of workers is so huge that they start to become idle, remove
        # this line.
        await task

    # try:
    #     await task
    # except KeyboardInterrupt:
    #     task.cancel()
    #     raise


if __name__ == '__main__':
    # asyncio.run(main(), debug=True)
    asyncio.run(main())
