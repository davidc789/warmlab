""" The simulation master controller. """

import json
import logging
import asyncio
import math
from itertools import chain

from typing import Optional, NamedTuple, Literal

import aiosqlite
import aiohttp

from tqdm import tqdm

from ..config import config, Target, WorkerData
from .. import warm
from ..ContextManagers import DatabaseManager

logger = logging.getLogger(__name__)


class SimulationContext(NamedTuple):
    """ Global context for the simulation controller. """
    pending_simulation: asyncio.Queue[tuple[str, int, int, warm.WarmSimData] | tuple[str, float, None, None]]
    pending_storage: asyncio.Queue[warm.WarmSimData]
    db: aiosqlite.Connection
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
                VALUES ({sim.trialId}, {sim.t}, '{sim.model.id}', '{sim.root}', '{json.dumps(sim.counts)}')
            """)
        elif on_failure == "ignore":
            await cursor.execute(f"""
                INSERT OR IGNORE INTO SimData (trialId, endTime, simId, root, counts)
                VALUES ({sim.trialId}, {sim.t}, '{sim.model.id}', '{sim.root}', '{json.dumps(sim.counts)}')
            """)
        elif on_failure == "replace":
            await cursor.execute(f"""
                REPLACE INTO SimData (trialId, endTime, simId, root, counts)
                VALUES ({sim.trialId}, {sim.t}, '{sim.model.id}', '{sim.root}', '{json.dumps(sim.counts)}')
            """)
        else:
            raise ValueError(f"Unknown failure action {on_failure}")

        await context.db.commit()

async def worker(
        context: SimulationContext,
        data: WorkerData
):
    """ Pass the job to a given subprocess. Simple and easy.

    :param context: The global simulation context.
    :param data: Worker data.
    """
    if data.worker_type == "http":
        async with aiohttp.ClientSession(data.url) as session:
            while True:
                # Fetches a new job and start working on it.
                # If the termination signal is received, stops working.
                sim_tuple = await context.pending_simulation.get()
                _, _, _, sim = sim_tuple

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
    else:
        raise ValueError(f"Unknown worker type {data.worker_type}")


async def simulation_manager(
        context: SimulationContext,
        target: Target,
        use_solver: bool = True
):
    """ Controls the simulation.

    :param context: The global simulation context.
    :param target: The simulation target.
    :param use_solver: Whether to use a solver.
    """
    # Initialises the progress bars if they are demanded.
    # pbar_completed = None
    # pbar_full = None
    #
    # if use_progress_bar:
    #     pbar_completed = stack.enter_context(tqdm(
    #         desc="Fully completed trials: ",
    #         total=target.n
    #     ))
    #     pbar_full = stack.enter_context(tqdm(
    #         desc="Overall: ",
    #         total=target.n * target.t
    #     ))

    cursor: aiosqlite.Cursor
    async with context.db.cursor() as cursor:
        # Obtains relevant information regarding the simulation of interest.
        # First query the master data.
        await cursor.execute(f"""
            SELECT simId, solution, completedTrials, maxTime FROM SimInfo 
            WHERE simId = '{target.model.id}'
        """)
        info_row = await cursor.fetchone()
        # TODO: use info row to cross-check data integrity.

        # Then query detailed sim data.
        await cursor.execute(f"""
            SELECT trialId, max(endTime) AS endTime, counts FROM SimData
            WHERE simId = '{target.model.id}'
            GROUP BY trialId
            ORDER BY trialId
        """)
        data_rows = sorted(await cursor.fetchall(), key=simulation_order)

    # Cleanse the data rows so that None is converted into an empty list.
    if data_rows is None:
        data_rows = []

    print(f" - {len(data_rows)} previous simulations found. Using those as well")
    existing_latest_sims = [warm.WarmSimData(
        model=target.model,
        root="0.0",
        counts=json.loads(data_row["counts"]),
        targetTime=data_row["endTime"] + config.time_step,
        t=data_row["endTime"],
        trialId=data_row["trialId"]
    ) for data_row in data_rows]
    latest_sims = existing_latest_sims + [warm.WarmSimData(
        model=target.model,
        root="0.0",
        counts=None,
        targetTime=config.time_step,
        t=0,
        trialId=i
    ) for i in range(len(data_rows), target.n)]

    # If solving is desired, solve it quickly.
    solution = None
    if use_solver:
        solution = warm.solve(model=target.model).to_json()
        print(" - Solver completed successfully")

    # Ensure the database know about this simulation and have the solution available.
    async with context.db.cursor() as cursor:
        await cursor.execute(f"""
            INSERT OR IGNORE INTO SimInfo (simId, model, solution)
            VALUES ('{target.model.id}', '{target.model.to_json()}', '{solution}')
        """)
        await cursor.execute(f"""
            UPDATE SimInfo
            SET model = '{target.model.to_json()}',
                solution = '{solution}'
            WHERE simId = '{target.model.id}'
        """)
        await context.db.commit()

    # Place a few initial simulations and keep track of completed simulations.
    completed_count = 0
    for sim in latest_sims:
        if sim.t < target.endTime:
            await context.pending_simulation.put((target.model.id, sim.t, sim.trialId, sim))
        else:
            completed_count += 1

    try:
        # if context.pbar_completed is not None:
        #     context.pbar_completed.update(completed_count)

        while completed_count < target.n:
            # Fetches a simulation result and stores it in the database.
            res = await context.pending_storage.get()

            # Creates a new entry in the data table.
            cursor: aiosqlite.Cursor
            await insert_simulation(context, res, "raise")
            print(f" - Trial {res.trialId} time {res.t} is written to the database")

            # Adds the next simulation item into the queue.
            if res.t < target.endTime:
                await context.pending_simulation.put((target.model.id, res.t, res.trialId, warm.WarmSimData(
                    model=res.model,
                    root=res.root,
                    counts=res.counts,
                    targetTime=res.t + config.time_step,
                    t=res.t,
                    trialId=res.trialId
                )))
            else:
                completed_count += 1
                print(f" - Trial {res.trialId} is completed")
                # if context.pbar_completed is not None:
                #     context.pbar_completed.update(1)

            context.pending_storage.task_done()

        # Overwrite the completed trials and maxtime.
        async with context.db.cursor() as cursor:
            await cursor.execute(f"""
                UPDATE SimInfo
                SET completedTrials = {target.n},
                    maxTime = {target.endTime}
                WHERE simId = '{target.model.id}'
            """)
            await context.db.commit()
    except (InterruptedError, KeyboardInterrupt):
        raise


async def main(args: Optional[list[str]] = None):
    """ The main coroutine serving as the entry point of the program.

    :param args: The command line arguments.
    :param use_progress_bar: Whether to use a progress bar.
    :param use_solver: Whether to use a solver.
    """
    targets = config.targets

    pending_simulation = asyncio.PriorityQueue()  # Items waiting to be simulated
    pending_storage = asyncio.Queue()             # Items pending storage

    async with DatabaseManager(config.db_path, is_readonly=False, use_dict_factory=True) as db:
        # Ensures the presence of data tables required.
        await create_siminfo_if_missing(db)
        await create_simdata_if_missing(db)

        # Creates the global context and state store.
        context = SimulationContext(
            pending_simulation=pending_simulation,
            pending_storage=pending_storage,
            db=db
        )

        # Spawn the manager tasks and run in concurrency.
        manager_tasks = [asyncio.create_task(simulation_manager(context, target, config.use_solver))
                         for target in targets]

        # Spawn the worker tasks and run in concurrency.
        worker_tasks = [asyncio.create_task(worker(context, data))
                        for data in config.workers]

        # Execute the task until completion.
        # Unless this is interrupted.
        try:
            for task in manager_tasks:
                await task

            # When all the managers exit, all trials must have been done.
            # Time to fire all the workers so they stop.
            for target in targets:
                await context.pending_simulation.put((target.model.id, math.inf, None, None))

            for task in worker_tasks:
                await task
        except (KeyboardInterrupt, InterruptedError) as error:
            for task in chain(worker_tasks, manager_tasks):
                task.cancel()

# TODO: Ultra-high priority. Centralise all workflows!
# TODO: High priority. More robust handling of database entries.
# TODO: High priority. A crash-preventing caching system for restarting
# TODO: Low priority. Better progress bar.


if __name__ == '__main__':
    asyncio.run(main(), debug=True)
    # asyncio.run(main())
