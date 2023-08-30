""" The simulation master controller. """
import dataclasses
import json
import logging
import asyncio
import math
from itertools import chain
from multiprocessing import Process, JoinableQueue, current_process

from typing import Optional, NamedTuple, Literal

import aiosqlite
import aiohttp

from tqdm import tqdm

from config import config, Target, WorkerData
import warm
from ContextManagers import DatabaseManager, ProcessManager, DataManager, CSVHandler

logger = logging.getLogger(__name__)


class SimulationContext(NamedTuple):
    """ Global context for the simulation controller. """
    pending_simulation: asyncio.Queue[tuple[str, int, int, warm.WarmSimData] | tuple[str, float, None, None]]
    pending_storage: asyncio.Queue[warm.WarmSimData]
    db: aiosqlite.Connection
    pbar_completed: Optional[tqdm] = None
    pbar_full: Optional[tqdm] = None


@dataclasses.dataclass()
class HpcContext(object):
    pending_simulation: JoinableQueue#[warm.WarmSimData]
    pending_storage: JoinableQueue#[warm.WarmSimData]
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


def hpc_worker(context: HpcContext):
    """ High-performance computing worker.

    Warning: This operation is synchronous and blocks. Always run it in another
    process.

    :param context: The simulation context.
    """
    for sim in iter(context.pending_simulation.get, None):
        res = warm.simulate(sim)
        context.pending_storage.put(res)
        context.pending_simulation.task_done()


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
    elif data.worker_type == "popen":
        async with ProcessManager(config.interpreter_path, config.worker_path) as proc:
            while True:
                # Fetches a new job and start working on it.
                # If the termination signal is received, stops working.
                sim: warm.WarmSimData
                sim_tuple = await context.pending_simulation.get()
                _, _, _, sim = sim_tuple

                if sim is None:
                    return

                try:
                    response, error = await proc.communicate(sim.to_json().encode())

                    if error:
                        raise error

                    res = warm.WarmSimData.from_json(response.decode())
                    await context.pending_storage.put(res)
                    context.pending_simulation.task_done()
                except ValueError as error:
                    logger.error(error)
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
            WHERE simId = '{target.simId}'
        """)
        info_row = await cursor.fetchone()
        # TODO: use info row to cross-check data integrity.

        # Then query detailed sim data.
        await cursor.execute(f"""
            SELECT trialId, max(endTime) AS endTime, counts FROM SimData
            WHERE simId = '{target.simId}'
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
        targetTime=data_row["endTime"] + config.TIME_STEP,
        t=data_row["endTime"],
        trialId=data_row["trialId"]
    ) for data_row in data_rows]
    latest_sims = existing_latest_sims + [warm.WarmSimData(
        model=target.model,
        root="0.0",
        counts=None,
        targetTime=config.TIME_STEP,
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
            VALUES ('{target.simId}', '{target.model.to_json()}', '{solution}')
        """)
        await cursor.execute(f"""
            UPDATE SimInfo
            SET model = '{target.model.to_json()}',
                solution = '{solution}'
            WHERE simId = '{target.simId}'
        """)
        await context.db.commit()

    # Place a few initial simulations and keep track of completed simulations.
    completed_count = 0
    for sim in latest_sims:
        if sim.t < target.t:
            await context.pending_simulation.put((target.simId, sim.t, sim.trialId, sim))
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
            if res.t < target.t:
                await context.pending_simulation.put((target.simId, res.t, res.trialId, warm.WarmSimData(
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
                # if context.pbar_completed is not None:
                #     context.pbar_completed.update(1)

            context.pending_storage.task_done()

        # Overwrite the completed trials and maxtime.
        async with context.db.cursor() as cursor:
            await cursor.execute(f"""
                UPDATE SimInfo
                SET completedTrials = {target.n},
                    maxTime = {target.t}
                WHERE simId = '{target.simId}'
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

    async with DatabaseManager(config.DB_LOCATION, is_readonly=False, use_dict_factory=True) as db:
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
                await context.pending_simulation.put((target.simId, math.inf, None, None))

            for task in worker_tasks:
                await task
        except (KeyboardInterrupt, InterruptedError) as error:
            for task in chain(worker_tasks, manager_tasks):
                task.cancel()


def hpc_main():
    # Set up the queues and global state stores.
    pending_simulation = JoinableQueue()
    pending_storage = JoinableQueue()

    # The context.
    context = HpcContext(
        pending_simulation=pending_simulation,
        pending_storage=pending_storage
    )

    # Start the worker processes.
    for i in range(config.n_proc):
        Process(target=hpc_worker, args=(context,)).start()

    # Compute the amount of tasks to be completed.
    task_count = sum(target.n for target in config.targets)
    completed_count = 0

    # Build an auxiliary target lookup.
    target_dict: dict[str, Target] = {target.simId: target for target in config.targets}

    # Places a few initial tasks into the queue.
    for target in config.targets:
        for i in range(target.n):
            context.pending_simulation.put(warm.WarmSimData(
                model=target.model,
                root="0.0",
                counts=None,
                targetTime=config.TIME_STEP,
                t=0,
                trialId=i
            ))

    # Observe the queue and add things in when necessary, until everything is
    # done.
    with DataManager(config.buffer_limit, handlers=[CSVHandler(config.CSV_PATH)]) as dm:
        while completed_count < task_count:
            res = context.pending_storage.get(timeout=config.time_out)
            dm.write({
                "trialId": res.trialId,
                "endTime": res.t,
                "simId": res.model.id,
                "root": res.root,
                "counts": json.dumps(res.counts)
            })

            if res.t < target_dict[res.model.id].t:
                context.pending_simulation.put(warm.WarmSimData(
                    model=res.model,
                    root=res.root,
                    counts=res.counts,
                    targetTime=res.t + config.TIME_STEP,
                    t=res.t,
                    trialId=res.trialId
                ))
            else:
                completed_count += 1
                if config.verbose:
                    print(f" - Simulation {res.trialId} trial {res.trialId} is completed")
                if config.use_progress_bar:
                    pass
                    # if context.pbar_completed is not None:
                    #     context.pbar_completed.update(1)

            context.pending_storage.task_done()

    # Sends the stop signal to the workers.
    print(f" - Stopping all process...")
    for _ in range(config.n_proc):
        context.pending_simulation.put(None)


# TODO: Ultra-high priority. Centralise all workflows!
# TODO: High priority. More robust handling of database entries.
# TODO: High priority. A crash-preventing caching system for restarting
# TODO: Low priority. Better progress bar.


if __name__ == '__main__':
    if config.use_hpc:
        logger.info("HPC mode is on")
        hpc_main()
    else:
        asyncio.run(main(), debug=True)
        # asyncio.run(main())
