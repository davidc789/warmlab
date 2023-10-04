""" Central workflow management. """

import dataclasses
import json
import time

from argparse import ArgumentParser
from multiprocessing import Process, JoinableQueue

from typing import Optional

from tqdm import tqdm

from .. import config
from .. import warm
from ..ContextManagers import DataManager, DBHandler, DatabaseManager


@dataclasses.dataclass()
class HpcContext(object):
    target_dict: dict[str, config.Target]
    pending_simulation: JoinableQueue#[warm.SimData]
    db_location: str
    time_step: int
    pbar_completed: Optional[tqdm] = None
    pbar_full: Optional[tqdm] = None


def worker(context: HpcContext):
    """ High-performance computing worker.

    Warning: This operation is synchronous and blocks. Always run it in another
    process.

    :param context: The simulation context.
    """
    with DataManager(lim=1000, handlers=[DBHandler("SimData", context.db_location)]) as dm:
        for sim_data in iter(context.pending_simulation.get, None):
            # Construct the simulation information object.
            sim_info = warm.SimInfo(
                simId=sim_data.simId,
                model=context.target_dict[sim_data.simId].model,
            )
            sim_data.calc_omega_x(sim_info.model)

            # Repeatedly simulate until the target is reached.
            while sim_data.endTime < context.target_dict[sim_data.simId].endTime:
                sim_data = warm.simulate(sim_info, sim_data)
                dm.write({
                    "root": sim_data.root,
                    "endTime": sim_data.endTime,
                    "trialId": sim_data.trialId,
                    "counts": json.dumps(sim_data.counts),
                    "simId": sim_data.simId,
                })
                sim_data.endTime += context.time_step

            # Inform the queue the job is done.
            context.pending_simulation.task_done()


def prepare_simulation_worker(params: tuple[HpcContext, config.Target]):
    """ Prepares the simulation.

    :param params: The parameters required for the worker.
    """
    # If solving is desired, solve it quickly.
    context, target = params
    solution = None
    completed_count = 0
    if config.config.use_solver:
        solution = warm.solve(model=target.model).to_json()
        print(" - Solver completed successfully")

    # Ensure the database know about this simulation and have the solution available.
    with DatabaseManager(config.config.db_path, is_readonly=False, use_dict_factory=True) as db:
        cursor = db.cursor()
        cursor.execute(f"""
            INSERT OR IGNORE INTO SimInfo (simId, model, solution)
            VALUES (?, ?, ?)
        """, [target.model.id, target.model.to_json(), solution])
        cursor.execute(f"""
            UPDATE SimInfo
            SET model = ?,
                solution = ?
            WHERE simId = ?
        """, [target.model.to_json(), solution, target.model.id])
        db.commit()

        # Determine the current simulation progress.
        cursor.execute(f"""
            SELECT trialId, max(endTime) AS endTime, counts FROM SimData
            WHERE simId = ?
            GROUP BY trialId
            ORDER BY trialId
        """, [target.model.id])
        data_rows = cursor.fetchall()

    # Start simulating from the existing ones.
    for data_row in data_rows:
        if data_row["endTime"] < target.endTime:
            context.pending_simulation.put(warm.SimData(
                root="0.0",
                counts=json.loads(data_row["counts"]),
                endTime=data_row["endTime"] + config.config.time_step,
                t=data_row["endTime"],
                trialId=data_row["trialId"],
                simId=target.model.id,
            ))
        else:
            completed_count += 1

    # Then place the new jobs.
    for i in range(len(data_rows), target.n):
        context.pending_simulation.put(warm.SimData(
            root="0.0",
            counts=None,
            endTime=config.config.time_step,
            t=0,
            trialId=i,
            simId=target.model.id,
        ))

    return completed_count


def manager():
    # Set up the queues and build an auxiliary target lookup, and then build the global context.
    nproc = config.config.n_proc
    pending_simulation = JoinableQueue()
    target_dict: dict[str, config.config.Target] = {
        target.model.id: target for target in config.config.targets
    }
    context = HpcContext(
        target_dict=target_dict,
        pending_simulation=pending_simulation,
        db_location=config.config.db_path,
        time_step=config.config.time_step,
    )

    # Compute the amount of tasks to be completed.
    completed_count = 0
    task_count = sum(target.n for target in config.config.targets)

    # Prepare the simulations in a paralleled manner. Not parallel yet.
    for target in config.config.targets:
        completed_count += prepare_simulation_worker((context, target))

    processes: list[Process] = []

    try:
        # Start the worker processes.
        for i in range(nproc):
            process = Process(target=worker, args=(context,))
            processes.append(process)
            process.start()

        # Monitor the simulation loosely.
        start_time = time.time()
        while completed_count < task_count:
            # Print statistics to the stdout.
            elapsed_time = time.time() - start_time
            new_completed_count = task_count - context.pending_simulation.qsize()
            print(f"{new_completed_count} out of {task_count} trials done ({elapsed_time})", flush=True)

            # Update the progress bar, if there is one.
            if config.config.use_progress_bar:
                if context.pbar_completed is not None:
                    context.pbar_completed.update(new_completed_count - completed_count)

            completed_count = new_completed_count
            time.sleep(30)

        # Sends the stop signal to the workers.
        print(f" - Stopping all process...", flush=True)
        for _ in range(nproc):
            context.pending_simulation.put(None)
    finally:
        # Wait for the processes to join before quitting.
        for p in processes:
            p.join()
            p.close()


def main(argv: Optional[list[str]] = None):
    # Create the parser.
    parser = ArgumentParser("main")
    parser.add_argument("--config", default=None, type=str,
                        help="Path to config file")
    parser.add_argument("--cores", default=None, type=int,
                        help="Number of cores to use. Overrides the config")

    # Read the argument.
    args = parser.parse_args(argv)
    if args.config is not None:
        config.load_config(args.config, args.cores)

    manager()


if __name__ == '__main__':
    main()
