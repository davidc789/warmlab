""" Central workflow management. """

import dataclasses
import json

from argparse import ArgumentParser
from multiprocessing import Process, JoinableQueue
from typing import Optional

from tqdm import tqdm

from .. import config
from .. import warm
from ..ContextManagers import DataManager, CSVHandler


@dataclasses.dataclass()
class HpcContext(object):
    pending_simulation: JoinableQueue#[warm.WarmSimData]
    pending_storage: JoinableQueue#[warm.WarmSimData]
    pbar_completed: Optional[tqdm] = None
    pbar_full: Optional[tqdm] = None


def worker(context: HpcContext):
    """ High-performance computing worker.

    Warning: This operation is synchronous and blocks. Always run it in another
    process.

    :param context: The simulation context.
    """
    for sim in iter(context.pending_simulation.get, None):
        res = warm.simulate(sim)
        context.pending_storage.put(res)
        context.pending_simulation.task_done()


def manager():
    # Set up the queues and the timer.
    pending_simulation = JoinableQueue()
    pending_storage = JoinableQueue()

    # The global context.
    context = HpcContext(
        pending_simulation=pending_simulation,
        pending_storage=pending_storage
    )

    # Start the worker processes.
    for i in range(config.config.n_proc):
        Process(target=worker, args=(context,)).start()

    # Compute the amount of tasks to be completed.
    task_count = sum(target.n for target in config.config.targets)
    completed_count = 0

    # Build an auxiliary target lookup.
    target_dict: dict[str, config.config.Target] = {target.model.id: target for target in config.config.targets}

    # Places a few initial tasks into the queue.
    for target in config.config.targets:
        for i in range(target.n):
            context.pending_simulation.put(warm.WarmSimData(
                model=target.model,
                root="0.0",
                counts=None,
                targetTime=config.config.time_step,
                t=0,
                trialId=i
            ))

    # Observe the queue and add things in when necessary, until everything is
    # done.
    with DataManager(config.config.buffer_limit, handlers=[CSVHandler(config.config.csv_path)]) as dm:
        while completed_count < task_count:
            res = context.pending_storage.get(timeout=config.config.time_out)
            dm.write({
                "trialId": res.trialId,
                "endTime": res.t,
                "simId": res.model.id,
                "root": res.root,
                "counts": json.dumps(res.counts)
            })

            if res.t < target_dict[res.model.id].t:
                if config.config.verbose:
                    print(f" - Simulation {res.model.id} trial {res.trialId} time {res.t} is completed", flush=True)
                context.pending_simulation.put(warm.WarmSimData(
                    model=res.model,
                    root=res.root,
                    counts=res.counts,
                    targetTime=res.t + config.config.time_step,
                    t=res.t,
                    trialId=res.trialId
                ))
            else:
                completed_count += 1
                if config.config.verbose:
                    print(f" - Simulation {res.model.id} trial {res.trialId} is completed", flush=True)
                if config.config.use_progress_bar:
                    pass
                    # if context.pbar_completed is not None:
                    #     context.pbar_completed.update(1)

            context.pending_storage.task_done()

    # Sends the stop signal to the workers.
    print(f" - Stopping all process...")
    for _ in range(config.config.n_proc):
        context.pending_simulation.put(None)


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
