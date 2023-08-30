import logging.config
import os
from pathlib import Path
from typing import NamedTuple, Optional, Literal

import warm

logger = logging.getLogger(__name__)


class WorkerData(NamedTuple):
    worker_type: Literal["http", "proc"]
    url: Optional[str]


class Target(NamedTuple):
    """ Specifying a simulation target. """
    t: int                             # Target end time.
    n: int                             # Target number of simulations.
    simId: str                         # The simulation ID.
    model: warm.WarmModel              # The underlying WARM model.


class Config():
    # Whether to display progress bars.
    use_progress_bar: bool = True

    # Whether to apply the solver.
    use_solver: bool = True

    # Whether to use high-performance computing mode.
    use_hpc: bool = True

    # Control level of outputting. Disable this in hpc mode to further boost
    # performance. Note that this does not disable the progress bars, so
    # disable them manually if necessary.
    verbose = True

    # Location of the database.
    DB_LOCATION = "../../data.sqlite"

    # Directory location for storing csv output.
    CSV_PATH = "../../output/"

    # Frequency of database entry.
    TIME_STEP = 1_000_000

    # Maximum number of rows to buffer. For hpc, consider increasing this.
    buffer_limit = 100_000

    # Number of processes.
    n_proc = 2

    # The list of simulation targets.
    targets = ([
        Target(
            t=10_000_000,
            n=1,
            simId=f"ring_2d_{i}",
            model=warm.WarmModel(
                is_graph=True,
                graph=warm.ring_2d_graph(i),
                model_id=f"ring_2d_{i}"
            ),
        ) for i in range(3, 11)
    ])

    # A list of workers.
    workers = (
        # + [WorkerData(worker_type="html", url=f"http://127.0.0.1:{port}") for port in range(8080,     8081)]
        # + [WorkerData(worker_type="html", url=f"http://10.0.0.1:{port}" ) for port in range(8081 - 8, 8081)]
        # + [WorkerData(worker_type="html", url=f"http://10.0.0.3:{port}" ) for port in range(8081 - 8, 8081)]
        # + [WorkerData(worker_type="html", url=f"http://10.0.0.4:{port}" ) for port in range(8081 - 8, 8081)]
        # + [WorkerData("http://warmlab.azurewebsites.net")]
    )

    # A list of process workers
    proc_workers = (
        [WorkerData(worker_type="proc", url=None) for _ in range(n_proc)]
    )

    # Time-out for the main manager. Set this to be large, so it only catches
    # extremely abnormal events like one of the process got killed.
    time_out = 600


config = Config()
