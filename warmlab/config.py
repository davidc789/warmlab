import logging.config
from dataclasses import dataclass, field
from os import PathLike
from typing import Optional, Literal

from . import warm

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkerData(warm.JsonSerialisable):
    worker_type: Literal["http", "proc"]
    url: Optional[str]


@dataclass(slots=True)
class Target(warm.JsonSerialisable):
    """ Specifying a simulation target. """
    endTime: int                             # Target end time.
    n: int                             # Target number of simulations.
    model: warm.WarmModel              # The underlying WARM model.
    # progress: Optional[list[warm.WarmSimData]] = None  # Current simulation progress, if applicable.

    def to_dict(self):
        return {
            "t": self.endTime,
            "n": self.n,
            "model": self.model.to_dict(),
            # "progress": self.progress
        }

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(
            endTime=dct["t"],
            n=dct["n"],
            model=warm.WarmModel.from_dict(dct["model"]),
            # "progress": [warm.WarmSimData.from_dict(x) for x in dct["progress"]]
        )


@dataclass()
class Config(warm.JsonSerialisable):
    # Whether to display progress bars.
    use_progress_bar: bool = True

    # Whether to apply the solver.
    use_solver: bool = True

    # Whether to use high-performance computing mode.
    use_hpc: bool = True

    # Control level of outputting. Disable this in hpc mode to further boost
    # performance. Note that this does not disable the progress bars, so
    # disable them manually if necessary.
    verbose: bool = True

    # Location of the database.
    db_path: str = "./data.sqlite"

    # Directory location for storing csv output.
    csv_path: str = "./data/"

    # Frequency of database entry.
    time_step: int = 1_000_000

    # Maximum number of rows to buffer. For hpc, consider increasing this.
    buffer_limit: int = 100_000

    # Number of processes.
    n_proc: int = 2

    # The list of simulation targets.
    targets: list[Target] = field(default_factory=lambda: [
        Target(
            endTime=10_000_000,
            n=1,
            model=warm.WarmModel(
                is_graph=True,
                graph=warm.ring_2d_graph(i),
                model_id=f"ring_2d_{i}"
            ),
        ) for i in range(3, 11)
    ])

    # A list of workers.
    workers: list[WorkerData] = (
        # + [WorkerData(worker_type="html", url=f"http://127.0.0.1:{port}") for port in range(8080,     8081)]
        # + [WorkerData(worker_type="html", url=f"http://10.0.0.1:{port}" ) for port in range(8081 - 8, 8081)]
        # + [WorkerData(worker_type="html", url=f"http://10.0.0.3:{port}" ) for port in range(8081 - 8, 8081)]
        # + [WorkerData(worker_type="html", url=f"http://10.0.0.4:{port}" ) for port in range(8081 - 8, 8081)]
        # + [WorkerData("http://warmlab.azurewebsites.net")]
    )

    # Time-out for the main manager. Set this to be large, so it only catches
    # extremely abnormal events like one of the process got killed.
    time_out: int = 6000

    def to_dict(self):
        return self.__dict__ | {
            "targets": [x.to_dict() for x in self.targets],
            "workers": [x.to_dict() for x in self.workers]
        }

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**(dct | {
            "targets": [Target.from_dict(x) for x in dct["targets"]],
            "workers": [WorkerData.from_dict(x) for x in dct["workers"]]
        }))


config = Config()


def load_config(path: PathLike, n_proc: Optional[int] = None):
    with open(path) as f:
        global config
        config = Config.from_json(f.read())
        if n_proc is not None:
            config.n_proc = n_proc
