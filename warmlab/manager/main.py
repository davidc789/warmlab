from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from .. import warm
from ..config import Config, Target

DEFAULT_JOBS_DIR = "./data/jobs/"


def write_jobs_handler(args: Namespace):
    """ Write jobs. """
    config = Config()
    job_dir = Path(DEFAULT_JOBS_DIR)

    with open(job_dir / "sample_job.json", "w") as f:
        f.write(config.to_json(indent=4))

    # Customise targets here
    config.targets = [
        Target(
            t=1_000_000_000,
            n=100,
            model=warm.WarmModel(
                is_graph=True,
                graph=warm.ring_2d_graph(i),
                model_id=f"ring_2d_{i}"
            ),
        ) for i in range(3, 11)
    ]

    with open(job_dir / "job1.json", "w") as f:
        f.write(config.to_json(indent=4))


def main(argv: Optional[list[str]] = None):
    # Add the parser.
    parser = ArgumentParser("warmlab")
    subparsers = parser.add_subparsers(required=True)

    write_parser = subparsers.add_parser("write")
    write_parser.set_defaults(func=write_jobs_handler)

    # Parse the arguments.
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
