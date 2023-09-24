from argparse import ArgumentParser, Namespace
from csv import DictReader
from pathlib import Path
from typing import Optional

import pandas as pd

from ..ContextManagers import DatabaseManager
from .. import warm
from .. import config

DEFAULT_DATA_DIR = "./data"
DEFAULT_JOBS_DIR = "./data/jobs/"


def write_jobs_handler(args: Namespace):
    """ Write jobs. """
    cfg = config.config.Config()
    jobs_dir = Path(DEFAULT_JOBS_DIR)
    data_dir = Path(DEFAULT_DATA_DIR)

    with open(jobs_dir / "sample_job.json", "w") as f:
        f.write(cfg.to_json(indent=4))

    # Job 1: small graphs in the ring family.
    # Estimated time: ~10 hours with 32 cores.
    cfg.targets = [
        config.Target(
            endTime=1_000_000_000,
            n=600,
            model=warm.WarmModel(
                is_graph=True,
                graph=warm.ring_2d_graph(i),
                model_id=f"ring_2d_{i}"
            ),
        ) for i in range(3, 11)
    ]
    cfg.csv_path = str(data_dir / "job1")

    with open(jobs_dir / "job1.json", "w") as f:
        f.write(cfg.to_json(indent=4))

    # Job 2: large graphs in the ring family.
    # Estimated time: ~31 hours with 32 cores.
    # Estimated time: ~21 hours with 48 cores.
    cfg.targets = [
        config.Target(
            endTime=1_000_000_000,
            n=500,
            model=warm.WarmModel(
                is_graph=True,
                graph=warm.ring_2d_graph(i),
                model_id=f"ring_2d_{i}"
            ),
        ) for i in range(11, 21)
    ]
    cfg.csv_path = str(data_dir / "job2")

    with open(jobs_dir / "job2.json", "w") as f:
        f.write(cfg.to_json(indent=4))

    # Job 3: small graphs in the donut family
    # Estimated time: unknown
    cfg.targets = [
        config.Target(
            endTime=1_000_000_000,
            n=200,
            model=warm.WarmModel(
                is_graph=True,
                graph=warm.donut_2d_graph(i),
                model_id=f"donut_2d_{i}"
            ),
        ) for i in range(2, 11)
    ]
    cfg.csv_path = str(data_dir / "job3")

    with open(jobs_dir / "job3.json", "w") as f:
        f.write(cfg.to_json(indent=4))

    # Job 4: large graphs in the donut family
    # Estimated time: unknown
    cfg.targets = [
        config.Target(
            endTime=1_000_000_000,
            n=200,
            model=warm.WarmModel(
                is_graph=True,
                graph=warm.donut_2d_graph(i),
                model_id=f"donut_2d_{i}"
            ),
        ) for i in range(11, 21)
    ]
    cfg.csv_path = str(data_dir / "job4")

    with open(jobs_dir / "job4.json", "w") as f:
        f.write(cfg.to_json(indent=4))


def import_handler(args: Namespace):
    """ Imports external data into the database. """
    with DatabaseManager(db_path=config.config.db_path) as db:
        pd.read_csv(Path(args.path) / "SimInfo.csv").to_sql("SimInfo", db, if_exists="append")
        pd.read_csv(Path(args.path) / "SimData.csv").to_sql("SimData", db, if_exists="append")


def export_handler(args: Namespace):
    with DatabaseManager(config.config.db_path) as db:
        pd.read_sql(f"""
            SELECT * FROM SimInfo
        """, db).to_csv(Path(args.path) / "SimInfo.csv", index=False)
        if args.full:
            pd.read_sql(f"""
                SELECT * FROM SimData
            """, db).to_csv(Path(args.path) / "SimData.csv", index=False)
        else:
            trials = pd.read_csv(Path(args.path) / "SimDataSummary.csv").to_dict("records")
            pd.DataFrame([pd.read_sql(f"""
                SELECT * FROM SimData
                WHERE simId = ? AND trialId = ? AND endTime > ?
            """, db, params=[d["simId"], d["trialId"], d["endTime"]])
                          for d in trials]).to_csv(Path(args.path) / "SimData.csv")


def summarise_handler(args: Namespace):
    with DatabaseManager(config.config.db_path) as db:
        pd.read_sql(f"""
            SELECT simId, trialId, max(endTime) AS endTime FROM SimData
            GROUP BY simId, trialId
            ORDER BY simId, trialId
        """, db).to_csv(Path(args.path) / "SimDataSummary.csv", index=False)


def analyser_handler(args: Namespace):
    pass


def main(argv: Optional[list[str]] = None):
    # Add the parser.
    parser = ArgumentParser("warmlab")
    subparsers = parser.add_subparsers(required=True)

    # Write parser.
    write_parser = subparsers.add_parser("write")
    write_parser.set_defaults(func=write_jobs_handler)

    # Import (data) parser.
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("path")
    import_parser.set_defaults(func=import_handler)

    # Export (data) parser.
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("path")
    export_parser.add_argument("--full", action="store_true")
    export_parser.set_defaults(func=export_handler)

    # Generate a data summary.
    summarise_parser = subparsers.add_parser("summarise")
    summarise_parser.add_argument("path")
    summarise_parser.set_defaults(func=summarise_handler)

    # Parse the arguments and invoke the relevant handler.
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
