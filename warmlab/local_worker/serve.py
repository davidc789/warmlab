from argparse import ArgumentParser
from subprocess import Popen
from typing import Optional

from app import app


def serve_multiprocess(base_url: str, n_process: int = 4, ports: list[int] = None):
    proc_pool: list[Popen] = []

    try:
        if ports is None:
            ports = range(8080 + 1 - n_process, 8080 + 1)
        for port in ports:
            proc = Popen(f"waitress-serve --host {base_url} --port {port} app:app")
            proc_pool.append(proc)
            print(f" - Serving on {base_url} port {port}")
        for proc in proc_pool:
            proc.wait()
    except KeyboardInterrupt:
        try:
            print(" - Killing all the processes...")
            for proc in proc_pool:
                proc.kill()
            for proc in proc_pool:
                proc.wait()
        except KeyboardInterrupt:
            print(" - Terminating everything...")
            for proc in proc_pool:
                proc.terminate()
            for proc in proc_pool:
                proc.wait()


def main(args: Optional[list[str]] = None):
    # Creates the CLI parser.
    parser = ArgumentParser("serve")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")

    # Parse the CLI arguments.
    args = parser.parse_args(args)

    serve_multiprocess(base_url=args.host, n_process=args.workers)


if __name__ == "__main__":
    main()
