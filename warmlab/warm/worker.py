""" Sub-process workers for the main. """

import warm

import logging
import sys
import traceback

logger = logging.Logger(__name__)


def main():
    """ Main. """
    logger.info(" ! Worker ready")
    try:
        sim = warm.WarmSimData.from_json(sys.stdin.buffer.read().decode())
        res = warm.simulate(sim)
        sys.stdout.buffer.write(res.to_json().encode())
    except InterruptedError:
        return 1

    logger.info(" ! Worker exiting")
    return 0


if __name__ == '__main__':
    main()
