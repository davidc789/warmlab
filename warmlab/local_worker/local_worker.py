import json
import sys
import traceback
import socket

import warm


async def remote_worker(ip: str, port: int):
    # ip = '10.0.0.1'  # Server ip
    # port = 4000

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((ip, port))

    print("Server Started")

    while True:
        data, addr = s.recvfrom(1024)
        data = data.decode('utf-8')
        print("Message from: " + str(addr))
        print("From connected user: " + data)
        data = data.upper()
        print("Sending: " + data)
        s.sendto(data.encode('utf-8'), addr)

    c.close()

async def local_worker():
    """

    :return: Exit code.
    """
    try:
        sim = warm.WarmSimulationData.from_json(sys.stdin.read())
        warm.simulate(sim)
        print(sim.to_json())
    except json.JSONDecodeError:
        traceback.format_exc()
        return 1

    return 0


def main(args):
    local_worker()


if __name__ == '__main__':
    main(sys.argv)
