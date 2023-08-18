import traceback

from flask import Flask, request

import warm

app = Flask("simulate")


@app.post("/api/simulate")
def simulate_function():
    """ Runs simulation before returning the results back. """
    try:
        sim = warm.WarmSimulationData.from_dict(request.json)
        res = warm.simulate(sim)
        return res.to_dict(), 200
    except ValueError as error:
        return {
            "traceback": traceback.format_exc()
        }, 400


if __name__ == '__main__':
   app.run()
