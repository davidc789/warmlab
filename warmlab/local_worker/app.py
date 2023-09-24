import traceback

from flask import Flask, request

from .. import warm

app = Flask("simulate")


@app.post("/api/simulate")
def simulate_function():
    """ Runs simulation before returning the results back. """
    try:
        sim = warm.WarmSimData.from_dict(request.json)
        res = warm.simulate(sim, None)
        return res.to_dict(), 200
    except ValueError as error:
        return {
            "traceback": traceback.format_exc()
        }, 400


if __name__ == '__main__':
   app.run()
