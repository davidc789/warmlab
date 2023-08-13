""" Runs the simulation on Azure. """
import http
import traceback

import azure.functions as func
import warm

app = func.FunctionApp()

@app.function_name(name="SimulationTrigger")
@app.route(route="simulate")
def simulate_function(req: func.HttpRequest) -> func.HttpResponse:
    """ Runs simulation before returning the results back. """
    try:
        sim_json = req.get_json()
        sim = warm.WarmSimulationData.from_dict(sim_json)
        res = warm.simulate(sim)
        return func.HttpResponse(res.to_json())
    except ValueError as error:
        return func.HttpResponse(
            traceback.format_exc(),
            status_code=http.HTTPStatus.BAD_REQUEST
        )
