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
        sim = req.get_json()
        sim_data = warm.WarmSimulationData.from_json(sim)
        res_data = warm.simulate(sim_data)
        return func.HttpResponse(res_data.to_json())
    except ValueError as error:
        return func.HttpResponse(
            traceback.format_exc(),
            status_code=http.HTTPStatus.BAD_REQUEST
        )
