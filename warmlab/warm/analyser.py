import json

import warm
import cvxpy as cp
import sqlite3

from dash import Dash, html
import dash_cytoscape as cyto


class WarmResults(object):
    _problem: cp.Problem
    _vardict: dict[str, cp.Variable]
    _x: list[int]

    def __init__(self, problem: cp.Problem):
        self._problem = problem
        self._vardict = problem.var_dict
        self._x = problem.var_dict["x"].value

    @property
    def x(self):
        return self._x

    @property
    def problem(self):
        return self._problem

class WarmResultAnalyser(object):
    pass


def warm_analyse_mean(db: sqlite3.Connection, simId: str):
    # Fetch data
    cursor: sqlite3.Cursor
    with db.cursor() as cursor:
        cursor.execute(f"""
            SELECT trialId, endTime, root, counts, model, solution
            FROM SimData
            INNER JOIN SimInfo USING (simId);
        """)
        data = cursor.fetchall()

        warm.WarmSimulationData(
            model=data["model"],
            root=data["root"],
            targetTime=data["targetTime"],
            t=data["t"],
            trialId=data["trailId"]
        )

        [x["trialId"] for x in data]
        json.loads([x["counts"] for x in data])


def warm_analyse_covariance(db: sqlite3.Cursor, simId: str):
    pass


# class WarmGraphResults(WarmResults):
#     _graph: ig.Graph
#
#     def __init__(self, problem: cp.Problem, graph: ig.Graph):
#         super().__init__(problem)
#         self._graph = graph
#
#     def draw(self):
#         fig, ax = plt.subplots()
#         ig.plot(
#             self._graph,
#             vertex_size=20,
#             vertex_label=['first', 'second', 'third', 'fourth'],
#             edge_width=[1, 4],
#             edge_color=['black', 'grey'],
#             target=ax
#         )


if __name__ == '__main__':
    # with DataManager(config.DB_LOCATION) as db:
    #     warm_analyse_convergence(db, "ring_2d_3")
    app = Dash(__name__)

    app.layout = html.Div([
        # html.P("Dash Cytoscape:"),
        cyto.Cytoscape(
            id='cytoscape',
            elements=[
                {'data': {'id': 'ca', 'label': 'Canada'}},
                {'data': {'id': 'on', 'label': 'Ontario'}},
                {'data': {'id': 'qc', 'label': 'Quebec'}},
                {'data': {'source': 'ca', 'target': 'on'}},
                {'data': {'source': 'ca', 'target': 'qc'}}
            ],
            layout={'name': 'breadthfirst'},
            style={'width': '400px', 'height': '500px'}
        )
    ])

    app.run_server(debug=True)
