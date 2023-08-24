import json
from typing import TypedDict, Generic, TypeVar, Optional

import warm
import cvxpy as cp
import sqlite3
import pandas as pd
import networkx as nx

from dash import Dash, dcc, html, Input, Output, callback
import dash_cytoscape as cyto
import plotly.express as px

from DataManager import DataManager
from config import config

# Canvas dimension, in px.
width = 1048
height = 576


T = TypeVar("T")


class PositionDict(TypedDict):
    x: int
    y: int


class NodeDict(TypedDict):
    id: Optional[str]
    name: Optional[str]
    label: Optional[str]
    name: Optional[str]
    value: Optional[str]


class EdgeDict(TypedDict):
    label: Optional[str]
    source: str
    target: str


class CytoElement(TypedDict):
    data: dict
    position: Optional[PositionDict]


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


def warm_analyse_mean(simId: str):
    # Fetch data.
    with DataManager(config.DB_LOCATION, use_dict_factory=True) as db:
        cursor: sqlite3.Cursor
        cursor = db.cursor()
        cursor.execute(f"""
            SELECT trialId, endTime, root, counts, model, solution
            FROM SimData
            INNER JOIN SimInfo USING (simId)
            WHERE simId = '{simId}'
        """)
        sims = [warm.WarmSimData.from_dict({
            "model": json.loads(x["model"]),
            "root": x["root"],
            "t": x["endTime"],
            "targetTIme": x["endTime"],
            "trialId": x["trialId"],
            "counts": json.loads(x["counts"]),
            "solution": json.loads(x["solution"])
        }) for x in cursor.fetchall()]

    # TODO: fix spelling error
    proportions = [x.x for x in sims]

    # Convert everything to cytoscape. Phew.
    graph = sims[0].model.graph
    nodes: list[NodeDict]
    edges: list[EdgeDict]
    nodes, edges = graph.to_dict_list(use_vids=False)

    # Change 'name' into 'id' for the nodes. I know.
    for node in nodes:
        node["id"] = node["name"]
        node["label"] = node["name"]
        del node["name"]  # For efficiency

    # Add edge labels.
    for i, edge in enumerate(edges):
        edge["label"] = str(i)

    # Compute layouts. Default layout should be between 0 and 1 so scales can
    # be applied consistently for a canvas of arbitrary size.
    layout = graph.layout()
    positions = [PositionDict(x=coord[0] * width, y=coord[1] * height) for coord in layout]

    # Wrap into Cyto and output the result.
    elements: list[CytoElement] = [CytoElement(
        data=node,
        position=position
    ) for node, position in zip(nodes, positions)] + [CytoElement(
        data=edge
    ) for edge in edges]
    print(elements)
    return elements


# Sets up the dash app.
app = Dash(__name__)
app.layout = html.Div([
    html.Header("Warmlab"),
    cyto.Cytoscape(
        id="cytoscape",
        elements=warm_analyse_mean("ring_2d_3"),
        layout={"name": "preset"},
        style={"width": f"{width}px", "height": f"{height}px"}
    )
])


# @callback(
#     Output('graph-with-slider', 'figure'),
#     Input('year-slider', 'value'))
# def update_figure(selected_year):
#     filtered_df = df[df.year == selected_year]
#
#     fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
#                      size="pop", color="continent", hover_name="country",
#                      log_x=True, size_max=55)
#
#     fig.update_layout(transition_duration=500)
#
#     return fig


if __name__ == '__main__':
    # with DataManager(config.DB_LOCATION) as db:
    #     warm_analyse_convergence(db, "ring_2d_3")
    app.run_server(debug=True)
