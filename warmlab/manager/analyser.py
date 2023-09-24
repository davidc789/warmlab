import logging
import sqlite3
from typing import TypedDict, TypeVar, Optional, NamedTuple

import dash_cytoscape as cyto
import plotly.express as px
import pandas as pd

from dash import Dash, dcc, html, Input, Output, callback

from .. import warm
from ..ContextManagers import DatabaseManager
from ..config import config

logger = logging.getLogger(__name__)

# Canvas dimension.
width = 400
height = 400


T = TypeVar("T")
SimMap = dict[tuple[str, int, int], warm.WarmSimData]
GraphFamilies = ["ring_2d"]


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


class InfoTuple(NamedTuple):
    model: warm.WarmModel
    solution: warm.WarmSolution


def escape_string(string: str, escape_symb: str = "\\", escape_chars: list[str] = ("%", "_")):
    """ Escape the original string. """
    out_buf = []

    for s in string:
        if s in escape_chars:
            out_buf.append(escape_symb)
        out_buf.append(s)

    return "".join(out_buf)


def parse_model(model: warm.WarmModel) -> list[CytoElement]:
    nodes: list[NodeDict]
    edges: list[EdgeDict]
    nodes, edges = model.graph.to_dict_list(use_vids=False)

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
    layout = warm.ring_2d_layout(model.bins_count // 2)
    positions = [PositionDict(x=coord[0] * width, y=coord[1] * height) for coord in layout]

    # Wrap into Cyto and output the result.
    elements: list[CytoElement] = [CytoElement(
        data=node,
        position=positions[int(node["id"])]
    ) for node in nodes] + [CytoElement(
        data=edge
    ) for edge in edges]
    return elements


def fetch_data(graph_family_selected: str) -> tuple[pd.DataFrame, dict[str, InfoTuple]]:
    if graph_family_selected in GraphFamilies:
        # Fetch data.
        with DatabaseManager(config.db_path) as db:
            logger.info("Data fetching commenced")
            id_pattern = escape_string(graph_family_selected) + "%"
            df = pd.read_sql(f"""
                SELECT simId, trialId, endTime, root, counts
                FROM SimData
                WHERE simId LIKE ? ESCAPE '\\'
            """, con=db, params=[id_pattern])

            cursor: sqlite3.Cursor
            cursor = db.cursor()
            cursor.execute(f"""
                SELECT simId, model, solution
                FROM SimInfo
                WHERE simId LIKE ? ESCAPE '\\'
            """, [id_pattern])
            sims = {
                x[0]: InfoTuple(
                    model=warm.WarmModel.from_json(x[1]),
                    solution=warm.WarmSolution.from_json(x[2])
                ) for x in cursor.fetchall()
            }
            logger.info("Data fetched and transformed successfully")
            return df, sims
    else:
        logger.warning("Don't try to hack me! I am vulnerable.")


def update_slider_props(df: pd.DataFrame):
    model_indices = df["endTime"].unique()
    marks = {x: str(x) for x in range(min(model_indices), max(model_indices) + 1)}
    return min(model_indices), max(model_indices), marks


# Construct the default values.
df, sims = fetch_data("ring_2d")


# Sets up the dash app.
app = Dash(__name__)
app.layout = html.Div([
    dcc.Store(id="simId-store", storage_type="local"),
    html.Header("Warmlab"),
    dcc.Dropdown(
        GraphFamilies,
        "ring_2d",
        id="graph-family-dropdown"
    ),
    dcc.Slider(
        min=update_slider_props(df)[0],
        max=update_slider_props(df)[1],
        step=1,
        marks=update_slider_props(df)[2],
        value=5,
        id="family-index-slider"
    ),
    cyto.Cytoscape(
        id="graph-canvas",
        elements=parse_model(sims["ring_2d_5"].model),
        layout={"name": "preset"},
        style={"width": f"{width}px", "height": f"{height}px"}
    ),
    dcc.Graph(
        id="mean-conv-graph"
    ),
    dcc.Graph(
        id="var-conv-graph"
    )
])


# @callback(Output("data-store", "data"),
#           Input("graph-family-dropdown", "value"),
#           background=True)
# def fetch_data_cb(graph_family_selected: str):
#     return fetch_data(graph_family_selected)


@callback(Output("simId-store", "data"),
          Input("family-index-slider", "value"))
def update_sim_id(value: int):
    return f"ring_2d_{value}"


@callback(Output("graph-canvas", "elements"),
          Input("simId-store", "data"))
def render_cyto_graph(sim_id: str):
    # Recompute the graph layout.
    return parse_model(sims[sim_id].model)


@callback(Output("mean-conv-graph", "figure"),
          Output("var-conv-graph", "figure"),
          Input("simId-store", "data"))
def render_convergence_plot(sim_id: str):
    # Filter.
    filtered_df = df[df["simId"] == sim_id].copy()

    # Transform.
    filtered_df["counts"] = filtered_df["counts"].str.strip("[]")
    ncol = filtered_df["counts"].str.split(",").transform(len).max()
    x_names = [f"x_{i}" for i in range(ncol)]
    filtered_df.loc[x_names] = filtered_df["counts"].str.split(',', expand=True).astype("int")
    filtered_df.loc[x_names] = filtered_df[x_names].div(filtered_df["endTime"] + ncol, axis=0)
    mean_df = filtered_df.groupby("endTime")[x_names].mean().reset_index()
    var_df = filtered_df.groupby("endTime")[x_names].var().reset_index()

    # Plot.
    mean_conv_fig = px.scatter(mean_df, x="endTime", y="x_0")
    var_conv_fig = px.scatter(var_df, x="endTime", y="x_0")
    return mean_conv_fig, var_conv_fig


if __name__ == '__main__':
    # with DataManager(config.DB_LOCATION) as db:
    #     warm_analyse_convergence(db, "ring_2d_3")
    app.run_server(debug=True)
