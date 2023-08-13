""" Implements the WARM model. """
import abc
import logging


from collections.abc import Sequence
from typing import TypedDict, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
import json

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import igraph as ig
import cvxpy as cp

logger = logging.getLogger(__name__)


class BinDict(TypedDict):
    group: list[int]
    omega: float


class Action(TypedDict):
    trial_number: Optional[int]
    t_max: float

Actions = list[Action]

def fast_choose(q: float, arr: list[int], p: list[float]):
    cp = np.cumsum(p)

    for i, x in enumerate(cp):
        if x >= q:
            return arr[i]

    raise ValueError(f"q={q} is out of [0, 1]")


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


class WarmModel(object):
    """ A generic WARM model specification. """
    bins: list[list[int]]                    # The grouping of elements.
    probs: np.ndarray                        # Probability of bins.
    bins_count: int                          # Number of bins.
    elem_count: int                          # Number of elements.
    is_graph: bool                           # Whether the model is graph-based.
    graph: Optional[ig.Graph]                # If is_graph, gives the graph.
    _inc_matrix: Optional[np.ndarray]        # The incidence matrix.
    rev_bin_map: list[list[int]]             # The reversed bin mapping.

    def __init__(self, probs: Optional[list[float]] = None,
                 is_graph: bool = False,
                 graph: Optional[ig.Graph] = None, bins: list[list[int]] = None,
                 elem_count: Optional[int] = None,
                 weight_func: Optional[Callable[[int], float]] = None):
        """ Constructs a WARM model.

        Two construction methods are currently supported.
        For graph-based models, use is_graph = True and specify the graph using
        the graph parameter; for general models, specify bins, elem_count and
        weight_func manually.

        :param probs: Corresponds to p_A in a WARM model.
        :param is_graph: Whether the model is a graph model.
        :param graph: The graph specification of the model.
        :param bins: Corresponds to A in a WARM model.
        :param elem_count: The number of elements in the model.
        :param weight_func: Corresponds to W in a WARM model.
        """
        if is_graph:
            if graph is None:
                raise ValueError("graph must be supplied with is_graph true")

            self.bins = graph.get_inclist()
            self.graph = graph
            self.elem_count = graph.ecount()
        else:
            if weight_func is None:
                def weight_func(x):
                    return x

                # TODO: weight func support

            self.bins = bins
            self.elem_count = elem_count

        if probs is None:
            probs = np.ones(len(self.bins))
        else:
            probs = np.array(probs)
        self.probs = probs / np.sum(probs)

        # Post-init computations
        self.is_graph = is_graph
        self.bins_count = len(self.bins)
        self._inc_matrix = None
        self.rev_bin_map = calc_rev_bin_map(self.bins, self.elem_count)

    @property
    def incidence_matrix(self) -> np.ndarray:
        """ The incidence matrix representation of the model, lazily computed. """
        if self._inc_matrix is None:
            def _to_flag(l, m):
                arr = [0 for _ in range(m)]
                for i in l:
                    arr[i] = 1
                return arr

            self._inc_matrix = np.array([_to_flag(b, self.elem_count) for b in self.bins])

        return self.incidence_matrix

    def to_dict(self):
        """ A no-copy dictionary conversion. """
        if self.is_graph:
            graph_dict = self.graph.to_dict_dict()
        else:
            graph_dict = None
        return {
            "probs": self.probs.tolist(),
            "is_graph": self.is_graph,
            "graph": graph_dict,
            "bins": self.bins,
            "elem_count": self.elem_count
        }

    @classmethod
    def from_dict(cls, dct: dict):
        """ Constructs a WARM model from a given dictionary form.

        :param dct: The dictionary format.
        """
        is_graph = dct["is_graph"]
        graph: Optional[ig.Graph] = None

        if is_graph:
            graph = ig.Graph.DictDict(dct["graph"])

        return cls(
            probs=dct["probs"],
            is_graph=is_graph,
            graph=graph,
            bins=dct["bins"],
            elem_count=dct["elem_count"]
        )

    def to_json(self, *args, **kargs):
        """ Converts the class to a json string. """
        return json.dumps(self.to_dict(), *args, **kargs)

    @classmethod
    def from_json(cls, string: str, *args, **kargs):
        """ Constructs the object from a json serialisation.

        :param string: The json serialised format.
        """
        return cls.from_dict(json.loads(string, *args, **kargs))


@dataclass(slots=True)
class WarmSimulationData(object):
    model: WarmModel        # The model.
    root: str               # The starting point of the simulation.
    targetTime: int         # Target time.
    t: int                  # Current time.
    trialId: Optional[int] = None          # The trialId. Used only for tagging.
    counts: Optional[list[int]] = None     # Current counts.
    x: list[float] = field(init=False)     # Current proportions.
    omegas: list[int] = field(init=False)  # Current node sums.

    def __post_init__(self):
        if self.counts is None:
            self.counts = [1 for _ in range(self.model.elem_count)]
        self.x = [x / (self.t + self.model.elem_count) for x in self.counts]
        self.omegas = [sum(self.counts[i] for i in g) for g in self.model.bins]

    @classmethod
    def from_dict(cls, dct: dict):
        """ Loads the model from a dictionary representation.

        :param dct: The dictionary to load from.
        """
        return cls(
            model=WarmModel.from_dict(dct["model"]),
            root=dct["root"],
            targetTime=dct["n"],
            t=dct["t"],
            trialId=dct["trialId"],
            counts=dct["count"]
        )

    def to_dict(self):
        """ Converts the model to a dictionary representation.

        :return: The dictionary representation.
        """
        return {
            "model": self.model.to_dict(),
            "root": self.root,
            "n": self.targetTime,
            "t": self.t,
            "trialId": self.trialId,
            "count": self.counts
        }

    def to_json(self, *args, **kargs):
        """ Converts the class to a json string. """
        return json.dumps(self.to_dict(), *args, **kargs)

    @classmethod
    def from_json(cls, string: str, *args, **kargs):
        """ Constructs the object from a json serialisation.

        :param string: The json serialised format.
        """
        return cls.from_dict(json.loads(string, *args, **kargs))


def simulate(sim: WarmSimulationData):
    """

    :param sim: The simulation data object.
    :param rep: Number of repetitions.
    """
    # The reversed map tells us which bins need to reevaluate when e is updated.
    rev_bin_map = sim.model.rev_bin_map
    vertices = np.array(range(sim.model.bins_count))
    vs: np.ndarray = np.random.choice(vertices, size=sim.targetTime - sim.t, p=sim.model.probs)
    qs = np.random.rand(sim.targetTime - sim.t)

    for v, q in zip(vs, qs):
        groups, omega = sim.model.bins[v], sim.omegas[v]
        weights = [sim.counts[i] / omega for i in groups]

        # Selects a random edge and updates omegas based on
        e = fast_choose(q, groups, p=weights)
        sim.counts[e] += 1
        for i in rev_bin_map[e]:
            sim.omegas[i] += 1

    sim.t = sim.targetTime
    return sim


def solve(model: WarmModel):
    """ Runs the solver on the graph to obtain the equilibrium.

    :param model: The model to solve.
    """
    x = cp.Variable(model.elem_count, name="x")
    A = model.incidence_matrix
    b = model.probs
    omega = A @ x

    # Construct the problem.
    objective = (cp.sum(x)
                 - cp.sum(cp.multiply(cp.log(omega), np.array(b))))
    constraints = [cp.sum(x) == 1, omega >= b, x >= 0]
    prb = cp.Problem(cp.Minimize(objective), constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prb.solve()

    # The optimal value for x is stored in `x.value`.
    values = [y.value for y in x]
    print(values)

    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    print(constraints[0].dual_value)
    print(constraints[1].dual_value)
    print(constraints[2].dual_value)

    return result, values


# TODO
def analyze(model: WarmModel, x0: Sequence[int]):
    """
    The analyzer only works in the linear case.

    :param model: The model.
    :param x0: The initial starting point.
    :return: Analysed result.
    """
    rev_bin_map = model.rev_bin_map

    def f(t: float, x: np.array) -> np.array:
        omegas = np.array([sum(x[e] for e in g) for g in model.bins])
        return -x - np.array([xe * sum(model.probs[u] / omegas[u] for u in rev_bin_map[e])
                              for e, xe in enumerate(x)])

    return solve_ivp(fun=f, t_span=[0, 1e-3], y0=x0, max_step=1e-8)


def calc_rev_bin_map(bins: Sequence[Sequence[int]], e_count: int) -> list[list[int]]:
    return [[i for i, g in enumerate(bins) if e in g] for e in range(e_count)]


def ring_2d_graph(m: int):
    """ Constructs a 2D ring graph of index m.

    :param m: Index of the graph.
    :return: The constructed graph.
    """
    outer_edges = [(i, (i + 1) % m) for i in range(m)]
    edges = (outer_edges + [(i + m, j + m) for i, j in outer_edges]
             + [(i, i + m) for i in range(m)])
    graph = ig.Graph(n=2 * m, edges=edges)
    return graph


def grid(m: int):
    """ Creates a 2D grid graph.

    :param m: Index of the graph.
    :return: The constructed graph.
    """
    # Horizontal + Vertical
    edges = ([(m * j + i, m * j + i + 1) for i in range(m - 1) for j in range(m)]
           + [(m * j + i, m * j + i + m) for i in range(m) for j in range(m - 1)])
    graph = ig.Graph(n = m * m, edges=edges)
    return graph


def donut_2d_graph(m: int):
    """ Creates a 2D donut graph.

    :param m: Index of the graph.
    :return: The constructed graph.
    """
    # Horizontal + Vertical
    horizontals = [(i, (i + 1) % m) for i in range(m)]
    verticals = [(i * m, j * m) for (i, j) in horizontals]
    edges = ([(m * k + i, m * k + j) for (i, j) in horizontals for k in range(m)]
           + [(i + k, j + k) for (i, j) in verticals for k in range(m)])
    graph = ig.Graph(n=m * m, edges=edges)
    return graph


if __name__ == '__main__':
    m = 3
    graph = ring_2d_graph(m)
    model = WarmModel(is_graph=True, graph=graph)
    sim = simulate(WarmSimulationData(model=model, root="0.0", t=0,
                                      targetTime=1_000_000))
    print(sim.to_json(indent=True))
