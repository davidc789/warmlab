""" Implements the WARM model. """

import abc
import logging
from math import pi
from itertools import chain

from collections.abc import Sequence
from typing import TypedDict, Optional, Callable
from dataclasses import dataclass, field, asdict
import json

import igraph
import numpy as np
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
    i = 0
    acc = 0
    while i < len(p):
        acc += p[i]

        if acc >= q:
            return arr[i]

        i += 1
    raise ValueError(f"q={q} is out of [0, 1]")


class JsonSerialisable(object, metaclass=abc.ABCMeta):
    """ A mixin for json-serialisation. """
    __slots__ = ()

    @abc.abstractmethod
    def to_dict(self):
        """ Converts the object to a dictionary representation.

        :return: The dictionary representation.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, dct: dict):
        """ Loads the object from a dictionary representation.

        :param dct: The dictionary to load from.
        """
        pass

    def to_json(self, *args, **kargs):
        """ Converts the class to a json string.

        :return: The json string representation.
        """
        return json.dumps(self.to_dict(), *args, **kargs)

    @classmethod
    def from_json(cls, string: str, *args, **kargs):
        """ Constructs the object from a json serialisation.

        :param string: The json serialised format.
        """
        return cls.from_dict(json.loads(string, *args, **kargs))


class WarmModel(JsonSerialisable):
    """ A generic WARM model specification. """
    id: Optional[str]                        # The model ID, for tagging purpose.
    probs: np.ndarray                        # Probability of bins.
    is_graph: bool                           # Whether the model is graph-based.
    graph: Optional[ig.Graph]                # If is_graph, gives the graph.
    bins: list[list[int]]                    # The grouping of elements.
    bins_count: int                          # Number of bins.
    elem_count: int                          # Number of elements.
    _inc_matrix: Optional[np.ndarray] = None         # The incidence matrix.
    _rev_bin_map: Optional[list[list[int]]] = None   # The reversed bin mapping.

    def __init__(self, model_id: Optional[str] = None, probs: Optional[list[float]] = None,
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
        self.id = model_id

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

        return self._inc_matrix

    @property
    def rev_bin_map(self) -> list[list[int]]:
        """ The reverse-bin-map, lazily computed. """
        if self._rev_bin_map is None:
            self._rev_bin_map = calc_rev_bin_map(self.bins, self.elem_count)

        return self._rev_bin_map

    def to_dict(self):
        """ A no-copy dictionary conversion. """
        if self.is_graph:
            graph_dict = self.graph.to_dict_dict()
        else:
            graph_dict = None
        return {
            "id": self.id,
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
            model_id=dct["id"],
            probs=dct["probs"],
            is_graph=is_graph,
            graph=graph,
            bins=dct["bins"],
            elem_count=dct["elem_count"]
        )


@dataclass(slots=True, frozen=True)
class WarmSolution(JsonSerialisable):
    obj_value: float
    duals: tuple[float, list[float], float]
    x_opt: list[float]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)


@dataclass(slots=True)
class WarmSimData(JsonSerialisable):
    model: WarmModel        # The actual model.
    root: str               # The starting point of the simulation.
    targetTime: int         # Target time.
    t: int                  # Current time.
    trialId: Optional[int] = None            # The trialId. Used only for tagging.
    counts: Optional[list[int]] = None       # Current counts.
    solution: Optional[WarmSolution] = None  # Solution data.
    x: list[float] = field(init=False)       # Current proportions.
    omegas: list[int] = field(init=False)    # Current node sums.

    def __post_init__(self):
        if self.counts is None:
            self.counts = [1 for _ in range(self.model.elem_count)]
        self.x = [x / (self.t + self.model.elem_count) for x in self.counts]
        self.omegas = [sum(self.counts[i] for i in g) for g in self.model.bins]

    @classmethod
    def from_dict(cls, dct: dict):
        solution = None
        if dct["solution"] is not None:
            solution = WarmSolution.from_dict(dct["solution"])

        return cls(
            model=WarmModel.from_dict(dct["model"]),
            root=dct["root"],
            targetTime=dct["targetTIme"],
            t=dct["t"],
            trialId=dct["trialId"],
            counts=dct["counts"],
            solution=solution
        )

    def to_dict(self):
        solution_dict = None
        if self.solution is not None:
            solution_dict = self.solution.to_dict()

        return {
            "model": self.model.to_dict(),
            "root": self.root,
            "targetTIme": self.targetTime,
            "t": self.t,
            "trialId": self.trialId,
            "counts": self.counts,
            "solution": solution_dict
        }


def simulate(sim: WarmSimData):
    """ Conducts simulation as directed.

    :param sim: The simulation data object.
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

    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    return WarmSolution(
        obj_value=result,
        duals=(constraints[0].dual_value, constraints[1].dual_value.tolist(), constraints[2].dual_value.tolist()),
        x_opt=values
    )


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


def ring_2d_layout(m: int, r1: float = 0.5, r2_to_r1: float = 0.5, x_offset: int = 0, y_offset: int = 0):
    """ An optimal layout for the ring_2d family.

    :param m: Index of the graph.
    :param r1: Outer radius.
    :param r2_to_r1: Inner radius as a fraction of outer radius.
    :param x_offset: Shift in the x direction.
    :param y_offset: Shift in the y direction.
    """
    r2 = r2_to_r1 * r1

    omegas = np.linspace(0, 2 * pi, m)
    xs_outer = r1 * np.cos(omegas) + x_offset
    xs_inner = r2 * np.cos(omegas) + x_offset
    ys_outer = r1 * np.sin(omegas) + y_offset
    ys_inner = r1 * np.sin(omegas) + y_offset

    return igraph.Layout([(x, y) for x, y in zip(
        chain(xs_outer, xs_inner), chain(ys_outer, ys_inner))])


def grid(m: int):
    """ Creates a 2D grid graph.

    :param m: Index of the graph.
    :return: The constructed graph.
    """
    # Horizontal + Vertical
    edges = ([(m * j + i, m * j + i + 1) for i in range(m - 1) for j in range(m)]
             + [(m * j + i, m * j + i + m) for i in range(m) for j in range(m - 1)])
    graph = ig.Graph(n=m * m, edges=edges)
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

    sim = simulate(WarmSimData(model=model, root="0.0", t=0,
                               targetTime=1_000_000))
    sim.solution = solve(model)
    print(sim.to_json(indent=True))
