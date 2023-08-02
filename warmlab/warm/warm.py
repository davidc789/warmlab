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


class JsonSerialisable(object, metaclass=abc.ABCMeta):
    """ A base class for implementing Json Serialisation. """
    __slots__ = ()

    @abc.abstractmethod
    def to_dict(self):
        """ A no-copy dictionary conversion.

        :return: The dictionary representation of the object.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, dct: dict):
        """ Constructs the object from a given dictionary form.

        :param dct: The dictionary format.
        """
        pass

    def to_json(self):
        """ Converts the class to a json string. """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, string: str):
        """ Constructs the object from a json serialisation.

        :param string: The json serialised format.
        """
        return json.loads(string, object_hook=cls.from_dict)


class WarmGraphResults(WarmResults):
    _graph: ig.Graph

    def __init__(self, problem: cp.Problem, graph: ig.Graph):
        super().__init__(problem)
        self._graph = graph

    def draw(self):
        fig, ax = plt.subplots()
        ig.plot(
            self._graph,
            vertex_size=20,
            vertex_label=['first', 'second', 'third', 'fourth'],
            edge_width=[1, 4],
            edge_color=['black', 'grey'],
            target=ax
        )


class WarmModel(JsonSerialisable):
    """ A generic WARM model specification. """
    bins: list[list[int]]                    # The grouping of elements.
    probs: list[float]                       # Probability of bins.
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
            self.probs = [1 / len(self.bins) for _ in range(len(self.bins))]
        else:
            self.probs = standardise(probs)

        # Post-init computations
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
        return {
            "probs": self.probs,
            "is_graph": self.is_graph,
            "graph": self.graph.to_dict_dict,
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
            graph = ig.Graph(dct["graph"])

        return cls(
            probs=dct["probs"],
            is_graph=is_graph,
            graph=graph,
            bins=dct["bins"],
            elem_count=dct["elem_count"]
        )


@dataclass(slots=True)
class WarmSimulationData(JsonSerialisable):
    model: WarmModel     # The model.
    root: str            # The starting point of the simulation.
    targetTime: int      # Target time.
    t: int               # Current time.
    counts: list[int]    # Current counts.
    x: list[float] = field(init=False)      # Current proportions.
    omegas: list[int] = field(init=False)   # Current omegas.

    def __post_init__(self):
        self.x = [x / (self.t + self.model.elem_count) for x in self.counts]
        self.omegas = calc_omega(self.model.bins, self.counts)

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
            "count": self.counts
        }


def simulate(sim: WarmSimulationData):
    """

    :param sim: The simulation data object.
    :param rep: Number of repetitions.
    """
    # The reversed map tells us which bins need to reevaluate when e is updated.
    rev_bin_map = sim.model.rev_bin_map
    data: list[BinDict] = [{
        "group": g,
        "omega": omega
    } for [g, omega] in zip(sim.model.bins, sim.omegas)]

    while sim.t <= sim.targetTime:
        bin_data: BinDict = np.random.choice(data, p=sim.model.probs)
        groups = bin_data["group"]
        weights = [sim.counts[i] / bin_data["omega"] for i in groups]

        # Selects a random edge and updates omegas based on
        e = np.random.choice(groups, weights)
        sim.counts[e] += 1
        omega_updates = calc_omega(rev_bin_map[e], sim.counts)
        for (i, omega) in zip(rev_bin_map[e], omega_updates):
            sim.omegas[i] = omega
        sim.t += 1

    return sim


def solve(model: WarmModel):
    """ Runs the solver on the graph to obtain the equilibrium.

    :param model: The model to solve.
    """
    x = cp.Variable(model.elem_count, name="x")
    A = model.incidence_matrix
    b = np.array(model.probs)
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


def calc_omega(bins: Sequence[Sequence[int]], counts: Sequence[int]) -> list[int]:
    return [sum(counts[i] for i in g) for g in bins]


def calc_rev_bin_map(bins: Sequence[Sequence[int]], e_count: int) -> list[list[int]]:
    return [[i for i, g in enumerate(bins) if e in g] for e in range(e_count)]


def standardise(vec: list[float]):
    return [x / sum(vec) for x in vec]


def ring_2d_graph(m: int):
    """ Constructs a 2D ring graph of index m. """
    edges = ([(i, (i + 1) % m) for i in range(m)]
             + [(i, (i + 1) % (2*m)) for i in range(m, 2 * m)]
             + [(i, i + m) for i in range(m)])
    graph = ig.Graph(n=2 * m, edges=edges)
    return graph


if __name__ == '__main__':
    m = 3
    graph = ring_2d_graph(m)
    model = WarmModel(is_graph=True, graph=graph)
    sim = simulate(WarmSimulationData(model=model, root="0.0", t=2*m,
                                      targetTime=10_000_000,
                                counts=[1 for _ in range(2*m)]))
