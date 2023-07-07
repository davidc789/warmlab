#%% Imports
from collections.abc import Sequence
from typing import TypedDict, Optional, Callable

import cvxpy
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import igraph as ig
import plotly.express as px
import cvxpy as cp


class BinDict(TypedDict):
    group: list[int]
    omega: float


class WarmResults(object):
    _problem: cvxpy.Problem
    _vardict: dict[str, cvxpy.Variable]
    _x: list[int]

    def __init__(self, problem: cvxpy.Problem):
        self._problem = problem
        self._vardict = problem.var_dict
        self._x = problem.var_dict["x"].value

    @property
    def x(self):
        return self._x

    @property
    def problem(self):
        return self._problem


class WarmGraphResults(WarmResults):
    _graph: ig.Graph

    def __init__(self, problem: cvxpy.Problem, graph: ig.Graph):
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


class WarmModel(object):
    """ An immutable object specifying a WA-Reinforcement Model (WARM).

    """
    _weightFunc: Callable[[int], float]
    _bins: list[list[int]]
    _probs: list[float]
    _incidence_matrix: np.array
    _is_graph: bool

    def __init__(self, bins: list[list[int]], elem_count: int,
                 probs: Optional[list[float]] = None,
                 weight_func: Optional[Callable[[int], float]] = None):
        """

        :param bins: Corresponds to A in a WARM model.
        :param probs: Corresponds to p_A in a WARM model.
        :param weight_func: Corresponds to W in a WARM model.
        """
        if weight_func is None:
            def weight_func(x):
                return x
        if probs is None:
            probs = [1 / len(bins) for _ in range(len(bins))]

        self._weightFunc = weight_func
        self._bins = bins
        self._probs = [x / sum(probs) for x in probs]
        self._elem_count = elem_count

    @property
    def probs(self):
        return self._probs

    @property
    def weightFunc(self):
        return self._weightFunc

    @property
    def bins(self):
        return self._bins

    @property
    def inc_matrix(self):
        """ Gives the dense incidence matrix.

        Warning: This property is lazy, which can be costly in the first call.
        """
        if self._incidence_matrix is None:
            def _to_flag(l, m):
                arr = [0 for _ in range(m)]
                for i in l:
                    arr[i] = 1
                return arr
            self._incidence_matrix = np.array([_to_flag(b, self._elem_count) for b in self._bins])

        return self._incidence_matrix

    @property
    def sp_inc_matrix(self):
        """ Gives the sparse incidence matrix.

        Warning: This property is lazy, which can be costly in the first call.
        :return:
        """
        # TODO
        raise NotImplemented

    # TODO: Output the result to a db
    # TODO: Store it somewhere
    def simulate(self,
                 initial_counts: Optional[Sequence[int]] = None,
                 n: int = 100000,
                 rep: int = 5):
        if initial_counts is None:
            initial_counts = np.repeat(1, len(self._bins))

        # The reversed map tells us which bins need to reevaluate when e is updated.
        reverse_bin_map = calc_rev_bin_map(self._bins, len(initial_counts))
        results = []
        for _rep in range(rep):
            omegas = calc_omega(self._bins, initial_counts, self._weightFunc)
            data: list[BinDict] = [{
                "group": g,
                "omega": omega
            } for [g, omega] in zip(self._bins, omegas)]
            counts = initial_counts.copy()
            for _i in range(n):
                bin_data: BinDict = np.random.choice(data, p=self._probs)
                groups = bin_data["group"]
                weights = [self._weightFunc(counts[i]) / bin_data["omega"] for i in groups]

                # Selects a random edge and update omegas based on
                e = np.random.choice(groups, weights)
                counts[e] += 1
                omega_updates = calc_omega(reverse_bin_map[e], counts, self._weightFunc)
                for (i, omega) in zip(reverse_bin_map[e], omega_updates):
                    omegas[i] = omega

            results.append(np.array(counts) / (n + sum(initial_counts)))

        return results

    def solve(self):
        x = cp.Variable(self._elem_count, name="x")
        A = self.inc_matrix
        b = np.array(self._probs)
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
    def analyze(self, x0: Sequence[int]):
        """
        The analyzer only works in the linear case.

        :param x0:
        :return:
        """
        rev_bin_map = calc_rev_bin_map(self._bins, len(x0))

        def f(t: float, x: np.array) -> np.array:
            omegas = np.array([sum(x[e] for e in g) for g in self._bins])
            return -x - np.array([xe * sum(self._probs[u] / omegas[u] for u in rev_bin_map[e])
                                  for e, xe in enumerate(x)])

        return solve_ivp(fun=f, t_span=[0, 1e-3], y0=x0, max_step=1e-8)


class WarmGraph(WarmModel):
    def __init__(self, graph: ig.Graph, probs: Optional[Sequence[float]] = None,
                 weight_func: Optional[Callable[[int], float]] = None):
        """

        :param bins: Corresponds to A in a WARM model.
        :param probs: Corresponds to p_A in a WARM model.
        :param weight_func: Corresponds to W in a WARM model.
        """
        super().__init__(graph.get_inclist(), graph.ecount(), probs, weight_func)


def calc_omega(bins: Sequence[Sequence[int]], counts: Sequence[int],
               weight_func: Callable) -> list[int]:
    return [sum(weight_func(counts[i]) for i in g) for g in bins]


def calc_rev_bin_map(bins: Sequence[Sequence[int]], e_count: int) -> list[list[int]]:
    return [[i for i, g in enumerate(bins) if e in g] for e in range(e_count)]


if __name__ == '__main__':
    graph = ig.Graph(n=8, edges=[[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                              [6, 7], [7, 4], [3, 4]])
    model = WarmGraph(graph)
    solution, values = model.solve()
    print(solution, values)

    # graph = ig.Graph(n=3, edges=[[0, 1], [0, 2]])
    # model = WarmGraph(graph)
    # solution, values = model.solve()
    # print(solution, values)
    #
    # graph = ig.Graph(n=4, edges=[[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
    # model = WarmGraph(graph)
    # solution, values = model.solve()
    # print(solution, values)
    #
    # graph = ig.Graph(n=6, edges=[[0, 1], [1, 2], [2, 3], [2, 4], [2, 5]])
    # model = WarmGraph(graph)
    # solution, values = model.solve()
    # print(solution, values)
    #
    # graph = ig.Graph(n=6, edges=[[0, 1], [1, 2], [2, 3], [2, 4], [2, 5]])
    # model = WarmGraph(graph)
    # solution, values = model.solve()
    # print(solution, values)
    #
    # graph = ig.Graph(n=8, edges=[[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
    #                           [6, 7], [7, 4], [3, 4]])
    # model = WarmGraph(graph)
    # solution, values = model.solve()
    # print(solution, values)

    # x = cp.Variable(5, name="x")
    # A = np.matrix([
    #     [1, 1, 1, 1, 0],
    #     [1, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 1],
    #     [0, 0, 0, 0, 1],
    #     [0, 0, 1, 0, 0],
    #     [0, 0, 0, 1, 0]
    # ])
    # b = np.repeat(1/6, 6)
    # omega = A @ x
    #
    # # Construct the problem.
    # objective = (cp.sum(x)
    #     - cp.sum(cp.multiply(cp.log(omega), np.array(b))))
    # constraints = [cp.sum(x) == 1, omega >= b, x >= 0]
    # prb = cp.Problem(cp.Minimize(objective), constraints)
    #
    # # The optimal objective value is returned by `prob.solve()`.
    # result = prb.solve()
    # # The optimal value for x is stored in `x.value`.
    # values = [y.value for y in x]
    # print(values)
    # # The optimal Lagrange multiplier for a constraint is stored in
    # # `constraint.dual_value`.
    # print(constraints[0].dual_value)
    # print(constraints[1].dual_value)
    # print(constraints[2].dual_value)
