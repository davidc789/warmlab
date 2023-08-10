import warm
import cvxpy as cp


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
