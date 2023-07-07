""" An adaptor for the json format supported in the C++ module.
"""
import typing

import graphviz
import dataclasses
import json
from typing import TypeVar

_Edge = TypeVar("_Edge")


class DataClassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclasses.dataclass
class WeightedEdge:
    count: int
    weight: float

    @classmethod
    def fromJson(cls, dct: dict[str]):
        return WeightedEdge(dct["count"], dct["weight"])

    def toJson(self):
        return json.dumps(self)


@dataclasses.dataclass
class Graph(object):
    nodeCount: int
    edgeCount: int
    adjacencyList: list[dict[int, int]]
    edgeList: list[_Edge]
    isDirected: bool

    def __init__(self, n: int = 0, is_directed: bool = False):
        self.nodeCount = 0
        self.isDirected = is_directed
        self.addNodes(n)

    def getNodeCount(self) -> int:
        return self.nodeCount

    def getEdgeCount(self) -> int:
        return self.edgeCount

    def setEdge(self, u: int, v: int, edge: _Edge) -> typing.Self:
        self._nodePairCheck(u, v)

        if v in self.adjacencyList[u]:
            self.edgeList[self.adjacencyList[u][v]] = edge
        else:
            self.adjacencyList[u][v] = self.edgeCount
            if not self.isDirected:
                self.adjacencyList[v][u] = self.edgeCount
            self.edgeList.append(edge)
            self.edgeCount += 1
        return self

    def getEdge(self, u: int, v: int) -> _Edge:
        self._nodePairCheck(u, v)
        return self.edgeList[self.adjacencyList[u][v]]

    def getAdjacentNodes(self, u: int) -> dict[int, int]:
        self._nodeRangeCheck(u)
        return self.adjacencyList[u]

    def getEdges(self) -> list[_Edge]:
        return self.edgeList

    def addEdge(self, u: int, v: int, edge: _Edge):
        return self.setEdge(u, v, edge)

    def addNodes(self, count: int):
        for i in range(count):
            self.adjacencyList.append({})
        self.nodeCount += count

    def isEmpty(self) -> bool:
        return self.nodeCount == 0

    def __getitem__(self, item):
        return self.adjacencyList[item]

    @classmethod
    def fromJson(cls, dct: dict[str]):
        graph = Graph(n=dct["nodeCount"], is_directed=dct["isDirected"])
        graph.edgeCount = dct["edgeCount"]
        graph.adjacencyList = dct["adjacencyList"]
        graph.edgeList = [WeightedEdge.fromJson(x) for x in dct["edgeList"]]

        return graph

    def toJson(self):
        return json.dumps(self)

    def toGraphviz(self, node_annotations: list[str], edge_annotations: list[str]) -> graphviz.Graph | graphviz.Digraph:
        if len(node_annotations) != self.nodeCount:
            raise ValueError("node annotations length unmatched")
        if len(edge_annotations) != self.edgeCount:
            raise ValueError("edge annotations length unmatched")

        if self.isDirected:
            dot = graphviz.Digraph(comment="")
            raise NotImplemented
        else:
            dot = graphviz.Graph(comment="graph")
            for i, node_annotation in enumerate(node_annotations):
                dot.node(str(i), node_annotation)

            for u, d in enumerate(self.adjacencyList):
                for v, e in d.values():
                    dot.edge(str(v), str(u), edge_annotations[e])

        return dot

    def _nodeRangeCheck(self, u: int):
        if not 0 <= u < self.nodeCount:
            raise ValueError("Index out of bound")

    def _nodePairCheck(self, u: int, v: int):
        self._nodeRangeCheck(u)
        self._nodeRangeCheck(v)
        if u == v:
            raise ValueError("Self-loops are not supported")

    def _edgeIndexCheck(self, index: int):
        if not 0 <= index < self.edgeCount:
            raise ValueError("Edge index out of bound")
