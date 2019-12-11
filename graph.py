from collections import defaultdict
import numpy as np


def dist(p1, p2):
    # Euclidean distance between p1 and p2 points
    return np.sqrt(np.sum((p1 - p2) ** 2))


class Graph:

    def __init__(self, v=None, values=None):
        self.v = v
        self.graph = defaultdict(list)
        self.edge_lengths = {}
        self.vertex_values = {i: values[i] for i in range(len(values))}

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.edge_lengths[(u, v)] = dist(self.vertex_values[u],
                                         self.vertex_values[v])
        self.edge_lengths[(v, u)] = dist(self.vertex_values[u],
                                         self.vertex_values[v])

    def remove_edge(self, u, v):
        self.graph[u].remove(v)
        self.graph[v].remove(u)
        del self.edge_lengths[(u, v)]
        del self.edge_lengths[(v, u)]

    def is_reachable(self, u, v):
        visited = [False] * self.v

        queue = [u]

        visited[u] = True
        res = False
        while queue:
            n = queue.pop(0)

            if n == v:
                res = True
                break

            lst = list(self.graph.items())
            iter_item = []
            for item in lst:
                if item[0] == n:
                    iter_item = item[1]
            for i in iter_item:
                i = i[0] if type(i) == list else i
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True
        return res

    def is_isolated(self, u):
        isolated = True
        for node in self.graph.items():
            if node[0] != u and self.is_reachable(u, node[0]):
                isolated = False
                break
        return isolated

    def set_values(self, values):
        self.vertex_values = {i: values[i] for i in range(self.v)}

    def edges(self):
        return self.edge_lengths.copy()

    def isolated(self):
        return [node for node in range(self.v) if self.is_isolated(node)]

    def isolated_values(self):
        return {i: self.vertex_values[i] for i in range(len(self.vertex_values))
                if self.is_isolated(list(self.vertex_values.keys())[i])}

    def values(self):
        return self.vertex_values.copy()

    def clusters(self):
        clusters = []
        for node in self.graph:

            queue = [node]

            seen = {node}

            while queue:

                cur = queue.pop(0)
                for i in self.graph[cur]:
                    if i not in seen:
                        queue.append(i)
                        seen.add(i)
            res = list(sorted(list(seen)))
            if res not in clusters:
                clusters.append(res)
        return clusters
