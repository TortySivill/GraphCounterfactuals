from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import networkx as nx
import numpy as np


class FACE:
    def __init__(self, data, clf, dist_threshold=1, density_threshold=0.001, pred_threshold=0.9,
                 dist_metric='euclidean', kde=gaussian_kde):
        self.clf = clf
        self.dist_threshold = dist_threshold
        self.density_threshold = density_threshold
        self.pred_threshold = pred_threshold
        self.dist_metric = dist_metric
        self.kde = kde(data.T)
        self.data = data
        self.graph = self.create_graph()
        self.data = data[data.index.isin(list(self.graph.nodes()))]

    def create_graph(self):
        m, n = self.data.shape
        w_ij = np.zeros(int(m * (m - 1) / 2))
        edge_weights = []
        q = 0
        for edge_from in range(m):
            k = 1 + edge_from
            for edge_to in range(k, m):
                dist = cdist(self.data.values[edge_from].reshape(1, -1), self.data.values[edge_to].reshape(1, -1),
                             metric=self.dist_metric).squeeze()
                if dist < self.dist_threshold:
                    w_ij[q] = -np.log(self.kde((self.data.values[edge_from] + self.data.values[edge_to]) / 2) * dist)
                edge_weights.append((edge_from, edge_to, {'weight': w_ij[q]}))
                q += 1

        nonzero_edge = []
        for i in range(len(edge_weights)):
            if edge_weights[i][2]['weight'] != 0:
                nonzero_edge.append(edge_weights[i])

        G = nx.Graph()
        G.add_nodes_from(range(len(self.data)))
        G.add_edges_from(nonzero_edge)

        low_density = self.data[self.kde(self.data.T) < self.density_threshold].index
        G.remove_nodes_from(low_density)

        return G

    def nearest_training_point(self, example):
        pair_dist = cdist(self.data, example, metric=self.dist_metric)
        index_sorted = np.argsort(pair_dist, axis=0)
        i = 0
        nearest = self.data.values[index_sorted[i]].reshape(1, -1)
        if not self.clf.predict(example) == self.clf.predict(nearest):
            i += 1
            nearest = self.data.values[index_sorted[i]].reshape(1, -1)
        assert pair_dist[index_sorted[i]] < self.dist_threshold, 'Data point does not meet distance threshold'
        return nearest, index_sorted[i].item()

    def generate_counterfactual(self, example):
        _, start_node = self.nearest_training_point(example)

        target = int(abs(self.clf.predict(example) - 1))
        target_data = self.data[self.clf.predict(self.data) == target]
        target_nodes = list(set(list(self.graph.nodes())).intersection(target_data.index))

        _, path = nx.multi_source_dijkstra(self.graph, target_nodes, target=start_node)
        path = path[::-1]
        pred_prob = self.clf.predict_proba(target_data.loc[path[-1]].values.reshape(1, -1)).squeeze()[target]

        while pred_prob < self.pred_threshold:
            target_data = target_data.drop(path[-1])
            target_nodes = list(set(list(self.graph.nodes())).intersection(target_data.index))
            _, path = nx.multi_source_dijkstra(self.graph, target_nodes, target=start_node)
            path = path[::-1]
            pred_prob = self.clf.predict_proba(target_data.loc[path[-1]].values.reshape(1, -1)).squeeze()[target]

        return self.data.loc[path].reset_index(drop=True), pred_prob


