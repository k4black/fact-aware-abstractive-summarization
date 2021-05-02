from torch.utils.data import Dataset, DataLoader
import fasttext
from datasets import load_dataset

from openie import get_triplets

ft = fasttext.load_model('cc.en.128.bin')

import numpy as np
import torch


def _get_nodes_features(nodes, features_dim=128):
    features = np.empty(shape=(len(nodes), features_dim))

    for n, i in nodes.items():
        features[i] = ft.get_sentence_vector(n)

    return features


def _get_graph(triplets, device='cpu'):
    nodes = set()
    for t in triplets:
        nodes.update(t)
    nodes = {n: i for i, n in enumerate(nodes)}

    # shape = (N, FIN), where N is the number of nodes and FIN is the number of input features
    node_features = _get_nodes_features(nodes)

    # Normalize the features
    node_features = node_features / np.clip(node_features.sum(1), a_min=1, a_max=None)[:,None]

    # shape = (N, 1)
    # node_labels = np.array(list(nodes.values()))
    node_labels = np.arange(0, len(nodes))[:,None]

    # Build edge index explicitly (faster than nx ~100 times and as fast as PyGeometric imp but less complex)
    # shape = (2, E), where E is the number of edges, and 2 for source and target nodes. Basically edge index
    # contains tuples of the format S->T, e.g. 0->3 means that node with id 0 points to a node with id 3.
    topology = []
    for s, r, o in triplets:
        topology.append((nodes[s], nodes[r]))
        topology.append((nodes[r], nodes[o]))
    _topology = np.empty(shape=(2, len(topology))).T
    if len(topology) > 0:
        _topology[:] = topology

    # Convert to dense PyTorch tensors

    # Needs to be long int type (in implementation 3) because later functions like PyTorch's index_select expect it
    topology = torch.tensor(_topology.T, dtype=torch.long, device=device)
    node_labels = torch.tensor(node_labels, dtype=torch.long, device=device)  # Cross entropy expects a long int
    node_features = torch.tensor(node_features.astype(np.float32), device=device)

    return node_features, node_labels, topology


class CnnDmDataset(Dataset):
    def __init__(self, _type='train', transform=None):
        self.dataset = load_dataset("cnn_dailymail", '3.0.0')[_type]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.dataset[i]

        article = data['article']

        triplets = get_triplets(article)
        node_features, node_labels, topology = _get_graph(triplets)

        highlight = data['highlights']

        return {
            'article': article,
            'node_features': node_features,
            'node_labels': node_labels,
            'topology': topology,
            'summary': highlight,
        }

    @staticmethod
    def collate_fn(batch):
        return {
            'article': [i['article'] for i in batch],
            'node_features': [i['node_features'] for i in batch],
            'node_labels': [i['node_labels'] for i in batch],
            'topology': [i['topology'] for i in batch],
            'summary': [i['summary'] for i in batch],
        }
