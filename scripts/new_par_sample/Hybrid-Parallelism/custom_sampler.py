import torch
from torch.utils.data import Sampler
import math

class NodeDistributedSampler(Sampler):
    """
    Sampler that splits dataset by node (not per GPU).
    - All GPUs in the same node get the same indices.
    - Different nodes get different indices.
    """
    def __init__(self, dataset, num_nodes, node_rank, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.node_rank = node_rank # Which node am I on?
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        if self.drop_last and len(self.dataset) % self.num_nodes != 0:
            self.num_samples = math.floor(len(self.dataset) / self.num_nodes)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_nodes)

        self.total_size = self.num_samples * self.num_nodes

    def __iter__(self):
        # generate all indices
        indices = torch.arange(len(self.dataset)).tolist()

        # shuffle deterministically based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # pad if needed
        if not self.drop_last and len(indices) < self.total_size:
            indices += indices[:(self.total_size - len(indices))]

        # split by node
        start = self.node_rank * self.num_samples
        end = start + self.num_samples
        node_indices = indices[start:end]

        return iter(node_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """Sets the epoch for deterministic shuffling."""
        self.epoch = epoch
