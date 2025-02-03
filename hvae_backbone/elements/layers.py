import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from .distributions import generate_distribution
from ..utils import SerializableModule


class FixedStdDev(SerializableModule):
    def __init__(self, std):
        super(FixedStdDev, self).__init__()
        self.std = std

    def forward(self, x):
        return torch.concatenate([x, self.std * torch.ones_like(x)], dim=1)

    def serialize(self):
        serialized = super().serialize()
        serialized["params"] = dict(
            std=self.std
        )
        return serialized

    @staticmethod
    def deserialize(serialized):
        return FixedStdDev(**serialized["params"])


class Flatten(torch.nn.Flatten, SerializableModule):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__(start_dim=start_dim, end_dim=end_dim)

    def serialize(self):
        serialized = super().serialize()
        serialized["params"] = dict(
            start_dim=self.start_dim,
            end_dim=self.end_dim
        )
        return serialized

    @staticmethod
    def deserialize(serialized):
        return Flatten(**serialized["params"])


class Unflatten(torch.nn.Unflatten, SerializableModule):
    def __init__(self, dim, unflattened_size):
        super(Unflatten, self).__init__(dim, unflattened_size)
        self.unflattened_size = unflattened_size
        self.dim = dim

    def serialize(self):
        serialized = super().serialize()
        serialized["params"] = dict(
            dim=self.dim,
            unflattened_size=self.unflattened_size
        )
        return serialized

    @staticmethod
    def deserialize(serialized):
        return Unflatten(**serialized["params"])


class Sample(SerializableModule):

    def __init__(self, distribution):
        self.distribution = distribution
        super(Sample, self).__init__()

    def forward(self, x):
        dist = generate_distribution(x, self.distribution)
        return dist.rsample()

    def serialize(self):
        serialized = super().serialize()
        serialized["params"] = dict(
            distribution=self.distribution
        )
        return serialized

    @staticmethod
    def deserialize(serialized):
        return Sample(**serialized["params"])



