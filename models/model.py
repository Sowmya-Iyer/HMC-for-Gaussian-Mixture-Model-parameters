#
import abc
import torch
from typing import List, Tuple


class Model(abc.ABC, torch.nn.Module):
    R"""
    GMM model.
    """
    @abc.abstractmethod
    def initialize(self, rng: torch.Generator) -> int:
        R"""
        Initialize parameters
        """
        #
        ...

    @abc.abstractmethod
    def forward(self, /, *ARGS) -> List[torch.Tensor]:
        R"""
        Forward.
        """
        #
        return []

    @abc.abstractmethod
    def loss(self, /, *ARGS) -> torch.Tensor:
        R"""
        Loss function.
        """
        #
        ...

    @abc.abstractmethod
    def metrics(self, /, *ARGS) -> List[Tuple[float, int]]:
        R"""
        Metric function.
        """
        #
        ...
