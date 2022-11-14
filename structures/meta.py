#
import abc
import numpy as onp
import numpy.typing as onpt
from typing import List


class Meta(abc.ABC):
    R"""
    Meta structure.
    """
    # Default float precision is float32.
    PRECISION = onp.float32

    @abc.abstractmethod
    def __len__(self, /) -> int:
        R"""
        Get class length.
        """
        #
        ...

    @abc.abstractmethod
    def minibatch(
        self,
        indices: List[int],
        /,
    ) -> List[onpt.NDArray[onp.generic]]:
        R"""
        Get minibatch.
        """
        #
        ...

    @abc.abstractmethod
    def fullbatch(self, /) -> List[onpt.NDArray[onp.generic]]:
        R"""
        Get full batch.
        """
        #
        ...