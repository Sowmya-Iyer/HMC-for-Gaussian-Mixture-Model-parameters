#
import torch
from typing import Tuple


#
MetaDDI = (
    Tuple[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor],
    ]
)