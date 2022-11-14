#
import numpy as onp
import numpy.typing as onpt
import torch
import os
import random
from typing import List, Tuple


def transfer(
    buf_array: List[onpt.NDArray[onp.generic]],
    device: str,
    /,
) -> List[torch.Tensor]:
    R"""
    Transfer numpy arrays to torch tensors.
    """
    #
    return [torch.from_numpy(array).to(device) for array in buf_array]


def sgen(
    squeue: str, description: str, python: str, cmds: List[str], maxlen: int,
    /,
    *,
    num_gpus: int,
) -> None:
    R"""
    Generate sbatch file.
    """
    #
    if squeue == "urca":
        #
        squeue = "--partition=urca-gpu"
    elif squeue == "scholar":
        #
        squeue = "--account=scholar"
    elif squeue == "gpu":
        #
        squeue = "--account=gpu"
    else:
        # UNEXPECT:
        # Unknown Slurm queue.
        raise NotImplementedError(
            "Unknown Slurm queue \"{:s}\".".format(squeue),
        )

    #
    if not os.path.isdir("sbatch"):
        #
        os.makedirs("sbatch")

    #
    tops = []
    tops.append("#!/bin/bash")
    tops.append("")
    tops.append("#SBATCH --job-name={:s}".format(description))
    tops.append("#SBATCH --output=sbatch/{:s}.stdout.txt".format(description))
    tops.append("#SBATCH --error=sbatch/{:s}.stderr.txt".format(description))
    tops.append("#SBATCH {:s}".format(squeue))
    tops.append("#SBATCH --gres=gpu:{:d}".format(num_gpus))
    tops.append("")
    tops.append("#")
    tops.append(
        "/usr/bin/time -f \"Max CPU Memory: %M KB\\nElapsed: %e sec\" \\",
    )
    if len(cmds) > 0:
        #
        tops.append("python -u {:s} \\".format(python))
    else:
        #
        tops.append("python -u {:s}".format(python))

    #
    if len(cmds) > 0:
        #
        args = ["    "]
    else:
        #
        args = []
    for cmd in cmds:
        #
        if len(args[-1]) + 1 - int(args[-1].isspace()) + len(cmd) > maxlen:
            #
            args[-1] = args[-1] + " \\"
            args.append("    " + cmd)
        elif args[-1].isspace():
            #
            args[-1] = args[-1] + cmd
        else:
            #
            args[-1] = args[-1] + " " + cmd

    #
    spath = os.path.join("sbatch", "{:s}.sh".format(description))
    with open(spath, "w") as file:
        #
        for line in tops + args:
            #
            file.write(line + "\n")


def negative_sampling(
    edge_index: torch.Tensor, num_nodes: int, num_neg_samples: int,
    /,
) -> torch.Tensor:
    r"""
    Samples random negative edges.
    """
    #
    (idx, population) = edge_index_to_vector(edge_index, num_nodes)
    if not idx.numel() < population:
        # UNEXPECT:
        # Positive edges occupied all possible edge IDs.
        raise NotImplementedError(
            "Positive edge IDs occupy all possible edge IDs",
        )

    # Probability to sample a negative edge.
    prob = 1.0 - float(idx.numel()) / float(population)

    # Over-sample to make it more likely to get enough negative samples.
    sample_size = int(1.1 * num_neg_samples / prob)

    # Mark positive edges as insamplable.
    mask = idx.new_ones(population, dtype=torch.bool)
    mask[idx] = False

    # Sample multiple times to make it more likely to get enough negative
    # samples.
    neg_idx = torch.sum(torch.Tensor([0]))
    for _ in range(3):
        #
        rnd = sample(population, sample_size, idx.device)
        rnd = rnd[mask[rnd]]
        neg_idx = rnd if neg_idx.ndim == 0 else torch.cat([neg_idx, rnd])

        #
        if neg_idx.numel() >= num_neg_samples:
            # Early stop.
            neg_idx = neg_idx[:num_neg_samples]
            break
        else:
            # Mark sampled edges as insamplable.
            mask[neg_idx] = False
    return vector_to_edge_index(neg_idx, num_nodes)


def sample(population: int, k: int, device: torch.device) -> torch.Tensor:
    R"""
    Uniformly sample from given population size.
    """
    #
    if population <= k:
        #
        return torch.arange(population, device=device)
    else:
        #
        return torch.tensor(random.sample(range(population), k), device=device)


def edge_index_to_vector(
    edge_index: torch.Tensor, num_nodes: int,
    /,
) -> Tuple[torch.Tensor, int]:
    R"""
    Translate edge tuples into a vector unique edge IDs.
    """
    # We remove self-loops as we do not want to take them into account
    # when sampling negative values.
    (src, dst) = edge_index
    mask = src != dst
    (src, dst) = (src[mask], dst[mask])

    # Shift down destination node ID which is larger than source node ID as the
    # result removing self-loops.
    dst[src < dst] -= 1

    # Get node IDs, and maximum number of unique IDs.
    idx = src.mul_(num_nodes - 1).add_(dst)
    population = num_nodes * num_nodes - num_nodes
    return (idx, population)


def vector_to_edge_index(idx: torch.Tensor, num_nodes: int, /) -> torch.Tensor:
    R"""
    Translate a vector of unique edge IDs to edge tuples.
    """
    # Recover from self-loop removal.
    row = idx.div(num_nodes - 1, rounding_mode="floor")
    col = idx % (num_nodes - 1)
    col[row <= col] += 1
    return torch.stack([row, col], dim=0)


def select_batchable_seqints(
    seqint: onpt.NDArray[onp.int64], batch_size: int, seed: int,
    /,
) -> Tuple[onpt.NDArray[onp.int64], onpt.NDArray[onp.int64]]:
    R"""
    Select batchable sequences of integers.
    """
    #
    batchable_size = (len(seqint) - 1) // batch_size * batch_size
    if batchable_size == len(seqint) - 1:
        #
        bias = 0
    else:
        #
        bias = (
            onp.random.RandomState(seed)
            .randint(0, len(seqint) - 1 - batchable_size)
        )
    seqint_input = seqint[bias:bias + batchable_size]
    seqint_target = seqint[bias + 1:bias + batchable_size + 1]
    seqints_input = (
        onp.reshape(
            seqint_input, (batch_size, len(seqint_input) // batch_size),
        )
    )
    seqints_target = (
        onp.reshape(
            seqint_target, (batch_size, len(seqint_target) // batch_size),
        )
    )
    return (seqints_input, seqints_target)
