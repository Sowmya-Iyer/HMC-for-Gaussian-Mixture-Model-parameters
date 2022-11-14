#
import argparse
import os
import torch
import hashlib
import shutil
from typing import Sequence, List
from structures.gmm import MetaGMM
from models.gmm import ModelGMM
from models.hmc import ParameterSamplerHMC
from utils import transfer, sgen


def describe(hash: bool, /,) -> str:
    R"""
    Get description.
    """
    #
    description = "hmc"

    #
    if hash:
        #
        description = hashlib.md5(description.encode()).hexdigest()
    return description


def sbatch(
    squeue: str, description: str,
    /,
    *,
    source: str, seed: int, device: str,
) -> None:
    R"""
    Generate sbatch file.
    """
    #
    cmds = (
        [
            "--source {:s}".format(source), "--random-seed {:d}".format(seed),
            "--device {:s}".format(device),
        ]
    )
    sgen(
        squeue, description, "main_hmc.py", cmds, 79,
        num_gpus=0 if device == "cpu" else 1,
    )


@torch.no_grad()
def evaluate(metaset: MetaGMM, model: ModelGMM, /, *, device: str) -> float:
    R"""
    Evaluate.
    """
    #
    model.eval()
    inputs_ondev = transfer(metaset.fullbatch(), device)
    ((sum_nll, size_nll),) = model.metrics(*inputs_ondev)
    return sum_nll / float(size_nll)


def train(
    metaset: MetaGMM, model: ModelGMM,
    optimizers: Sequence[torch.optim.Optimizer],
    /,
    *,
    device: str,
) -> None:
    R"""
    Train.
    """
    #
    (optimizer_mix, optimizer_gauss) = optimizers
    model.train()
    optimizer_gauss.zero_grad()
    optimizer_mix.zero_grad()
    inputs_ondev = transfer(metaset.fullbatch(), device)
    loss = model.loss(*inputs_ondev)
    loss.backward()
    optimizer_gauss.step()
    optimizer_mix.step()


@torch.no_grad()
def posterior(
    metaset: MetaGMM, model: ModelGMM,
    /,
    *,
    device: str,
) -> List[float]:
    R"""
    Posterior.
    """
    #
    inputs_ondev = (
        transfer(
            metaset.minibatch(metaset.pairs_interest.T.flatten().tolist()),
            device,
        )
    )
    prob_clusters = model.posterior(*inputs_ondev)
    prob_clusters = (
        torch.reshape(
            prob_clusters,
            (2, len(metaset.pairs_interest), metaset.num_clusters),
        )
    )
    prob_same = torch.sum(prob_clusters[0] * prob_clusters[1], dim=1)
    return prob_same.cpu().tolist()


def main(*ARGS) -> None:
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="Main Execution (Homework 2)")
    parser.add_argument(
        "--sbatch",
        type=str, required=False, default="", help="Slurm queue.",
    )
    parser.add_argument(
        "--source",
        type=str, required=False, default="data",
        help="Path to data directory.",
    )
    parser.add_argument(
        "--random-seed",
        type=int, required=False, default=47, help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str, required=False, default="cpu", help="Device.",
    )
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    #
    use_sbatch = args.sbatch
    source = args.source
    seed = args.random_seed
    device = args.device

    #
    description = describe(False)
    print("\x1b[104;30m{:s}\x1b[0m".format(description))

    #
    if use_sbatch:
        #
        sbatch(
            use_sbatch, description,
            source=source, seed=seed, device=device,
        )
        return

    #
    metaset = MetaGMM.from_load(os.path.join(source, "gmm.npy"))

    #
    if os.path.isdir("fig"):
        #
        shutil.rmtree("fig")
        while os.path.isdir("fig"):
            #
            pass
    os.makedirs("fig", exist_ok=True)

    #
    model = ModelGMM(metaset.num_feats_target, metaset.num_clusters)
    thrng = torch.Generator("cpu")
    thrng.manual_seed(seed)
    model.initialize(thrng)
    model = model.to(device)

    #
    epochs = 2000

    # Use different optimization configurations for different parts.
    optimizer_mix = torch.optim.SGD([model.weights], lr=0.001)
    optimizer_gauss = (
        torch.optim.SGD([model.means, model.stdvs], lr=0.01, momentum=0.99)
    )
    optimizers = [optimizer_mix, optimizer_gauss]

    #
    print("Pretrain {:d} epochs before HMC ...".format(epochs))
    posterior1 = posterior(metaset, model, device=device)
    print("{:s} {:s}".format("-" * 5, "-" * 8))
    print("{:>5s} {:>8s}".format("Epoch", "Mean NLL"))
    print("{:s} {:s}".format("-" * 5, "-" * 8))
    mean_nll = evaluate(metaset, model, device=device)
    print("{:>5d} {:.6f}".format(0, mean_nll))
    for epoch in range(1, epochs + 1):
        #
        train(metaset, model, optimizers, device=device)
        if epoch % 500 > 0:
            #
            continue
        mean_nll = evaluate(metaset, model, device=device)
        print("{:>5d} {:.6f}".format(epoch, mean_nll))
    print("{:s} {:s}".format("-" * 5, "-" * 8))
    posterior2 = posterior(metaset, model, device=device)

    #
    param_sample_stdv = 1.0
    leapfrog_deltas = [0.001, 0.01, 0.01]
    num_leapfrogs = 50
    num_samples = 100

    #
    param_sampler = ParameterSamplerHMC(model)
    thrng = torch.Generator("cpu")
    thrng.manual_seed(seed)

    #
    print(
        "Sample {:d} parameter configurations by HMC ...".format(num_samples),
    )
    param_sample_batch = transfer(metaset.fullbatch(), device)
    param_samples = (
        param_sampler.sample(
            num_samples, *param_sample_batch,
            inits=[param.data for param in model.parameters()],
            stdv_or_stdvs=param_sample_stdv, rng=thrng, deltas=leapfrog_deltas,
            num_leapfrogs=num_leapfrogs,
        )
    )
    buf_posterior = []
    for inits in param_samples:
        #
        for (param, init) in zip(model.parameters(), inits):
            #
            param.data.copy_(init)
        buf_posterior.append(posterior(metaset, model, device=device))
    posterior3 = torch.mean(torch.Tensor(buf_posterior), dim=0).tolist()

    #
    print("-" * 7, "-" * 10, "-" * 8, "-" * 5)
    print("Initial", "Pretrained", "HMC Mean", "Label")
    print("-" * 7, "-" * 10, "-" * 8, "-" * 5)
    for (i, (p1, p2, p3)) in (
        enumerate(zip(posterior1, posterior2, posterior3))
    ):
        #
        print(
            "{:>7s} {:>10s} {:>8s} {:>5s}".format(
                "{:.3f}".format(p1), "{:.3f}".format(p2), "{:.3f}".format(p3),
                "Same" if i % 2 == 0 else "Diff",
            ),
        )
    print("-" * 7, "-" * 10, "-" * 8, "-" * 5)


if __name__ == "__main__":
    #
    main()
