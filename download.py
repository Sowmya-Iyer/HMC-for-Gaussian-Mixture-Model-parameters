#
import argparse
import os
import requests # type: ignore[import]
from structures.gmm import MetaGMM

def gmm(directory: str, /) -> None:
    R"""
    Prepare GMM dataset.
    """
    #
    print("Generating GMM raw data ...")
    n_samples = [100, 100, 100, 50]
    cluster_std = 1.6
    seed = 42
    metaset = MetaGMM.from_generate(n_samples, cluster_std, seed=seed)
    metaset.render(os.path.join(directory, "gmm.png"))
    metaset.save(os.path.join(directory, "gmm.npy"))


def ddi(path: str, /) -> None:
    R"""
    Prepare DDI dataset.
    """
    #
    print("Downloading DDI raw data ...")
    src = (
        "https://raw.githubusercontent.com/gao462/DrugDrugInteraction/main/ddi"
        ".npy"
    )
    tar = os.path.join(path, "ddi.npy")
    if os.path.isfile(tar):
        #
        return
    remote = requests.get(src)
    with open(tar, "wb") as file:
        #
        file.write(remote.content)


# def ptb(path: str, /) -> None:
#     R"""
#     Prepare PTB dataset.
#     """
#     #
#     print("Downloading PTB raw data ...")
#     directory = os.path.join(path, "ptb")
#     if not os.path.isdir(directory):
#         #
#         os.makedirs(directory)
#     for filename in ("ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"):
#         #
#         src = (
#             "https://raw.githubusercontent.com/gao462/PennTreebank/master/{:s}"
#             .format(filename)
#         )
#         tar = os.path.join(directory, filename)
#         if os.path.isfile(tar):
#             #
#             continue

#         #
#         remote = requests.get(src)
#         with open(tar, "wb") as file:
#             #
#             file.write(remote.content)
#     metaset = MetaPTB.from_raw(directory)
#     metaset.save(os.path.join(path, "ptb.npy"))


def main(*ARGS) -> None:
    R"""
    Main execution.
    """
    #
    parser = argparse.ArgumentParser(description="Download Execution")
    parser.add_argument(
        "--source",
        type=str, required=False, default="data",
        help="Source root directory for data.",
    )
    parser.add_argument("--gmm", action="store_true", help="Prepare GMM data.")
    parser.add_argument("--ddi", action="store_true", help="Prepare DDI data.")
    parser.add_argument("--ptb", action="store_true", help="Prepare PTB data.")
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    #
    root = args.source
    do_gmm = args.gmm
    do_ddi = args.ddi
    do_ptb = args.ptb

    #
    if not os.path.isdir(root):
        #
        os.makedirs(root, exist_ok=True)

    #
    if do_gmm:
        #
        gmm(root)
    if do_ddi:
        #
        ddi(root)
    if do_ptb:
        #
        ptb(root)


#
if __name__ == "__main__":
    #
    main()