#
from __future__ import annotations
import matplotlib.pyplot as plt # type: ignore[import]
import seaborn as sns # type: ignore[import]
import numpy as onp
import numpy.typing as onpt
import os
from typing import List, Optional
from sklearn.datasets import make_blobs # type: ignore[import]
from .meta import Meta


class MetaGMM(Meta):
    R"""
    GMM simulation metaset.
    """
    #
    PRECISION = onp.float32

    def __init__(
        self,
        feats_target: onpt.NDArray[onp.generic],
        labels_target: onpt.NDArray[onp.generic],
        /,
    ) -> None:
        R"""
        Initialize the class.
        """
        # Get number of clusters.
        num_unique_labels_target = len(onp.unique(labels_target))
        vmax_label_target = onp.max(labels_target).item()
        if num_unique_labels_target != vmax_label_target + 1:
            # EXPECT:
            # Improper target label data.
            raise RuntimeError(
                "Given target labels are not tightly consecutive from 0.",
            )
        self.num_clusters = num_unique_labels_target

        #
        (self.num_samples, self.num_feats_target) = feats_target.shape
        self.num_labels_target = 1

        #
        self.feats_target = feats_target.astype(self.PRECISION)
        self.labels_target = labels_target.astype(onp.int64)

        #
        buf_indices_clusters = []
        for y in range(self.num_clusters):
            #
            (indices,) = onp.where(self.labels_target == y)
            buf_indices_clusters.append(indices[0:2].tolist())

        #
        buf_pairs_interest = []
        for i in range(len(buf_indices_clusters)):
            #
            buf_pairs_interest.append(buf_indices_clusters[i])
            buf_pairs_interest.append(
                [
                    buf_indices_clusters[i][0],
                    buf_indices_clusters[(i + 1) % len(buf_indices_clusters)]
                    [0],
                ],
            )
        self.pairs_interest = onp.array(buf_pairs_interest)

    @classmethod
    def from_generate(
        cls,
        n_samples: List[int], cluster_std: float,
        /,
        *,
        seed: int,
    ) -> MetaGMM:
        R"""
        Initialize the class from generation.
        """
        #
        (feats_target, labels_target) = (
            make_blobs(
                n_samples=n_samples, centers=None, cluster_std=cluster_std,
                random_state=seed,
            )
        )
        feats_target = feats_target[:, ::-1].copy()
        return cls(feats_target, labels_target)

    @classmethod
    def from_load(cls, path: str, /) -> MetaGMM:
        R"""
        Initialize the class from loading.
        """
        #
        with open(path, "rb") as file:
            #
            feats_target = onp.load(file)
            labels_target = onp.load(file)
        return cls(feats_target, labels_target)

    def save(self, path: str, /) -> str:
        R"""
        Save data.
        """
        #
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            #
            os.makedirs(directory, exist_ok=True)

        #
        with open(path, "wb") as file:
            #
            onp.save(file, self.feats_target)
            onp.save(file, self.labels_target)
        return path

    def render(self, path: str, /) -> str:
        R"""
        Render data statistics.
        """
        #
        return render(self.feats_target, self.labels_target, None, path=path)

    def __len__(self, /) -> int:
        R"""
        Get class length.
        """
        #
        return self.num_samples

    def minibatch(
        self,
        indices: List[int],
        /,
    ) -> List[onpt.NDArray[onp.generic]]:
        R"""
        Get minibatch.
        """
        #
        return [self.feats_target[indices]]

    def fullbatch(self, /) -> List[onpt.NDArray[onp.generic]]:
        R"""
        Get full batch.
        """
        #
        return [self.feats_target]


def render(
    feats: onpt.NDArray[onp.generic], labels: onpt.NDArray[onp.generic],
    means: Optional[onpt.NDArray[onp.generic]],
    /,
    *,
    path: str,
) -> str:
    R"""
    Render data statistics.
    """
    #
    sns.set()

    #
    vmin_x = onp.min(feats[:, 0])
    vmax_x = onp.max(feats[:, 0])
    vptp_x = vmax_x - vmin_x
    vmin_x = vmin_x - 0.05 * vptp_x
    vmax_x = vmax_x + 0.05 * vptp_x
    vmin_y = onp.min(feats[:, 1])
    vmax_y = onp.max(feats[:, 1])
    vptp_y = vmax_y - vmin_y
    vmin_y = vmin_y - 0.05 * vptp_y
    vmax_y = vmax_y + 0.05 * vptp_y

    #
    (fig, ax) = plt.subplots(1, 1)
    sns.scatterplot(
        x=feats[:, 0], y=feats[:, 1], hue=labels, alpha=0.5,
        palette=sns.color_palette("hls", len(onp.unique(labels))), ax=ax,
    )
    if means is not None:
        #
        sns.scatterplot(
            x=means[:, 0], y=means[:, 1], color="black", marker="X", ax=ax,
        )
    ax.set_xlim(vmin_x, vmax_x)
    ax.set_ylim(vmin_y, vmax_y)
    fig.savefig(path)
    plt.close(fig)
    return path
