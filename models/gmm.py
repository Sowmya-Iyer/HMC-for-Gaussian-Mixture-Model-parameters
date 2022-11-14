#
import torch
import torch.distributions as thdists
from typing import List, Tuple
from .model import Model


class ModelGMM(Model):
    R"""
    GMM model.
    """
    def __init__(
        self,
        feat_size_target: int, label_size_target: int,
        /,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.feat_size_target = feat_size_target
        self.label_size_target = label_size_target

        #
        self.weights = (
            torch.nn.parameter.Parameter(torch.zeros(self.label_size_target))
        )
        self.means = (
            torch.nn.parameter.Parameter(
                torch.zeros(self.label_size_target, self.feat_size_target),
            )
        )
        self.stdvs = (
            torch.nn.parameter.Parameter(
                torch.zeros(self.label_size_target, self.feat_size_target),
            )
        )

    def initialize(self, rng: torch.Generator) -> int:
        R"""
        Initialize parameters
        """
        #
        self.weights.data.fill_(1.0 / float(self.label_size_target))
        resetted = self.weights.numel()

        #
        self.means.data.normal_(0, 1, generator=rng)
        self.stdvs.data.normal_(0, 10, generator=rng)
        resetted += self.means.numel()
        resetted += self.stdvs.numel()
        return resetted

    def forward(self, /, *ARGS) -> List[torch.Tensor]:
        R"""
        Forward.
        """
        # Parse.
        (observations,) = ARGS

        # Get log distribution of input observations.
        weights = torch.softmax(self.weights, dim=0)
        mix = thdists.Categorical(weights)
        components = (
            thdists.Independent(thdists.Normal(self.means, self.stdvs ** 2), 1)
        )
        gmm = thdists.MixtureSameFamily(mix, components)
        return [gmm.log_prob(observations)]

    def loss(self, /, *ARGS) -> torch.Tensor:
        R"""
        Loss function.
        """
        #
        (log_prob,) = self.forward(*ARGS)
        return torch.mean(-log_prob)

    def metrics(self, /, *ARGS) -> List[Tuple[float, int]]:
        R"""
        Metric function.
        """
        #
        (log_prob,) = self.forward(*ARGS)
        sum_nll = torch.sum(-log_prob).item()
        size_nll = len(log_prob)
        return [(sum_nll, size_nll)]

    def energy(self, /, *ARGS) -> torch.Tensor:
        R"""
        Energy function.
        """
        #
        (log_prob,) = self.forward(*ARGS)
        return torch.mean(-log_prob)

    def posterior(self, /, *ARGS) -> torch.Tensor:
        R"""
        Posterior distribution.
        """
        # Parse.
        (observations,) = ARGS

        #
        buf = []
        weights = torch.softmax(self.weights, dim=0)
        for i in range(len(weights)):
            #
            mix = thdists.Categorical(weights[[i]])
            components = (
                thdists.Independent(
                    thdists.Normal(self.means[[i]], self.stdvs[[i]] ** 2), 1,
                )
            )
            gmm = thdists.MixtureSameFamily(mix, components)
            buf.append(gmm.log_prob(observations) + torch.log(weights[[i]]))
        log_prob_clusters = torch.stack(buf, dim=1)
        prob_clusters = torch.exp(log_prob_clusters)
        prob_clusters = (
            prob_clusters / torch.sum(prob_clusters, dim=1, keepdim=True)
        )
        return prob_clusters
