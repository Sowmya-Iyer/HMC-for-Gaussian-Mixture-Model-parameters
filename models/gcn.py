#
import torch
import numpy as onp
from typing import List, Tuple, cast
from .model import Model


class GCN(torch.nn.Module):
    R"""
    GCN.
    """
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        R"""
        Initialize the class.
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        #
        self.weight = (
            torch.nn.parameter.Parameter(
                torch.zeros(self.num_outputs, self.num_inputs),
            )
        )
        self.bias = (
            torch.nn.parameter.Parameter(torch.zeros(self.num_outputs,))
        )

    @classmethod
    def degree_normalizor(
        cls,
        num_nodes: int, edge_indices: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Get in-degree normalizors.
        """
        #
        in_degrees = (
            torch.zeros(
                num_nodes,
                dtype=torch.long, device=edge_indices.device,
            )
        )
        in_degrees.index_add_(
            0, edge_indices[1], torch.ones_like(in_degrees)[edge_indices[0]],
        )
        in_degrees = 1.0 / torch.sqrt(in_degrees)
        return in_degrees[edge_indices[0]] * in_degrees[edge_indices[1]]

    def forward(
        self,
        node_embeds: torch.Tensor, edge_indices: torch.Tensor,
        edge_weights: torch.Tensor,
        /
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        # YOU NEED TO FILL IN THIS PART.
        ...
        ns= node_embeds.shape
        
        A = torch.zeros(ns[0], ns[1], device = node_embeds.device)
        com = node_embeds[edge_indices[1]] * edge_weights.unsqueeze(1)
        A.index_add_(
            0, edge_indices[0], com
        )

        F= torch.matmul(A,self.weight) + self.bias

        H= torch.nn.functional.relu(F)

        return H

def glorot_linear(linear: torch.nn.Linear, rng: torch.Generator, /) -> int:
    R"""
    Glorot initializor.
    """
    #
    (num_outs, num_ins) = linear.weight.data.size()
    a = onp.sqrt(6.0 / float(num_ins + num_outs))
    linear.weight.data.uniform_(-a, a, generator=rng)
    linear.bias.data.zero_()
    return linear.weight.numel() + linear.bias.numel()


def glorot_gcn(gcn: GCN, rng: torch.Generator, /) -> int:
    R"""
    Glorot initializor.
    """
    #
    (num_outs, num_ins) = gcn.weight.data.size()
    a = onp.sqrt(6.0 / float(num_ins + num_outs))
    gcn.weight.data.uniform_(-a, a, generator=rng)
    gcn.bias.data.zero_()
    return gcn.weight.numel() + gcn.bias.numel()


class ModelGCN(Model):
    R"""
    GMM model.
    """
    def __init__(
        self,
        embed_size_hidden: int, num_nodes: int, num_layers: int,
        /,
        *,
        positional: bool,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.embed_size_hidden = embed_size_hidden
        self.num_nodes = num_nodes
        self.num_layers= num_layers
        self.positional = positional

        #
        self.node_embeds = (
            torch.nn.parameter.Parameter(
                torch.zeros(self.num_nodes, self.embed_size_hidden),
            )
        )
        self.node_embeds.requires_grad = self.positional

        #
        self.gcns = (
            torch.nn.ModuleList(
                [
                    GCN(self.embed_size_hidden, self.embed_size_hidden)
                    for _ in range(self.num_layers)
                ],
            )
        )

        #
        self.classifier = (
            torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.embed_size_hidden, self.embed_size_hidden,
                    ),
                    torch.nn.Linear(self.embed_size_hidden, 1),
                ],
            )
        )

    def initialize(self, rng: torch.Generator) -> int:
        R"""
        Initialize parameters
        """
        #
        if self.positional:
            #
            self.node_embeds.data.normal_(0, 1, generator=rng)
            resetted = self.node_embeds.numel()
        else:
            #
            self.node_embeds.data.fill_(1.0)
            resetted = 0

        #
        for gcn in self.gcns:
            #
            resetted += glorot_gcn(gcn, rng)
        for linear in self.classifier:
            #
            resetted += glorot_linear(linear, rng)
        return resetted

    def forward(self, /, *ARGS) -> List[torch.Tensor]:
        R"""
        Forward.
        """
        #
        ((edge_indices, edge_weights), edges_eval) = ARGS
        edge_scores = (
            self.decode(self.encode(edge_indices, edge_weights), edges_eval)
        )
        return [edge_scores]

    def encode(self, /, *ARGS) -> torch.Tensor:
        R"""
        Encode nodes.
        """
        #
        (edge_indices, edge_weights) = ARGS

        #
        node_embeds = cast(torch.Tensor, self.node_embeds)
        for gcn in self.gcns[:-1]:
            #
            node_embeds = gcn.forward(node_embeds, edge_indices, edge_weights)
            node_embeds = torch.relu(node_embeds)
        node_embeds = (
            self.gcns[-1].forward(node_embeds, edge_indices, edge_weights)
        )
        return node_embeds

    def decode(
        self,
        node_embeds: torch.Tensor, edges: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Decode node embeddings for link prediction.
        """
        #
        (edges_src, edges_dst) = edges
        node_embeds_src = node_embeds[edges_src]
        node_embeds_dst = node_embeds[edges_dst]
        edge_embeds = node_embeds_src * node_embeds_dst

        #
        for linear in self.classifier[:-1]:
            #
            edge_embeds = linear.forward(edge_embeds)
            edge_embeds = torch.relu(edge_embeds)
        edge_embeds = self.classifier[-1].forward(edge_embeds)
        return torch.sigmoid(edge_embeds)

    def loss(self, /, *ARGS) -> torch.Tensor:
        R"""
        Loss function.
        """
        # UNEXPECT:
        # Not implemented.
        raise NotImplementedError("Not implemented.")

    def metrics(self, /, *ARGS) -> List[Tuple[float, int]]:
        R"""
        Metric function.
        """
        # UNEXPECT:
        # Not implemented.
        raise NotImplementedError("Not implemented.")
