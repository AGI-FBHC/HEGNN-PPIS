import dhg
import torch.nn.functional as F
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch_geometric
from torch import Tensor
from torch_geometric.nn import (
    Aggregation,
    MeanAggregation,
    MessagePassing,
    SumAggregation,
)
from torch_geometric.typing import Adj, OptPairTensor, OptTensor

EPS = 1e-5

class HGNNPConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, residual=False):
        super().__init__()
        self.residual = residual
        self.theta = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        residual = x
        x = self.theta(x)
        x = hg.v2v(x, aggr="mean")
        x = F.relu(x)
        if self.residual:
            x = x + residual
        return x


class KHGNNPConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=bias)
        self.p = nn.Parameter(torch.FloatTensor(1, 1))
        self.p_min = 0
        self.p_max = 2
        self.mu = 1

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        X = self.poly_agg_v2e(X=X, hg=hg)
        X = hg.v2e_update(X)
        X = hg.e2v(X, "mean", drop_rate=0.0)
        X = F.relu(X)
        return X

    def poly_agg_v2e(self, X: torch.Tensor, hg: dhg.Hypergraph):
        # X = self.norm(X)
        p = torch.clamp(self.p, self.p_min, self.p_max)
        min_u = torch.min(X) - self.mu - EPS
        X = (
                torch.sparse.mm(hg.H_T, torch.pow(X - min_u, p + 1)).div(
                    torch.sparse.mm(hg.H_T, torch.pow(X - min_u, p)) + EPS
                )
                + min_u
        )
        return X


class HGATConv(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            drop_rate: float = 0.0,
            atten_neg_slope: float = 0.2,
    ):
        super().__init__()
        self.atten_dropout = nn.Dropout(drop_rate)
        self.atten_act = nn.LeakyReLU(atten_neg_slope)
        self.act = nn.ELU()
        self.theta_vertex = nn.Linear(in_dim, out_dim, bias=bias)
        self.theta_hyperedge = nn.Linear(in_dim, out_dim, bias=bias)
        self.atten_vertex = nn.Linear(out_dim, 1, bias=False)
        self.atten_hyperedge = nn.Linear(out_dim, 1, bias=False)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta_vertex(X)
        Y = self.theta_hyperedge(Y)
        x_for_vertex = self.atten_vertex(X)
        y_for_hyperedge = self.atten_hyperedge(Y)
        v2e_atten_score = x_for_vertex[hg.v2e_src] + y_for_hyperedge[hg.v2e_dst]
        e2v_atten_score = y_for_hyperedge[hg.e2v_src] + x_for_vertex[hg.e2v_dst]
        v2e_atten_score = self.atten_dropout(self.atten_act(v2e_atten_score).squeeze())
        e2v_atten_score = self.atten_dropout(self.atten_act(e2v_atten_score).squeeze())
        Y_ = hg.v2e(X, aggr="softmax_then_sum", v2e_weight=v2e_atten_score)
        X_ = hg.e2v(Y_, aggr="softmax_then_sum", e2v_weight=e2v_atten_score)
        X_ = self.act(X_)
        Y_ = self.act(Y_)
        return X_, Y_


class CoorsNorm(nn.Module):
    """https://github.com/lucidrains/egnn-pytorch"""

    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class EGNNLayer(MessagePassing):
    """E(n)-equivariant Message Passing Layer
    Is currently not compatible with the Pytorch Geometric HeteroConv class, because are returning here
    only the updated target nodes features.
    TODO: Change this to conform with general Pytorch Geometric interface.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_features: int,
        out_features: int,
        act: nn.Module,
        dropout: float = 0.5,
        node_aggr: Aggregation = SumAggregation,
        cord_aggr: Aggregation = MeanAggregation,
        residual: bool = True,
        update_coords: bool = True,
        norm_coords: bool = True,
        norm_coors_scale_init: float = 1e-2,
        norm_feats: bool = True,
        initialization_gain: float = 1,
        return_pos: bool = True,
        attention = False
    ):
        super().__init__(aggr=None)
        self.node_aggr = node_aggr()
        self.cord_aggr = cord_aggr()
        self.residual = residual
        self.update_coords = update_coords
        self.act = act
        self.initialization_gain = initialization_gain
        self.return_pos = return_pos
        self.attention = attention

        if (node_features != out_features) and residual:
            raise ValueError(
                "Residual connections are only compatible with the same input and output dimensions."
            )

        self.message_net = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_features),
            nn.Dropout(dropout),
            act,
            nn.Linear(hidden_features, hidden_features),
            act
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_features, 1),
                nn.Sigmoid()
            )

        self.update_net = nn.Sequential(
            nn.Linear(node_features + hidden_features, hidden_features),
            nn.Dropout(dropout),
            act,
            nn.Linear(hidden_features, out_features),
        )

        layer = nn.Linear(hidden_features, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.pos_net = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.Dropout(dropout),
            act,
            layer,
        )

        self.node_norm = (
            torch_geometric.nn.norm.LayerNorm(node_features) if norm_feats else nn.Identity()
        )
        self.coors_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init) if norm_coords else nn.Identity()
        )

        # self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            if (type(self.act) is nn.SELU):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear", mode="fan_in")
                nn.init.zeros_(module.bias)
            else:
                # seems to be needed to keep the network from exploding to NaN with greater depths
                nn.init.xavier_normal_(module.weight, gain=self.initialization_gain)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        pos: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptPairTensor = None,
    ):
        # TODO: Think about a better solution for the residual connection
        if self.residual:
            residual = x if isinstance(x, Tensor) else x[1]
        x_dest, pos = self.propagate(
            edge_index, x=x, pos=pos, edge_attr=edge_attr, edge_weight=edge_weight
        )

        if self.residual:
            x_dest = x_dest + residual

        out = (x_dest, pos) if self.return_pos else x_dest
        return out

    def message(
        self, x_i: Tensor, x_j: Tensor, pos_i: Tensor, pos_j: Tensor, edge_weight: OptTensor = None
    ):
        """Create messages"""
        pos_dir = pos_i - pos_j
        dist = torch.norm(pos_dir, dim=-1, keepdim=True)
        input = [self.node_norm(x_i), self.node_norm(x_j), dist]
        input = torch.cat(input, dim=-1)
        node_message = self.message_net(input)
        if self.attention:
            node_message = self.att_mlp(node_message) * node_message
        pos_message = self.coors_norm(pos_dir) * self.pos_net(node_message)
        if edge_weight is not None:
            node_message = node_message * edge_weight.unsqueeze(-1)
            pos_message = pos_message * edge_weight.unsqueeze(-1)

        return node_message, pos_message

    def aggregate(
        self,
        inputs: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Tensor = None,
        dim_size: int = None,
    ) -> Tensor:
        node_message, pos_message = inputs
        agg_node_message = self.node_aggr(node_message, index, ptr, dim_size)
        agg_pos_message = self.cord_aggr(pos_message, index, ptr, dim_size)
        return agg_node_message, agg_pos_message

    def update(
        self,
        message: Tuple[Tensor, Tensor],
        x: Union[Tensor, OptPairTensor],
        pos: Union[Tensor, OptPairTensor],
    ):
        node_message, pos_message = message
        x_, pos_ = (x, pos) if isinstance(x, Tensor) else (x[1], pos[1])
        input = torch.cat((x_, node_message), dim=-1)
        x_new = self.update_net(input)
        pos_new = pos_ + pos_message if self.update_coords else pos
        return x_new, pos_new


class EGNNGlobalNodeHetero(nn.Module):
    """E(n)-equivariant Message Passing Network"""

    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        out_features,
        num_layers,
        act=nn.SiLU(),
        dropout=0.0,
        node_aggr=SumAggregation,
        cord_aggr=MeanAggregation,
        update_coords=True,
        residual=True,
        norm_coords=True,
        norm_coors_scale_init=1e-2,
        norm_feats=True,
        initialization_gain=1,
        weight_share=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.weight_share = weight_share
        if weight_share:
            # Use a single layer that will be shared across all iterations
            self.shared_layer = self.create_layer(
                node_features,
                edge_features,
                hidden_features,
                out_features,
                act,
                dropout,
                node_aggr,
                cord_aggr,
                update_coords,
                residual,
                norm_coords,
                norm_coors_scale_init,
                norm_feats,
                initialization_gain,
            )
        else:
            # Create a list of layers, one for each iteration
            self.layers = nn.ModuleList(
                [
                    self.create_layer(
                        node_features,
                        edge_features,
                        hidden_features,
                        out_features,
                        act,
                        dropout,
                        node_aggr,
                        cord_aggr,
                        update_coords,
                        residual,
                        norm_coords,
                        norm_coors_scale_init,
                        norm_feats,
                        initialization_gain,
                    )
                    for _ in range(num_layers)
                ]
            )

    def create_layer(
        self,
        node_features,
        edge_features,
        hidden_features,
        out_features,
        act,
        dropout,
        node_aggr,
        cord_aggr,
        update_coords,
        residual,
        norm_coords,
        norm_coors_scale_init,
        norm_feats,
        initialization_gain,
    ):
        # Centralized layer creation logic
        return nn.ModuleDict(
            {
                "atom_to_atom": EGNNLayer(
                    node_features=node_features,
                    edge_features=edge_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act=act,
                    dropout=dropout,
                    node_aggr=node_aggr,
                    cord_aggr=cord_aggr,
                    residual=residual,
                    update_coords=update_coords,
                    norm_coords=False,
                    norm_coors_scale_init=norm_coors_scale_init,
                    norm_feats=False,
                    initialization_gain=initialization_gain,
                    attention=True
                ),
                "atom_to_global_node": EGNNLayer(
                    node_features=node_features,
                    edge_features=edge_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act=act,
                    dropout=dropout,
                    node_aggr=node_aggr,
                    cord_aggr=cord_aggr,
                    residual=residual,
                    update_coords=update_coords,
                    norm_coords=norm_coords,
                    norm_coors_scale_init=norm_coors_scale_init,
                    norm_feats=norm_feats,
                    initialization_gain=initialization_gain,
                    attention=False
                ),
                "global_node_to_atom": EGNNLayer(
                    node_features=node_features,
                    edge_features=edge_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act=act,
                    dropout=dropout,
                    node_aggr=node_aggr,
                    cord_aggr=cord_aggr,
                    residual=residual,
                    update_coords=update_coords,
                    norm_coords=norm_coords,
                    norm_coors_scale_init=norm_coors_scale_init,
                    norm_feats=norm_feats,
                    initialization_gain=initialization_gain,
                    attention=False
                ),
            }
        )


    def forward(
        self,
        x_atom,
        pos_atom,
        x_global_node,
        pos_global_node,
        edge_index_atom_atom,
        edge_index_atom_global_node,
        edge_index_global_node_atom,
    ):
        for i in range(self.num_layers):
            layer = self.shared_layer if self.weight_share else self.layers[i]
            x_atom, pos_atom = layer["atom_to_atom"](
                x=(x_atom, x_atom),
                edge_index=edge_index_atom_atom,
                pos=(pos_atom, pos_atom),
            )
            pairwise_atom = x_atom
            x_global_node, pos_global_node = layer["atom_to_global_node"](
                x=(x_atom, x_global_node),
                edge_index=edge_index_atom_global_node,
                pos=(pos_atom, pos_global_node),
            )
            x_atom, pos_atom = layer["global_node_to_atom"](
                x=(x_global_node, x_atom),
                edge_index=edge_index_global_node_atom,
                pos=(pos_global_node, pos_atom),
            )

        return x_atom, x_global_node, pos_atom, pos_global_node, pairwise_atom