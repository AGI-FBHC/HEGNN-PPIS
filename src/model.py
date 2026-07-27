"""
HEGNN-PPIS dual-branch selective hypergraph model.

Architecture:
- VN-EGNN backbone (shared)
- Full hypergraph branch (local_full)
- Selective hypergraph branch (local_selective)
- Residual fusion: h = h_VN + alpha_full * h_full + alpha_selective * h_selective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import HGNNPConv, EGNNGlobalNodeHetero, EGNNLayer
from config import Config
import warnings
warnings.filterwarnings("ignore")


class HEGNNPPIS_Dual(nn.Module):
    """
    Dual-branch HEGNN-PPIS with separate full and selective hypergraph branches.

    Fusion: node_feat = global_h + alpha_full * local_full + alpha_selective * local_selective
    """
    def __init__(self, in_dim=67, in_edge_dim=1, hidden_dim=67, layers=4, dropout=0.0,
                 alpha_full=0.05, alpha_selective=0.10,
                 use_virtual_nodes=True, use_hyperedges=True):
        super(HEGNNPPIS_Dual, self).__init__()
        self.layers = layers
        self.use_virtual_nodes = use_virtual_nodes
        self.use_hyperedges = use_hyperedges
        self.alpha_full = alpha_full
        self.alpha_selective = alpha_selective

        # Global branch (with or without virtual nodes)
        if use_virtual_nodes:
            for i in range(layers):
                self.add_module("global_%d" % i, EGNNGlobalNodeHetero(
                    node_features=hidden_dim, edge_features=in_edge_dim,
                    hidden_features=hidden_dim, dropout=dropout,
                    num_layers=1, out_features=hidden_dim))
        else:
            for i in range(layers):
                self.add_module("atom_%d" % i, EGNNLayer(
                    node_features=hidden_dim, edge_features=in_edge_dim,
                    hidden_features=hidden_dim, out_features=hidden_dim,
                    act=nn.SiLU(), dropout=dropout, residual=True,
                    update_coords=True, norm_coords=False, norm_feats=False,
                    attention=True))

        # Local branch 1: full hypergraph
        # Local branch 2: selective hypergraph
        if use_hyperedges:
            for i in range(layers):
                self.add_module("local_full_%d" % i, HGNNPConv(
                    in_dim=hidden_dim, out_dim=hidden_dim, residual=True))
                self.add_module("local_selective_%d" % i, HGNNPConv(
                    in_dim=hidden_dim, out_dim=hidden_dim, residual=True))

        # Input projection
        self._in = nn.Linear(in_dim, hidden_dim)

        # Output
        self.out = nn.Linear(hidden_dim, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.drop = nn.Dropout(dropout)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=Config.learning_rate,
                                          weight_decay=Config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.6, patience=5, min_lr=1e-6)

    def _forward_layers(self, node_feat, node_pos, hyper_node_feat, hyper_node_pos,
                        edge_index, n2h_edge_index, h2n_edge_index,
                        hypergraph_full, hypergraph_selective):
        """Shared forward logic, returns final node_feat and branch outputs from last layer."""
        edge_index = edge_index.squeeze(0)
        node_feat = node_feat.squeeze(0)
        hyper_node_feat = hyper_node_feat.squeeze(1)
        n2h_edge_index = n2h_edge_index.squeeze(0)
        h2n_edge_index = h2n_edge_index.squeeze(0)

        node_feat = self._in(node_feat)
        hyper_node_feat = self._in(hyper_node_feat)

        global_h_last = None
        local_full_last = None
        local_selective_last = None

        for i in range(self.layers):
            if self.use_virtual_nodes:
                global_h, hyper_node_feat, node_pos, hyper_node_pos, pairwise_atom = \
                    self._modules["global_%d" % i](
                        node_feat, node_pos, hyper_node_feat,
                        hyper_node_pos, edge_index, n2h_edge_index, h2n_edge_index)
            else:
                global_h, node_pos = self._modules["atom_%d" % i](
                    x=(node_feat, node_feat),
                    edge_index=edge_index,
                    pos=(node_pos, node_pos))
                pairwise_atom = global_h

            if self.use_hyperedges:
                local_full = self._modules["local_full_%d" % i](pairwise_atom, hypergraph_full)
                local_selective = self._modules["local_selective_%d" % i](pairwise_atom, hypergraph_selective)
                node_feat = global_h + self.alpha_full * local_full + self.alpha_selective * local_selective
                global_h_last = global_h
                local_full_last = local_full
                local_selective_last = local_selective
            else:
                node_feat = global_h
                global_h_last = global_h

        return node_feat, global_h_last, local_full_last, local_selective_last

    def forward(self, node_feat, node_pos, hyper_node_feat, hyper_node_pos,
                edge_index, n2h_edge_index, h2n_edge_index,
                hypergraph_full, hypergraph_selective):
        """
        Forward pass with dual hypergraphs.

        Args:
            hypergraph_full: dhg.Hypergraph (full coverage, e.g., baseline A)
            hypergraph_selective: dhg.Hypergraph (selective, e.g., S2 hotspot)
        """
        node_feat, _, _, _ = self._forward_layers(
            node_feat, node_pos, hyper_node_feat, hyper_node_pos,
            edge_index, n2h_edge_index, h2n_edge_index,
            hypergraph_full, hypergraph_selective)
        out = self.out(node_feat)
        return out

    def forward_diagnostic(self, node_feat, node_pos, hyper_node_feat, hyper_node_pos,
                           edge_index, n2h_edge_index, h2n_edge_index,
                           hypergraph_full, hypergraph_selective):
        """
        Forward pass with diagnostic branch norms.

        Returns:
            logits: (N, 2) output logits
            diag: dict with branch norms and ratios
        """
        node_feat, global_h, local_full, local_selective = self._forward_layers(
            node_feat, node_pos, hyper_node_feat, hyper_node_pos,
            edge_index, n2h_edge_index, h2n_edge_index,
            hypergraph_full, hypergraph_selective)

        out = self.out(node_feat)

        diag = {}
        if global_h is not None:
            global_norm = torch.norm(global_h, dim=1).mean().item()
            diag['global_norm'] = global_norm
        else:
            diag['global_norm'] = 0.0

        if local_full is not None:
            full_contrib = self.alpha_full * local_full
            full_norm = torch.norm(full_contrib, dim=1).mean().item()
            raw_full_norm = torch.norm(local_full, dim=1).mean().item()
            diag['full_norm'] = full_norm
            diag['raw_full_norm'] = raw_full_norm
            diag['full_ratio'] = full_norm / (diag['global_norm'] + 1e-8)
        else:
            diag['full_norm'] = 0.0
            diag['raw_full_norm'] = 0.0
            diag['full_ratio'] = 0.0

        if local_selective is not None:
            sel_contrib = self.alpha_selective * local_selective
            sel_norm = torch.norm(sel_contrib, dim=1).mean().item()
            raw_sel_norm = torch.norm(local_selective, dim=1).mean().item()
            diag['selective_norm'] = sel_norm
            diag['raw_selective_norm'] = raw_sel_norm
            diag['selective_ratio'] = sel_norm / (diag['global_norm'] + 1e-8)
            if diag['full_norm'] > 1e-8:
                diag['selective_full_ratio'] = sel_norm / (diag['full_norm'] + 1e-8)
            else:
                diag['selective_full_ratio'] = 0.0
        else:
            diag['selective_norm'] = 0.0
            diag['raw_selective_norm'] = 0.0
            diag['selective_ratio'] = 0.0
            diag['selective_full_ratio'] = 0.0

        return out, diag


def get_dual_model(alpha_full=0.05, alpha_selective=0.10, **kwargs):
    """Factory function to create dual-branch HEGNN-PPIS model."""
    return HEGNNPPIS_Dual(alpha_full=alpha_full, alpha_selective=alpha_selective, **kwargs)
