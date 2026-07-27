"""Matched inference ablations for the frozen HEGNNPPIS_Dual architecture."""

from model import HEGNNPPIS_Dual


class HEGNNPPISDualAblation(HEGNNPPIS_Dual):
    """Keep checkpoint-compatible parameters while bypassing one component."""

    def __init__(self, *args, ablation="full", **kwargs):
        if ablation not in ("full", "wo_ha_vn", "wo_dhre"):
            raise ValueError(f"unknown ablation: {ablation}")
        super().__init__(*args, **kwargs)
        self.ablation = ablation

    def _forward_layers(self, node_feat, node_pos, hyper_node_feat, hyper_node_pos,
                        edge_index, n2h_edge_index, h2n_edge_index,
                        hypergraph_full, hypergraph_selective):
        edge_index = edge_index.squeeze(0)
        node_feat = node_feat.squeeze(0)
        hyper_node_feat = hyper_node_feat.squeeze(1)
        n2h_edge_index = n2h_edge_index.squeeze(0)
        h2n_edge_index = h2n_edge_index.squeeze(0)
        node_feat = self._in(node_feat)
        hyper_node_feat = self._in(hyper_node_feat)

        global_last = full_last = selective_last = None
        for i in range(self.layers):
            global_h, hyper_node_feat, node_pos, hyper_node_pos, pairwise_h = self._modules[
                f"global_{i}"
            ](node_feat, node_pos, hyper_node_feat, hyper_node_pos, edge_index,
              n2h_edge_index, h2n_edge_index)
            # pairwise_h is captured before global-node-to-residue propagation.
            backbone_h = pairwise_h if self.ablation == "wo_ha_vn" else global_h
            if self.ablation == "wo_dhre":
                node_feat = backbone_h
                full_h = selective_h = None
            else:
                full_h = self._modules[f"local_full_{i}"](pairwise_h, hypergraph_full)
                selective_h = self._modules[f"local_selective_{i}"](
                    pairwise_h, hypergraph_selective
                )
                node_feat = (backbone_h + self.alpha_full * full_h
                             + self.alpha_selective * selective_h)
            global_last, full_last, selective_last = backbone_h, full_h, selective_h
        return node_feat, global_last, full_last, selective_last
