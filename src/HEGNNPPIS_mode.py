from layers import *
from config import Config
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HEGNNPPIS(nn.Module):
    def __init__(self, in_dim=67, in_edge_dim=1, hidden_dim=67, layers=4, dropout=0.0, alpha=Config.feature_fusion_alpha):
        super(HEGNNPPIS, self).__init__()
        self.layers = layers
        for i in range(layers):
            self.add_module("global_%d" % i, EGNNGlobalNodeHetero(node_features=hidden_dim, edge_features=in_edge_dim,
                                                                  hidden_features=hidden_dim, dropout=dropout,
                                                                  num_layers=1, out_features=hidden_dim))
            self.add_module("local_%d" % i, HGNNPConv(in_dim=hidden_dim, out_dim=hidden_dim, residual=True))
        self.alpha = alpha
        self._in = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.drop = nn.Dropout(dropout)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=5,
                                                                    min_lr=1e-6)

    def forward(self, node_feat, node_pos, hyper_node_feat, hyper_node_pos, edge_index, n2h_edge_index,
                h2n_edge_index, hypergraph):
        edge_index = edge_index.squeeze(0)
        node_feat = node_feat.squeeze(0)
        hyper_node_feat = hyper_node_feat.squeeze(1)
        n2h_edge_index = n2h_edge_index.squeeze(0)
        h2n_edge_index = h2n_edge_index.squeeze(0)
        node_feat = self._in(node_feat)
        hyper_node_feat = self._in(hyper_node_feat)
        for i in range(self.layers):
            global_h, hyper_node_feat, node_pos, hyper_node_pos, pairwise_atom = self._modules["global_%d" % i](
                node_feat, node_pos, hyper_node_feat,
                hyper_node_pos, edge_index, n2h_edge_index, h2n_edge_index)
            local_h = self._modules["local_%d" % i](pairwise_atom, hypergraph)
            node_feat = global_h * (1 - self.alpha) + local_h * self.alpha
        out = self.out(node_feat)
        return out