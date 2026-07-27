import itertools
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


from config import Config
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

def init():
    SEED = 2020
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(SEED)
    pass

def embedding(sequence_name):
    pssm_feature = np.load(Config.feature_path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Config.feature_path + "hmm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)


def get_pssm_features(sequence_name):
    dssp_feature = np.load(Config.feature_path + "pssm/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


def get_hmm_features(sequence_name):
    dssp_feature = np.load(Config.feature_path + "hmm/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load(Config.feature_path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Config.feature_path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)


def get_rsa_feature(sequence_name):
    rsa_feature = np.load(Config.feature_path + "rsa/" + sequence_name + '.npy')
    return rsa_feature.astype(np.float32)


def load_edge_index(sequence_name):
    radius = np.load(Config.graph_path + Config.center + 'edge_index/' + sequence_name + '.npy')
    return radius


def load_hypergraph(
    sequence_name,
    hypergraph_dir=os.path.join(Config.graph_path, Config.center, "hypergraph"),
):
    file_path = os.path.join(hypergraph_dir, sequence_name)
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph



def cal_adj(sequence_name, radius=Config.MAP_CUTOFF):
    dist_matrix = np.load(Config.feature_path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(np.int32)
    return adjacency_matrix


def cal_edge_index(sequence_name, radius=Config.MAP_CUTOFF):
    dist_matrix = np.load(Config.feature_path + "distance_map_" + Config.center + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(np.int64)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list


def graph_collate(samples):
    sequence_name, label, node_features, virtual_node_features, pos, virtual_pos, edge_index, A2V_edge_index, V2A_edge_index, hypergraph = map(list, zip(*samples))
    label = torch.Tensor(label)
    node_features = torch.cat(node_features)
    virtual_node_features = torch.cat(virtual_node_features)
    pos = torch.cat(pos)
    pos = torch.Tensor(pos)
    # hypergraph = torch.Tensor(hypergraph)
    virtual_pos = torch.cat(virtual_pos)
    virtual_pos = torch.Tensor(virtual_pos)
    edge_index = torch.Tensor(edge_index)
    A2V_edge_index = torch.Tensor(A2V_edge_index[0])
    V2A_edge_index = torch.Tensor(V2A_edge_index[0])
    return sequence_name, label, node_features, virtual_node_features, pos, virtual_pos, edge_index, A2V_edge_index, V2A_edge_index, hypergraph


def graph_collate_dual(samples):
    """Collate function for dual hypergraph (full + selective)."""
    sequence_name, label, node_features, virtual_node_features, pos, virtual_pos, \
        edge_index, A2V_edge_index, V2A_edge_index, hypergraph_full, hypergraph_selective = map(list, zip(*samples))
    label = torch.Tensor(label)
    node_features = torch.cat(node_features)
    virtual_node_features = torch.cat(virtual_node_features)
    pos = torch.cat(pos)
    pos = torch.Tensor(pos)
    virtual_pos = torch.cat(virtual_pos)
    virtual_pos = torch.Tensor(virtual_pos)
    edge_index = torch.Tensor(edge_index)
    A2V_edge_index = torch.Tensor(A2V_edge_index[0])
    V2A_edge_index = torch.Tensor(V2A_edge_index[0])
    return (sequence_name, label, node_features, virtual_node_features, pos, virtual_pos,
            edge_index, A2V_edge_index, V2A_edge_index, hypergraph_full, hypergraph_selective)


class ProDataset(Dataset):
    def __init__(self, dataframe, radius=Config.MAP_CUTOFF, dist=Config.DIST_NORM,
                 psepos_path=Config.Train335_psepos_path, hypernodes=3,
                 hypergraph_dir=os.path.join(Config.graph_path, Config.center, "hypergraph"),
                 random_virtual_rotations=True):
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))

        # Filter dataframe to only include proteins with psepos data
        valid_indices = [i for i, name in enumerate(dataframe['ID'].values) if name in self.residue_psepos]
        if len(valid_indices) < len(dataframe):
            print(f"Warning: {len(dataframe) - len(valid_indices)} proteins missing from psepos file, skipping them")

        self.names = dataframe['ID'].values[valid_indices]
        self.sequences = dataframe['sequence'].values[valid_indices]
        self.labels = dataframe['label'].values[valid_indices]
        self.radius = radius
        self.dist = dist
        self.hypernodes = hypernodes
        self.hypergraph_dir = hypergraph_dir
        self.random_virtual_rotations = random_virtual_rotations

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        pos = self.residue_psepos[sequence_name]
        nodes_num = len(sequence)
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos)

        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        rsa_features = get_rsa_feature(sequence_name)
        res_atom_features = get_res_atom_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features, rsa_features, res_atom_features], axis=1)
        node_features = torch.from_numpy(node_features)
        node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

        hyper_node_features = torch.stack(
            [torch.mean(node_features, dim=0, keepdim=True) for _ in range(self.hypernodes)]
        ).squeeze()

        edge_index = load_edge_index(sequence_name)

        centroid = torch.mean(pos, dim=0, keepdim=True)
        radius = torch.max(torch.norm(pos - centroid, dim=1))
        hyper_pos = sample_global_node_starting_positions(
            centroid=centroid, radius=radius, num_points=self.hypernodes,
            random_rotations=self.random_virtual_rotations
        )

        src_atom = list(
            itertools.chain.from_iterable(
                [list(range(nodes_num)) for i in range(self.hypernodes)]
            )
        )
        dst_global_node = list(
            itertools.chain.from_iterable(
                [[i] * nodes_num for i in range(self.hypernodes)]
            )
        )
        hypergraph = load_hypergraph(sequence_name, self.hypergraph_dir)
        A2V_edge_index = torch.LongTensor(
            [src_atom, dst_global_node]
        )
        V2A_edge_index = torch.LongTensor(
            [dst_global_node, src_atom]
        )
        node_features = node_features.detach().numpy()
        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
        return sequence_name, label, node_features, hyper_node_features, pos, hyper_pos, edge_index, A2V_edge_index, V2A_edge_index, hypergraph

    def __len__(self):
        return len(self.labels)


class ProDatasetDual(Dataset):
    """Dataset that loads two hypergraphs: full (baseline) and selective (e.g., S2 hotspot)."""
    def __init__(self, dataframe, radius=Config.MAP_CUTOFF, dist=Config.DIST_NORM,
                 psepos_path=Config.Train335_psepos_path,
                 hypernodes=3,
                 hypergraph_dir_full=os.path.join(Config.graph_path, Config.center, "hypergraph"),
                 hypergraph_dir_selective=os.path.join(
                     Config.graph_path, Config.center, "hypergraph_surface", "hotspot_surface_r10"
                 ),
                 random_virtual_rotations=True):
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))

        valid_indices = [i for i, name in enumerate(dataframe['ID'].values) if name in self.residue_psepos]
        if len(valid_indices) < len(dataframe):
            print(f"Warning: {len(dataframe) - len(valid_indices)} proteins missing from psepos file, skipping them")

        self.names = dataframe['ID'].values[valid_indices]
        self.sequences = dataframe['sequence'].values[valid_indices]
        self.labels = dataframe['label'].values[valid_indices]
        self.radius = radius
        self.dist = dist
        self.hypernodes = hypernodes
        self.hypergraph_dir_full = hypergraph_dir_full
        self.hypergraph_dir_selective = hypergraph_dir_selective
        self.random_virtual_rotations = random_virtual_rotations

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        pos = self.residue_psepos[sequence_name]
        nodes_num = len(sequence)
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos)

        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        rsa_features = get_rsa_feature(sequence_name)
        res_atom_features = get_res_atom_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features, rsa_features, res_atom_features], axis=1)
        node_features = torch.from_numpy(node_features)
        node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

        hyper_node_features = torch.stack(
            [torch.mean(node_features, dim=0, keepdim=True) for _ in range(self.hypernodes)]
        ).squeeze()

        edge_index = load_edge_index(sequence_name)

        centroid = torch.mean(pos, dim=0, keepdim=True)
        radius = torch.max(torch.norm(pos - centroid, dim=1))
        hyper_pos = sample_global_node_starting_positions(
            centroid=centroid, radius=radius, num_points=self.hypernodes,
            random_rotations=self.random_virtual_rotations
        )

        src_atom = list(
            itertools.chain.from_iterable(
                [list(range(nodes_num)) for i in range(self.hypernodes)]
            )
        )
        dst_global_node = list(
            itertools.chain.from_iterable(
                [[i] * nodes_num for i in range(self.hypernodes)]
            )
        )
        hypergraph_full = load_hypergraph(sequence_name, self.hypergraph_dir_full)
        hypergraph_selective = load_hypergraph(sequence_name, self.hypergraph_dir_selective)
        A2V_edge_index = torch.LongTensor(
            [src_atom, dst_global_node]
        )
        V2A_edge_index = torch.LongTensor(
            [dst_global_node, src_atom]
        )
        node_features = node_features.detach().numpy()
        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
        return (sequence_name, label, node_features, hyper_node_features, pos, hyper_pos,
                edge_index, A2V_edge_index, V2A_edge_index, hypergraph_full, hypergraph_selective)

    def __len__(self):
        return len(self.labels)


def random_rotation_matrix():
    """Generate a random 3x3 rotation matrix using PyTorch."""
    theta = 2 * torch.pi * torch.rand(1)  # Random rotation around the z-axis
    phi = torch.acos(2 * torch.rand(1) - 1)  # Random rotation around the y-axis
    psi = 2 * torch.pi * torch.rand(1)  # Random rotation around the x-axis

    Rz = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    Ry = torch.tensor(
        [[torch.cos(phi), 0, torch.sin(phi)], [0, 1, 0], [-torch.sin(phi), 0, torch.cos(phi)]]
    )
    Rx = torch.tensor(
        [[1, 0, 0], [0, torch.cos(psi), -torch.sin(psi)], [0, torch.sin(psi), torch.cos(psi)]]
    )
    R = torch.mm(Rz, torch.mm(Ry, Rx))  # Combined rotation matrix
    return R


def sample_global_node_starting_positions(
    centroid: torch.tensor,
    radius: torch.tensor,
    num_points: int,
    random_rotations: bool = True,
) -> torch.tensor:
    golden_ratio = (1.0 + torch.sqrt(torch.tensor(5.0))) / 2.0

    theta = 2 * torch.pi * torch.arange(num_points).float() / golden_ratio
    phi = torch.acos(1 - 2 * (torch.arange(num_points).float() + 0.5) / num_points)
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)

    points = torch.stack((x, y, z), dim=1)
    if random_rotations:
        rotation_matrix = random_rotation_matrix()
        points = torch.mm(points, rotation_matrix.T)  # Corrected rotation step

    points = centroid + points

    return points
