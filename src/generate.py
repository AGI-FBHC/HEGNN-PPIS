from dataloader import *
from dhg import Graph, Hypergraph
import math
import freesasa
import numpy as np
import torch

pdb_path = './Dataset/pdb/'
Dataset_Path = "./Dataset/"


def generate_hypergraph():
    with open(Dataset_Path + "Test_335.pkl", "rb") as f:
        Train_335 = pickle.load(f)
        Train_335.pop('2j3rA')  # remove the protein with error sequence in the train dataset
    # Btest_31_6 = {}
    # with open(Config.dataset_path + "bound_unbound_mapping31-6.txt", "r") as f:
    #     lines = f.readlines()[1:]
    # for line in lines:
    #     bound_ID, unbound_ID, _ = line.strip().split()
    #     Btest_31_6[bound_ID] = Train_335[bound_ID]
    IDs, sequences, labels = [], [], []
    # Train_335 = Btest_31_6
    _dict = {}
    for ID in Train_335:
        IDs.append(ID)
        item = Train_335[ID]
        sequences.append(item[0])
        radius = np.load(Config.graph_path + Config.center + 'edge_index/' + ID + '.npy')
        edge_index = torch.from_numpy(radius)
        edge_list = edge_index.t().tolist()
        residue_psepos = pickle.load(open('./Feature/psepos/Test60_psepos_SC.pkl', 'rb'))
        pos = residue_psepos[ID]
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos)
        sequence_embedding = embedding(ID)
        structural_features = get_dssp_features(ID)
        rsa_features = get_rsa_feature(ID)
        res_atom_features = get_res_atom_features(ID)
        node_features = np.concatenate([sequence_embedding, structural_features, rsa_features, res_atom_features],
                                       axis=1)
        node_features = torch.from_numpy(node_features)
        node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / 15],
                                  dim=-1)
        node_feat = node_features
        graph = Graph(node_feat.shape[0], edge_list)
        # hypergraph = Hypergraph.from_graph(graph)
        # hypergraph = Hypergraph.from_feature_kNN(node_feat, k=2)
        hypergraph = Hypergraph.from_graph_kHop(k=1, graph=graph, only_kHop=True)
        # hypergraph.add_hyperedges_from_graph_kHop(graph, k=1, only_kHop=True)
        hypergraph.add_hyperedges_from_feature_kNN(node_feat, k=2)
        hypergraph.add_hyperedges_from_graph_kHop(graph, k=2, only_kHop=True)
        # hypergraph.add_hyperedges_from_graph_kHop(graph, k=3, only_kHop=True)
        save_path = './Graph/SC/hypergraph/' + ID
        with open(save_path, 'wb') as f:
            pickle.dump(hypergraph, f, protocol=pickle.HIGHEST_PROTOCOL)
    pass


def generate_RSA_feature(protein, chains):
    RSA_dict = {}
    structure = freesasa.Structure(pdb_path + protein + '.pdb')
    result = freesasa.calc(structure, freesasa.Parameters({'algorithm': freesasa.LeeRichards, 'n-slices': 100, 'probe-radius': 1.4}))
    residueAreas = result.residueAreas()
    RSA = []
    for c in chains:
        for r in residueAreas[c.upper()].keys():
            RSA_AA = []
            RSA_AA.append(min(1, residueAreas[c.upper()][r].relativeTotal))
            RSA_AA.append(min(1, residueAreas[c.upper()][r].relativePolar))
            RSA_AA.append(min(1, residueAreas[c.upper()][r].relativeApolar))
            RSA_AA.append(min(1, residueAreas[c.upper()][r].relativeMainChain))
            if math.isnan(residueAreas[c.upper()][r].relativeSideChain):
                RSA_AA.append(0)
            else:
                RSA_AA.append(min(1,residueAreas[c.upper()][r].relativeSideChain))
            RSA.append(RSA_AA)
    RSA_dict[protein] = RSA
    return RSA_dict