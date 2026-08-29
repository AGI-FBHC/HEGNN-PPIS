"""
Generate Surface Patch Hyperedges and compute Hyperedge Enrichment.

5 versions:
A: Current k-hop + feature kNN (baseline)
B: Surface-r8 (small surface patch)
C: Surface-r10 (recommended)
D: Surface-r12 (large surface patch)
E: Current + Surface-r10 (complementary)

Surface Patch definition:
    e_i = {j | RSA_i > 0.2, RSA_j > 0.2, d(i,j) < r}
    min_hyperedge_size = 3
    max_hyperedge_size = 20 or 30

Hyperedge Enrichment:
    Enrichment(e) = (#interface_residues_in_e / |e|) / (#interface_residues_in_protein / N)
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
from dhg import Graph, Hypergraph
from tqdm import tqdm
import argparse

SOURCE_DIR = Path(__file__).resolve().parent


def compute_distance_matrix(pos):
    """Compute pairwise distance matrix."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist


def generate_surface_patch_hyperedges(pos, rsa, radius=10.0, rsa_threshold=0.2,
                                       min_size=3, max_size=30):
    """
    Generate surface patch hyperedges.

    Args:
        pos: (N, 3) residue positions
        rsa: (N,) RSA values (1-dim, total RSA)
        radius: spatial radius in Angstroms
        rsa_threshold: RSA threshold for surface residues
        min_size: minimum hyperedge size
        max_size: maximum hyperedge size

    Returns:
        hyperedges: list of lists, each list contains residue indices
    """
    N = len(rsa)
    dist_matrix = compute_distance_matrix(pos)

    hyperedges = []

    for i in range(N):
        # Center residue must be on surface
        if rsa[i] < rsa_threshold:
            continue

        # Find surface neighbors within radius
        neighbors = []
        for j in range(N):
            if i == j:
                continue
            if rsa[j] >= rsa_threshold and dist_matrix[i, j] < radius:
                neighbors.append(j)

        # Create hyperedge: center + neighbors
        hyperedge = [i] + neighbors

        # Apply size constraints
        if len(hyperedge) < min_size:
            continue
        if len(hyperedge) > max_size:
            # Keep closest neighbors
            distances = [dist_matrix[i, j] for j in neighbors]
            sorted_indices = np.argsort(distances)[:max_size - 1]
            hyperedge = [i] + [neighbors[idx] for idx in sorted_indices]

        hyperedges.append(hyperedge)

    return hyperedges


def generate_conserved_surface_patch_hyperedges(pos, rsa, pssm, radius=10.0,
                                                rsa_threshold=0.2, cons_percentile=60,
                                                min_size=3, max_size=20):
    """
    Generate conserved surface patch hyperedges (Version S1).

    e_i = {j | RSA_i > tau, RSA_j > tau, d(i,j) < r, Cons_j > q}

    Conservation proxy: PSSM max score per residue.

    Args:
        pos: (N, 3) residue positions
        rsa: (N,) RSA values
        pssm: (N, 20) PSSM matrix
        radius: spatial radius in Angstroms
        rsa_threshold: RSA threshold for surface residues
        cons_percentile: conservation percentile threshold (top 40% -> 60th percentile)
        min_size: minimum hyperedge size
        max_size: maximum hyperedge size

    Returns:
        hyperedges: list of lists
    """
    N = len(rsa)
    dist_matrix = compute_distance_matrix(pos)

    # Compute conservation scores from PSSM
    cons_scores = pssm.max(axis=1)
    cons_thresh = np.percentile(cons_scores, cons_percentile)

    # Masks
    surface_mask = rsa > rsa_threshold
    conserved_mask = cons_scores >= cons_thresh

    hyperedges = []

    for i in range(N):
        # Center must be surface + conserved
        if not (surface_mask[i] and conserved_mask[i]):
            continue

        # Find neighbors that are surface + conserved + within radius
        neighbors = []
        for j in range(N):
            if i == j:
                continue
            if surface_mask[j] and conserved_mask[j] and dist_matrix[i, j] < radius:
                neighbors.append(j)

        hyperedge = [i] + neighbors

        if len(hyperedge) < min_size:
            continue
        if len(hyperedge) > max_size:
            distances = [dist_matrix[i, j] for j in neighbors]
            sorted_indices = np.argsort(distances)[:max_size - 1]
            hyperedge = [i] + [neighbors[idx] for idx in sorted_indices]

        hyperedges.append(hyperedge)

    return hyperedges


def generate_hotspot_surface_patch_hyperedges(pos, rsa, sequence, radius=10.0,
                                               rsa_threshold=0.2, min_size=3, max_size=20):
    """
    Generate hotspot-centered surface patch hyperedges (Version S2).

    e_i = {j | RSA_i > tau, a_i in A_hot, RSA_j > tau, d(i,j) < r}

    Hotspot residues (broad): W, Y, F, R, L, I, V, M

    Args:
        pos: (N, 3) residue positions
        rsa: (N,) RSA values
        sequence: str, amino acid sequence
        radius: spatial radius in Angstroms
        rsa_threshold: RSA threshold for surface residues
        min_size: minimum hyperedge size
        max_size: maximum hyperedge size

    Returns:
        hyperedges: list of lists
    """
    N = len(rsa)
    dist_matrix = compute_distance_matrix(pos)

    # Hotspot residue types (broad)
    hotspot_types = set('WYFRILVM')

    hotspot_mask = np.array([aa in hotspot_types for aa in sequence])
    surface_mask = rsa > rsa_threshold

    hyperedges = []

    for i in range(N):
        # Center must be surface + hotspot
        if not (surface_mask[i] and hotspot_mask[i]):
            continue

        # Find neighbors that are surface + within radius
        neighbors = []
        for j in range(N):
            if i == j:
                continue
            if surface_mask[j] and dist_matrix[i, j] < radius:
                neighbors.append(j)

        hyperedge = [i] + neighbors

        if len(hyperedge) < min_size:
            continue
        if len(hyperedge) > max_size:
            distances = [dist_matrix[i, j] for j in neighbors]
            sorted_indices = np.argsort(distances)[:max_size - 1]
            hyperedge = [i] + [neighbors[idx] for idx in sorted_indices]

        hyperedges.append(hyperedge)

    return hyperedges


def generate_current_hyperedges(node_feat, edge_list, num_nodes):
    """
    Generate current hyperedges (k-hop + feature kNN).

    Args:
        node_feat: (N, D) node features
        edge_list: list of [src, dst] edges
        num_nodes: number of nodes

    Returns:
        hyperedges: list of lists
    """
    graph = Graph(num_nodes, edge_list)

    # Current method: k-hop + feature kNN
    hypergraph = Hypergraph.from_graph_kHop(k=1, graph=graph, only_kHop=True)
    hypergraph.add_hyperedges_from_feature_kNN(node_feat, k=2)
    hypergraph.add_hyperedges_from_graph_kHop(graph, k=2, only_kHop=True)

    # Convert to list of lists
    hyperedges = []
    for e_idx in range(hypergraph.num_e):
        edge_nodes = list(hypergraph.e[0][e_idx])
        if len(edge_nodes) >= 2:
            hyperedges.append(edge_nodes)

    return hyperedges


def generate_combined_hyperedges(node_feat, edge_list, num_nodes, pos, rsa,
                                  surface_radius=10.0, rsa_threshold=0.2):
    """
    Generate combined hyperedges: current + surface patch.

    Args:
        node_feat: (N, D) node features
        edge_list: list of [src, dst] edges
        num_nodes: number of nodes
        pos: (N, 3) positions
        rsa: (N,) RSA values
        surface_radius: radius for surface patches
        rsa_threshold: RSA threshold

    Returns:
        hyperedges: list of lists
    """
    # Current hyperedges
    current_hyperedges = generate_current_hyperedges(node_feat, edge_list, num_nodes)

    # Surface patch hyperedges
    surface_hyperedges = generate_surface_patch_hyperedges(
        pos, rsa, radius=surface_radius, rsa_threshold=rsa_threshold)

    # Combine
    combined = current_hyperedges + surface_hyperedges

    return combined


def hyperedges_to_dhg(hyperedges, num_nodes):
    """Convert hyperedges list to dhg.Hypergraph."""
    if len(hyperedges) == 0:
        return Hypergraph(num_nodes)

    # Filter out empty or single-node edges
    valid_edges = [e for e in hyperedges if len(e) >= 2]

    if len(valid_edges) == 0:
        return Hypergraph(num_nodes)

    hg = Hypergraph(num_nodes, valid_edges)
    return hg


def compute_hyperedge_enrichment(hyperedges, labels, interface_threshold=1):
    """
    Compute Hyperedge Enrichment.

    Enrichment(e) = (#interface_residues_in_e / |e|) / (#interface_residues_in_protein / N)

    Args:
        hyperedges: list of lists
        labels: (N,) binary labels (1 = interface, 0 = non-interface)
        interface_threshold: not used, labels are already binary

    Returns:
        enrichments: list of enrichment values for each hyperedge
        mean_enrichment: mean enrichment across all hyperedges
    """
    N = len(labels)
    interface_rate_protein = np.sum(labels) / N

    if interface_rate_protein == 0:
        return [], 0.0

    enrichments = []
    for hyperedge in hyperedges:
        if len(hyperedge) == 0:
            continue

        # Count interface residues in hyperedge
        interface_in_edge = sum(1 for idx in hyperedge if labels[idx] == 1)
        interface_rate_edge = interface_in_edge / len(hyperedge)

        # Compute enrichment
        enrichment = interface_rate_edge / interface_rate_protein
        enrichments.append(enrichment)

    mean_enrichment = np.mean(enrichments) if enrichments else 0.0

    return enrichments, mean_enrichment


def process_protein(protein_id, dataset, rsa_dir, pos_dir, graph_dir, feature_dir,
                    version='C', surface_radius=10.0, rsa_threshold=0.2):
    """
    Process a single protein to generate hyperedges.

    Args:
        protein_id: protein ID
        dataset: dataset dict
        rsa_dir: RSA features directory
        pos_dir: positions directory
        graph_dir: edge_index directory
        feature_dir: features directory
        version: 'A', 'B', 'C', 'D', 'E'
        surface_radius: radius for surface patches
        rsa_threshold: RSA threshold

    Returns:
        hypergraph: dhg.Hypergraph
        enrichments: list of enrichment values
        mean_enrichment: mean enrichment
    """
    # Load data
    item = dataset[protein_id]
    sequence = item[0]
    labels = np.array(item[1])

    # Load RSA
    rsa_path = os.path.join(rsa_dir, f"{protein_id}.npy")
    if not os.path.exists(rsa_path):
        return None, [], 0.0
    rsa = np.load(rsa_path)
    if len(rsa.shape) > 1:
        rsa = rsa[:, 0]  # Use total RSA

    # Load positions
    if os.path.isfile(pos_dir) and pos_dir.endswith('.pkl'):
        # pos_dir is actually a path to an aggregated .pkl file
        psepos_dict = pickle.load(open(pos_dir, 'rb'))
        if protein_id not in psepos_dict:
            return None, [], 0.0
        pos = psepos_dict[protein_id]
    else:
        pos_path = os.path.join(pos_dir, f"{protein_id}.npy")
        if not os.path.exists(pos_path):
            return None, [], 0.0
        pos = np.load(pos_path)
    reference = pos[0]
    pos = pos - reference

    # Load edge_index
    edge_path = os.path.join(graph_dir, f"{protein_id}.npy")
    if not os.path.exists(edge_path):
        return None, [], 0.0
    edge_index = np.load(edge_path)
    edge_list = edge_index.T.tolist()

    # Load features
    pssm = np.load(os.path.join(feature_dir, "pssm", f"{protein_id}.npy"))
    hmm = np.load(os.path.join(feature_dir, "hmm", f"{protein_id}.npy"))
    dssp = np.load(os.path.join(feature_dir, "dssp", f"{protein_id}.npy"))
    rsa_feat = np.load(os.path.join(feature_dir, "rsa", f"{protein_id}.npy"))
    resaf = np.load(os.path.join(feature_dir, "resAF", f"{protein_id}.npy"))

    seq_embedding = np.concatenate([pssm, hmm], axis=1)
    node_features = np.concatenate([seq_embedding, dssp, rsa_feat, resaf], axis=1)
    node_features = torch.from_numpy(node_features).float()

    # Compute distance to centroid
    centroid_dist = np.sqrt(np.sum(pos ** 2, axis=1)).reshape(-1, 1) / 15.0
    node_features = np.concatenate([node_features.numpy(), centroid_dist], axis=1)
    node_features = torch.from_numpy(node_features).float()

    num_nodes = len(sequence)

    # Generate hyperedges based on version
    if version == 'A':
        # Current: k-hop + feature kNN
        hyperedges = generate_current_hyperedges(node_features, edge_list, num_nodes)
    elif version == 'B':
        # Surface-r8
        hyperedges = generate_surface_patch_hyperedges(
            pos, rsa, radius=8.0, rsa_threshold=rsa_threshold)
    elif version == 'C':
        # Surface-r10 (recommended)
        hyperedges = generate_surface_patch_hyperedges(
            pos, rsa, radius=10.0, rsa_threshold=rsa_threshold)
    elif version == 'D':
        # Surface-r12
        hyperedges = generate_surface_patch_hyperedges(
            pos, rsa, radius=12.0, rsa_threshold=rsa_threshold)
    elif version == 'E':
        # Current + Surface-r10
        hyperedges = generate_combined_hyperedges(
            node_features, edge_list, num_nodes, pos, rsa,
            surface_radius=10.0, rsa_threshold=rsa_threshold)
    elif version == 'F':
        # E-filtered: Start with E, then filter out noise
        hyperedges = generate_combined_hyperedges(
            node_features, edge_list, num_nodes, pos, rsa,
            surface_radius=10.0, rsa_threshold=rsa_threshold)

        # Filter 1: Remove size == 2 (just graph edges, not hyperedges)
        # Filter 2: Remove oversized + low purity (size > 100, purity < 0.05)
        interface_indices = set(np.where(labels == 1)[0])
        filtered = []
        for edge in hyperedges:
            size = len(edge)
            if size == 2:
                continue
            if size > 100:
                edge_interface = sum(1 for n in edge if n in interface_indices)
                purity = edge_interface / size if size > 0 else 0
                if purity < 0.05:
                    continue
            filtered.append(edge)
        hyperedges = filtered
    elif version == 'S1':
        # Conserved Surface Patch: surface + spatial + conservation
        hyperedges = generate_conserved_surface_patch_hyperedges(
            pos, rsa, pssm, radius=surface_radius, rsa_threshold=rsa_threshold)
    elif version == 'S2':
        # Hotspot-centered Surface Patch: surface + hotspot residue center
        hyperedges = generate_hotspot_surface_patch_hyperedges(
            pos, rsa, sequence, radius=surface_radius, rsa_threshold=rsa_threshold)
    else:
        raise ValueError(f"Unknown version: {version}")

    # Compute enrichment
    enrichments, mean_enrichment = compute_hyperedge_enrichment(hyperedges, labels)

    # Convert to dhg.Hypergraph
    hypergraph = hyperedges_to_dhg(hyperedges, num_nodes)

    return hypergraph, enrichments, mean_enrichment


def main():
    parser = argparse.ArgumentParser(description="Generate Surface Patch Hyperedges")
    parser.add_argument("--dataset", type=str, default=str(SOURCE_DIR / "Dataset/Train_335.pkl"),
                        help="Dataset pickle file")
    parser.add_argument("--output_dir", type=str, default=str(SOURCE_DIR / "Graph/SC/hypergraph_surface"),
                        help="Output directory")
    parser.add_argument("--rsa_dir", type=str, default=str(SOURCE_DIR / "Feature/rsa"),
                        help="RSA directory")
    parser.add_argument("--pos_dir", type=str, default=str(SOURCE_DIR / "Feature/psepos"),
                        help="Position directory")
    parser.add_argument("--graph_dir", type=str, default=str(SOURCE_DIR / "Graph/SC/edge_index"),
                        help="Edge index directory")
    parser.add_argument("--feature_dir", type=str, default=str(SOURCE_DIR / "Feature"),
                        help="Feature directory")
    parser.add_argument("--version", type=str, default="C",
                        choices=['A', 'B', 'C', 'D', 'E', 'F', 'S1', 'S2'],
                        help="Hyperedge version")
    parser.add_argument("--surface_radius", type=float, default=10.0,
                        help="Surface patch radius")
    parser.add_argument("--rsa_threshold", type=float, default=0.2,
                        help="RSA threshold")

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    if '2j3rA' in dataset:
        dataset.pop('2j3rA')

    # Create output directory
    version_name = {
        'A': 'current',
        'B': 'surface_r8',
        'C': 'surface_r10',
        'D': 'surface_r12',
        'E': 'current_plus_surface_r10',
        'F': 'current_plus_surface_r10_filtered',
        'S1': 'conserved_surface_r10',
        'S2': 'hotspot_surface_r10',
    }
    output_dir = os.path.join(args.output_dir, version_name[args.version])
    os.makedirs(output_dir, exist_ok=True)

    # Process each protein
    print(f"Processing {len(dataset)} proteins with version {args.version}...")

    all_enrichments = []
    protein_enrichments = {}

    for protein_id in tqdm(dataset.keys()):
        output_path = os.path.join(output_dir, f"{protein_id}")

        # Check if already processed
        if os.path.exists(output_path):
            # Load and compute enrichment
            with open(output_path, 'rb') as f:
                hypergraph = pickle.load(f)

            # Recompute enrichment for analysis
            item = dataset[protein_id]
            labels = np.array(item[1])
            hyperedges = []
            for e_idx in range(hypergraph.num_e):
                edge_nodes = list(hypergraph.e[0][e_idx])
                hyperedges.append(edge_nodes)
            enrichments, mean_enrichment = compute_hyperedge_enrichment(hyperedges, labels)
            all_enrichments.extend(enrichments)
            protein_enrichments[protein_id] = mean_enrichment
            continue

        try:
            hypergraph, enrichments, mean_enrichment = process_protein(
                protein_id, dataset, args.rsa_dir, args.pos_dir, args.graph_dir,
                args.feature_dir, version=args.version,
                surface_radius=args.surface_radius, rsa_threshold=args.rsa_threshold)

            if hypergraph is not None:
                # Save hypergraph
                with open(output_path, 'wb') as f:
                    pickle.dump(hypergraph, f, protocol=pickle.HIGHEST_PROTOCOL)

                all_enrichments.extend(enrichments)
                protein_enrichments[protein_id] = mean_enrichment

        except Exception as e:
            print(f"Error processing {protein_id}: {e}")
            continue

    # Print summary
    print("\n" + "="*80)
    print(f"HYPEREDGE ENRICHMENT SUMMARY - Version {args.version} ({version_name[args.version]})")
    print("="*80)

    if all_enrichments:
        print(f"\nTotal hyperedges: {len(all_enrichments)}")
        print(f"Mean enrichment: {np.mean(all_enrichments):.4f}")
        print(f"Std enrichment: {np.std(all_enrichments):.4f}")
        print(f"Median enrichment: {np.median(all_enrichments):.4f}")
        print(f"Enrichment > 1 (interface-enriched): {sum(1 for e in all_enrichments if e > 1)} ({sum(1 for e in all_enrichments if e > 1)/len(all_enrichments)*100:.1f}%)")

        # Per-protein statistics
        mean_per_protein = np.mean(list(protein_enrichments.values()))
        print(f"\nMean enrichment per protein: {mean_per_protein:.4f}")

    # Save enrichment data
    enrichment_file = os.path.join(args.output_dir, f"enrichment_{version_name[args.version]}.pkl")
    with open(enrichment_file, 'wb') as f:
        pickle.dump({
            'version': args.version,
            'all_enrichments': all_enrichments,
            'protein_enrichments': protein_enrichments
        }, f)
    print(f"\nEnrichment data saved to: {enrichment_file}")


if __name__ == "__main__":
    import torch
    main()
