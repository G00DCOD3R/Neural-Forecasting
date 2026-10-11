import argparse
import os
import pandas as pd
import numpy as np
import torch
from reader import MicronsReader

def load_cave_edges(structural_session, nucleus_ids, mat_version, proofread_only, data_dir='dataset'):
    """Aggregate CAVE synapses between the given nuclei into (pre_nucleus_id, post_nucleus_id, n_synapses) edges."""
    from caveclient import CAVEclient

    client = CAVEclient('minnie65_public')
    client.materialize.version = mat_version

    # Nucleus IDs are stable across versions, root IDs change with every proofreading edit
    nuc = client.materialize.query_table(
        'nucleus_detection_v0',
        filter_in_dict={'id': list(nucleus_ids)},
        select_columns=['id', 'pt_root_id'],
    )
    nuc = nuc[nuc['pt_root_id'] != 0]
    nuc = nuc[~nuc['pt_root_id'].duplicated(keep=False)]  # roots holding several nuclei are merge errors
    root2nuc = dict(zip(nuc['pt_root_id'], nuc['id']))
    roots = list(root2nuc)

    # Same cache file as cave_sanity_check.py
    cache_path = os.path.join(data_dir, f'cave_synapses_s{structural_session}_v{mat_version}.pkl')
    if os.path.exists(cache_path):
        syn = pd.read_pickle(cache_path)
    else:
        syn = client.materialize.query_table(
            'synapses_pni_2',
            filter_in_dict={'pre_pt_root_id': roots, 'post_pt_root_id': roots},
            select_columns=['pre_pt_root_id', 'post_pt_root_id', 'size'],
        )
        syn.to_pickle(cache_path)
    syn = syn[syn['pre_pt_root_id'] != syn['post_pt_root_id']]  # drop autapses

    if proofread_only:
        proof = client.materialize.query_table('proofreading_status_and_strategy')
        proof_roots = set(proof.loc[proof['status_axon'] == True, 'pt_root_id'])
        syn = syn[syn['pre_pt_root_id'].isin(proof_roots)]

    cave_edges = (
        syn.groupby(['pre_pt_root_id', 'post_pt_root_id'])
        .size().rename('n_synapses').reset_index()
    )
    cave_edges['pre_nucleus_id'] = cave_edges['pre_pt_root_id'].map(root2nuc)
    cave_edges['post_nucleus_id'] = cave_edges['post_pt_root_id'].map(root2nuc)
    return cave_edges[['pre_nucleus_id', 'post_nucleus_id', 'n_synapses']]

def build_dataset(structural_session, hf_session, edge_source='bossdb', mat_version=1926, data_dir='dataset', out_dir='processed', out_name=None):
    print(f"\n=== BUILDING DATASET FOR SESSION {hf_session} ===")

    # 1. Setup Output Directory
    session_out = os.path.join(out_dir, out_name or f"session_{hf_session}")
    os.makedirs(session_out, exist_ok=True)

    # 2. Load Structural Data (Nodes & Edges)
    print("Loading structural connectome (BossDB pickles)...")
    nodes = pd.read_pickle(os.path.join(data_dir, 'node_data_v1.pkl'))

    session_nodes = nodes[nodes['scan_session'] == structural_session].copy()
    valid_nodes = session_nodes.dropna(subset=['unit_id'])
    
    ordered_nucleus_ids = valid_nodes['nucleus_id'].tolist()
    golden_unit_ids = valid_nodes['unit_id'].astype(int).tolist()
    
    num_nodes = len(ordered_nucleus_ids)
    print(f"Selected {num_nodes} target neurons with both physical and functional data.")
    # 3. Build PyTorch Geometric Graph (Topology)
    print(f"Constructing PyTorch Geometric Edge Index (edge source: {edge_source})...")
    nuc2idx = {nid: i for i, nid in enumerate(ordered_nucleus_ids)}

    if edge_source == 'bossdb':
        edges = pd.read_pickle(os.path.join(data_dir, 'edge_data_v1.pkl'))
    else:
        print(f"Loading CAVE synapses (materialization v{mat_version})...")
        edges = load_cave_edges(structural_session, ordered_nucleus_ids, mat_version,
                                proofread_only=(edge_source == 'cave_proofread'), data_dir=data_dir)

    valid_edges = edges[
        edges['pre_nucleus_id'].isin(nuc2idx) & 
        edges['post_nucleus_id'].isin(nuc2idx)
    ].copy()

    valid_edges = valid_edges[valid_edges['n_synapses'] > 0]

    valid_edges['src_idx'] = valid_edges['pre_nucleus_id'].map(nuc2idx)
    valid_edges['dst_idx'] = valid_edges['post_nucleus_id'].map(nuc2idx)

    src = valid_edges['src_idx'].values
    dst = valid_edges['dst_idx'].values
    edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    edge_weight = torch.tensor(valid_edges['n_synapses'].values, dtype=torch.float32)

    torch.save(edge_index, os.path.join(session_out, 'edge_index.pt'))
    torch.save(edge_weight, os.path.join(session_out, 'edge_weight.pt'))
    with open(os.path.join(session_out, 'edge_source.txt'), 'w') as f:
        f.write(f"{edge_source}\n" if edge_source == 'bossdb' else f"{edge_source} (CAVE v{mat_version})\n")
    print(f"Graph constructed: {edge_index.shape[1]} edges.")

    # 4. Extract Functional Time-Series (Signals)
    print("Extracting functional calcium traces (Hugging Face H5)...")
    trial_list = []

    with MicronsReader(os.path.join(data_dir, 'microns.h5')) as reader:
        session_hashes = reader.get_hashes_by_session(hf_session, return_unique=True)
        
        # Read unit_ids from H5 to align arrays
        all_session_unit_ids = reader.f[f'sessions/{hf_session}/meta/unit_ids'][:].tolist()
        indices = [all_session_unit_ids.index(uid) for uid in golden_unit_ids if uid in all_session_unit_ids]
        # A missing unit would shift every later column of X out of line with the graph nodes
        assert len(indices) == num_nodes, f"{num_nodes - len(indices)} unit_ids missing from the H5 file"

        for hash_val in session_hashes:
            data = reader.get_full_data_by_hash(hash_val)
            for trial in data['trials']:
                if trial['session'] != hf_session:
                    continue
                    
                trial_traces = trial['responses'][indices, :].T
                pupil_data = trial['pupil'].T
                treadmill_data = trial['treadmill'].reshape(-1, 1)
                trial_behavior = np.hstack((treadmill_data, pupil_data))
                trial_list.append((trial['stim_times'], hash_val, trial['trial_idx'], trial_traces, trial_behavior))

    # Hashes come from a set, so put the trials back in recording order (stim_times are seconds since session start)
    trial_list.sort(key=lambda t: t[0][0])
    X = np.vstack([t[3] for t in trial_list])
    U = np.vstack([t[4] for t in trial_list])
    rows = pd.DataFrame({
        'condition_hash': np.concatenate([[t[1]] * len(t[0]) for t in trial_list]),
        'trial_id': np.concatenate([[i] * len(t[0]) for i, t in enumerate(trial_list)]),
        'trial_name': np.concatenate([[t[2]] * len(t[0]) for t in trial_list]),
        'time': np.concatenate([t[0] for t in trial_list]),
    })

    np.save(os.path.join(session_out, 'X_features.npy'), X)
    np.save(os.path.join(session_out, 'U_exogenous.npy'), U)
    rows.to_csv(os.path.join(session_out, 'conditions.csv'), index=False)

    print(f"\n=== SUCCESS ===")
    print(f"Output saved to: {session_out}/")
    print(f"Node Features (X): {X.shape}")
    print(f"Exogenous (U):     {U.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PyTorch Spatiotemporal Dataset from MICrONS")
    parser.add_argument('--struct_session', type=float, default=4.0, help='BossDB Structural Session (e.g., 4.0)')
    parser.add_argument('--hf_session', type=str, default='4_7', help='HuggingFace Functional Session (e.g., 4_7)')
    parser.add_argument('--edge_source', type=str, default='bossdb', choices=['bossdb', 'cave_proofread', 'cave_all'],
                        help='bossdb: v1 release edges; cave_proofread: CAVE synapses from proofread axons; cave_all: all CAVE synapses')
    parser.add_argument('--mat_version', type=int, default=1926, help='CAVE materialization version (cave_* edge sources only)')
    parser.add_argument('--out_name', type=str, default=None, help='Output folder inside processed/ (default: session_<hf_session>)')
    args = parser.parse_args()

    build_dataset(args.struct_session, args.hf_session, args.edge_source, args.mat_version, out_name=args.out_name)