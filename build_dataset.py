import argparse
import os
import pandas as pd
import numpy as np
import torch
from reader import MicronsReader

def build_dataset(structural_session, hf_session, data_dir='dataset', out_dir='processed'):
    print(f"\n=== BUILDING DATASET FOR SESSION {hf_session} ===")
    
    # 1. Setup Output Directory
    session_out = os.path.join(out_dir, f"session_{hf_session}")
    os.makedirs(session_out, exist_ok=True)

    # 2. Load Structural Data (Nodes & Edges)
    print("Loading structural connectome (BossDB pickles)...")
    nodes = pd.read_pickle(os.path.join(data_dir, 'node_data_v1.pkl'))
    edges = pd.read_pickle(os.path.join(data_dir, 'edge_data_v1.pkl'))

    session_nodes = nodes[nodes['scan_session'] == structural_session].copy()
    valid_nodes = session_nodes.dropna(subset=['unit_id'])
    
    ordered_nucleus_ids = valid_nodes['nucleus_id'].tolist()
    golden_unit_ids = valid_nodes['unit_id'].astype(int).tolist()
    
    num_nodes = len(ordered_nucleus_ids)
    print(f"Selected {num_nodes} target neurons with both physical and functional data.")
    # 3. Build PyTorch Geometric Graph (Topology)
    print("Constructing PyTorch Geometric Edge Index...")
    nuc2idx = {nid: i for i, nid in enumerate(ordered_nucleus_ids)}
    
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
    print(f"Graph constructed: {edge_index.shape[1]} edges.")

    # 4. Extract Functional Time-Series (Signals)
    print("Extracting functional calcium traces (Hugging Face H5)...")
    trace_list, behavior_list, condition_hashes = [], [], []

    with MicronsReader(os.path.join(data_dir, 'microns.h5')) as reader:
        session_hashes = reader.get_hashes_by_session(hf_session, return_unique=True)
        
        # Read unit_ids from H5 to align arrays
        all_session_unit_ids = reader.f[f'sessions/{hf_session}/meta/unit_ids'][:].tolist()
        indices = [all_session_unit_ids.index(uid) for uid in golden_unit_ids if uid in all_session_unit_ids]
        
        for hash_val in session_hashes:
            data = reader.get_full_data_by_hash(hash_val)
            for trial in data['trials']:
                if trial['session'] != hf_session:
                    continue
                    
                trial_traces = trial['responses'][indices, :].T
                pupil_data = trial['pupil'].T
                treadmill_data = trial['treadmill'].reshape(-1, 1)
                trial_behavior = np.hstack((treadmill_data, pupil_data))
                
                trace_list.append(trial_traces)
                behavior_list.append(trial_behavior)
                condition_hashes.extend([hash_val] * trial_traces.shape[0])

    X = np.vstack(trace_list)
    U = np.vstack(behavior_list)

    np.save(os.path.join(session_out, 'X_features.npy'), X)
    np.save(os.path.join(session_out, 'U_exogenous.npy'), U)
    pd.Series(condition_hashes, name='condition_hash').to_csv(os.path.join(session_out, 'conditions.csv'), index=False)

    print(f"\n=== SUCCESS ===")
    print(f"Output saved to: {session_out}/")
    print(f"Node Features (X): {X.shape}")
    print(f"Exogenous (U):     {U.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PyTorch Spatiotemporal Dataset from MICrONS")
    parser.add_argument('--struct_session', type=float, default=4.0, help='BossDB Structural Session (e.g., 4.0)')
    parser.add_argument('--hf_session', type=str, default='4_7', help='HuggingFace Functional Session (e.g., 4_7)')
    args = parser.parse_args()
    
    build_dataset(args.struct_session, args.hf_session)