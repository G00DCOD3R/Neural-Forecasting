import os
import pandas as pd
from caveclient import CAVEclient

# Same check as sanity_check.py, but the edges come from the full CAVE synapse
# table instead of the BossDB v1 release (which only covers 148 proofread axons).
# Requires a CAVE token: https://caveconnectome.github.io/CAVEclient/tutorials/authentication/

target_session = 4.0
datastack = 'minnie65_public'
mat_version = None  # None = latest public materialization
cache_dir = 'dataset'

print("Loading pickles...")
nodes = pd.read_pickle('dataset/node_data_v1.pkl')
edges = pd.read_pickle('dataset/edge_data_v1.pkl')

client = CAVEclient(datastack)
if mat_version is not None:
    client.materialize.version = mat_version
mat_version = client.materialize.version
print(f"Checking session: {target_session} (CAVE materialization v{mat_version})\n")

# 1. Filter golden nodes (identical to sanity_check.py)
session_nodes = nodes[nodes['scan_session'] == target_session].copy()
valid_nodes = session_nodes.dropna(subset=['unit_id'])
golden_nucleus_ids = set(valid_nodes['nucleus_id'].tolist())
total_neurons = len(golden_nucleus_ids)

# 2. Map nucleus IDs to current segment (root) IDs. Nucleus IDs are stable
# across versions, root IDs change with every proofreading edit.
nuc = client.materialize.query_table(
    'nucleus_detection_v0',
    filter_in_dict={'id': list(golden_nucleus_ids)},
    select_columns=['id', 'pt_root_id'],
)
nuc = nuc[nuc['pt_root_id'] != 0]
# A root holding several nuclei is a merge error; its synapses can't be attributed
shared_roots = nuc['pt_root_id'].duplicated(keep=False)
nuc = nuc[~shared_roots]
root2nuc = dict(zip(nuc['pt_root_id'], nuc['id']))
roots = list(root2nuc)

print("=== NUCLEUS -> SEGMENT MAPPING ===")
print(f"Golden nuclei: {total_neurons}")
print(f"Mapped to a unique segment: {len(roots)}")
print(f"Dropped (no segment or merged with another nucleus): {total_neurons - len(roots)}")
print("==================================\n")

# 3. Query all synapses within the functional subgraph (cached, the query is slow)
cache_path = os.path.join(cache_dir, f'cave_synapses_s{target_session}_v{mat_version}.pkl')
if os.path.exists(cache_path):
    syn = pd.read_pickle(cache_path)
else:
    print("Querying synapses_pni_2 (may take a few minutes)...")
    syn = client.materialize.query_table(
        'synapses_pni_2',
        filter_in_dict={'pre_pt_root_id': roots, 'post_pt_root_id': roots},
        select_columns=['pre_pt_root_id', 'post_pt_root_id', 'size'],
    )
    syn.to_pickle(cache_path)
syn = syn[syn['pre_pt_root_id'] != syn['post_pt_root_id']]  # drop autapses (mostly segmentation errors)

# 4. Aggregate synapses into weighted edges
cave_edges = (
    syn.groupby(['pre_pt_root_id', 'post_pt_root_id'])
    .agg(n_synapses=('size', 'size'), synapse_size=('size', 'sum'))
    .reset_index()
)
cave_edges['pre_nucleus_id'] = cave_edges['pre_pt_root_id'].map(root2nuc)
cave_edges['post_nucleus_id'] = cave_edges['post_pt_root_id'].map(root2nuc)

# Flag edges whose presynaptic axon has been proofread at this version
proof = client.materialize.query_table('proofreading_status_and_strategy')
proof_roots = set(proof.loc[proof['status_axon'] == True, 'pt_root_id'])
cave_edges['pre_proofread'] = cave_edges['pre_pt_root_id'].isin(proof_roots)

print(f"Synapses in subgraph: {len(syn)}")
print(f"Edges in subgraph: {len(cave_edges)}")
print(f"Distinct presynaptic neurons: {cave_edges['pre_nucleus_id'].nunique()}")
print(f"  of which proofread axons: {cave_edges.loc[cave_edges['pre_proofread'], 'pre_nucleus_id'].nunique()}")
print(f"Edges from proofread axons: {cave_edges['pre_proofread'].sum()}\n")


def in_degree_of(edge_df):
    deg = edge_df.groupby('post_nucleus_id').size()
    deg = deg.reindex(list(golden_nucleus_ids), fill_value=0)
    deg.name = 'in_degree'
    return deg


# 5. Print Graph Density, for increasingly strict synapse-count thresholds.
# Unproofread axons carry false merges; requiring several synapses per edge
# filters out most of the resulting spurious single-synapse edges.
for min_syn in [1, 2, 3]:
    for label, subset in [('all axons', cave_edges), ('proofread axons only', cave_edges[cave_edges['pre_proofread']])]:
        in_degree = in_degree_of(subset[subset['n_synapses'] >= min_syn])
        print(f"=== GRAPH DENSITY CHECK (n_synapses >= {min_syn}, {label}) ===")
        for threshold in [1, 3, 5]:
            count = (in_degree >= threshold).sum()
            pct = (count / total_neurons) * 100
            print(f">= {threshold} presynaptic partners: {count} ({pct:.1f}%)")
        print(f"mean in-degree: {in_degree.mean():.2f}, median: {in_degree.median():.0f}, "
              f"edges: {(subset['n_synapses'] >= min_syn).sum()}")
        print()

# 6. Print Stats (all axons, >= 1 synapse)
in_degree = in_degree_of(cave_edges)
print("=== ACTUAL DEGREE STATS (all axons, n_synapses >= 1) ===")
print(in_degree.describe().to_string())

print("\n=== TOP 5 MOST CONNECTED NEURONS ===")
top_5 = in_degree.sort_values(ascending=False).head(5).reset_index()
top_5['nucleus_id'] = top_5['post_nucleus_id']
print(top_5[['nucleus_id', 'post_nucleus_id', 'in_degree']].to_string())

# 7. Agreement with the BossDB v1 release
bossdb = edges[
    edges['pre_nucleus_id'].isin(golden_nucleus_ids) &
    edges['post_nucleus_id'].isin(golden_nucleus_ids) &
    (edges['n_synapses'] > 0)
]
bossdb_pairs = set(zip(bossdb['pre_nucleus_id'], bossdb['post_nucleus_id']))
cave_pairs = set(zip(cave_edges['pre_nucleus_id'], cave_edges['post_nucleus_id']))
print("\n=== AGREEMENT WITH BOSSDB v1 ===")
print(f"BossDB v1 edges: {len(bossdb_pairs)}")
print(f"Also present in CAVE v{mat_version}: {len(bossdb_pairs & cave_pairs)}")
print("================================")
