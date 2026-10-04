import pandas as pd

print("Loading pickles...")
nodes = pd.read_pickle('dataset/node_data_v1.pkl')
edges = pd.read_pickle('dataset/edge_data_v1.pkl')

target_session = 4.0
print(f"Checking session: {target_session}\n")

# 1. Filter golden nodes
session_nodes = nodes[nodes['scan_session'] == target_session].copy()
valid_nodes = session_nodes.dropna(subset=['unit_id'])
golden_nucleus_ids = set(valid_nodes['nucleus_id'].tolist())

total_neurons = len(golden_nucleus_ids)

# 2. Filter edges strictly to our functional subgraph
valid_edges = edges[
    edges['pre_nucleus_id'].isin(golden_nucleus_ids) & 
    edges['post_nucleus_id'].isin(golden_nucleus_ids)
]
valid_edges = valid_edges[valid_edges['n_synapses'] > 0]

# 3. Calculate in-degree (how many inputs each neuron receives)
in_degree = valid_edges.groupby('post_nucleus_id').size()
in_degree = in_degree.reindex(list(golden_nucleus_ids), fill_value=0)
in_degree.name = 'in_degree'

# 4. Print Graph Density 
print("=== GRAPH DENSITY CHECK ===")
print(f"Total neurons: {total_neurons}")
for threshold in [1, 3, 5]:
    count = (in_degree >= threshold).sum()
    pct = (count / total_neurons) * 100
    print(f">= {threshold} presynaptic partners: {count} ({pct:.1f}%)")
print("===========================\n")

# 5. Print Stats
print("=== ACTUAL DEGREE STATS ===")
print(in_degree.describe().to_string())

print("\n=== TOP 5 MOST CONNECTED NEURONS ===")
top_5 = in_degree.sort_values(ascending=False).head(5).reset_index()
top_5['nucleus_id'] = top_5['post_nucleus_id']
print(top_5[['nucleus_id', 'post_nucleus_id', 'in_degree']].to_string())