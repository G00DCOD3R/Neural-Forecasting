import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm

# ==========================================
# 0. SETUP & DATA LOADING
# ==========================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join("results", f"baselines_{timestamp}")
os.makedirs(results_dir, exist_ok=True)
summary_file = os.path.join(results_dir, "metrics_summary.txt")
best_gru_path = os.path.join(results_dir, "best_gru_model.pth")

print(f"Run initialized. Outputs will be saved to: {results_dir}")

print("Loading data...")
data_dir = os.path.join('processed', 'session_4_7')
X = np.load(os.path.join(data_dir, 'X_features.npy'))
U = np.load(os.path.join(data_dir, 'U_exogenous.npy'))
conditions = pd.read_csv(os.path.join(data_dir, 'conditions.csv'))['condition_hash'].values

# Handle both NaNs and Infinities explicitly per the review recommendations
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)

# 70-15-15 Chronological Split
total_steps = X.shape[0]
train_idx = int(total_steps * 0.70)
val_idx = int(total_steps * 0.85)

X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
U_train, U_val, U_test = U[:train_idx], U[train_idx:val_idx], U[val_idx:]
conds_train, conds_val, conds_test = conditions[:train_idx], conditions[train_idx:val_idx], conditions[val_idx:]

history_len = 10
num_neurons = X.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Trial Boundary Handler
def get_valid_starts(conds_split, hist_len):
    """Ensures a sliding window does not cross different video stimulus trials."""
    valid = []
    for t in range(len(conds_split) - hist_len):
        if conds_split[t] == conds_split[t + hist_len]:
            valid.append(t)
    return valid

train_starts = get_valid_starts(conds_train, history_len)
val_starts = get_valid_starts(conds_val, history_len)
test_starts = get_valid_starts(conds_test, history_len)

# ==========================================
# 1. COPY-LAST BASELINE
# ==========================================
def evaluate_copy_last(X_data, valid_starts, hist_len):
    """Evaluates Copy-last ONLY on the exact same valid target rows as the GRU."""
    preds, targets = [], []
    for t in valid_starts:
        preds.append(X_data[t + hist_len - 1, :])
        targets.append(X_data[t + hist_len, :])
        
    preds = np.array(preds)
    targets = np.array(targets)
    mse = np.mean((preds - targets) ** 2)
    
    corrs = []
    for i in range(targets.shape[1]):
        # Skip zero-variance neurons to prevent undefined correlation
        if np.std(preds[:, i]) == 0 or np.std(targets[:, i]) == 0:
            continue
        r, _ = pearsonr(preds[:, i], targets[:, i])
        if np.isfinite(r):
            corrs.append(r)
            
    mean_corr = np.mean(corrs) if corrs else np.nan
    return mse, mean_corr

test_mse, test_corr = evaluate_copy_last(X_test, test_starts, history_len)
print("\n=== COPY-LAST BASELINE (TEST SET) ===")
print(f"MSE: {test_mse:.6f} | Pearson r: {test_corr:.4f}")

with open(summary_file, "a", encoding="utf-8") as f:
    f.write("=== COPY-LAST BASELINE ===\n")
    f.write(f"MSE: {test_mse:.6f} | Pearson r: {test_corr:.4f}\n\n")

# ==========================================
# 2. INDEPENDENT GRU BASELINE
# ==========================================
class NeuralDataset(Dataset):
    def __init__(self, X_data, U_data, valid_starts, hist_len):
        self.X = torch.FloatTensor(X_data)
        self.U = torch.FloatTensor(U_data)
        self.valid_starts = valid_starts
        self.history_len = hist_len
        self.num_neurons = X_data.shape[1]

    def __len__(self):
        return len(self.valid_starts) * self.num_neurons

    def __getitem__(self, idx):
        t_idx = idx // self.num_neurons
        n_idx = idx % self.num_neurons
        
        t_start = self.valid_starts[t_idx]
        t_end = t_start + self.history_len
        
        x_hist = self.X[t_start:t_end, n_idx].unsqueeze(1)
        u_hist = self.U[t_start:t_end, :]
        features = torch.cat([x_hist, u_hist], dim=1)
        
        target = self.X[t_end, n_idx]
        return features, target

class IndependentGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

train_dataset = NeuralDataset(X_train, U_train, train_starts, history_len)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

val_dataset = NeuralDataset(X_val, U_val, val_starts, history_len)
val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

# Derive input dimension dynamically from exogenous array
input_dim = 1 + U.shape[1]
model = IndependentGRU(input_dim=input_dim, hidden_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction='sum') 

epochs = 5 # THIS IS ONLY TO TEST, NORMALLY INCREASE IT TO HIGHER
best_val_loss = float('inf')
patience = 4
patience_counter = 0

print("\nTraining Independent GRU Baseline...")
for epoch in range(epochs):
    model.train()
    total_train_sse = 0.0
    total_train_preds = 0
    
    for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        preds = model(batch_x).squeeze(-1) # Squeeze(-1) preserves neuron dim
        loss = criterion(preds, batch_y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_train_sse += loss.item()
        total_train_preds += batch_y.numel()
        
    train_mse = total_train_sse / total_train_preds
    
    model.eval()
    total_val_sse = 0.0
    total_val_preds = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x).squeeze(-1)
            loss = criterion(preds, batch_y)
            
            total_val_sse += loss.item()
            total_val_preds += batch_y.numel()
            
    val_mse = total_val_sse / total_val_preds
    print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
    
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        patience_counter = 0
        torch.save(model.state_dict(), best_gru_path)
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"\nValidation loss plateaued. Early stopping triggered at epoch {epoch+1}!")
        break

print("\nLoading best GRU weights for Test Set evaluation...")
model.load_state_dict(torch.load(best_gru_path))
model.eval()

test_X_tensor = torch.FloatTensor(X_test).to(device)
test_U_tensor = torch.FloatTensor(U_test).to(device)

all_preds = []
all_targets = []

with torch.no_grad():
    for t_start in tqdm(test_starts, desc="Eval Steps"):
        t_end = t_start + history_len
        
        x_hist = test_X_tensor[t_start:t_end, :].T.unsqueeze(-1)
        u_hist = test_U_tensor[t_start:t_end, :].unsqueeze(0).expand(num_neurons, -1, -1)
        
        features = torch.cat([x_hist, u_hist], dim=-1)
        preds = model(features).squeeze(-1)
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(X_test[t_end, :])

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

gru_mse = np.mean((all_preds - all_targets) ** 2)

corrs = []
for i in range(num_neurons):
    pred_i = all_preds[:, i]
    target_i = all_targets[:, i]
    if np.std(pred_i) == 0 or np.std(target_i) == 0:
        continue
    r, _ = pearsonr(pred_i, target_i)
    if np.isfinite(r):
        corrs.append(r)

print("\n=== GRU BASELINE (TEST SET) ===")
print(f"MSE: {gru_mse:.6f} | Pearson r: {np.mean(corrs):.4f}")

with open(summary_file, "a", encoding="utf-8") as f:
    f.write("=== INDEPENDENT GRU BASELINE ===\n")
    f.write(f"MSE: {gru_mse:.6f} | Pearson r: {np.mean(corrs):.4f}\n\n")

print(f"\nExecution complete. Model and metrics saved to: {results_dir}")