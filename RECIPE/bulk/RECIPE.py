import os, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import sparse as sp
from sklearn.metrics import r2_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops


# --------------------------
# Utils
# --------------------------
def set_seed(seed: int):
    print(f"[Seed] {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cpm_log2p1(arr: np.ndarray) -> np.ndarray:
    med = np.median(arr)
    return np.log2(arr / med + 1.0)


def make_splits(n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


# --------------------------
# Model
# --------------------------
class NeuralGraph(nn.Module):
    def __init__(self, seq_dim=9216, hidden=32, dropout=0.2, aggr='sum'):
        super().__init__()
        self.fc_x = nn.Sequential(
            nn.Linear(1, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fc_paired = nn.Sequential(
            nn.Linear(1, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.encoder = nn.Sequential(
            nn.Linear(seq_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.conv = SAGEConv(hidden, hidden, aggr=aggr)
        self.conv_activation = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.regressor = nn.Linear(hidden, 1)
        self.regressor_activation = nn.ReLU()

    def forward(self, data: Data):
        x_num = self.fc_x(data.x)                        # (N, hidden)
        x_seq = self.encoder(data.seq)                   # (N, hidden)
        x = x_num + x_seq
        x = torch.cat([x, self.fc_paired(data.pause)], dim=1)  # (N, 2*hidden)
        x = self.fc(x)
        z = self.conv(x, data.edge_index)
        z = self.conv_activation(z)
        out = self.regressor_activation(self.regressor(z))
        return out, z


# --------------------------
# Training loop
# --------------------------
def train_one(args):
    set_seed(args.seed)

    # ---- Load data ----
    # sequence embeddings
    seq = np.load(args.seq_npy).astype(np.float32)   # shape (N, seq_dim)
    n_nodes, seq_dim = seq.shape
    if args.seq_dim is None:
        args.seq_dim = seq_dim
    else:
        assert args.seq_dim == seq_dim, f"--seq-dim({args.seq_dim}) != npy({seq_dim})"

    # meta (X / y / pause)
    df = pd.read_csv(args.meta_csv)
    # 可选：去掉小数点版本号
    if "protein_x" in df.columns:
        df['protein'] = df['protein_x'].astype(str).str.split('.').str[0]

    # X/y 列
    if args.norm == "cpm_log2p1":
        X = cpm_log2p1(df[[args.x_col]].values.astype(np.float32))
        y = cpm_log2p1(df[[args.y_col]].values.astype(np.float32))
    elif args.norm == "none":
        X = df[[args.x_col]].values.astype(np.float32)
        y = df[[args.y_col]].values.astype(np.float32)
    else:
        raise ValueError("--norm must be one of ['none','cpm_log2p1'].")

    # pause 列（缺失时填 0）
    if args.pause_col in df.columns:
        pause = df[[args.pause_col]].values.astype(np.float32)
        # 若有缺失，填 0
        pause = np.nan_to_num(pause, nan=0.0)
    else:
        print(f"[Warn] pause column '{args.pause_col}' not found in meta. Use zeros.")
        pause = np.zeros_like(X, dtype=np.float32)

    # PPI
    ppi_dense = pd.read_csv(args.ppi_csv)
    ppi = sp.coo_matrix(ppi_dense.values.astype(np.float32))
    edge_index, edge_weight = from_scipy_sparse_matrix(ppi)
    if args.add_self_loops:
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)

    # ---- tensors & device ----
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    X_t = torch.from_numpy(X).view(-1, 1).to(device)
    y_t = torch.from_numpy(y).view(-1, 1).to(device)
    pause_t = torch.from_numpy(pause).view(-1, 1).to(device)
    seq_t = torch.from_numpy(seq).to(device)
    data = Data(x=X_t, y=y_t, edge_index=edge_index.to(device), edge_attr=edge_weight.to(device))
    data.pause = pause_t
    data.seq = seq_t

    # ---- splits ----
    train_idx, val_idx, test_idx = make_splits(
        n_nodes, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    train_idx_t = torch.as_tensor(train_idx, device=device)
    val_idx_t = torch.as_tensor(val_idx, device=device)
    test_idx_t = torch.as_tensor(test_idx, device=device)

    # ---- model/opt ----
    model = NeuralGraph(seq_dim=args.seq_dim, hidden=args.hidden, dropout=args.dropout, aggr=args.aggr).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_r2 = -1e9
    best_snap = None
    patience_counter = 0

    # ---- train ----
    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        opt.zero_grad()
        out, _ = model(data)
        train_loss = criterion(out[train_idx_t], data.y[train_idx_t]).mean()
        train_loss.backward()
        opt.step()
        with torch.no_grad():
            train_r2 = r2_score(data.y[train_idx_t].cpu().numpy(), out[train_idx_t].cpu().numpy())

        # eval
        model.eval()
        with torch.no_grad():
            out_eval, _ = model(data)
            val_loss = criterion(out_eval[val_idx_t], data.y[val_idx_t]).mean()
            val_r2 = r2_score(data.y[val_idx_t].cpu().numpy(), out_eval[val_idx_t].cpu().numpy())
            test_loss = criterion(out_eval[test_idx_t], data.y[test_idx_t]).mean()
            test_r2 = r2_score(data.y[test_idx_t].cpu().numpy(), out_eval[test_idx_t].cpu().numpy())

        # early stop on val R2
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best = {
                "epoch": epoch,
                "train_loss": float(train_loss.detach().cpu().item()),
                "val_loss": float(val_loss.detach().cpu().item()),
                "test_loss": float(test_loss.detach().cpu().item()),
                "train_r2": float(train_r2),
                "val_r2": float(val_r2),
                "test_r2": float(test_r2),
            }
            best_snap = {k: v for k, v in best.items()}
            patience_counter = 0
            # save
            if args.save_path:
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                torch.save(model.state_dict(), args.save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # print progress（按你习惯的格式）
        print(
            f"Epoch: {epoch:04d}, "
            f"Train Loss: {train_loss:.3f}, Train R²: {train_r2:.3f}| "
            f"Val Loss: {val_loss:.3f}, Val R²: {val_r2:.3f}|"
            f"Test Loss: {test_loss:.3f}, Test R²: {test_r2:.3f}"
        )

    # ---- summary ----
    if best_snap is not None:
        print("\n[Best @ Epoch {epoch}] Train Loss: {train_loss:.3f}, Train R²: {train_r2:.3f}| "
              "Val Loss: {val_loss:.3f}, Val R²: {val_r2:.3f}| "
              "Test Loss: {test_loss:.3f}, Test R²: {test_r2:.3f}".format(**best_snap))
    else:
        print("\n[Warn] No improvement observed.")

    return best_snap


# --------------------------
# CLI
# --------------------------
def build_parser():
    p = argparse.ArgumentParser(description="GraphSAGE protein prediction (CLI)")

    # paths
    p.add_argument("--seq-npy", type=str, required=True, help="Path to sequence embedding .npy")
    p.add_argument("--ppi-csv", type=str, required=True, help="Path to PPI adjacency CSV (NxN)")
    p.add_argument("--meta-csv", type=str, required=True, help="Metadata CSV containing x/y columns")
    p.add_argument("--save-path", type=str, default="./model/best_model.pth", help="Where to save best model")

    # columns & preprocessing
    p.add_argument("--x-col", type=str, default="rNC2")
    p.add_argument("--y-col", type=str, default="NC3")
    p.add_argument("--pause-col", type=str, default="High_Pause_Countsnc",
                   help="Pause column name in meta CSV; if missing, zeros used")
    p.add_argument("--norm", type=str, default="cpm_log2p1", choices=["none", "cpm_log2p1"])

    # model
    p.add_argument("--seq-dim", type=int, default=None, help="Sequence embedding dim; auto from npy if None")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--aggr", type=str, default="sum", choices=["sum", "mean", "max"])
    p.add_argument("--add-self-loops", action="store_true")

    # train
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.07)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=8)

    # split
    p.add_argument("--train-ratio", type=float, default=0.75)
    p.add_argument("--val-ratio", type=float, default=1/6)   # 0.1667
    p.add_argument("--test-ratio", type=float, default=1/12) # 0.0833

    return p


def main():
    args = build_parser().parse_args()
    train_one(args)


if __name__ == "__main__":
    main()
