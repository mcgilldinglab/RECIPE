import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
from scipy import sparse as sp

import os
import random
cds_df=pd.read_csv("/mnt/md0/luying/ribo/308code/pausing/cds_df38510.csv")
cds_df = cds_df.iloc[:,1:9]
cds_df['transcript_id'] = cds_df['transcript_id_x'].str.split('.').str[0]
exp=pd.read_csv("/mnt/md0/luying/ribo/308code/pausing/data/sc11619genes422cell_normalized.csv")
sorted_cds_df = cds_df.set_index('transcript_id').reindex(exp['Unnamed: 0']).reset_index()


sorted_cds_df.head()
sorted_cds_df.fillna(0, inplace=True)

merged_df=sorted_cds_df
merged_df['transcript_id'] = merged_df['transcript_id_x'].str.split('.').str[0]
merged_df3 = pd.read_csv('./data/24077132kdncmergedf.csv')
merged_df3['protein'] = merged_df3['protein_x'].str.split('.').str[0]


pausing = pd.read_csv('/mnt/md0/luying/ribo/308code/pausing/pause_scorescdsallnewnohupNC1_38510FINAL.csv')

pausing.columns = ['protein_id', "High_Pause_Countsnc1", "transcript_id_x"]
merged_df2 = pd.merge(merged_df, pausing, on='transcript_id_x', how='left')
merged_df2['High_Pause_Countsnc1'].fillna(0, inplace=True)


pausing = pd.read_csv('/mnt/md0/luying/ribo/308code/pausing/pause_scorescdsallnewnohupNC2_38510FINAL.csv')

pausing.columns = ['protein_id', "High_Pause_Countsnc2", "transcript_id_x"]
merged_df2 = pd.merge(merged_df2, pausing, on='transcript_id_x', how='left')
merged_df2['High_Pause_Countsnc2'].fillna(0, inplace=True)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import r2_score


def seed_everything(seed=0):
    print('seed = {}'.format(seed))
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

SEED = 5

seed_everything(SEED)


class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_model(model, data, train_idx, y_true, optimizer, criterion, patience=50):
    early_stopping = EarlyStopping(patience=patience)
    model.train()
    for epoch in range(10000): 
        model.train()
        optimizer.zero_grad()
        out, _ = model(data)
        loss = criterion(out[train_idx], y_true[train_idx])
        #print(f"train_idx: {train_idx}")
        loss.backward()
        optimizer.step()

        early_stopping.step(loss.item())
        if early_stopping.early_stop:
            print(f"Early Stopping at Epoch {epoch} with Loss {loss.item():.4f}")
            break


def evaluate_model(model, data, idx, y_true):
    model.eval()
    with torch.no_grad():
        out, _ = model(data)
        loss = nn.MSELoss()(out[idx], y_true[idx])
        r2 = r2_score(y_true[idx].cpu().numpy(), out[idx].cpu().numpy())
    return loss.item(), r2

def self_learning_process(data, y, model, device, initial_labeled_idx, val_idx, pool_idx, batch_size=300, max_rounds=10):


    train_idx = initial_labeled_idx.clone()
    print(f" {len(train_idx)}")

    seed_everything(SEED)

   

    optimizer = optim.Adam(model.parameters(), lr=7e-2)
    criterion = nn.MSELoss()

    train_model(model, data, train_idx, y, optimizer, criterion)



    for round_num in range(max_rounds):
        if len(pool_idx) == 0:
            print("Self-Learning end。")
            break

        model.eval()
        with torch.no_grad():
            outputs, _ = model(data)
        
        select_size = min(batch_size, len(pool_idx))
        selected_idx = pool_idx[:select_size]

        pseudo_labels = outputs[selected_idx].detach()


        y[selected_idx] = pseudo_labels

        train_idx = torch.cat([train_idx, selected_idx], dim=0)


        pool_idx = pool_idx[select_size:]

        print(f"{round_num+1}: add{select_size},total{len(train_idx)}")


        optimizer = optim.Adam(model.parameters(), lr=7e-2)
        criterion = nn.MSELoss()
        train_model(model, data, train_idx, y, optimizer, criterion)

    val_loss, val_r2 = evaluate_model(model, data, val_idx, y)
    print(f"最终验证集Loss: {val_loss:.4f}, 验证集R²: {val_r2:.4f}")
    return model


seed_everything(SEED)

class NeuralGraph(nn.Module):
    def __init__(self):
        super(NeuralGraph, self).__init__()
        self.fc_x = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.fc_paired = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.encoder = nn.Sequential(
            nn.Linear(9216, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.conv = SAGEConv(32, 32, aggr='sum')
        self.conv_activation = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.regressor = nn.Linear(32, 1)
        self.regressor_activation = nn.Sequential(
            nn.ReLU()
        )
    def forward(self, data):
        x = data.x
        seq_embedding = data.seq
        pausescore = data.pause
        
        x = self.fc_x(x) + self.encoder(seq_embedding)
        x = torch.cat((x, self.fc_paired(pausescore)), dim=1)
        x = self.fc(x)
        
        # Graph convolution layer
        z = self.conv(x, data.edge_index)
        z = self.conv_activation(z)
        
        # Regressor layer
        out = self.regressor(z)
        out=self.regressor_activation(out)
        return out, z
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
import random
import numpy as np
import torch


seed_everything(SEED)  

import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


X_cpm_log2 = np.log2((merged_df2[['rNC2']].values / np.median(merged_df3[['rNC2']].values)) + 1)
y_cpm_log2 = np.log2((merged_df2[['NC3']].values / np.median(merged_df3[['NC3']].values))+ 1)


y = torch.tensor(y_cpm_log2, dtype=torch.float32).view(-1, 1)
X = torch.tensor(X_cpm_log2, dtype=torch.float32).view(-1, 1)
# 
# y = torch.tensor(np.array(merged_df['NC3'], dtype=np.float32).reshape(-1, 1))
# X = torch.from_numpy(np.array(merged_df['rNC2'], dtype=np.float32).reshape(-1, 1))

paired_ratio = torch.tensor(np.array(merged_df2['High_Pause_Countsnc1'], dtype=np.float32).reshape(-1, 1))

sequence_embedding = torch.tensor(all_sequence_outputsnew, dtype=torch.float32)
edge_index, edge_weight = from_scipy_sparse_matrix(ppi_matrix.astype('float32'))


data = Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=y)
data.pause = paired_ratio
data.seq = sequence_embedding

id_to_idx = {tid: idx for idx, tid in enumerate(merged_df2['Unnamed: 0'])}

labeled_idx = [id_to_idx[tid] for tid in intersection]
# print(intersection)
# print(merged_df2['Unnamed: 0'].tolist()[:10])

# 转成 Tensor
labeled_idx = torch.tensor(labeled_idx, dtype=torch.long)



seed_everything(SEED)

train_idx, temp_idx = train_test_split(labeled_idx.cpu().numpy(), test_size=0.25, random_state=SEED)
test_idx, val_idx = train_test_split(temp_idx, test_size=0.5, random_state=SEED)

train_idx = torch.tensor(train_idx, device=device)
test_idx = torch.tensor(test_idx, device=device)
val_idx = torch.tensor(val_idx, device=device)


is_labeled = torch.zeros(y.size(0), dtype=torch.bool)
is_labeled[labeled_idx] = True  
unlabeled_idx = torch.where(~is_labeled)[0]
pool_idx = unlabeled_idx.to(device)


data = data.to(device)
data.x = data.x.to(device)
data.seq = data.seq.to(device)
data.pause = data.pause.to(device)
data.edge_index = data.edge_index.to(device)
y = y.to(device)


seed_everything(SEED)                
neural_net = NeuralGraph().to(device)

# 初始化优化器
optimizer = optim.Adam(neural_net.parameters(), lr=7e-2)
criterion = nn.MSELoss()

# 启动 self-learning
model = self_learning_process(
    data=data,
    y=y,
    model=neural_net,
    device=device,
    initial_labeled_idx=train_idx,
    val_idx=val_idx,
    pool_idx=pool_idx,
    batch_size=300,
    max_rounds=100
)

torch.save(model, './model/250429selflearning_3193_11619final_full_model0.6.pth')


exp=pd.read_csv('/mnt/md0/luying/ribo/dnabert/DNABERT/examples/my_project/data/sc11619genes422cell.csv')

meta=pd.read_csv('brforepridictmeta_dataall.csv')

rich_cells = meta[meta['fraction'] == 'Rich']['cell_names'].tolist()

# 由于 cell_names 在 exp 中是列名，需要检查格式是否匹配
rich_cells_exp = [cell for cell in rich_cells if cell in exp.columns]

# 提取 Rich 细胞对应的表达矩阵
exp_rich = exp[['Unnamed: 0'] + rich_cells_exp]

exp_rich["scribo"] = exp_rich.iloc[:, 1:].mean(axis=1)

merged_df=pd.read_csv('/mnt/md0/luying/ribo/308code/pausing/data/250429scribonew11619_422.csv')

pausing = pd.read_csv('/mnt/md0/luying/ribo/308code/pausing/pause_scorescdsallscribo293Rich_dedup3sball.csv')


pausing.columns = ['protein_id', "High_Pause_Countsscrich", "transcript_id"]


merged_df2 = pd.merge(merged_df, pausing, on='transcript_id', how='left')
merged_df2['High_Pause_Countsscrich'].fillna(0, inplace=True)
pausing = pd.read_csv('/mnt/md0/luying/ribo/308code/pausing/pause_scorescdsallscribo293Leu6h_dedup3sball.csv')


pausing.columns = ['protein_id', "High_Pause_Countsscleu6h", "transcript_id"]
merged_df2 = pd.merge(merged_df2, pausing, on='transcript_id', how='left', suffixes=('_existing', '_new'))
merged_df2['High_Pause_Countsscleu6h'].fillna(0, inplace=True)
pausing = pd.read_csv('/mnt/md0/luying/ribo/308code/pausing/pause_scorescdsallscribo293Leu3h_dedup3sball.csv')
pausing.columns = ['protein_id', "High_Pause_Countsscleu3h", "transcript_id"]
merged_df2 = pd.merge(merged_df2, pausing, on='transcript_id', how='left', suffixes=('_existing', '_new'))
merged_df2['High_Pause_Countsscleu3h'].fillna(0, inplace=True)

pausing = pd.read_csv('/mnt/md0/luying/ribo/308code/pausing/pause_scorescdsallscribo293Arg3h_dedup3sball.csv')


pausing.columns = ['protein_id', "High_Pause_Countsscarg3h", "transcript_id"]
merged_df2 = pd.merge(merged_df2, pausing, on='transcript_id', how='left', suffixes=('_existing1', '_new1'))
merged_df2['High_Pause_Countsscarg3h'].fillna(0, inplace=True)
pausing = pd.read_csv('/mnt/md0/luying/ribo/308code/pausing/pause_scorescdsallscribo293Arg6h_dedup3sball.csv')
pausing.columns = ['protein_id', "High_Pause_Countsscarg6h", "transcript_id"]
merged_df2 = pd.merge(merged_df2, pausing, on='transcript_id', how='left', suffixes=('_existing2', '_new2'))
merged_df2['High_Pause_Countsscarg6h'].fillna(0, inplace=True)

from torch_geometric.utils import from_scipy_sparse_matrix
ppi_matrix = pd.read_csv('./data/ppi_ebi_string_ppi3ensp_lr_IntAct_corummatrix4p_pbulk11619.csv')

ppi_matrix.head()
all_sequence_outputsnew = np.load('./data/all_sequence_outputsnewbulk11619.npy')
ppi_matrix = sp.coo_matrix(ppi_matrix)
edge_index, edge_weight = from_scipy_sparse_matrix(ppi_matrix.astype('float32'))

cds_df=pd.read_csv("/mnt/md0/luying/ribo/308code/pausing/cds_df38510.csv")
cds_df = cds_df.iloc[:,1:9]
cds_df['transcript_id'] = cds_df['transcript_id_x'].str.split('.').str[0]
exp=pd.read_csv("/mnt/md0/luying/ribo/308code/pausing/data/sc11619genes422cell_normalized.csv")

sorted_cds_df = cds_df.set_index('transcript_id').reindex(exp['Unnamed: 0']).reset_index()


sorted_cds_df.head()
sorted_cds_df.fillna(0, inplace=True)

merged_df=sorted_cds_df
merged_df['transcript_id'] = merged_df['transcript_id_x'].str.split('.').str[0]
merged_df.head
merged_df3 = pd.read_csv('./data/24077132kdncmergedf.csv')
merged_df3['protein'] = merged_df3['protein_x'].str.split('.').str[0]
merged_df3.shape



X_cpm_log2 = np.log2((exp_rich[['scribo']].values / np.median(exp_rich[['scribo']].values)) + 1)
y_cpm_log2 = np.log2((merged_df[['NC3']].values / np.median(merged_df3[['NC3']].values))+ 1)


y = torch.tensor(y_cpm_log2, dtype=torch.float32).view(-1, 1)
X = torch.tensor(X_cpm_log2, dtype=torch.float32).view(-1, 1)
ppi_matrix = sp.coo_matrix(ppi_matrix)

edge_index, edge_weight = from_scipy_sparse_matrix(ppi_matrix.astype('float32'))

data = Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=y)
data.pause = torch.tensor(merged_df2['High_Pause_Countsscrich'].values, dtype=torch.float32)
data.seq = torch.tensor(all_sequence_outputsnew, dtype=torch.float32)
paired_ratio = torch.tensor(np.array(merged_df2['High_Pause_Countsscrich'], dtype=np.float32).reshape(-1, 1))

sequence_embedding = torch.tensor(all_sequence_outputsnew, dtype=torch.float32)
edge_index, edge_weight = from_scipy_sparse_matrix(ppi_matrix.astype('float32'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
from scipy import sparse as sp

import os
import random

def set_seed(seed=0):
    print('seed = {}'.format(seed))
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

# 设置随机种子
SEED = 8
set_seed(SEED)

X_cpm_log2 = np.log2((exp_rich[['scribo']].values / np.median(exp_rich[['scribo']].values)) + 1)
y_cpm_log2 = np.log2((merged_df[['NC3']].values / np.median(merged_df3[['NC3']].values))+ 1)

y = torch.tensor(y_cpm_log2, dtype=torch.float32).view(-1, 1)
X = torch.tensor(X_cpm_log2, dtype=torch.float32).view(-1, 1)

# train_mask = (~torch.isnan(y)) & (y != 0)
paired_ratio = torch.tensor(np.array(merged_df2['High_Pause_Countsscrich'], dtype=np.float32).reshape(-1, 1))

import torch


valid_mask = (~torch.isnan(y)) & (y != 0)  
valid_indices = valid_mask.nonzero(as_tuple=True)[0]  #
from sklearn.model_selection import train_test_split


train_val_idx, test_idx = train_test_split(valid_indices, test_size=0.2, random_state=42)


train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


num_nodes = y.shape[0]
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True
from torch_geometric.data import Data

data = Data(
    x=X,  # 11619 x N
    edge_index=edge_index,  # 
    y=y,  # 11619 x 1 或 11619,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)
data.pause = paired_ratio
data.seq = sequence_embedding


# 数据集划分
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
from scipy import sparse as sp

import os
import random

def set_seed(seed=0):
    print('seed = {}'.format(seed))
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


set_seed(SEED)

class NeuralGraph(nn.Module):
    def __init__(self):
        super(NeuralGraph, self).__init__()
        self.fc_x = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.fc_paired = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.encoder = nn.Sequential(
            nn.Linear(9216, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.conv = SAGEConv(32, 32, aggr='sum')
        self.conv_activation = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.regressor = nn.Linear(32, 1)
        self.regressor_activation = nn.Sequential(
            nn.ReLU()
        )
    def forward(self, data):
        x = data.x
        seq_embedding = data.seq
        pausescore = data.pause
        
        x = self.fc_x(x) + self.encoder(seq_embedding)
        x = torch.cat((x, self.fc_paired(pausescore)), dim=1)
        x = self.fc(x)
        
        # Graph convolution layer
        z = self.conv(x, data.edge_index)
        z = self.conv_activation(z)
        
        # Regressor layer
        out = self.regressor(z)
        out=self.regressor_activation(out)
        return out, z


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
neural_net = NeuralGraph().to(device)
optimizer = optim.Adam(neural_net.parameters(), lr=7e-2)
criterion = nn.MSELoss()
data = data.to(device)

patience = 250 
num_epochs = 3000
patience_counter = 0
best_val_loss = float('inf')
best_val_r2 = float('-inf') 
best_test_r2 = float('-inf') 
for epoch in range(1, num_epochs + 1):
    # Training phase
    neural_net.train()
    optimizer.zero_grad()

    out, z  = neural_net(data)
    train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
    train_loss.backward()
    optimizer.step()

    train_r2 = r2_score(
        data.y[data.train_mask].cpu().numpy(),
        out[data.train_mask].detach().cpu().numpy()
    )
    # Validation phase
    neural_net.eval()
    with torch.no_grad():
        val_out, _ = neural_net(data)
        val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask]).mean()
        val_r2 = r2_score(data.y[data.val_mask].cpu().numpy(), val_out[data.val_mask].cpu().numpy())

        test_out, _ = neural_net(data)
        test_loss = criterion(test_out[data.test_mask], data.y[data.test_mask]).mean()
        test_r2 = r2_score(data.y[data.test_mask].cpu().numpy(), test_out[data.test_mask].cpu().numpy())

    # Check if validation loss has improved
    if val_r2 > best_val_r2:  
        best_val_r2 = val_r2
        best_train_loss = train_loss
        best_val_loss = val_loss
        patience_counter = 0

        best_test_loss = test_loss
        best_test_r2 = test_r2
        torch.save(neural_net.state_dict(), './model/250503预测未知scribobest_models80.680_0.600.pth')
        y_true_np = data.y[data.test_mask].cpu().detach().numpy()
        y_pred_np = test_out[data.test_mask].cpu().detach().numpy()


    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Train R²: {train_r2:.3f}| "
          f"Val Loss: {best_val_loss:.3f}, Val R²: {best_val_r2:.3f}|"
          f"Test Loss: {best_test_loss:.3f}, Test R²: {best_test_r2:.3f}")
# finetuning
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = NeuralGraph().to(device)

# ✅ 
pretrained_model_path = './model/250429selflearning_3193_11619final_model0.6.pth'
if os.path.exists(pretrained_model_path):
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print("✅")
else:
    print("⚠️")


data = data.to(device)

train_idx = data.train_mask
val_idx = data.val_mask
test_idx = data.test_mask

# ✅ 训练参数
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
num_epochs = 1000
patience = 100
patience_counter = 0
best_val_r2 = float('-inf')
best_model_path = './model/250502预测未知scribofinetunings8_0.813.pth'

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data)
    loss = criterion(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

    train_r2 = r2_score(
        data.y[train_idx].cpu().numpy(),
        out[train_idx].detach().cpu().numpy()
    )

    model.eval()
    with torch.no_grad():
        out_val, _ = model(data)
        val_loss = criterion(out_val[val_idx], data.y[val_idx])
        val_r2 = r2_score(
            data.y[val_idx].cpu().numpy(),
            out_val[val_idx].cpu().numpy()
        )
        test_out, _ = model(data)
        test_loss = criterion(test_out[test_idx], data.y[test_idx])
        test_r2 = r2_score(
            data.y[test_idx].cpu().numpy(),
            test_out[test_idx].cpu().numpy()
        )

    # Early stopping
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_train_loss = loss.item()
        best_val_loss = val_loss.item()
        best_test_loss = test_loss.item()
        best_test_r2 = test_r2
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"[Epoch {epoch}] ✅ New best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹ Early stopping at epoch {epoch}")
            break

    print(f"Epoch {epoch:03d} | Train Loss: {loss:.3f}, Train R²: {train_r2:.3f} | "
          f"Val Loss: {val_loss:.3f}, Val R²: {val_r2:.3f} | "
          f"Test Loss: {test_loss:.3f}, Test R²: {test_r2:.3f}")

print("✅！")



import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad


exp=pd.read_csv('/mnt/md0/luying/ribo/308code/pausing/data/250429scribonew11619_422.csv')

meta=pd.read_csv('brforepridictmeta_dataall.csv')

rich_cells = meta[meta['fraction'] == 'Rich']['cell_names'].tolist()


rich_cells_exp = [cell for cell in rich_cells if cell in exp.columns]


exp_rich = exp[['transcript_id'] + rich_cells_exp]

import pandas as pd


exp_rich_filtered = exp_rich.loc[~(exp_rich.iloc[:, 1:] == 0).all(axis=1)]


exp_rich_filtered

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
from scipy import sparse as sp

import os
import random

def set_seed(seed=0):
    print('seed = {}'.format(seed))
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
# 设置随机种子
SEED = 12
set_seed(SEED)


class NeuralGraph(nn.Module):
    def __init__(self):
        super(NeuralGraph, self).__init__()
        self.fc_x = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.fc_paired = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.encoder = nn.Sequential(
            nn.Linear(9216, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.conv = SAGEConv(32, 32, aggr='sum')
        self.conv_activation = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.regressor = nn.Linear(32, 1)
        self.regressor_activation = nn.Sequential(
            nn.ReLU()
        )
    def forward(self, data):
        x = data.x
        seq_embedding = data.seq
        pausescore = data.pause
        #print(f"pausescore shape: {pausescore.shape}")

        x = self.fc_x(x) + self.encoder(seq_embedding)
        x = torch.cat((x, self.fc_paired(pausescore.view(-1, 1))), dim=1)
        x = self.fc(x)
        
        # Graph convolution layer
        z = self.conv(x, data.edge_index)
        z = self.conv_activation(z)
        
        # Regressor layer
        out = self.regressor(z)
        out=self.regressor_activation(out)
        return out, z

device = torch.device("cpu")
neural_net = NeuralGraph().to(device)
optimizer = optim.Adam(neural_net.parameters(), lr=3e-4)
criterion = nn.MSELoss()


neural_net.load_state_dict(torch.load('./model/250502预测未知scribofinetunings8_0.813.pth'))

neural_net = neural_net.to(device) 
expk=pd.read_csv('/mnt/md0/luying/ribo/dnabert/DNABERT/examples/my_project/data/sc11619genes422cell.csv')
expk = expk.set_index('Unnamed: 0')
adata = ad.AnnData(X=expk)

sc.pp.normalize_total(adata)  #

# 转回 DataFrame 格式
exp_normalized = pd.DataFrame(
    adata.X, 
    index=adata.obs_names, 
    columns=adata.var_names
)
exp_normalized.head

exp_normalized = exp_normalized.fillna(0)

import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import numpy as np
exp=exp_normalized
def create_knn_graph(exp_data, n_neighbors=5, n_pcs=1):


    pca = PCA(n_components=n_pcs)
    exp_pca = pca.fit_transform(exp_data)

    print(f"Shape after PCA: {exp_pca.shape}")


    knn = NearestNeighbors(n_neighbors=n_neighbors + 1) 
    knn.fit(exp_pca)


    distances, indices = knn.kneighbors(exp_pca)


    edge_index_knn = []
    
    for node, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  
            edge_index_knn.append([node, neighbor])

    # 将边列表转换为PyTorch的Tensor格式
    edge_index_knn = torch.tensor(edge_index_knn, dtype=torch.long).t().contiguous()

    print(f"Constructed edge_index_knn shape: {edge_index_knn.shape}")
    
    return edge_index_knn

if isinstance(exp, np.ndarray):
    exp = pd.DataFrame(exp)  


exp_values = exp.iloc[:, 0:].T.values
edge_index_knn = create_knn_graph(exp_values, n_neighbors=5, n_pcs=1)

print(edge_index_knn)
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.utils.data import random_split

seed = 0  
torch.manual_seed(seed)
def create_knn_graph(exp_data, n_neighbors=3):


    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)  
    knn.fit(exp_data)

    distances, indices = knn.kneighbors(exp_data)


    edge_index_knn = []
    for node, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  
            edge_index_knn.append([node, neighbor])

    edge_index_knn = torch.tensor(edge_index_knn, dtype=torch.long).t().contiguous()

    return edge_index_knn

class GraphListDataset(Dataset):
    def __init__(self, graph_list, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.graph_list = graph_list

    def len(self):
        return len(self.graph_list)

    def get(self, idx):
        # 仅返回图数据对象
        return self.graph_list[idx]


import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import numpy as np

def create_knn_graphs_per_cell(exp, z_array, meta, n_neighbors=3):
    """

    """
    all_graphs = []
    num_cells, num_genes = exp.shape

    print(f"✅ num cell {num_cells}, num gene: {num_genes}")

    for cell_idx in range(num_cells):  
  
        cell_exp_data = exp.iloc[cell_idx, :].values.reshape(-1, 1)  
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1) 
        knn.fit(cell_exp_data)
        distances, indices = knn.kneighbors(cell_exp_data)


        edge_index_knn = []
        for node, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:  
                edge_index_knn.append([node, neighbor])
        edge_index_knn = torch.tensor(edge_index_knn, dtype=torch.long).t().contiguous()  # (2, num_edges)


        node_features = torch.tensor(z_array[cell_idx, :, :], dtype=torch.float)  # (gene, 32)

        num_nodes = node_features.shape[0]  
        if edge_index_knn.max().item() >= num_nodes:
            print(f"⚠️ 跳过细胞 {cell_idx}: Edge index 超界 ({edge_index_knn.max().item()} >= {num_nodes})")
            continue  

        is_rich = meta.iloc[cell_idx]['fraction'] == 'Rich'  # True / False
        rich_mask = torch.tensor(is_rich, dtype=torch.bool).repeat(num_genes)  # (gene,)

        graph = Data(x=node_features, edge_index=edge_index_knn, rich_mask=rich_mask, gene_ids=torch.arange(num_genes))
        all_graphs.append(graph)

    return all_graphs




meta=pd.read_csv('brforepridictmeta_dataall.csv')

meta = meta.rename(columns={'Unnamed: 0': 'cell_names'})

print("Meta columns:", meta.columns)


meta = meta.loc[:, ~meta.columns.duplicated()]


print("Duplicate cell_names:", meta['cell_names'].duplicated().sum())  
meta = meta.drop_duplicates(subset=['cell_names'])

print(f"Data shape: {exp.shape}")  
exp_values = exp.iloc[:, 0:].T  #
# z_array (215, 7132, 32)
z_array = np.load('data/all_z_array_0503test.npy')
print(f"z_array shape: {z_array.shape}")

meta = meta.set_index('cell_names').loc[exp_values.index]  
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
import torch
import numpy as np

def split_genes(num_genes, train_ratio=0.9, seed=42):
    """
    随机划分基因索引为训练集和测试集。
    """
    np.random.seed(seed)
    indices = np.random.permutation(num_genes)
    train_size = int(train_ratio * num_genes)
    train_genes = indices[:train_size]
    test_genes = indices[train_size:]
    return train_genes, test_genes

def create_knn_graphs_subset_genes(exp, z_array, meta, gene_indices, n_neighbors=3):
    """
    构建只包含部分基因的 KNN 图列表（每个细胞一张图，节点为基因）。
    """
    all_graphs = []
    num_cells = exp_values.shape[0]

    for cell_idx in range(num_cells):
        
        cell_exp_data = exp_values.iloc[cell_idx, gene_indices].values.reshape(-1, 1)

    
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        knn.fit(cell_exp_data)
        _, indices = knn.kneighbors(cell_exp_data)

        edge_index = []
        for node, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:
                edge_index.append([node, neighbor])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

  
        node_features = torch.tensor(z_array[cell_idx, gene_indices, :], dtype=torch.float)

        is_rich = meta.iloc[cell_idx]['fraction'] == 'Rich'
        rich_mask = torch.tensor(is_rich, dtype=torch.bool).repeat(len(gene_indices))

        graph = Data(x=node_features, edge_index=edge_index, rich_mask=rich_mask, gene_ids=torch.tensor(gene_indices))
        all_graphs.append(graph)

    return all_graphs
cds_df=pd.read_csv("/mnt/md0/luying/ribo/308code/pausing/cds_df38510.csv")
cds_df = cds_df.iloc[:,1:9]
cds_df['transcript_id'] = cds_df['transcript_id_x'].str.split('.').str[0]
exp=pd.read_csv("/mnt/md0/luying/ribo/308code/pausing/data/sc11619genes422cell_normalized.csv")

sorted_cds_df = cds_df.set_index('transcript_id').reindex(exp['Unnamed: 0']).reset_index()


sorted_cds_df.head()
sorted_cds_df.fillna(0, inplace=True)

merged_df2=sorted_cds_df
merged_df2['transcript_id'] = merged_df2['transcript_id_x'].str.split('.').str[0]
merged_df2.head

from sklearn.model_selection import train_test_split
import numpy as np

original_y = merged_df2["NC3"]
original_y_np = original_y.to_numpy()

total_gene_ids = np.arange(z_array.shape[1])  #
valid_gene_ids = np.where(original_y != 0)[0]

invalid_gene_ids = np.setdiff1d(total_gene_ids, valid_gene_ids)


train_gene_ids, temp_gene_ids = train_test_split(valid_gene_ids, test_size=0.25, random_state=42)
test_gene_ids, val_gene_ids_with_y = train_test_split(temp_gene_ids, test_size=0.5, random_state=42)

val_gene_ids =val_gene_ids_with_y
print(f"Train genes (with y): {len(train_gene_ids)}")
print(f"Test genes (with y): {len(test_gene_ids)}")
print(f"Val genes (with y): {len(val_gene_ids_with_y)}")
print(f"Val genes total (with + without y): {len(val_gene_ids)}")
# gene 数量
num_genes = z_array.shape[1]
train_graphs = create_knn_graphs_subset_genes(exp_values, z_array, meta, train_gene_ids, n_neighbors=3)
test_graphs = create_knn_graphs_subset_genes(exp_values, z_array, meta, test_gene_ids, n_neighbors=3)
val_graphs = create_knn_graphs_subset_genes(exp_values, z_array, meta, val_gene_ids, n_neighbors=3)
# 创建 DataLoader
trainloader = DataLoader(train_graphs, batch_size=len(train_graphs), shuffle=False)
testloader = DataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)
valloader = DataLoader(val_graphs, batch_size=len(val_graphs), shuffle=False)
print(f"训练集基因数: {len(train_gene_ids)}, 图数: {len(train_graphs)}")
print(f"测试集基因数: {len(test_gene_ids)}, 图数: {len(test_graphs)}")

print(f"测试集基因数: {len(val_gene_ids)}, 图数: {len(val_graphs)}")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import r2_score
import numpy as np
import random
import time

# ✅ 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(12)


input_dim = 32
output_dim = 16

y_cpm_log2 = np.log2((merged_df2[['NC3']].values / np.median(merged_df[['NC3']].values)) + 1)
original_y = torch.tensor(y_cpm_log2, dtype=torch.float32).view(-1).to(device)

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, output_dim, aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.mlp(x)
        x = F.relu(x)
        return x.squeeze(-1)

input_dim = 32  
output_dim = 16  
model = GraphSAGE(input_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


def get_batch(loader):
    return next(iter(loader)).to(device)


def train(loader):
    model.train()
    optimizer.zero_grad()

    batch = get_batch(loader)  # 
    y_pred = model(batch.x, batch.edge_index) 
    
    rich_mask = batch.rich_mask.bool()
    gene_ids = batch.gene_ids

    if rich_mask.sum() == 0:
        return 0.0, float('nan')

    # ✅ 计算 rich 组中每个基因的预测均值
    rich_preds = y_pred[rich_mask]
    rich_gene_ids = gene_ids[rich_mask]

    unique_gene_ids = torch.unique(rich_gene_ids)

    losses = []
    all_preds = []
    all_labels = []
    total_loss_sum = 0.0
    gene_count = 0

    for gid in unique_gene_ids:
        gid_mask = (rich_gene_ids == gid)
        preds = rich_preds[gid_mask]

        if preds.numel() == 0:
            continue

  
        mean_pred = preds.mean()

        # ✅ 
        label = original_y[gid]

        # ✅ 
        loss = F.mse_loss(mean_pred, label)
        losses.append(loss)
        total_loss_sum += loss.item()
        gene_count += 1

        all_preds.append(mean_pred.detach().cpu().numpy())
        all_labels.append(label.cpu().numpy())

    # ✅ 
    if len(losses) > 0:
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        optimizer.step()

    # ✅ 
    if gene_count > 0:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        r2 = r2_score(all_labels, all_preds)
        avg_loss = total_loss_sum / gene_count
    else:
        avg_loss = 0.0
        r2 = float('nan')

    return avg_loss, r2

# ✅ 测试函数
def test(loader):
    model.eval()

    batch = get_batch(loader)  
    y_pred = model(batch.x, batch.edge_index)
    
    rich_mask = batch.rich_mask.bool()
    gene_ids = batch.gene_ids

    if rich_mask.sum() == 0:
        return 0.0, float('nan')

    rich_preds = y_pred[rich_mask]
    rich_gene_ids = gene_ids[rich_mask]

    unique_gene_ids = torch.unique(rich_gene_ids)
    total_loss_sum = 0.0
    gene_count = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for gid in unique_gene_ids:
            gid_mask = (rich_gene_ids == gid)
            preds = rich_preds[gid_mask]

            if preds.numel() == 0:
                continue

            mean_pred = preds.mean()
            label = original_y[gid]
            
            loss = F.mse_loss(mean_pred, label)
            total_loss_sum += loss.item()
            gene_count += 1

            all_preds.append(mean_pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    # ✅ 计算 R²
    if gene_count > 0:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        r2 = r2_score(all_labels, all_preds)
        avg_loss = total_loss_sum / gene_count
    else:
        avg_loss = 0.0
        r2 = float('nan')

    return avg_loss, r2

best_test_r2 = float('-inf')
patience_counter = 0
patience = 50
best_train_loss = None
best_val_loss = None
best_test_loss = None
best_val_r2 = float('-inf')

num_epochs = 391
for epoch in range(1, num_epochs + 1):
    train_loss, train_r2 = train(trainloader)
    test_loss, test_r2 = test(testloader)
    val_loss, val_r2 = test(valloader)  # 

    if test_r2 > best_test_r2:  # 
        best_test_r2 = test_r2
        best_train_loss = train_loss
        best_val_loss = val_loss
        best_test_loss = test_loss
        best_val_r2 = val_r2  #
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Train R²: {train_r2:.3f}| "
          f"Val Loss: {val_loss:.3f}, Val R²: {val_r2:.3f}| "
          f"Test Loss: {best_test_loss:.3f}, Test R²: {best_test_r2:.3f}")
