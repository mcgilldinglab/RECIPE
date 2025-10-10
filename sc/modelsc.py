#12+12 Epoch: 3000, Train Loss: 0.130, Train R²: 0.866| Val Loss: 0.464, Val R²: 0.616|Test Loss: 0.482, Test R²: 0.515
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

# 设定随机种子
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

# 设置随机种子8：1:1
#SEED = 12#Train R²: 0.866| Val Loss: 0.438, Val R²: 0.675|Test Loss: 0.576, Test R²: 0.379
SEED = 5

seed_everything(SEED)

# 早停器
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

# 训练函数
def train_model(model, data, train_idx, y_true, optimizer, criterion, patience=50):
    early_stopping = EarlyStopping(patience=patience)
    model.train()
    for epoch in range(10000):  # 大循环，由early stopping控制
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

# 验证函数
def evaluate_model(model, data, idx, y_true):
    model.eval()
    with torch.no_grad():
        out, _ = model(data)
        loss = nn.MSELoss()(out[idx], y_true[idx])
        r2 = r2_score(y_true[idx].cpu().numpy(), out[idx].cpu().numpy())
    return loss.item(), r2

# 主Self-Learning流程
def self_learning_process(data, y, model, device, initial_labeled_idx, val_idx, pool_idx, batch_size=300, max_rounds=10):

    # 初始训练
    train_idx = initial_labeled_idx.clone()
    print(f"初始训练集大小: {len(train_idx)}")
    # 每次新模型初始化前，设置一次随机种子！
    seed_everything(SEED)

    # 定义优化器
   

    optimizer = optim.Adam(model.parameters(), lr=7e-2)
    criterion = nn.MSELoss()
    #optimizer = optim.Adam(train_model.parameters(), lr=7e-2)
    # 开始训练
    train_model(model, data, train_idx, y, optimizer, criterion)


    # 伪标签循环
    for round_num in range(max_rounds):
        if len(pool_idx) == 0:
            print("没有更多未标记样本，结束Self-Learning。")
            break

        model.eval()
        with torch.no_grad():
            outputs, _ = model(data)
        
        # 从pool里选前batch_size个
        select_size = min(batch_size, len(pool_idx))
        selected_idx = pool_idx[:select_size]
        #print(selected_idx)
        # 拿当前模型预测这些节点作为伪标签
        pseudo_labels = outputs[selected_idx].detach()

        # 更新y，把伪标签赋值
        y[selected_idx] = pseudo_labels

        # 将伪标签样本加入train_idx
        train_idx = torch.cat([train_idx, selected_idx], dim=0)

        # pool里移除这些样本
        pool_idx = pool_idx[select_size:]

        print(f"第{round_num+1}轮: 添加{select_size}个伪标签样本，总训练集大小{len(train_idx)}")

        # 重新训练
        optimizer = optim.Adam(model.parameters(), lr=7e-2)
        criterion = nn.MSELoss()
        train_model(model, data, train_idx, y, optimizer, criterion)

    # 最后在验证集上测试
    val_loss, val_r2 = evaluate_model(model, data, val_idx, y)
    print(f"最终验证集Loss: {val_loss:.4f}, 验证集R²: {val_r2:.4f}")
    return model

# 设置随机种子
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

#SEED = 42  # 你自己定义的种子数，比如42或者任何数

seed_everything(SEED)  
# # Python内置随机
# random.seed(SEED)
# # numpy随机
# np.random.seed(SEED)
# # torch随机
# torch.manual_seed(SEED)
# # 如果用的是GPU，也加上下面这行
# torch.cuda.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)  # 如果有多个GPU

# # 确保CUDA中的卷积算法确定性
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
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


# 只选有标签的数据
X_cpm_log2 = np.log2((merged_df2[['rNC2']].values / np.median(merged_df3[['rNC2']].values)) + 1)
y_cpm_log2 = np.log2((merged_df2[['NC3']].values / np.median(merged_df3[['NC3']].values))+ 1)

# 数据准备
y = torch.tensor(y_cpm_log2, dtype=torch.float32).view(-1, 1)
X = torch.tensor(X_cpm_log2, dtype=torch.float32).view(-1, 1)
# 
# y = torch.tensor(np.array(merged_df['NC3'], dtype=np.float32).reshape(-1, 1))
# X = torch.from_numpy(np.array(merged_df['rNC2'], dtype=np.float32).reshape(-1, 1))

paired_ratio = torch.tensor(np.array(merged_df2['High_Pause_Countsnc1'], dtype=np.float32).reshape(-1, 1))

sequence_embedding = torch.tensor(all_sequence_outputsnew, dtype=torch.float32)
edge_index, edge_weight = from_scipy_sparse_matrix(ppi_matrix.astype('float32'))

# 创建图数据对象
data = Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=y)
data.pause = paired_ratio
data.seq = sequence_embedding
# labeled_idx = torch.where(y.squeeze() != 0)[0]
# unlabeled_idx = torch.where(y.squeeze() == 0)[0]
# 假设merged_df2和merged_df3都有'transcript_id'这一列
# 建一个从transcript_id到索引的映射
# 先建一个 merged_df2['Unnamed: 0'] 到 index 的映射
id_to_idx = {tid: idx for idx, tid in enumerate(merged_df2['Unnamed: 0'])}

# 然后只查 intersection 中的
labeled_idx = [id_to_idx[tid] for tid in intersection]
# print(intersection)
# print(merged_df2['Unnamed: 0'].tolist()[:10])

# 转成 Tensor
labeled_idx = torch.tensor(labeled_idx, dtype=torch.long)

#print(f"最终labeled蛋白数量: {len(labeled_idx)}")  # 应该是4258


# 划分
# 划分

seed_everything(SEED)

train_idx, temp_idx = train_test_split(labeled_idx.cpu().numpy(), test_size=0.25, random_state=SEED)
test_idx, val_idx = train_test_split(temp_idx, test_size=0.5, random_state=SEED)

train_idx = torch.tensor(train_idx, device=device)
test_idx = torch.tensor(test_idx, device=device)
val_idx = torch.tensor(val_idx, device=device)

# pool里面是其他未标记的
# 假设你的样本总数是N（也就是y的样本数）
is_labeled = torch.zeros(y.size(0), dtype=torch.bool)
is_labeled[labeled_idx] = True  # 有标签的设为True
unlabeled_idx = torch.where(~is_labeled)[0]
pool_idx = unlabeled_idx.to(device)


# 初始化模型
#optimizer = optim.Adam(neural_net.parameters(), lr=7e-2)
# criterion = nn.MSELoss()
data = data.to(device)
data.x = data.x.to(device)
data.seq = data.seq.to(device)
data.pause = data.pause.to(device)
data.edge_index = data.edge_index.to(device)
y = y.to(device)

# 创建模型前，再设置一次种子 —— 确保模型的参数初始化是一样的
seed_everything(SEED)                 # ✅ 这里才是最关键的！
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