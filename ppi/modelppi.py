
from scipy import sparse as sp
import pandas as pd
import numpy as np
import torch

import torch.nn as nn
all_sequence_outputsnew=np.load('./data/all_sequence_outputsnew7132.npy')
all_sequence_outputsnew.shape
ppi_matrix=pd.read_csv('./data/9606ppi_matrix.csv')
ppi_matrix = sp.coo_matrix(ppi_matrix)
# featureDF.to_csv("./data/24077132kdncmergedf.csv")
merged_df=pd.read_csv('./data/24077132kdncmergedf.csv')
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv  # 使用SAGEConv替代GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.utils import set_seed
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops


SEED = 12
set_seed(SEED)


X_cpm_log2 = np.log2((merged_df[['rNC2']].values / np.median(merged_df[['rNC2']].values)) + 1)
y_cpm_log2 = np.log2((merged_df[['NC3']].values / np.median(merged_df[['NC3']].values))+ 1)


y = torch.tensor(y_cpm_log2, dtype=torch.float32).view(-1, 1)
X = torch.tensor(X_cpm_log2, dtype=torch.float32).view(-1, 1)

X_cpm_log2 = np.log2((merged_df[['rKD2']].values / np.median(merged_df[['rKD2']].values)) + 1)
y_cpm_log2 = np.log2((merged_df[['KD3']].values / np.median(merged_df[['KD3']].values))+ 1)


y1 = torch.tensor(y_cpm_log2, dtype=torch.float32).view(-1, 1)
X1 = torch.tensor(X_cpm_log2, dtype=torch.float32).view(-1, 1)
# 转换为PyTorch Tensor
edge_index, edge_weight = from_scipy_sparse_matrix(ppi_matrix.astype('float32'))


train_data = Data(
    x=X,
    edge_index=edge_index,
    edge_weight=edge_weight,
    y=y,
    seq=torch.tensor(all_sequence_outputsnew, dtype=torch.float32),
    pause=torch.tensor(merged_df['High_Pause_Countsnc'].values, dtype=torch.float32)
)



train_data.edge_index, train_data.edge_attr = add_self_loops(train_data.edge_index, train_data.edge_weight)


test_data = Data(
    x=X1,
    edge_index=edge_index,
    edge_weight=edge_weight,
    y=y1,
    seq=torch.tensor(all_sequence_outputsnew, dtype=torch.float32),
    pause=torch.tensor(merged_df['High_Pause_Countskd'].values, dtype=torch.float32)
)



test_data.edge_index, test_data.edge_attr = add_self_loops(test_data.edge_index, test_data.edge_weight)

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
        x = torch.cat((x, self.fc_paired(pausescore.view(-1, 1))), dim=1)
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
optimizer = optim.Adam(neural_net.parameters(), lr=3e-4)
criterion = nn.MSELoss()
train_data = train_data.to(device)
test_data = test_data.to(device)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv  # 使用SAGEConv替代GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from src.utils import set_seed
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops

model=NeuralGraph().to(device)
neural_net.load_state_dict(torch.load('./model/241121预测yizhicpmlog1p均一化e_200best_models12new2.pth'))

neural_net = neural_net.to(device)

model.eval()
from torch_geometric.utils import negative_sampling


positive_edge_index = train_data.edge_index


num_nodes = train_data.num_nodes
negative_edge_index = negative_sampling(
    edge_index=positive_edge_index,
    num_nodes=num_nodes,
    num_neg_samples=positive_edge_index.size(1)  # 生成与正样本数相同的负样本
)
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class EdgeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x1, x2 = self.features[idx]
        label = self.labels[idx]
        return x1, x2, label


neural_net.eval()
with torch.no_grad():
    _, node_embeddings = neural_net(train_data)  # 只计算一次


positive_features = [(node_embeddings[positive_edge_index[0, i]].unsqueeze(0), 
                      node_embeddings[positive_edge_index[1, i]].unsqueeze(0))
                     for i in range(positive_edge_index.size(1))]


negative_edge_index = negative_sampling(
    edge_index=positive_edge_index,  
    num_nodes=node_embeddings.size(0),  
    num_neg_samples=positive_edge_index.size(1) 
)


negative_features = [(node_embeddings[negative_edge_index[0, i]].unsqueeze(0), 
                      node_embeddings[negative_edge_index[1, i]].unsqueeze(0))
                     for i in range(negative_edge_index.size(1))]


features = positive_features + negative_features
labels = torch.cat([torch.ones(len(positive_features), 1), torch.zeros(len(negative_features), 1)], dim=0)


batch_size = 512 
dataset = EdgeDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 1),  
            nn.Sigmoid()        
        )

    def forward(self, x1, x2):

        x = torch.cat((x1, x2), dim=-1) 
        output = self.mlp(x)
        return output.squeeze(-1)  
mlp_model = MLP().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class EdgeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x1, x2 = self.features[idx]
        label = self.labels[idx]
        return x1, x2, label


neural_net.eval()
with torch.no_grad():
    _, node_embeddings = neural_net(train_data)  

positive_features = [(node_embeddings[positive_edge_index[0, i]].unsqueeze(0), 
                      node_embeddings[positive_edge_index[1, i]].unsqueeze(0))
                     for i in range(positive_edge_index.size(1))]

negative_edge_index = negative_sampling(
    edge_index=positive_edge_index, 
    num_nodes=node_embeddings.size(0),  
    num_neg_samples=positive_edge_index.size(1)  
)


negative_features = [(node_embeddings[negative_edge_index[0, i]].unsqueeze(0), 
                      node_embeddings[negative_edge_index[1, i]].unsqueeze(0))
                     for i in range(negative_edge_index.size(1))]


features = positive_features + negative_features
labels = torch.cat([torch.ones(len(positive_features), 1), torch.zeros(len(negative_features), 1)], dim=0)


batch_size = 512  
dataset = EdgeDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),  
            nn.ReLU(),
            nn.Linear(32, 1),   
            nn.Sigmoid()        
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1) 
        output = self.mlp(x)
        

        return output.squeeze(-1)  


mlp_model = MLP().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)


num_epochs = 10000  
for epoch in range(num_epochs):
    mlp_model.train()
    total_loss = 0

    for x1, x2, label in dataloader:
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)

        optimizer.zero_grad()
        predictions = mlp_model(x1, x2)
        
   
        loss = criterion(predictions, label)
        
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}')
torch.save(mlp_model, './model/241127mlp_model283epochloss3087.pth')
mlp_model = MLP().to(device)


model_path = './model/241127mlp_model283epochloss3087.pth'
mlp_model = torch.load(model_path)


mlp_model.eval()
import torch
from itertools import combinations
import csv


num_nodes = train_data.num_nodes
candidate_edges = list(combinations(range(num_nodes), 2))  # 所有可能的节点对 (i, j)
edge_candidates = torch.tensor(candidate_edges).t().contiguous()  # Shape: [2, num_candidate_edges]

neural_net.eval()
with torch.no_grad():
    _, node_embeddings = neural_net(train_data)


x1_candidates = node_embeddings[edge_candidates[0]]  # Shape: [num_candidate_edges, embedding_dim]
x2_candidates = node_embeddings[edge_candidates[1]]  # Shape: [num_candidate_edges, embedding_dim]


mlp_model.eval()
with torch.no_grad():
    edge_probs = mlp_model(x1_candidates, x2_candidates).squeeze(-1)


threshold = 0.8


edge_probs_cpu = edge_probs.cpu()
new_edges = edge_candidates[:, edge_probs_cpu >= threshold]
new_edges = new_edges.t().contiguous()  # Shape: [2, num_new_edges]

print(f"The new PPI contains {new_edges.shape} edges.")


new_edges_list = new_edges.cpu().numpy().tolist()
with open('new_ppi_edgesnew.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['node1', 'node2'])  # Header
    writer.writerows(new_edges_list)

print("New PPI edges exported to 'new_ppi_edgesnew.csv'")
import torch
import numpy as np
import csv


adj_matrix = torch.zeros((num_nodes, num_nodes))


adj_matrix[edge_candidates[0], edge_candidates[1]] = edge_probs.cpu()
adj_matrix[edge_candidates[1], edge_candidates[0]] = edge_probs.cpu()  # 确保矩阵对称
adj_matrix_np = adj_matrix.numpy()  # 转为 NumPy 数组
np.savetxt("full_adj_matrix.csv", adj_matrix_np, delimiter=",", fmt="%.6f")

print("Full adjacency matrix exported to 'full_adj_matrix.csv'")

