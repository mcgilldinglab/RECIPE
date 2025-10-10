import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse as sp
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv  
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.utils import set_seed
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops


SEED = 12
set_seed(SEED)


all_sequence_outputsnew = np.load('./data/all_sequence_outputsnew7132.npy')
ppi_matrix = pd.read_csv('./data/9606ppi_matrix.csv').values
ppi_matrix = sp.coo_matrix(ppi_matrix)
merged_df = pd.read_csv('./data/24077132kdncmergedf.csv')

X_cpm_log2 = np.log2((merged_df[['rNC2']].values / np.median(merged_df[['rNC2']].values)) + 1)
y_cpm_log2 = np.log2((merged_df[['NC3']].values / np.median(merged_df[['NC3']].values))+ 1)


y = torch.tensor(y_cpm_log2, dtype=torch.float32).view(-1, 1)
X = torch.tensor(X_cpm_log2, dtype=torch.float32).view(-1, 1)

X_cpm_log2 = np.log2((merged_df[['rKD2']].values / np.median(merged_df[['rKD2']].values)) + 1)
y_cpm_log2 = np.log2((merged_df[['KD3']].values / np.median(merged_df[['KD3']].values))+ 1)


y1 = torch.tensor(y_cpm_log2, dtype=torch.float32).view(-1, 1)
X1 = torch.tensor(X_cpm_log2, dtype=torch.float32).view(-1, 1)

edge_index, edge_weight = from_scipy_sparse_matrix(ppi_matrix.astype('float32'))


train_data = Data(
    x=X,
    edge_index=edge_index,
    edge_weight=edge_weight,
    y=y,
    seq=torch.tensor(all_sequence_outputsnew, dtype=torch.float32),
    pause=torch.tensor(merged_df['High_Pause_Countsnc'].values, dtype=torch.float32)
)


# 添加自环
train_data.edge_index, train_data.edge_attr = add_self_loops(train_data.edge_index, train_data.edge_weight)


# 创建测试集Data对象
test_data = Data(
    x=X1,
    edge_index=edge_index,
    edge_weight=edge_weight,
    y=y1,
    seq=torch.tensor(all_sequence_outputsnew, dtype=torch.float32),
    pause=torch.tensor(merged_df['High_Pause_Countskd'].values, dtype=torch.float32)
)

# 添加自环
test_data.edge_index, test_data.edge_attr = add_self_loops(test_data.edge_index, test_data.edge_weight)
print('：', test_data)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 32)
        )

        self.encoder = nn.Sequential(
            nn.Linear(9216, 32),
            nn.GELU(),
        )

        self.conv1 = SAGEConv(64, 32)
        self.regressor = nn.Linear(32, 1)
        self.regressor_activation = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, data):
        x = data.x.view(-1, 1)
        seq_embedding = data.seq
        edge_index = data.edge_index
        pausescore = data.pause.view(-1, 1)

        x = self.fc(x) + self.encoder(seq_embedding)
        x = torch.cat((self.fc(pausescore), x), dim=1)
        x = F.gelu(x)
        x = self.conv1(x, edge_index)  
        z = F.gelu(x)
        x = self.regressor(z)
        #x = self.regressor_activation(x)
        return x,z

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
neural_net = NeuralNet().to(device)

# 损失函数和优化器
optimizer = optim.Adam(neural_net.parameters(), lr=1e-2)
criterion = nn.MSELoss()

num_epochs = 13000

train_data = train_data.to(device)
test_data = test_data.to(device)

# 初始化最佳 R² 值
best_train_r2 = -float('inf')
early_stop_patience = 50  
patience_counter = 0

for epoch in range(num_epochs):
    neural_net.train()

    optimizer.zero_grad()

    y_pred, _ = neural_net(train_data)
    loss = criterion(y_pred.view(-1), train_data.y.view(-1))

    loss.backward()
    optimizer.step()
    
    if epoch % 1 == 0:
        neural_net.eval()
        with torch.no_grad():
            y_train_pred, _ = neural_net(train_data)
            y_train_pred = y_train_pred.view(-1).cpu()

            train_r2 = r2_score(train_data.y.cpu().numpy(), y_train_pred.numpy())
            
            y_test_pred, _ = neural_net(test_data)
            y_test_pred = y_test_pred.view(-1).cpu()
            test_r2 = r2_score(test_data.y.cpu().numpy(), y_test_pred.numpy())
        
        
        if train_r2 > best_train_r2:
            best_train_r2 = train_r2
            patience_counter = 0  
        
            torch.save(neural_net.state_dict(), './model/241121cpmlog1pz.pth')
        else:            
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break
        print(f'Epoch: {epoch}, Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}')
        

