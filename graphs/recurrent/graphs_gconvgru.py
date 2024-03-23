import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU
from . import graphs_base as gb

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
 

class ModelOps(gb.BaseGrafModelOps):
    def __init__(self) -> None:
        super().__init__()
        
    def train(self, loader,train_dataset, num_train=1):
        return super().train(loader,train_dataset, RecurrentGCN(node_features = loader.lags),num_train=num_train )
      
    
