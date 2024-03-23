import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN, A3TGCN
from sklearn.metrics import precision_recall_fscore_support as score
from . import graphs_base as gb
 
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(RecurrentGCN, self).__init__()
        self.recurrent = A3TGCN(node_features, 32, periods)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h  
 
class ModelOps(gb.BaseGrafModelOps):
    def __init__(self) -> None:
        super().__init__()
    def train(self, loader,train_dataset, num_train=1):
        return super().train(loader,train_dataset, RecurrentGCN(node_features = 1, periods= loader.lags),num_train=num_train )
    
    