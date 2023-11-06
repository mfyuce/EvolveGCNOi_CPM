import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from sklearn.metrics import precision_recall_fscore_support as score
from . import graphs_base as gb


class RecurrentGCN(torch.nn.Module):
    def __init__(self,  node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = EvolveGCNO(node_features)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
 


class ModelOps(gb.BaseGrafModelOps):
    def __init__(self) -> None:
        super().__init__()
        
    def train(self, loader,train_dataset, num_train=1, plot_model=True, calc_perf=True):
        model = RecurrentGCN(node_features = loader.lags)
        for param in model.parameters():
            param.retain_grad()
        return super().train(loader,train_dataset, model, num_train=num_train, plot_model=plot_model, calc_perf=calc_perf )
         