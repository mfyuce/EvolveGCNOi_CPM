import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNH
from sklearn.metrics import precision_recall_fscore_support as score
from . import graphs_base as gb


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_count, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = EvolveGCNH(node_count, node_features)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
 


class ModelOps(gb.BaseGrafModelOps):
    def __init__(self) -> None:
        super().__init__()
    def train(self, loader,train_dataset, num_train=1,plot_model=False):
        number_of_nodes=len(loader._dataset["node_labels"])
        return super().train(loader,train_dataset, RecurrentGCN(node_features = loader.lags, node_count = number_of_nodes),num_train=num_train,plot_model=plot_model )
      
