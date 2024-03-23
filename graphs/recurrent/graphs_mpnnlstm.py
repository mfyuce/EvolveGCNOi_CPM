import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import MPNNLSTM
from . import graphs_base as gb
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

class RecurrentGCN(torch.nn.Module):
    def __init__(self, number_of_nodes):
        super(RecurrentGCN, self).__init__()
        self.recurrent = MPNNLSTM(32, 32, number_of_nodes, 1, 0.5)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
 


class ModelOps(gb.BaseGrafModelOps):
    def __init__(self) -> None:
        super().__init__()
        self.trainer = None

    def train(self, loader,train_dataset, num_train=1):
        number_of_nodes=len(loader._dataset["node_labels"])
        return super().train(loader,train_dataset, RecurrentGCN(number_of_nodes = number_of_nodes),num_train=num_train )
         