import torch
import torch.nn.functional as F
from .evolvegcnh_improved import EvolveGCNHImproved
from sklearn.metrics import precision_recall_fscore_support as score
from . import graphs_base as gb
from torch_geometric.nn import GCNConv

class RecurrentGCN1(torch.nn.Module):
    def __init__(self, node_count, node_features):
        super(RecurrentGCN1, self).__init__()
        self.recurrent = EvolveGCNHImproved(node_count, node_features)
        self.conv1 = GCNConv(node_features, 4)
        # self.conv2 = GCNConv(4, 4)
        # self.conv3 = GCNConv(4, 2)
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)


        h = self.conv1(h, edge_index)
        # h = h.tanh()
        # h = self.conv2(h, edge_index)
        # h = h.tanh()
        # h = F.dropout(h, p=0.5, training=self.training)
        # h = self.conv3(h, edge_index)
        out = h.tanh()  # Final GNN embedding space.



        # h = F.relu(h)
        # h = F.relu6(h)
        # h = F.selu(h)
        h = self.linear(h)
        return h, out

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_count, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = EvolveGCNHImproved(node_count, node_features)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        # h = F.relu(h)
        # h = F.relu6(h)
        h = F.selu(h)
        h = self.linear(h)
        return h,None
 


class ModelOps(gb.BaseGrafModelOps):
    def __init__(self) -> None:
        super().__init__()
    
    def snapshot_train(self): 
        y_hat, out =  self.model(self.snapshot.x, self.snapshot.edge_index, self.snapshot.edge_attr )
        return y_hat
    
    def train(self, loader,train_dataset, num_train=1,plot_model=False, calc_perf=True):
        number_of_nodes=len(loader._dataset["node_labels"])
        return super().train(loader,train_dataset, RecurrentGCN1(node_features = loader.lags, node_count = number_of_nodes),num_train=num_train,plot_model=plot_model, calc_perf=calc_perf )
      
