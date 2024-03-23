import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM
 
from . import graphs_base as gb

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0


class ModelOps(gb.BaseGrafModelOps):
    def __init__(self) -> None:
        super().__init__()

    def snapshot_train(self):
        number_of_nodes=len(self.loader._dataset["node_labels"])
        if self.extras.get("h") is None :
            self.extras["h"] = None
            self.extras["c"] = None

        y_hat, self.extras["h"], self.extras["c"] = \
            self.model(self.snapshot.x, self.snapshot.edge_index, self.snapshot.edge_attr, \
                            self.extras["h"], self.extras["c"])
        return y_hat

    def snapshot_eval(self): 
        y_hat, self.extras["h"], self.extras["c"] = \
            self.model(self.snapshot.x, self.snapshot.edge_index, self.snapshot.edge_attr, \
                            self.extras["h"], self.extras["c"])
        return y_hat

    def snapshot_epoch(self, epoch):
        super().snapshot_epoch(epoch, self)
        self.extras["h"] = None
        self.extras["c"] = None

    def train(self, loader,train_dataset, num_train=1):
        number_of_nodes=len(loader._dataset["node_labels"])
        model = RecurrentGCN(node_features = loader.lags )
        #train(loader,train_dataset, model,num_train=1, snapshot_train_func=snapshot_train, snapshot_epoch_func=snapshot_epoch):
        return super().train(loader,train_dataset, model,num_train=num_train  )
         
 