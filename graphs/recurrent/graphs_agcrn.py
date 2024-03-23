import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import AGCRN
from sklearn.metrics import precision_recall_fscore_support as score
from . import graphs_base as gb


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, number_of_nodes):
        super(RecurrentGCN, self).__init__()
        self.recurrent = AGCRN(number_of_nodes = number_of_nodes,
                              in_channels = node_features,
                              out_channels = 2,
                              K = 2,
                              embedding_dimensions = 4)
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x, e, h):
        h_0 = self.recurrent(x, e, h)
        y = F.relu(h_0)
        y = self.linear(y)
        return y, h_0
     

class ModelOps(gb.BaseGrafModelOps):
    def __init__(self) -> None:
        super().__init__()

    def train(self, loader,train_dataset, num_train=1):
        return super().train(loader,train_dataset, RecurrentGCN(node_features = 1, periods= loader.lags),num_train=num_train )
    
    def snapshot_train(self):
        number_of_nodes=len(self.loader._dataset["node_labels"])
        if self.extras.get("e") is None :
            e = torch.empty(number_of_nodes, 4)
            torch.nn.init.xavier_uniform_(e)
            self.extras["e"] = e
            self.extras["h"] = None

        x = self.snapshot.x.view(1, number_of_nodes, self.loader.lags)
        y_hat, self.extras["h"] = self.model(x, self.extras["e"], self.extras["h"])
        return y_hat

    def snapshot_eval(self):
        x = self.snapshot.x.view(1, self.snapshot.num_nodes, self.snapshot.num_features)
        y_hat, self.extras["h"] = self.model(x, self.extras["e"], self.extras["h"])
        return y_hat


    def snapshot_epoch(self, epoch):
        super().snapshot_epoch(epoch)
        self.extras["h"] = None

    def train(self, loader,train_dataset, num_train=1):
        number_of_nodes=len(loader._dataset["node_labels"])
        model = RecurrentGCN(node_features = loader.lags,number_of_nodes=number_of_nodes )
        #train(loader,train_dataset, model,num_train=1, snapshot_train_func=snapshot_train, snapshot_epoch_func=snapshot_epoch):
        return super().train(loader,train_dataset, model,num_train=num_train )
 
    
 