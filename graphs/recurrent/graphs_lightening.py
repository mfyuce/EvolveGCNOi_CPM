import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU
from . import graphs_base as gb

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split


class LitDiffConvModel(pl.LightningModule):

    def __init__(self, node_features, filters):
        super().__init__()
        self.recurrent = DCRNN(node_features, filters, 1)
        self.linear = torch.nn.Linear(filters, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch.x
        y = train_batch.y.view(-1, 1)
        edge_index = train_batch.edge_index
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        loss = F.mse_loss(h, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch.x
        y = val_batch.y.view(-1, 1)
        edge_index = val_batch.edge_index
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        loss = F.mse_loss(h, y)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics
 
 

class ModelOps(gb.BaseGrafModelOps):
    def __init__(self) -> None:
        super().__init__()
        self.trainer = None

    def train(self, loader,train_dataset, num_train=1):
        model = LitDiffConvModel(node_features = loader.lags,filters=16)
        early_stop_callback = EarlyStopping(monitor='val_loss',
                                            min_delta=0.00,
                                            patience=10,
                                            verbose=False,
                                            mode='max')

        self.trainer = pl.Trainer(callbacks=[early_stop_callback])
        return super().train(loader,train_dataset, LitDiffConvModel(node_features = loader.lags,filters=16),num_train=num_train )
         
