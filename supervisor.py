import pytorch_lightning as pl
import torch
from torch import optim
from STNET.model import GWNET

from libs import utils


class Predictor(pl.LightningModule):
    def __init__(self, model, hparams, A, scaler, **kwargs):
        super(Predictor, self).__init__()
        self.model = model
        self.lr = hparams['OPTIMIZER']['lr']
        self.weight_decay = hparams['OPTIMIZER']['weight_decay']
        self.scaler = scaler
        self.target_metric = hparams['OPTIMIZER']['target_metric']
        self.monitor_metric = hparams['OPTIMIZER']['monitor_metric']

        # for logging computational graph in tensorboard
        self.example_input_array = torch.rand(
            hparams['DATA']['batch_size'], hparams['MODEL']['in_features'], A.size(0), hparams['DATA']['seq_len'])
        self.save_hyperparameters('hparams')

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {self.monitor_metric: 0})

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, idx):
        x, y = batch
        pred = self.scaler.inverse_transform(self.forward(x))
        loss = utils.masked_MAE(pred, y)
        self.log(self.target_metric, loss, prog_bar=False)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        pred = self.scaler.inverse_transform(self.forward(x))
        return {'y': y, 'pred': pred}

    def validation_epoch_end(self, outputs):
        y = torch.cat([output['y'] for output in outputs], dim=0)
        pred = torch.cat([output['pred'] for output in outputs], dim=0)
        loss = utils.masked_MAE(pred, y)
        self.log_dict(
            {self.monitor_metric: loss, "step": self.current_epoch}, prog_bar=True, on_epoch=True)

    def test_step(self, batch, idx):
        x, y = batch
        pred = self.scaler.inverse_transform(self.forward(x))
        return {'y': y, 'pred': pred}

    def test_epoch_end(self, outputs):
        # target and prediction
        y = torch.cat([output['y'] for output in outputs], dim=0)
        pred = torch.cat([output['pred'] for output in outputs], dim=0)

        # calculate error for each metric
        loss = {'mae': utils.masked_MAE(pred, y, dim=(0, -1)),
                'rmse': utils.masked_RMSE(pred, y, dim=(0, -1)),
                'mape': utils.masked_MAPE(pred, y, dim=(0, -1))}

        # error for each horizon
        for h in range(len(loss["mae"])):
            print(f"Horizon {h+1} ({5*(h+1)} min) - ", end="")
            print(f"MAE: {loss['mae'][h]:.2f}", end=", ")
            print(f"RMSE: {loss['rmse'][h]:.2f}", end=", ")
            print(f"MAPE: {loss['mape'][h]:.2f}")
            if self.logger:
                for m in loss:
                    self.logger.experiment.add_scalar(
                        f"Test/{m}", loss[m][h], h)

        # aggregated error
        print("Aggregation - ", end="")
        print(f"MAE: {loss['mae'].mean():.2f}", end=", ")
        print(f"RMSE: {loss['rmse'].mean():.2f}", end=", ")
        print(f"MAPE: {loss['mape'].mean():.2f}")

        self.test_results = pred.cpu()

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr,
                          weight_decay=self.weight_decay)


class GWNETPredictor(Predictor):
    def __init__(self, hparams, A, scaler, **kwargs):
        super().__init__(GWNET(in_dim=hparams['MODEL']['in_features'],
                               enc_in_dim=hparams['MODEL']['hidden_dim'],
                               enc_hid_dim=hparams['MODEL']['hidden_dim'],
                               enc_out_dim=hparams['MODEL']['hidden_dim'] * 8,
                               num_enc_blocks=hparams['MODEL']['num_enc_blocks'],
                               dec_hid_dim=hparams['MODEL']['hidden_dim'] * 16,
                               dec_out_dim=hparams['DATA']['horizon'],
                               kernel_size=hparams['MODEL']['kernel_size'],
                               num_gnn_layers=hparams['MODEL']['num_gnn_layers'],
                               num_temp_layers=hparams['MODEL']['num_temp_layers'],
                               adj_mode=hparams['MODEL']['adj_mode'],
                               adj_type=hparams['MODEL']['adj_type'],
                               num_adaptive_adj=hparams['MODEL']['num_adaptive_adj'],
                               A=A,
                               dropout=hparams['OPTIMIZER']['dropout']), hparams, A, scaler)
