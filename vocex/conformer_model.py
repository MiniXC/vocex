import torch
from torch import nn
import lightning.pytorch as pl
from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer
from .scaler import GaussianMinMaxScaler
from .utils import Transpose

class Vocex(pl.LightningModule):

    def __init__(
        self,
        layers=8,
        measures=["energy", "pitch", "srmr", "snr"],
        in_channels=80,
        filter_size=256,
        kernel_size=3,
        dropout=0.1,
        depthwise=True,
        measure_nlayers=4,
        dvector_dim=256,
        dvector_nlayers=2,
        noise_factor=0.1,
        lr=1e-4,
    ):
        super().__init__()
        self.measures = measures
        self.noise_factor = noise_factor
        in_channels = in_channels
        filter_size = filter_size
        kernel_size = kernel_size
        dropout = dropout
        depthwise = depthwise
        num_outputs = len(self.measures)

        self.loss_compounds = self.measures + ["dvector"]
        
        self.in_layer = nn.Linear(in_channels, filter_size)

        self.positional_encoding = PositionalEncoding(filter_size)

        self.layers = TransformerEncoder(
            ConformerLayer(
                filter_size,
                2,
                conv_in=filter_size,
                conv_filter_size=filter_size,
                conv_kernel=(kernel_size, 1),
                batch_first=True,
                dropout=dropout,
                conv_depthwise=True,
            ),
            num_layers=measure_nlayers,
        )

        self.linear = nn.Sequential(
            nn.Linear(filter_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs),
        )

        dvector_dim = dvector_dim
        
        self.dvector_layers = TransformerEncoder(
            ConformerLayer(
                filter_size,
                2,
                conv_in=filter_size,
                conv_filter_size=filter_size,
                conv_kernel=(kernel_size, 1),
                batch_first=True,
                dropout=dropout,
                conv_depthwise=True,
            ),
            num_layers=dvector_nlayers,
        )

        dvector_input_dim = filter_size * 2
        
        self.dvector_linear = nn.Sequential(
            nn.Linear(dvector_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, dvector_dim),
        )

        self.scaler_dict = {
            k: GaussianMinMaxScaler(10) for k in self.measures
        }
        self.scaler_dict["mel"] = GaussianMinMaxScaler(10)
        self.scaler_dict["dvector"] = GaussianMinMaxScaler(10, sqrt=False)
        self.scaler_dict = nn.ModuleDict(self.scaler_dict)

        self.has_teacher = False
        self.lr = lr

        self.apply(self._init_weights)

    def set_teacher(self, teacher):
        self._external_teacher = teacher
        self.scaler_dict = teacher.scaler_dict
        self.has_teacher = True
        # freeze teacher
        for param in self._external_teacher.parameters():
            param.requires_grad = False

    @property
    def scalers(self):
        return self.scaler_dict

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, mel, dvector=None, measures=None, inference=False):
        if self.scalers["mel"]._n <= 1_000_000:
            self.scalers["mel"].partial_fit(mel)
        x = self.scalers["mel"].transform(mel)
        x = x + torch.randn_like(x) * self.noise_factor
        x = self.in_layer(x)
        x = self.positional_encoding(x)
        out_conv = self.layers(x)
        out = self.linear(out_conv)
        out = out.transpose(1, 2)
        measure_results = {}
        measure_true = {}
        loss = 0
        loss_dict = {}
        assert not (self.has_teacher and (measures is not None))
        assert (self.has_teacher or (measures is not None) or inference)
        ### teacher distillation
        if self.has_teacher:
            measures_loss = 0
            teacher_results = self._external_teacher(mel)["logits"]
            for i, measure in enumerate(self.measures):
                m_loss = nn.MSELoss()(out[:, i], teacher_results[:, i])
                measure_results[measure] = out[:, i]
                loss_dict[measure] = m_loss
                measures_loss = measures_loss + m_loss
            loss = loss + measures_loss / len(self.measures)
        ### measures (without teacher)
        if measures is not None:
            loss_dict = {}
            for i, measure in enumerate(self.measures):
                measure_out = out[:, i]
                if self.scalers[measure]._n <= 1_000_000:
                    self.scalers[measure].partial_fit(measures[measure])
                measure_results[measure] = measure_out #self.scalers[measure].transform(measure_out)
                measure_true[measure] = self.scalers[measure].transform(measures[measure])
            measures_loss = 0
            for measure in self.measures:
                m_loss = nn.MSELoss()(measure_results[measure], measure_true[measure])
                loss_dict[measure] = m_loss
                measures_loss += m_loss
            loss = measures_loss / len(self.measures)
            loss = loss + measures_loss / len(self.measures)
        ### d-vector
        # predict d-vector using global average and max pooling as input
        out_dvec = self.dvector_layers(x)
        dvector_input = torch.cat(
            [
                torch.mean(out_dvec, dim=1),
                torch.max(out_dvec, dim=1)[0],
            ],
            dim=1,
        )
        dvector_pred = self.dvector_linear(dvector_input)
        if dvector is not None:
            if self.scalers["dvector"]._n <= 1_000_000:
                self.scalers["dvector"].partial_fit(dvector)
            true_dvector = self.scalers["dvector"].transform(dvector)
            dvector_loss = nn.MSELoss()(dvector_pred, true_dvector)
            loss_dict["dvector"] = dvector_loss
            loss = loss + dvector_loss
        return {
            "loss": loss,
            "compound_losses": loss_dict,
            "logits": out,
            "logits_dvector": dvector_pred,
        }

    def training_step(self, batch, batch_idx):
        mel = batch["mel"]
        dvector = batch["dvector"]
        measures = batch["measures"]
        loss = self(mel, dvector, measures)
        result_dict = {
            f"train_loss_{k}": v for k, v in loss["compound_losses"].items()
        }
        result_dict["train_loss"] = loss["loss"]
        self.log_dict(
            result_dict,
            prog_bar=True,
            sync_dist=True,
        )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        mel = batch["mel"]
        dvector = batch["dvector"]
        measures = batch["measures"]
        loss = self(mel, dvector, measures)
        for measure in self.measures:
            # calculate MAE
            measure_results = self.scalers[measure].inverse_transform(
                loss["logits"][:, self.measures.index(measure)]
            )
            measure_true = measures[measure]
            mae = nn.L1Loss()(measure_results, measure_true)
            norm_mae = mae / torch.mean(measure_true)
            self.log_dict(
                {
                    f"val_loss_{measure}": loss["compound_losses"][measure],
                    f"val_mae_{measure}": mae,
                },
                prog_bar=True,
                sync_dist=True,
            )
        # d-vector
        dvector_results = self.scalers["dvector"].inverse_transform(
            loss["logits_dvector"]
        )
        dvector_true = dvector
        dvector_mae = nn.L1Loss()(dvector_results, dvector_true)
        norm_dvector_mae = dvector_mae / torch.mean(dvector_true)
        self.log_dict(
            {
                "val_loss_dvector": loss["compound_losses"]["dvector"],
                "val_mae_dvector": dvector_mae,
            },
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }