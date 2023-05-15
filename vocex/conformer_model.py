import torch
from torch import nn
from tqdm.auto import tqdm
from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer
from .scaler import GaussianMinMaxScaler
from .utils import Transpose, NoamLR
from .tpu_metrics_delta import MetricsDelta

class Vocex(nn.Module):

    def __init__(
        self,
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

        self.apply(self._init_weights)

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

    def fit_scalers(self, dataloader, n_batches=1000):
        for i, batch in tqdm(enumerate(dataloader), desc="Fitting scalers", total=n_batches):
            self.scalers["mel"].partial_fit(batch["mel"])
            for k in self.measures:
                self.scalers[k].partial_fit(batch["measures"][k])
            self.scalers["dvector"].partial_fit(batch["dvector"])
            if i >= n_batches:
                break
        for k in self.scalers:
            self.scalers[k].is_fit = True

    def forward(self, mel, dvector=None, measures=None, inference=False):
        if not self.scalers["mel"].is_fit:
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
        if measures is not None:
            loss_dict = {}
            for i, measure in enumerate(self.measures):
                measure_out = out[:, i]
                if not self.scalers[measure].is_fit:
                    self.scalers[measure].partial_fit(measures[measure])
                measure_results[measure] = measure_out
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
            if not self.scalers["dvector"].is_fit:
                self.scalers["dvector"].partial_fit(dvector)
            true_dvector = self.scalers["dvector"].transform(dvector)
            dvector_loss = nn.MSELoss()(dvector_pred, true_dvector)
            loss_dict["dvector"] = dvector_loss
            loss = loss + dvector_loss
        results = {
            "loss": loss,
            "compound_losses": loss_dict,
            "logits": out,
            "logits_dvector": dvector_pred,
        }
        return results