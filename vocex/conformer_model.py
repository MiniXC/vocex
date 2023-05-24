import torch
from torch import nn
from tqdm.auto import tqdm
from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer
from .scaler import GaussianMinMaxScaler
from .utils import Transpose, NoamLR
from .tpu_metrics_delta import MetricsDelta
from .softdtw import SoftDTW

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
        use_softdtw=False,
        softdtw_gamma=1.0,
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
                conv_depthwise=depthwise,
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
        
        self.dvector_conv_in_layer = nn.Linear(filter_size, filter_size)
        self.dvector_x_in_layer = nn.Linear(in_channels, filter_size)

        self.dvector_layers = TransformerEncoder(
            ConformerLayer(
                filter_size,
                2,
                conv_in=filter_size,
                conv_filter_size=filter_size,
                conv_kernel=(kernel_size, 1),
                batch_first=True,
                dropout=dropout,
                conv_depthwise=depthwise,
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
            if not k.endswith("_binary")
        }
        self.scaler_dict["mel"] = GaussianMinMaxScaler(10)
        self.scaler_dict["dvector"] = GaussianMinMaxScaler(10)
        self.scaler_dict = nn.ModuleDict(self.scaler_dict)
        self.use_softdtw = use_softdtw
        if self.use_softdtw:
            self.softdtw = SoftDTW(gamma=softdtw_gamma)

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
        # reset scalers
        for k in self.scalers:
            self.scalers[k].is_fit = False
        for i, batch in tqdm(enumerate(dataloader), desc="Fitting scalers", total=n_batches):
            self.scalers["mel"].partial_fit(batch["mel"])
            for k in self.measures:
                if not k.endswith("_binary"):
                    self.scalers[k].partial_fit(batch["measures"][k])
            self.scalers["dvector"].partial_fit(batch["dvector"])
            if i >= n_batches:
                break
        for k in self.scalers:
            self.scalers[k].is_fit = True

    def forward(self, mel, dvector=None, measures=None, inference=False):
        mel_padding_mask = mel.sum(dim=-1) != 0
        mel_padding_mask = mel_padding_mask.to(torch.float32)
        if not self.scalers["mel"].is_fit:
            self.scalers["mel"].partial_fit(mel)
        x = self.scalers["mel"].transform(mel)
        x = x + (torch.randn_like(x) * x.std() + x.mean()) * self.noise_factor
        out = self.in_layer(x)
        out = self.positional_encoding(out)
        out_conv = self.layers(out)
        out = self.linear(out_conv)
        out = out.transpose(1, 2)
        measure_results = {}
        measure_true = {}
        loss_dict = {}
        for i, measure in enumerate(self.measures):
            measure_out = out[:, i]
            if not measure.endswith("_binary") and not self.scalers[measure].is_fit:
                self.scalers[measure].partial_fit(measures[measure])
            measure_results[measure] = measure_out
            if measure.endswith("_binary"):
                measure_results[measure] = torch.sigmoid(measure_results[measure])
        if measures is not None:
            loss_dict = {}
            for measure in self.measures:
                if not measure.endswith("_binary"):
                    measure_true[measure] = self.scalers[measure].transform(measures[measure])
                else:
                    measure_true[measure] = measures[measure]
            measures_losses = []
            for measure in self.measures:
                if measure.endswith("_binary"):
                    m_loss = nn.BCELoss()(measure_results[measure], measure_true[measure])
                else:
                    if not self.use_softdtw:
                        m_loss = nn.MSELoss()(measure_results[measure]*mel_padding_mask, measure_true[measure]*mel_padding_mask)
                    else:
                        m_loss = self.softdtw(
                            measure_results[measure]*mel_padding_mask,
                            measure_true[measure]*mel_padding_mask,
                        ) / 1000
                loss_dict[measure] = m_loss
                measures_losses.append(m_loss)
            loss = sum(measures_losses) / len(self.measures)
        else:
            loss = None
        ### d-vector
        # predict d-vector using global average and max pooling as input
        out_conv_dvec = self.dvector_conv_in_layer(out_conv)
        x = self.dvector_x_in_layer(x)
        out_dvec = self.dvector_layers(out_conv_dvec + x)   
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
            dvector_loss = nn.L1Loss()(dvector_pred, true_dvector)
            loss_dict["dvector"] = dvector_loss
            if loss is not None:
                loss += dvector_loss
                loss /= 2
            else:
                loss = dvector_loss
        if not inference:
            results = {
                "loss": loss,
                "compound_losses": loss_dict,
            }
            return results
        else:
            # transform back to original scale
            for measure in self.measures:
                if not measure.endswith("_binary"):
                    measure_results[measure] = self.scalers[measure].inverse_transform(
                        measure_results[measure]
                    )
            dvector_pred = self.scalers["dvector"].inverse_transform(dvector_pred)
            results = {
                "loss": loss,
                "compound_losses": loss_dict,
                "measures": measure_results,
                "dvector": dvector_pred,
            }
            return results