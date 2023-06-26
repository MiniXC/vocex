import torch
from torch import nn
from tqdm.auto import tqdm
from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer
from .scaler import GaussianMinMaxScaler
from .utils import Transpose, NoamLR
from .softdtw import SoftDTW
from .pooling import AttentiveStatsPooling

class VocexModel(nn.Module):

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

        self.verbose = False

        self.apply(self._init_weights)

        # save hparams
        self.hparams = {
            "measures": measures,
            "in_channels": in_channels,
            "filter_size": filter_size,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "depthwise": depthwise,
            "measure_nlayers": measure_nlayers,
            "dvector_dim": dvector_dim,
            "dvector_nlayers": dvector_nlayers,
            "noise_factor": noise_factor,
            "use_softdtw": use_softdtw,
            "softdtw_gamma": softdtw_gamma,
        }

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

    def forward(self, mel, dvector=None, measures=None, inference=False, return_activations=False, return_attention=False):
        mel_padding_mask = mel.sum(dim=-1) != 0
        mel_padding_mask = mel_padding_mask.to(mel.dtype)

        if return_activations:
            self.layers.return_additional_layers = list(range(self.hparams["measure_nlayers"]))

        if not self.scalers["mel"].is_fit:
            self.scalers["mel"].partial_fit(mel)
        x = self.scalers["mel"].transform(mel)
        is_onnx = (hasattr(self, "onnx_export") and self.onnx_export)
        if is_onnx:
            random_noise = (torch.randn_like(x, dtype=float) * x.std() + x.mean()) * self.noise_factor
        else:
            random_noise = (torch.randn_like(x) * x.std() + x.mean()) * self.noise_factor
        if is_onnx:
            x_dtype = x.dtype
            x = x + random_noise
            x = x.to(x_dtype)
        else:
            x = x + random_noise
        out = self.in_layer(x)
        out = self.positional_encoding(out)
        res = self.layers(out, src_key_padding_mask=mel_padding_mask, need_weights=return_attention)
        if return_activations:
            activations = res["activations"]
            out_conv = res["output"]
            self.layers.return_additional_layers = None
        if return_attention:
            attention = res["attention"]
            out_conv = res["output"]
        if not return_activations and not return_attention:
            out_conv = res
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
                    m_loss = nn.BCEWithLogitsLoss()(measure_results[measure]*mel_padding_mask, measure_true[measure]*mel_padding_mask)
                else:
                    if not self.use_softdtw:
                        m_loss = nn.MSELoss()(measure_results[measure]*mel_padding_mask, measure_true[measure]*mel_padding_mask)
                    else:
                        if self.verbose:
                            print(measure_results[measure], measure_true[measure])
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
        out_dvec = self.dvector_layers(out_conv_dvec + x, src_key_padding_mask=mel_padding_mask)
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
        if (not inference) and not (hasattr(self, "onnx_export") and self.onnx_export):
            results = {
                "loss": loss,
                "compound_losses": loss_dict,
            }
            if return_activations:
                results["activations"] = [a.detach() for a in activations]
            if return_attention:
                results["attention"] = [a.detach() for a in attention]
            return results
        else:
            # transform back to original scale
            for measure in self.measures:
                if not measure.endswith("_binary"):
                    measure_results[measure] = self.scalers[measure].inverse_transform(
                        measure_results[measure]
                    )
                else:
                    measure_results[measure] = torch.sigmoid(measure_results[measure]).detach()
            dvector_pred = self.scalers["dvector"].inverse_transform(dvector_pred)
            if is_onnx:
                return [measure_results[measure] for measure in self.measures] + [dvector_pred]
            results = {
                "loss": loss,
                "compound_losses": loss_dict,
                "measures": measure_results,
                "dvector": dvector_pred,
            }
            if return_activations:
                results["activations"] = [a.detach() for a in activations]
            if return_attention:
                results["attention"] = [a.detach() for a in attention]
            return results

class Vocex2Model(nn.Module):
    """
    Version 2 of the model, which is trained using Version 1 as teacher model.
    We augment the input with various augmentations and train the model to predict
    the teacher's output on the original input.
    Due to this, we disregard SNR and SRMR for now.
    Also, instead of using externally trained d-vectors (or ones from the teacher), we simply use CrossEntropyLoss to predict which
    speaker the input belongs to, and use the corresponding embedding as a d-vector replacement.
    """

    def __init__(
        self,
        measures=["energy", "pitch", "voice_activity_binary"],
        in_channels=80,
        filter_size=256,
        kernel_size=3,
        dropout=0.1,
        depthwise=True,
        nlayers=4,
        speaker_emb_dim=256,
    ):
        super().__init__()
        self.measures = measures
        in_channels = in_channels
        num_outputs = len(self.measures)

        self.loss_compounds = self.measures + ["dvector", "augmentations"]

        self.in_layer = nn.Linear(in_channels, filter_size)

        self.positional_encoding = PositionalEncoding(filter_size)

        self.transformer = TransformerEncoder(
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
            num_layers=nlayers,
        )

        self.measure_output_layer = nn.Linear(filter_size, num_outputs)

        self.speaker_output_layer = nn.Sequential(
            nn.Linear(filter_size, speaker_emb_dim),
            AttentiveStatsPooling(speaker_emb_dim, filter_size),
            nn.BatchNorm1d(speaker_emb_dim * 2),
            nn.Linear(speaker_emb_dim * 2, speaker_emb_dim),
            nn.BatchNorm1d(speaker_emb_dim),
        )

        self.apply(self._init_weights)

        # save hparams
        self.hparams = {
            "measures": measures,
            "in_channels": in_channels,
            "filter_size": filter_size,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "depthwise": depthwise,
            "nlayers": nlayers,
            "speaker_emb_dim": speaker_emb_dim,
        }

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, mel):
        mel_padding_mask = mel.sum(dim=-1) != 0
        mel_padding_mask = mel_padding_mask.to(mel.dtype)
        x = self.in_layer(mel)
        x = self.positional_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mel_padding_mask)
        measures = self.measure_output_layer(x)
        speaker_emb = self.speaker_output_layer(x)
        return {
            "measures": {
                m: measures[:, :, i] for i, m in enumerate(self.measures)
            },
            "speaker_embedding": speaker_emb,
            "padding_mask": mel_padding_mask,
        }

