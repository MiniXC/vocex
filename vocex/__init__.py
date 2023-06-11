import gzip

import torch
import torchaudio.transforms as T
from librosa.filters import mel as librosa_mel
import numpy as np
from torch import nn

from .conformer_model import VocexModel
from .image_helpers import QuantizeToGivenPalette, transformYIQ2RGB

class Vocex():
    """ 
    This is a wrapper class for the vocex model. 
    It is used to load the model and perform inference on given audio file(s).
    """

    @staticmethod
    def _drc(x, C=1, clip_val=1e-7):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    @staticmethod
    def from_pretrained(model_file, compressed=True, for_onnx=False, **postprocess_kwargs):
        """ 
        Load a pretrained model from a given .pt file.
        Also accepts huggingface model names.
        """
        if compressed:
            # decompress
            with gzip.open(model_file, "rb") as f:
                checkpoint = torch.load(f)
        else:
            checkpoint = torch.load(model_file)
        model_args = checkpoint["model_args"]
        model = VocexModel(**model_args)
        model.load_state_dict(checkpoint["state_dict"])
        return Vocex(model, for_onnx, **postprocess_kwargs)

    def __init__(self, model, seed=42, **postprocess_kwargs):
        super().__init__()
        self.model = model
        self.mel_spectrogram = T.Spectrogram(
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            pad=0,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        mel_basis = librosa_mel(
            sr=22050,
            n_fft=1024,
            n_mels=80,
            fmin=0,
            fmax=8000,
        )
        self.mel_basis = torch.from_numpy(mel_basis).float()
        self.postprocess_kwargs = postprocess_kwargs
        for measure in self.model.measures:
            if measure not in self.postprocess_kwargs:
                self.postprocess_kwargs[measure] = {
                    "use_convolution": True,
                    "convolution_window_size": 10,
                    "convolution_window": "hann",
                    "interpolate": True,
                }
                if measure == "voice_activity":
                    self.postprocess_kwargs[measure]["normalize"] = True
                else:
                    self.postprocess_kwargs[measure]["normalize"] = False
                if measure == "pitch":
                    self.postprocess_kwargs[measure]["vad_threshold_min"] = 0.5
                    self.postprocess_kwargs[measure]["vad_threshold_max"] = 1.0
                elif measure == "snr":
                    self.postprocess_kwargs[measure]["vad_threshold_min"] = 0.4
                    self.postprocess_kwargs[measure]["vad_threshold_max"] = 1.0
                    self.postprocess_kwargs[measure]["convolution_window_size"] = 200
                elif measure == "srmr":
                    self.postprocess_kwargs[measure]["vad_threshold_min"] = 0.5
                    self.postprocess_kwargs[measure]["vad_threshold_max"] = 1.0
                    self.postprocess_kwargs[measure]["convolution_window_size"] = 200
                elif measure == "energy":
                    self.postprocess_kwargs[measure]["vad_threshold_min"] = 0.0
                    self.postprocess_kwargs[measure]["vad_threshold_max"] = 1.0
                
        # set to eval mode
        self.model.eval()
        self.seed = seed

    def save_checkpoint(self, path, compressed=True):
        """ Save the model to a given path. """
        # get state dict
        state_dict = self.model.state_dict()
        # get model args
        model_args = self.model.hparams

        if compressed:
            # compress
            with gzip.open(path, "wb") as f:
                torch.save({
                    "state_dict": state_dict,
                    "model_args": model_args,
                }, f)
        else:
            torch.save({
                "state_dict": state_dict,
                "model_args": model_args,
            }, path)

    @staticmethod
    def _interpolate(x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]
        nans, y = nan_helper(x)
        if not np.all(nans):
            x[nans] = np.interp(y(nans), y(~nans), x[~nans])
        else:
            x = np.zeros_like(x)
        return torch.tensor(x)

    def _preprocess(self, audio, sr=22050):
        """ Preprocess audio. """
        if isinstance(audio, list):
            # pad to same length
            max_length = max([a.shape[-1] for a in audio])
            audio = [torch.from_numpy(a).float() if isinstance(a, np.ndarray) else a for a in audio]
            audio = [torch.nn.functional.pad(a, (0, max_length - a.shape[-1]), mode="constant", value=0) for a in audio]
            # stack
            audio = torch.stack(audio)
        # convert to tensor
        elif isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        # resample if necessary
        if sr != 22050:
            if audio.dtype == torch.int16:
                audio = audio / 32768
            elif audio.dtype == torch.float64:
                audio = audio.to(torch.float32)
            audio = T.Resample(sr, 22050)(audio)
        if audio.ndim == 1:
            # normalize unbatched
            audio = audio / audio.abs().max()
        elif audio.ndim == 2:
            # normalize batched
            audio = audio / audio.abs().max(dim=-1, keepdim=True)[0]
        elif audio.ndim == 3:
            # normalize batched and make mono
            audio = audio.mean(dim=-1)
            audio = audio / audio.abs().max(dim=-1, keepdim=True)[0]
        # convert to spectrogram
        mel = self.mel_spectrogram(audio)
        # convert to mel spectrogram
        mel = torch.matmul(self.mel_basis, mel)
        # dynamic range compression
        mel = self._drc(mel)
        # transpose
        if mel.ndim == 2:
            # add batch dimension if necessary
            mel = mel.unsqueeze(0)
        mel = mel.transpose(1, 2)
        return mel

    def postprocess(
        self,
        measure,
        vad=None,
        use_convolution=True,
        convolution_window_size=10,
        convolution_window="hann",
        interpolate=True,
        normalize=False,
        vad_threshold_min=0.0,
        vad_threshold_max=1.0,
    ):
        """ Postprocess a given measure. """
        def _interpolate(measure, vad):
            if vad is not None and (vad_threshold_min > 0 or vad_threshold_max < 1):
                # apply voice activity detection
                measure[vad < vad_threshold_min] = torch.nan
                measure[vad > vad_threshold_max] = torch.nan
                if interpolate:
                    # interpolate nan values
                    if measure.shape[0] == 1:
                        measure[0] = Vocex._interpolate(measure[0])
                    else:
                        # for batch size > 0, we want to interpolate each batch element separately
                        measure = torch.stack([Vocex._interpolate(m) for m in measure])
                else:
                    # set missing values to 0
                    measure[torch.isnan(measure)] = 0
            return measure

        if use_convolution:
            measure = _interpolate(measure, vad)
            # apply convolution
            if convolution_window == "hann":
                window = torch.hann_window(convolution_window_size)
            elif convolution_window == "hamming":
                window = torch.hamming_window(convolution_window_size)
            elif convolution_window == "blackman":
                window = torch.blackman_window(convolution_window_size)
            else:
                raise ValueError("convolution_window must be one of 'hann', 'hamming', or 'blackman'")
            window = window / window.sum()
            window = window.unsqueeze(0).unsqueeze(0)
            # use convolution from torch.nn.functional
            # pad first (using reflection padding)
            if not isinstance(measure, torch.Tensor):
                measure = torch.tensor(measure)
            if measure.shape[-1] < convolution_window_size:
                # if measure is too short, we need to pad it with zeros on 
                measure = torch.nn.functional.pad(measure, (convolution_window_size // 2, convolution_window_size // 2), mode="constant", value=0)
                print("WARNING: measure is too short, padding with zeros")
            else:
                measure = torch.nn.functional.pad(measure, (convolution_window_size // 2, convolution_window_size // 2), mode="reflect")
            if measure.ndim == 1:
                # add batch dimension if necessary
                measure = measure.unsqueeze(0)
            if measure.shape[0] == 1:
                measure = torch.nn.functional.conv1d(measure, window, padding="same")
            else:
                measure = torch.stack([torch.nn.functional.conv1d(m.unsqueeze(0), window, padding="same") for m in measure]).squeeze(1)
            # remove padding
            measure = measure[:, convolution_window_size // 2:-convolution_window_size // 2]
        if normalize:
            # normalize
            measure = measure / measure.abs().max(dim=-1, keepdim=True)[0]
        measure = _interpolate(measure, vad)
        return measure

    def forward(self, audio, sr=22050, return_activations=False, return_attention=False, speaker_avatar=False):
        """ Perform inference on given audio. """
        is_onnx = hasattr(self.model, "onnx_export") and self.model.onnx_export

        # preprocess
        mel = self._preprocess(audio, sr)
        # forward pass
        with torch.no_grad():
            if self.seed is not None:
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)
            out = self.model(mel, inference=True, return_activations=return_activations, return_attention=return_attention)

        if return_activations:
            out["activations"] = torch.stack(out["activations"]).transpose(0, 1).cpu().numpy()
        
        if return_attention:
            out["attention"] = torch.stack(out["attention"]).transpose(0, 1).cpu().numpy()

        if not is_onnx:
            del out["loss"]
            del out["compound_losses"]

        # convert to numpy
        if not is_onnx:
            for measure in out["measures"].keys():
                out[measure] = out["measures"][measure].cpu().numpy()
        # do voice activity postprocessing first
        voice_activity = None
        if "voice_activity_binary" in out or (is_onnx and "voice_activity_binary" in self.model.measures):
            if not is_onnx:
                voice_activity_vals = out["voice_activity_binary"]
            else:
                # use index of voice activity in self.model.measures
                voice_activity_vals = out[self.model.measures.index("voice_activity_binary")]
            voice_activity = self.postprocess(voice_activity_vals, None, **self.postprocess_kwargs["voice_activity_binary"])
            if not is_onnx:
                out["voice_activity_binary"] = voice_activity
        # do other postprocessing
        for measure in self.model.measures:
            if measure != "voice_activity_binary":
                if not is_onnx:
                    measure_vals = out["measures"][measure]
                else:
                    # use index of measure in self.model.measures
                    measure_vals = out[self.model.measures.index(measure)]
                measure_vals = self.postprocess(measure_vals, voice_activity, **self.postprocess_kwargs[measure])
                if not is_onnx:
                    out[measure] = measure_vals
                else:
                    # use index of measure in self.model.measures
                    out[self.model.measures.index(measure)] = measure_vals
        
        if not is_onnx:
            del out["measures"]
            out["dvector"] = out["dvector"].cpu().numpy()
        if speaker_avatar:
            from scipy import ndimage
            import seaborn as sns
            avatars = []
            for dvec in out["dvector"]:
                # convert to 8 x 8 x 2
                dvec = dvec.reshape(8, 8, 2, 2).mean(axis=-1)
                # normalize
                dvec = dvec / 0.1
                # create 8 x 8 image using the 2 channels as i and q values at y=0.5
                # for this, we add one channel with .5 values
                dvec = np.concatenate([np.ones_like(dvec) * 0.5, dvec], axis=-1)
                dvec = dvec[:, :, [0, 2, 3]]
                # move to correct range for i (-0.5957, 0.5957)
                dvec[:, :, 1] = dvec[:, :, 1] * 0.5957
                # move to correct range for q (-0.5226, 0.5226)
                dvec[:, :, 2] = dvec[:, :, 2] * 0.5226
                # convert to rgb
                dvec = transformYIQ2RGB(dvec)
                # scale up to 256 x 256
                dvec = ndimage.zoom(dvec, (32, 32, 1), order=1)
                # make symmetric
                dvec = np.concatenate([dvec, dvec[:, ::-1]], axis=1)
                dvec = np.concatenate([dvec, dvec[::-1]], axis=0)
                inPalette = np.array(sns.color_palette("hls", 5))
                dvec = QuantizeToGivenPalette(dvec, inPalette)
                # add to list
                avatars.append(dvec)
            out["avatars"] = np.stack(avatars)

        if not is_onnx:
            for measure in out.keys():
                if isinstance(out[measure], torch.Tensor):
                    out[measure] = out[measure].cpu().numpy()
            if "snr" in out and "voice_activity_binary" in out:
                out["overall_snr"] = (out["snr"] * out["voice_activity_binary"]).sum(axis=-1) / out["voice_activity_binary"].sum(axis=-1)
            if "srmr" in out and "voice_activity_binary" in out:
                va_mask = out["voice_activity_binary"] > 0.5
                out["overall_srmr"] = (out["srmr"] * va_mask).sum(axis=-1) / va_mask.sum(axis=-1)
                if np.isnan(out["overall_srmr"]).any():
                    out["overall_srmr"][np.isnan(out["overall_srmr"])] = out["srmr"][np.isnan(out["overall_srmr"])].mean(axis=-1)
        return out