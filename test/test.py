from pathlib import Path

from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
import pytest
from scipy.stats import pearsonr
from tqdm.auto import tqdm

from vocex import Vocex
from test.wada_snr import wada_snr
from srmrpy import srmr

# def main_plot():
#     # load gigaspeech validation set in streaming mode
#     dataset = load_dataset("speechcolab/gigaspeech", "xs", use_auth_token=True, streaming=True, split="validation")

#     # load vocex model
#     vocex = Vocex.from_pretrained("models/checkpoint_half.ckpt")

#     # get the first example
#     example = next(iter(dataset))

#     # get the audio
#     audio = example["audio"]["array"]

#     # get the sample rate
#     sr = example["audio"]["sampling_rate"]

#     # run inference
#     output = vocex(audio, sr, return_activations=True, return_attention=True)

#     # plot the output
#     Path("plots").mkdir(exist_ok=True)
#     for measure in output.keys():
#         if measure not in ["dvector", "activations", "attention"]:
#             plt.figure(figsize=(20, 5))
#             mel = vocex._preprocess(audio, sr)[0]
#             plt.imshow(mel.T, aspect="auto", origin="lower", cmap="magma", alpha=0.5)
#             plt.twinx()
#             values = output[measure][0]
#             sns.lineplot(x=range(len(values)), y=values)
#             plt.title(measure)
#             plt.tight_layout()
#             plt.xlim(0, len(values))
#             plt.ylim(0, values.max()*1.5)
#             plt.savefig(f"plots/{measure}.png")
#             plt.close()

#     # test batched inference
#     batch_size = 4
#     audio_list = []
#     for item in dataset:
#         audio_list.append(item["audio"]["array"])
#         if len(audio_list) == batch_size:
#             break

#     output = vocex(audio_list, sr, return_activations=True, return_attention=True)

#     def collate_fn(batch):
#         # pad audio to the same length
#         lengths = [len(item["audio"]["array"]) for item in batch]
#         max_len = max(lengths)
#         for item in batch:
#             item["audio"]["array"] = np.pad(item["audio"]["array"], (0, max_len - len(item["audio"]["array"])))
#         return {
#             "audio": torch.tensor([item["audio"]["array"] for item in batch]),
#             "lengths": torch.tensor(lengths),
#             "sampling_rate": batch[0]["audio"]["sampling_rate"]
#         }

#     dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

#     num_samples = 0
#     max_num_samples = 1000
#     for batch in dataloader:
#         output = vocex(batch["audio"], batch["sampling_rate"])
#         print(output["snr"].mean(dim=1))
#         for i, item in enumerate(batch["audio"]):
#             item = item[:batch["lengths"][i]]
#             print(wada_snr(item))
#         num_samples += batch_size
#         if num_samples >= max_num_samples:
#             break
    

# if __name__ == "__main__":
#     main_plot()

# tests
@pytest.fixture
def setup_data_and_model():
    dataset = load_dataset("speechcolab/gigaspeech", "xs", use_auth_token=True, streaming=True, split="validation")
    vocex = Vocex.from_pretrained("models/checkpoint_half.ckpt")
    def collate_fn(batch):
        # pad audio to the same length
        lengths = [len(item["audio"]["array"]) for item in batch]
        max_len = max(lengths)
        for item in batch:
            item["audio"]["array"] = np.pad(item["audio"]["array"], (0, max_len - len(item["audio"]["array"])))
        return {
            "audio": torch.tensor([item["audio"]["array"] for item in batch]),
            "lengths": torch.tensor(lengths),
            "sampling_rate": batch[0]["audio"]["sampling_rate"]
        }
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    return dataset, dataloader, vocex

def test_single_inference(setup_data_and_model):
    dataset, _, vocex = setup_data_and_model
    example = next(iter(dataset))
    audio = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    output = vocex(audio, sr, return_activations=True, return_attention=True)
    assert output["energy"].shape == (1,712)
    assert output["snr"].shape == (1,712)
    assert output["pitch"].shape == (1,712)
    assert output["dvector"].shape == (1,256)
    assert output["activations"].shape == (1,vocex.model.hparams["measure_nlayers"],712,256)
    assert output["attention"].shape == (1,vocex.model.hparams["measure_nlayers"],712,712)

def test_single_inference_plot(setup_data_and_model):
    dataset, _, vocex = setup_data_and_model
    example = next(iter(dataset))
    audio = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    output = vocex(audio, sr, return_activations=True, return_attention=True)
    Path("plots").mkdir(exist_ok=True)
    for measure in output.keys():
        if measure not in ["dvector", "activations", "attention", "overall_snr", "overall_srmr"]:
            plt.figure(figsize=(20, 5))
            mel = vocex._preprocess(audio, sr)[0]
            plt.imshow(mel.T, aspect="auto", origin="lower", cmap="magma", alpha=0.5)
            plt.twinx()
            values = output[measure][0]
            sns.lineplot(x=range(len(values)), y=values)
            plt.title(measure)
            plt.tight_layout()
            plt.xlim(0, len(values))
            plt.ylim(0, values.max()*1.5)
            plt.savefig(f"plots/{measure}.png")
            plt.close()
    assert True

def test_batched_inference(setup_data_and_model):
    dataset, _, vocex = setup_data_and_model
    batch_size = 4
    audio_list = []
    sr = None
    for item in dataset:
        audio_list.append(item["audio"]["array"])
        sr = item["audio"]["sampling_rate"]
        if len(audio_list) == batch_size:
            break
    vocex.model.noise_factor = 0.0
    output = vocex(audio_list, sr, return_activations=True, return_attention=True)
    output_single = vocex(audio_list[0], sr, return_activations=True, return_attention=True)
    assert np.allclose(output["energy"][0]/output["energy"][0].max(), output_single["energy"][0]/output_single["energy"][0].max(), atol=1e-2)
    assert np.allclose(output["snr"][0]/output["snr"][0].max(), output_single["snr"][0]/output_single["snr"][0].max(), atol=1e-2)
    assert np.allclose(output["pitch"][0]/output["pitch"][0].max(), output_single["pitch"][0]/output_single["pitch"][0].max(), atol=1e-2)
    assert np.allclose(output["dvector"][0]/output["dvector"][0].max(), output_single["dvector"][0]/output_single["dvector"][0].max(), atol=1e-2)
    assert output["energy"].shape == (4,712)
    assert output["snr"].shape == (4,712)
    assert output["pitch"].shape == (4,712)
    assert output["dvector"].shape == (4,256)
    assert output["activations"].shape == (4,vocex.model.hparams["measure_nlayers"],712,256)
    assert output["attention"].shape == (4,vocex.model.hparams["measure_nlayers"],712,712)

def test_snr_correlated(setup_data_and_model):
    _, dataloader, vocex = setup_data_and_model
    num_samples = 0
    max_num_samples = 100
    snr_list = []
    wada_snr_list = []
    for batch in tqdm(dataloader, total=max_num_samples//4, desc="Testing SNR"):
        output = vocex(batch["audio"], batch["sampling_rate"])
        for i, item in enumerate(batch["audio"]):
            item = item[:batch["lengths"][i]]
            snr_list.append(output["overall_snr"][i])
            wada_snr_list.append(wada_snr(item))    
        num_samples += batch["audio"].shape[0]
        if num_samples >= max_num_samples:
            break
    plt.figure(figsize=(5,5))
    sns.scatterplot(x=snr_list, y=wada_snr_list)
    plt.xlabel("Vocex SNR")
    plt.ylabel("WADA SNR")
    plt.tight_layout()
    plt.savefig("plots/snr_correlation.png")
    corr = pearsonr(snr_list, wada_snr_list)
    print(f"SNR correlation: {corr[0]:.4f} (p={corr[1]:.4f})")
    assert corr[0] > 0.2
    assert corr[1] < 0.05

def test_srmr_correlated(setup_data_and_model):
    _, dataloader, vocex = setup_data_and_model
    num_samples = 0
    max_num_samples = 100
    srmr_list = []
    srmrpy_list = []
    for batch in tqdm(dataloader, total=max_num_samples//4, desc="Testing SRMR"):
        output = vocex(batch["audio"], batch["sampling_rate"])
        for i, item in enumerate(batch["audio"]):
            item = item[:batch["lengths"][i]]
            try:
                srmrpy_list.append(srmr(item.numpy(), batch["sampling_rate"])[0])
                srmr_list.append(output["overall_srmr"][i])
            except ValueError:
                print(f"SRMRpy failed on {i}, model output: {output['overall_srmr'][i]}")
                pass
        num_samples += batch["audio"].shape[0]
        if num_samples >= max_num_samples:
            break
    plt.figure(figsize=(5,5))
    sns.scatterplot(x=srmr_list, y=srmrpy_list)
    plt.xlabel("Vocex SRMR")
    plt.ylabel("SRMRpy SRMR")
    plt.tight_layout()
    plt.savefig("plots/srmr_correlation.png")
    corr = pearsonr(srmr_list, srmrpy_list)
    print(f"SRMR correlation: {corr[0]:.4f} (p={corr[1]:.4f})")
    assert corr[0] > 0.2
    assert corr[1] < 0.05