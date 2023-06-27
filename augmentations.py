import numpy as np
import audiomentations as A
import random
import colorednoise as cn
from audiomentations import (
    AddBackgroundNoise, 
    PolarityInversion,
    RoomSimulator,
    BandPassFilter,
    ClippingDistortion,
)
import torchaudio
import torch
import pydub
from pathlib import Path
import os
import warnings
import tempfile

aug_polarity_inversion = PolarityInversion(p=1.0)

SAMPLE_RATE = 22050

def _random_flag(prob):
    return np.random.uniform(0, 1) < prob

def calculate_power(signal):
    return np.sum(np.square(signal)) / len(signal)

def resample(signal):
    target_sampling_rate = random.choices(
        [8000, 16000, 44100], [0.4, 0.4, 0.1]
    )[0]
    resample_algorithm = random.choices(["sinc_interpolation", "kaiser_window"])[0]
    signal = torch.tensor(signal).unsqueeze(0)
    if signal.dtype == torch.float64:
        signal = signal.float()
    signal = torchaudio.transforms.Resample(SAMPLE_RATE, target_sampling_rate, resample_algorithm)(signal)
    signal = torchaudio.transforms.Resample(target_sampling_rate, SAMPLE_RATE, resample_algorithm)(signal)
    signal = signal.squeeze(0).numpy()
    return signal, f"resample_{target_sampling_rate//1000}k_{resample_algorithm}"

def combine_signal_and_noise(signal, noise, desired_snr):
    signal_power = calculate_power(signal)
    noise_power = calculate_power(noise)
    noise_scaling_factor = np.sqrt(signal_power / (noise_power * (10 ** (desired_snr / 10))))
    return signal + noise * noise_scaling_factor

def add_colored_noise(signal):
    noise_type = random.choices(["white", "pink", "brown"], [0.5, 0.25, 0.25])[0]
    if noise_type == "white":
        noise = cn.powerlaw_psd_gaussian(1, signal.shape)
    elif noise_type == "pink":
        noise = cn.powerlaw_psd_gaussian(2, signal.shape)
    elif noise_type == "brown":
        noise = cn.powerlaw_psd_gaussian(3, signal.shape)
    else:
        raise ValueError(f"Unknown noise type {noise_type}!")
    snr = random.choices(["clean", "low", "medium", "high", "very_high"], [0.15, 0.3, 0.35, 0.2, 0.2])[0]
    if snr == "very_high" and noise_type == "white":
        snr = "high"
    if snr == "clean":
        snr_val = np.random.uniform(20, 30)
    elif snr == "low":
        snr_val = np.random.uniform(10, 20)
    elif snr == "medium":
        snr_val = np.random.uniform(0, 10)
    elif snr == "high":
        snr_val = np.random.uniform(-10, 0)
    elif snr == "very_high":
        snr_val = np.random.uniform(-20, -10)
    return combine_signal_and_noise(signal, noise, snr_val), f"{noise_type}_noise_{snr}"

reverb_rooms = {
    "small_absorbent": RoomSimulator(
        min_target_rt60=0.1,
        max_target_rt60=0.3,
        min_absorption_value=0.3,
        max_absorption_value=0.4,
        leave_length_unchanged=True,
        use_ray_tracing=False,
        p=1.0,
    ),
    "small_reflective": RoomSimulator(
        min_target_rt60=0.1,
        max_target_rt60=0.3,
        min_absorption_value=0.1,
        max_absorption_value=0.1,
        leave_length_unchanged=True,
        use_ray_tracing=False,
        p=1.0,
    ),
    "medium_absorbent": RoomSimulator(
        min_target_rt60=0.3,
        max_target_rt60=1,
        min_absorption_value=0.3,
        max_absorption_value=0.4,
        leave_length_unchanged=True,
        use_ray_tracing=False,
        p=1.0,
    ),
    "medium_reflective": RoomSimulator(
        min_target_rt60=0.3,
        max_target_rt60=1,
        min_absorption_value=0.1,
        max_absorption_value=0.1,
        leave_length_unchanged=True,
        use_ray_tracing=False,
        p=1.0,
    ),
    "large_absorbent": RoomSimulator(
        min_target_rt60=1,
        max_target_rt60=1.2,
        min_absorption_value=0.3,
        max_absorption_value=0.4,
        leave_length_unchanged=True,
        use_ray_tracing=False,
        p=1.0,
    ),
    "large_reflective": RoomSimulator(
        min_target_rt60=1,
        max_target_rt60=1.2,
        min_absorption_value=0.0,
        max_absorption_value=0.0,
        leave_length_unchanged=True,
        use_ray_tracing=False,
        p=1.0,
    ),
}

def add_reverberation(signal):
    room = random.choices([
        "small_absorbent",
        "small_reflective",
        "medium_absorbent",
        "medium_reflective",
        "large_absorbent",
        "large_reflective",
    ], [0.2, 0.2, 0.2, 0.2, 0.15, 0.05])[0]
    return reverb_rooms[room](samples=signal, sample_rate=SAMPLE_RATE), f"reverb_{room}"

def aug_add_colored_noise(wave, _):
    if _random_flag(0.5):
        return wave
    wave, _ = add_colored_noise(wave)
    return wave

aug_bandpass = BandPassFilter(
    min_center_freq=300,
    max_center_freq=4000,
    p=1.0
)

def add_telephone_or_radio(wave):
    wave = aug_bandpass(wave, sample_rate=SAMPLE_RATE)
    return wave, "telephone"

aug_clip = ClippingDistortion(
    min_percentile_threshold=10,
    max_percentile_threshold=20,
    p=1.0
)

def add_clipping(wave):
    return aug_clip(wave, sample_rate=SAMPLE_RATE), "clipping"

aug_background_noise = AddBackgroundNoise(
    sounds_path="/home/christoph.minixhofer/ESC-50-master/audio_22k",
    min_snr_in_db=0,
    max_snr_in_db=30.0,
    noise_transform=aug_add_colored_noise,
    p=1.0
)

def add_codecs(wave):
    codec = random.choices(["mp3", "ogg", "flac", "adts"], [0.25, 0.25, 0.25, 0.25])[0]
    bitrate = random.choices([16, 32, 64, 128, 320], [0.2, 0.2, 0.2, 0.2, 0.2])[0]
    if not Path("/tmp/codecs").exists():
        Path("/tmp/codecs").mkdir()
    tmp_file = tempfile.mktemp(dir="/tmp/codecs", suffix=".wav")
    if wave.dtype == np.float64:
        wave = wave.astype(np.float32)
    torchaudio.save(tmp_file, torch.tensor(wave).unsqueeze(0), SAMPLE_RATE)
    # remove bitrate if not applicable
    if codec in ["adts", "ogg"]:
        bitrate = None
    else:
        bitrate = f"{bitrate}k"
    pydub.AudioSegment.from_wav(tmp_file).export(tmp_file, format=codec, bitrate=bitrate)
    new_wave = np.array(pydub.AudioSegment.from_file(tmp_file).get_array_of_samples())
    # remove wav file
    os.remove(tmp_file)
    if new_wave.shape != wave.shape:
        # interpolate
        new_wave = np.interp(
            np.linspace(0, 1, len(wave)), 
            np.linspace(0, 1, len(new_wave)), 
            new_wave
        )
    assert new_wave.shape == wave.shape
    if bitrate is not None:
        return new_wave, f"codec_{codec}_{bitrate}"
    return new_wave, f"codec_{codec}"

def wave_augmentation_func(wave, return_name=False):
    old_shape = wave.shape
    augmentations = []
    # probabilties = {
    #     "polarity_inversion": 0.1,
    #     "real_noise": 0.2,
    #     "reverb": 0.1,
    #     "resample": 0.1,
    #     "telephone": 0.1,
    #     "colored_noise": 0.1,
    #     "clipping": 0.05,
    #     "codecs": 0.1,
    # }
    probabilties = {
        "polarity_inversion": 0.1,
        "real_noise": 0.2,
        "reverb": 0.2,
        "resample": 0.0, # untested (uses torchaudio, so could be slow)
        "telephone": 0.2,
        "colored_noise": 0.2,
        "clipping": 0.1,
        "codecs": 0.0, # untested (uses ffmpeg, could be problematic with multiple workers)
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # polarity inversion
        if _random_flag(probabilties["polarity_inversion"]):
            wave = aug_polarity_inversion(wave, sample_rate=SAMPLE_RATE)
        # real noise
        if _random_flag(probabilties["real_noise"]/2):
            wave = aug_background_noise(wave, sample_rate=SAMPLE_RATE)
            if wave.dtype == np.float64:
                wave = wave.astype(np.float32)
            augmentations.append("real_noise")
        # reverberation
        if _random_flag(probabilties["reverb"]):
            wave, aug = add_reverberation(wave)
            augmentations.append(aug)
        # real noise
        if _random_flag(probabilties["real_noise"]/2) and "real_noise" not in augmentations:
            wave = aug_background_noise(wave, sample_rate=SAMPLE_RATE)
            augmentations.append("real_noise")
        # resample
        if _random_flag(probabilties["resample"]):
            wave, aug = resample(wave)
            augmentations.append(aug)
        # telephone or radio
        if _random_flag(probabilties["telephone"]):
            wave, aug = add_telephone_or_radio(wave)
            augmentations.append(aug)
        # additive noise
        if _random_flag(probabilties["colored_noise"]):
            wave, aug = add_colored_noise(wave)
            augmentations.append(aug)
        # clipping
        if _random_flag(probabilties["clipping"]):
            wave, aug = add_clipping(wave)
            augmentations.append(aug)
        # codecs
        if _random_flag(probabilties["codecs"]):
            wave, aug = add_codecs(wave)
            augmentations.append(aug)
    if len(augmentations) == 0:
        augmentations.append("none")

    if old_shape != wave.shape:
        # interpolate
        wave = np.interp(
            np.linspace(0, 1, old_shape[0]), 
            np.linspace(0, 1, wave.shape[0]), 
            wave
        )

    if wave.dtype == np.float64:
        wave = wave.astype(np.float32)

    if wave.dtype == np.int16:
        wave = wave.astype(np.float32)
    
    wave = wave / (np.max(np.abs(wave)) + 1e-9)

    if wave.dtype == np.float64:
        wave = wave.astype(np.float32)

    if return_name:
        return wave, "|".join(augmentations)
    return wave

def aug_mask_frequncy_bands(mel):
    mask_size = random.choices(["small", "medium", "large"], [0.2, 0.6, 0.2])[0]
    if mask_size == "small":
        num_bands = random.randint(1, 5)
    elif mask_size == "medium":
        num_bands = random.randint(5, 10)
    elif mask_size == "large":
        num_bands = random.randint(10, 20)
    start = random.randint(0, mel.shape[0]-num_bands)
    end = start + num_bands
    mel_std = torch.std(mel)
    mel_mean = torch.mean(mel)
    mel[start:end] = torch.randn_like(mel[start:end]) * mel_std + mel_mean
    return mel, f"mask_frequncy_bands_{mask_size}"

def mel_augmentation_func(mel, return_name=False):
    augmentations = []
    probabilties = {
        "mask_frequncy_bands": 1,
    }
    # mask frequncy bands
    if _random_flag(probabilties["mask_frequncy_bands"]):
        mel, aug = aug_mask_frequncy_bands(mel)
        augmentations.append(aug)
    if len(augmentations) == 0:
        augmentations.append("none")
    if return_name:
        return mel, "|".join(augmentations)
    return mel

# from datasets import load_dataset

# dataset = load_dataset("/home/christoph.minixhofer/libritts-r-aligned/libritts-r-aligned.py", split="dev")

# for item in dataset:
#     for i in range(1000):
#         wave, sr = torchaudio.load(item["audio"])
#         wave_shape = wave.shape
#         wave = wave[0].numpy()
#         old_wave = wave
#         wave, aug = wave_augmentation_func(wave, return_name=True)
#         print(wave.dtype, np.max(np.abs(wave)).dtype, aug)
#         wave = torch.tensor(wave).unsqueeze(0)
#         if wave_shape != wave.shape:
#             print("Shape changed!")
#             print(aug, wave_shape, wave.shape)
#         torchaudio.save(f"test/augs/{aug}.wav", wave, sr)
#     break

# test mel augmentation
# import matplotlib.pyplot as plt
# import librosa

# for item in dataset:
#     for i in range(10):
#         wave, sr = torchaudio.load(item["audio"])
#         wave_shape = wave.shape
#         wave = wave[0].numpy()
#         mel = librosa.feature.melspectrogram(wave, sr=sr, n_fft=1024, hop_length=256, n_mels=80, fmin=0, fmax=8000)
#         mel = np.log(mel + 1e-9)
#         mel_shape = mel.shape
#         mel = mel_augmentation_func(mel)
#         print(mel_shape, mel.shape)
#         plt.imshow(mel, origin="lower")
#         plt.savefig(f"test/augs/mel_{i}.png")
#         break
#     break