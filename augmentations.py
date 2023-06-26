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

aug_polarity_inversion = PolarityInversion(p=1.0)

SAMPLE_RATE = 22050

def _random_flag(prob):
    return np.random.uniform(0, 1) < prob

def calculate_power(signal):
    return np.sum(np.square(signal)) / len(signal)

def resample(signal):
    target_sampling_rate = random.choices(
        [8000, 16000, 22050, 44100], [0.1, 0.4, 0.4, 0.1]
    )[0]
    resample_algorithm = random.choices(["sinc_interp_hann", "sinc_interp_kaiser"])[0]
    signal = torchaudio.transforms.Resample(SAMPLE_RATE, target_sampling_rate, resample_algorithm)(signal)
    return signal, f"resample_{target_sampling_rate//1000}k_{resample_algorithm.split('_')[-1]}"

def combine_signal_and_noise(signal, noise, desired_snr):
    # Calculate power of the signal and noise
    signal_power = calculate_power(signal)
    noise_power = calculate_power(noise)

    # Calculate current SNR
    current_snr = 10 * np.log10(signal_power / noise_power)

    # Calculate required power of noise to achieve desired SNR
    required_noise_power = signal_power / (10 ** (desired_snr / 10))

    # Calculate the scaling factor for the noise
    noise_scaling_factor = np.sqrt(required_noise_power / noise_power)

    # Scale the noise and add it to the signal
    return signal + noise_scaling_factor * noise

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
    snr = random.choices(["clean", "low", "medium", "high"], [0.15, 0.3, 0.35, 0.2])[0]
    if snr == "clean":
        snr_val = np.random.uniform(20, 30)
    elif snr == "low":
        snr_val = np.random.uniform(10, 20)
    elif snr == "medium":
        snr_val = np.random.uniform(0, 10)
    elif snr == "high":
        snr_val = np.random.uniform(-10, 0)
    return combine_signal_and_noise(signal, noise, snr_val), f"{noise_type}_noise_{snr}"

reverb_rooms = {
    "small_absorbent": RoomSimulator(
        min_target_rt60=0.1,
        max_target_rt60=0.3,
        min_absorption_value=0.3,
        max_absorption_value=0.4,
        leave_length_unchanged=True,
        use_ray_tracing=True,
        p=1.0,
    ),
    "small_reflective": RoomSimulator(
        min_target_rt60=0.1,
        max_target_rt60=0.3,
        min_absorption_value=0.1,
        max_absorption_value=0.1,
        leave_length_unchanged=True,
        use_ray_tracing=True,
        p=1.0,
    ),
    "medium_absorbent": RoomSimulator(
        min_target_rt60=0.3,
        max_target_rt60=1,
        min_absorption_value=0.3,
        max_absorption_value=0.4,
        leave_length_unchanged=True,
        use_ray_tracing=True,
        p=1.0,
    ),
    "medium_reflective": RoomSimulator(
        min_target_rt60=0.3,
        max_target_rt60=1,
        min_absorption_value=0.1,
        max_absorption_value=0.1,
        leave_length_unchanged=True,
        use_ray_tracing=True,
        p=1.0,
    ),
    "large_absorbent": RoomSimulator(
        min_target_rt60=1,
        max_target_rt60=2,
        min_absorption_value=0.3,
        max_absorption_value=0.4,
        leave_length_unchanged=True,
        use_ray_tracing=True,
        p=1.0,
    ),
    "large_reflective": RoomSimulator(
        min_target_rt60=1,
        max_target_rt60=2,
        min_absorption_value=0.0,
        max_absorption_value=0.0,
        leave_length_unchanged=True,
        use_ray_tracing=True,
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
    sounds_path="/home/christoph.minixhofer/ESC-50-master/audio",
    min_snr_in_db=0,
    max_snr_in_db=30.0,
    noise_transform=aug_add_colored_noise,
    p=1.0
)

def add_codecs(wave):
    codec = random.choices(["mp3", "ogg", "flac"], [0.25, 0.25, 0.25])[0]
    bitrate = random.choices([64, 128, 192, 256, 320], [0.2, 0.2, 0.2, 0.2, 0.2])[0]
    random_hash = random.randint(0, 100000)
    if not Path("/tmp/codecs").exists():
        Path("/tmp/codecs").mkdir()
    tmp_file = f"/tmp/codecs/{random_hash}.wav"
    torchaudio.save(tmp_file, torch.tensor(wave).unsqueeze(0), SAMPLE_RATE)
    pydub.AudioSegment.from_wav(tmp_file).export(tmp_file, format=codec, bitrate=f"{bitrate}k")
    new_wave = np.array(pydub.AudioSegment.from_file(tmp_file).array)
    assert new_wave.shape == wave.shape
    return new_wave, f"codec_{codec}"

def wave_augmentation_func(wave):
    augmentations = []
    probabilties = {
        "polarity_inversion": 0.1,
        "real_noise": 0.2,
        "reverb": 1,
        "resample": 0.1,
        "telephone": 0.1,
        "colored_noise": 0.1,
        "clipping": 0.05,
        "codecs": 0.1,
    }
    # polarity inversion
    if _random_flag(probabilties["polarity_inversion"]):
        wave = aug_polarity_inversion(wave, sample_rate=SAMPLE_RATE)
    # real noise
    if _random_flag(probabilties["real_noise"]/2):
        wave = aug_background_noise(wave, sample_rate=SAMPLE_RATE)
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
    return wave, "|".join(augmentations)

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
    mel[start:end] = 0.0
    return mel, f"mask_frequncy_bands_{mask_size}"


def mel_augmentation_func(mel):
    return mel

from datasets import load_dataset

dataset = load_dataset("/home/christoph.minixhofer/libritts-r-aligned/libritts-r-aligned.py", split="dev")

for item in dataset:
    wave, sr = torchaudio.load(item["audio"])
    wave = wave[0].numpy()
    wave, aug = wave_augmentation_func(wave)
    torchaudio.save(f"test/augs/{aug}.wav", torch.tensor(wave).unsqueeze(0), sr)
    break