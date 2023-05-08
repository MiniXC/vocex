import argparse
import os
import hashlib
from pathlib import Path

import pickle
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from alignments.datasets.libritts import LibrittsDataset
from tqdm.contrib.concurrent import process_map
from tqdm.auto import tqdm
from multiprocessing import cpu_count
from phones.convert import Converter
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import DataLoader, Dataset

from training.collator import Collator
from training.measures import (
    EnergyMeasure,
    PitchMeasure,
    SNRMeasure,
    SRMRMeasure,
)

_URL = "https://www.openslr.org/resources/60/"
_URLS = {
    "dev-clean": _URL + "dev-clean.tar.gz",
    "dev-other": _URL + "dev-other.tar.gz",
    "test-clean": _URL + "test-clean.tar.gz",
    "test-other": _URL + "test-other.tar.gz",
    "train-clean-100": _URL + "train-clean-100.tar.gz",
    "train-clean-360": _URL + "train-clean-360.tar.gz",
    "train-other-500": _URL + "train-other-500.tar.gz",
}

class DfDataset(Dataset):
    def __init__(self, df, max_lengths, min_size=10_000):
        self.list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="loading data"):
            # 10kB is the minimum size of a wav file for our purposes
            if Path(row["audio"]).stat().st_size >= min_size:
                if len(row["phones"]) < max_lengths["phone"]:
                    result = {
                        "id": row["basename"],
                        "speaker": row["speaker"],
                        "text": row["text"],
                        "start": row["start"],
                        "end": row["end"],
                        "phones": row["phones"],
                        "phone_durations": row["duration"],
                        "audio": str(row["audio"]),
                    }
                    self.list.append(result)
        # convert to dataframe for easier indexing and to avoid "too many open files" error
        self.df = pd.DataFrame(self.list)
        # set datatypes
        self.df["id"] = self.df["id"].apply(lambda x: x.encode("utf-8"))
        self.df["speaker"] = self.df["speaker"].apply(lambda x: x.encode("utf-8"))
        self.df["text"] = self.df["text"].apply(lambda x: x.encode("utf-8"))
        self.df["phones"] = self.df["phones"].apply(lambda x: [p.encode("utf-8") for p in x])
        self.df["phone_durations"] = self.df["phone_durations"].apply(lambda x: [int(d) for d in x])
        self.df["audio"] = self.df["audio"].apply(lambda x: x.encode("utf-8"))
        self.df["start"] = self.df["start"].astype(np.float32)
        self.df["end"] = self.df["end"].astype(np.float32)
        # index
        self.df = self.df.set_index("id")
        self.df = self.df.sort_index()
        self.df = self.df.reset_index()
        del self.list

    def __getitem__(self, idx):
        return self.df.iloc[idx]
        
    def __len__(self):
        return len(self.df)

class LibriTTSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=32,
        measures=[
            EnergyMeasure(),
            PitchMeasure(),
            SRMRMeasure(),
            SNRMeasure(),
        ],
        audio_args={
            "sample_rate": 22050,
            "hop_length": 256,
            "win_length": 1024,
            "n_fft": 1024,
            "n_mels": 80,
        },
        max_lengths={
            "frame": 512,
            "phone": 384,
        },
        min_audio_file_size=10_000,
        seed=42,
        num_workers=0,
        use_cache=True,
        verbose=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = os.path.join(self.data_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.batch_size = batch_size
        self.audio_args = argparse.Namespace(**audio_args)
        self.seed = seed
        self.collator = Collator(
            measures=measures,
            audio_args=audio_args,
            max_lengths=max_lengths,
        )
        self.max_lengths = max_lengths
        self.min_audio_file_size = min_audio_file_size
        self.num_workers = num_workers
        self.use_cache = use_cache
        self.verbose = verbose

    def _create_alignments_ds(self, name, url):
        self.empty_textgrids = 0
        ds_hash = hashlib.md5(os.path.join(self.data_dir, f"{name}-alignments").encode()).hexdigest()
        pkl_path = os.path.join(self.cache_dir, f"{ds_hash}.pkl")
        if os.path.exists(pkl_path) and self.use_cache:
            ds = pickle.load(open(pkl_path, "rb"))
        else:
            tgt_dir = os.path.join(self.data_dir, f"{name}-alignments")
            src_dir = os.path.join(self.data_dir, f"{name}-data")
            if os.path.exists(tgt_dir):
                src_dir = None
                url = None
            elif os.path.exists(src_dir):
                url = None
            ds = LibrittsDataset(
                target_directory=tgt_dir,
                source_directory=src_dir,
                source_url=url,
                verbose=self.verbose,
                tmp_directory=os.path.join(self.data_dir, f"{name}-tmp"),
                chunk_size=1000,
            )
            pickle.dump(ds, open(pkl_path, "wb"))
        return ds, ds_hash

    def _create_data(self, data):
        entries = []
        self.phone_cache = {}
        self.phone_converter = Converter()
        if not isinstance(data, list):
            data = [data]
        hashes = [ds_hash for ds, ds_hash in data]
        ds = [ds for ds, ds_hash in data]
        self.ds = ds
        del data
        for i, ds in enumerate(ds):
            if os.path.exists(os.path.join(self.cache_dir, f"{hashes[i]}-entries.pkl")) and self.use_cache:
                add_entries = pickle.load(open(os.path.join(self.cache_dir, f"{hashes[i]}-entries.pkl"), "rb"))
            else:
                add_entries = [
                    entry
                    for entry in process_map(
                        self._create_entry,
                        zip([i] * len(ds), np.arange(len(ds))),
                        chunksize=10_000,
                        max_workers=cpu_count(),
                        desc=f"processing dataset {hashes[i]}",
                        tqdm_class=tqdm,
                    )
                    if entry is not None
                ]
                pickle.dump(add_entries, open(os.path.join(self.cache_dir, f"{hashes[i]}-entries.pkl"), "wb"))
            entries += add_entries
        if self.empty_textgrids > 0:
            logger.warning(f"Found {self.empty_textgrids} empty textgrids")
        return pd.DataFrame(
            entries,
            columns=[
                "phones",
                "duration",
                "start",
                "end",
                "audio",
                "speaker",
                "text",
                "basename",
            ],
        )
        del self.ds, self.phone_cache, self.phone_converter

    def _create_entry(self, dsi_idx):
        dsi, idx = dsi_idx
        item = self.ds[dsi][idx]
        start, end = item["phones"][0][0], item["phones"][-1][1]
        phones = []
        durations = []
        for i, p in enumerate(item["phones"]):
            s, e, phone = p
            phone.replace("ˌ", "")
            r_phone = phone.replace("0", "").replace("1", "")
            if len(r_phone) > 0:
                phone = r_phone
            if "[" not in phone:
                o_phone = phone
                if o_phone not in self.phone_cache:
                    phone = self.phone_converter(
                        phone, "arpabet", lang=None
                    )[0]
                    self.phone_cache[o_phone] = phone
                phone = self.phone_cache[o_phone]
            phones.append(phone)
            durations.append(
                int(
                    np.round(e * self.audio_args.sample_rate / self.audio_args.hop_length)
                    - np.round(s * self.audio_args.sample_rate / self.audio_args.hop_length)
                )
            )
        if start >= end:
            self.empty_textgrids += 1
            return None
        return (
            phones,
            durations,
            start,
            end,
            item["wav"],
            str(item["speaker"]).split("/")[-1],
            item["transcript"],
            Path(item["wav"]).name,
        )

    def setup(self, stage: str):
        # check if datasets are cached
        cache_paths = [
            os.path.join(self.cache_dir, f"{name}_cache.pkl")
            for name in ["train", "dev", "test"]
        ]
        if all([os.path.exists(path) for path in cache_paths]) and self.use_cache:
            self.data_train, self.data_dev, self.data_test = [
                pickle.load(open(path, "rb")) for path in cache_paths
            ]
            return
        ds_dict = {}
        for name, url in _URLS.items():
            ds_dict[name] = self._create_alignments_ds(name, url)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        cache_dict = {
            "train": os.path.join(self.cache_dir, "train_cache.pkl"),
            "dev": os.path.join(self.cache_dir, "dev_cache.pkl"),
            "test": os.path.join(self.cache_dir, "test_cache.pkl"),
        }
        if os.path.exists(cache_dict["train"]) and self.use_cache:
            self.data_train = pickle.load(open(cache_dict["train"], "rb"))
        else:
            self.data_train = DfDataset(
                self._create_data([ds_dict["train-clean-100"], ds_dict["train-clean-360"], ds_dict["train-other-500"]]),
                max_lengths=self.max_lengths,
                min_size=self.min_audio_file_size,
            )
            pickle.dump(self.data_train, open(cache_dict["train"], "wb"))
        if os.path.exists(cache_dict["dev"]) and self.use_cache:
            self.data_dev = pickle.load(open(cache_dict["dev"], "rb"))
        else:
            self.data_dev = DfDataset(
                self._create_data([ds_dict["dev-clean"], ds_dict["dev-other"]]),
                max_lengths=self.max_lengths,
                min_size=self.min_audio_file_size,
            )
            pickle.dump(self.data_dev, open(cache_dict["dev"], "wb"))
        if os.path.exists(cache_dict["test"]) and self.use_cache:
            self.data_test = pickle.load(open(cache_dict["test"], "rb"))
        else:
            self.data_test = DfDataset(
                self._create_data([ds_dict["test-clean"], ds_dict["test-other"]]),
                max_lengths=self.max_lengths,
                min_size=self.min_audio_file_size,
            )
            pickle.dump(self.data_test, open(cache_dict["test"], "wb"))
        # self.data_all = pd.concat([self.data_train, self.data_dev, self.data_test])

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_dev,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate_fn,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            collate_fn=self.collator.collate_fn,
            num_workers=self.num_workers,
        )

    def teardown(self, stage: str):
        pass