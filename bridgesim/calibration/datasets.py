"""
Dataset utilities for Bench2Drive/MetaDrive UniAD image pairs.

Pairs are produced by build_bench2drive_metadrive_pairs.py. This module mirrors
the preprocessing strategy from precompute_uniad_image_features.py so you can
load, preprocess, and collate pairs on the fly (no feature caching).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import json
import random

import cv2
import numpy as np
from mmcv.parallel.collate import collate as mm_collate_to_batch_form
from torch.utils.data import Dataset


@dataclass
class PairEntry:
    scenario: str
    frame: str
    camera: str
    bench2drive_path: Path
    metadrive_path: Path


def load_pairs(pairs_path: Path) -> List[PairEntry]:
    """
    Load image pairs from a JSONL produced by build_bench2drive_metadrive_pairs.py.
    """
    entries: List[PairEntry] = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            scenario = payload.get("scenario") or payload.get("scenario_name")
            if not scenario:
                raise ValueError(f"Missing 'scenario' key on line {line_no}")
            pairs = payload.get("pairs") or []
            for pair in pairs:
                try:
                    entries.append(
                        PairEntry(
                            scenario=scenario,
                            frame=str(pair["frame"]),
                            camera=str(pair["camera"]),
                            bench2drive_path=Path(pair["bench2drive_path"]),
                            metadrive_path=Path(pair["metadrive_path"]),
                        )
                    )
                except KeyError as exc:
                    raise ValueError(f"Missing field {exc} in pair on line {line_no}") from exc
    return entries


def train_test_split(
    entries: List[PairEntry], train_ratio: float = 0.8, seed: int = 42
) -> tuple[List[PairEntry], List[PairEntry]]:
    """Deterministic train/test split across the flat pair list."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    idxs = list(range(len(entries)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    cut = int(len(entries) * train_ratio)
    train = [entries[i] for i in idxs[:cut]]
    test = [entries[i] for i in idxs[cut:]]
    return train, test


def build_dummy_meta() -> Dict[str, object]:
    """
    Provide minimal metadata keys that UniAD's inference-only pipeline expects.
    """
    return {
        "timestamp": 0.0,
        "l2g_r_mat": np.eye(3, dtype=np.float32),
        "l2g_t": np.zeros(3, dtype=np.float32),
        "command": 0,
    }


def _default_loader(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


class UniADPairDataset(Dataset):
    """
    Torch dataset that yields processed Bench2Drive/MetaDrive image pairs.

    Each item is a tuple of (PairEntry, metadrive_sample, bench2drive_sample),
    where the samples are the outputs of the provided `pipeline` (typically
    the inference-only Compose pipeline built for UniAD). Items with missing
    images are skipped by returning None; use `collate_uniad_pairs` to drop them.
    """

    def __init__(
        self,
        pairs: Path | List[PairEntry],
        pipeline: Callable[[Dict], Dict],
        *,
        split: str = "all",
        train_ratio: float = 0.8,
        split_seed: int = 42,
        loader: Callable[[Path], Optional[np.ndarray]] = _default_loader,
    ) -> None:
        self.entries = load_pairs(Path(pairs)) if isinstance(pairs, (str, Path)) else list(pairs)
        if split not in {"all", "train", "test", "val"}:
            raise ValueError(f"split must be one of ['all','train','test','val'], got {split}")
        if split != "all":
            train_entries, test_entries = train_test_split(self.entries, train_ratio=train_ratio, seed=split_seed)
            if split == "train":
                self.entries = train_entries
            else:
                self.entries = test_entries
        self.pipeline = pipeline
        self.loader = loader

    def __len__(self) -> int:
        return len(self.entries)

    def _process_image(self, img_path: Path) -> Optional[Dict]:
        img = self.loader(img_path)
        if img is None:
            return None
        sample = build_dummy_meta()
        sample["img"] = [img]
        return self.pipeline(sample)

    def __getitem__(self, idx: int):
        pair = self.entries[idx]
        if not pair.metadrive_path.is_file() or not pair.bench2drive_path.is_file():
            return None
        md_proc = self._process_image(pair.metadrive_path)
        b2d_proc = self._process_image(pair.bench2drive_path)
        if md_proc is None or b2d_proc is None:
            return None
        return pair, md_proc, b2d_proc


def collate_uniad_pairs(batch_items: List):
    """
    Collate function that filters out None items and applies mmcv collate.
    """
    batch_items = [b for b in batch_items if b is not None]
    if not batch_items:
        return None
    pairs_batch, md_samples, b2d_samples = zip(*batch_items)
    md_batch = mm_collate_to_batch_form(md_samples, samples_per_gpu=len(md_samples))
    b2d_batch = mm_collate_to_batch_form(b2d_samples, samples_per_gpu=len(b2d_samples))
    return list(pairs_batch), md_batch, b2d_batch


def build_train_val_datasets(
    pairs: Path | List[PairEntry],
    pipeline: Callable[[Dict], Dict],
    *,
    train_ratio: float = 0.8,
    split_seed: int = 42,
    loader: Callable[[Path], Optional[np.ndarray]] = _default_loader,
) -> Tuple[UniADPairDataset, UniADPairDataset]:
    """
    Convenience helper to create (train, val) datasets with a single split pass.
    """
    entries = load_pairs(Path(pairs)) if isinstance(pairs, (str, Path)) else list(pairs)
    train_entries, val_entries = train_test_split(entries, train_ratio=train_ratio, seed=split_seed)

    train_ds = UniADPairDataset(pairs=train_entries, pipeline=pipeline, split="all", loader=loader)
    val_ds = UniADPairDataset(pairs=val_entries, pipeline=pipeline, split="all", loader=loader)
    return train_ds, val_ds


class UniADPairDataModule:
    """
    Simple factory to build and reuse train/val UniAD pair datasets.
    Mirrors the DatasetFactory pattern used elsewhere in training code.
    """

    def __init__(
        self,
        pairs: Path | List[PairEntry],
        pipeline: Callable[[Dict], Dict],
        *,
        train_ratio: float = 0.8,
        split_seed: int = 42,
        loader: Callable[[Path], Optional[np.ndarray]] = _default_loader,
    ) -> None:
        entries = load_pairs(Path(pairs)) if isinstance(pairs, (str, Path)) else list(pairs)
        train_entries, val_entries = train_test_split(entries, train_ratio=train_ratio, seed=split_seed)
        self.train_dataset = UniADPairDataset(train_entries, pipeline=pipeline, loader=loader, split="all")
        self.val_dataset = UniADPairDataset(val_entries, pipeline=pipeline, loader=loader, split="all")

    def get_split(self, split: str) -> UniADPairDataset:
        if split == "train":
            return self.train_dataset
        if split in {"val", "test"}:
            return self.val_dataset
        if split == "all":
            # Concatenate-style view by recreating a dataset over both splits.
            combined_entries = list(self.train_dataset.entries) + list(self.val_dataset.entries)
            return UniADPairDataset(combined_entries, pipeline=self.train_dataset.pipeline, split="all", loader=self.train_dataset.loader)  # type: ignore[arg-type]
        raise ValueError(f"Unsupported split: {split}")
