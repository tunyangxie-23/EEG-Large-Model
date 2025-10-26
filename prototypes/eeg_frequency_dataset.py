"""
Prototype utilities for EEG frequency-band autoregressive experiments.
"""
import os
from glob import glob
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _validate_npz_keys(npz: np.lib.npyio.NpzFile, path: str):
    if 'cumulative_bands' not in npz or npz['cumulative_bands'].ndim != 3:
        raise ValueError(f'Expected `cumulative_bands` with shape (bands, channels, time) in {path}')
    if 'bands' not in npz:
        raise ValueError(f'Expected `bands` frequencies metadata in {path}')
    if 'signal' not in npz:
        raise ValueError(f'Expected `signal` entry (channels, time) in {path}')


class EEGFrequencyDataset(Dataset):
    """
    Dataset for FIR-decomposed EEG segments saved as .npz files.

    Each sample returns:
        cumulative_bands: Tensor[float32] shaped (bands, channels, time)
        label: LongTensor scalar (always 0 placeholder)
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        recursive: bool = True,
        extensions: Sequence[str] = ('.npz',),
    ):
        self.root = os.path.abspath(root)
        self.data_dir = os.path.join(self.root, split) if split else self.root
        pattern = '**/*' if recursive else '*'
        files: List[str] = []
        for ext in extensions:
            files.extend(glob(os.path.join(self.data_dir, f'{pattern}{ext}'), recursive=recursive))
        self.paths = sorted(files)
        if not self.paths:
            raise FileNotFoundError(f'No files found at {self.data_dir}')
        
        with np.load(self.paths[0]) as probe:
            _validate_npz_keys(probe, self.paths[0])
            self.input_shape: Tuple[int, int, int] = tuple(probe['cumulative_bands'].shape)
            self.bands = probe['bands'].astype(np.float32)
        self.num_classes = 1  # unconditional prototype
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with np.load(path) as npz:
            _validate_npz_keys(npz, path)
            cumulative = torch.from_numpy(npz['cumulative_bands'].astype(np.float32))
        label = torch.zeros((), dtype=torch.long)
        return cumulative, label


def build_datasets(root: str, recursive: bool = True):
    train = EEGFrequencyDataset(root, split='train', recursive=recursive)
    try:
        val = EEGFrequencyDataset(root, split='val', recursive=recursive)
    except FileNotFoundError:
        val = EEGFrequencyDataset(root, split='', recursive=recursive)
    return dict(
        num_classes=train.num_classes,
        train=train,
        val=val,
        input_shape=train.input_shape,
        bands=train.bands,
    )
