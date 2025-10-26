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
    Dataset for FIR-decomposed EEG segments.
    Each item returns:
      - tensor shaped (bands, channels, time) with cumulative frequency content
      - dummy label (always 0) to keep interface compatible with conditional VAR
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        recursive: bool = True,
        allowed_extensions: Sequence[str] = ('.npz',),
    ):
        self.root = os.path.abspath(root)
        if split:
            self.data_root = os.path.join(self.root, split)
        else:
            self.data_root = self.root
        pattern = '**/*' if recursive else '*'
        paths: List[str] = []
        for ext in allowed_extensions:
            paths.extend(glob(os.path.join(self.data_root, f'{pattern}{ext}'), recursive=recursive))
        self.paths = sorted(paths)
        if not self.paths:
            raise FileNotFoundError(f'No EEG segment files found under {self.data_root}')
        
        probe = np.load(self.paths[0])
        try:
            _validate_npz_keys(probe, self.paths[0])
        finally:
            probe.close()
        with np.load(self.paths[0]) as probe:
            self.input_shape: Tuple[int, int, int] = tuple(probe['cumulative_bands'].shape)
            self.frequency_bands = probe['bands'].astype(np.float32)
        self.num_classes = 1  # unconditional generation by default
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with np.load(path) as npz:
            _validate_npz_keys(npz, path)
            cumulative = npz['cumulative_bands'].astype(np.float32)  # (bands, channels, time)
        cumulative_tensor = torch.from_numpy(cumulative)
        label_tensor = torch.zeros((), dtype=torch.long)
        return cumulative_tensor, label_tensor.long()


def build_eeg_frequency_datasets(
    data_path: str,
    recursive: bool = True,
) -> Tuple[int, EEGFrequencyDataset, EEGFrequencyDataset, Tuple[int, int, int], np.ndarray]:
    train_set = EEGFrequencyDataset(data_path, split='train', recursive=recursive)
    try:
        val_set = EEGFrequencyDataset(data_path, split='val', recursive=recursive)
    except FileNotFoundError:
        val_set = EEGFrequencyDataset(data_path, split='', recursive=recursive)
    num_classes = train_set.num_classes
    input_shape = train_set.input_shape
    frequency_bands = train_set.frequency_bands
    return num_classes, train_set, val_set, input_shape, frequency_bands
