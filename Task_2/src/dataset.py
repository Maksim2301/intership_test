import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class SatelliteMatchingDataset(Dataset):
    """PyTorch Dataset for pairing cross-seasonal satellite patches."""
    def __init__(self, dir_a: str, dir_b: str, img_size: tuple = (512, 512)):
        self.dir_a = dir_a
        self.dir_b = dir_b
        self.img_size = img_size

        # Identify common filenames present in both seasonal directories
        files_a = set(os.listdir(dir_a))
        files_b = set(os.listdir(dir_b))
        self.filenames = sorted(list(files_a & files_b))

    def __len__(self):
        return len(self.filenames)

    def _read_and_resize(self, path: str) -> np.ndarray:
        """Helper method to load a image in grayscale and resize it."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Unable to open: {path}")
        img = cv2.resize(img, self.img_size)
        return img

    def __getitem__(self, idx):
        """Returns normalized 1-channel image tensors for Season A and Season B."""
        filename = self.filenames[idx]
        path_a = os.path.join(self.dir_a, filename)
        path_b = os.path.join(self.dir_b, filename)

        # Load grayscale arrays
        img0 = self._read_and_resize(path_a)
        img1 = self._read_and_resize(path_b)

        # Convert to PyTorch tensors with shape (1, H, W) and scale to [0, 1]
        tensor0 = torch.from_numpy(img0).float().unsqueeze(0) / 255.0
        tensor1 = torch.from_numpy(img1).float().unsqueeze(0) / 255.0

        return {
            'image0': tensor0,
            'image1': tensor1,
        }