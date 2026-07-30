import os
import glob
import cv2
import numpy as np
import rasterio
from typing import List, Tuple

def load_sentinel_rgb_from_files(b04_path: str, b03_path: str, b02_path: str) -> np.ndarray:
    """Reads Sentinel-2 single bands (Red, Green, Blue) and stacks them into an 8-bit RGB image."""
    # Read raster bands using rasterio
    with rasterio.open(b04_path) as r_band:
        red = r_band.read(1)
    with rasterio.open(b03_path) as g_band:
        green = g_band.read(1)
    with rasterio.open(b02_path) as b_band:
        blue = b_band.read(1)

    # Stack single channels into a 3D float32 array
    rgb_stack = np.dstack((red, green, blue)).astype(np.float32)

    # Contrast stretching using 2nd and 98th percentiles
    p2, p98 = np.percentile(rgb_stack, (2, 98))
    rgb_clipped = np.clip(rgb_stack, p2, p98)

    # Normalize values to 0-255 range and convert to uint8
    rgb_8bit = cv2.normalize(rgb_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return rgb_8bit

def tile_image(image: np.ndarray, tile_size: int = 512, overlap: int = 64) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """Crops a large satellite image into overlapping smaller square patches"""
    h, w = image.shape[:2]
    stride = tile_size - overlap
    tiles = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            patch = image[y:y + tile_size, x:x + tile_size]
            tiles.append((patch, (x, y)))

    return tiles