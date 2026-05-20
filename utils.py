import math
import os
from typing import List, Tuple

import numpy as np
from PIL import Image


def ensure_dir(path: str) -> None:
    # If path is empty (e.g. "visualisation.png" in current folder), no directory to create.
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def load_frames(frames_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    supported = {".png", ".jpg", ".jpeg", ".bmp"}
    names = sorted(
        f for f in os.listdir(frames_dir) if os.path.splitext(f.lower())[1] in supported
    )
    frames = []
    for name in names:
        p = os.path.join(frames_dir, name)
        img = Image.open(p).convert("RGB")
        frames.append(np.asarray(img, dtype=np.float32))
    return frames, names


def save_rgb_frame(rgb: np.ndarray, path: str) -> None:
    img = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    out = np.stack([y, cb, cr], axis=-1)
    return out.astype(np.float32)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1] - 128.0
    cr = ycbcr[..., 2] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    out = np.stack([r, g, b], axis=-1)
    return out.astype(np.float32)


def pad_to_block(channel: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = channel.shape
    new_h = ((h + block_size - 1) // block_size) * block_size
    new_w = ((w + block_size - 1) // block_size) * block_size
    padded = np.pad(channel, ((0, new_h - h), (0, new_w - w)), mode="edge")
    return padded, (h, w)


def unpad(channel: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    h, w = original_shape
    return channel[:h, :w]


def dct_matrix(n: int = 8) -> np.ndarray:
    mat = np.zeros((n, n), dtype=np.float32)
    alpha0 = math.sqrt(1.0 / n)
    alpha = math.sqrt(2.0 / n)
    for k in range(n):
        for i in range(n):
            a = alpha0 if k == 0 else alpha
            mat[k, i] = a * math.cos((math.pi * (2 * i + 1) * k) / (2 * n))
    return mat


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return 99.0
    return 10.0 * math.log10((255.0 ** 2) / mse)
