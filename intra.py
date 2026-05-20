from typing import Dict, Tuple

import numpy as np

from utils import dct_matrix, pad_to_block, unpad


Q_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)

Q_CHROMA = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float32,
)


def _block_dct_quant(channel: np.ndarray, qmat: np.ndarray, quality_scale: float) -> np.ndarray:
    block_size = 8
    d = dct_matrix(block_size)
    padded, _ = pad_to_block(channel, block_size)
    h, w = padded.shape
    qcoeff = np.zeros((h, w), dtype=np.int16)
    q = np.maximum(1.0, qmat * quality_scale)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = padded[i : i + block_size, j : j + block_size] - 128.0
            dct_block = d @ block @ d.T
            qcoeff[i : i + block_size, j : j + block_size] = np.round(dct_block / q).astype(
                np.int16
            )
    return qcoeff


def _dequant_idct(
    qcoeff: np.ndarray, qmat: np.ndarray, original_shape: Tuple[int, int], quality_scale: float
) -> np.ndarray:
    block_size = 8
    d = dct_matrix(block_size)
    h, w = qcoeff.shape
    recon = np.zeros((h, w), dtype=np.float32)
    q = np.maximum(1.0, qmat * quality_scale)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_q = qcoeff[i : i + block_size, j : j + block_size].astype(np.float32)
            deq = block_q * q
            spatial = d.T @ deq @ d + 128.0
            recon[i : i + block_size, j : j + block_size] = spatial
    recon = np.clip(recon, 0, 255)
    return unpad(recon, original_shape)


def encode_intra(ycbcr: np.ndarray, quality_scale: float = 1.0) -> Dict:
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1]
    cr = ycbcr[..., 2]
    qy = _block_dct_quant(y, Q_LUMA, quality_scale)
    qcb = _block_dct_quant(cb, Q_CHROMA, quality_scale)
    qcr = _block_dct_quant(cr, Q_CHROMA, quality_scale)
    return {
        "type": "I",
        "shape": y.shape,
        "quality_scale": quality_scale,
        "qy": qy,
        "qcb": qcb,
        "qcr": qcr,
    }


def decode_intra(encoded: Dict) -> np.ndarray:
    shape = tuple(encoded["shape"])
    quality_scale = float(encoded.get("quality_scale", 1.0))
    ry = _dequant_idct(encoded["qy"], Q_LUMA, shape, quality_scale)
    rcb = _dequant_idct(encoded["qcb"], Q_CHROMA, shape, quality_scale)
    rcr = _dequant_idct(encoded["qcr"], Q_CHROMA, shape, quality_scale)
    return np.stack([ry, rcb, rcr], axis=-1).astype(np.float32)
