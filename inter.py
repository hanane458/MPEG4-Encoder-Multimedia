from typing import Dict, List, Tuple

import numpy as np

from intra import _block_dct_quant, _dequant_idct, Q_LUMA


def _motion_estimation(
    curr: np.ndarray, prev: np.ndarray, block_size: int = 16, search_range: int = 4
) -> List[Tuple[int, int, int, int]]:
    h, w = curr.shape
    vectors = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = curr[y : y + block_size, x : x + block_size]
            best_sad = float("inf")
            best_dx, best_dy = 0, 0
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    py, px = y + dy, x + dx
                    if py < 0 or px < 0 or py + block_size > h or px + block_size > w:
                        continue
                    ref = prev[py : py + block_size, px : px + block_size]
                    sad = float(np.sum(np.abs(block - ref)))
                    if sad < best_sad:
                        best_sad = sad
                        best_dx, best_dy = dx, dy
            vectors.append((y, x, best_dy, best_dx))
    return vectors


def _build_prediction(
    prev: np.ndarray, vectors: List[Tuple[int, int, int, int]], block_size: int = 16
) -> np.ndarray:
    h, w = prev.shape
    pred = np.zeros((h, w), dtype=np.float32)
    for y, x, dy, dx in vectors:
        pred[y : y + block_size, x : x + block_size] = prev[
            y + dy : y + dy + block_size, x + dx : x + dx + block_size
        ]
    return pred


def encode_inter(curr_ycbcr: np.ndarray, prev_recon_ycbcr: np.ndarray, quality_scale: float = 1.0) -> Dict:
    curr_y = curr_ycbcr[..., 0]
    prev_y = prev_recon_ycbcr[..., 0]
    vectors = _motion_estimation(curr_y, prev_y, block_size=16, search_range=4)
    pred_y = _build_prediction(prev_y, vectors, block_size=16)
    resid = curr_y - pred_y
    qres = _block_dct_quant(resid + 128.0, Q_LUMA, quality_scale)
    return {
        "type": "P",
        "shape": curr_y.shape,
        "quality_scale": quality_scale,
        "vectors": vectors,
        "qres": qres,
    }


def decode_inter(encoded: Dict, prev_recon_ycbcr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    shape = tuple(encoded["shape"])
    quality_scale = float(encoded.get("quality_scale", 1.0))
    vectors = encoded["vectors"]
    pred_y = _build_prediction(prev_recon_ycbcr[..., 0], vectors, block_size=16)
    resid = _dequant_idct(encoded["qres"], Q_LUMA, shape, quality_scale) - 128.0
    recon_y = np.clip(pred_y + resid, 0, 255)

    recon_ycbcr = prev_recon_ycbcr.copy()
    recon_ycbcr[..., 0] = recon_y
    return recon_ycbcr.astype(np.float32), pred_y
