from typing import Dict, List, Tuple

import numpy as np

from inter import decode_inter
from intra import decode_intra
from utils import ycbcr_to_rgb


def decode_sequence(frames_encoded: List[Dict], gop: int) -> Tuple[List[np.ndarray], List[np.ndarray], List]:
    recon_ycbcr = []
    recon_rgb = []
    pred_list = []

    prev_recon = None
    for i, frame in enumerate(frames_encoded):
        is_iframe = (i % gop == 0) or frame["type"] == "I" or prev_recon is None
        if is_iframe:
            rec = decode_intra(frame)
            pred = np.zeros(rec[..., 0].shape, dtype=np.float32)
        else:
            rec, pred = decode_inter(frame, prev_recon)
        prev_recon = rec
        recon_ycbcr.append(rec)
        recon_rgb.append(ycbcr_to_rgb(rec))
        pred_list.append(pred)
    return recon_ycbcr, recon_rgb, pred_list
