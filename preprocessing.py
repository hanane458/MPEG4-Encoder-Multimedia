from typing import Dict, List

import numpy as np

from utils import rgb_to_ycbcr


def preprocess_frames(frames_rgb: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
    processed = []
    for frame in frames_rgb:
        ycbcr = rgb_to_ycbcr(frame)
        processed.append(
            {
                "ycbcr": ycbcr,
                "Y": ycbcr[..., 0],
                "Cb": ycbcr[..., 1],
                "Cr": ycbcr[..., 2],
            }
        )
    return processed
