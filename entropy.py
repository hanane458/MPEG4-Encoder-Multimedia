import pickle
import zlib
from typing import Any, Dict, List

import numpy as np


def rle_encode_1d(values: List[int]) -> List[List[int]]:
    if not values:
        return []
    out = []
    prev = values[0]
    count = 1
    for v in values[1:]:
        if v == prev:
            count += 1
        else:
            out.append([int(prev), int(count)])
            prev = v
            count = 1
    out.append([int(prev), int(count)])
    return out


def rle_decode_1d(rle: List[List[int]]) -> np.ndarray:
    out = []
    for value, count in rle:
        out.extend([value] * count)
    return np.array(out, dtype=np.int16)


def _pack_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    packed = {k: v for k, v in frame.items() if k not in {"qy", "qcb", "qcr", "qres"}}
    for key in ("qy", "qcb", "qcr", "qres"):
        if key in frame:
            arr = frame[key].astype(np.int16)
            packed[key] = {"shape": arr.shape, "rle": rle_encode_1d(arr.flatten().tolist())}
    return packed


def _unpack_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(frame)
    for key in ("qy", "qcb", "qcr", "qres"):
        if key in frame:
            shape = tuple(frame[key]["shape"])
            rle = frame[key]["rle"]
            out[key] = rle_decode_1d(rle).reshape(shape)
    return out


def save_bitstream(frames_data: List[Dict[str, Any]], output_path: str, metadata: Dict[str, Any]) -> int:
    payload = {
        "metadata": metadata,
        "frames": [_pack_frame(f) for f in frames_data],
    }
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(raw, level=9)
    with open(output_path, "wb") as f:
        f.write(compressed)
    return len(compressed)


def load_bitstream(input_path: str) -> Dict[str, Any]:
    with open(input_path, "rb") as f:
        compressed = f.read()
    raw = zlib.decompress(compressed)
    payload = pickle.loads(raw)
    payload["frames"] = [_unpack_frame(f) for f in payload["frames"]]
    return payload
