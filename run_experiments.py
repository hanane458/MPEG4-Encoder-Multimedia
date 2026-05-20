"""Sweep quality_scale and GOP; save plots + print CSV for report."""
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from decoder import decode_sequence
from entropy import load_bitstream, save_bitstream
from inter import encode_inter
from intra import encode_intra
from preprocessing import preprocess_frames


def _make_frames_dir(n: int = 20, h: int = 120, w: int = 160) -> str:
    d = tempfile.mkdtemp(prefix="tp_frames_")
    for i in range(n):
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[..., 0] = np.linspace(0, 255, w)
        rgb[..., 1] = 40 + 8 * np.sin(np.linspace(0, 4 * np.pi, w) + 0.1 * i)
        rgb[..., 2] = 200 - 3 * i
        cy, cx = h // 2 + (i % 7) * 3, w // 2 + (i % 5) * 4
        rgb[max(0, cy - 12) : cy + 12, max(0, cx - 20) : cx + 20] = [240, 60, 60]
        rgb = np.clip(rgb, 0, 255)
        Image.fromarray(rgb.astype(np.uint8), "RGB").save(os.path.join(d, f"f{i:03d}.png"))
    return d


def _encode_all(processed, gop: int, quality_scale: float):
    encoded = []
    prev_recon = None
    for i, frame in enumerate(processed):
        ycbcr = frame["ycbcr"]
        if i % gop == 0 or prev_recon is None:
            enc = encode_intra(ycbcr, quality_scale=quality_scale)
            prev_recon = ycbcr.copy()
        else:
            enc = encode_inter(ycbcr, prev_recon, quality_scale=quality_scale)
            prev_recon[..., 0] = ycbcr[..., 0]
        encoded.append(enc)
    return encoded


def _compress_ratio(frames_rgb, encoded, gop: int, quality_scale: float, bin_path: str):
    from utils import psnr

    names = [f"{i}.png" for i in range(len(frames_rgb))]
    meta = {"gop": gop, "frame_names": names, "quality_scale": quality_scale}
    comp = save_bitstream(encoded, bin_path, meta)
    raw = sum(int(np.prod(f.shape)) for f in frames_rgb)
    ratio = raw / max(1, comp)
    payload = load_bitstream(bin_path)
    _, recon_rgb, _ = decode_sequence(payload["frames"], gop=gop)
    mpsnr = float(np.mean([psnr(o, r) for o, r in zip(frames_rgb, recon_rgb)]))
    return comp, ratio, mpsnr


def main():
    frames_dir = _make_frames_dir(20, 120, 160)
    from utils import load_frames

    frames_rgb, _ = load_frames(frames_dir)
    processed = preprocess_frames(frames_rgb)
    out_dir = os.path.join(os.path.dirname(__file__), "experiment_plots")
    os.makedirs(out_dir, exist_ok=True)
    tmp_bin = os.path.join(out_dir, "_tmp.bin")

    # Courbe 1: quality_scale (GOP fixe = 5)
    gop_fix = 5
    qs_list = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    r1, p1, c1 = [], [], []
    print("quality_scale,compressed_bytes,ratio,mean_psnr")
    for qs in qs_list:
        enc = _encode_all(processed, gop_fix, qs)
        comp, ratio, mpsnr = _compress_ratio(frames_rgb, enc, gop_fix, qs, tmp_bin)
        r1.append(ratio)
        p1.append(mpsnr)
        c1.append(comp)
        print(f"{qs},{comp},{ratio:.4f},{mpsnr:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(qs_list, r1, "o-", color="#1f77b4")
    ax.set_xlabel("Facteur de quantification quality_scale")
    ax.set_ylabel("Taux de compression (taille brute / taille .bin)")
    ax.set_title(f"Taux de compression vs quality_scale (GOP={gop_fix}, 20 frames synthétiques)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1_path = os.path.join(out_dir, "courbe1_compression_vs_quality.png")
    fig.savefig(p1_path, dpi=150)
    plt.close(fig)

    # Courbe 2: GOP (quality fixe = 1.0)
    qs_fix = 1.0
    gop_list = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    r2, c2 = [], []
    print("\ngop,compressed_bytes,ratio,n_iframes")
    for g in gop_list:
        enc = _encode_all(processed, g, qs_fix)
        comp, ratio, _ = _compress_ratio(frames_rgb, enc, g, qs_fix, tmp_bin)
        n_i = sum(1 for i in range(len(processed)) if i % g == 0)
        r2.append(ratio)
        c2.append(comp)
        print(f"{g},{comp},{ratio:.4f},{n_i}")

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(gop_list, r2, "s-", color="#d62728")
    ax2.set_xlabel("Taille du GOP (nombre de images entre deux I-frames)")
    ax2.set_ylabel("Taux de compression (taille brute / taille .bin)")
    ax2.set_title(f"Taux de compression vs GOP (quality_scale={qs_fix}, 20 frames)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2_path = os.path.join(out_dir, "courbe2_compression_vs_gop.png")
    fig2.savefig(p2_path, dpi=150)
    plt.close(fig2)

    print(f"\nPlots: {p1_path}\n       {p2_path}")


if __name__ == "__main__":
    main()
