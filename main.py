import argparse 
import os 

import matplotlib.pyplot as plt 
import numpy as np 
 
from decoder import decode_sequence
from entropy import load_bitstream, save_bitstream
from inter import encode_inter
from intra import encode_intra
from preprocessing import preprocess_frames
from utils import ensure_dir, load_frames, psnr, save_rgb_frame


def visualize_example(
    original_rgb: np.ndarray,
    ycbcr: np.ndarray,
    recon_rgb: np.ndarray,
    output_path: str,
) -> None:
    ensure_dir(os.path.dirname(output_path))
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(np.clip(original_rgb / 255.0, 0, 1))
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Y")
    plt.imshow(y, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Cb")
    plt.imshow(cb, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("Cr")
    plt.imshow(cr, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Reconstructed")
    plt.imshow(np.clip(recon_rgb / 255.0, 0, 1))
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("Absolute Error")
    err = np.mean(np.abs(original_rgb - recon_rgb), axis=-1)
    plt.imshow(err, cmap="hot")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_pipeline(frames_dir: str, output_bin: str, gop: int, quality_scale: float) -> None:
    frames_rgb, names = load_frames(frames_dir)
    if not frames_rgb:
        raise ValueError(f"Aucune image trouvee dans {frames_dir}")

    processed = preprocess_frames(frames_rgb)
    encoded_frames = []
    prev_recon = None
    iframe_count = 0
    pframe_count = 0

    for i, frame in enumerate(processed):
        ycbcr = frame["ycbcr"]
        if i % gop == 0 or prev_recon is None:
            enc = encode_intra(ycbcr, quality_scale=quality_scale)
            prev_recon = ycbcr.copy()
            iframe_count += 1
        else:
            enc = encode_inter(ycbcr, prev_recon, quality_scale=quality_scale)
            prev_recon[..., 0] = ycbcr[..., 0]
            pframe_count += 1
        encoded_frames.append(enc)

    metadata = {
        "gop": gop,
        "frame_names": names,
        "quality_scale": quality_scale,
    }
    compressed_size = save_bitstream(encoded_frames, output_bin, metadata)
    payload = load_bitstream(output_bin)

    recon_ycbcr, recon_rgb, _ = decode_sequence(payload["frames"], gop=gop)
    recon_dir = os.path.join(os.path.dirname(output_bin), "reconstructed")
    ensure_dir(recon_dir)
    for name, rgb in zip(names, recon_rgb):
        save_rgb_frame(rgb, os.path.join(recon_dir, name))

    original_size = sum(int(np.prod(f.shape)) for f in frames_rgb)
    ratio = original_size / max(1, compressed_size)
    psnr_values = [psnr(o, r) for o, r in zip(frames_rgb, recon_rgb)]
    mean_psnr = float(np.mean(psnr_values))

    print("=== RESULTATS ===")
    print(f"Nombre frames: {len(frames_rgb)}")
    print(f"I-frames: {iframe_count}")
    print(f"P-frames: {pframe_count}")
    print(f"Taille compressee (bytes): {compressed_size}")
    print(f"Taux de compression approx: {ratio:.2f}")
    print(f"PSNR moyen (dB): {mean_psnr:.2f}")
    print(f"Fichier binaire: {output_bin}")
    print(f"Images reconstruites: {recon_dir}")

    fig_out = os.path.join(os.path.dirname(output_bin), "visualisation.png")
    visualize_example(frames_rgb[0], processed[0]["ycbcr"], recon_rgb[0], fig_out)
    print(f"Visualisation: {fig_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini encodeur video simplifie")
    parser.add_argument("--frames_dir", default="frames", help="Dossier des frames")
    parser.add_argument("--output_bin", default="output.bin", help="Bitstream de sortie")
    parser.add_argument("--gop", type=int, default=5, help="Taille du GOP")
    parser.add_argument(
        "--quality_scale",
        type=float,
        default=1.0,
        help="Facteur quantification (>1 plus compresse, <1 meilleure qualite)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.frames_dir, args.output_bin, args.gop, args.quality_scale)
