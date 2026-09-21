"""
cli.py
======

Command-line interface for the JPEG AI forensic pilot reference
implementation (see core.py for the equations/algorithms this wraps).

Usage
-----
Single-image mode (Algorithms 1-2 on one image; saves the four panels of
Fig. 3 in the paper plus a metrics JSON):

    python -m jpeg_ai_forensic_pilot.cli single \\
        --image photo.jpg --donor other.jpg \\
        --quality 75 --patch-fraction 0.18 --threshold-percentile 92 \\
        --seed 42 --out-dir results/single

Batch mode (Algorithm 3 across a directory of images; reproduces the
paper's Table 1/2 as CSV + JSON):

    python -m jpeg_ai_forensic_pilot.cli batch \\
        --images-dir ./my_images --quality 75 --patch-fraction 0.18 \\
        --threshold-percentile 92 --out-csv results.csv --out-json results.json

Rate-distortion mode (Table 1-style PSNR/bitrate at several quality points,
no tampering):

    python -m jpeg_ai_forensic_pilot.cli rd \\
        --images-dir ./my_images --qualities 50 75 90 --out-csv rd.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import core

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    _save_image((mask * 255).astype(np.uint8), path)


def _discover_images(images_dir: str) -> "list[tuple[str, np.ndarray]]":
    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if len(paths) < 2:
        raise SystemExit(f"Need at least 2 images in {images_dir}, found {len(paths)}.")
    return [(p.name, core.load_image_square(str(p))) for p in paths]


def cmd_single(args: argparse.Namespace) -> None:
    target = core.load_image_square(args.image)
    donor = core.load_image_square(args.donor) if args.donor else None

    result = core.run_pipeline(
        target, donor,
        quality=args.quality,
        patch_frac=args.patch_fraction,
        threshold_percentile=args.threshold_percentile,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    _save_image(result.decoded, out_dir / "a_decoded.png")
    _save_image(result.tampered, out_dir / "b_tampered.png")
    _save_mask(result.gt_mask, out_dir / "c_ground_truth_mask.png")
    _save_mask(result.pred_mask, out_dir / "d_predicted_mask.png")

    metrics = {
        "psnr_db": round(result.psnr_single, 2),
        "bitrate_bpp": round(result.bpp, 3),
        "psnr_after_tamper_db": round(result.psnr_tamper, 2),
        "precision": round(result.precision, 3),
        "recall": round(result.recall, 3),
        "f1": round(result.f1, 3),
        "iou": round(result.iou, 3),
        "quality": args.quality,
        "patch_fraction": args.patch_fraction,
        "threshold_percentile": args.threshold_percentile,
        "seed": args.seed,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved panels (a)-(d) and metrics.json to: {out_dir}")


def cmd_batch(args: argparse.Namespace) -> None:
    images = _discover_images(args.images_dir)
    rows = core.evaluate_batch(
        images,
        quality=args.quality,
        patch_frac=args.patch_fraction,
        threshold_percentile=args.threshold_percentile,
    )

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.out_csv}")

    summary = {
        "quality": args.quality, "patch_fraction": args.patch_fraction,
        "threshold_percentile": args.threshold_percentile, "n_images": len(rows),
        "rows": rows,
        "average": {
            k: round(float(np.mean([r[k] for r in rows])), 3)
            for k in ("psnr", "bpp", "psnr_tamper", "precision", "recall", "f1", "iou")
        },
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.out_json}")

    print(json.dumps(summary["average"], indent=2))


def cmd_rd(args: argparse.Namespace) -> None:
    images = _discover_images(args.images_dir)
    rows = []
    for name, img in images:
        row = {"image": name}
        for q in args.qualities:
            decoded, n_bytes = core.jpeg_proxy_encode_decode(img, q)
            row[f"psnr_q{q}"] = round(core.psnr(img, decoded), 2)
            row[f"bpp_q{q}"] = round(core.bits_per_pixel(n_bytes, img.shape[0], img.shape[1]), 3)
        rows.append(row)

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.out_csv}")
    else:
        print(json.dumps(rows, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jpeg_ai_forensic_pilot",
        description="Reference implementation of Algorithms 1-3 (pixel-domain / D_pix branch) "
                    "from the JPEG AI forensic framework paper.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("single", help="Run Algorithms 1-2 on one image.")
    ps.add_argument("--image", required=True, help="Path to the target image x.")
    ps.add_argument("--donor", default=None, help="Path to a donor image (optional; synthetic donor used if omitted).")
    ps.add_argument("--quality", type=int, default=75, help="Proxy JPEG quality, 1-95 (default: 75).")
    ps.add_argument("--patch-fraction", type=float, default=0.18, help="Tamper patch fraction p (default: 0.18).")
    ps.add_argument("--threshold-percentile", type=float, default=92.0, help="ELA threshold percentile tau (default: 92).")
    ps.add_argument("--seed", type=int, default=None, help="Random seed for reproducible tamper placement.")
    ps.add_argument("--out-dir", default="./out_single", help="Directory to save panels + metrics.json.")
    ps.set_defaults(func=cmd_single)

    pb = sub.add_parser("batch", help="Run Algorithm 3 across a directory of images.")
    pb.add_argument("--images-dir", required=True, help="Directory of images (>=2).")
    pb.add_argument("--quality", type=int, default=75)
    pb.add_argument("--patch-fraction", type=float, default=0.18)
    pb.add_argument("--threshold-percentile", type=float, default=92.0)
    pb.add_argument("--out-csv", default="results.csv")
    pb.add_argument("--out-json", default="results.json")
    pb.set_defaults(func=cmd_batch)

    pr = sub.add_parser("rd", help="Rate-distortion table (PSNR/bitrate) across quality points, no tampering.")
    pr.add_argument("--images-dir", required=True)
    pr.add_argument("--qualities", type=int, nargs="+", default=[50, 75, 90])
    pr.add_argument("--out-csv", default=None)
    pr.set_defaults(func=cmd_rd)

    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
