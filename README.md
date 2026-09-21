# jpeg_ai_forensic_pilot — Python reference implementation

Python reference implementation of the pixel-domain (Class B) branch of the
dual-branch forensic framework proposed in the companion paper *"Forensic
Integrity and Tamper Localization in JPEG AI-Compressed Images: A Problem
Formulation and Research Framework."* This is the Python counterpart to the
paper's browser-based companion interface and pilot study (Section 5) —
same equations, same algorithms, same conventional-codec proxy scope.

## Scope note (read this first)

The official JPEG AI reference Verification Model is not a pip-installable
dependency, so — exactly as in the paper's pilot study — this implementation
substitutes a conventional, standards-based JPEG codec (via Pillow) for the
JPEG AI analysis/synthesis transforms g_a, g_s of Eq. (1)–(2). **Every PSNR,
F1, and IoU value this code produces is a conventional-codec proxy number,
not a JPEG AI performance figure.** All internal logic besides
`jpeg_proxy_encode_decode()` is codec-agnostic; swap that one function for
real JPEG AI encode/decode calls to turn this into the genuine JPEG AI
pipeline.

Class C (latent-domain tampering, Eq. 5) and the latent-aware detector
D_lat are **not implemented** — `latent_splice()` raises `NotImplementedError`
with an explanation rather than approximating something it structurally
cannot be. This matches the companion browser interface's documented
coverage exactly.

## What's implemented

| Paper reference | Function |
|---|---|
| Eq. (1)–(2) (proxy) | `jpeg_proxy_encode_decode()` |
| Eq. (4) | `psnr()` |
| Eq. (6) | `ela_localize()` |
| Eq. (9) | `precision_recall_f1_iou()` |
| Algorithm 1 (Class B) | `splice_tamper()` |
| Algorithm 1 (Class C) | `latent_splice()` — raises `NotImplementedError` |
| Algorithm 2 (D_pix branch) | `run_pipeline()` |
| Algorithm 3 | `evaluate_batch()` |

## Installation

```bash
pip install -r requirements.txt
```

No other setup is required — this is pure Python + NumPy + Pillow, with no
compiled dependencies.

## Usage as a library

```python
from jpeg_ai_forensic_pilot import load_image_square, run_pipeline

target = load_image_square("photo.jpg")     # 256x256 RGB, cover-cropped
donor  = load_image_square("other.jpg")

result = run_pipeline(target, donor, quality=75, patch_frac=0.18,
                       threshold_percentile=92, seed=42)

print(result.psnr_single, result.f1, result.iou)
```

## Usage from the command line

Single-image mode (saves the four panels of the paper's Fig. 3, plus a
metrics JSON):

```bash
python -m jpeg_ai_forensic_pilot.cli single \
    --image photo.jpg --donor other.jpg \
    --quality 75 --patch-fraction 0.18 --threshold-percentile 92 \
    --seed 42 --out-dir results/single
```

Batch mode (Algorithm 3 across a directory; reproduces the paper's Table
1/2 structure as CSV + JSON, including the averaged row):

```bash
python -m jpeg_ai_forensic_pilot.cli batch \
    --images-dir ./my_images \
    --quality 75 --patch-fraction 0.18 --threshold-percentile 92 \
    --out-csv results.csv --out-json results.json
```

Rate–distortion mode (Table 1-style PSNR/bitrate at several quality points,
no tampering):

```bash
python -m jpeg_ai_forensic_pilot.cli rd \
    --images-dir ./my_images --qualities 50 75 90 --out-csv rd.csv
```

## Reproducing the paper's pilot numbers

The paper's extended pilot study (Section 5) used 24 real images at three
proxy quality points (q = 50, 75, 90) with `patch_fraction=0.18` and
`threshold_percentile=92`. Running `rd` and `batch` with those parameters
over an equivalent image set will reproduce results of the same character
(exact figures depend on which images are used and on JPEG-encoder version
differences between environments).

## File layout

```
jpeg_ai_forensic_pilot/
├── __init__.py       # public API re-exports
├── core.py           # Eq. (1)-(9) and Algorithms 1-3
├── cli.py            # command-line interface
├── requirements.txt
└── README.md
```
