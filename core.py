"""
core.py
=======

Python reference implementation of the pixel-domain (Class B) branch of the
dual-branch forensic framework proposed in:

    "Forensic Integrity and Tamper Localization in JPEG AI-Compressed
    Images: A Problem Formulation and Research Framework"

This module implements, in one place, the equations and algorithms the
paper defines:

    Eq. (1)-(2)  analysis/synthesis transform stand-ins   -> jpeg_proxy_encode_decode()
    Eq. (4)      PSNR                                     -> psnr()
    Eq. (5)      Class C latent-domain splice              -> NOT IMPLEMENTED (see note below)
    Eq. (6)      ELA-style difference map / predicted mask -> ela_localize()
    Eq. (9)      Precision / Recall / F1 / IoU             -> precision_recall_f1_iou()
    Algorithm 1  tamper-sample generation (Class B)        -> splice_tamper()
    Algorithm 2  dual-branch detection & localization      -> run_pipeline()  (D_pix branch only)
    Algorithm 3  end-to-end evaluation loop                -> evaluate_batch()

SCOPE NOTE (read this first)
-----------------------------
Exactly as in the paper's pilot study (Section 5) and its companion browser
interface, this reference implementation substitutes a conventional,
standards-based JPEG codec (via Pillow) for the JPEG AI analysis/synthesis
transforms g_a, g_s of Eq. (1)-(2), because the official JPEG AI reference
Verification Model is not available as a pip-installable dependency. Every
PSNR, F1 and IoU value this module produces is therefore a conventional-codec
PROXY number, not a JPEG AI performance figure -- exactly as documented in
Section 5 of the paper. Swap `jpeg_proxy_encode_decode()` for real JPEG AI
encode/decode calls to turn this into the genuine JPEG AI evaluation
pipeline; every other function is codec-agnostic and needs no change.

Class C (latent-domain tampering, Eq. 5) and the latent-aware detector D_lat
are NOT implemented here, for the same reason they are not implemented in
the companion browser interface: both require access to the actual JPEG AI
encoder/decoder internals (the latent tensor y-bar itself), which a
conventional codec does not expose in a comparable form. `latent_splice()`
below raises NotImplementedError with this same explanation rather than
silently doing something unrelated.

Dependencies: numpy, Pillow (see requirements.txt).
"""

from __future__ import annotations

import io
import dataclasses
from typing import Optional, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "load_image_square",
    "make_synthetic_donor",
    "jpeg_proxy_encode_decode",
    "psnr",
    "splice_tamper",
    "latent_splice",
    "ela_localize",
    "precision_recall_f1_iou",
    "PipelineResult",
    "run_pipeline",
    "evaluate_batch",
]

DEFAULT_SIZE = 256  # matches the 256x256 pilot protocol used throughout the paper (Section 5)


# --------------------------------------------------------------------------
# Image I/O helpers
# --------------------------------------------------------------------------

def load_image_square(path: str, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Load an image and center-crop/resize it to a size x size RGB square.

    This is a "cover" fit (scale to fill, then center-crop), matching the
    behaviour of the companion browser interface's drawCover() and the
    256x256 protocol used in Section 5 of the paper.
    """
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = max(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    im = im.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    im = im.crop((left, top, left + size, top + size))
    return np.array(im, dtype=np.uint8)


def make_synthetic_donor(size: int = DEFAULT_SIZE, seed: int = 1234) -> np.ndarray:
    """Generate a synthetic gradient/blob donor image.

    Used as a fallback in run_pipeline() when the caller does not supply a
    donor image, so the pipeline can still execute end-to-end (mirrors the
    companion interface's makeSyntheticDonor()).
    """
    rng = np.random.default_rng(seed)
    # Simple diagonal color gradient
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    c0 = rng.uniform(0, 255, size=3)
    c1 = rng.uniform(0, 255, size=3)
    t = ((xx + yy) / 2)[..., None]
    img = (c0 * (1 - t) + c1 * t).astype(np.uint8)
    # A few translucent-looking blobs for texture
    for _ in range(10):
        cx, cy = rng.uniform(0, size, size=2)
        r = rng.uniform(10, 45)
        color = rng.uniform(0, 255, size=3)
        yy_idx, xx_idx = np.ogrid[:size, :size]
        mask = (xx_idx - cx) ** 2 + (yy_idx - cy) ** 2 <= r ** 2
        img[mask] = (0.5 * img[mask] + 0.5 * color).astype(np.uint8)
    return img


# --------------------------------------------------------------------------
# Eq. (1)-(2): proxy analysis/synthesis transform (see SCOPE NOTE above)
# --------------------------------------------------------------------------

def jpeg_proxy_encode_decode(img: np.ndarray, quality: int) -> Tuple[np.ndarray, int]:
    """Proxy stand-in for JPEG AI's g_a (analysis) + g_s (synthesis), Eq. (1)-(2).

    Encodes `img` with a conventional JPEG codec at the given quality and
    decodes it back, returning the decoded image and the encoded size in
    bytes (used to estimate bits-per-pixel).

    Parameters
    ----------
    img : (H, W, 3) uint8 RGB array
    quality : JPEG quality, 1-95. Stands in for the JPEG AI rate point lambda
        of Eq. (3): lower quality ~ higher lambda ~ lower bitrate.

    Returns
    -------
    decoded : (H, W, 3) uint8 RGB array, same shape as `img`
    n_bytes : size of the encoded bitstream, in bytes
    """
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    n_bytes = buf.tell()
    buf.seek(0)
    decoded = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return decoded, n_bytes


def bits_per_pixel(n_bytes: int, height: int, width: int) -> float:
    """Bitrate in bits-per-pixel from an encoded size in bytes."""
    return (n_bytes * 8) / (height * width)


# --------------------------------------------------------------------------
# Eq. (4): PSNR
# --------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio between two uint8 images, Eq. (4).

    Returns 99.0 for a perfect (zero-MSE) match, matching the convention
    used in the companion interface and the paper's pilot study.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 99.0
    return 10.0 * np.log10((255.0 ** 2) / mse)


# --------------------------------------------------------------------------
# Algorithm 1 (Class B): pixel-domain splice tamper
# --------------------------------------------------------------------------

@dataclasses.dataclass
class SpliceResult:
    tampered: np.ndarray      # tampered image, same shape as the target
    mask: np.ndarray          # (H, W) uint8 {0,1} ground-truth tamper mask
    region: Tuple[int, int, int, int]  # (x0, y0, w, h) of the pasted patch


def splice_tamper(
    target: np.ndarray,
    donor: np.ndarray,
    patch_frac: float = 0.18,
    rng: Optional[np.random.Generator] = None,
) -> SpliceResult:
    """Algorithm 1, Class B: splice a square patch from `donor` into `target`.

    Parameters
    ----------
    target : (H, W, 3) uint8 RGB array to be tampered
    donor : (H, W, 3) uint8 RGB array supplying the pasted patch
    patch_frac : side length of the square patch, as a fraction of min(H, W)
    rng : optional numpy random Generator for reproducibility

    Returns
    -------
    SpliceResult(tampered, mask, region)
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = target.shape[:2]
    dh, dw = donor.shape[:2]
    ph = max(8, round(min(h, w) * patch_frac))
    pw = ph

    x0 = int(rng.integers(0, max(1, w - pw)))
    y0 = int(rng.integers(0, max(1, h - ph)))
    x1 = int(rng.integers(0, max(1, dw - pw)))
    y1 = int(rng.integers(0, max(1, dh - ph)))

    tampered = target.copy()
    tampered[y0:y0 + ph, x0:x0 + pw] = donor[y1:y1 + ph, x1:x1 + pw]

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y0 + ph, x0:x0 + pw] = 1

    return SpliceResult(tampered=tampered, mask=mask, region=(x0, y0, pw, ph))


def latent_splice(*_args, **_kwargs):
    """Eq. (5), Class C latent-domain splice: NOT IMPLEMENTED.

    Class C tampering operates on the JPEG AI latent tensor y-bar itself
    (y-bar_tamper = M (o) y-bar_d + (1-M) (o) y-bar), which requires access
    to the actual JPEG AI encoder/decoder internals. A conventional codec
    (used as the proxy throughout this reference implementation) has no
    comparable latent representation, so this function intentionally raises
    rather than approximating Class C with something it is not. See the
    SCOPE NOTE at the top of this module, Section 4.3 (threat model) and
    Section 5 of the paper.
    """
    raise NotImplementedError(
        "Class C (latent-domain) tampering requires the actual JPEG AI "
        "encoder/decoder internals and is out of scope for this "
        "conventional-codec proxy reference implementation. See the "
        "module docstring's SCOPE NOTE."
    )


# --------------------------------------------------------------------------
# Eq. (6): ELA-style pixel-only detector D_pix
# --------------------------------------------------------------------------

def ela_localize(tampered: np.ndarray, quality: int, threshold_percentile: float = 92.0) -> np.ndarray:
    """Eq. (6): Error-Level-Analysis-style tamper mask prediction.

    Re-encodes `tampered` at `quality`, differences it against itself, and
    thresholds the per-pixel absolute difference at `threshold_percentile`.

    Returns
    -------
    pred_mask : (H, W) uint8 {0,1} array
    """
    re_decoded, _ = jpeg_proxy_encode_decode(tampered, quality)
    diff = np.abs(tampered.astype(np.int16) - re_decoded.astype(np.int16)).sum(axis=2)
    thresh = np.percentile(diff, threshold_percentile)
    return (diff >= thresh).astype(np.uint8)


# --------------------------------------------------------------------------
# Eq. (9): Precision / Recall / F1 / IoU
# --------------------------------------------------------------------------

def precision_recall_f1_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Eq. (9): pixel-level Precision, Recall, F1 and IoU of a predicted mask."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / union if union > 0 else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou)}


# --------------------------------------------------------------------------
# Algorithm 2: dual-branch detection & localization (D_pix branch only)
# --------------------------------------------------------------------------

@dataclasses.dataclass
class PipelineResult:
    decoded: np.ndarray
    tampered: np.ndarray
    gt_mask: np.ndarray
    pred_mask: np.ndarray
    psnr_single: float
    bpp: float
    psnr_tamper: float
    precision: float
    recall: float
    f1: float
    iou: float


def run_pipeline(
    target: np.ndarray,
    donor: Optional[np.ndarray],
    quality: int = 75,
    patch_frac: float = 0.18,
    threshold_percentile: float = 92.0,
    seed: Optional[int] = None,
) -> PipelineResult:
    """Run Algorithm 1 (Class B) + Algorithm 2 (D_pix branch) on one image.

    This is the reference-implementation equivalent of the companion
    interface's single-image "Run Pipeline" action. Only the pixel-only
    branch D_pix is computed (see SCOPE NOTE); the dual-branch fusion of
    Eq. (7)-(8) reduces to w1=1, w2=0 as in Algorithm 2's pixel-only
    fallback path.

    Parameters
    ----------
    target : (H, W, 3) uint8 RGB array, the pristine image x
    donor : optional (H, W, 3) uint8 RGB array; a synthetic donor is
        generated via make_synthetic_donor() if None
    quality : proxy rate point (JPEG quality, 1-95)
    patch_frac : Algorithm 1 tamper patch size, as a fraction of image side
    threshold_percentile : Eq. (6) ELA threshold percentile
    seed : optional int for reproducible tamper placement

    Returns
    -------
    PipelineResult
    """
    rng = np.random.default_rng(seed)
    if donor is None:
        donor = make_synthetic_donor(size=target.shape[0], seed=(seed or 777))

    decoded, n_bytes = jpeg_proxy_encode_decode(target, quality)
    psnr_single = psnr(target, decoded)
    bpp = bits_per_pixel(n_bytes, target.shape[0], target.shape[1])

    splice = splice_tamper(decoded, donor, patch_frac=patch_frac, rng=rng)
    tampered_re, _ = jpeg_proxy_encode_decode(splice.tampered, quality)
    psnr_tamper = psnr(decoded, tampered_re)

    pred_mask = ela_localize(tampered_re, quality, threshold_percentile)
    metrics = precision_recall_f1_iou(pred_mask, splice.mask)

    return PipelineResult(
        decoded=decoded,
        tampered=tampered_re,
        gt_mask=splice.mask,
        pred_mask=pred_mask,
        psnr_single=psnr_single,
        bpp=bpp,
        psnr_tamper=psnr_tamper,
        **metrics,
    )


# --------------------------------------------------------------------------
# Algorithm 3: end-to-end batch evaluation loop
# --------------------------------------------------------------------------

def evaluate_batch(
    images: "list[tuple[str, np.ndarray]]",
    quality: int = 75,
    patch_frac: float = 0.18,
    threshold_percentile: float = 92.0,
    donor_offset: Optional[int] = None,
    seed: int = 1000,
) -> "list[dict]":
    """Algorithm 3: run the pipeline across a batch of images.

    Each image is used once as the tamper target and, for another image in
    the set offset by `donor_offset` positions (default: half the batch
    size, matching the companion interface's batch pairing), as that
    image's donor.

    Parameters
    ----------
    images : list of (name, image_array) pairs, each image_array already
        loaded to a common size via load_image_square()
    quality, patch_frac, threshold_percentile : see run_pipeline()
    donor_offset : index offset used to pick each image's donor;
        defaults to len(images) // 2 + 1
    seed : base seed; image i uses seed + i for reproducibility

    Returns
    -------
    rows : list of dicts, one per image, with keys:
        image, psnr, bpp, psnr_tamper, precision, recall, f1, iou
    """
    n = len(images)
    if n < 2:
        raise ValueError("evaluate_batch requires at least 2 images (for donor pairing).")
    if donor_offset is None:
        donor_offset = n // 2 + 1

    rows = []
    for i, (name, img) in enumerate(images):
        donor_name, donor_img = images[(i + donor_offset) % n]
        result = run_pipeline(
            img, donor_img, quality=quality, patch_frac=patch_frac,
            threshold_percentile=threshold_percentile, seed=seed + i,
        )
        rows.append({
            "image": name,
            "psnr": round(result.psnr_single, 2),
            "bpp": round(result.bpp, 3),
            "psnr_tamper": round(result.psnr_tamper, 2),
            "precision": round(result.precision, 3),
            "recall": round(result.recall, 3),
            "f1": round(result.f1, 3),
            "iou": round(result.iou, 3),
        })
    return rows
