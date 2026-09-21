"""
jpeg_ai_forensic_pilot
=======================

Python reference implementation of the pixel-domain (Class B) branch of the
dual-branch forensic framework proposed in "Forensic Integrity and Tamper
Localization in JPEG AI-Compressed Images: A Problem Formulation and
Research Framework" (companion paper).

See core.py for the equation-by-equation, algorithm-by-algorithm
implementation, and cli.py for a command-line entry point. See README.md
for installation and usage instructions and the scope note explaining the
conventional-codec proxy substitution used throughout.
"""

from .core import (
    load_image_square,
    make_synthetic_donor,
    jpeg_proxy_encode_decode,
    bits_per_pixel,
    psnr,
    splice_tamper,
    SpliceResult,
    latent_splice,
    ela_localize,
    precision_recall_f1_iou,
    PipelineResult,
    run_pipeline,
    evaluate_batch,
)

__version__ = "1.0.0"

__all__ = [
    "load_image_square",
    "make_synthetic_donor",
    "jpeg_proxy_encode_decode",
    "bits_per_pixel",
    "psnr",
    "splice_tamper",
    "SpliceResult",
    "latent_splice",
    "ela_localize",
    "precision_recall_f1_iou",
    "PipelineResult",
    "run_pipeline",
    "evaluate_batch",
]
