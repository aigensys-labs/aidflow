"""
Context-Aware Inference Engine — Production Scoring API.

This module implements the `Predictor`, the primary inference
interface for applying trained AIDflow pipelines to new, unseen
datasets. It provides a symmetrical API that mirrors the training
pipeline's structure, ensuring that all preprocessing — from data
cleaning through feature stabilization to WOE encoding — is
executed with the exact parameters locked during training.

Training-Serving Symmetry
-------------------------
The Predictor guarantees that inference follows the identical
transformation path used during model development:

    raw data → fe_clean → fe_stabilize → obs_stabilize
    → num_encode → cat_encode → ixn_encode
    → woe_encode → woe_stabilize → agent predict

Each stage applies the trained standards (binning thresholds,
imputation values, stability filters, encoding maps) — not
re-derived statistics. This eliminates training-serving skew.
"""

from .predictor import Predictor