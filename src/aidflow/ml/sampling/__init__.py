"""
AIDflow — Data Sampling & Partitioning Utilities.

This package provides the internal utilities for data splitting,
stratified sampling, and cross-validation dataset construction.
It is the foundational layer consumed by fitters, agents, and
diagnostic components across the pipeline.

Submodules
----------
- **sampling**: Train/test and train/validation splitting.
  Supports both random (stratified) and time-based (chronological)
  splits. Used by:
  - GBM agents (early stopping validation split).
  - GLM agents (diagnostic validation split).
  - Data Encoder (train/test creation).

- **subsampling**: Stratified subsampling for large datasets.
  Produces a representative reduced sample that preserves the
  target distribution. Used by:
  - WOE Stabilizer (reduces data for the consistency test).
  - Feature importance computation (permutation tests).

- **dataset**: Cross-validation dataset utilities.
  Provides ``DatasetCV`` (fold construction with stratification
  and group awareness) and ``DataUtils`` (data manipulation
  helpers: concatenation, alignment, index operations). Used
  by:
  - Feature selection (CV folds for regularization search).
  - Shadow pruning (CV for permutation testing).
  - General data alignment across pipeline stages.
"""

from .sampling import (
    Sampling,
    Bootstrap
)
from .subsampling import (
    Subsampling,
    SubsamplingSpec
)
from .dataset import (
    DatasetCV,
    DataUtils
)