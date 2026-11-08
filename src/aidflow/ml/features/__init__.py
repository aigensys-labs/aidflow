"""
AIDflow — Feature Engineering Toolkit.

This package provides the internal ML utilities for feature
analysis, transformation, and quality assessment. It is the
foundational layer consumed by the pipeline's fitters and
agents — not a user-facing API.

Submodules
----------
- **reduction**: Correlation analysis and feature redundancy
  detection (identifies collinear or redundant feature groups
  before selection).

- **binning**: Optimal bin discovery for ordinal features.
  Provides bin boundaries, per-bin statistics, and the
  encoding map used by the WOE encoder.

- **woe**: Weight of Evidence (WOE) encoding. Transforms
  ordinal features into a monotonic, bin-based representation
  that linearizes the relationship with the target in
  log-odds space.

- **stability**: Temporal consistency analysis. Tests whether
  a feature's target-separation is maintained across time
  units (the quality gate behind the WOE Stabilizer stage).

- **interaction**: Tree-based feature interaction mining.
  Discovers non-linear combinations and conditional rules
  between features (used for feature engineering insight).

- **pca**: Principal Component analysis. Provides
  dimensionality reduction diagnostics (Variance explained,
  component loadings) for understanding feature space
  structure.
"""

from .binning import (
    Binning,
    MinBinResult,
    EncodingMap
)
from .encoding import (
    Encoding
)
from .interaction import (
    TreeMiner, 
    RuleStats, 
    Rule
)
from .woe import (
    WoE,
    WoEPlot
)
from .stability import (
    FeConsistency,
    FEConsistencyPlot
)
from .reduction import (
    Correlation
)
from .pca import (
    PCV, 
    PCAnalysis
)