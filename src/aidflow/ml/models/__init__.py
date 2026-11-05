"""
AIDflow — ML Model Wrappers & Selection Backends.

This package provides the internal model implementations and
feature selection engines consumed by the training agents.
It abstracts the underlying ML libraries (scikit-learn,
LightGBM, XGBoost) behind a consistent interface.

Submodules
----------
- **models**: Core estimator wrappers.

  | Class        | Backend              | Use Case                          |
  |------------- |----------------------|-----------------------------------|
  | `Logit`      | statsmodels          | Standard logistic regression (final GLM fit) |
  | `Lasso`      | scikit-learn         | L1 regularized selection          |
  | `ElasticNet` | scikit-learn         | L1+L2 regularized selection       |
  | `LogReg`     | scikit-learn         | Unpenalized logit (baseline)      |
  | `LGB`        | LightGBM             | Gradient boosting (GBM agent)     |
  | `XGB`        | XGBoost              | Gradient boosting (GBM agent)     |
  | `ModelUtils` | —                    | Shared model utilities (is_tree, is_lr,...) |
  | `ModelView`  | —                    | Unified model introspection interface |

- **selection**: Sparse (stepwise) feature selection engine.
  Sequential forward/backward addition based on statistical
  significance. Produces the smallest interpretable model.
  Consumed by the `SparseBinaryGLM` agent.

- **pls**: Partial Least Squares selection and regression.
  Maximizes covariance between features and target. Best for
  high-dimensional, correlated feature spaces. Consumed by
  the `PLSBinaryGLM` agent.
"""

from .models import (
    Logit,
    Lasso,
    ElasticNet,
    LogReg,
    LGB,
    XGB,
    ModelUtils,
    ModelView
)
from .selection import (
    SparseSelector
)
from .pls import (
    PLSSelector,
    BinaryPLSRegression
)