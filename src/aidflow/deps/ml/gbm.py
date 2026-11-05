"""
The Dependency Abstraction Layer (Dependency Firewall).

This module serves as the sole authorized gateway for all external dependencies 
and Python standard library imports across the entire AIDflow ecosystem. 
Following a strict architectural mandate, no module within the AIDflow package 
is permitted to import third-party libraries or native Python modules directly; 
instead, they must consume these dependencies via the `.deps` abstraction layer.
"""

from xgboost import (
    XGBClassifier
)

from xgboost.callback import (
    TrainingCallback
)

from lightgbm import (
    LGBMClassifier, 
    LGBMRegressor
)

import optuna