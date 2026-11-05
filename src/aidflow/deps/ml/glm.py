"""
The Dependency Abstraction Layer (Dependency Firewall).

This module serves as the sole authorized gateway for all external dependencies 
and Python standard library imports across the entire AIDflow ecosystem. 
Following a strict architectural mandate, no module within the AIDflow package 
is permitted to import third-party libraries or native Python modules directly; 
instead, they must consume these dependencies via the `.deps` abstraction layer.
"""

from statsmodels.api import  (
    Logit as LogitSM
)

from sklearn.linear_model import  (
    LogisticRegression,
    LogisticRegressionCV,
    ElasticNetCV,
    LassoCV
)

from sklearn.cross_decomposition import (
    PLSRegression
)