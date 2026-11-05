"""
The Dependency Abstraction Layer (Dependency Firewall).

This module serves as the sole authorized gateway for all external dependencies 
and Python standard library imports across the entire AIDflow ecosystem. 
Following a strict architectural mandate, no module within the AIDflow package 
is permitted to import third-party libraries or native Python modules directly; 
instead, they must consume these dependencies via the `.deps` abstraction layer.
"""

from scipy.stats import (
    t, 
    chi2,
    norm, 
    ks_2samp,
    rankdata,
    kendalltau,
    gaussian_kde
)

from statsmodels.tsa.stattools import (
    acf, 
    adfuller
)
from statsmodels.tsa.arima_process import (
    ArmaProcess
)

from sklearn.metrics import (
    log_loss, 
    auc, 
    roc_curve, 
    roc_auc_score,
    classification_report, 
    confusion_matrix,
    mean_squared_error,
    r2_score
)