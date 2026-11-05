"""
Sensitivity Sampling — Stratified Subsampling for Importance & Pruning Analysis.

This module provides a shared sampling utility for sensitivity
analysis modules (feature importance assessment and shadow
pruning). Both modules operate on potentially large datasets and
require a representative subsample to keep computation tractable
without losing statistical power.

What It Provides
----------------
- **Sensitivity-aware subsampling**: Automatically selects an
  appropriate sample size based on the model type:
  - **GLM models**: Larger sample (coefficient estimation is
    stable with moderate samples).
  - **GBM models**: Smaller sample (tree-based fitting is
    expensive per sample; diminishing returns beyond a certain
    size).
- **Sample tracking**: Records the subsample ratio and
  observation count for reporting and transparency.
"""

from ...deps.data   import  pDF, pSe
from ...ml.sampling import  Subsampling
from .estimator     import  FeatureEstimator, OutputModel


class SamplingUtils:
    """
    Sensitivity-aware subsampling resolver.

    Determines the appropriate sample size based on the model
    type (GLM vs. GBM) and produces a stratified subsample
    that preserves the target distribution.

    Parameters
    ----------
    model : OutputModel
        The fitted model (used to determine GLM vs. GBM).
    params : dict
        Configuration parameters containing `random_state`.
    """

    def __init__(self, model: OutputModel, params: dict):
        self.params = params
        self.is_gbm = self._is_gbm(model)

    @staticmethod
    def _is_gbm(model) -> bool:
        return FeatureEstimator(model).is_non_linear_estimator()
    
    def subsample(self, X:pDF, y:pSe) -> tuple[pDF, pSe]:
        """
        Produce a stratified subsample appropriate for the model type.

        Parameters
        ----------
        X : pDF
            Full feature matrix.
        y : pSe
            Full target series.

        Returns
        -------
        tuple[pDF, pSe]
            (X_subsampled, y_subsampled) — stratified
            subsample preserving the target distribution.
        """        
        rs = self.params.get('random_state', 42)
        ss = Subsampling(X, y, rs)
        return ss.sensitivity(self.is_gbm)


class SamplingMixin:
    """
    Mixin providing subsampling and sample-spec tracking.

    Adds to any host class, that defines `self.fitter.model`, the ability to 
    subsample data for sensitivity analysis and report the subsample ratio.
    """

    def get_sample_ratio(self) -> float:
        """
        Returns the sample ratio used for the importance evaluation.

        When subsampling was applied (large dataset), this is
        the fraction of the original data used. 1.0 means the
        full dataset was used.
        """
        return self._sample_spec[1]

    def get_sample_obs(self) -> float:
        """
        Return the sample size used for the sensitivity evaluation.
        """
        return self._sample_spec[0]          

    def _get_subsample(self, X:pDF, y:pSe, params:dict) -> tuple[pDF, pSe]:
        """
        Subsample the data and record the sample specification.

        Parameters
        ----------
        X : pDF
            Full feature matrix.
        y : pSe
            Full target series.
        params : dict
            Configuration (must contain `random_state`).

        Returns
        -------
        tuple[pDF, pSe]
            (X_subsampled, y_subsampled)
        """        
        su = SamplingUtils(self.fitter.model, params)
        X_, y_ = su.subsample(X, y)
        self._set_sample_spec(X, X_)
        return X_, y_
    
    def _set_sample_spec(self, X:pDF, X_:pDF) -> None:
        """
        Record the subsample observation count and ratio.

        Parameters
        ----------
        X : pDF
            Original full dataset.
        X_ : pDF
            Subsampled dataset.
        """        
        obs = len(X_)
        self._sample_spec = (
            obs, round( obs / len(X), 4)
        )  