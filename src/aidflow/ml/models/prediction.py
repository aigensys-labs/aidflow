"""
Predictive interface mixin for feature-selected model implementations.

This module provides a standardized prediction layer that can be injected into 
any selection-based estimator to ensure consistent behavior across the framework. 
It implements a strict input validation and subsetting mechanism, guaranteeing 
that only the specifically selected significant features are passed to the 
underlying predictive model, regardless of the size or composition of the raw 
input dataset.
"""

from ...deps.data       import  pDF, pSe
from ...deps.types      import  (
                                Any, 
                                NDArray, 
                                ArrayLike
                                )
from ...deps.ml.base    import  ClassifierMixin


class PredictSelectionMixin:
    """Provides standard prediction behavior for Selection Models."""
    
    model:  ClassifierMixin
    result: list
    
    def __getattr__(self, name:str):
        return getattr(self.model, name)    
    
    def _set_predict_model(self, X:pDF, y:pSe) -> None:
        raise NotImplementedError('AIG SAFETY ► PredictBase must define prediction base model.')
    
    def _verify_is_predict(self) -> None:
        if any([getattr(self, attr, None) is None for attr in ['model', 'result']]):            
            raise NotImplementedError('AIG SAFETY ► PredictBase must define base `model` and `result`.')
    
    def _get_predict_subset(self, X:pDF|NDArray, verbose=True) -> pDF|NDArray:
        """Standardizes feature selection logic."""
        
        is_right_shape = X.shape[1] == len(self.result)

        if hasattr(X, 'columns'):
            missing = [f for f in self.result if f not in X.columns]
            if missing:
                raise ValueError(f"Input X is missing required features: {missing}")
            
            if not is_right_shape and verbose:
                self._subset_help(len(self.result), X.shape[1])
        
            return X[self.result]
        
        if is_right_shape:
            return X                    
            
        raise ValueError(
            "Input X must be a DataFrame with column names of selected features"
            " or np.ndarray with same features order and length as selection."
        )

    @staticmethod
    def _subset_help(result_len:int, X_fes_len:int, hint:str='.features | .result') -> None:
        """Prints help info when input X has different shape than expected."""
        print(
            f"Warning: Input X has {X_fes_len} features. "
            f"Applying subset of selected (fitted) {result_len} features."
            f"\nHint: Call {hint} to obtain selected features "
            "or supress this note: verbose=False."
        )
             
    def predict(self, X:pDF|NDArray, verbose=True, **kwargs) -> ArrayLike:
        """
        Predict response variable of a model given exogenous variables.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        array
            Fitted values at exog.
        """    
        self._verify_is_predict()
        X_sub = self._get_predict_subset(X, verbose)
        return self.model.predict(X_sub, **kwargs)

    def predict_proba(self, X:pDF|NDArray, verbose=True, **kwargs) -> NDArray:
        """
        Predict response variable of a model given exogenous variables.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        array
            NDArray([1-p, p]).
        """
        self._verify_is_predict()
        X_sub = self._get_predict_subset(X, verbose)
        return self.model.predict_proba(X_sub, **kwargs)

    def get_model(self) -> Any:
        """Returns Predictor Model of chosen significant features."""        
        return self.model

    @property
    def features(self) -> list:
        """Returns List of chosen significant features."""        
        return self.result      