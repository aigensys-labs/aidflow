"""
Core contractual definitions and base interfaces for all Fitter implementations.

This module defines the mandatory signatures and state-management protocols 
that every ML execution component (Fitter) must follow within AIDflow. By 
establishing a strict hierarchy of base classes, it ensures seamless 
interoperability between different pipeline phases (Data, Transformation, and Training).

Key architectural roles provided by this module:
1. Standardized Signatures: Enforces mandatory attributes (e.g., `_model`) across 
   all fitters to ensure predictable orchestration by the Model layer.
   
2. State-Safe Access: Implements guardrails to prevent access to results, features, 
   or models before the fitting process has successfully completed.

3. Pickle-Safety & Sanitization: Provides automated stripping of heavy objects 
   (like `stats` or `dash`) during serialization to ensure efficient model 
   persistence and deployment.

4. Lifecycle Decoupling: Separates the inner ML implementation (FitterBase) from 
   the final pipeline wrapper (FitterModelBase), allowing for clean integration 
   into the global FitRegistry.
"""

from ...deps.data   import  pDF
from ...deps.types  import  (
                            NDArray, 
                            ArrayLike,
                            Generic, 
                            TypeVar
                            )
from ..foundation   import  AigParam
from ..factory      import  (
                            Dataclass,
                            ValidationDataclass
                            )


@Dataclass.base(update_specs=dict(slots=False))
class FitParamsBase:
    """Fit params input design."""       

@Dataclass.base
class FitParamsCfg:
    """Fit params input design."""       

@Dataclass.base
class FitParams(FitParamsBase, FitParamsCfg):
    """Fit params input design."""  

@Dataclass.base
class FitResult:
    """Fit params input design."""  

@Dataclass.base
class FitStats:
    """Fit params input design.""" 

@Dataclass.base
class FitDash:
    """Fit params input design.""" 

class _NoExternalModel:    
    pass

class _DataModel:

    def transform() -> pDF:
        """
        FitterDataBase fit IDE helper function.
        Redirects to self._model with is_fitted verification → raise if False.
        """ 
        ...

    def predict() -> pDF:
        """
        FitterDataBase fit IDE helper function.
        Redirects to self._model with is_fitted verification → raise if False.
        """ 
        ...

class _LearnModel:

    def predict_proba() -> NDArray:
        """
        FitterLearnBase fit IDE helper function.
        Redirects to self._model with is_fitted verification → raise if False.
        """ 
        ...

    def predict() -> ArrayLike:
        """
        FitterLearnBase fit IDE helper function.
        Redirects to self._model with is_fitted verification → raise if False.
        """ 
        ...

class IOBase:

    _sanitize:  list = ['stats', 'dash']

    def __aig_copy__(self):
        """Prepares the state for sanitization, slimming down."""
        model = object.__new__(type(self))
        state  = self.__getstate__()
        model.__setstate__(state)
        return model
    
    def __getstate__(self) -> dict:
        """Prepares the state for pickling, slimming down."""

        if hasattr(super(), '__getstate__'):
            state = super().__getstate__()
        else:
            state = self.__dict__
        
        state = state.copy()
        for s in self._sanitize:
            state.pop(s, None)
        
        return state
    
    def __setstate__(self, state: dict) -> None:
        """Re-hydrates the model into a proper object instance."""
        self.__dict__.update(state)
        
        if hasattr(super(), '__setstate__'):
            super().__setstate__(state)
        

type FitModel = _DataModel | _LearnModel

class FitterBase(IOBase):
    
    _model:  FitModel = None
    result: FitResult = None
    stats:   FitStats = None
    dash:     FitDash = None
    features:    list = None

    def __getattr__(self, name:str):
        if ( self._model is None ) or isinstance(self._model, _NoExternalModel):
            raise AttributeError(name)
        return getattr(self.model, name)
    
    def fit(self):
        raise NotImplemented('Fitter must define custom "fit" implementation.')
    
    def is_fitted(self) -> bool:
        return self.result is not None    

    def is_model(self) -> bool:
        return self._model is not None        

    @staticmethod
    def get_estimator_shared_setup(estimator, params: FitParams) -> dict:
        """
        Allows AIG Fitter to obtain arbitrary kwargs addressing the inner estimator responsible for ML.
        """
        sig = AigParam.get_definition(estimator)
        if not sig:
            return {}
        return {k:v for k,v in params.items() if k in sig}
        
    def _verify_model(self):
        """Verify that FitWorker model has been fitted and `holds populated self._model`."""
        if not self.is_model():
            raise ValueError('Use .fit() prior calling inference model.')                   
    
    @property
    def model(self):
        """
        Fitter underlying model for inference.
        Raises Exception if accessed prior fit().
        """        
        self._verify_model()
        return self._model
    

class FitterDataBase(FitterBase, ValidationDataclass):
    """
    Base class for inner Fitter model of DataPhase | TransformationPhase.
    Fitters can be found in e.g.: ./data/transform/models/num_discretizor.py.

    **The Fitter must define _model if *inner external model* is in use:**
    - _model : type

    **Suplemented Fitter cls attrs=None:**
    - result
    - stats
    - dash
    - features
    """
    _model = _NoExternalModel()

    def transform(self) -> pDF:
        """
        FitterDataBase transform IDE helper function.
        Redirects to self._fitter with is_fitted verification → raise if False.
        """
        ...

    def process(self) -> pDF:
        """
        FitterDataBase process IDE helper function.
        Redirects to self._fitter with is_fitted verification → raise if False.
        """
        ...


class FitterTrainBase(FitterBase, ValidationDataclass):
    """
    Base class for inner Fitter model mainly of TrainingPhase.
    Fitters can be found in e.g.: ./learner/agents/models/glm_selector.py.
    
    **The Fitter class must define:**
    - _model : type

    **Suplemented Fitter cls attrs=None:**
    - result
    - stats
    - dash
    - features

    Access to Fitter fitted results:
    - model  → self._model → raise if not fitted
    - features → self.features → raise if not fitted
    """
                
    def predict_proba(self):
        """
        FitterBase predict_proba IDE helper function.
        Redirects to self._fitter with is_fitted verification → raise if False.
        """
        ...
    
    def predict(self):
        """
        FitterBase predict IDE helper function.
        Redirects to self._fitter with is_fitted verification → raise if False.
        """
        ...

FitterT = TypeVar('FitterT', bound=FitterDataBase | FitterTrainBase)

class FitterModelBase(Generic[FitterT], ValidationDataclass):
    """
    Base class providing fitted state safe access functionality for 
    `Inner Fitter Models` (e.g.: .../x/models/inner_fitter.py).
    
    **The wrapper class must define:**
    - _fitter : type
    - _params : type    

    **The wrapper optionally can define attrs to sanitize by:**
    - _estimator : type → propagates inner estimator signature for extra config
    - _config    : dict → internal estimator config provided by kwargs
    - _sanitize  : list → `default: ['stats', 'dash]`

    Designed for the final pipeline wrapper class of the actual Fitter. 
    **Mandatory Fitter attrs:**
    - result
    - stats

    Handles `stats` stripping for pickle-safety and provides 
    unified access to Fitter results:
    - fitter → self._fitter
    - result → self._fitter.result
    - stats  → self._fitter.stats
    - model  → self._fitter.model (TrainingPhase)
    - features → self._fitter.features (TrainingPhase)

    By the AIG design, the Inner Fitter Model is registred to FitRegistry
    by the `Main Model` (Orchestrator) (e.g.: .../x/main_worker.py).
    """
    _fitter: FitterT = None
    _estimator: type = None

    _params = None
    _config = None

    def __getattr__(self, name:str):
        if self._fitter is None:
            raise AttributeError(name)
        return getattr(self.fitter, name)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def get_estimator(cls) -> type:
        """
        Fitter underlying internal estimator.
        - use .model to get fitter's model → estimator with AIG pre/post processing.
        - use .get_estimator() to get fitter's model raw internal estimator class.
        """
        return cls._estimator
        
    def get_estimator_config(self, params: FitParams, kwargs: dict) -> dict:
        """
        Allows AIG Fitter to obtain arbitrary kwargs addressing the inner estimator
        responsible for ML.
        """
        est = self.get_estimator()
        if est is None:
            return {}
        sig = AigParam.get_definition(est)
        if not sig:
            return {}
        prm = Dataclass.get_names(params)
        return {k:v for k,v in kwargs.items() if ( k not in prm ) and ( k in sig )}
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def is_fitted(self) -> bool:
        return self._fitter is not None

    def _is_learn(self, name:str) -> bool:
        if not hasattr(self._fitter, name): 
            return False
        return True
    
    def _validate_worker_design(self, name:str) -> None:
        if (
            ( not self.is_fitted() ) or 
            ( hasattr(self._fitter, name) )
        ):
            return
        raise ValueError(
            'AIG SAFETY ► Inner Fitter-Worker design must '
            f'follow AIG API standards "features - result - stats" ► missing attr "{name}"'
        )
            
    def _verify_fitted(self):
        if not self.is_fitted():
            raise ValueError('Use .fit() prior calling fitter model.')   
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @property
    def fitter(self) -> FitterT:
        """
        Access to inner Fitter for inference.
        Raises Exception if accessed prior fit().
        """
        self._verify_fitted()
        return self._fitter
    
    @property
    def params(self) -> FitParams:
        """Access params used during fitting."""        
        return self._params
    
    @property
    def config(self) -> dict:
        """Access config used during fitting."""        
        return self._config  
      
    @property
    def model(self) -> FitModel:
        """
        Fitter underlying model for inference.
        Raises Exception if accessed prior fit().
        """
        return self.fitter.model
        
    @property
    def features(self) -> list:
        """Fitter resulting chosen features for inference."""
        if not self.is_fitted():
            return None
        self._validate_worker_design('features')
        return self.fitter.features
        
    @property
    def result(self) -> FitResult:
        """Fitter fitted result cointainer → instance of FitResult dataclass."""
        if not self.is_fitted():
            return None
        self._validate_worker_design('result')
        return self.fitter.result

    @property
    def stats(self) -> FitStats:
        """Fitter stats created during fit to supplement dashboard."""
        if not self.is_fitted():
            return None
        self._validate_worker_design('stats')
        return self.fitter.stats    

    @property
    def dash(self) -> FitDash:
        """Fitter extra dash link created during fit to supplement dashboard."""
        if not self.is_fitted():
            return None
        self._validate_worker_design('dash')
        return self.fitter.dash