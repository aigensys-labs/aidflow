"""
Immutable model adapter for consistent inference via transparent lazy proxying.

This module implements lineage hash verification to ensure that the loaded 
model remains identical to its state at the time of creation. By validating 
the provenance hash, it eliminates risks associated with external dependency 
drift, ensuring absolute reproducibility during the inference phase.
"""

from ...deps.data               import  pDF
from ..structure                import  (
                                        AIGFLOW, 
                                        AIGMODELS
                                        )
from ..registry                 import  ProjectModelRegistry
from .model                     import  (
                                        ModelBase,
                                        ModelResult
                                        )
from .worker                    import  WorkerBase
    

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# MODEL → ADAPTER → DEPENDENCY
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ReportAdapter:
    """Creates independent report adapter for inference (transparent virtual lazy proxy)."""

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._report)))
    
    def __getattr__(self, name):
        if self.__dict__.get('_adapter') is None:
            raise AttributeError(name)        

        if name in self._persist:
            return self._persist_report_attr[name]
        
        try:
            self._adapter._model_route_io()
        except AttributeError as e:
            raise RuntimeError(f"Failed to load underlying model during lazy initialization: {e}") from e
        
        return getattr(self._report , name)

    def __init__(self, adapter: "AdapterBase"):
        self._adapter = adapter
        self._persist = adapter.PERSIST_FROM_REPORT or []
        self._extract_persist_attr()
    
    @property
    def _report(self):
        return self._adapter.model.report
    
    def _extract_persist_attr(self):
        self._persist_report_attr = {}
        for p in self._persist:
            self._persist_report_attr[p] = getattr(self._report , p)       
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class AdapterBase:
    """
    Creates independent model adapter for inference (transparent virtual lazy proxy).
    Stored ModelAdapter with lineage hash verification to eliminate any external dependency change.
    """
    PERSIST_FROM_REPORT = None    
    model  = None

    def __dir__(self):
        self._model_route_io()
        return sorted(set(super().__dir__()) | set(dir(self.model)))
    
    def __getstate__(self):
        state = self.__dict__        
        state = state.copy()
        state.pop('model', None)        
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None

    def __init__(self, model_instance: ModelBase | WorkerBase):
        self._verify_model(model_instance)
        self.model   = model_instance
        self.MODEL   = model_instance.MODEL
        self.project = model_instance.setup.project
        self.report  = ReportAdapter(self)
        self._name   = type(model_instance).__name__

        if isinstance(model_instance, ModelBase):
            self._set_signatures(model_instance)

    def __call__(self, *args, **kwargs):
        """Worker route active only during fitting."""
        w = self.model(*args, **kwargs)
        self._set_signatures(w)
        return w
    
    def __getattr__(self, name):
        self._init_model()
        return getattr(self.model, name)
    
    def get_model_signature(self) -> str:
        """Returns model signature under which model was trained."""
        return self._m_sign

    def get_project_signature(self) -> str:
        """Returns project signature under which model was trained."""
        return self._p_sign

    def _set_signatures(self, model_instance: ModelBase | WorkerBase):
        self._m_sign = self._get_model_signature(model_instance)
        self._p_sign = self._get_project_signature(model_instance)

    def _get_model_signature(self, model_instance: ModelBase) -> str:
        return model_instance.get_model_signature()

    def _get_project_signature(self, model_instance: ModelBase) -> str:
        return model_instance.get_project_signature()

    def _init_model(self):
        try:
            self._model_route_io()
        except AttributeError as e:
            raise RuntimeError(f"Failed to load underlying model during lazy initialization: {e}") from e
    
    def _model_route_io(self) -> None:
        if self.__dict__.get('project') is None:
            raise AttributeError
        
        if self.__dict__.get('model') is not None:
            return        
        
        self._set_predictor()        
        self.model: ModelBase = self.Predictor(self.project).get_model(self.MODEL)            
        
        self._verify_main_timestamp(self.model)

    def _set_predictor(self):
        if self.__dict__.get('Predictor') is not None:
            return
        from ...predictor.predictor import  Predictor        
        self.Predictor = Predictor

    @staticmethod
    def _verify_model(model: ModelBase) -> None:
        try:
            m = model.MODEL
        except: 
            m = None
        if m in AIGMODELS:
            return
        raise TypeError(
            f'\n ► AIG SAFETY ► Provided model {m} is not registred in AIG REGISTRY'
            f'\n ► Please use model from following: \n ► {'\n ► '.join(AIGMODELS)}'
        )

    def _verify_main_timestamp(self, model:ModelBase) -> None:        
        """Model timestamp verification."""
        
        m_ts = self._get_model_signature(model)
        
        if self._m_sign == m_ts:
            return        
            
        raise ValueError(
            f'\n► AIG SAFETY ► Loaded Model {self.MODEL} signature: {m_ts} differs '
            f'from the stored sig.: {self._m_sign}.'
            f'\n► Please refit and store current worker-model to align dependencies.'
        ) 
                       
    def _verify_dependecy_timestamp(self, model:ModelBase) -> None:        
        """
        Model dependency timestamp verification.
        Dependency timestamp must precede main model by the aigflow design.
        """
        
        m_ts = self._get_model_signature(model)        
        p_ts = self._get_project_signature(model)

        if ( m_ts < self._m_sign ) and ( self._p_sign == p_ts ):
            return

        raise ValueError('► AIG SAFETY'
            f'\n► 1. Loaded Dependecy Model {model.MODEL} signature must precede the Main Model '
            f'{self.MODEL} which uses it, \n  Main Model signature {self._m_sign} | dependency: {m_ts}.'
            f'\n► 2. Main Model and its dependency must be trained and stored under the '
            f'same project signature \n  project expected signature: {self._p_sign} | loaded: {p_ts}'                        
            f'\n► Please refit and store current worker-model and its dependecy.'
        ) 
    
class ModelAdapterBase(AdapterBase):
    """
    Creates independent model adapter for inference (transparent virtual lazy proxy).
    Stored ModelAdapter with lineage hash verification to eliminate any external dependency change.
    """

    def __repr__(self):
        return f'ModelAdapter({self._name})'
    
    def _get_model(self, model: str) -> ModelBase:
        m = ProjectModelRegistry.get_model(self._p_sign, model)
        if m is None:
            # loaded state wo active project
            p = self.Predictor(self.project)
            m = p.get_model(model)
        self._verify_dependecy_timestamp(m)        
        return m

    def _preprocess(self, data:pDF) -> pDF:
        """
        Data preprocess functionality if data phase dependencies have been stored 
        during project training.
        """
        c = AIGFLOW.Data.fe_cleaner
        s = AIGFLOW.Data.fe_stabilizer

        mc = self._get_model(c)
        ms = self._get_model(s)

        data_ = mc.transform(data).data
        return ms.transform(data_).data

    def transform(self, data, preprocess:bool=False, **kwargs) -> ModelResult:
        """
        Adapter extension of model API - returns prediction data - transform().data.

        Adds data preprocess functionality for model loaded from storage if data phase 
        dependencies have been stored during project training.
        """        
        self._model_route_io()
        
        if preprocess:
            data = self._preprocess(data)
        
        return self.model.transform(data, **kwargs)
    
    def predict(self, data, preprocess:bool=False, **kwargs) -> pDF:
        """Adapter extension of model API - returns transform data - transform().data."""        
        return self.transform(data, preprocess, **kwargs).data
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬