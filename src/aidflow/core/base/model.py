"""
The architectural backbone and master blueprint for all AIDflow Model implementations.

This module defines the mandatory structural hierarchy that every model must 
follow to be compatible with the AIDflow orchestration engine. By enforcing a 
strict inheritance chain—comprising processing, transformation, fitting, and 
UI layers—it enables an "auto-plug" mechanism where new models are instantly 
integrated into the global pipeline with full awareness of their I/O signatures.

The Model backbone manages four critical lifecycle dimensions:
1. Execution Flow: Orchestrates the sequence from Model → Processor → Transformer → Fitter.

2. Dynamic UI Generation: Automatically derives interactive parameter panels via `OptionUIBase` 
   and `WorkerUIBase`, bridging the gap between code and user configuration.

3. State Synchronization: Maintains a unified `ModelState` that tracks fit results, 
   hyperparameters, and statistical metadata across the model's lifetime.

4. Pipeline Integration: Provides standardized `fit_transform` and `update_transform` 
   hooks, ensuring consistent behavior during both initial training and production 
   inference phases.

Any class inheriting from `Model` is guaranteed to be a first-class citizen 
within the AIDflow ecosystem, inheriting full support for lineage tracking, 
dashboarding, and parallel execution.
"""

from ...deps.common     import  wraps
from ...deps.struct     import  Enum
from ...deps.data       import  pDF
from ...deps.types      import  ( 
                                Union,
                                Generic,
                                TypeVar,
                                Callable,
                                Optional,
                                Type,
                                FuncType
                                )

from ..parameters       import  (
                                AIGFLOW,
                                Parameters, 
                                PRM_links,
                                )
from ..registry         import  (
                                ProjectModelRegistry,
                                ModelRegistry,
                                FitRegister,
                                FitConfig,
                                FitRegOpt, 
                                DashRegister, 
                                DashRegistry,
                                RegGet
                                )
from ..factory          import  (
                                Dataclass, 
                                DictClass,
                                DictDataclass,
                                ValidationDataclass,
                                MappingDataclass
                                )
from ..foundation       import  (
                                AigParam, 
                                AigStop,
                                Dynamic
                                )
from ..io               import  (
                                ReportIO,
                                MetaIO
                                )
from ..display          import  (
                                DashMixin, 
                                DashInterface,
                                PanelConstructor
                                )
from ..display.repr     import  (
                                REPR, 
                                InputsRepr,
                                ModelRepr,
                                ReprCreator
                                )
from ..logging          import  (
                                Logger, 
                                LogSignature
                                )
from .fitted            import  (
                                FitterDataRelay,
                                ModelDataResult,
                                FittedResult
                                )

MODEL = AIGFLOW
FREG  = FitRegister
DREG  = DashRegister
DESC  = 'AIG STAGE'


class FIT(DictClass):
    """fit-update | fit-transform auto link constants."""      
    UPDATE      = '_fit_update_data_link'
    TRANSFORM   = '_fit_transform_inputs_link'
    SYNC        = '_background_task_sync'
    SPEC        = 'spec'
    STATS       = 'stats'
    FITTED      = 'fitted'
    RESULT      = '_result'
    GRESULT     = '_get_result'

class SIGN(DictClass):
    """Persistance signature model - fitter."""      
    MS = 'model_signature'


@Dataclass.base
class ModelConfigOverrides:
    """
    Pipeline Signature Model Input, providing flexible
    Configuration Overrides.

    Inherited by all pipeline model inputs (`ModelInput`).
    Allows the user to experiment and steer any pipeline step
    without modifying the underlying flow context.

    All fields are optional. When set, they override the
    corresponding value resolved from the flow context
    (`ctx.setup`). When None, the flow default is used.

    Attributes
    ----------
    target : str | None, default=None
        Target column name. Overrides the flow-level target
        for this specific model call. Useful when running
        multiple target experiments from the same pipeline.

    time_var : str | None, default=None
        Time/date column name. Used for:
        - Time-based (chronological) train/test splits.
        - OOT (out-of-time) evaluation logic.
        - Temporal ordering in the stabilizer stage.

        If None, random (stratified) splits are used.

    fx_choice : str | None, default=None
        Strategy identifier for this model step. Overrides
        the default choice registered in the flow. Enables
        switching algorithms without modifying the pipeline
        definition.

    fx_params : dict | None, default=None
        Runtime parameter overrides for the chosen strategy.
        Merged over the flow-level parameters at resolution time.

    fx_config : dict | None, default=None
        Configuration overrides for the chosen strategy. Merged 
        over the flow-level configuration at resolution time.

    save_path : str | None, default=None
        Custom save location for the model artifact. If
        None, the model is saved to the default flow
        directory.

    verbose : int | None, default=None
        Display verbosity level for this model step.
        Controls progress output and diagnostic display.
        If None, inherits the flow-level verbosity.
    """
    target:     str  = None
    time_var:   str  = None
    fx_choice:  str  = None
    fx_params:  dict = None
    fx_config:  dict = None
    save_path:  str  = None  
    verbose:    int  = None 

class ModelInput(ModelConfigOverrides):
    def _repr_html_(self):
        t = {REPR.T: type(self).__name__}
        p = InputsRepr.get_params(self)
        h = InputsRepr.help()
        return ReprCreator(t, p, h).html()   

@Dataclass.base
class ModelEnvelope:
    """Model Envelope Input Signature Design."""
    inputs: ModelInput
    config: dict | list | tuple | type

@Dataclass.base
class ModelResult:
    """Model Output Signature Design."""


class Validator(Enum):
    fitter = (
        "Model's Fitter must be defined as Fitter(FitterBase)"
        "\n ► Where FREG and DREG are linked to local FREG and DREG respectively from registers."
    )
    fitter_transformer = (
        "Model's Fitter must be defined in Transfomer body by 'ModelFitter'."
    )
    processor_base = (
        'Define Processor as Processor(ProcessorBase)'
        '\n ► Where local Transformer is linked in the Processor class body by "ModelTransformer".'            
    )
    processor_choice = (
        'Define Processor as Processor(ProcessorChoiceBase)'
        '\n ► Where MODEL, DESC and Config are linked to local MODEL, DESC and FitParamsCfg respectively.'
    )
    worker = (
        "Worker's Model must be defined in ModelUI body by 'Model'. OptionUI link by 'Option'."        
    )
    worker_base = (
        "Worker(WorkerBase) must define its MODEL, TITLE, Worker, Input, Result."        
    )    
    def __call__(self, *args):
        if None in args:
            raise RuntimeError(f'\n ► AIG SAFETY ► Model Design ► {self.value}')
        

class IOBase:

    _sanitize: list = [FIT.UPDATE]

    def __aig_copy__(self):
        """Prepares the state for sanitization."""
        model = object.__new__(type(self))
        state  = self.__getstate__()
        model.__setstate__(state)
        return model
    
    def __getstate__(self) -> dict:
        """Prepares the state for pickling."""

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

class NoReport:
    def __init__(self, model:str):
        self.model = model

    def __getattr__(self, name):
        print(f'{self.model} has no active report.', end='\r')

@Dataclass.base
class InternalFitterParams:
    dummy: str = 'T'

class FitParamsMapping(MappingDataclass):
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(InternalFitterParams)
    setup  = {
        'flow_mapping_to': params.dummy
    }

class FitParamsCfg:
    """Flow data params input design."""      
    ...

class FitParamsBase:
    """Outliers params input design."""
    ...

class FitParams(FitParamsBase):
    """Fitter params input design."""
    ...

class FitOptions(FitRegOpt):
    """Module Fit Options - Each Model is Auto Registered to the main REGISTRY."""       
    
    # @FREG.register(MyModel)
    class XFitter(FitParams):
        config: FitParamsCfg
        def _build(self): ...        

class FitStandards:
    """Functions used to enforce data standard type."""
    ...

class FitResult:
    """Fitter output design."""
    ...

class FitStats:
    """Fitter output design."""
    ...

FitParamsT = TypeVar('FitParamsT', bound=FitParams)

class FitterBase(Generic[FitParamsT], IOBase, DashMixin, ValidationDataclass):
    """
    MODEL Fitter Base.

    *Fitter may define:*
    - self.result = self.model.get_result() → result provides auto offload capability
    - self.stats  = self.model.get_stats()
    
    - self._relay = Dataclass → _relay provides auto offload capability

    *Default sanitization:*
    - _sanitize: list = ['result', 'stats', '_dashboard', FIT.UPDATE, FIT.SYNC]

    *FitterBase Provides standard init:*
    
    >>> 
    model  = None
    result = None by FittedResult()
    stats  = None
    _relay = None by FitterDataRelay()

    >>> 
    def __init__(
        self, 
        fx_choice:  RegGet.Choice | str | dict, 
        params:     FitParams, 
        setup:      Parameters = None, 
        verbose:    bool = True
    ):
        self.fx:     FitConfig = self.FREG.get(fx_choice)
        self.params: FitParams = params
        self.setup: Parameters = setup     
        self.verbose:     bool = verbose        
    >>>     
        self.model = self.fx.build(params=self.prm)                
        self._fx   = fx_choice   
    """  
    _sanitize: list = ['stats', '_dashboard', FIT.UPDATE, FIT.SYNC]
    
    FREG: FitRegister  = None
    DREG: DashRegister = None

    model:  FitConfig  = None
    result: FitResult  = FittedResult()
    stats:  FitStats   = None

    _relay = FitterDataRelay()
    _is_no_model: bool = False

    def __init__(self, 
        fx_choice:  RegGet.Choice | str | dict, 
        params:     FitParamsT, 
        setup:      Parameters = None, 
        verbose:    bool = True
    ):   
        Validator.fitter(self.FREG)

        self.fx = self.FREG.get(fx_choice)
        self.model = self.fx.build(params = params)
        
        self.params: FitParamsT = params
        self.config: dict = fx_choice.config
        
        self.setup      = setup
        self.verbose    = verbose        
        self.fx_choice  = fx_choice

    def _init_dash(self, verbose: None|bool = None, **kwargs):
        """
        Registred Dashboard init pipeline. All kwargs go to the fitter's dashboard.
        
        **Direct params:**
        - verbose : verbosity level; None → resolved by self.verbose
        
        Parameters
        ----------
        build_dash_after_init : bool = None
            - None → `verbose rule` → verbose = True → .build() after init.
            - False → do not trigger .build() after init; True otherwise.
        build_at_background : bool = False
            True → Async build at the background by BackgroundProcess; False otherwise.
        with_background_sync : threading.Event = None
            When build_at_background = True and threading Event via with_background_sync
            provided, once dashboard is constructed, Event is set as ready.
        **kwargs : 
            Additional parameters propagated to the dashboard constructor.        
        """
        Validator.fitter(self.DREG)
        verbose = AigParam.di(verbose, self.verbose)
        self.DREG.init(self.fx_choice, self, verbose=verbose, **kwargs)
        self._log_summary(kwargs)
    
    def _log_summary(self, kwargs: dict) -> None:
        """Activity log only (no UI output)."""
        if any([
            self.verbose,                         
            not self.setup.logging,
            kwargs.get('build_dash_after_init')
        ]):
            return
        
        sync = kwargs.get('with_background_sync')
        summary = lambda x=None: self.dashboard.summary(False)

        if sync is None:
            summary()
        else:
            Logger.background_logger(sync, summary)

    def _fit_no_model(self, model:str, features:list, result:"FitResult" = None) -> bool:
        """
        Features needed for a model fit:
        - no features → _is_no_model = True; return True
        - otherwise False; False

        Optionally when True and result instance provided:
        - self.result = Dataclass.create_default_instance(result)
        """
        if features:
            return False
        self._is_no_model = True
        if result is not None:
            self.result = Dataclass.create_default_instance(result)
        i = [f'Model {model} not Applicable', 'No Features Available']
        if self.verbose:
            print(' ► '.join(i)) 
        Logger(self).info(' - '.join(i))
        return True

    @property
    def MODEL(self) -> str:
        """AIG model ID."""
        return self.FREG.MODEL
    
class Fitter(FitterBase):
    """MODEL Fitter."""   

    ...

    def fit(self): 
        self.result = FitResult(...)
        self._init_dash()        
        return self
    
    def update(self): 
        self.result = FitResult(...)
        self._init_dash()
        return self
    
    def transform(self, data:pDF): 
        return data


class TransParams:
    """Transformer params input design."""    

class TransResData:
    """Transfomer data output design."""
    ...

class TransResCols:
    """Transfomer cols output design."""
    ...
    
class TransUtils:
    """MODEL Transformer Utils."""    
    ...


FitterT = TypeVar('FitterT', bound = FitterBase)
FitStatsT = TypeVar('FitStatsT', bound = FitStats)

TransformerC = TypeVar('TransformerC', bound = TransResCols)
TransformerR = TypeVar('TransformerR', bound = TransResData)

class _IsFitted:

    @staticmethod
    def verify(instance) -> bool:
        if isinstance(instance, ModelProcessorBase):
            return instance.processor._fx is not None
        elif isinstance(instance, ProcessorBase):
            return instance._fx is not None
        elif isinstance(instance, TransformerBase):
            return instance.fitter is not None
        
class _IsNoModel:

    @staticmethod
    def verify(instance) -> bool:
        if isinstance(instance, ModelProcessorBase):
            instance: "TransformerBase" = instance.fx
        fitter: FitterBase = instance.fitter
        return fitter._is_no_model
    
class _FitStatsT:
    """IDE aware link to Fitter `stats` (property breaks Generic[...])."""
    def __init__(self, link: Callable):
        self.link  = link

    def __get__(self, instance: "TransformerBase | ModelProcessorBase", owner) -> FitStatsT:
        if not _IsFitted.verify(instance):
            return        
        if _IsNoModel.verify(instance):
            return
        stats = self.link(instance)
        if stats is None:
            # View to Model's state after save
            model = getattr(instance, 'MODEL', None) # defined in Processor
            setup = getattr(instance, 'setup', None) # defined in Processor
            if None in [setup, model]:
                return
            try:
                stats = MetaIO(setup).load(model)
            except:
                pass
        return stats
    
    def __set__(self, instance, owner):
        raise RuntimeError('AIG SAFETY ► Link to Fitter stats is read only.')
    

class TransformerBase(Generic[FitterT, FitStatsT, TransformerC], IOBase, ValidationDataclass):
    """
    MODEL Transformer: Allowing meta-processing between Fitter-Model.
    
    Provides standard init:
    
    >>> 
    def __init__(
        self, 
        fit_choice: RegGet.Choice, 
        fit_params: FitParams, 
        setup:      Parameters,
        trn_params: TransParams = None,
        verbose:    bool = True
    ):
        self.f_fx    = fit_choice
        self.f_prm   = fit_params
        self.setup   = setup  
        self.t_prm   = trn_params
        self.verbose = verbose
    """
    ModelFitter: FitterT = None
    
    fitter: FitterT = None
    stats:  FitStatsT = _FitStatsT(lambda x: x.fitter.stats)
    cols:   TransformerC = None

    def __init__(self, 
        fit_choice: RegGet.Choice, 
        fit_params: FitParamsT, 
        setup:      Parameters,
        trn_params: TransParams = None,
        verbose:    bool = True
    ):
        Validator.fitter_transformer(self.ModelFitter)

        if (
            not isinstance(fit_choice, RegGet.Choice) or
            not isinstance(setup, Parameters)
        ):
            raise TypeError(
                '\n ► AIG SAFETY ► Transformer signature must follow mandatory design:'
                '\n ► 1. fit_choice, 2. fit_params, 3. setup, 4.trn_params'
            )
        
        self.f_fx    = fit_choice
        self.f_prm   = fit_params
        self.setup   = setup    
        self.t_prm   = trn_params
        self.verbose = verbose
    
    @property
    def fitted(self) -> FitResult | None:
        """
        Transformer link to its `fitter` result state → connected to ModelBase.fitted.
        - Direct use for Local Transformer which `TransResData inherits directly from Fitter result`.        
        - Local impl. req. for Transformer doing extras on Fitter result → mapping Fitter-Transformer result.
        """
        if self.fitter is not None:
            return self.fitter.result

    @property
    def dashboard(self) -> DashInterface | None:
        """
        The dashboard of fitted fitter or None.

        Safe access with a clear error if the dashboard has not
        yet been created (i.e., the primary fit step has not
        been executed, or the background thread has not
        completed).

        Raises
        ------
        AigStop
            If accessed before the parent worker has executed
            its primary fit/process step.

        Notes
        -----
        When `build_at_background=True`, this property may raise
        if accessed before the background thread completes.
        Use `with_background_sync` (threading.Event) to
        synchronize if sequential access is required.
        """        
        if self.fitter is None:
            return
        return self.fitter.dashboard
    
    def fit(self, inputs: ModelInput, *args, **kwargs):
        """
        Standard instantiation and fit of Model's Fitter:
        
        >>>
        self.fitter = Fitter(self.f_fx, self.f_prm, self.setup, verbose = self.verbose)
        self.fitter.fit(data)        
        """
        self.fitter = self.ModelFitter(self.f_fx, self.f_prm, self.setup, verbose=self.verbose)
        args, kwargs = self._route_fit_data(inputs, args, kwargs)            
        self.fitter.fit(*args, **kwargs)
        self._has_set_cols()
        return self
    
    def fit_update(self, *args, **kwargs):
        """
        Update Fitter Output with respect to user dynamic TH change in UI.

        This method triggeres 
        1. fitter.update()
        2. _set_cols().
        """
        self.fitter.update(*args, **kwargs)
        self._has_set_cols()
        return self
    
    def trans_update(self):
        """
        Update Transformer Output with respect to user dynamic features choice in UI.

        This method triggeres `_set_cols()` without any fitter refit.
        Intended for custom feature selection by CUSTOM UPDATE.
        """
        self._has_set_cols()
        return self

    def _fit_data(self, inputs: ModelInput, *args, **kwargs) -> pDF | tuple[pDF]:
        """
        `Prior` fitter.fit template method to define `input data to fitter.fit`.
        """
    
    def _set_cols(self) -> None:
        """
        `Post` fitter.fit template method to set Transformer resulting `self.cols`.
        """

    def _route_fit_data(self, inputs: ModelInput, args: tuple, kwargs: dict) -> pDF | tuple[pDF]:
        """
        Routing logic with respect to Transformer implementations.
        """
        # 1. Explicit Route → Tranformer defined routing to fitter.fit
        if '_fit_data' in type(self).__dict__:
            d = self._fit_data(inputs, *args, **kwargs)
            if isinstance(d, DictDataclass):
                return (d,), {}
            if isinstance(d, dict):
                return (), d
            if not isinstance(d, tuple):
                d = (d,)
            return d, {}
        
        # 2. Direct Route → inputs doesn't contain data for fitter.fit
        if args or kwargs:
            return args, kwargs
        
        # 3. Implicit Route → data are present in the default attr inputs.data 
        if isinstance(inputs, ModelInput):
            return (inputs.data,), {}

    def _has_set_cols(self) -> bool:
        if '_set_cols' in type(self).__dict__:
            self.cols = self._set_cols()
     

class Transformer(TransformerBase[Fitter, FitStats, TransResCols]):
    """MODEL Transformer: Allowing meta-processing between Fitter-Model."""

    def fit(self, inputs: ModelInput, *args): 
        super().fit(inputs.data, *args)
        self.cols : TransformerC  = TransResCols(...)
        return self 
    
    def fit_update(self): 
        self.fitter.update(...)
        return self
    
    def trans_update(self):
        self.cols: TransformerC = TransUtils(...)
        return self
    
    def transform(self, data:pDF): 
        d = self.fitter.transform(data)
        return TransResData(...)


TransformerT = TypeVar('TransformerT', bound = Transformer)


class _TransfromerT:
    """IDE aware link to Transfromer `fx` (property breaks Generic[...])."""
    def __init__(self, link: Callable[[type], type], verify: bool = False):
        self.link   = link
        self.verify = verify
    
    def __get__(self, instance: "ProcessorBase | ModelProcessorBase", owner) -> TransformerT:
        transformer = self.link(instance)
        if self.verify: 
            AigStop.fx(transformer, instance.desc, 'pred')
        return transformer
    
    def __set__(self, instance, owner):
        raise RuntimeError('AIG SAFETY ► Link to Transformer instance is read only.')
    
class _TransfromerC:
    """IDE aware link to Transfromer `cols` (property breaks Generic[...])."""
    def __init__(self, link: Callable[[type], type]):
        self.link = link

    def __get__(self, instance: "ProcessorBase | ModelProcessorBase", owner) -> TransformerC:
        if not _IsFitted.verify(instance):
            return
        return self.link(instance)
    
    def __set__(self, instance, owner):
        raise RuntimeError('AIG SAFETY ► Link to Transformer cols is read only.')


class ProcessorBase(Generic[TransformerT, TransformerC, TransformerR]):    
    """
    Transformer Processor: Validation and delegation to Transformer.
    """

    ModelTransformer: Type[TransformerT] = None  # module must define local Transformer in local Processor
    
    _fx: TransformerT = None

    fx:   TransformerT = _TransfromerT(lambda x: x._fx, verify=True)
    cols: TransformerC = _TransfromerC(lambda x: x.fx.cols)

    
    def __init__(self, stop_desc: str = DESC, transformer: Type[TransformerT] = None):
        self.ModelTransformer: Type[TransformerT] = transformer or self.ModelTransformer
        Validator.processor_base(self.ModelTransformer)
        self.desc = stop_desc        
    
    def _set_signature(self):
        setattr(self, SIGN.MS, LogSignature.get(self.fx.fitter.MODEL))        
    
    @staticmethod
    def _verify(verify:bool|None) -> bool:
        if verify is None:
            return True
        return verify
    
    def fit(self, 
        inputs: ModelInput, 
        /, 
        *args,
        fx_choice:  RegGet.Choice = None,
        params:     tuple[FitParams, Parameters, Optional[TransParams]] = None, 
        verify:     bool  = None,
        **kwargs
    ):
        """Processor of Transformer.fit() with verification."""
        with Logger(self.ModelTransformer).model_logger(fx_choice, inputs, params[0]):
            AigStop.fx(self._verify(verify), self.desc, 'fit')
            self._fx = self.ModelTransformer(fx_choice, *params, verbose = inputs.verbose)
            self._fx.fit(inputs, *args, **kwargs)
            self._set_signature()
            return self

    def update(self, data:pDF, /, *args, **kwargs):
        """Processor of Transformer.fit_update()."""
        with Logger(self.ModelTransformer).model_logger(args=args, kwargs=kwargs):
            self.fx.fit_update(data, *args, **kwargs)
            self._set_signature()
            return self
    
    def transform(self, 
        data: pDF, 
        /, 
        *args, 
        verify: bool = None, 
        **kwargs
    ) -> TransformerR:
        """Processor of Transformer.transform() with verification."""
        AigStop.cols(self.desc, verify)
        if ( data is None ) or ( not len(data) ):
            return None
        return self.fx.transform(data, *args, **kwargs)

class LocalProcessor(ProcessorBase):
    """Transformer Update Processor."""  

    fx: Transformer

    def update(self, th_ts:float) -> None:
        """Update fitted function."""
        if th_ts is None: 
              self.fx.trans_update()
        else: self.fx.fit_update(th_ts)  

    @staticmethod
    def _fx_with_signature(fx: Callable) -> Callable:
        def wrap(self:ProcessorBase, *args, **kwargs):
            self._set_signature()
            return fx(self, *args, **kwargs)
        return wrap

    @classmethod
    def _with_signature(cls, processor: ProcessorBase) -> Callable:
        out = (staticmethod, classmethod, property)

        methods = [
            name for name, value in vars(processor).items()
            if callable(value) and not name.startswith('__')
            and not isinstance(value, out)
        ]        
        for m in methods:
            fx = getattr(processor, m)
            setattr(processor, m, cls._fx_with_signature(fx))

        return processor

    @classmethod
    def _get_base(cls, processor:Union["LocalProcessor", None]) -> ProcessorBase:
        if processor is None:
            return ProcessorBase        
        if not issubclass(processor, ProcessorBase):
            raise TypeError('AIG SAFETY ► Custom Processor Update must use "ProcessorBase" as its base.')
        return cls._with_signature(processor)

class ModelProcessorBase(Generic[TransformerT, TransformerC, TransformerR], ValidationDataclass):
    """
    Model's standard processing pipeline → Fitter Choice+Params → Processor → Transformer → Fitter.
    - Model must have Processor on the left side in the inheritance chain follwed by ModelBase.

    **Local Processor must define:**
    - MODEL (str) → link to local MODEL
    - DESC (str) → link to local DESC
    - Config (type) → link to local FitParamsCfg
    - Mapping (type) → link to local FitParamsMapping
    - ModelTransformer (type) → link to local Transformer

    *Optional*:
    - ExtendedProcessor (type) → link to LocalProcessor to tweak ProcessorBase behavior on specific methods.
    
    **Init**
    - creates link to processor responsible for `verification and delagation to the Transformer`:
    - get_fit_config() → standard functionality to create fx_config :
      - fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
    - get_fit_params() → standard functionality to create fx_params :
     - fx_params = self.get_params(fx_choice, inputs.fx_params, params):

    >>>
    def __init__(self, setup:Parameters):
        self.processor = ProcessorBase(DESC, self.ModelTransformer)
        self.setup = setup    
    """   

    MODEL:   str = None
    DESC:    str = None

    Config:  FitParamsCfg = None
    Mapping: FitParamsMapping = None 

    ModelTransformer:  Type[TransformerT] = None
    ExtendedProcessor: LocalProcessor = None

    processor: ProcessorBase[TransformerT, TransformerC, TransformerR]

    fx:    TransformerT = _TransfromerT(lambda x: x.processor.fx)
    cols:  TransformerC = _TransfromerC(lambda x: x.processor.cols)
    stats: FitStatsT    = _FitStatsT(lambda x: x.fx.stats)

    def __init__(self, setup: Parameters):
        Validator.processor_choice(self.MODEL, self.DESC, self.Config, self.Mapping)        
        processor_base = LocalProcessor._get_base(self.ExtendedProcessor)

        self.processor = processor_base(self.DESC, self.ModelTransformer)
        self.setup = setup

    @property
    def fitter(self) -> FitOptions:
        """Model access to chosen internal Fitter responsible for computation."""
        if not _IsFitted.verify(self):
            return
        return self.fx.fitter.model
    
    @property
    def report(self) -> DashInterface | None:
        """Model report access."""        
        if not (
            self.MODEL in DashRegistry and 
            len(DashRegistry[self.MODEL])
        ):
            return NoReport(self.MODEL)
                
        if not _IsFitted.verify(self):
            return        
        if _IsNoModel.verify(self):
            return NoReport(self.MODEL)
        
        d = self.fx.fitter._dashboard
        if d is not None:
            return d
        return ReportIO(self.setup).load(self.MODEL)
        
    def get_model_dependecy(self, model:str|list) -> "ModelBase | dict[str, ModelBase]":
        """
        Returns model instance / dictionary populated by requested model instances 
        if they were fitted under current project, otherwise raise.
        """
        ts = self.setup.project_signature

        def get_model(m: str):
            model = ProjectModelRegistry.get_model(ts, m)
            if model is not None:
                return model
            raise ValueError(
                f'AIG SAFETY ► Model dependecy {m} must be fitted prior {self.MODEL} model fit.'
            )

        if isinstance(model, str):
            return get_model(model)        
        reg = {}
        for m in model:
            reg[m] = get_model(m)
        return reg
    
    def get_fit_config(self, fx_choice:str, fx_config:dict[dict], defaults:FitParamsCfg) -> dict:
        """
        Applies RegGet.config to create standard config by provided fx_choice and fx_config
        - defaults are returned when provided fx_config is empty.
        
        **Returns populated dict by FitParamsCfg signature + extras if provided in fx_config**.
        """ 
        config = RegGet.config(fx_choice, fx_config) # user custom config
        if not config: 
            return dict(defaults)
        
        config_ = {**config}
        for p,v in dict(defaults).items():
            config_[p] = AigParam.di(config.get(p), v)
        return config_
    
    def get_fit_params(self, fx_choice:str, fx_params:dict[dict], params:dict) -> FitParamsBase:
        """
        Applies mapping by provided `Mapping`
        - defaults to mapped flow params when params are empty.
        
        **Returns populated FitParamsBase**.
        """
        # 1. UI mode
        # incoming params by WorkerUI-OptioUI keys are by the flow design
        # filled by OptionUI - init run with ctx defaults (self.setup), 
        # rerun (CONFIRM button) with user dynamic changes
        if params:
            pass

        # 2. no UI → low verbosity mode → no OptionUI
        else:
            params = self.Mapping(self.setup, default=True)        
        
        # a. defaults
        fitter_params = self.Mapping(params)                 
        
        # b. user custom params → user provides fitter params keys directly
        fx_params_ = RegGet.config(fx_choice, fx_params) 
        
        if not fx_params_:
            return fitter_params
        
        params_ = {**fx_params_}
        for p,v in dict(fitter_params).items():
            params_[p] = AigParam.di(fx_params_.get(p), v)            
        
        return self.Mapping.create_prm_instance(params_)
    
class Processor(ModelProcessorBase[Transformer, TransResCols, TransResData]):
    """Transformer - Fitter Choice-Params - Processor."""    
    
    MODEL  = MODEL
    DESC   = DESC

    Config  = FitParamsCfg
    Mapping = FitParamsMapping

    ModelTransformer  = Transformer
    ExtendedProcessor = LocalProcessor    

    def get_choice(self, fx_choice:str, inputs:ModelInput):
        defaults = self.Config(
            time_var = inputs.time_var,
            verbose  = inputs.verbose
        )  
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  

    def get_params(self, fx_choice:str, inputs:ModelInput, params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params)
        fit = FitParams(...)
        trn = TransParams(...)
        return fit, self.setup, trn


@Dataclass.base
class ModelState:
    """Model Fitted State Design."""
    fitter:     object | FitRegister
    params:     tuple  | tuple[FitParams, Parameters] | tuple[FitParams, TransParams, Parameters]
    choice:     RegGet.Choice
    report:     DashInterface | NoReport
    features:   object | TransResCols
    stats:      object | FitStats = None

class ModelBase(IOBase):
    """
    MODEL Base: Mandatory signature to all AIG models.

    Each model must inherit from Processor and ModelBase, wheras ModelBase on the right.    

    Default sanitization:
    _sanitize: list = [FIT.INPUTS, FIT.UPDATE]

    *ModelBase Provides standard property:*
    - result
    - fitted
    - state
    - model_signature

    *fitted property*:
    - Default return is model's fitter result by link *self.fx.fitted*
    - By implementing `_get_result`, *fitted* will return *self._get_result(self.fx.fitted)*
    - Note that self.fx.fitted default resolution is Transformer's self.fitter.result

    *ModelBase Provides standard init:*
    
    >>> 
    fitter: FitterT       = None
    params: Parameters    = None
    choice: RegGet.Choice = None
    report: DashInterface = None
    cols:   TransformerC  = None
    stats:  FitStatsT     = None

    *ModelBase Provides standard fit-transform functionality for its worker:*
    
    >>> 
    fit_transform() → fit - transform
    update_transform() → update - transform
    

    Attributes
    ----------
    inputs : ModelInput
        Registered populated inputs containing execution timestamps and feature definitions.

    result : ModelResult
        Execution output payload containing model predictions and fit results

    fitter : Fitter
        Fitted state returns registered fitted fitter.

    state : ModelState
        Returns model state of all key attributes - fitter, choice, params, features, report, stats.

    report : DashboardInterface
        Interactive report interface. Call `report.show()` to render the visualization dashboard.

    stats : FitStats
        Performance metrics and training metadata gathered during fitting.

    features : TransResCols
        Feature schema dataclass required for downstream inference.

    params : tuple[FitParams, Parameters, ...]
       Tuple of fit settings, execution parameters, optionally transformer parameters.

    choice : RegGet.Choice
        Active configuration parameters including internal fitter choice (`fx_choice`).
    
    model_signature : str
        Unique hash identifier for the fitted model instance.
    """  
    _sanitize: list = [FIT.SPEC, FIT.UPDATE]
    _result = ModelDataResult()
    FIT.RESULT

    MODEL: str
    DESC:  str
    
    setup:     Parameters
    processor: ProcessorBase
    fx:        TransformerT

    fitter: FitterT         = None
    inputs: ModelInput      = None
    params: tuple           = None
    choice: RegGet.Choice   = None
    report: DashInterface   = None
    cols:   TransformerC    = None
    stats:  FitStatsT       = None

    def __str__(self):
        return f"Model: {self.MODEL} ► {self.DESC}"
    
    def __repr__(self):
        if not _IsFitted.verify(self):
            return f"Model: {self.MODEL} ► Not Fitted ► {self.DESC}"
        return self._repr()
    
    def _repr(self, is_worker:bool=False):
        return ModelRepr(self).help(is_worker)    

    def fit(self, inputs: ModelInput, fx_choice:str='AIG', **params) -> "ModelBase":
        self._update_data = None        
        self.choice = self.get_choice(fx_choice, inputs)
        self.params = self.get_params(fx_choice, inputs, params)
        raise NotImplementedError(
            'Workers must implement `fit` method prior fit_transform is called.'
        )
    
    def update(self):
        raise NotImplementedError(
            'Workers must implement `update` method prior update_transform is called.'
        )
    
    def transform(self, data:pDF):
        raise NotImplementedError(
            'Workers must implement `transform` method prior fit_transform is called.'
        )

    def _set_io(self, io:type) -> None:
        """Sets `inputs` to worker-model instance: dynamic context aware input/output."""
        setattr(self, FIT.SPEC, io)   

    def _set_update_data(self, data: FuncType) -> None:
        """
        Use this method for models with implemented `update()`. 
        Sets `data` to the model instance under recognized update link.

        **Data must be provided as a function, e.g. lambda: inputs.train_set.**
        """
        if not isinstance(data, FuncType):
            raise TypeError (
                'AIG SAFETY ► Update data must be provided as a function type.'
            )
        setattr(self, FIT.UPDATE, data)

    def _get_update_data(self) -> pDF | None:
        """Retrieves stored upadate data by `_set_update_data()`."""
        d = getattr(self, FIT.UPDATE, None)
        if d is not None:
            return d()   

    def _is_train_data(self, data:pDF) -> bool:
        """Verifies whether stored `update data` are equal to provided data."""
        return data is self._get_update_data()

    def _get_transform_inputs(self, args, kwargs) -> ModelInput:
        a_k = AigParam.get_ar_kw_values_by_sig(self.fit, args, kwargs)        
        inputs = a_k.pop(next(iter(a_k)), None) # inputs as mandatory first param
        
        if ( 
            inputs is None or
            not isinstance(inputs, (ModelInput, ModelEnvelope))
        ):  raise TypeError('Worker .fit() method must define data as the first arg WorkerInput type.')
        
        if isinstance(inputs, ModelEnvelope):
            return inputs.inputs
        
        return inputs
        
    def _fit_transform(self, f_t:str|None, args:tuple, kwargs:dict):
        sig = AigParam.get_definition(self.transform)
        if not sig:
            return self.transform()
        
        inputs = self._get_transform_inputs(args, kwargs)

        if not f_t:
            # (a) f_t is None → the first value is expected value for transform()
            cfg  = AigParam.get_definition(ModelConfigOverrides)            
            data = inputs[ inputs[len(cfg)] ]
        
        elif f_t == FIT.UPDATE: 
            # (b) use stored data in FIT.UPDATE link 
            data = getattr(self, f_t, None)
            data = Dynamic.data(data)

        else:
            # (c) use value by FIT.TRANSFORM link specified via fit_transform_inputs_link 
            data = inputs[f_t]
        
        return self.transform(data, **{p: inputs[p] for p in sig[1:] if p in inputs.keys()}) 
            
    def fit_transform(self, *args, **kwargs):
        """
        Pipeline fit-transform helper to allow train-test transformation at once.
        """
        f_t = kwargs.pop(FIT.TRANSFORM, None)  # provided by .model_ui.fit orchestrator
        self.fit(*args, **kwargs)
        return self._fit_transform(f_t, args, kwargs)

    def update_transform(self, *args, **kwargs): 
        self.update(*args, **kwargs)
        data = self._get_update_data()
        if data is None:
            raise TypeError(f'Worker .fit() method must store {FIT.UPDATE} when .update_transform() is engaged.')
        return self.transform(data)

    @property
    def result(self) -> ModelResult | None:
        """AIG Flow `Model` Result - Output."""        
        io = getattr(self, FIT.SPEC, None)
        if io is not None:
            return io.outputs

    def _get_result(self, res: TransformerR) -> ModelResult:
        """
        Implementing this method is a route declaration, AIG idiom, to use Fitter's result
        as the final Model training result.

        - By implementing `_get_result`, *fitted* returns *self._get_result(self.fx.fitted)*.
        - If not implemented, `fitted` returns `self.fx.fitted` as-is.
        """
        FIT.GRESULT

    def _has_get_result(self) -> bool:
        """Checks whether Model has implemented routing `_get_result` method."""
        return FIT.GRESULT in type(self).__dict__
    
    @property
    def fitted(self) -> ModelResult | TransformerR | FitResult | None:
        """
        AIG Flow link to Model | Fitter Result.
        
        - Default return is model's fitter result by link *self.fx.fitted*
        - By implementing `_get_result`, *fitted* returns *self._get_result(self.fx.fitted)*       
        """        
        if not _IsFitted.verify(self):
            return

        result = self.fx.fitted

        if self._has_get_result():
            return self._get_result(result)
        
        return result

    @property
    def features(self) -> TransformerC:
        """AIG Flow Model Features."""        
        return self.cols

    @property
    def state(self) -> ModelState:
        """AIG Flow Model State."""        
        return ModelState(
            fitter   = self.fitter,
            choice   = self.choice,
            params   = self.params,
            features = self.cols,
            report   = self.report,
            stats    = self.stats
        ) 

    @property
    def fitters(self) -> DictClass:
        """
        Returns all available fitters of the model that can be choosen for fit() by fx_choice.
        """
        return ModelRegistry[self.MODEL]
    
    @property
    def model_signature(self):
        """Returns model signature under which model was trained."""
        return getattr(self.processor, SIGN.MS, None)

    @property
    def is_fitted(self) -> bool:
        """True if the internal model has been fitted; False otherwise."""
        return self.model_signature is not None
    
    def get_project_signature(self) -> str:
        """Returns project signature under which model was trained."""
        return self.setup.project_signature

    def get_model_signature(self) -> str:
        """Returns model signature under which model was trained."""
        return self.model_signature
    
    def get_all_params(self) -> dict:
        """Returns all model parameters in one dict: params + config"""
        return dict(self.params[0]) | self.choice.config
    
class ModelBaseDual(ModelBase):
    """
    *ModelBaseDual for Workers with explicit Train-Test approach*

    MODEL Base: Mandatory signature to all AIG models.

    Each model must inherit from Processor and ModelBase, wheras ModelBase on the right.    

    Default sanitization:
    _sanitize: list = [FIT.INPUTS, FIT.UPDATE]

    *ModelBase Provides standard property:*
    - result
    - fitted
    - state
    - model_signature

    *ModelBase Provides standard init:*
    
    >>> 
    fitter: FitterT       = None
    params: Parameters    = None
    choice: RegGet.Choice = None
    report: DashInterface = None
    cols:   TransformerC  = None
    stats:  FitStatsT     = None

    *ModelBase Provides standard fit-transform functionality for its worker:*
    
    >>> 
    fit_transform - pipe fit - transform
    update_transform - pipe update - transform
    """     
    def fit_transform(self, *args, **kwargs):
        """
        Pipeline fit-transform helper to allow train-test transformation at once.
        To engage transform_dual, `inputs as WorkerInput param is required in both` 
        .fit() and .transform_dual() signatures to declare the intention of the worker.
        """
        f_t = kwargs.pop(FIT.TRANSFORM, None)  # provided by .model_ui.fit orchestrator
        
        self.fit(*args, **kwargs)

        if f_t is not None:
            return self._fit_transform(f_t, args, kwargs)
        
        inputs = self._get_transform_inputs(args, kwargs)

        sig = AigParam.get_definition(self.transform_dual)  
        return self.transform_dual(inputs, **{p: inputs[p] for p in sig[1:] if p in inputs}) 
    
    def transform_dual(self, inputs:ModelInput):
        raise NotImplementedError(
            'Workers must implement `transform_dual()` method prior fit_transform ' \
            'is called with WorkerInput as mandatory first data arg in transform_dual().'
        )

class AigModel(Processor, ModelBase):
    """AIG MODEL."""


class OptionUIBase:    
    """UI Dynamic Parameters Panel."""

    Mapping: MappingDataclass = None
    title:   str = 'AIG PANEL'
    split:   int = 2
    width:   int = 130
    
    def __init__(self, setup:Parameters, inputs:ModelInput):
        if self.Mapping is None or not AigParam.issubclass(self.Mapping, MappingDataclass):
            raise TypeError(
                'Wroker must implement parameters mapping → Model to inner Fitter. by '
                f'{MappingDataclass}'
            )
        self.prm = self._params(setup, inputs)
        self.wg  = PanelConstructor(self.prm, self.split, desc_width=f'{self.width}px')
    
    def _params(self, setup:Parameters, inputs:ModelInput):
        
        default_params = self.Mapping(setup, default=True)
        fx_params = RegGet.config(inputs.fx_choice, inputs.fx_params) 
        
        if not fx_params:
            return default_params
        
        params_ = {
            p_flow: fx_params.get(p_source) 
            for p_flow, p_source in self.Mapping._setup.items()
        }
        for p,v in dict(default_params).items():
            params_[p] = AigParam.di(params_.get(p), v) 

        return self.Mapping.create_default_instance(params_)
    
    def options(self):
        """Shows Main UI Options for ML fx fit."""    
        if self.Mapping is None:
            return
        desc = self.Mapping.get_desc()

        params = {}
        for p in self.prm.keys():
            if p not in PRM_links: 
                raise ValueError('Worker mapping must link Flow Parameters and inner Fitter params.')
            prm = PRM_links[p]
            params.update( prm.get_panel_defaults(desc[p]) ) 

        self.wg.params_to_sliders(params)

    def panel(self, *args, **kwargs):
        self.options()
        self.wg.panel(self.title)

    def get_ui_setting(self):
        return self.wg.get_ui_setting()
    
class ModelUIRunnerBase:
    """
    Wrapper around base model class:
    Adds UI choices and pass to a Worker for training given by the dynamic User Choices.
    """    
    _fitted = None

    @property
    def fitted(self): 
        """Obtain dynamic state of `Model` result."""
        return self._fitted

ModelT  = TypeVar('ModelT',  bound = AigModel)
OptionT = TypeVar('OptionT', bound = OptionUIBase)

class ModelUIBase(Generic[ModelT, OptionT]):
    """
    ModelUIBase mandatory signature for all AIG model UI layer\
    - direct bridge to model's worker.
    
    Transparent proxy with auto-execution of its Model fit-transform in one go:
    - run `fit()` followed by `transform()` (optional) and store output to `fitted`
    - construct UI features - Dynamic User Panel - Dashboard - Report
    
    ModelUI must define:
    - Model: ModelBase
    - Options: OptionUIBase | False to skip UI construction but provide model_ui API

    >>>
    def __init__(self, setup: Parameters, inputs: ModelInput, **kwargs): 
        self.model:     ModelT | AigModel = self.Model(setup, **kwargs)
        self.model_ui:  ModelUIRunner = ModelUIRunner(self.model, verbose=inputs.verbose)   
        self.option_ui: OptionT | OptionUIBase = self.Option(setup, inputs, **kwargs)    
    """
    
    Model:  ModelT  | AigModel = None
    Option: OptionT | OptionUIBase = None

    def __init__(self, setup: Parameters, inputs: ModelInput, runner_wg_id: int = None, **kwargs): 
        from .model_ui import ModelUIRunner

        Validator.worker(self.Model, self.Option)        

        self.model:     ModelT | AigModel = self.Model(setup, **kwargs)
        self.model_ui:  ModelUIRunner = ModelUIRunner(self.model, verbose=inputs.verbose, wg_id=runner_wg_id)   
        self.option_ui: OptionT | OptionUIBase = self.Option(setup, inputs, **kwargs) if self.Option else None
        
        self.fit_dynamic_input = AigParam.signature(self.model.fit, 1)

    def __getattr__(self, name):
        return getattr(self.model, name)
    
    def __repr__(self):
        return self.model._repr(True)

    @wraps(ModelBase.fit)
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    @wraps(ModelBase.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)
    
    @property
    def fitted(self) -> ModelResult: 
        """
        AIG Flow link to Model Result.
        
        Use standard UI link **self.model_ui.fitted**:
        - Models design with fit_transform_pipeline = True:        
          - returns result of fitted *model.transform()*

        - Models with fit_transform_pipeline = False:
          - returns result of *model.fitted*
          - i.e. model instance fitted property is exposed
          - most likely defining custom **_get_result()** in model is required
        """          
        return self.model_ui.fitted

    @property
    def state(self) -> ModelState: 
        """AIG Flow Model State."""
        return self.model.state

    @property
    def stats(self) -> FitStatsT: 
        """AIG Flow Model Stats."""
        return self.model.stats

    def panel(self, *args, **kwargs):  
        """Construct Choice Panel - Fitter options."""
        self.model_ui.panel(*args, **kwargs)

    def parent_input_data(self, *args, **kwargs):
        """Dynamically provided data params to fit method."""
        return Dynamic.get_parent_args_kwargs(args, kwargs, self.fit_dynamic_input)
    
    def inject_to_params(self, *args, **kwargs):
        """Extension of dynamic params to fit."""
        params = self.option_ui.get_ui_setting()
        kwargs = kwargs | dict(params)
        return args, kwargs
    
    def _get_ui_fit_defaults(self, output_wait:bool=False, **kwargs) -> DictDataclass:
        """
        Provides default kwargs to Model UI fit() with respect to shared dynamic design.
        """
        if self.option_ui is None:
            return self.model_ui.UIParams(
                args_kwargs_to_verify_na = self.fit_dynamic_input,
                parent_data_check_update = self.parent_input_data,
                show_ui                  = False,
                **kwargs
            )

        panel = kwargs.pop('gui_static_before_confirm_1', self.option_ui.panel)

        return self.model_ui.UIParams(
            args_kwargs_to_verify_na    = self.fit_dynamic_input,
            parent_data_check_update    = self.parent_input_data,
            gui_static_before_confirm_1 = panel,
            dynamic_to_args_kwargs_FX_4 = self.inject_to_params,
            output_wait                 = output_wait,
            **kwargs
        )