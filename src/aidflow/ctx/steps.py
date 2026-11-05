"""
Pipeline Step Context and Immutable I/O Registry.

This module implements the strict state management system for individual 
pipeline steps, ensuring that every worker's inputs and outputs are validated, 
registered, and locked upon execution. It serves as the primary auditing 
mechanism of AIDflow, preventing accidental modification of data during 
the pipeline flow and providing a transparent trace of model provenance.

Key architectural roles provided by this module:
1. Atomic I/O Locking: Utilizes a specialized `LockedDict` to enforce absolute 
   immutability of worker inputs and outputs. Once a step is registered, its 
   configuration is locked, ensuring that the data used for training or 
   inference cannot be altered mid-flight.

2. Dynamic Input Resolution: The `StepsContext` orchestrates the complex 
   merging of global setup parameters, user-provided overrides, and model-specific 
   defaults into a final, validated `ModelInput` object before execution.

3. Execution Provenance (Tracing): Automatically captures metadata for every 
   worker execution—including creation timestamps and processing durations—via 
   the `WorkerIORepr`, enabling detailed performance profiling and audit trails.

4. Type-Safe Registry: Enforces strict structural validation of inputs and 
   outputs through the `_WorkerIORegistry`, ensuring that only compatible 
   dataclass schemas are committed to the project state.
"""
                                
from ..deps.common          import  (
                                    threading,
                                    weakref
                                    )
from ..deps.weakref         import  WeakGetter
from ..core.parameters      import  Parameters
from ..core.registry        import  ProjectWorkerRegistry
from ..core.base            import  (
                                    FIT,
                                    ModelEnvelope,
                                    ModelInput, 
                                    ModelResult, 
                                    ModelConfigOverrides,
                                    ModelBase,
                                    ModelUIBase
                                    )
from ..core.factory         import  Dataclass
from ..core.foundation      import  (
                                    AigParam, 
                                    AigStop, 
                                    )
from ..core.display.repr    import  (
                                    REPR, 
                                    ReprCreator, 
                                    InputsRepr
                                    )
from ..core.accessor        import  Format
from ..utils.builtins       import  Misc
from ..display.rgb          import  RgbANSI



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# STEP INPUT CONTEXT - PROVIDES STANDARDIZED INPUT TO ALL AIG STEPS FOR TRAINING
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬   
class LockedDict(dict):
    """
    A dictionary that enforces strict immutability when locked.
    Designed for atomic 'Write-and-Lock' operations.
    """

    def __init__(self, data: dict = None, locked: bool = False):
        super().__init__(data or {})
        self._locked = locked
    
    @staticmethod
    def _warn(action:str=''):
        a = f'Cannot {action} ► ' if action else ''
        AigStop.custom(
            f"AIG SAFETY ► {a}Worker Inputs/Outputs are locked and immutable "
            f"► Write to storage of Worker is reserved to the pipeline workers.",
            bold=False, red=False
        )

    def _check_lock(self, action: str):
        if self._locked:
            self._warn(action)
        
    def __setitem__(self, key, value):
        self._check_lock("modify")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._check_lock("delete")
        super().__delitem__(key)

    def pop(self, *args, **kwargs):
        self._check_lock("pop")
        return super().pop(*args, **kwargs)

    def popitem(self):
        self._check_lock("popitem")
        return super().popitem()

    def clear(self):
        self._check_lock("clear")
        super().clear()

    def update(self, *args, **kwargs):
        self._check_lock("update")
        super().update(*args, **kwargs)

    def setdefault(self, key, default=None):
        self._check_lock("setdefault")
        return super().setdefault(key, default)

    def lock(self):
        self._locked = True
    
    def unlock(self):
        self._locked = False
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class InputsWithDefaultRepr:
    """Model Inputs with the standard dataclass __repr__ string."""
    
    def __init__(self, inputs: ModelInput):
        if not isinstance(inputs, (ModelInput, ModelEnvelope)):
            raise TypeError(
                f'AIG SAFETY ► Worker inputs must be created with signature '
                'of ModelInput | ModelEnvelope type'
            )
        self._inputs = inputs

    def __getattr__(self, name):
        return getattr(self._inputs, name)

    def __repr__(self):
        attrs = []
        for p in Dataclass.get_names(self._inputs):
            a = getattr(self._inputs, p)
            attrs.append(f"{p}={a!r}")        
        return f"{self._inputs.__class__.__qualname__}({', '.join(attrs)})"

    def _repr_html_(self):
        return None
    
class _WorkerRepr:
        
    def __init__(self, worker:"WorkerInputsRepr"):
        self.worker = worker
    
    def help(self) -> list:
        o = '' if isinstance(self.worker.trace, str) else ' & outputs'
        h = []
        h.append(
            f"► {RgbANSI.bold(f'Access to worker inputs{o}:')}"
            "\n• inputs           → View to actual worker's inputs used during pipeline execution"
            "\n• outputs          → View to actual worker's results during pipeline execution"
            "\n• inputs_class     → Worker inputs class for tuning/testing with custom inputs"
            "\n• config_overrides → Overview of optional worker's custom config overriding flow's default config"
        )
        h.append( InputsRepr.help() )
        h.append(
           f"\n► {RgbANSI.bold('Providing free parameters to a worker:')}"
            "\n• kwargs  → workers accept kwargs to allow direct assignment of parameters in the <i> inputs </i> signature"
            "\n• default → kwargs allow to provide any individual parameter while the rest of inputs are auto filled by the flow"
            "\n  -- for instance, worker1(fx_config = dict(n_estimators=500))"
            "\n  -- free param <i>fx_config = dict(n_estimators=500)</i> targets internal fitter config, the rest of inputs are provided by the flow"
        )        
        return h

    def html(self):
        t = {
            REPR.T: self.worker.inputs_class.__name__, 
            REPR.H: f"creation time: {self.worker.creation_time}"
        }
        help = []
        for i in self.help():
            help.extend([
                i, Format.splitter(show=False, style=Format.StyleSplitter(to_html=False))
            ])
        return ReprCreator(t, InputsRepr.get_params(self.worker.inputs), help).html()    

@Dataclass.base
class WorkerInputsRepr:
    model:              str
    inputs:             InputsWithDefaultRepr
    inputs_class:       type
    config_overrides:   ModelConfigOverrides
    creation_time:      str
    
    def _repr_html_(self):
        return _WorkerRepr(self).html()

    def __repr__(self):
        return self.help(False)
    
    def help(self, show:bool=True) -> None:
        h = '\n'.join(_WorkerRepr(self).help())
        if not show: return h
        print(h)

    @property
    def trace(self) -> str:
        return self.creation_time
    
    @property
    def outputs(self) -> str:
        print(f'{self.model} ► Outputs has not been registred yet, please run the worker to register it.')
    
    @property
    def creation_trace(self) -> str:
        print(f'{self.model} ► Outputs has not been registred yet, please run the worker to register it.')

@Dataclass.base(dynamic_params=True)
class WorkerIORepr(WorkerInputsRepr):
    outputs:        ModelResult
    creation_trace: dict

    @property
    def trace(self) -> dict:
        return self.creation_trace
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬    

STORAGE = '_storage'

class _WorkerIORegistry:
    """
    Manages the lifecycle and immutability of inputs for a specific pipeline step.
    
    This class acts as the internal engine for the `inputs` descriptor. It ensures that
    worker inputs are strictly typed, validated against defaults, and immediately locked
    (made immutable) upon registration. It prevents accidental modification of inputs
    after they have been committed to the step context.
    """
    __slots__ = (STORAGE,)

    def __init__(self):
        object.__setattr__(self, STORAGE, LockedDict())    

    def __setattr__(self, name, value):
        LockedDict._warn()
    
    def __getattr__(self, name:str):
        AigStop.jn_getattr(name)
        
        if name in self._model_io:
            return self._model_io[name]
        
        AigStop.custom(
            f"Model Input '{name}' is not registered." + self.stored_inputs_repr,            
            bold=False, red=False
        )

    @property
    def _model_io(self) -> LockedDict:
        return getattr(self, STORAGE)
    
    @property
    def stored_inputs_repr(self) -> str:
        return f"\nRegistred worker inputs:\n- {'\n- '.join(list(self._model_io))}"
    
    @staticmethod
    def _verify_type(inputs:ModelInput, defaults:ModelInput):
        inputs_cls = type(defaults)
        if type(inputs) is not inputs_cls:
            raise TypeError(f'AIG SAFETY ► Worker inputs must be created from: {inputs_cls}')
        
    def _verify_result(self, 
        model:   str, 
        worker:  ModelUIBase | ModelBase, 
        result:  ModelResult
    ):
        if not isinstance(self._model_io[model], WorkerInputsRepr):
            raise TypeError(f'AIG SAFETY ► Worker inputs must be created and registred prior outputs')
                
        if not isinstance(worker, ModelUIBase | ModelBase):
            raise TypeError(f'AIG SAFETY ► Worker results must be created based on type: {ModelUIBase|ModelBase}')

        if callable(result):
            result = result()

        if not isinstance(result, ModelResult):
            raise TypeError(f'AIG SAFETY ► Worker results must be created with signature of ModelResult type')
        
    @staticmethod
    def _get_overrides(inputs:ModelInput) -> ModelConfigOverrides:
        cfg = {p:v for p,v in inputs.items() if p in AigParam.get_definition(ModelConfigOverrides)}
        return ModelConfigOverrides(**cfg)

    def _add_trace(self, model:str, timestamp:str) -> list:        
        model:WorkerInputsRepr = self._model_io[model]
        return dict(
            inputs      = model.creation_time, 
            outputs     = timestamp, 
            processing  = Misc.time_diff(model.creation_time, timestamp)
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _register_inputs(self, model:str, inputs:ModelInput, defaults:ModelInput) -> None:
        """
        Registers a new input set for a specific model/step in the LockedDict.
        
        Execution Logic:
        1. Verify the type of `inputs` matches `defaults`.
        2. Temporarily UNLOCK the `LockedDict` to allow atomic writes.
        3. WRITE the new `WorkerInputRepr` object containing inputs, metadata, and overrides.
        4. LOCK the dictionary to ensure immutability.
        
        This guarantees that once this method returns, the data for `model` is immutable.
        
        Parameters
        ----------
        model : str
            The unique identifier or name of the step/model.
        inputs : WorkerInput
            Optional user data to store.
        defaults : WorkerInput
            The default pipeline worker data.
        """
        self._verify_type(inputs, defaults)            
        self._model_io.unlock()
        self._model_io[model] = WorkerInputsRepr(
            model            = model,
            inputs           = InputsWithDefaultRepr(inputs),
            inputs_class     = type(inputs),
            config_overrides = self._get_overrides(inputs),
            creation_time    = Misc.timestamp('human')
        )
        self._model_io.lock()
    
    def _register_outputs(self, 
        model:  str, 
        worker: ModelUIBase | ModelBase, 
        result: ModelResult = None
    ) -> None:
        """
        Registers a new outputs set for a specific model/step in the LockedDict.
        
        Execution Logic:
        1. Verify the type of `outputs`.
        2. Temporarily UNLOCK the `LockedDict` to allow atomic writes.
        3. WRITE the new `WorkerIORepr` object containing inputs, outputs, metadata, and overrides.
        4. LOCK the dictionary to ensure immutability.
        
        This guarantees that once this method returns, the data for `model` is immutable.
        
        Parameters
        ----------
        model : str
            The unique identifier or name of the step/model.
        worker : WorkerGUIBase
            Pipeline Worker which holds fitted results.
        """        
        if result is None:
              result = WeakGetter.Attr(worker, FIT.FITTED)
        self._verify_result(model, worker, result)
        self._model_io.unlock()
        self._model_io[model] = WorkerIORepr(
            **self._model_io[model],
            outputs = result,
            creation_trace = self._add_trace(model, Misc.timestamp('human'))
        )
        self._model_io.lock()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class _WorkerIOAccessor:
    """Data Descriptor: Entry point to the registry."""
    _lock = threading.Lock()
    
    def __init__(self):
        self._reg = weakref.WeakKeyDictionary()

    def __get__(self, instance:"StepsContext", owner) -> _WorkerIORegistry:
        if instance is None:
            return self
        with self._lock:        
            if instance not in self._reg:
                self._reg[instance] = _WorkerIORegistry()
            return self._reg[instance]
    
    def __set__(self, instance, owner) -> None:
        LockedDict._warn()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class StepsContext:
    """
    Context manager for pipeline worker inputs - outputs and configuration.
    
    Orchestrates the creation, validation, and registration of worker inputs and outputs
    for a specific pipeline step. It handles the resolution of configuration values
    (merging setup parameters, global overrides, and defaults) and ensures that once
    inputs are created, they are immediately registered and locked.

    Parameters
    ----------
    setup : Parameters
        The core Parameters instance providing global setup values.
    """    
    __slots__ = ('cfg_setup', 'cfg_over', '__weakref__')

    _steps  = _WorkerIOAccessor()
    
    def __init__(self, setup: Parameters):
        self.cfg_setup = setup
        self.cfg_over  =  ModelConfigOverrides()

    def __getattr__(self, name:str):
        if name == STORAGE:
            raise AttributeError
        return getattr(self._steps, name)
    
    def __repr__(self):
        return ( 
            'Pipeline Workers Inputs Registry and Access' + self._steps.stored_inputs_repr
        )
    
    @staticmethod
    def _keep_dynamic_inputs(inputs: ModelInput) -> dict:
        """Dynamic link keeping live connection to the worker.fitted."""
        inputs_ = {}
        for p in inputs.keys():
            inputs_[p] = (lambda p=p: inputs[p]) if inputs[p] is not None else None
        return inputs_

    def _create_inputs(self, 
        fx_choice:  str, 
        inputs:     ModelInput | None, 
        defaults:   ModelInput, 
        kwargs:     dict
    ) -> ModelInput:
        flow_defaults = dict(fx_choice = fx_choice) | {
            p: getattr(self.cfg_setup, p) for p in self.cfg_over if hasattr(self.cfg_setup, p)
        }
        defaults_ = self._keep_dynamic_inputs(defaults) | flow_defaults  # | +default overrides
        input_cls = type(defaults)        

        if ( inputs is None ) and ( not kwargs ):            
            return input_cls(**defaults_)
        
        inputs_d = {}
        if inputs is not None:
            _WorkerIORegistry._verify_type(inputs, defaults)        
            inputs_d = self._keep_dynamic_inputs(inputs)

        inputs_ = {}
        in_kw = kwargs | inputs_d   # inputs have precedence over free kwargs         
        for p, v in defaults_.items():
            inputs_[p] = AigParam.di(in_kw.get(p), v)
        
        return input_cls(**inputs_)
    
    def create_inputs(self, 
        model:      str, 
        fx_choice:  str, 
        inputs:     ModelInput, 
        defaults:   ModelInput | ModelEnvelope, 
        params:     dict={}
    ) -> ModelInput | ModelEnvelope:
        """
        Create, validate, and `register inputs` for a specific pipeline worker.
        
        This is the primary entry point for setting up inputs. It resolves the final
        parameter values (merging setup, overrides, and defaults), validates the
        structure, and atomically registers the result in the `InputsRegistry`.
        
        Parameters
        ----------
        model (str) : The unique identifier for the step or model.
        fx_choice (str) : The unique identifier for the choice of internal fitter of the model.
        inputs (WorkerInput) : Optional user-provided input data.
        defaults (WorkerInput) : The default pipeline input schema.
        params (dict) : Free parameters allowing direct assigment to input signature. 

        Returns
        -------
        WorkerInput
            The validated and registered input instance.
        """
        envelope_cls = None

        if isinstance(defaults, ModelEnvelope):
            envelope_cls = type(defaults)
            d_config  = defaults.config
            defaults  = defaults.inputs

        inputs_ = self._create_inputs(fx_choice, inputs, defaults, params)
        self._steps._register_inputs(model, inputs_, defaults)
        
        if envelope_cls is None:
            return inputs_
        
        return envelope_cls(inputs = inputs_, config = d_config)
    
    def register_inputs(self, model:str, inputs:ModelInput) -> None:
        """
        Register inputs for a specific pipeline worker.
        
        This is the secondary entry point to register inputs to the `InputsRegistry`.
        Aims workers / models which were fitted out of the interface scope.
        
        Parameters
        ----------
        model (str) : The unique identifier for the step or model.
        inputs (WorkerInput) : Optional user-provided input data.
        """
        self._steps._register_inputs(model, inputs, inputs)

    def register_outputs(self, 
        model:  str, 
        worker: ModelUIBase | ModelBase, 
        result: ModelResult = None
    ) -> None:
        """
        Validate and `register outputs` to a specific pipeline worker inputs.
                
        Parameters
        ----------
        model : str
            The unique identifier for the step or model.
        worker : WorkerUIBase | ModelBase
            Worker' model Fitted Instance as a dynamic link to fitted outputs.
        result: ModelResult, optional
            Worker result provided directly; AIG standard fetch dynamic data by worker.fitted.
        """         
        self._steps._register_outputs(model, worker, result)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
        
class WorkersContext:
    """
    Pipeline Workers (Nodes) Dynamic IO Context.
    
    Workers are a context aware orchestrators of models in a given flow step while
    holding live reference to pipeline input-output context as io of the models.
    """
        
    def __init__(self, setup: Parameters):
        self._io  = StepsContext(setup)       
        self._sig = setup.project_signature
    
    def __getattr__(self, name) -> type | None:
        AigStop.jn_getattr(name)

        w = self._get_worker(name)
        if w is not None:
            return w
                
        print(
            f'Requested worker {name} has not been registered, available workers:'
            f'{self._help()}'
        )

    def __repr__(self):
        return (
            'Pipeline Workers Dynamic Registry and Access'
            f'\nRegistred workers: {self._help()}' 
        )
    
    @property
    def io(self) -> StepsContext:
        """
        Context manager for pipeline worker inputs - outputs and configuration.
        
        Orchestrates the creation, validation, and registration of worker inputs and outputs
        for a specific pipeline step. It handles the resolution of configuration values
        (merging setup parameters, global overrides, and defaults) and ensures that once
        inputs are created, they are immediately registered and locked.        
        """
        return self._io

    def _get_worker(self, name: str) -> type | None:
        return ProjectWorkerRegistry.get_worker(self._sig, name)

    def _get_registred(self) -> list[str]:
        return ProjectWorkerRegistry.get_registred(self._sig)
            
    def _help(self) -> str:
        return f'\n - {"\n - ".join( self._get_registred() )}'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬