"""
Global Model and Dashboard Registry: The Central Service Locator for AIDflow.

This module implements the core registration and discovery mechanisms that 
allow AIDflow to dynamically manage models, fitters, and dashboards throughout 
the pipeline lifecycle. It acts as a high-performance Global State Service (Multiton), 
providing seamless access to fitted model instances and their corresponding 
configuration options via unique project signatures.

Key architectural components:
1. Model & Dash Registries: Automated discovery systems that populate all available 
   main MODELs and Fitter options upon package import, enabling a "Plug-and-Play" 
   experience for new ML components.

2. Three-Layer API Architecture: A strict structural enforcement for `FitConfig` 
   that decouples the Public API (Framework-Controlled), the Defining Layer 
   (Subclass-Implemented), and Custom Extensions, ensuring absolute stability 
   regardless of user customizations.

3. Dynamic Proxying & Building: Utilizes a lazy-loading proxy pattern to handle 
   the transition from a "Registered Fitter" (config) to a "Built Fitter" 
   (mutable instance with an internal estimator), optimizing memory usage and 
   initialization speed.

4. Project Provenance Tracking: Through `ProjectModelRegistry`, the system maintains 
   a mapping of fitted models tied to specific project signatures, preventing 
   cross-project state contamination and ensuring absolute auditability.
   
5. Integrated Safety Verification: A rigorous internal validation engine (`_Verify`) 
   that prevents name collisions and ensures all registered components adhere 
   to the mandatory AIDflow architectural contracts.
"""

from ..deps.common      import  (
                                weakref,
                                threading
                                )
from ..deps.types       import  (
                                Any,
                                Literal,
                                get_type_hints,
                                extract_types
                                )

from .parameters        import  Parameters
from .factory           import  (
                                DictClass, 
                                Dataclass, 
                                DictDataclass
                                )
from .foundation        import  (
                                AigParam, 
                                AigStop
                                )
from .structure         import  AIGMODELS
from .accessor          import  Format
from .display.dash      import  (
                                DashMixin, 
                                DashInterface
                                )
from .display.repr      import  (
                                REPR, 
                                ReprCreator,
                                ReprUtils
                                )
from ..display.rgb      import  RgbANSI

from ..utils.builtins   import  List


class ProjectRegistryBase:
    _storage: dict[str, weakref.WeakValueDictionary]
    _lock = threading.Lock()

    @classmethod
    def get_registred(cls, project_sign: str) -> list:
        """Provides registred models under project signature ID."""
        return list(cls._storage[project_sign])

    @classmethod
    def get_setup(cls, project_sign: str) -> Parameters | None:
        """Returns project's setup under project signature ID."""
        with cls._lock:
            if project_sign not in cls._storage:
                return
            
            registred = cls.get_registred(project_sign)
            if not registred:
                return
            
            project = cls._storage[project_sign]
            
            from .base import ModelBase
            ins : ModelBase = project[registred[0]]

            return ins.setup
    
    @classmethod
    def _register(cls, instance) -> tuple[str, str]:
        from .base import ModelBase, ModelUIBase
        from .base.worker import WorkerBase
    
        ins : ModelBase | ModelUIBase | WorkerBase = instance

        MODEL = ins.MODEL   
        _Verify.model(MODEL)

        p_sig = ins.setup.project_signature  

        with cls._lock:
            if p_sig not in cls._storage:
                cls._prune()
                cls._storage[p_sig] = weakref.WeakValueDictionary()

            cls._storage[p_sig][MODEL] = instance
    
    @classmethod
    def _get_model(cls, project_sign: str, model: str) -> object | None:
        """Provides instantiated model / worker under project signature ID."""
        with cls._lock:
            project_bucket = cls._storage.get(project_sign)
            if project_bucket:
                return project_bucket.get(model)
            return None

    @classmethod
    def _prune(cls) -> None:
        """
        During registering a new project, verifies whether previous 
        projects are active. Empty project record is pruned.
        """
        for p in list(cls._storage):
            if not len(cls._storage[p]):
                cls._storage.pop(p)

class ProjectModelRegistry(ProjectRegistryBase):
    """
    AIGFLOW Project Registry of fitted Models → Auto populated by unique 
    project ID signature with fitted models during project training.

    Serves as Context Registry / Global State Service (Multiton).
    """
    _storage: dict[str, weakref.WeakValueDictionary] = {}

    @classmethod
    def register(cls, model_instance: object) -> None:
        """Registers model during training under project signature ID."""        
        from .base import ModelBase, ModelUIBase

        model: ModelBase = (
            model_instance if not isinstance(model_instance, ModelUIBase)
            else model_instance.model
        )
        if not isinstance(model, ModelBase):
            raise TypeError(
                'AIG SAFETY ► ProjectModelRegistry accepts only ModelBase objects.'
            )

        cls._register(model)

    @classmethod
    def get_model(cls, project_sign: str, model: str) -> object | None:
        """Provides fitted model under project signature ID."""
        return cls._get_model(project_sign, model)

class ProjectWorkerRegistry(ProjectRegistryBase):
    """
    AIGFLOW Project Registry of instantiated Workers → Auto populated by unique 
    project ID signature  with fitted models during project training.

    Serves as Context Registry / Global State Service (Multiton).
    """
    _storage: dict[str, weakref.WeakValueDictionary] = {}

    @classmethod
    def register(cls, worker_instance: object) -> None:
        """Registers worker during training under project signature ID."""        
        from .base.worker import WorkerBase

        if not isinstance(worker_instance, WorkerBase):
            raise TypeError(
                'AIG SAFETY ► ProjectWorkerRegistry accepts only WorkerBase objects.'
            )
        
        cls._register(worker_instance)
        
    @classmethod
    def get_worker(cls, project_sign: str, model: str) -> object | None:
        """Provides instantiated model's Worker under project signature ID."""
        return cls._get_model(project_sign, model)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
#     
class RegGet:
    """Helper to allow various ways to obtain chosen model from Model/Dash registry."""

    @Dataclass.base
    class Choice:
        """Placeholder of chosen fitter by its name and config."""
        name:   str
        config: dict = None 
    
    @classmethod
    def choice(cls, name:str, config:dict=None) -> Choice:
        """
        Params standard API → name-config → FitRegister/FitOptions.get(**choice) from registry.
        
        Fitter first param is always fx → can be provided as:
         - str (name) → fitter uses default config values
         - dict {'name', 'config'} 
         - Choice instance
        """
        match name:
            case cls.Choice() | dict() as d:
                config = config or d.get('config')
                name   = d['name']
            case str():
                pass
            case _:
                raise ValueError(
                f"Valid choice params can be provided in various ways:\n"
                " - str (name) → fitter uses default config values\n"
                " - dict {'name':str, 'config':dict} \n"
                " - instance of Choice(name, config)"
                )
        config = cls.config(name, config)
        return cls.Choice(name=name, config=config)
    
    @staticmethod
    def config(name:str, config:dict=None) -> dict:
        """
        Config can be provided either as:
        - per fit option (name):
        >>> config = dict(
                name1: dict(verbose=0), 
                name2: ...
            )
            
        - general setting for all options:
        >>> config = dict(verbose=0)
        """
        match config:
            case None:
                return {}
            case dict() as d:
                return d.get(name) or d
            case _:
                return dict(config)
# ─────────────────────────────────────────────────────────────────────────────────────────────────    

class ModelRegistry(DictClass, frozen=False):
    """
    AIGFLOW Models Registry → Auto populated by all used models and their 
    fitter options during the AID package import.
    
    >>> Models Registry Overview via AID UI:
    ctx = PipelineContext()
    reg = ctx.registry

    ► reg.list()
      → provides all available main MODELs linked to their modules
      → each main MODEL may have several fit options (Fitters) given their "name"    

    Get Fitter from Registry:
    >>> reg.get(MODEL, name)
    <or>
    >>> reg.MODEL.name

    Help for Fitter from Registry:
    >>> reg.get(MODEL, name).help()
    <or>
    >>> reg.MODEL.name.help() 
    """
    
    @classmethod
    def _get_model(cls, MODEL:str, name:str, config:dict|None):
        """Helper to obtain chosen model by name under the given MODEL module."""        
        _Verify.model(MODEL)
        fitters = cls[MODEL]
        if name is None:
            name = list(fitters)[0]
        if name not in fitters:
            raise ValueError(f"No fitter '{name}' in {fitters.__name__}. Available: {list(fitters)}")
        name, config = RegGet.choice(name, config).values()
        fitter:FitConfig = fitters[name]
        if not config:
            return fitter()                    
        return fitter(config=config)
        
    @classmethod
    def list(cls) -> dict[str, dict]:
        return {
            k:{name: type(f) for name, f in fitters.items()} for k,fitters 
            in cls.to_dict().items() if k not in ['list','build','get','help']            
        }
        
    @classmethod
    def build(cls, MODEL:str, name:str=None, config:dict=None, **kwargs) -> Any:
        fitter:FitConfig = cls._get_model(MODEL, name, config)
        return fitter.build(**kwargs)
            
    @classmethod
    def get(cls, MODEL:str, name:str=None, config:dict=None) -> Any:
        return cls._get_model(MODEL, name, config)
        
    @classmethod
    def help(cls) -> None:
        print(f"""
    ► {RgbANSI.bold('Models Registry Overview:')}
        CTX = PipelineContext()
        r = CTX.registry

    ► {RgbANSI.bold('r.list()')}
      → provides all available main MODELs linked to their modules
      → each main MODEL may have several fit options (Fitters) given their "name"          

    ► Get Fitter from Registry with default params:
      {RgbANSI.bold('reg_fitter = r.get(MODEL, name)')}
      <or>
      {RgbANSI.bold('reg_fitter = r.MODEL.name')}

    ► Help for Fitter from Registry:
      {RgbANSI.bold('reg_fitter.help()')}

    ► Build Fitter to create internal model for fit/transform:
      (opt: use custom params/config)
      {RgbANSI.bold('fx_instance = reg_fitter(params=params, config=config)')}
      <or>
      {RgbANSI.bold('fx_instance = r.build(MODEL, name, params=params, config=config)')}

    ► View built fitter model in detail:
      {RgbANSI.bold('fx_instance.model')}

    ► Fit built fitter model:
      {RgbANSI.bold('fx_instance.fit(data,...)')}
    """)
# ─────────────────────────────────────────────────────────────────────────────────────────────────    

class DashRegistry(DictClass, frozen=False):
    """
    AIGFLOW Models Dashboard Registry → Auto populated by all used Models 
    during the AID package import.
    """
    ...

    @classmethod
    def list(cls) -> dict[str, dict]:
        return {k:m for k,m in cls.to_dict().items() if k not in ['list','help']}    

class DashRegister:
    """Register and retrieve Dash for given MODEL/module."""                 

    def __init__(self, MODEL:str):
        _Verify.model(MODEL)
        self.MODEL = MODEL

        if MODEL not in DashRegistry:
            DashRegistry[MODEL] = type(MODEL, (DictClass,), {}, frozen=False)   
   
    def register(self, name:str|list):
        """Register Model's Fitter Dash."""                         
        names = List.ensure(name)
        for name in names:
            _Verify.new_name(self.MODEL, name, is_dash=True)
        def decorator(cls: DashInterface):
            for name in names:
                DashRegistry[self.MODEL][name] = cls
            return cls
        return decorator
        
    def get(self, name:str) -> DashInterface:
        """Retrieve Model's Fitter Dash.""" 
        name, _ = RegGet.choice(name).values()
        _Verify.has_name(self.MODEL, name, is_dash=True)
        return DashRegistry[self.MODEL][name]

    def init(self, name:str, fitter_instance:DashMixin, **kwargs) -> None:
        """Instantiate=Build Model's Fitter Dash to Fitter instance → activates: self.dashboard."""
        dash: DashInterface = self.get(name)
        fitter_instance._init_dashboard(dash, **kwargs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class _FitVerify:

    _base_collision_methods = ('_state', '_proxy', 'get_params', 'get_model', 'get_estimator')
    _base_forbidden_methods = ('build', 'fit', 'model', 'estimator', 'params', 'help')
    _base_required_methods  = ('_build',)
    
    _V = RgbANSI.bold('AIG SAFETY ► REGISTRY DESIGN:')
    
    def __init__(self, cls_name:str, method_name:str):
        self.cls_id = f"'{method_name}' in {RgbANSI.bold(cls_name)}"
        self.method = method_name

    def _base_collision_message(self):
        return (self._V +
        f"\n  → Fitter cannot override Base method {self.cls_id}."
        f"\n  → Please avoid any of following internals:\n{self._base_collision_methods}"
        )
    
    def _base_forbidden_message(self):
        return (self._V +
        f"\n  → Fitter cannot override API layer method {self.cls_id}."
        f"\n  → Defining method has to be used instead:\n    def _{self.method}(...):\n     ..."
        )
    
    def _base_required_message(self):
        return (self._V +     
        f"\n  → Fitter must definy {self.cls_id}."
        f"\n  → def _build() is mandatory to definy target fitter worker (ChosenFitter)."
        )     

class FitVerifyApi:
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "FitConfig":
            return
        
        verify = lambda x: _FitVerify(cls.__name__, x)

        for name in _FitVerify._base_collision_methods:
            if name in cls.__dict__:
                raise TypeError(verify(name)._base_collision_message())

        for name in _FitVerify._base_forbidden_methods:
            if name in cls.__dict__:
                raise TypeError(verify(name)._base_forbidden_message())
            
        for name in _FitVerify._base_required_methods:
            if name not in cls.__dict__:
                raise TypeError(verify(name)._base_required_message())
# ─────────────────────────────────────────────────────────────────────────────────────────────────    

class _FitInternals:
    """FitConfig internal space for model state."""
    model:     Any = None
    estimator: Any = None
    params:    Any = None  
    config:    Any = None      

class _FitIntProxy:
    """Descriptor for internal FitConfig namespace."""

    def __get__(self, instance, owner) -> "_FitMethods":
        """
        Access to internal namespace.
        First access creates internal model state on Fitter instance.
        """
        if instance is None:
            return self        
        if '_state' not in instance.__dict__:
            object.__setattr__(instance, '_state', _FitInternals())
        return _FitMethods(instance)    

    def __set__(self, instance, value):
        raise ValueError('AIG SAFETY ► Internal Fitter State is read only.')

class _FitMethods:
    """FitConfig internal methods namespace."""
            
    def __init__(self, fit_instance:"FitConfig"):       # FitConfig
        self.fit_instance = fit_instance                # FitConfig Fitter child
        self.state:_FitInternals = fit_instance._state  # Fitter child internals
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_model(self):        
        if self.state.model is None: 
            raise RuntimeError("Model not built yet. Call `.build()` first.")
        return self.state.model
    
    def get_estimator(self):        
        if self.state.estimator is None:
            raise RuntimeError("Model not built yet. Call `.build()` first.")
        return self.state.estimator
    
    def set_internal(self, attr:Literal["estimator", "model", "params", "config"], int_obj):
        """
        Set mutable internal models to Fitter instance's __dict__ → next to dataclass __slots__ for params.
        This approach allows to keep original DictDataclass functionality, signature and behavior intact.
        In other words, opt back into dynamic attributes, but only for internals as MRO respects it.
        i.e. python-level equivalent of dynamically enabling tp_dictoffset in CPython.
        """
        object.__setattr__(self.state, attr, int_obj)
    
    def is_built(self) -> bool:
        return self.state.params is not None
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_name(self) -> str:
        """Fitter wrapper class name."""
        return self.fit_instance.__class__.__name__
    
    @staticmethod
    def _target_sig_est(fx:type) -> type:
        """Standard link to `fitter's` external estimator class - FitterModelBase."""
        e = getattr(fx, 'get_estimator', None) 
        if not callable(e):
            return fx
        fx_ = e()
        if isinstance(fx_, type):
            return fx_ 
        return fx
    
    @staticmethod
    def _target_sig_prm(fx:type) -> tuple[type, dict]:
        """
        Standard link to `fitter's` w/o external estimator but
        with standard AIG signature → params: dict | DictDataclass.
        """
        i = getattr(fx, '__init__', None)
        if not i:
            return fx, {}
        sig  = {}
        anno = get_type_hints(i)
        for a in anno.values():
            types_ = extract_types(a)
            for t in types_:
                if isinstance(t, type) and issubclass(t, DictDataclass):
                    sig.update( {i:p for i,p in enumerate(Dataclass.get_names(t))} )
        return fx, sig
        
    @classmethod
    def target_sig(cls, fx:type) -> tuple[type, dict]:
        """Obtain Target Fitter (ChosenFitter) params or custom aig fitters."""
        fx_ = cls._target_sig_est(fx)
        if fx_ is fx:
            fx_, sig = cls._target_sig_prm(fx)
            if sig:
                return fx_, sig
        return fx_, {
            i:x for i,x in AigParam.signature(fx_).items() 
            if x not in ['args', 'kwargs']
        }
        
    def get_target(self) -> tuple[str, dict|None]:
        """Target Fitter (ChosenFitter) class name."""
        fit: "FitConfig" = self.fit_instance
        fx = fit._build.__annotations__.get('return') # help can be called before build
        if not fx:
            return '', None
        fx, prm = self.target_sig(fx)
        name = f'► {fx.__name__}'
        return name, prm

    def verify_target(self, method:str) -> None:
        """Target Fitter (ChosenFitter) called method check existance."""
        if not hasattr(self.get_model(), method):
            AigStop.custom(
                f'{_Verify._V} Target Model {self.get_target()[0]} '
                f'does not have {method} method ► Define custom _fit/_transform prior ML action.'
            )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def default_config(self) -> dict:
        """Helper to provide default dynamic parameters."""
        return self.fit_instance.config or {}
    
    def default_params(self) -> dict:
        """Helper to provide default dynamic parameters."""
        return {p:v for p,v in dict(self.fit_instance).items() if p != 'config'}
    
    def get_params_state(self) -> dict:
        """Helper to provide default or built values of parameters."""
        if self.state.params is not None:
            return self.state.params
        return self.default_params()

    def get_config_state(self) -> dict:
        """Helper to provide default or built values of parameters."""
        if self.state.config is not None:
            return self.state.config
        return self.default_config()
    
    def verify_params(self, params:dict) -> None:
        default = self.default_params()
        missing = [k for k in default if k not in params]
        if not missing:
            return
        AigStop.custom(
            f'{_Verify._V} '
            f'Default parameters does not match provided build parameters:\n'
            f'Parameters Design:  {default}\n'
            f'Parameters Missing: {missing}'
        )

    def verify_config(self, config:dict) -> None:
        default = self.default_config()
        missing = [k for k in default if k not in config]
        if not missing:
            return
        AigStop.custom(
            f'{_Verify._V} '
            f'Default config does not match provided build config:\n'
            f'Config Design:  {default}\n'
            f'Config Missing: {missing}'
        )
    
    def set_internal_params(self, params:dict) -> None:
        """
        Helper to set values of parameters from build.
        Fitters firstly inherit default params -> set from build.
        """
        self.verify_params(params)
        default = self.default_params()
        params  = {k: params[k] for k in default}
        self.set_internal('params', params)
    
    def set_internal_config(self, config:dict) -> None:
        """
        Helper to set values of config from build.
        Fitters firstly inherit default config -> set from build.
        """
        self.verify_config(config)
        default = self.default_config()
        config  = {k: config[k] for k in default}
        self.set_internal('config', config)
    
    def set_build_params(self, params:dict) -> None:
        """Set build parameters values."""
        self.set_internal_params(params)
        for p in self.default_params(): 
            object.__setattr__(self.fit_instance, p, params[p])

    def set_build_config(self, config:dict) -> None:
        """Set build config values."""
        self.set_internal_config(config)
        object.__setattr__(self.fit_instance, 'config', config)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def update_model(self, model):
        """
        Ensure returned object is valid and updated version of the target self by fit.
        When custom _fit is defiend we allow to update internal state of the _internal_model 
        as it is expected that _fit deals with nuances in returned obj across libraries.
        """
        if (
            type(self.fit_instance).__dict__.get("_fit", None) is None and 
            ( model is not None ) and 
            ( not isinstance(model, type(self.state.model)) )
        ):
            raise RuntimeError("fit() returned a different object → Please define custom _fit()")
        if model is not None:            
            self.set_internal('model', model)

    def get_custom_or_target(self, method:str):
        """
        Standardized fit/transform call across ML libraries.
        _fit | _update | _transform can be defined to customize target respective methods.

        For instance, we can define custom _fit to introduce one more 
        layer between api and inner model as:

        >>> def _fit(self, *args, **kwargs):
                return self.model.train(*args, **kwargs)

        Where self.model is the standard reference to internally built target instance.
        """
        fx = type(self.fit_instance).__dict__.get(f"_{method}", None)
        if fx is not None:
            return getattr(self.fit_instance, f"_{method}")
        self.verify_target(method)
        return getattr(self.state.model, method)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _repr_html(self):
        t = {
            REPR.T: self.get_name(), 
            REPR.H: "built" if self.is_built() else "not built"
        }
        help = []
        for i in self.get_help(p=False):
            help.extend([
                i, Format.splitter(show=False, style=Format.StyleSplitter(to_html=False))
            ])
        return ReprCreator(t, self.get_params(), help).html()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _norm_val(self, val) -> str:
        if isinstance(val,str):
            return f'"{val}"'
        return ReprUtils.normalize_val(val)

    def _default_info(self, prm_cfg:dict[dict]) -> None:
        state = 'default' if not self.is_built() else 'built'
        info  = []
        for p_c in prm_cfg:
            prm = '\n'.join([f'  "{k}": {self._norm_val(v)}' for k,v in prm_cfg[p_c].items()] + ['}'])

            if p_c == 'params':
                t = RgbANSI.bold(f'► Available dynamic params and their {state} values:\n')
                i = ( '\n► Dynamic params can be provided to the Fitter either as:',
                      '\n  → args/kwargs or',
                      '\n  → params=dict(...)' )
            else:
                t = RgbANSI.bold(f'\n► {state.capitalize()} general config:\n')
                i = (
                    f'\n► Default values of general config can be updated using',
                    f' → Fitter( {RgbANSI.bold("update_config=dict(...)")} )',
                    '\n► Convenient way to keep the default config with a few tweaks via update_config=dict(...)' 
                )
            info.append( t + p_c + ' = {\n' + prm + ''.join(i) )
        return info

    def _specific_info(self, fx_name:str, prm:dict|None) -> str:
        info = []
        join = lambda *i: info.append(''.join(i))
        join(
            RgbANSI.bold('\n► Usage to parametrize target fx:'),
            '\n  • get fitter with defaults: reg_fitter = REGISTRY.get(MODEL, name)'
            '\n  • build with default parametrization: fx = reg_fitter()'
            '\n  • build with custom  parametrization: '
                  'fx = reg_fitter( params=dict(...), config=dict(...), config_update=dict(...) )'
            f'\n  • view built internal backend model in detail: {RgbANSI.bold_blue("fx.model")}'
        )
        join(
            RgbANSI.bold('\n► Fit - Result - Transform of built model:'),
            '\n  • fx.fit(...)'
            '\n  • fx.result'
            '\n  • fx.transform(data, ...)'
        )
        if not prm: 
            return info
        join(
            '\n► Config can set more parameters for the '+ RgbANSI.bold_blue('internal estimator')+':',
            '\n  • Chosen config subset is considered as the most important for the use-case',
            '\n  • Use config or config_update to directly adjust parametrization of the internal estimator',
            '\n\n'+ RgbANSI.bold(f'{fx_name} (internal estimator) full set of available parameters via config | config_update:'),
            '\n'+'\n'.join([f'  {k}: {v}' for k,v in prm.items()])
        )
        return info
    
    def get_help(self, prm_cfg:dict=None, p:bool=True):
        if prm_cfg is None:
            prm_cfg = self.get_params()

        fx_name, prm = self.get_target()
        h1 = self._default_info(prm_cfg)
        h2 = self._specific_info(fx_name, prm)
        hi = h1 + h2
        
        if not p: 
            return hi   

        Format.title(RgbANSI.bold_blue(self.get_name()+' '+fx_name))        
        for i in hi:
            print(i)
            Format.splitter()

    def get_params(self, verbose=False) -> dict|str:
        """Helper to provide params and info about parametrization options."""        
        prm_cfg = {'params': self.get_params_state(), 'config': self.get_config_state()}
        if not verbose:
            return prm_cfg        
        self.get_help()
        return prm_cfg
# ─────────────────────────────────────────────────────────────────────────────────────────────────
     
class FitConfig(DictDataclass, FitVerifyApi):
    """
    **Base Class for any Registered Fitter option.**

    FitConfig provides standard API for end-users with methods/properties as below.
    
    Architecture of Layer 1 - Layer 2 - Layer 3:
    - Layer 1: Public API → Framework-Controlled, Cannot Be Overridden → build / fit / model / estimator / params / help
    - Layer 2: Defining   → Subclass-Controlled, Must Be Implemented   → `_build` / _fit / _transform / _model /...
    - Layer 3: Custom     → Free Extensions, Any Custom Public methods → get_result / get_extra / extract_features /...
    \n─────────────────────────────────────────────────────────────\n   

    - Fitter `Defining Methods at Layer 2` (below public API):
        - _build()     → mandatory: constructs internal model (ChosenFitter)
        - _model       → recommended: when using custom _fit/_transform/etc (IDE support)
        - _fit()       → optional: overrides default fit routing
        - _process()   → optional: overrides default process routing
        - _update()    → optional: overrides default update routing
        - _transform() → optional: overrides default transform routing
    \n─────────────────────────────────────────────────────────────\n   
    - NOTE Fitter must define _build not build \n
    \n─────────────────────────────────────────────────────────────\n   
    
    - `_build()`:
        - the only `mandatory` construction block
        - defines `ChosenFitter` and how parameters are processed and propagated to it
        - the ._build() is used when calling `Fitter API via .build()` :
            - e.g. fx = Fitter(config=config).build(params=params) 
                ► this builds internal ► `self.model = self._build(...) ◄` of targeted ChosenFitter
            - once fx is built, we can operate on it as other defining methods show  

    >>> def _build(self, params:FitParams, config:dict) -> ChosenFitter:
    >>>     ...
    >>>     setup  = dict(params) | config
    >>>     return ChosenFitter(**setup) # builds internal self.model inside Fitter (Fitter holds ChosenFitter state)

    ▼ ──────────── built self.model (ChosenFitter) usage ─────────── ▼

    - `_model`:
        - when custom _fit() / _transform() / _update() / _process() is used `_model` should be defined
        - define _model to obtain IDE support as a simple property link and use in custom _fit/_transform: 
    
    >>> @property
    >>> def _model(self) -> ChosenFitter: 
    >>>   return self.model
    \n─────────────────────────────────────────────────────────────\n   

    - `_fit()`:
        - if defined it `provides full control` of ChosenFitter fit behavior
        - not defining falls back to execute `default` when calling fx.fit(...): self.model.fit(*args, **kwargs)    
        - `returns self`
        - fit/_fit always `returns self of the Fitter` (not ChosenFitter) 
        - thus other custom methods of Fitter can be chained

    >>> def _fit(self, ar1, ..., kw1=1, ...) -> ChosenFitter:
    >>>     return self._model.fit_custom(ar1,...)
    NOTE that we are operating on built self.model its own api..thus any `ChosenFitter method` can be used.\n
    Thus, we can standardize call to various libraries/packages to our API neeeds.
    \n─────────────────────────────────────────────────────────────\n   

    - `_transform()`:
        - if defined it `provides full control` of ChosenFitter transform/predict etc.
        - not defining falls back to execute `default` when calling fx.transform(...): self.model.transform(*args, **kwargs)   
        - returns fx(*args, **kwargs)

    >>> def _transform(self, data:pDF) -> np.ndarray:
    >>>     pred = self._model.predict(data)
    >>>     return np.where(pred == -1, 1, 0) # standard to 0 / 1        
    NOTE that we are operating on built self.model its own api..thus any `ChosenFitter method` can be used.\n
    Thus, we can standardize call output of various libraries/packages to our API neeeds.
    \n─────────────────────────────────────────────────────────────\n   

    - `_update()`:
        - if defined it runs custom logic of _update otherwise tries to call ChosenFitter.update()
        - returns fx(*args,**kwargs)

    - `_process()`:
        - if defined it runs custom logic of _result otherwise tries to call ChosenFitter.result()
        - returns fx(*args,**kwargs)
        
    \n─────────────────────────────────────────────────────────────\n   
    - Any `Custom Methods` can be defined in Fitter and called directly:
        - e.g. `get_result()`, `get_extra()`, ...
        - serves as helpers to obtain any custom output from ChosenFitter fitted state
        - for convenience get_result() is linked to `result` property

    >>> def get_result(self):
    >>>     return {k: self._model.result[k] for k in Dataclass.get_names(FitResult)}

    >>> def get_extra(self, data: pDF) -> np.ndarray:            
    >>>     return scale(self._model.decision_function(data))             

    \n─────────────────────────────────────────────────────────────\n   
    - `Built model's Methods at Layer 1` (public API) to use registred fitter with:
        - help():       provides guidelines about parameters and usage of fitter
        - build():      executes `_build()` defined in fitter and creates internal `self.model` of ChosenFitter
        - fit():        default executes self.model.fit(...)       | or custom defined _fit()
        - process():    default executes self.model.process(...)   | or custom defined _process()
        - update():     default executes self.model.update(...)    | or custom defined _update()
        - transform():  default executes self.model.transform(...) | or custom defined _transform()        
        
    - `Built model's Custom Methods at Layer 1` defined directly in Fitter (goes to Layer 1 directly), e.g.:
        - get_result(): define if custom result of self.model is required → direct link to property `result`
        - get_extra():  define if custom extras (any custom output) from self.model is required
    
    \n─────────────────────────────────────────────────────────────\n   
    - `Built model's standard properties at Layer 1`:
        - model:     provides built internal self.model once .build() is executed → mutable state given fit() etc.
        - estimator: provides built internal self.estimator once .build() is executed → non-mutable state, i.e. orig self.model
        - params:    provides default or built params of Fitter
        - `result`:  default provides built model | or returns custom `get_result() if defined in Fitter`                
    """
    config: dict = {}    
    _state = _FitInternals()
    _proxy = _FitIntProxy()
        
    def __call__(
        self,
        *args,
        params: dict = {}, 
        config: dict = {}, 
        config_update: dict = {},
        **kwargs
    ):
        """
        Build Fit Choice while providing friendly parametrization.

        Dynamic params of the Fit can be provided either as args/kwargs or as one dict params=dict(...).        

        General config of the Fit can keep its default values while be partially updated by cofig_update=dict(...).
        """
        return self.build(
            *args, 
            params = params, 
            config = config, 
            config_update = config_update, 
            **kwargs
        )

    def __getattr__(self, name:str):
        AigStop.jn_getattr(name)
        if not self._proxy.is_built():
            raise AttributeError(name)
        return getattr(self.model, name)
    
    def __repr__(self) -> str:
        return self._proxy.get_help()
    
    def _repr_html_(self) -> str:
        return self._proxy._repr_html()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def help(self) -> None: 
        """Helper to describe how-to."""
        _ = self._proxy.get_help()

    @property
    def params(self) -> dict: 
        """Return params and config state."""
        return self._proxy.get_params()
    
    @property
    def model(self) -> Any: 
        """Expose the built model - mutable instance of estimator for fitting."""        
        return self._proxy.get_model()
    
    @property
    def estimator(self) -> Any: 
        """Expose the built estimator - holds original estimator."""
        return self._proxy.get_estimator()
    
    @property
    def result(self):
        return self.get_result()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_params(self, verbose=False) -> dict:
        """Helper to provide params and info about parametrization options → dict['params', 'config']."""
        return self._proxy.get_params(verbose)

    def get_model(self) -> Any:
        """Helper to provide state of the built model or build and provide model with default params."""
        if self._state.model is None:
            return self.build().model
        return self.model
    
    def get_estimator(self) -> Any:
        """Helper to provide state of the built estimator or build and provide estimator with default params."""
        if self._state.estimator is None:
            return self.build().estimator
        return self.estimator
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def build(
        self, 
        *args,
        params: dict = {}, 
        config: dict = {}, 
        config_update: dict = {},
        **kwargs
    ):
        """
        Build Fit Choice while providing friendly parametrization.

        Dynamic params of the Fit can be provided either as args/kwargs or as one dict params=dict(...).        

        General config of the Fit can keep its default values while be partially updated by cofig_update=dict(...).
        """        
        p_def   = self._proxy.default_params() 
        params_ = AigParam.param_ar_kw(p_def, params, args, kwargs, self._proxy.get_name())   
        self._proxy.set_build_params(params_)

        config_ = config or self.config
        config_ = dict(config_) | config_update
        self._proxy.set_build_config(config_)

        self._proxy.set_internal('estimator', self._build(params_, config_))    # holds original estimator
        self._proxy.set_internal('model',     self._build(params_, config_))    # mutable instance for fitting
        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _build(self, params:DictDataclass, config:dict):
        raise NotImplementedError("Fitter must implement _build()")

    @property
    def _model(self):
        raise NotImplementedError("Fitter should implement _model() when custom _fit/_transform is used") 
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, *args, **kwargs) -> Any:
        """Standardized fit (self update) call across ML libraries."""
        fx = self._proxy.get_custom_or_target('fit')
        model = fx(*args, **kwargs)
        self._proxy.update_model(model)
        return self
    
    def process(self, *args, **kwargs) -> Any:
        """Standardized process (fit w/o mutable self) call across ML libraries."""
        fx = self._proxy.get_custom_or_target('process')
        return fx(*args, **kwargs)    

    def update(self, *args, **kwargs) -> Any:
        """Standardized fit/process update call across ML libraries."""
        fx = self._proxy.get_custom_or_target('update')
        return fx(*args, **kwargs)    
        
    def transform(self, *args, **kwargs) -> Any:
        """Standardized transform/predict call across ML libraries."""
        fx = self._proxy.get_custom_or_target('transform')
        return fx(*args, **kwargs)    
    
    def predict_proba(self, *args, **kwargs) -> Any:
        """Standardized predict_proba call across ML libraries."""
        fx = self._proxy.get_custom_or_target('predict_proba')
        return fx(*args, **kwargs)  

    def get_result(self) -> Any:
        """Obtain Result during fit pipeline."""
        return self.model.result

    def get_stats(self) -> Any:
        """Obtain Result stats during fit pipeline."""
        return self.model.stats
        
    def get_extra(self) -> Any:
        """Extras needed during fit pipeline."""
        return self.model
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class FitRegister:
    """Register and retrieve a fit option for a given MODEL/module."""                 

    def __init__(self, MODEL:str):
        _Verify.model(MODEL)
        self.MODEL = MODEL

        if MODEL not in ModelRegistry:
            ModelRegistry[MODEL] = type(MODEL, (DictClass,), {}, frozen=False)   

    def register(self, name: str):
        """
        Register a Fitter option under chosen name of the main MODEL (flow-step) to REGISTRY.

        Base class `FitConfig` provides standard API for end-users with methods/properties as below.
            
        Architecture of Layer 1 - Layer 2 - Layer 3:
        - Layer 1: Public API → Framework-Controlled, Cannot Be Overridden → build / fit / model / estimator / params / help
        - Layer 2: Defining   → Subclass-Controlled, Must Be Implemented   → `_build` / _fit / _transform / _model /...
        - Layer 3: Custom     → Free Extensions, Any Custom Public methods → get_result / get_extra / extract_features /...
        \n─────────────────────────────────────────────────────────────\n   

        - Fitter `Defining Methods at Layer 2` (below public API):
            - _build()     → mandatory: constructs internal model (ChosenFitter)
            - _model       → recommended: when using custom _fit/_transform/etc (IDE support)
            - _fit()       → optional: overrides default fit routing
            - _process()   → optional: overrides default process routing
            - _update()    → optional: overrides default update routing
            - _transform() → optional: overrides default transform routing
        \n─────────────────────────────────────────────────────────────\n   
        - NOTE Fitter must define _build not build \n
        \n─────────────────────────────────────────────────────────────\n   
        
        - `_build()`:
            - the only `mandatory` construction block
            - defines `ChosenFitter` and how parameters are processed and propagated to it
            - the ._build() is used when calling `Fitter API via .build()` :
                - e.g. fx = Fitter(config=config).build(params=params) 
                    ► this builds internal ► `self.model = self._build(...) ◄` of targeted ChosenFitter
                - once fx is built, we can operate on it as other defining methods show  

        >>> def _build(self, params:FitParams, config:dict) -> ChosenFitter:
        >>>     ...
        >>>     setup  = dict(params) | config
        >>>     return ChosenFitter(**setup) # builds internal self.model inside Fitter (Fitter holds ChosenFitter state)

        ▼ ──────────── built self.model (ChosenFitter) usage ─────────── ▼

        - `_model`:
            - when custom _fit() / _transform() / _update() / _process() is used `_model` should be defined
            - define _model to obtain IDE support as a simple property link and use in custom _fit/_transform: 
        
        >>> @property
        >>> def _model(self) -> ChosenFitter: 
        >>>   return self.model
        \n─────────────────────────────────────────────────────────────\n   

        - `_fit()`:
            - if defined it `provides full control` of ChosenFitter fit behavior
            - not defining falls back to execute `default` when calling fx.fit(...): self.model.fit(*args, **kwargs)    
            - `returns self`
            - fit/_fit always `returns self of the Fitter` (not ChosenFitter) 
            - thus other custom methods of Fitter can be chained

        >>> def _fit(self, ar1, ..., kw1=1, ...) -> ChosenFitter:
        >>>     return self._model.fit_custom(ar1,...)
        NOTE that we are operating on built self.model its own api..thus any `ChosenFitter method` can be used.\n
        Thus, we can standardize call to various libraries/packages to our API neeeds.
        \n─────────────────────────────────────────────────────────────\n   

        - `_transform()`:
            - if defined it `provides full control` of ChosenFitter transform/predict etc.
            - not defining falls back to execute `default` when calling fx.transform(...): self.model.transform(*args, **kwargs)   
            - returns fx(*args, **kwargs)

        >>> def _transform(self, data:pDF) -> np.ndarray:
        >>>     pred = self._model.predict(data)
        >>>     return np.where(pred == -1, 1, 0) # standard to 0 / 1        
        NOTE that we are operating on built self.model its own api..thus any `ChosenFitter method` can be used.\n
        Thus, we can standardize call output of various libraries/packages to our API neeeds.
        \n─────────────────────────────────────────────────────────────\n   

        - `_update()`:
            - if defined it runs custom logic of _update otherwise tries to call ChosenFitter.update()
            - returns fx(*args,**kwargs)

        - `_process()`:
            - if defined it runs custom logic of _result otherwise tries to call ChosenFitter.result()
            - returns fx(*args,**kwargs)
            
        \n─────────────────────────────────────────────────────────────\n   
        - Any `Custom Methods` can be defined in Fitter and called directly:
            - e.g. `get_result()`, `get_extra()`, ...
            - serves as helpers to obtain any custom output from ChosenFitter fitted state
            - for convenience get_result() is linked to `result` property

        >>> def get_result(self):
        >>>     return {k: self._model.result[k] for k in Dataclass.get_names(FitResult)}

        >>> def get_extra(self, data: pDF) -> np.ndarray:            
        >>>     return scale(self._model.decision_function(data))             

        \n─────────────────────────────────────────────────────────────\n   
        - `Built model's Methods at Layer 1` (public API) to use registred fitter with:
            - help():       provides guidelines about parameters and usage of fitter
            - build():      executes `_build()` defined in fitter and creates internal `self.model` of ChosenFitter
            - fit():        default executes self.model.fit(...)       | or custom defined _fit()
            - process():    default executes self.model.process(...)   | or custom defined _process()
            - update():     default executes self.model.update(...)    | or custom defined _update()
            - transform():  default executes self.model.transform(...) | or custom defined _transform()        
            
        - `Built model's Custom Methods at Layer 1` defined directly in Fitter (goes to Layer 1 directly), e.g.:
            - get_result(): define if custom result of self.model is required → direct link to property `result`
            - get_extra():  define if custom extras (any custom output) from self.model is required
        
        \n─────────────────────────────────────────────────────────────\n   
        - `Built model's standard properties at Layer 1`:
            - model:     provides built internal self.model once .build() is executed → mutable state given fit() etc.
            - estimator: provides built internal self.estimator once .build() is executed → non-mutable state, i.e. orig self.model
            - params:    provides default or built params of Fitter
            - `result`:  default provides built model | or returns custom `get_result() if defined in Fitter`  
        """                         
        _Verify.new_name(self.MODEL, name)
        def decorator(cls):
            cls = Dataclass.base(cls, base=FitConfig, top_base=False)
            ModelRegistry[self.MODEL][name] = cls()
            return cls
        return decorator

    def get(self, name: str, config:dict=None) -> FitConfig:
        """Retrieve a fitter instance by chosen name of the main MODEL (flow-step)."""
        name, config = RegGet.choice(name, config).values()
        _Verify.has_name(self.MODEL, name)
        fx:FitConfig = type(ModelRegistry[self.MODEL][name])
        if not config:
            return fx() # default config design of the chosen fitter - from the register point
        return fx(config=config) # custom config from the API-user
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class _FitOptRepr(type):
    def __repr__(cls):
        return cls.help(show=False)
    
class FitRegOpt(metaclass=_FitOptRepr):
    """
    Helper of the MODEL Module → Instantiate and build a fitter model.

    FitOptions.build() is present for consistency and convenience.
    FitOptions operates on nameFitter inside given MODEL module.
    We can obtain chosen fitter by various ways:

        >>> FitOptions.build(nameFitter, params=params, config=config)
        >>> FitOptions.get(nameFitter).build(params=params, config=config)
        >>> FitOptions.get(nameFitter, config).build(params=params)

        >>> FitRegistry(MODEL).get(name).build(params=params, config=config)
        >>> FitRegistry(MODEL).get(name, config).build(params=params)

        >>> REGISTRY.build(MODEL, name, params=params, config=config)
        >>> REGISTRY.get(MODEL, name, config).build(params=params)

        >>> REGISTRY.MODEL.name(config=config).build(params=params)
        >>> REGISTRY.MODEL.name().build(params=params, config=config)
        >>> REGISTRY.MODEL.name(**params, config=config).build()
    """

    @classmethod
    def _fitters(cls) -> dict[str, type]:
        """Return a dict of all fitter classes defined inside this registry."""
        return {
            name: obj for name, obj in vars(cls).items()
            if ( AigParam.issubclass(obj, FitConfig) or isinstance(obj, FitConfig) )
        }

    @classmethod
    def _get_model(cls, name:str, config:dict) -> FitConfig:
        """Helper to obtain chosen model by name under the given MODEL module."""
        fitters = cls._fitters()
        if name is None:
            name = list(fitters)[0]
        if name not in fitters:
            raise ValueError(f"No fitter '{name}' in {cls.__name__}. Available: {list(fitters)}")
        name, config = RegGet.choice(name, config).values()        
        fitter:FitConfig = fitters[name]
        if not config:
            return fitter()                    
        return fitter(config=config)
        
    @classmethod
    def list(cls, short: bool = True) -> list[str]:
        """List names of all fitters available in this registry."""
        fitters = cls._fitters()
        if short:
            return list(fitters.keys())
        return [
            f"{name}: {getattr(obj, '__doc__', '').strip() or '(no doc)'}"
            for name, obj in fitters.items()
        ]
    
    @classmethod
    def help(cls, name: str = None, show:bool=True):
        """Show help for all or one specific fitter."""
        h = []
        if name is None:
            h.append(RgbANSI.bold(
                f"{cls.__name__} Registry \nChoose model by name and obtain it via "
                ".get(name, config:Optional) <or> .build(name, config:Optional, **kwargs):\n"
            ))
            for n, c in cls._fitters().items():
                h.append(f"\n• {n}\n{getattr(c, '__doc__', '').strip() or '(no doc)'}")
        else:
            fcls = cls._fitters().get(name)
            if not fcls:
                raise ValueError(f"No fitter named '{name}' found in {cls.__name__}")
            h.append(f"Help for {name}:")
            h.append(fcls.__doc__ or "(no docstring)")
            h.append("\nDefault config:\n", getattr(fcls, 'config', None))    

        h = ''.join(h)
        if not show:
            return h
        print(h)

    @classmethod
    def build(cls, nameFitter:str=None, config:dict=None, **kwargs) -> Any:
        """
        Helper of the MODEL Module → Instantiate and build a fitter model.

        FitOptions.build() is present for consistency and convenience.
        FitOptions operates on nameFitter inside given MODEL module.
        We can obtain chosen fitter by various ways:

            >>> FitOptions.build(nameFitter, params=params, config=config)
            >>> FitOptions.get(nameFitter).build(params=params, config=config)
            >>> FitOptions.get(nameFitter, config).build(params=params)

            >>> FitRegistry(MODEL).get(name).build(params=params, config=config)
            >>> FitRegistry(MODEL).get(name, config).build(params=params)
            
            >>> REGISTRY.build(MODEL, name, params=params, config=config)
            >>> REGISTRY.get(MODEL, name, config).build(params=params)

            >>> REGISTRY.MODEL.name(config=config).build(params=params)
            >>> REGISTRY.MODEL.name().build(params=params, config=config)
            >>> REGISTRY.MODEL.name(**params, config=config).build()
        """
        fitter = cls._get_model(nameFitter, config)
        return fitter.build(**kwargs)
    
    @classmethod
    def get(cls, nameFitter:str=None, config:dict=None) -> FitConfig:
        """Helper to obtain chosen model by name under the given MODEL module."""
        return cls._get_model(nameFitter, config)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class _Verify:

    _V = 'FIT REGISTRY ►'
    _D = 'DASH REGISTRY ►'
    
    @classmethod
    def _get_reg(cls, is_dash=False) -> tuple[DashRegistry|ModelRegistry, str]:    
        if is_dash: 
            return DashRegistry,  cls._D
        else: return ModelRegistry, cls._V
    
    @classmethod
    def model(cls, MODEL:str) -> None:
        if MODEL in AIGMODELS:
            return
        AigStop.custom(
            f"{cls._V} To register MODEL's fitting options, the MODEL has to be defined in AIGFLOW general package structure prior → ./core/structure\n\n"+
            f"Defined Main Models (Flow-Steps):\n► { '\n► '.join(AIGMODELS) }"
        )

    @classmethod
    def new_name(cls, MODEL:str, name:str, is_dash=False) -> None:
        REG, _v = cls._get_reg(is_dash)
        if name not in REG[MODEL]:
            return
        AigStop.custom(
            f"{_v} {MODEL} ► '{name}' is already registred fit function, please define new name for new option under the same MODEL."
        )

    @classmethod
    def has_name(cls, MODEL:str, name:str, is_dash=False) -> None:
        REG, _v = cls._get_reg(is_dash)    
        if name in REG[MODEL]:
            return
        AigStop.custom(
            f"{_v} {MODEL} ► '{name}' has not been registred as fit function, please define new fit option prior calling it."+
            f"\nAvailable fit options under {MODEL}:\n{ModelRegistry[MODEL].keys()}"
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬