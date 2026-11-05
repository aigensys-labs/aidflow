"""
Foundation Utilities and Low-Level System Primatives.

This module provides the essential operational tools and reflection utilities 
that underpin all other layers of the AIDflow framework. It implements a suite 
of high-order functions designed to handle dynamic parameter resolution, 
object provenance, and runtime execution control, ensuring that the framework's 
high-level API remains clean while maintaining strict technical rigor.
"""

from ..deps.common import  (
    inspect, 
    logging, 
    datetime, 
    copy, 
    Path, 
    wraps
)
from ..deps.struct import  (
    is_dataclass, 
    asdict,
)
from ..deps.types import  (
    Any, 
    Literal,
    TypeVar,
    FuncType,
)
from ..display.rgb import (
    RgbANSI
)
from ..display.verbosity import (
    Verbosity
)
from .factory import  (
    DictDataclass, 
    Dataclass
)

T = TypeVar("T", bound=type)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# AIG → PARAMETERIZATION
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class AigParam:
    """
    Parameter resolution and hierarchy utilities.

    Provides the core mechanism for resolving user-supplied
    parameters against a target class signature. Handles:

    - **Extraction**: Pull relevant parameter values from an
      instance or dict into a target class's signature.
    - **Hierarchy**: Resolve value priority — direct (explicit
      user input) > indirect (flow standard) > default.
    - **Standardization**: Accept arbitrary args/kwargs and
      enforce them against a signature definition, catching
      typos and unknown parameters.
    - **Runtime Merging**: Decorator that merges instance-level
      defaults with call-time overrides.
    """

    @staticmethod
    def get_definition(param_cls: type | FuncType) -> list:
        """
        Returns the ordered list of parameter names from a class
        or function signature.

        Parameters
        ----------
        param_cls : type | callable
            Class or function to inspect.

        Returns
        -------
        list
            Parameter names in definition order.
        """
        return list( inspect.signature(param_cls).parameters )
           
    @classmethod
    def get_values(cls, 
        for_target_cls: type, 
        from_param_instance: object, 
        **kwargs
    ) -> dict:
        """
        Extract parameter values from a source instance that match
        the target class's signature.

        Parameters
        ----------
        for_target_cls : type
            The class whose signature defines which parameters
            to extract.
        from_param_instance : object | dict | DictDataclass
            Source of values (instance, dict, or dataclass).
        **kwargs
            Additional values to merge into the extraction pool.

        Returns
        -------
        dict
            {param_name: value} for parameters present in both
            the source and the target signature.
        """
        prm = cls.get_definition(for_target_cls)
        vars_from_instance:dict = (
            vars(from_param_instance) 
            if not isinstance(from_param_instance, (dict, DictDataclass)) 
            else dict(from_param_instance)
        )
        all_vars = vars_from_instance | kwargs
        return {k: v for k,v in all_vars.items() if k in prm}    

    @classmethod
    def make_instance(cls, 
        target_cls_to_make, 
        use_params_from_instance, 
        **kwargs
    ) -> object:
        """
        Instantiate a target class by extracting matching parameter
        values from a source instance.

        Parameters
        ----------
        target_cls_to_make : type
            The class to instantiate.
        use_params_from_instance : object | dict
            Source of parameter values.
        **kwargs
            Override values (highest priority).

        Returns
        -------
        instance
            A new instance of `target_cls_to_make`.
        """ 
        vals = cls.get_values(target_cls_to_make, use_params_from_instance, **kwargs)
        return target_cls_to_make(**vals)

    @classmethod
    def user_setup(cls, parameters_instance: type) -> dict:        
        """
        Extract user-facing flow parameters from a Parameters instance.

        Returns
        -------
        dict
            User-specified parameter values (excludes internal/
            private attributes).
        """
        from .parameters import Parameters
        return cls.get_values(Parameters, parameters_instance)

    @classmethod
    def user_complete_setup(cls, 
        parameters_instance: type, 
        convert_path: bool=True
    ) -> dict:
        """
        Extract the full parameter set (including private) from a
        Parameters instance.

        Parameters
        ----------
        parameters_instance : object
            The Parameters instance to extract from.
        convert_path : bool, default=True
            If True, convert all `Path` values to strings
            (for serialization compatibility).

        Returns
        -------
        dict
            Complete parameter mapping.
        """
        setup = vars(parameters_instance).items()
        setup = {k: v for k,v in setup if not k.startswith('_')}    # _ excludes private attributes         
        if convert_path: 
            return cls._setup_path_to_string(setup)
        return setup

    @classmethod
    def _setup_path_to_string(cls, setup: dict | Path | Any) -> dict:
        """
        Recursively convert all `Path` objects in a nested dict
        to their string representations.

        Parameters
        ----------
        setup : dict | Path | Any
            The value (or nested dict) to convert.

        Returns
        -------
        dict | str | Any
            The value with all Path instances converted to str.
        """
        if isinstance(setup, dict):   
            to_str = {}                     
            for k, v in setup.items():
                to_str[k] = cls._setup_path_to_string(v)
            return to_str
        else: 
            return setup if not isinstance(setup, Path) else str(setup)   
        
    @staticmethod
    def get_ar_kw_values_by_sig(fx, args: tuple, kwargs: dict) -> dict:
        """
        Bind positional and keyword arguments to a function's
        signature, filling in defaults for absent parameters.

        Parameters
        ----------
        fx : callable
            The function whose signature defines the binding.
        args : tuple
            Positional arguments to bind.
        kwargs : dict
            Keyword arguments to bind.

        Returns
        -------
        dict
            {param_name: value} with defaults applied for
            parameters not explicitly provided.
        """
        sig = inspect.signature(fx)
        kw  = {k:v for k,v in kwargs.items() if k in sig.parameters}
        bound = sig.bind_partial(*args, **kw)

        for name, param in sig.parameters.items():
            if name not in bound.arguments:
                if param.default is not inspect._empty:
                    bound.arguments[name] = param.default

        return dict(bound.arguments) 
    
    @staticmethod
    def di(p_direct: Any, p_indirect: Any) -> Any:
        """
        Resolve parameter value by hierarchy: direct (explicit)
        takes priority over indirect (flow standard/default).

        Parameters
        ----------
        p_direct : Any
            Explicit user-supplied value (highest priority).
        p_indirect : Any
            Flow standard or default value (fallback).

        Returns
        -------
        Any
            `p_direct` if not None, else `p_indirect`.
        """
        if p_direct is not None:  
            return p_direct
        return p_indirect        
    
    @staticmethod
    def di_lambda(p_direct: Any, p_indirect: Any) -> FuncType:
        """
        Resolve a deferred (callable) parameter by hierarchy.
        Ensures the result is always a callable — wrapping static
        values in a zero-arg lambda if needed.

        Parameters
        ----------
        p_direct : Any | FuncType
            Explicit value or callable (highest priority).
        p_indirect : Any | FuncType
            Fallback value or callable.

        Returns
        -------
        FuncType
            A callable that returns the resolved value.
        """
        if ( p_direct is not None ):  
            if not isinstance(p_direct, FuncType):
                return lambda: p_direct
            return p_direct
        if not isinstance(p_indirect, FuncType):
            return lambda: p_indirect
        return p_indirect
    
    @classmethod
    def ctx(cls, 
        p_direct: Any = None, 
        ctx: dict = None, 
        p_context: str = None
    ) -> Any:
        """
        Resolve a parameter from the flow context (`ctx` dict),
        with direct value taking priority.

        Parameters
        ----------
        p_direct : Any
            Explicit value (highest priority).
        ctx : dict | None
            Flow context dictionary to look up from.
        p_context : str | None
            Key in `ctx` to retrieve.

        Returns
        -------
        Any
            `p_direct` if not None, else `ctx[p_context]`.
        """  
        p_c = None
        if ( ctx is not None ) and ( ctx.get(p_context) is not None ):
            p_c = ctx[p_c]
        return cls.di(p_direct, p_c)  
    
    @classmethod
    def signature(cls, 
        fx: FuncType, 
        to_param: int = 100, 
        from_param: int = 0
    ) -> dict:   
        """
        Obtain a subset of a function's parameter names by index
        range.

        Parameters
        ----------
        fx : callable
            The function to inspect.
        from_param : int, default=0
            Start index (inclusive).
        to_param : int, default=100
            End index (exclusive).

        Returns
        -------
        dict
            {index: param_name} for the requested range.
        """
        return {
            i:x for i,x in enumerate(cls.get_definition(fx)[from_param:to_param])
        }
        
    @staticmethod
    def issubclass(cls: type|Any, parent_to_verify:type) -> bool:
        """
        Safe issubclass check — returns False if the first argument
        is not a class (preventing TypeError from non-type inputs).

        Parameters
        ----------
        cls : type | Any
            The class to check.
        parent_to_verify : type
            The parent class to verify against.

        Returns
        -------
        bool
            True if `cls` is a class and is a subclass of
            `parent_to_verify`.
        """
        return inspect.isclass(cls) and issubclass(cls, parent_to_verify)

    @classmethod
    def _ar_kw_sig(cls, params_def: dict | DictDataclass) -> dict | DictDataclass:
        """
        Normalize a parameter signature definition to a consistent
        form (dict of default values or DictDataclass instance).

        Accepts:
        - `dict`
        - `DictDataclass` instance
        - `DictDataclass` class (extracts default values)

        Parameters
        ----------
        params_def : dict | DictDataclass
            The signature definition.

        Returns
        -------
        dict | DictDataclass
            Normalized signature with default values.
        """
        if cls.issubclass(params_def, DictDataclass):            
            params_def = Dataclass.get_default_values(params_def)
        elif not isinstance(params_def, (dict, DictDataclass)):
            AigStop.custom(
                'AIG SAFETY ► Only instance of dict | DictDataclass or DictDataclass as cls is allowed.'
            )
        return params_def
    
    @classmethod
    def _ar_kw(cls, 
        params_def: dict | DictDataclass, 
        args: tuple, 
        kwargs: dict
    ) -> dict:
        """
        Extract user-supplied parameter values from arbitrary
        args/kwargs, matched against the signature definition.

        Positional args are matched by index; keyword args by
        name. Only parameters present in the signature are
        extracted.

        Parameters
        ----------
        params_def : dict | DictDataclass
            Signature definition (parameter names + defaults).
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        dict
            {param_name: value} for user-supplied values.
        """
        if not (args or kwargs):
            return {}
        params_def = cls._ar_kw_sig(params_def)

        ak = lambda i,p: args[i] if args[i:] else kwargs.get(p)
        return {
            p: ak(i,p) for i,p in enumerate(params_def) 
            if ak(i,p) is not None
        }
    
    @staticmethod
    def _ar_kw_param_standardize(
        p_def:  dict, 
        params: dict, 
        ar_kw:  dict, 
        allow_extra_params: bool = True, 
        warn_dupl: bool = False
    ) -> dict:
        """
        Validate user-supplied parameters against the signature
        definition.

        - Raises if unknown parameters are provided and
          `allow_extra_params=False`.
        - Warns if the same parameter is provided in both
          `params` and `ar_kw` (ar_kw wins).
        - Filters out unknown parameters if allowed.

        Parameters
        ----------
        p_def : dict
            Signature definition (valid parameter names).
        params : dict
            User-supplied named parameters.
        ar_kw : dict
            User-supplied args/kwargs (highest priority).
        allow_extra_params : bool, default=True
            If False, raise on unknown parameters.
        warn_dupl : bool, default=False
            If True, print a warning when a parameter appears in
            both `params` and `ar_kw`.

        Returns
        -------
        dict
            Validated parameter dict (unknown params removed).
        """
        params = dict(params)
        is_not_sig = [p for p in ar_kw | params if p not in p_def]
        if is_not_sig and not allow_extra_params:
            AigStop.custom(
            f'AIG SAFETY ► Obtained unkown parameters:\n'+
            f'{is_not_sig}\n'+
            f'Accepted custom parameters are: {list(p_def)}'
            )
        if warn_dupl and (
            [p for p in params if p in ar_kw ] or
            [p for p in ar_kw  if p in params]
        ):
            print('WARNING: Provided same parameter via args/kwargs and params → using dynamic ar_kw values')
            print('arg/kw ►', ar_kw)
            print('params ►',params)
        return {p:v for p,v in params.items() if p not in is_not_sig}
    
    @classmethod
    def param_ar_kw(cls, 
        params_def:  T | DictDataclass | dict, 
        params:      DictDataclass | dict | None, 
        args:        tuple, 
        kwargs:      dict,
        new_name:    str = 'ParamClass',
        allow_extra: bool = True,
        warn_dupl:   bool = False
    ) -> T | DictDataclass:
        """
        Resolve the final parameter set by merging user input
        against a signature definition.

        Hierarchy (highest → lowest priority):
        1. `ar_kw` — positional/keyword args (runtime overrides).
        2. `params` — named parameter dict (e.g., from config).
        3. `p_def` — defaults from the signature.

        Parameters
        ----------
        params_def : type | DictDataclass | dict
            Signature definition (class, instance, or dict of
            defaults).
        params : dict | DictDataclass | None
            Named parameter values (medium priority).
        args : tuple
            Positional arguments (highest priority).
        kwargs : dict
            Keyword arguments (highest priority).
        new_name : str, default='ParamClass'
            Name for dynamically created parameter classes.
        allow_extra : bool, default=True
            If False, raise on unknown parameters.
        warn_dupl : bool, default=False
            If True, warn on parameter conflicts.

        Returns
        -------
        DictDataclass
            The resolved parameter instance with all values
            merged by hierarchy.
        """
        ar_kw  = cls._ar_kw(params_def, args, kwargs)
        p_def  = cls._ar_kw_sig(params_def)            
        params = cls._ar_kw_param_standardize(p_def, (params or {}), ar_kw, allow_extra, warn_dupl)
        params = p_def | params | ar_kw
        if cls.issubclass(params_def, DictDataclass):
            return params_def(**params)
        return Dataclass.create_params(params, new_name) 

    @staticmethod
    def verbose_std(
        kwargs: dict | None, 
        VERBOSE: int | bool, 
        verbose: str = 'verbose'
    ) -> tuple[Verbosity,bool]:
        """
        Standardize verbosity from kwargs or a class-level default.

        Parameters
        ----------
        kwargs : dict | None
            Call-time keyword arguments.
        VERBOSE : int | bool
            Class-level verbosity default.
        verbose : str, default='verbose'
            Key name to look up in kwargs.

        Returns
        -------
        tuple[Verbosity, bool]
            (normalized Verbosity instance, is_GUI flag)
        """
        k = kwargs or {}
        k = dict(k) # duplicate
        v = AigParam.di(k.get(verbose), VERBOSE)
        v = Verbosity.normalize(v)
        return v, (v == Verbosity.GUI)

    @staticmethod
    def with_setattr_runtime(*runtime_def_none, **runtime_w_defaults) -> FuncType:
        """
        Decorator that resolves runtime parameters by merging
        instance-level defaults with call-time overrides, then
        sets them as instance attributes and passes to the
        wrapped function.

        Parameters
        ----------
        *runtime_def_none : str
            Parameter names that require no default (resolved
            from instance or kwargs).
        **runtime_w_defaults : Any
            Parameter names with fallback defaults.

        Returns
        -------
        FuncType
            Decorated method.

        Notes
        -----
        Priority: `kwargs` (call-time) > instance attribute >
        declared default.

        Example
        -------
        >>> @AigParam.with_runtime_params(verbose=True, dry_run=True)
        ... def build(self, ..., **kwargs): ...
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                
                def priority(k, default=None):
                    if hasattr(self, k):
                        runtime[k] = AigParam.di( kwargs.pop(k, None), getattr(self, k) )
                    else:
                        runtime[k] = kwargs.pop(k, default)

                runtime = {}
                for k in runtime_def_none:
                    priority(k)

                for k, default in runtime_w_defaults.items():
                    priority(k, default)
                
                for k, v in runtime.items():
                    setattr(self, k, v)
                
                return func(self, *args, **kwargs, **runtime)
            return wrapper
        return decorator 
    
    @staticmethod
    def verify_data_arg(default_value:Any, ith=0, is_method=True) -> FuncType:
        """
        Decorator that short-circuits a method if its i-th
        argument is None or empty, returning a default value
        instead.

        Parameters
        ----------
        default_value : Any
            Value to return if the data argument is None/empty.
        ith : int, default=0
            Zero-based index of the data argument
            (0 = first arg after self for methods).
        is_method : bool, default=True
            If True, accounts for `self` in the index.

        Returns
        -------
        FuncType
            Decorated method.
        """
        def decor(fx):
            i = ith+1 if is_method else ith
            @wraps(fx)
            def wrapper(*args, **kwargs):
                if args[i] is None or not len(args[i]): 
                    return default_value
                return fx(*args, **kwargs)
            return wrapper
        return decor
        
    @staticmethod
    def verify_ui_params(param_cls:DictDataclass, gui_desc:dict) -> None:
        """
        Validate that all keys in a UI description dict correspond
        to actual parameters in the parameter class.

        Raises
        ------
        AigStop
            If any key in `gui_desc` is not found in
            `param_cls`.

        Parameters
        ----------
        param_cls : DictDataclass
            The parameter class to validate against.
        gui_desc : dict
            GUI description mapping (keys must be valid
            parameter names).
        """
        prm = param_cls.keys()
        for k in gui_desc:
            if k in prm:
                continue
            AigStop.custom(
                'AIG SAFETY ► WRONG OptionGUI setting ► '
                f'parameter class {type(param_cls).__name__} has no {k} attribute. '
                f'Provided gui description: \n{gui_desc}'
            )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# AIG → MODELS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class AigModel:
    """
    Model lifecycle utilities — copy, sanitize, and reconstruct.

    Provides the infrastructure for safely serializing and
    restoring fitted model instances. Handles nested object
    graphs, custom copy protocols (`__aig_copy__`), and
    attribute removal/reconstruction for IO operations.
    """

    @staticmethod
    def structure(AIGFLOW:dict, REG:list, exclude:list=None) -> list:
        """
        Extract model identifiers from the flow structure dict by
        registry keys.

        Parameters
        ----------
        AIGFLOW : dict
            The full flow structure (nested dict of model names).
        REG : list
            Registry keys to extract (e.g., `['agent1', 'agent2']`).
        exclude : list | None, default=None
            Model names to exclude from the result.

        Returns
        -------
        list
            Flat list of model identifiers found under the
            specified registry keys.
        """
        AIGMODELS = []
        def _scan_structure(aigflow):    
            if isinstance(aigflow, str):
                AIGMODELS.append(aigflow)
                return
            for k in aigflow:
                _scan_structure(aigflow[k])
        _scan_structure({ml: AIGFLOW[ml] for ml in REG})    
        return [m for m in AIGMODELS if m not in (exclude or [])]
    # ─────────────────────────────────────────────────────────────────────────────────────────────
       
    @staticmethod
    def safe_copy(instance:object) -> object:
        """
        Perform a deep copy of a fitted model instance that
        respects custom copy protocols.

        Traverses the object graph recursively:
        - Objects with `__aig_copy__` use their custom copy.
        - Standard objects are shallow-copied at the top level
          and deep-copied for nested attributes.
        - Types (classes), `None`, and dataclasses are
          passed through by reference (not copied).

        Parameters
        ----------
        instance : object
            The fitted model instance to copy.

        Returns
        -------
        object
            A new instance with the same state, safely
            independent of the original.
        """
        skip_types = (type, DictDataclass)
        skip = lambda x: (
            ( x is None ) or isinstance(x, skip_types) or not hasattr(x, '__dict__')
        )        
        mem = {}
        def _copy(original:object):                        
            
            if skip(original):
                return original
                        
            orig_id = id(original)
            if orig_id in mem:
                return mem[orig_id]
            
            clone = copy.copy(original)
            mem[orig_id] = clone

            for attr, orig_val in original.__dict__.items():                
                if skip(orig_val):
                    continue
                
                if hasattr(orig_val, '__aig_copy__'):
                    # custom copy
                    fitter = orig_val.__aig_copy__()                    
                    clone_val = _copy(fitter)
                    
                else:
                    # continue to nested objects
                    clone_val = _copy(orig_val)                    
                
                object.__setattr__(clone, attr, clone_val)

            return clone
        
        return _copy(instance)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    @classmethod
    def sanitize(cls, 
        fit_instance: object, 
        attrs_to_sanitize: list[str] | str | None = None
    ) -> list[tuple[object, str, Any]] | None:
        """
        Temporarily remove specified attributes from a fitted
        instance (for serialization), returning a reconstruction
        list to restore them afterward.

        Supports nested attribute paths (e.g., `'fitter.model'`).

        Parameters
        ----------
        fit_instance : object
            The fitted model instance to sanitize.
        attrs_to_sanitize : str | list[str] | None
            Attribute name(s) to remove. Supports dot-notation
            for nested paths.

        Returns
        -------
        list[tuple[object, str, Any]] | None
            Reconstruction list — pass to `reconstruct()` after
            IO to restore the removed attributes. None if nothing
            was sanitized.
        """
        if (
            fit_instance is None or
            not attrs_to_sanitize
        ):
            return None

        if isinstance(attrs_to_sanitize, str):
            attrs_to_sanitize = [attrs_to_sanitize]

        to_remove    = [a.split('.') for a in attrs_to_sanitize]
        reconstruct  = []

        def _nested_attr(f_i, attr):
            for a in attr[:-1]:
                f_i = getattr(f_i, a, None)
                if f_i is None:
                    return None
            attr = attr[-1]
            val  = getattr(f_i, attr, None)
            if val is None:
                return None            
            reconstruct.append( (f_i, attr, val) )
            return f_i, attr
                
        for attr in to_remove:
            try:
                fi_at = _nested_attr(fit_instance, attr)
                if fi_at is None:
                    continue
                delattr(*fi_at)
            except AttributeError:
                logging.warning(f"Sanitization failed for '{attr}' in {type(fit_instance).__name__}")
        
        return reconstruct 
    
    @staticmethod
    def reconstruct(reconstruct: list[tuple[object, str, Any]] | None) -> None:
        """
        Restore attributes previously removed by `sanitize()`.

        Parameters
        ----------
        reconstruct : list[tuple[object, str, Any]] | None
            The reconstruction list returned by `sanitize()`.
            If None, this is a no-op.
        """        
        if not reconstruct: 
            return
        for instance, attr, value in reconstruct:
            setattr(instance, attr, value)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def timestamp() -> str:
        """
        Returns the current local timestamp as a formatted string
        (`YYYY-MM-DD, HH:MM:SS`).

        Returns
        -------
        str
            Formatted timestamp.
        """        
        return datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DYNAMIC EVAL
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Dynamic:
    """
    Dynamic evaluation utilities for deferred (lazy) objects.

    Resolves objects that are stored as references or callables
    into their current concrete state. Enables the framework to
    maintain live references to data, parameters, and arguments
    without premature evaluation — critical for the flow
    architecture where data may change between stages.

    Resolves:
    - **Data**: Dataclasses with callable fields, or direct
      callables.
    - **Params**: Args/kwargs containing deferred objects.
    - **Injection**: Transformation functions applied to
      args/kwargs at the point of use.
    """
    
    @staticmethod
    def data(data_fx: Any | FuncType, *args, **kwargs) -> Any:
        """
        Resolve a deferred data object to its current concrete
        value.

        Handles:
        - Plain values (returned as-is).
        - Dataclasses with callable fields (evaluated in place).
        - Direct callables (invoked with args/kwargs).

        Parameters
        ----------
        data_fx : Any | FuncType
            The data object to resolve.
        *args, **kwargs
            Forwarded to callables if present.

        Returns
        -------
        Any
            The resolved concrete value.
        """
        data = data_fx
        if is_dataclass(data_fx):
            is_d = True
            if isinstance(data_fx, DictDataclass):
                d = dict(data_fx)
            else: 
                d = asdict(data_fx)
            n = {}
            for k,v in d.items():
                n[k] = v() if isinstance(v, FuncType) else v
                is_d = is_d and n[k] is v
            if not is_d:
                data = type(data_fx)(**n)
        if isinstance(data_fx, FuncType):
            data = data_fx(*args, **kwargs)
        return data
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def params(cls, 
        args: tuple | list = None, 
        kwargs: dict = None
    ) -> tuple[list, dict]:    
        """
        Resolve all deferred objects in args/kwargs to their
        current concrete values.

        Parameters
        ----------
        args : list | tuple | None
            Positional arguments (may contain deferred objects).
        kwargs : dict | None
            Keyword arguments (values may be deferred objects).

        Returns
        -------
        tuple[list, dict]
            (resolved_args, resolved_kwargs)

        Raises
        ------
        ValueError
            If `kwargs` is not a dict.
        """
        args = args or ()
        kwargs = kwargs or {}

        if not isinstance(args, (list,tuple)):
            args = [args]   # This should be generally OK to allow functions returning only one object
            
        if not isinstance(kwargs, dict):
            raise ValueError(
                RgbANSI.bold('AIG ► Dynamic kwargs evaluation') +
                ' ► Provided kwargs for evaluation have to be in a form of dictionary'
            )
        
        d_args = []
        d_kwargs = {}
        
        for a in args:
            a = cls.data(a)
            d_args.append(a)
            
        for p,v in kwargs.items():                
            v = cls.data(v)
            d_kwargs[p] = v
            
        return d_args, d_kwargs
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def get_parent_args_kwargs(cls, 
        args: tuple, 
        kwargs: dict, 
        pos_param: dict = None
    ) -> tuple:    
        """
        Resolve deferred objects at specific argument positions.

        Parameters
        ----------
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.
        pos_param : dict, default = `{0: 'inputs'}`
            Mapping of {index: kwarg_name} for the positions to
            resolve. `None` values skip that position.

        Returns
        -------
        tuple
            Resolved values at the specified positions.
        """
        prm = pos_param or {0: 'inputs'}

        params = []    
        for i,x in prm.items():
            if x is None: 
                continue
            v = args[i] if args[i:] else kwargs[x] 
            params.append( cls.data(v) )
        return tuple(params)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def inject_params(
        fx: FuncType | None, 
        args:  tuple = None, 
        kwargs: dict = None
    ) -> tuple[list, dict]:       
        """
        Apply a transformation function to args/kwargs at the
        point of use.

        Parameters
        ----------
        fx : FuncType | None
            Transformation function (`(*args, **kwargs) → (args, kwargs)`).
            If None, args/kwargs are returned unchanged.
        args : list | tuple | None
            Positional arguments to transform.
        kwargs : dict | None
            Keyword arguments to transform.

        Returns
        -------
        tuple[list, dict]
            Transformed (args, kwargs).
        """
        ar = args or ()
        kw = kwargs or {}
        if fx is not None:
            ar, kw = fx(*ar, **kw)
        return ar, kw
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PHASE STOP
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class StopExecution(Exception):
    """
    Clean execution halt — suppresses traceback rendering for
    user-facing error messages (IPython/Jupyter compatible).
    """    
    def _render_traceback_(self):
        return

class AigStop:
    """
    Standardized execution termination with user-facing diagnostic
    messages.

    All error messages follow a consistent format:
    - **RED BOLD**: The flow step identifier (what failed).
    - **BOLD**: The explanation (why it failed).
    - **DETAIL**: Specific values (missing variables, paths, etc.)

    Provides both generic errors (`custom`) and specific
    scenario handlers (untrained model, missing columns,
    path misconfiguration, flow pause).
    """

    @staticmethod
    def custom(
        message: str, 
        condition: bool = True, 
        bold: bool = True, 
        red: bool = True
    ) -> None:
        """
        Raise a custom execution halt with a formatted message.

        Parameters
        ----------
        message : str
            The error message to display.
        condition : bool, default=True
            If False, no message is shown and no exception is
            raised (used for conditional checks).
        bold : bool, default=True
            Apply bold formatting.
        red : bool, default=True
            Apply red+bold formatting (overrides bold).
        """        
        f = lambda x: x
        if bold: f = RgbANSI.bold
        if red:  f = RgbANSI.bold_red
        if condition:
            print(f(message))
        raise StopExecution
    
    @staticmethod
    def fx(
        condition: Any, 
        flow_step: str, 
        phase: Literal['fit','pred'] = 'fit'
    ) -> None:
        """
        Verify that a fitted model exists before execution.
        Raises with a user-facing message if the condition
        indicates the model is not ready.

        Parameters
        ----------
        condition : Any
            Truthy if the model is ready. Falsy (None, False)
            triggers the halt.
        flow_step : str
            The flow step identifier (displayed in red).
        phase : 'fit' | 'pred', default='fit'
            Determines the error message context.
        """
        if ( condition is not None ) and condition:
            return
        if phase == 'fit': 
              m = "\n► Model has to be trained on training sample. Please provide data-variables for training."
        else: m = "\n► Model has not been trained. Please train the model before prediction."
        message = RgbANSI.bold_red(flow_step), '\n', RgbANSI.bold(m)
        print(*message)
        raise StopExecution

    @staticmethod
    def cols(flow_step:str, vars_miss:list) -> None:
        """
        Halt execution if required variables are missing from
        the scoring data.

        Parameters
        ----------
        flow_step : str
            The flow step identifier (displayed in red).
        vars_miss : list
            Missing variable names. If empty, no halt occurs.
        """        
        if not vars_miss:
            return
        m = ( 
            f'\n► AIG EXECUTION HAS BEEN STOPPED ► Stored Trained Model has {len(vars_miss)}'
            f' Variables in use which none is present in provided Scoring Data'
        )
        v = f'\n► Missing Variables:\n {', '.join(vars_miss)}'        
        message = RgbANSI.bold_red(flow_step), RgbANSI.bold(m), v
        print(*message)
        raise StopExecution

    @staticmethod
    def flow_execution(
        pause: bool = True, 
        phase: Literal['Train','Score'] = 'Train'
    ) -> None:
        """
        Optionally pause automatic flow execution at a phase
        boundary (for interactive verification).

        Parameters
        ----------
        pause : bool, default=True
            If True, halt execution and display a completion
            message. If False, continue silently.
        phase : 'Train' | 'Score', default='Train'
            The completed phase (determines the message).
        """        
        if not pause:
            return
        m = lambda p: f' ► Auto execution stopped ► Please continue to the next phase once {p} is verified'
        if phase == 'Train': message = RgbANSI.bold_green('TRAINING PHASE DONE √') + m(phase)
        if phase == 'Score': message = RgbANSI.bold_green('SCORING PHASE DONE √')  + m(phase)
        print(message)
        raise StopExecution
    
    @staticmethod
    def is_flow_init(instance: object, attr: str) -> None:
        """
        Verify that a flow instance has been properly initialized
        (saved). Raises if the specified attribute is missing.

        Parameters
        ----------
        instance : object
            The model/report instance to check.
        attr : str
            The attribute that must be present (e.g., `'config'`).
        """        
        v = getattr(instance, attr, None)
        if v is not None:
            return
        m = RgbANSI.bold_red(
            'Models-Reports can be operated (load/save) only for saved instance of AID.'
            '\nPlease use Predict class while specifying PROJECT/PATH under which AID '
            'models were trained and saved.'
        )
        print(m)
        raise StopExecution
    
    @staticmethod
    def verify_model_path(model: str, path: str|None) -> None:
        """
        Verify that a save path is provided when required by the
        flow configuration.

        Parameters
        ----------
        model : str
            Model identifier (for the error message).
        path : str | None
            The save path. If None, raises (misconfiguration).
        """        
        if path is not None:
            return
        m = RgbANSI.bold_red(
            f'Model-Dashboard save IO obtained unexpected setting during ► {model} ◄ flow step.'
            '\nConfiguration of `save=True`, path=None must not occur by AIG STRUCTURE default design.'
            '\nCustom Saving without providing config Parameters requires custom path to target destination.'
            '\nOtherwise please check - CTX.project_paths.models / .reports - each model must have its '
            'own unique default path.'
        )
        print(m)
        raise StopExecution
    
    @staticmethod
    def verify_context(
        model: str, 
        instance: object, 
        attr: tuple | str = ('config','cfg')
    ) -> None:
        """
        Verify that a loaded model instance has the required
        context attributes for dashboard operations.

        Parameters
        ----------
        model : str
            Model identifier (for the error message).
        instance : object
            The loaded model instance to check.
        attr : str | tuple[str], default=('config', 'cfg')
            Attribute name(s) that must be present.
        """        
        if attr is None:
            return
        if isinstance(attr, str):
            attr = [attr]            
        if any(hasattr(instance, a) for a in attr):
            return
        m = RgbANSI.bold_red(
            f'Stored dashboard of loaded {model} model can be saved only by providing a custom_path.'
        )
        print(m)
        raise StopExecution

    @staticmethod
    def jn_getattr(name:str) -> None:
        """
        Block IPython/Jupyter internal attribute access
        (`_ipython_*`, `_repr_*`) to prevent rendering
        artifacts in notebooks.

        Parameters
        ----------
        name : str
            The attribute name being accessed.

        Raises
        ------
        AttributeError
            Always raised (prevents the attribute access).
        """
        if any([name.startswith(x) for x in ['_ipython_', '_repr_']]):
            raise AttributeError(name)

    @classmethod
    def pause_auto_execution(cls, 
        pause: bool = True, 
        phase: Literal['Train','Score'] = 'Train'
    ) -> None:
        """
        Convenience helper to pause `Run All` execution at a
        phase boundary.

        Parameters
        ----------
        pause : bool, default=True
            If True, halt execution. If False, continue.
        phase : 'Train' | 'Score', default='Train'
            The completed phase.
        """                
        cls.flow_execution(pause=pause, phase=phase)        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬