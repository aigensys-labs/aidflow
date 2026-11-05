"""
Type Factory providing high-performance, immutable schema definitions.

This module implements a specialized dataclass framework designed for ML worker 
arguments and results. By synthesizing Python's `dataclasses` with `__slots__`, 
the `Mapping` interface, and custom metaclasses, it provides:

1. Memory Efficiency: Up to 5x faster look-up speeds and significantly reduced 
   memory footprints via slot optimization.

2. Strict Immutability: Frozen structures that prevent accidental runtime 
   modification of pipeline configurations (Provenance Insurance).

3. Hybrid Access: Type-safe attribute access combined with dictionary-like 
   mapping for seamless integration with external config files and APIs.
   
4. Dynamic Validation: Automated validation and mapping of parameters through 
   `MappingDataclass` to ensure structural integrity across the execution flow.
"""

from ..deps.common import (
    ABCMeta,
    inspect, 
    wraps,
    wrapt
)
from ..deps.struct import (
    Mapping, 
    dataclass, 
    is_dataclass,
    field, 
    fields, 
    asdict, 
    MISSING
)
from ..deps.types import (
    Any, 
    Literal,
    Callable,
    TypeVar,
    ClassVar, 
    Iterator, 
    Iterable, 
    Union, 
    FuncType,
    MappingProxyType,
    get_origin, 
    get_args, 
    dataclass_transform,
    extract_types
)

T  = TypeVar("T",  bound=type)
P1 = TypeVar("P1", bound=type)
P2 = TypeVar("P2", bound=type)


class MemoryCell(wrapt.ObjectProxy):
    """Transparent reference cell. 
    All holders share the same Cell object = mem ref.
    """
    def __init__(self, wrapped=None):
        super().__init__(wrapped)

    def __repr__(self):
        return repr(self.value)

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

    @property
    def value(self):
        return self.__wrapped__

    @value.setter
    def value(self, new_value):
        self.__wrapped__ = new_value


class DictDataclass(Mapping):
    """
    Base Mapping-like class to provide dict features to dataclass.
    
    Usage Example
    -------------
    >>> @dataclass(frozen=True, slots=True)
        class Standards(DictStandards):
            ids: list
            col1: str = "transaction_date"
            col2: str = "churn_flag"
    
    1) 2-5x faster look-up speed when using slots and half mem-usage
    2) frozen ensures immutable data structure of directly defined attrs
    3) DictDaclass provides both exact type check for vars and dict-like behavior
    """
    
    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            # Allowing Function to be part of input as AIG flow uses dynamic inputs
            if value is None:
                continue  # allow None for optional fields
            self._enforce_type(f.name, value, f.type)

    def _enforce_type(self, field_name: str, value: Any, expected_type: Any):
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Handle Union / Optional        
        if origin is Union:
            if not any(
                arg is type(None) or isinstance(value, arg) for arg in args
            ):
                types_str = ", ".join(a.__name__ if a is not type(None) else "None" for a in args)
                raise TypeError(f"Field '{field_name}' must be one of ({types_str}), got {type(value).__name__}")
            return  # valid, skip further checks

        # Handle base types
        if origin is None:
            if not isinstance(value, expected_type):
                raise TypeError(f"Field '{field_name}' must be {expected_type.__name__}, got {type(value).__name__}")
            return

        # Containers
        if origin in (list, set):
            if not isinstance(value, origin):
                raise TypeError(f"Field '{field_name}' must be {origin.__name__}")
            if args:
                inner_type = args[0]
                for i, item in enumerate(value):
                    self._enforce_type(f"{field_name}[{i}]", item, inner_type)

        elif origin is tuple:
            if args and args[-1] is Ellipsis:
                # variable-length tuple of type args[0]
                for i, item in enumerate(value):
                    self._enforce_type(f"{field_name}[{i}]", item, args[0])
            elif args:
                # fixed-length tuple
                if len(args) != len(value):
                    raise TypeError(f"Field '{field_name}' must be a tuple of length {len(args)}, got {len(value)}")
                for i, (item, t) in enumerate(zip(value, args)):
                    self._enforce_type(f"{field_name}[{i}]", item, t)
            else:
                # plain tuple with no args, just check type
                if not isinstance(value, tuple):
                    raise TypeError(f"Field '{field_name}' must be a tuple")
    
    def __getitem__(self, key) -> Any:
        if isinstance(key, int):
            return tuple(self)[key] # tuple-like access    
        return getattr(self, key)
    
    def __iter__(self) -> Iterator[str]: 
        return (f.name for f in fields(self))
    
    def __len__(self)  -> int:  
        return len(fields(self))
    
    def keys(self) -> tuple:
        return tuple(self)
    
    def values(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self))

    def items(self) -> tuple:
        return tuple((f.name, getattr(self, f.name)) for f in fields(self))
    
    def key(self) -> str:   
        """Returns the first key."""
        return next(iter(self)) # double perf. than list(dict)[0]
    
    def value(self) -> Any: 
        """Returns the first value."""        
        return next(iter(self.values()))
    
    def item(self) -> tuple[str, Any]:
        """Returns the first item."""
        return next(iter(self.items()))
    
    def to_dict(self) -> dict:
        """Returns dictionary repr of `ALL sub-dataclasses (!)` within the full structure."""
        return asdict(self)  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class ValidationDataclass:
    """Base class that auto-validates dataclass-typed init arguments."""
    
    def __init_subclass__(cls):
        """Intercept subclass creation to wrap its __init__ automatically."""
        orig_init = cls.__init__
        
        try:
            true_sig = inspect.signature(orig_init)
        except ValueError:
            true_sig = None

        @wraps(orig_init)
        def __init__(self, *args, **kwargs):
            # Bind args to parameter names
            bound = inspect.signature(orig_init).bind_partial(self, *args, **kwargs)
            bound.apply_defaults()

            # Inspect type annotations
            for name, param in cls.__annotations__.items():
                if not inspect.isclass(param) or not is_dataclass(param):
                    continue # skip non-dataclass types

                value = bound.arguments.get(name, getattr(self, name, None))

                # Get default value if defined
                default = getattr(cls, name, MISSING)
                allow_none = default is None

                if value is None:
                    if not allow_none:
                        raise TypeError(f"Parameter '{name}' cannot be None.")
                    continue
                
                if not isinstance(value, param):
                    raise TypeError(
                        f"Parameter '{name}' must be of type {param.__name__}, "
                        f"got {type(value).__name__}."
                    )

            orig_init(self, *args, **kwargs)
        
        if true_sig:
            __init__.__signature__ = true_sig
            
        cls.__init__ = __init__
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@dataclass_transform()
class _SchemaMapping(Mapping):
    
    def __init__(self, schema:dict):
        self._d = schema

    def __getitem__(self, key) -> Any:
        if isinstance(key, int): 
            key = tuple(self._d)[key] 
        return self._d[key]
    
    def __iter__(self) -> Iterator[str]: 
        return iter(self._d)
    
    def __len__(self)  -> int:  
        return len(self._d)
    

class _DataclassMapping:
    params: type
    setup:  dict[str, str]

type M = Union["MappingDataclass", _DataclassMapping]

class MappingDataclass:
    """
    Base class that auto-validates dataclass-typed init arguments.

    Parameters
    ----------
    params : dict | DictDataclass
        Values of parameters, supports to pass the full instance of the Flow Parameters. 
        When params are provided to Resulting Mapping, returned instance is the Source Param class
        populated by values from params.

    default : bool, default False
        When True returned instance is aligned with source params - subset by defined mapping -
        with params name by setup keys, i.e. Flow Parameters with values populated from params.
    """
    
    def __init_subclass__(cls: M):
        """Intercept subclass creation to wrap its __init__ automatically."""

        if any([not hasattr(cls, x) for x in _DataclassMapping.__annotations__]):
            raise ValueError('Mapping dataclass must define params and setup.')
        
        prm_schema = type(cls.params)

        if not issubclass(prm_schema, _SchemaMapping):
            raise TypeError(f'Mapping dataclass must define params by Dataclass.schema(SourceParams)' )
        
        cls._prm_source_cls = prm_schema.__dict__['__cls__']  # design in Dataclass.schema
        
        if (
            not isinstance(cls._prm_source_cls, type) or 
            not issubclass(cls._prm_source_cls, DictDataclass)
        ):
            raise TypeError(f'Source Params must be DictDataclass type.' )

        cls._setup, cls._panel_desc = cls._get_setup()
        cls._prm_names = Dataclass.get_names(cls._prm_source_cls)

    def __new__(cls: M, params: dict, default: bool = False) -> DictDataclass:
        
        if is_dataclass(params):
            if isinstance(params, DictDataclass):
                  params = dict(params)
            else: params = asdict(params)

        params_ = params        
        
        # Mapping from flow params naming to the source params
        if all([p_flow in params for p_flow in cls._setup]):
            params_ = {p_source: params[p_flow] for p_flow, p_source in cls._setup.items()}
        
        # Verify that we have the right params input
        if not all([p in cls._prm_names for p in params_]):
            raise ValueError('Provided params are not matching with the source:'
                f'\n► Source Params Base: {cls._prm_source_cls}'
                f'\n► Provided mapping:\n {cls._setup}'
            )
        
        if default:
            return cls.create_default_instance(params)

        return cls.create_prm_instance(params_)
    
    @staticmethod
    def _get_prm_desc(p_source: str|tuple) -> tuple[str, None|str]:
        if isinstance(p_source, str):
            return p_source, None
        e = lambda: TypeError(
            'Description to UI Panel options must follow expected design:'
            '\n► The second element in a tuple must be a string type '
            'representing widget name of the parameter; while first source param name.'
            '\n► e.g. setup = { PRM.LOAD.. : (params.starting_point, "Custom Option Desc") }'
        )                
        if not isinstance(p_source, (tuple, list)):
            raise e()
        if not len(p_source) == 2:
            raise e()
        return p_source
    
    @classmethod
    def _get_setup(cls: M) -> tuple[dict[str,str], dict[str, None|str]]:
        if not cls.setup:
            return {}, {}
        raw = {flow: cls._get_prm_desc(src) for flow, src in cls.setup.items()}
        setup = {k: v[0] for k, v in raw.items()}
        desc  = {k: v[1] for k, v in raw.items()}
        return setup, desc     
    
    @classmethod
    def get_desc(cls: M):
        """
        Returns optional description of UI Panel widget options if provided in Mapping;
        otherwise defaults to populated params dict with None.
        """
        return cls._panel_desc
    
    @classmethod
    def create_default_instance(cls:M, params:dict) -> DictDataclass:
        """Returns instance of mapped defaults to fitter class with provided params."""
        return Dataclass.create_params({p_flow: params[p_flow] for p_flow in cls._setup})   
    
    @classmethod
    def create_prm_instance(cls: M, params:dict) -> M:
        """Returns instance of registred fitter class with provided params."""
        return cls._prm_source_cls(**params)   
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DataclassUtils:
    """Helper to provide common functionalities to Dataclass."""

    anno = '__annotations__'

    @staticmethod
    def has_slots(specs:dict|None) -> bool:
        return isinstance(specs, dict) and specs.get('slots', False)
    
    @staticmethod
    def get_cls(class_or_instance) -> type:
        """Helper to return class type regardless if input is instance or class."""
        if not isinstance(class_or_instance, type):
            return type(class_or_instance)
        return class_or_instance
    
    @staticmethod
    def get_names(class_or_instance) -> list:
        """Helper returning fields names of Dataclass."""
        return [k.name for k in fields(class_or_instance)]

    @classmethod
    def get_params_type(cls, params:dict) -> dict[str, Any]:
        """Returns annotations and parameters types of dataclass."""
        def types(v):
            if not isinstance(v, (list, dict)):
                return v
            return field(default_factory = lambda: v)        
        annots = {k: type(v)  for k,v in params.items()}
        params = {k: types(v) for k,v in params.items()}        
        return {cls.anno: annots, **params}
    
    @classmethod
    def collect_params(cls, target_cls) -> dict[str, Any]:
        """Collect all class attributes from cls and its parents."""
        annots = {}
        params = {}
        for base in list(reversed(target_cls.__mro__))[:-1]:
            if base is object:
                continue            
            dct = {
                k:v for k,v in base.__dict__.items()
                if not (k.startswith('_') or callable(v))
            }
            prm = cls.get_params_type(dct)
            ant = prm.pop(cls.anno)
            annots = annots | ant
            params = params | prm

        tdct = target_cls.__dict__
        tant = tdct.get(cls.anno, {})
        tdct = {k:v for k,v in tdct.items() if k!=cls.anno}
        f = {cls.anno: annots | tant, **params, **tdct}
        return {cls.anno: annots | tant, **params, **tdct}

class _DynamicDataclass:

    class DictAddBase:
        pass   

    @staticmethod
    def _none_id(val: Any) -> Any:
        if isinstance(val, MemoryCell):
            if val.value is None:
                return None
        return val

    @classmethod
    def _get_val(cls, val: Any) -> Any:
        val = cls._none_id(val)
        return val() if isinstance(val, FuncType) else val

    @staticmethod
    def _is_cell(is_cell:bool, c_type:Any, val:Any, existing:Any=None) -> bool:
        if not is_cell:
            return False
        if c_type is None:
            return True
        if isinstance(existing, MemoryCell):
            return True
        return type(val) is c_type
    
    class SlotParam:
        """
        Allows `slots based param` classes to return actual value 
        when requested while it can be stored as a callable.
        We engage the fact that slots' attrs are data descriptors.
        """
        __slots__ = ('_param', '_is_cell', '_c_type')
                
        def __init__(self, param, set_cell: bool = False, cell_type: Any = None):
            self._param   = param
            self._is_cell = set_cell
            self._c_type  = cell_type

        def __get__(self, instance, owner):
            if instance is None:
                return self
            val = self._param.__get__(instance, owner)
            return _DynamicDataclass._get_val(val)

        def __set__(self, instance, value):
            try:
                existing = self._param.__get__(instance, None)
            except AttributeError:
                existing = None

            if _DynamicDataclass._is_cell(self._is_cell, self._c_type, value, existing):
                if isinstance(existing, MemoryCell):
                    existing.value = value 
                    return # mutate in place — holders stay linked
                
                value = value if isinstance(value, MemoryCell) else MemoryCell(value)

            self._param.__set__(instance, value)
            
        def __delete__(self, instance):
            self._param.__delete__(instance)

    class DictParam:
        """
        Allows `dict based param` classes to return actual value 
        when requested while it can be stored as a callable.
        We wrap the class and dynamically resolve args call.
        """
        def __repr__(self):
            instance = self.__dict__.get('_instance')
            return repr(instance) if instance else repr(self._cls)
        
        def __init__(self, cls:type, set_cell:bool=False, cell_type:Any=None):
            self.__dict__['_cls'] = cls
            self._is_cell = set_cell
            self._c_type = cell_type

        def __call__(self, *args, **kwargs):
            ar, kw = self._set_cell(args, kwargs)
            self.__dict__['_instance'] = self._cls(*ar, **kw)
            return self
        
        def __getattr__(self, name):
            instance = self.__dict__.get('_instance')
            target = instance if instance is not None else self._cls
            
            try:
                val = getattr(target, name)
            except AttributeError:
                raise AttributeError(f"'{target.__name__}' has no attribute '{name}'")

            return _DynamicDataclass._get_val(val)

        def __setattr__(self, name, value):
            ins = self.__dict__['_instance']
            dict_: dict = ins.__dict__
            existing = dict_.get(name)

            if _DynamicDataclass._is_cell(self._is_cell, self._c_type, value, existing):
                if name in dict_ and isinstance(dict_[name], MemoryCell):
                    dict_[name].value = value       # mutate existing Cell
                else:
                    dict_[name] = MemoryCell(value) # first write → create Cell
            else:
                dict_[name] = value

        def _set_cell(self, args:tuple, kwargs:dict) -> tuple[list, dict]:
            if not self._is_cell:
                return args, kwargs
                        
            c = lambda v: (
                MemoryCell(v) 
                if _DynamicDataclass._is_cell(self._is_cell, self._c_type, v) and
                not isinstance(v, MemoryCell)
                else v
            )

            new_ar = []
            for a in args:
                new_ar.append(c(a))

            new_kw = {}
            for k,v in kwargs.items():
                new_kw[k] = c(v)

            return new_ar, new_kw

class _DataclassIO:
    """
    Allow `internal objects` in dynamic __dict__ for `slots=True dataclass`.
    """

    @staticmethod
    def get_all_slots(cls:type) -> list:
        """
        Extract all slot-based attribute names, handling mangling 
        and infrastructure descriptors. Matches copyreg behavior.
        """        
        # extended v. in copyreg._slotnames
        
        # Return Cache if we stored it already
        if hasattr(cls, "__slotnames__"):
            return cls.__slotnames__
            
        names = []
        for base in cls.__mro__:
            slots = getattr(base, '__slots__', [])
            if slots is None:
                continue     

            # __slots__ can be a string or an iterable
            if isinstance(slots, str):
                slots = (slots,)

            for name in slots:
                # Skip infrastructure
                if name in ("__dict__", "__weakref__"):
                    continue
                
                # Handle Name Mangling: __attr -> _ClassName__attr
                if name.startswith('__') and not name.endswith('__'):
                    # Ensure we use the base class name where the slot was defined
                    stripped = base.__name__.lstrip('_')
                    names.append(f"_{stripped}{name}")
                else:
                    names.append(name)
        
        # Attempt to Cache
        try:
            cls.__slotnames__ = names
        except (AttributeError, TypeError):
            pass                    
        return names
            
    @classmethod
    def get_set_state(cls, new_cls:type, specs:dict) -> None:
        """Allow `internal objects` in dynamic __dict__ for `slots=True dataclass`."""
        if not DataclassUtils.has_slots(specs):
            return
        
        def __getstate__(self):
            """Persistent helper for internal objects."""      
            slots = cls.get_all_slots(self.__class__)   # collects attrs from slots in mro
            state = {slot: getattr(self, slot) for slot in slots if hasattr(self, slot)}
            if hasattr(self, "__dict__"): 
                state.update(self.__dict__)             # dynamic dict for internal state
            return state

        def __setstate__(self, state:dict):
            """Persistent helper for internal objects."""
            for k, v in state.items():
                object.__setattr__(self, k, v)  # we have to use __setattr__ as slots=True
                                                # by definition doesn't allow setattr
        new_cls.__getstate__ = __getstate__
        new_cls.__setstate__ = __setstate__

class _DataclassSignature(ABCMeta):
    """
    Provides correct signature to dynamically create new type.
    Redirects checks to look at the original blueprints.
    """

    def __subclasscheck__(cls, subclass):
        """Redirects subclass checks to look at the original blueprints."""
        if not isinstance(subclass, type):
            raise TypeError("issubclass() arg 1 must be a class")
            
        # Extract the original target if the class was mutated by the factory
        sub_orig = getattr(subclass, '__original__', subclass)
        cls_orig = getattr(cls, '__original__', cls)
        
        # 1. Direct identity check of the true blueprints
        if sub_orig is cls_orig:
            return True
            
        # 2. Fall back to checking the original base history
        sub_bases = getattr(sub_orig, '__bases__', ())
        return (
            any(cls_orig == getattr(b, '__original__', b) for b in sub_bases) or
            super().__subclasscheck__(subclass)
        )

    def __instancecheck__(cls, instance):
        """Redirects isinstance checks seamlessly."""
        return cls.__subclasscheck__(type(instance))
    
class _DataclassSpecs:
    """
    Helper to provide specs standard to target_cls with:
    - multiple bases (base can be tuple of classes)
    - dataclass machinery with given specs
    - optional dynamic internal slots support
    - internal state helpers (_getstate/_setstate) for frozen/mutable objects
    """

    @staticmethod
    def _is_mro_slots(cls:type) -> bool:
        """
        Returns True if __slots__ is explicitly defined anywhere 
        in the class inheritance hierarchy.
        """        
        if cls.__dict__.get('__slots__'):
            return True
        return any(layer.__dict__.get('__slots__') for layer in cls.__mro__)
                
    @classmethod
    def _ensure_dict(cls, bases:tuple[type], specs:dict) -> tuple:
        """Allow `internal objects` in dynamic __dict__ for `slots=True dataclass`."""
        if not DataclassUtils.has_slots(specs):
            return ()
         
        for base in bases:
            for cls_ in base.__mro__:
                if cls_ is object:
                    continue
                slots = cls_.__dict__.get('__slots__')
                # If an ancestor completely omits __slots__, it natively has a __dict__
                if slots is None:
                    return ()                
                if isinstance(slots, str): 
                    slots = (slots,)
                if '__dict__' in slots:
                    return ()
                
        return (_DynamicDataclass.DictAddBase,)
   
    @classmethod
    def _dynamic_params(cls, new_cls:type, specs:dict, cell_params:bool, cell_type:Any):
        """Allow `internal objects` in dynamic __dict__ for `slots=True dataclass`."""        
        if is_dataclass(new_cls):
            params = DataclassUtils.get_names(new_cls)
        else: 
            params = inspect.signature(new_cls).parameters
        
        is_slots = DataclassUtils.has_slots(specs)

        if not is_slots:
            return _DynamicDataclass.DictParam(new_cls, cell_params, cell_type) 
        
        for p in params:
            v = getattr(new_cls, p, None)
            setattr(new_cls, p, _DynamicDataclass.SlotParam(v, cell_params, cell_type))
        return new_cls

    @classmethod
    def _strip_slots(cls, base:type) -> type:
        """
        Removes both '__slots__' and their corresponding member descriptors 
        from parents to guarantee a pure metadata inheritance layout.        
        """
        if not cls._is_mro_slots(base):
            return base
        
        d = dict(base.__dict__)
        s = d.pop('__slots__', None)

        if s is not None:        
            for slot_key in s:
                d.pop(slot_key, None)
            
        return type(
            f"{base.__name__}__PureMeta", 
            tuple(cls._strip_slots(b) for b in base.__bases__),  
            d
        )

    @classmethod
    def _safe_base(cls, bases: tuple[type, ...]) -> tuple[type, ...]:
        """Unifies bases to provide a safe slot base."""
        safe_bases = []
        for b in bases:                
            safe_bases.append(cls._strip_slots(b))
        return tuple(safe_bases)

    @classmethod
    def _safe_slot_base(cls,
        target_cls: type, 
        bases: tuple[type, ...]
    ) -> tuple[type, tuple[type,...]]:
        """
        Unifies bases to provide a safe slot base.
        Removes both '__slots__' and their corresponding member descriptors 
        from parents to guarantee a pure metadata inheritance layout.
        """        
        return cls._strip_slots(target_cls), cls._safe_base(bases)
    
    @staticmethod
    def _base_order(target_cls:type, bases:tuple[type, ...], top_base:bool) -> tuple[type]:
        """Returns tuple of bases for class creation by type."""
        if (
            not bases or 
            all(issubclass(target_cls, b) for b in bases)
        ):
            return (target_cls,)
        if top_base:
              return (*bases, target_cls)
        return (target_cls, *bases) 


    @classmethod
    def _create_new_cls(cls, 
        target_cls:     type, 
        bases:          tuple[type, ...], 
        specs:          dict = None, 
        top_base:       bool = True
    ) -> type:
        """Creates new class with with desired mro order and dict-like design."""        
        
        target_cls_, bases_ = cls._safe_slot_base(target_cls, bases)        

        new_cls = _DataclassSignature(
            target_cls.__name__, 
            (
                *cls._base_order(target_cls_, bases_, top_base), 
                *cls._ensure_dict(bases + (target_cls.__bases__,), (specs or {}))
            ), 
            dict(target_cls_.__dict__)
        )

        new_cls.__original__ = target_cls
        new_cls.__module__   = target_cls.__module__
        new_cls.__qualname__ = target_cls.__qualname__    

        return new_cls
    
    @classmethod
    def _create_dataclass(cls, new_cls:type, specs:dict) -> type:
        new_cls_ = dataclass(**specs)(new_cls)        
        _DataclassIO.get_set_state(new_cls_, specs)
        return new_cls_
    
    @classmethod
    def apply(cls, 
        base = None, 
        specs: dict|None = None, 
        top_base: bool = True, 
        dynamic_params: bool = False,
        cell_params: bool = False,
        cell_type: Any = None
    ) -> Callable[[T], T]:
        """
        Universal decorator to dynamically wrap target_cls with:
        - multiple bases (base can be tuple of classes)
        - dataclass machinery with given specs
        - optional dynamic internal slots support
        - internal state helpers (_getstate/_setstate) for frozen/mutable objects
        """
        def decorator(target_cls: type):            
            base_ = base if isinstance(base, tuple) else((base,) if base else ())
            
            if (
                ( specs is None or not specs.get('slots') ) and 
                ( not base or all(issubclass(target_cls, b) for b in base_) )
            ):
                new_cls = target_cls                
            else:
                new_cls = cls._create_new_cls(target_cls, base_, specs, top_base)

            if specs is not None:
                new_cls = cls._create_dataclass(new_cls, specs)
            
            if dynamic_params or cell_params:
                new_cls = cls._dynamic_params(new_cls, specs, cell_params, cell_type)

            return new_cls
        return decorator
                
class Dataclass:
    """Dataclass base design for workers arguments/results."""

    @classmethod
    @dataclass_transform()
    def base(cls, 
        param_cls: T | None = None, 
        /, *, 
        base: tuple[type, ...] = (DictDataclass,),
        top_base:         bool = True,
        specs:    tuple | dict = (('frozen', True), ('slots',True), ('kw_only', True)),
        update_specs :    dict = None,
        dynamic_params:   bool = False,
        cell_params:      bool = False,
        cell_type:         Any = None
    ) -> Callable[[T], T]:
        """
        Decorator for Workers arguments/results base Dataclass design.

        **Default specs**
        - specs = `dict(frozen=True, slots=True, kw_only=True)`
        - update_specs = {}

        Usage 1 - without any kwargs
        --------        
        >>> @Dataclass.base
            class MyParams

        which is equivalent to :
        >>> @dataclass(frozen=True, slots=True, kw_only=True)
            class MyParams(DictDataclass)

        Usage 2 - with up to 2 kwargs
        --------        
        base  : base class for the param class
        specs : for dataclass(**specs)
        
        >>> @Dataclass.base(base=BaseClass, specs=dict(...))
            class MyParams

        it would be same as defining param class as:        
        >>> @dataclass(**specs)
            class MyParams(BaseClass)
        """
        specs = dict(specs) | (update_specs or {})
        
        wrap: Callable[[T], T] = _DataclassSpecs.apply(
            base, specs, top_base, dynamic_params, cell_params, cell_type
        )
        # Called either as @Dataclass.base or @Dataclass.base()
        if param_cls is None:            
            return wrap         # @Dataclass.base()
        return wrap(param_cls)  # @Dataclass.base

    @classmethod
    @dataclass_transform()
    def verify(cls, 
        worker_cls: T | None = None,
        /, *,
        base: tuple[type, ...] = (ValidationDataclass,)
    ) -> Callable[[T], T]:   
        """
        Decorator for Workers accepting Dataclass params base design → auto verification.

        Usage 1 - without any kwargs
        --------        
        >>> @verify
            class Worker

        which is equivalent to :
        >>> class Worker(DataclassParams)

        Usage 2 - with 1 kwarg
        --------        
        base  : base class for the worker params check class
        
        >>> @verify(base=BaseClass)
            class Worker

        it would be same as defining param class as:        
        >>> class Worker(BaseClass)
        """        
        wrap: Callable[[T], T] = _DataclassSpecs.apply(base)
        if worker_cls is None:            
            return wrap          # @Dataclass.verify()
        return wrap(worker_cls)  # @Dataclass.verify

    @classmethod    
    def mapping(cls, 
        param_cls: T | None = None, 
        /
    ) -> Callable[[MappingDataclass], MappingDataclass]:
        """
        Decorator helper for Workers params mapping to Internal Fitters.

        Mapping Class `must define 2 class attributes`:
        - params - where Params Source Class is linked as: params = Dataclass.schema(FitParamsBase)
        - setup - mapping dict between flow parameters and worker source params with opt. desc (prm_source, 'desc')
          - when no mapping, turn off: setup = None

        `Resulting Mapping Class` accepts only 2 args:
        - params : dict | DictDataclass
            - Values of parameters, supports to pass the full instance of the Flow Parameters. 
            When params are provided to Resulting Mapping, returned instance is the Source Param class
            populated by values from params.

        - default : bool, default False
            - When True returned instance is aligned with source params - subset by defined mapping -
            with params name by setup keys, i.e. Flow Parameters with values populated from params.        
        """
        wrap = _DataclassSpecs.apply(MappingDataclass)
        if param_cls is None:            
            return wrap         # @Dataclass.mapping()
        return wrap(param_cls)  # @Dataclass.mapping

    @classmethod
    def create_params(cls, params:dict, name:str='custom') -> DictDataclass:
        """Helper to create dynamic params instance in standard of DictDataclass."""
        return cls.base(
            type(f'dynamic_params_of_{name}', (), DataclassUtils.get_params_type(params))
        )()

    @classmethod
    def _schema_path(cls, 
        cls_: type, 
        path: Literal['last', 'full'] = 'full', 
        _prefix: str = ''
    ) -> dict:
        s = {}
        for f in fields(cls_):
            if is_dataclass(f.type):
                s[f.name] = cls._schema_path(f.type, path, f"{_prefix}{f.name}.")
            else:
                s[f.name] = f"{_prefix}{f.name}" if path == 'full' else f.name
        return type(
            f'dynamic_schema_of_{cls_.__name__}', (_SchemaMapping, ), 
            dict(__cls__ = cls_) | s
        )(s)
        
    @classmethod
    def schema(cls, class_or_instance: T, path_type:Literal['last', 'full'] = 'full') -> T:
        """
        Helper to create dynamic class from dataclass where `param returns its name`.        
        - path_type = 'last' → returns the last attr name from nested attrs
        - path_type = 'full' → returns the full path to the attr for nested dataclasses

        e.g. schema(ClsX.ClsY.attr2):
        - 'last' → 'attr2'
        - 'full' → 'ClsY.attr2'
        """
        cls_ = DataclassUtils.get_cls(class_or_instance)
        if not is_dataclass(cls_):
            raise TypeError(f'Source Params must be dataclass type.')
        return cls._schema_path(cls_, path_type)
    
    @staticmethod
    def get_names(class_or_instance) -> list:
        """Helper returning fields names of Dataclass."""
        return DataclassUtils.get_names(class_or_instance)
    
    @staticmethod
    def get_mandatory_fields(class_or_instance) -> list:
        """Helper returning fields which are mandatory to fill."""
        cls_ = DataclassUtils.get_cls(class_or_instance)
        d = []        
        for f in fields(cls_):
            if f.default is f.default_factory:
                d.append(f.name)        
        return d

    @classmethod
    def get_fields_type(cls, class_or_instance) -> dict:
        """Helper returning dict of param - its type."""        
        cls_ = DataclassUtils.get_cls(class_or_instance)
        d = {}
        for f in fields(cls_):
            d[f.name] = f.type
        return d
    
    @classmethod
    def get_default_values(cls, class_or_instance) -> dict:
        """
        Helper returning dict of params with default values. 
        Params w/o default (mandatory fields) are filled with None.
        """
        cls_ = DataclassUtils.get_cls(class_or_instance)
        m = cls.get_mandatory_fields(cls_)
        d = {}
        for f in fields(cls_):
            n = f.name
            if n in m:
                d[n] = None
            else:
                v = f.default
                if f.default_factory is not MISSING:
                    v = f.default_factory()
                d[n] = v
        return d

    @staticmethod
    def field(value: Any) -> field:
        """Helper returning field(default_factory=lambda:value)."""
        return field(default_factory=lambda:value)

    @staticmethod
    def _default_for_type(t) -> Any:
        """Instantiate a default value from a type annotation."""
        types = extract_types(t)

        if len(types) == 1:
            return types[0]()
        
        for t in types:
            if t is type(None):
                continue
            try:
                return t()
            except TypeError:
                return None
        
    @classmethod
    def create_default_instance(cls, class_or_instance:DictDataclass) -> DictDataclass:
        """
        Helper returning class instance with all params as empty fields by their type.
        NOTE that fields are created as f.type(), e.g.>
        
        class FitResult:
          x: dict
          y: list
          z: str

        → returned instance will be with:  
         - x = {}
         - y = []
         - z = ''
        """
        cls_ = DataclassUtils.get_cls(class_or_instance)
        default = {f.name: cls._default_for_type(f.type) for f in fields(cls_)}
        return cls_(**default)    

    @staticmethod        
    def merge(
        *parents: tuple[P1, P2],
        specs: dict = dict(frozen=True, slots=True, kw_only=True)
    ) -> Callable[[type[P1|P2]], type[P1|P2]]:
        """
        Class decorator merging multiple dataclass parents into unified dataclass.\n
        Allowing multiple parents with __slots__ → new class contains all slots.

        Currently python IDE does not support multiple types 
        - for IDE support direct inheritance is needed
        - to merge multiple dataclasses with slots=True → (update_specs=dict(slots=False))
        """        
        def decorator(target_cls):            
            new_cls = _DataclassSpecs._create_new_cls(target_cls, parents, specs)
            new_cls: type[P1|P2] = _DataclassSpecs._create_dataclass(new_cls, specs)
            return new_cls        
        return decorator

    @staticmethod    
    def is_dataclass(class_or_instance) -> bool:
        """Returns True if obj is a dataclass or an instance of a dataclass."""        
        return is_dataclass(class_or_instance)

    @classmethod
    def set_by_type(cls,
        instance: DictDataclass,
        type_to_fill: type | tuple[type, ...],
        values: list[Any] | None = None
    ) -> None:
        """
        Replace fields of a frozen dataclass that match a given type.

        Scans the default values of `instance` for fields whose type exactly
        matches `type_to_fill` and overwrites them in place using `object.__setattr__`, 
        bypassing the frozen dataclass guard. The object's identity (memory address) 
        is preserved.

        Parameters
        ----------
        instance : DictDataclass
            A frozen dataclass instance to mutate.

        type_to_fill : type | tuple[type, ...]
            The exact type to match against. Only fields whose value
            satisfies `type(value) is type_to_fill` are replaced.

        values : None | list = None
            Optional list of replacement values. Must have the same length
            as the number of matched fields. If `None`, all matched fields
            are set to `None`.

        Note
        ----
        Because mutation bypasses `frozen=True`, any hash computed from the
        field values (e.g. if the object is stored in a `set` or used as a
        `dict` key or used in `functools.lru_cache`) becomes stale after this call.
        """
        if not isinstance(type_to_fill, tuple):
            type_to_fill = (type_to_fill, )

        to_fill = []
        dv = cls.get_fields_type(instance)
        for x, t in dv.items():
            if t in type_to_fill:
                to_fill.append(x)

        if not to_fill:
            return

        if values is not None and not isinstance(values, list):
            raise ValueError(
                'AIG SAFETY ► parameter values must be provided as list with values to fill.'
            )

        if values is None:
            values = [None] * len(to_fill)

        if len(to_fill) != len(values):
            raise ValueError(
                'AIG SAFETY ► Mismatch length of provided values and types to fill ► '
                f'to fill: {len(to_fill)} | values: {len(values)}.'
            )            
        
        for i, x in enumerate(to_fill):
            object.__setattr__(instance, x, values[i])
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class _MappingMeta(type):
    """Metaclass to provide class-level dict-like behavior (including nested dicts)."""

    def _helper_steps(cls) -> str:
        return '► Keys to continue in the structure:\n' + '\n'.join(cls.keys())

    def __repr__(cls):
        return cls._helper_steps()

    def __new__(mcls, name, bases, namespace, **kw):
        cls = super().__new__(mcls, name, bases, namespace)
        cls._values = dict(getattr(cls, "_values", {}))
        forbid_init = kw.get("forbid_init", True)
        frozen = kw.get("frozen", True)
        cls._forbid_init = forbid_init
        cls._frozen = frozen
        return cls

    def __call__(cls, *args, **kwargs):
        if getattr(cls, "_forbid_init", False):
            raise TypeError(f"{cls.__name__} is a static placeholder - dict-like with IDE support - and cannot be instantiated.")
        return super().__call__(*args, **kwargs)
    
    def __setattr__(cls, name, value):
        if not name.startswith("_"):
            if getattr(cls, "_frozen", False):
                raise TypeError(f"{cls.__name__} is frozen — cannot modify attribute '{name}'")
            cls._values[name] = value
        super().__setattr__(name, value)
    
    def __setitem__(cls, key, value):
        if getattr(cls, '_frozen', False):
            raise TypeError(f"Cannot modify frozen DictClass: {cls.__name__}")
        cls._values[key] = value
        setattr(cls, key, value) 

    def __getitem__(cls:T, key: str) -> T | Any:
        if isinstance(key, int):            
            return cls.keys()[key]  # tuple-like access
        return cls._values[key]

    def __len__(cls) -> int:
        return len(cls._values)

    def __iter__(cls) -> Iterator[str]:
        return iter(cls._values)

    def keys(cls) -> Iterable[str]:
        return tuple(cls._values.keys())

    def values(cls) -> Iterable[Any]:
        return tuple(cls._values.values())
    
    def items(cls) -> Iterable[tuple[str, Any]]:
        return cls._values.items()

    def get(cls, key: str, default=None):
        return cls._values.get(key, default)

    def list(cls) -> list[Any]:
        """Returns list of values."""        
        return list(cls.values())

    def key(cls) -> str:   
        """Returns the first key."""
        return next(iter(cls.keys())) 
    
    def value(cls) -> Any: 
        """Returns the first value."""        
        return next(iter(cls.values()))
    
    def item(cls) -> tuple[str, Any]:
        """Returns the first item."""
        return next(iter(cls.items()))
    
    def to_dict(cls) -> dict[str, Any]:
        """Return a shallow-copy `dict repr of the full structure`."""
        def scan(d):
            if ( not inspect.isclass(d) or 
                 not issubclass(d, DictClass) ):
                return d  
            result = {}
            for k, v in d.items():
                result[k] = scan(v)
            return result
        
        return scan(cls)

    def schema(cls:T) -> T:
        """Helper to create dynamic class from the DictClass where param returns its name (key)."""
        p = cls.keys()
        return type(
            f'dynamic_schema_of_{cls.__name__}', (cls,), 
            dict(__cls__ = cls) | dict(zip(p,p))
        )


class DictClass(metaclass=_MappingMeta):
    """Base for class-level mapping behavior. Supports nested DictClass as nested dicts."""

    _values: ClassVar[dict[str, Any]] = {}
    __order__: ClassVar[list[str]] = []

    def __init_subclass__(cls, forbid_init=None, frozen=None, **kwargs):
        super().__init_subclass__(**kwargs)
        
        if forbid_init is None: 
            forbid_init = getattr(cls, "_forbid_init", True)
        if frozen is None: 
            frozen = getattr(cls, "_frozen", True)
        
        cls._forbid_init = forbid_init
        cls._frozen = frozen

        # Start from merged parent values (preserve parent ordering)
        merged: dict[str, Any] = {}
        for base in reversed(cls.__mro__[1:]):  # from top-most base down to immediate parent
            base_vals = getattr(base, "_values", None)
            if isinstance(base_vals, dict):
                # make shallow copy to avoid shared references
                for k, v in base_vals.items():
                    merged[k] = (v.copy() if isinstance(v, dict) else v)

        # Collect this class's attributes in definition order
        for name, attr in cls.__dict__.items():
            if name.startswith("_"):
                continue

            # If the attribute is a nested DictClass subclass -> use its _values as nested dict
            if isinstance(attr, type) and issubclass(attr, DictClass):
                merged[name] = getattr(attr, "_values", {}).copy()
                continue

            # Skip callables (methods, functions) and descriptors
            if callable(attr):
                continue

            # Regular class attribute -> include/override
            merged[name] = attr

        # Apply optional explicit ordering (__order__) if provided
        order = getattr(cls, "__order__", None) or []
        if order:
            ordered = {}
            for k in order:
                if k in merged:
                    ordered[k] = merged[k]
            # append any remaining keys in their insertion/collection order
            for k, v in merged.items():
                if k not in ordered:
                    ordered[k] = v
            merged = ordered
        
        cls._values = MappingProxyType(merged) if frozen else merged
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬