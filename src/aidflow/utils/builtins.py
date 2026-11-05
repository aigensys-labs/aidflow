"""
Atomic Utility Layer.

This module provides low-level, deterministic manipulation tools for Python's 
native collections and types. By centralizing these operations, AIDflow ensures 
that basic data transformations—such as list merging, string alignment, and 
dynamic class injection—are performed consistently across the entire framework.

These utilities are designed to eliminate common Python runtime errors and 
ensure that the high-level Worker and Model layers operate on a stable, 
predictable foundation.
"""

from ..deps.common import  (
    re, 
    uuid,
    datetime, 
)
from ..deps.data import  (
    pd
)
from ..deps.types import  (
    Any, 
    Callable,
    Literal, 
    MethodType,
    FuncType
)

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# BUILTINS TOOLS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class List:
    """
    Deterministic List & Iterable Orchestration.
    
    Provides a suite of tools to ensure that arguments are always handled as 
    iterables, preventing common 'TypeError' crashes when switching between 
    single-variable and multi-variable inputs.
    """

    ITERABLES = (list, tuple, set, dict)

    @staticmethod
    def yields(*lists):
        """Generator that flattens multiple iterable inputs into a single stream."""
        for lst in lists:
            yield from lst        

    @classmethod
    def merge(cls, *lists) -> list:        
        """
        Merge multiple iterables into a single, consolidated list.
        
        Ensures a deterministic combination of feature lists or parameter sets 
        without duplicating the underlying memory pointers.
        """
        return list(cls.yields(*lists))
    
    @classmethod
    def is_iterable(cls, arg:Any):
        """Check if an object is a compatible AIDflow iterable (list, tuple, set, dict)."""        
        return isinstance(arg, cls.ITERABLES)

    @classmethod
    def ensure(cls, *args) -> list|list[list]:
        """
        Ensures that all arguments are interables (list, tuple, set, dict).
        
        If a single string or integer is passed, it is automatically wrapped 
        in a list, ensuring that downstream loops never fail.
        """
        is_iterable = lambda a: cls.is_iterable(a)
        _iters = [a if a is None or is_iterable(a) else [a] for a in args]
        if len(_iters) == 1:
            return _iters[0]
        return _iters

    @staticmethod
    def subset(target:list, *args) -> list:
        """
        Deterministic Feature Subset Extraction.
        
        Returns the intersection of the provided lists against a target list. 
        Crucial for ensuring that only requested features are passed to a 
        specific model fitter while maintaining the original order.
        """
        res = []
        for a in args:
            res.append([x for x in a if x in target])
        return res[0] if len(args) == 1 else res
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Str:
    """
    Deterministic String & Text Manipulation.
    
    Provides specialized string utilities designed for ML metadata handling, 
    including interval parsing for binning and alignment tools for 
    generating human-readable audit logs.
    """
    
    @staticmethod
    def get_interval_numbers(x:str) -> list:
        """
        Extract Numeric Bounds from Interval Strings.
        
        Parses strings representing numeric intervals (e.g., "[10, 20]") 
        and returns a clean list of floats. Essential for automated 
        binning and discretization logic.
        """
        pattern = r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?|inf|-inf)"
        matches = re.findall(pattern, x)
        return [match.replace(",", "") for match in matches]

    @staticmethod
    def get_strftime_length(format_string:str) -> int:
        """Calculate the resulting character length of a specific datetime format."""
        dummy = datetime(2000, 1, 1)
        formatted_string = dummy.strftime(format_string)
        return len(formatted_string)

    @staticmethod
    def add_prefix(x:str, p:str='_') -> str:
        """Ensures string has a prefix (default '_')."""
        return p + x if x[0] != p else x

    @staticmethod
    def join(*args, s='_') -> str:
        """Joins args or input string by chosen factor (default '_')."""
        if len(args) == 1 and isinstance(args[0], str):
            args = args[0].split(' ')
        args = [a for a in args if a]    
        return s.join(args)

    @staticmethod
    def align(s0:str|int, s1:str|int, with_sx=None) -> str:
        """Returns blank space aligning 2 strings in the input."""
        a  = lambda s: len(str(s)) if not isinstance(s, int) else s
        l0 = a(s0)
        l1 = a(s1)
        if with_sx is not None:
            if len(str(with_sx)) >= max([l0,l1]):
                return ''
        return ' '*(l0-l1)
    
    @staticmethod
    def fx_align(list_of_str:list[str]) -> FuncType:
        """Returns a function aligning strings in the input list."""
        ml = max([len(x) for x in list_of_str])
        return lambda x: ' '*( ml-len(x) )

    @staticmethod
    def max_len(x:str, max_len:int=20, reduced_len:int=20, p:str='...') -> str:
        """
        Returns a string with respect to `max_len`. 
        When max_len is exceeded, *x[:reduced_len] + p*.
        """
        if len(x) <= max_len:
            return x
        return x[:reduced_len] + p
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Cls:
    """
    Dynamic Class & Attribute Orchestration.
    
    Implements a 'Mix-in' pattern to allow the dynamic injection of attributes 
    and methods into objects at runtime. This is the engine behind AIDflow's 
    ability to wrap Fitters into Models seamlessly.
    """

    @staticmethod
    def add_attribute(
        instance: object, attr_name: str, value: Any, over=False
    ) -> None:
        """
        Sets the named attribute on the given object to the specified value.
        setattr(x, 'y', v) is equivalent to x.y = v
        """
        if not hasattr(instance, attr_name) or over:
            setattr(instance, attr_name, value)
        else:
            print(f"Attribute '{attr_name}' already exists.")
    
    @staticmethod
    def add_method(
        instance: object, method_name: str, method: Callable, over: bool=False
    ) -> None:
        """
        Sets the method on the given object to the specified value.
        setattr(x, 'y', v) is equivalent to x.y = v
        """        
        if not hasattr(instance, method_name) or over:
            setattr(instance, method_name, MethodType(method, instance))
        else:
            print(f"Method '{method_name}' already exists.")

    @classmethod
    def merge_classes(cls, 
        self_to: type, self_from: type, over: bool=False, include_methods: bool=False
    ) -> None:
        """
        Merges 2 instances/classes → self_from is injected to self_to.
        
        Performs a deep merge of properties from one class/instance into another. 
        This is the core mechanism used to 'gear' different Fitters into the 
        standardized Model interface.
        """
        for key, value in self_from.__dict__.items():
            if callable(value) and include_methods:
                cls.add_method(self_to, key, value, over)
            else:
                cls.add_attribute(self_to, key, value, over)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Misc:
    """
    Standardized System & Temporal Utilities.
    
    Provides deterministic wrappers for UUID generation and timestamping 
    to ensure that session IDs and execution logs are consistent across 
    distributed environments.
    """

    @staticmethod
    def uuid(length:int=None) -> str:
        """Returns a uuid4 hex string of 32 chars. 
        Optionally, length slice can be applied.
        """
        u = uuid.uuid4().hex
        if not length:
            return u
        return u[:length]
    
    @staticmethod
    def timestamp(style: str | Literal['sec', 'iso', 'human'] = 'sec') -> str|float:
        """
        Returns a formatted timestamp string or float.
        
        Args:
            style : The format to return.
                - 'sec': Unix timestamp (float) (e.g. 1783351894.124526).
                - 'iso': ISO 8601 format string (e.g. '2023-10-27T14:30:00.123456').
                - 'human': Human-readable string (e.g. 'October 27, 2023, 2:30 PM').
                
        Returns:
            float if style is 'sec'.
            str if style is 'iso' or 'human'.
        """
        now = datetime.now()
                
        if style == 'sec'   : return now.timestamp()
        if style == 'iso'   : return now.isoformat()        
        if style == 'human' : return now.strftime('%B %d, %Y, %I:%M:%S %p')
        return now.strftime(style)
    
    @staticmethod
    def time_diff(
        t0: str|float|int, 
        t1: str|float|int, 
        format: str = '%B %d, %Y, %I:%M:%S %p', 
        **kwargs
    ) -> pd.Timedelta:
        """Delta between two timestamps."""
        kw = dict(format=format) | kwargs
        return pd.to_datetime(t1, **kw) - pd.to_datetime(t0, **kw) 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬