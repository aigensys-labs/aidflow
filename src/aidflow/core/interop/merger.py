"""
Dictionary Merger — Recursive Deep Merge of Configuration and Result Sets.

This module provides utilities for merging nested dictionary structures
while preserving type integrity. It handles:

- **Scalar values**: Last-write-wins (org overwrites new).
- **Nested dicts**: Recursively merged key-by-key.
- **Lists/Tuples**: Concatenated (order preserved).
- **Sets**: Union.
- **DataFrames/Series**: Concatenated along the specified axis.
- **DictDataclass**: Merged and re-instantiated with the original class.

This is the underlying merge engine used by the framework to
combine incremental result sets (e.g., per-agent diagnostics)
into unified output structures.
"""

from ...deps.types      import Any
from ...deps.data       import pDF, pSe

from ..factory          import DictDataclass
from ..interop          import Pd


class DictMerger:
    """
    Recursive deep-merge engine for nested dict structures.

    Merges two dictionaries (or a list of dictionaries) by
    traversing nested structures and combining values by type.
    Type safety is enforced — merging incompatible types raises
    a `TypeError` with a diagnostic message.
    """

    @classmethod
    def merge(cls, d_new: dict, d_org: dict, pd_axis:int=0) -> dict:
        """
        Merge `d_org` into `d_new` (org values take priority).

        Traverses both dicts recursively:
        - Keys only in `d_new`: preserved as-is.
        - Keys only in `d_org`: added (initialized by type).
        - Keys in both: merged by type (dict→recursive,
          list→concat, df→concat, etc.).

        Parameters
        ----------
        d_new : dict
            Base dictionary (lower priority).
        d_org : dict
            Source dictionary (higher priority — overwrites
            conflicts).
        pd_axis : int, default=0
            Concatenation axis for DataFrame merges.

        Returns
        -------
        dict
            The merged dictionary (`d_new` mutated in place).
        """ 
        for k, v in d_org.items():        
            if k not in d_new:
                d_new[k] = cls.init_value_by_type(v)
            if isinstance(v, dict):
                d_new[k] = cls.merge(d_new[k], v)
            elif cls.is_mergable_data_type(v):
                d_new[k] = cls._merge_by_type(d_new[k], v, pd_axis=pd_axis)
        return d_new

    @classmethod
    def merge_list(cls, 
        list_of_dict: list[dict | DictDataclass] | tuple[dict | DictDataclass], 
        pd_axis:int=0
    ) -> dict | DictDataclass:   
        """
        Merge a sequence of dictionaries into a single unified
        structure.

        If all elements are `DictDataclass` instances of the
        same type, the result is re-instantiated as that class.
        Otherwise, the result is a plain dict.

        Parameters
        ----------
        list_of_dict : list | tuple
            Sequence of dicts or DictDataclass instances to merge.
            All elements must be of the same type (dict or a
            single DictDataclass class).
        pd_axis : int, default=0
            Concatenation axis for DataFrame merges.

        Returns
        -------
        dict | DictDataclass
            The merged structure.

        Raises
        ------
        TypeError
            If elements are mixed types (e.g., dict and
            DictDataclass).
        """        
        d_cls: DictDataclass | None = None
        if all([isinstance(d, DictDataclass) for d in list_of_dict]):
            d_cls: DictDataclass = type(list_of_dict[0])
            list_of_dict = [dict(d) for d in list_of_dict]
            # NOTE 
            #  → dict(d)     → turns only the top level class (d_cls) to dict
            #  → d.to_dict() → turns all sub-dictdaclasses to dict (!)

        if not all([isinstance(d, dict) for d in list_of_dict]):
            type_ = [type(d) for d in list_of_dict]
            raise TypeError(f'\nAIG SAFETY\
        \n ► DICTIONARY MERGE FROM LIST ELEMENTS IS POSSIBLE ONLY WHEN ALL LIST ELEMENTS BEING DICTIONARY\
        \n ► TYPES OF ELEMENTS IN LIST: {type_}')
        
        new = {}    
        for d in list_of_dict:
            cls.merge(new, d, pd_axis)
        
        if d_cls is None:
            return new
        return d_cls(**new)
    
    @classmethod
    def _merge_by_type(cls, *args, pd_axis:int=0) -> Any:
        """
        Merge two or more values of the same type using type-
        appropriate concatenation.

        Type behaviors:
        - **list/tuple**: Flatten (concatenate all elements).
        - **dict/DictDataclass**: Deep merge.
        - **set**: Union.
        - **pDF**: `concat(axis=pd_axis)`.
        - **pSe**: `concat(axis=0)`.
        - **Other**: Return as tuple (no merge).

        Parameters
        ----------
        *args
            Two or more values of the same type to merge.
        pd_axis : int, default=0
            Axis for DataFrame concatenation.

        Returns
        -------
        Any
            The merged value.

        Raises
        ------
        TypeError
            If the arguments are not all the same type.
        """  
        t = args[0]
        c = [type(x) for x in args if type(x) != type(t)]
        if c:
            raise TypeError(
                'AIG SAFETY ► Merging is possible only for same types\n'
                f'► Org: {type(t)} | Mismatch: {c}'
            )
        if isinstance(t, (list, tuple)):
            merged = [r for res in args for r in res]
        elif isinstance(t, (dict, DictDataclass)):
            merged = cls.merge_list(args, pd_axis)
        elif isinstance(t, set):
            merged = set().union(*args)
        elif isinstance(t, pDF):
            merged = Pd.concat(args, axis=pd_axis)
        elif isinstance(t, pSe):
            merged = Pd.concat(args, axis=0)
        else: 
            merged = args
        return merged   

    @staticmethod
    def mergable_data_types() -> tuple:
        """
        Returns the tuple of data types supported for merging:
        `(list, tuple, set, dict, pDF, pSe)`.

        Returns
        -------
        tuple
            Supported mergable type identifiers.
        """
        return (list, tuple, set, dict, pDF, pSe)
    
    @classmethod
    def is_mergable_data_type(cls, value:Any) -> bool:
        """
        Check whether a value is of a type supported for merging.

        Parameters
        ----------
        value : Any
            The value to check.

        Returns
        -------
        bool
            True if the value's type is in `mergable_data_types()`.
        """
        return isinstance(value, cls.mergable_data_types())
    
    @classmethod
    def init_value_by_type(cls, value:Any):
        """
        Create an empty instance of the same type as the input
        value (for initializing merge targets).

        For mergable types (list, tuple, set, dict, pDF, pSe):
        returns an empty container of that type.
        For scalar types (int, float, str): returns the value
        as-is.

        Parameters
        ----------
        value : Any
            The value whose type to initialize.

        Returns
        -------
        Any
            An empty container of the same type, or the value
            itself for scalars.
        """
        if cls.is_mergable_data_type(value):
            return type(value)()  
        return value    