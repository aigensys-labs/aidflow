"""
Global Semantic Standards and Type-Normalization Layer.

This module defines the universal 'language' of the AIDflow ecosystem, 
establishing the naming conventions, type definitions, and structural 
standards used by all components to communicate state and identity. 
By centralizing these constants into a single standard, it eliminates 
semantic ambiguity across different pipeline phases.

Key operational roles:
1. Namespace Standardization: Implements `VarStandards` to enforce consistent 
   naming patterns for special feature types (e.g., WOE, Interactions, and 
   Groups), enabling automated feature discovery and lineage tracking.
   
2. Cross-Library Type Mapping: Provides a unified `TypeDefinitions` registry 
   that abstracts the differences between NumPy, Pandas, and Python native 
   types into stable, internal categories (e.g., 'numeric', 'datetime').

3. Platform-Stable Scalar Verification: Implements the `Scalar` utility to 
   provide a consistent verdict on data atomicity across different OS 
   architectures and library versions, ensuring stability during serialization.
   
4. Feature Provenance: Enables "reverse-lookup" of original input variables 
   from engineered features via semantic parsing of standardized suffixes.
"""

from ..deps.data    import np
from ..deps.struct  import dataclass
from ..deps.types   import Any, Iterable
from .factory       import Dataclass, DictClass


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# AIG PIPELINE STANDARDS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@dataclass(frozen=True, slots=True)
class VarType(DictClass):
    N = 'N'
    C = 'C'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    

@Dataclass.base
class VarStandards:
    """Holds Variable Naming Standards of the AIG flow."""

    MISSING:  str = "Missing"
    NA_TRAIN: str = "Not_in_Train"
    CAT_DIV:  str = " <|> "
    IXN:      str = 'ixn'
    WOE:      str = "woe"    
    T:        str = "_t"
    G:        str = "__g"   # __ signal → G is exposed to user across all transformed fes
    W:        str = "_w"
    ixn_len:  int = 4

    @property
    def WW(self) -> str: 
        return f"_{self.WOE}"
        
    def fe_t(self, x: str)   -> str: return f"{x}{self.T}"
    def fe_g(self, x: str)   -> str: return f"{x}{self.G}"
    def fe_w(self, x: str)   -> str: return f"{x}{self.W}"
    def fe_woe(self, x: str) -> str: return f"{x}{self.WW}"
    def fe_ixn(self, i: int) -> str: return f"{self.IXN}_{i:0{self.ixn_len}d}"

    def _fe_to_var(self, v:str) -> str:
        for suffix in [self.WW, self.W, self.G]:
            if v.endswith(suffix):
                v = v[:-len(suffix)]
        return v

    def _var_to_woe(self, v:str) -> str:
        fe = v
        for suffix in [self.G, self.WW]:
            fe += suffix
        return fe
    
    def fe_to_var(self, fe: str | list[str]) -> str | list:
        """
        Provides original input variable name from created FE in any stage of AIG flow.\n
        Holds only during AIG flow steps - applied on raw features might provide incorrect name.
        """
        if isinstance(fe, str):
            return self._fe_to_var(fe)
        fes = []
        for v in fe:            
            fes.append( self._fe_to_var(v) )
        return fes

    def var_to_woe(self, v: str | list[str]) -> str | list:
        """
        Provides WOE feature name from original feature.
        """
        if isinstance(v, str):
            return self._var_to_woe(v)
        fes = []
        for x in v:
            fes.append( self._var_to_woe(x) )
        return fes
    
    def _is_ixn_fe(self, fe: str) -> bool:
        """
        Returns True if fe is IXN kind; otherwise False.\n
        Verdict holds only during AIG flow steps - applied on raw features provides false positives.
        """
        ixn = fe.split('_')
        l = len(ixn)        
        if l not in (2, 3):
            return False
        v, i, *_ = ixn
        is_ixn = ( v == VarStd.IXN ) and ( len(i) == self.ixn_len and i.isdigit() )
        if l == 2:
            return is_ixn
        return is_ixn & fe.endswith(self.WW)
        
    def is_woe_fe(self, fe: str) -> bool:
        """
        Returns True if fe is WOE kind; otherwise False.\n
        Verdict holds only during AIG flow steps - applied on raw features provides false positives.
        """
        if self._is_ixn_fe(fe):
            s = self.WW
        else: 
            s = ''.join([self.G, self.WW])
        return fe.endswith(s)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


_BOOL  = (
    'bool', 'boolean',
    'bool[pyarrow]', 'boolean[pyarrow]'
)

_CAT   = (
    'category',
)

_OBJ   = (
    'object', 'str', 'string',
    'string[python]', 'string[pyarrow]', 
    'large_string[pyarrow]', 'string_view[pyarrow]'
)

_F64   = (
    'Float64', 'float64[pyarrow]', 'double[pyarrow]'
)
_F32   = (
    'Float32', 'float32[pyarrow]', 'float[pyarrow]'
)
_F16   = (
    'Float16', 'float16[pyarrow]', 'halffloat[pyarrow]'
)
_INT   = (
    'Int8',     'Int16',    'Int32',    'Int64',
    'UInt8',    'UInt16',   'UInt32',   'UInt64',

    'int8[pyarrow]',    'int16[pyarrow]',   'int32[pyarrow]',   'int64[pyarrow]',
    'uint8[pyarrow]',   'uint16[pyarrow]',  'uint32[pyarrow]',  'uint64[pyarrow]',
)
_NUM   = (
    *_INT, *_F16, *_F32, *_F64
)
_DATE  = (
    'Datetime64', 'Datetime64[ns]', 'Datetime64[s]', 'Datetime64[ms]', 'Datetime64[us]',

    'date32[day][pyarrow]',   'date64[ms][pyarrow]',
    'timestamp[s][pyarrow]',  'timestamp[ms][pyarrow]', 
    'timestamp[us][pyarrow]', 'timestamp[ns][pyarrow]'    
), ('<M8','<M8[ns]') 

_TIME  = (
    'Timedelta64', 'Timedelta64[ns]',

    'time32[s][pyarrow]',    'time32[ms][pyarrow]', 
    'time64[us][pyarrow]',   'time64[ns][pyarrow]',
    'duration[s][pyarrow]',  'duration[ms][pyarrow]', 
    'duration[us][pyarrow]', 'duration[ns][pyarrow]'    
), ('<m8','<m8[ns]') 

_NONE  = (
    '', 'NaN', 'NaT', 'N/A', 'NULL', 'None'
)


def _variation(definition:tuple, extra:tuple=()) -> tuple:
    definitions = ( (x, x.lower()) for x in definition )
    return tuple(dict.fromkeys( tuple([x for d in definitions for x in d]) + extra))

@Dataclass.base
class TypeDefinitions:
    """Holds Variable Data Type Standards."""
    bool:      tuple[str, ...] = _BOOL
    category:  tuple[str, ...] = _CAT
    object:    tuple[str, ...] = _OBJ
    float64:   tuple[str, ...] = _variation(_F64)
    float32:   tuple[str, ...] = _variation(_F32)
    float16:   tuple[str, ...] = _variation(_F16)
    integer:   tuple[str, ...] = _variation(_INT)
    numeric:   tuple[str, ...] = _variation(_NUM)
    datetime:  tuple[str, ...] = _variation(*_DATE)
    timedelta: tuple[str, ...] = _variation(*_TIME)
    missing:   tuple[str, ...] = _NONE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


VarStd  = VarStandards()
TypeDef = TypeDefinitions()
scalar_types  = [x for T in TypeDef.values() for x in T]
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class UniType:
    """Type Helper to provide all compatible types across arrow / native dtype."""

    _SCHEMA = Dataclass.schema(TypeDefinitions)
    
    @classmethod
    def get_exact(cls, dtype: str) -> tuple:
        """Returns compatible `category type` of provided dtype."""        
        return cls._get(dtype, (cls._SCHEMA.numeric,))

    @classmethod
    def get_group(cls, dtype: str) -> tuple:
        """Returns compatible `category group type` of provided dtype."""        
        e = (
            cls._SCHEMA.float64, 
            cls._SCHEMA.float32, 
            cls._SCHEMA.float16,
            cls._SCHEMA.integer
        )
        return cls._get(dtype, e)

    @classmethod
    def _get(cls, dtype: str, excluded:tuple) -> tuple:
        """Returns compatible `category group type` of provided dtype."""        
        for name, types in TypeDef.items():
            if name in excluded:
                continue
            if dtype in types:
                return types
        raise ValueError(
            f'AIG SAFETY ► Unkown dtype provided. Supported types:\n{TypeDef}'
        )    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Scalar:
    """Scalar Helper to provide stable verdict across platforms."""

    mapping = {
        'int':       'int64',
        'float':     'float64',
        'str':       'str',
        'bool':      'bool',
        'datetime':  'datetime64[ns]',
        'timedelta': 'timedelta64[ns]'
    }

    @classmethod
    def get_scalar_type_name(cls, x:Any) -> str:
        """
        Normalizes any scalar type into a clean string identifier 
        compatible with cross-platform list.
        """
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 'None' if x is None else 'nan'
        
        if hasattr(x, 'dtype'):
            return x.dtype.name 
        
        name = x.__class__.__name__
        return cls.mapping.get(name, name)
        
    @classmethod
    def is_scalar(cls, x:Any) -> bool:
        """
        Stable verification whether value is a scalar.
        Verdict is stable across packages/libs and platforms.
        """
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            return False        
        return cls.get_scalar_type_name(x) in scalar_types
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬