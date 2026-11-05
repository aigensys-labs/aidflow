"""
The Dependency Abstraction Layer (Dependency Firewall).

This module serves as the sole authorized gateway for all external dependencies 
and Python standard library imports across the entire AIDflow ecosystem. 
Following a strict architectural mandate, no module within the AIDflow package 
is permitted to import third-party libraries or native Python modules directly; 
instead, they must consume these dependencies via the `.deps` abstraction layer.
"""

from types import  (
    MethodType, 
    FunctionType,
    GeneratorType,
    UnionType,
    SimpleNamespace, 
    MappingProxyType
)

from typing import  (
    Type,
    Protocol,
    Any, 
    Literal, 
    Optional,
    ClassVar, 
    Iterator, 
    Iterable,
    Union,
    Generic,
    TypeVar,
    Callable,
    TYPE_CHECKING,
    cast,
    get_args, 
    get_origin,
    get_type_hints,
    overload,
    dataclass_transform
)

from numpy.typing import  (
    ArrayLike, 
    NDArray
)

from .weakref import WeakGetter


def extract_types(type_hint: Any) -> list:
    """
    Takes any type hint (single type or Union)
    and returns a flat list of raw classes.
    """
    origin = get_origin(type_hint)
    
    if origin in (Union, UnionType):
        return list(get_args(type_hint))
    
    return [type_hint]


FuncType = FunctionType | MethodType | WeakGetter.Attr | WeakGetter.PairAttr