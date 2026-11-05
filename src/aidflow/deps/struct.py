"""
The Dependency Abstraction Layer (Dependency Firewall).

This module serves as the sole authorized gateway for all external dependencies 
and Python standard library imports across the entire AIDflow ecosystem. 
Following a strict architectural mandate, no module within the AIDflow package 
is permitted to import third-party libraries or native Python modules directly; 
instead, they must consume these dependencies via the `.deps` abstraction layer.
"""

from dataclasses import  (
    dataclass, 
    is_dataclass,
    make_dataclass,
    fields, 
    field, 
    asdict,
    replace,
    MISSING,
)

from enum import  (
    Enum, 
    IntEnum, 
    unique
)

from collections import (
    namedtuple
)

from collections.abc import (
    Mapping
)