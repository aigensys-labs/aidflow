"""
Low-level infrastructure Arrow/pandas/dict ops providing high-performance 
data bridges and structural manipulation tools.

It serves as the primary engine for optimizing memory usage and ensuring 
data integrity during the hand-off between different execution layers.

Facilitates the use of PyArrow Tables for large-scale dataset handling, reducing 
memory overhead during high-volume data transformations.
"""

from .arrow         import Arrow
from .pandas        import Pd, GetValue, DataFrame
from .standard      import DataStandard, TypeStandard
from .merger        import DictMerger
from .containers    import Dict