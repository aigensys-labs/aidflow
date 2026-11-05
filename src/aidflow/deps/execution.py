"""
The Dependency Abstraction Layer (Dependency Firewall).

This module serves as the sole authorized gateway for all external dependencies 
and Python standard library imports across the entire AIDflow ecosystem. 
Following a strict architectural mandate, no module within the AIDflow package 
is permitted to import third-party libraries or native Python modules directly; 
instead, they must consume these dependencies via the `.deps` abstraction layer.
"""

class Ray:
    def __getattr__(self, name):
        return self._any
    def _any(self, *args, **kwargs):
        return
    
try:
    import ray
except:
    ray = Ray()
    
from multiprocess import (
    Pool, 
    RLock, 
    set_start_method,
    parent_process
)

from multiprocessing import (
    parent_process as _parent_process    
)

set_start_method("spawn", force=True)

def is_worker() -> bool:
    """Returns True when worker is in multiprocess."""
    if _parent_process() is not None:
        return True
    return parent_process() is not None