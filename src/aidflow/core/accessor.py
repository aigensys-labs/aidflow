"""
Global interface proxy for system configurations, styling, and execution 
utilities.

This module provides centralized access points:
- UI
- Rgb
- Format
- Spinner
- Parallel

to the  system's configuration manager using a lazy-loading proxy pattern. 
This ensures that configuration is resolved at runtime while maintaining 
strict type hinting for development and IDE support.
"""

from ..deps.types       import  (
                                Callable,
                                TYPE_CHECKING
                                )
from ..display.format   import  FormatHTML
from ..display.rgb      import  RgbHTML
from ..display.spinner  import  SpinnerHTML

from .execution         import  (
                                Config,
                                config_manager,
                                ParallelBase
                                )

class _Proxy:
    def __init__(self, getter: Callable[[], type]):
        self._getter = getter

    def __call__(self, *args, **kwargs):
        return self._getter()(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self._getter(), name)


UI       = _Proxy(lambda: config_manager.default)
Rgb      = _Proxy(lambda: config_manager.rgb)
Format   = _Proxy(lambda: config_manager.formatter)
Spinner  = _Proxy(lambda: config_manager.spinner)
Parallel = _Proxy(lambda: config_manager.parallel)


if TYPE_CHECKING:
    UI = Config
    Rgb = RgbHTML
    Format = FormatHTML
    Parallel = ParallelBase
    Spinner  = SpinnerHTML