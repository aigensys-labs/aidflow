"""
Global Environment Orchestrator and Runtime Configuration Manager with
High-performance compute orchestration layer for distributed data processing.
"""

from .manager import (
    UI, 
    THEME, 
    PARALLEL, 
    Config, 
    set_backend, 
    config_manager
)
from .parallel import (
    KW,
    Parallel, 
    ParallelBase
)
from .process import (
    BackgroundProcess
)