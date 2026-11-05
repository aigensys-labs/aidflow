"""
Global Communication and Log-Level Orchestration Engine.

This module defines the standardized verbosity protocol for all AIDflow 
components. It provides a centralized mechanism to control the volume and 
format of system feedback, ensuring that the framework can adapt its output 
to any execution environment—from highly interactive Jupyter Notebooks to 
silent headless production servers.
"""

from ..deps.struct import IntEnum, unique

@unique
class Verbosity(IntEnum):
    """
    Standardized enumeration for AIDflow communication levels.
    
    This class defines the three primary states of system feedback:
    - QUIET (0): Absolute silence; used for production execution and 
      minimalist logging.
    - INFO (1): Standard operational messages; provides essential tracking 
      of pipeline progress via console output.
    - GUI (2): Full interactive mode; enables all rich visual elements, 
      HTML tables, and animation frames.
    """

    QUIET = 0
    INFO  = 1
    GUI   = 2

    @classmethod
    def normalize(cls, value) -> "Verbosity":
        """
        Convert any user input to a valid Verbosity enum.

        Accepts:
        - raw int
        - string names ("quiet", "VERBOSE", etc.)
        - invalid → fallback to default
        """
        if isinstance(value, cls):
            return value

        try:
            if isinstance(value, str):
                return cls[value.strip().upper()]
            if isinstance(value, bool):
                value = max(Verbosity) if value else min(Verbosity)
            return cls(int(value))
        except Exception:            
            return cls.GUI