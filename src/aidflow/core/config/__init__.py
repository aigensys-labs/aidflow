"""
This module implements the primary user interface layer, transforming 
abstract parameter schemas defined in `core.parameters` into high-fidelity, 
interactive control panels within Jupyter environments. It serves as the 
bridge between human intuition and machine execution, allowing users to 
configure complex ML pipelines through a curated visual experience.
"""

from .css import THEME
from .config import (
    ProjectGenesis, 
    PipelineConfigurator, 
    ConfigCss
)