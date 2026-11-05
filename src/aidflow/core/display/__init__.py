"""
This module implements the standardized reporting layer used to communicate 
model performance, feature importance, and pipeline stability. It transforms 
raw execution results into high-fidelity, interactive dashboards through a 
decoupled architecture that separates data computation from visual rendering.
"""

from .dash import (
    DashInterface, 
    DashReportLog, 
    DashMixin, 
    DashPlotlyLink, 
    DashBackground
)
from .panel import PanelConstructor

from .display import Display
from .confirm import Confirm