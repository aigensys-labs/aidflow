"""
The Dependency Abstraction Layer (Dependency Firewall).

This module serves as the sole authorized gateway for all external dependencies 
and Python standard library imports across the entire AIDflow ecosystem. 
Following a strict architectural mandate, no module within the AIDflow package 
is permitted to import third-party libraries or native Python modules directly; 
instead, they must consume these dependencies via the `.deps` abstraction layer.
"""

import plotly.graph_objects  as go
import plotly.io             as pio
from   plotly.subplots       import make_subplots
from   plotly.colors         import qualitative

import matplotlib.pyplot     as plt
from   matplotlib.figure     import Figure  as plt_Figure
from   matplotlib.axes       import Axes    as plt_Axes