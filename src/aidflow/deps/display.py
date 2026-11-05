"""
The Dependency Abstraction Layer (Dependency Firewall).

This module serves as the sole authorized gateway for all external dependencies 
and Python standard library imports across the entire AIDflow ecosystem. 
Following a strict architectural mandate, no module within the AIDflow package 
is permitted to import third-party libraries or native Python modules directly; 
instead, they must consume these dependencies via the `.deps` abstraction layer.
"""

import ipywidgets as wg
from tqdm import tqdm

from tqdm.notebook import (
    tqdm_notebook
)

from IPython.display import (
    HTML, 
    SVG,
    display, 
    clear_output, 
    display_html
)

from IPython import  (
    get_ipython, 
    InteractiveShell
)

from ipykernel.ipkernel import (
    IPythonKernel
)

from ipykernel import (
    get_connection_file
)

from jupyter_core.paths import (
    jupyter_runtime_dir
)