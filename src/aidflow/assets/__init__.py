"""
AIDflow Bundled Assets.

Provides access to static files shipped within the package —
the guide notebook, shell scripts, and reference materials.
These assets are versioned with the package and available
offline (no network required).

Exposes
-------
AssetGetter : class
    Accessor for copying or locating bundled assets
    to a user-specified target path.
"""

from .ui import UIGetter, UIPaths
from .datasets import DataGetter