"""
Bundled assets accessor.

Provides a unified interface to locate and copy static files
(notebooks, scripts, reference documents) that are shipped
within the AIDflow package. Assets are co-located with this
module and versioned with the package — no network access
required.

Usage
-----
**Copy assets to the active dir:**

>>> from aidflow import UIGetter
>>> UIGetter.get_all()
"""

from ..deps.common import Path, shutil
from ..storage.folder import Folder


_DIR = Path(__file__).parent / 'files'

class UIPaths:
    NTB = 'notebook'
    CLI = 'console'

_PATHS = {
    UIPaths.NTB: 'aidflow.ipynb',
    UIPaths.CLI: 'console.py'
}


class UIGetter:
    """
    Bundled assets accessor.

    Locates and copies static files (notebooks, scripts)
    shipped within the AIDflow package to a user-specified
    target directory.
    """

    @classmethod
    def get_all(cls, target: str | Path = '') -> dict[str, Path]:
        """
        Copy all bundled assets to the target directory.

        Parameters
        ----------
        target : str | Path
            Target directory. Created if it does not exist.

        Returns
        -------
        dict[str, Path]
            Mapping of asset key → destination file path.
        """
        path = {}
        for k in _PATHS:
            path[k] = cls._copy(k, target)
        return path
    
    @classmethod
    def get_notebook(cls, target: str | Path = '') -> Path:
        """
        Copy the bundled guide notebook to the target path.

        Parameters
        ----------
        target : str | Path
            Destination folder path (e.g., `'./project'`).

        Returns
        -------
        Path
            The written file path.
        """
        return cls._copy(UIPaths.NTB, target)

    @classmethod
    def get_script(cls, target: str | Path = '') -> Path:
        """
        Copy the bundled shell script to the target path.

        Parameters
        ----------
        target : str | Path
            Destination folder path (e.g., `'./project'`).

        Returns
        -------
        Path
            The written file path.
        """
        return cls._copy(UIPaths.CLI, target)
    
    @classmethod
    def get_paths(cls) -> dict[str, Path]:
        """
        Return in-package paths for all registered assets (read-only).

        Returns
        -------
        dict[str, Path]
            Mapping of asset key → absolute path within the package.
        """
        return {k: _DIR / p for k,p in _PATHS.items()}

    @classmethod
    def _copy(cls, key:str, target: str | Path) -> Path:
        """Copy a registered asset to the target path."""
        path = Folder.create(target)
        path = path / _PATHS[key]
        shutil.copy(_DIR / _PATHS[key], path)
        return path