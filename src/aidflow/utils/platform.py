"""
Environment adaptation layer providing cross-platform compatibility and 
runtime context detection.

This module implements the critical infrastructure required to ensure that 
AIDflow remains environment-agnostic, allowing seamless execution across 
disparate deployment targets—from local Windows workstations and Linux 
Docker containers to managed JupyterHub clusters.
"""

from ..deps.common import (
    os, 
    sys, 
    psutil,
    pathlib
)
from ..deps.execution import (
    ray, 
    Ray,
    is_worker
)
from ..deps.display import (
    get_ipython
)


class Platform:

    @staticmethod
    def is_ray_active() -> bool:
        """Returns True when env. has installed ray library."""
        return not isinstance(ray, Ray)
    
    @staticmethod
    def is_main() -> bool:
        """Returns True under main process (no sub-multiprocess)."""
        return not is_worker()
        
    @staticmethod
    def is_in_jupyter() -> bool:
        """
        Detects if the code is running in a Jupyter Notebook or JupyterLab.
        Returns True for Jupyter environments, False otherwise.
        """
        try:
            shell = get_ipython().__class__.__name__
            if shell in ['ZMQInteractiveShell']:
                return True  # Jupyter Notebook or JupyterHub
            return False     # Other IPython shells (e.g., IPython terminal)
        except (NameError, ImportError):
            return False     # Standard Python shell or script

    @staticmethod
    def is_in_hub() -> bool:
        """
        Detects if the code is running in a JupyterHub environment.
        Looks for common JupyterHub environment variables.
        """
        os_env = os.environ
        is_hub = ("JUPYTERHUB_USER" in os_env) or ("JUPYTERHUB_API_TOKEN" in os_env)
        return is_hub

    @staticmethod
    def is_in_lx() -> bool:
        """Detects if the code is running on a Linux-based system."""
        return sys.platform.startswith("linux")

    @staticmethod
    def is_in_win() -> bool:
        """Detects if the code is running on a Windows system."""
        return os.name == "nt"

    @staticmethod
    def is_in_docker() -> bool:
        """Detects if the code is running inside a Docker container."""
        return os.path.exists("/.dockerenv")

    @classmethod
    def cross_platform_save_load(cls):
        """Workaround to allow cross platform save/load with Path."""
        sys.modules["pathlib._local"] = pathlib 
        if cls.is_in_win():
            pathlib.PosixPath = pathlib.WindowsPath
        else: 
            pathlib.WindowsPath = pathlib.PosixPath

    @staticmethod
    def joblib_dependecy_workaround():
        """Workaround to avoid error on certain OS platforms and versions."""
        cpu = 8
        try: 
            cpu = psutil.cpu_count(logical=True)
        except:
            try: 
                cpu = int(os.cpu_count())
            except: 
                pass        
        cpu = max(1, cpu-2)
        os.environ["LOKY_MAX_CPU_COUNT"] = str(cpu)