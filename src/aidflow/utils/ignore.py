"""
Environment stabilization and system noise suppression utilities.

This module ensures a clean execution environment for AIDflow by managing 
third-party library warnings and resolving OS-level resource conflicts. It 
is designed to provide a "silent" production experience, preventing redundant 
framework logs from obscuring critical application errors.
"""

from ..deps.common import (
    warnings, 
    logging
)
from ..deps.data import (
    pd
)
from ..deps.exceptions import (
    HessianInversionWarning, 
    ConvergenceWarning 
)


class Warnings:

    @staticmethod
    def local_redundant():
        """Supress Local Redundant Messages."""
        warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
        warnings.simplefilter('ignore', category=pd.errors.DtypeWarning)
        warnings.filterwarnings('ignore', category=FutureWarning, message="The default of observed=False is deprecated and will be changed to True in a future version of pandas.")
        warnings.filterwarnings('ignore', category=FutureWarning, message="Dtype inference on a pandas object .* is deprecated.*")
        warnings.filterwarnings('ignore', category=FutureWarning, message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8")
        warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API.*")
        warnings.filterwarnings('ignore', category=UserWarning, message="To exit: use 'exit', 'quit', or Ctrl-D.")   
        warnings.filterwarnings('ignore', category=UserWarning, message="The argument 'infer_datetime_format' is deprecated and will be removed in a future version.")
        warnings.filterwarnings('ignore', category=UserWarning, message="Setting penalty=None will ignore the C and l1_ratio parameters")
        warnings.filterwarnings('ignore', category=UserWarning, message='LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray')
        warnings.filterwarnings('ignore', category=UserWarning, message='This figure includes Axes that are not compatible with tight_layout')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message="overflow encountered in exp")
        warnings.filterwarnings('ignore', category=RuntimeWarning, message="divide by zero encountered in log")
        warnings.filterwarnings('ignore', category=HessianInversionWarning, message="Inverting hessian failed.*")
        warnings.filterwarnings('ignore', category=ConvergenceWarning, message="Maximum Likelihood optimization failed to converge.*")        
        logging.getLogger("shap").setLevel(logging.ERROR)    