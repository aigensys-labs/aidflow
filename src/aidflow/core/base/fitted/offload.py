"""
Fitted Offload Manager - Unified RAM Reclamation Entry Point.

This module provides a single, convenience API to offload **all** result
descriptor types for a given model (or list of models) from RAM to disk
in one atomic call.

The three descriptor types managed:
    - **FittedResult**: The primary fitted result (Fitter/ModelRunner data).
    - **ModelDataResult**: Secondary Model-level auxiliary outputs.
    - **FitterDataRelay**: Fitter relay data (temp data, stats, coefficients, etc.).

This manager exists to ensure that when a pipeline step is offloaded,
*all* associated data is reclaimed from memory consistently, preventing
partial offload states where some data remains in RAM while other data
has been written to disk.
"""

from .fitted    import  FittedResult
from .result    import  ModelDataResult
from .relay     import  FitterDataRelay


class FittedOffload:
    """
    Unified offload manager for all result descriptor types.

    Provides a single entry point to offload all RAM-resident data
    associated with a pipeline step. Internally, it delegates to the
    three descriptor classes' `offload()` classmethods:

    1. `FitterDataRelay.offload()` — Fitter relay data (stats, coefficients).
    2. `ModelDataResult.offload()` — Secondary Model-level outputs.
    3. `FittedResult.offload()` — Primary fitted result data.

    The order matters: relay data and secondary results are offloaded first,
    then the primary result. This ensures that if the primary result
    references relay data (e.g., a Dataclass containing both), the relay data
    is already safely on disk before the in-memory reference is severed.

    Parameters
    ----------
    project_signature : str
        The project signature used for registry lookups.

    Example:
        >>> offloader = FittedOffload(setup.project_signature)
        >>> 
        >>> # Free RAM for a single step after inspecting its results:
        >>> offloader.offload_to_disk('AIGFLOW.Data.fe_cleaner')
        >>> 
        >>> # Free RAM for all data-phase steps:
        >>> offloader.offload_to_disk([
        ...     'AIGFLOW.Data.loader',
        ...     'AIGFLOW.Data.fe_cleaner',
        ...     'AIGFLOW.Data.fe_stabilizer',
        ...     'AIGFLOW.Data.obs_stabilizer',
        ... ])
    """

    def __init__(self, project_signature: str):
        self.p_sig = project_signature

    def offload_to_disk(self, model: str | list[str]) -> None:
        """
        Offload all result data for the given model(s) from RAM to disk.

        Executes the offload sequence for each descriptor type:
        1. Fitter relay data (stats, fitted coefficients).
        2. Secondary Model-level data (auxiliary outputs).
        3. Primary fitted result (the main pipeline output).

        After execution, all in-memory DataFrame/Arrow references for the
        specified model(s) are set to None, `gc.collect()` is triggered,
        and the Arrow memory pool is released.

        Parameters
        ----------
        model : str | list[str]
            The model identifier string(s) to offload.
            Can be a single model (e.g., 'AIGFLOW.Data.fe_cleaner') or a
            list of models for batch offloading.

        Side Effects
        ------------
        - Data is written to Arrow/Feather files on disk.
        - In-memory references are severed (set to None).
        - Python garbage collector is triggered.
        - Arrow's internal memory pool releases unused buffers.
        - All data remains accessible via transparent re-hydration
            on next attribute access (lazy loading).

        Note:
            This operation is **reversible**. Offloaded data is not destroyed;
            it is persisted to disk and will be automatically re-hydrated
            when the user or pipeline accesses the result again.
        """        
        arg = (model, self.p_sig)
        FitterDataRelay.offload(*arg)
        ModelDataResult.offload(*arg)
        FittedResult.offload(*arg)