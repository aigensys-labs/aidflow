"""
Fitted Result Management - Package Public API.

This package provides the descriptor-based memory management system for
AIDflow's pipeline results. It exposes four complementary components:

**Descriptors** (attached to classes for transparent data access):

- **ModelFittedResult** (primary): For the mandatory fitted result that
  flows from Fitter → Model → Worker. Handles complex routing between
  the Fitter and ModelRunner layers based on whether the Model implements
  `_get_result()`.

- **ModelDataResult** (secondary): For auxiliary Model-level outputs that
  reside directly on the ModelBase instance. Simpler routing — always
  targets the Model itself.

**Relay** (data transfer between layers):

- **FitterDataRelay**: Manages the data handoff between Fitter and
  Model/Worker layers. Provides the "hop" mechanism that allows to 
  access relay data from any layer in the deeper hierarchy.

**Offload** (memory reclamation):

- **FittedOffload**: Unified offload manager that reclaims RAM for all
  result descriptor types in a single atomic call. Orchestrates the
  save-to-disk, reference-severing, and garbage collection sequence
  across all three descriptor classes.

All descriptors share the same underlying offload/re-hydration mechanism
defined in `base.py`, providing:
- Transparent disk offload of large DataFrames (Arrow/Feather + LZ4).
- Lazy re-hydration on attribute access.
- Per-model isolation via WeakKeyDictionary caches.
- Safe memory reclamation (gc.collect + Arrow pool release).

Usage:
    >>> from aidflow.core.base.fitted import (
    ...     FittedResult,
    ...     ModelDataResult,
    ...     FitterDataRelay,
    ...     FittedOffload,
    ... )
    >>>
    >>> # On a FitterBase subclass (primary result):
    >>> class MyFitter(FitterBase):
    ...     result = FittedResult()
    ...
    >>> # On a ModelBase subclass (secondary outputs):
    >>> class MyModel(ModelBase):
    ...     _result = ModelDataResult()
    >>>
    >>> # Offload all results from RAM to disk:
    >>> offloader = FittedOffload(setup.project_signature)
    >>> offloader.offload_to_disk('AIGFLOW.Data.fe_cleaner')
    >>>
    >>> # Clear offloaded data (free disk space):
    >>> ModelFittedResult.clear('AIGFLOW.Data.fe_cleaner', project_signature)
    
Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Worker / UI Layer                                                  │
    │  ┌───────────────────────────────────────────────────────────────┐  │
    │  │  WorkerUIBase / ModelUIBase                                   │  │
    │  │  .fitted  → FittedResult     (descriptor)                     │  │
    │  │  ._result → ModelDataResult  (descriptor)                     │  │
    │  └──────────────────────────┬────────────────────────────────────┘  │
    │                             │ routes via FittedID                   │
    │  ┌──────────────────────────▼────────────────────────────────────┐  │
    │  │  ModelBase                                                    │  │
    │  │  .fitted  → FittedResult     (descriptor) via _get_result()   │  │
    │  │  ._result → ModelDataResult  (descriptor)                     │  │
    │  └──────────────────────────┬────────────────────────────────────┘  │
    │                             │                                       │
    │  ┌──────────────────────────▼────────────────────────────────────┐  │
    │  │  FitterBase                                                   │  │
    │  │  .result → FittedResult (data holder)                         │  │
    │  │  ._relay → FitterDataRelay → data transfer to upper layers    |  │
    │  └───────────────────────────────────────────────────────────────┘  │
    │                                                                     │
    │  FittedOffload → unified RAM reclamation for all of the above       │
    └─────────────────────────────────────────────────────────────────────┘
"""

from .fitted    import FittedResult
from .result    import ModelDataResult
from .relay     import FitterDataRelay
from .offload   import FittedOffload