"""
The architectural backbone and master blueprint for all AIDflow Model implementations.

This module defines the mandatory structural hierarchy that every model must 
follow to be compatible with the AIDflow orchestration engine. By enforcing a 
strict inheritance chain—comprising processing, transformation, fitting, and 
UI layers—it enables an "auto-plug" mechanism where new models are instantly 
integrated into the global pipeline with full awareness of their I/O signatures.

The Model backbone manages four critical lifecycle dimensions:
1. Execution Flow: Orchestrates the sequence from Model → Processor → Transformer → Fitter.

2. Dynamic UI Generation: Automatically derives interactive parameter panels via `OptionUIBase` 
   and `WorkerUIBase`, bridging the gap between code and user configuration.

3. State Synchronization: Maintains a unified `ModelState` that tracks fit results, 
   hyperparameters, and statistical metadata across the model's lifetime.

4. Pipeline Integration: Provides standardized `fit_transform` and `update_transform` 
   hooks, ensuring consistent behavior during both initial training and production 
   inference phases.

Any class inheriting from `Model` is guaranteed to be a first-class citizen 
within the AIDflow ecosystem, inheriting full support for lineage tracking, 
dashboarding, and parallel execution.
"""

from .model import (
    FIT, 
    FitterBase,
    TransformerBase,
    ProcessorBase,
    ModelProcessorBase,
    ModelState, 
    ModelInput, 
    ModelEnvelope, 
    ModelResult, 
    ModelConfigOverrides, 
    ModelBase, 
    ModelBaseDual,
    OptionUIBase,
    ModelUIBase
)
from .fitter import (
    FitterDataBase, 
    FitterTrainBase, 
    FitterModelBase
)