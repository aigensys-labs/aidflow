"""
AIDflow — Automated Interactive Development Pipeline.

The public API surface for the AIDflow framework. Exposes the
three primary user-facing components:

- **ProjectContext**: Training pipeline — initialization,
  configuration, and execution of all ML stages.
- **Predictor**: Inference engine — applies trained pipelines
  to new data using locked production parameters.
- **ProductionExporter**: Deployment layer — exports the
  champion model for standalone scoring.

All other components - workers (api nodes),  core infrastructure, 
display, storage - are internal and accessed through these three 
entry points.
"""

from . import (
    ProjectContext,
    Predictor,
    ExportConfig,
    ProductionExporter,
    UIGetter
)
from .core.foundation import (
    AigStop
)
from .core.execution import (
    set_backend,
    config_manager
)
config = set_backend(
    config_manager, 
    name = 'env-config',    
)