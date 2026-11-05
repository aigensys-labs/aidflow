"""
Centralized Pipeline Context and Global State Coordinator.

This module implements the `PipelineContext`, the primary architectural hub that 
unifies all disparate subsystems of an active AIDflow project into a single, 
coherent interface (Control Plane). While the core library provides the tools, 
the PipelineContext  provides the session-specific state, binding together parameters, 
data graphs, model registries, and I/O handlers for a specific execution run.

Key architectural roles provided by this module:
1. Unified Access Point (Single Entry): Acts as the master coordinator (the 'CTX'), 
   providing seamless access to the `setup` (params), `workers` (execution), 
   `datasets` (state), and `registry` (discovery) without requiring complex 
   dependency injection across the pipeline.

2. subsystem Synchronization: Ensures that all I/O handlers (`model_io`, `report_io`, etc.) 
   and the `predictor` engine are initialized with the same project-specific 
   configuration, guaranteeing consistency between training and inference.

3. State Persistence Binding: Links the logical data structures (in `DataSets`) 
   to the physical file system (via `structure`), allowing the framework to 
   automatically resolve where a specific model's results should be read from or written to.

4. Lifecycle Integration: Bridges the gap between static design (`AIGFLOW`) and 
   dynamic execution (`WorkersContext`), enabling the pipeline to move fluidly 
   from initial data loading to final champion agent selection.
"""
 
from ..core.parameters  import  (
                                Parameters, 
                                AIGFLOW
                                )
from ..core.registry    import  (
                                ModelRegistry, 
                                AIGMODELS
                                )
from ..core.datanode    import  DataSets
from ..core.io          import  (
                                ModelIO, 
                                ReportIO, 
                                FeatureIO,
                                DataIO
                                )
from ..predictor        import  Predictor
from .steps             import  WorkersContext
from .interface         import  ExecutionContext


class PipelineContext:
    """
    Centralized Pipeline Context and Global State Coordinator for the AIG pipeline.
    config, I/O, features, transforms/predict, structure, registry.

   The primary architectural hub that unifies all disparate subsystems of an active 
   AIDflow project into a single, coherent interface (Control Plane). While the core 
   library provides the tools, the PipelineContext  provides the session-specific 
   state, binding together parameters, data graphs, model registries, and I/O handlers 
   for a specific execution run.
   """

    def __repr__(self) -> str:
        return self.help()

    @staticmethod
    def help() -> str:
        return ("AIG Context Access to the Key Components:"
        "\n► setup      : Pipeline params  → global parameters & settings - can be adjusted in a pipe-step (worker as a flow node)"
        "\n► flow       : Pipeline runner  → ordered execution plan & run of all pipeline workers"
        "\n► workers    : Pipeline workers → links to activated workers during flow execution with context-aware input-output"
        "\n► datasets   : Pipeline outputs → data-oriented structure of workers' results"
        "\n► models     : Pipeline models  → List of all workers' models → model orchestrates internal fitter, while worker orchestrates model"
        "\n► registry   : Registry and access to all available models and their internal fitters"
        "\n► model_io   : Load saved trained model  → model_io.load(model)"
        "\n► report_io  : Load saved model report   → report_io.load(model)"
        "\n► feature_io : Load saved model features → feature_io.load()[model]"
        "\n► data_io    : Data ingestion & export context-aware IO"
        "\n► predictor  : Inference engine of all trained models"
        "\n► structure  : Pipeline directory architecture & paths"
    )

    def __init__(self, setup: Parameters):
        self.setup      = setup
        self.workers    = WorkersContext(setup)
        self.datasets   = DataSets()
        self.flow       = ExecutionContext(self)
        self.models     = AIGMODELS
        self.registry   = ModelRegistry
        self.structure  = setup.project_paths
        self.model_io   = ModelIO(setup)
        self.report_io  = ReportIO(setup)
        self.data_io    = DataIO(setup)
        self.feature_io = FeatureIO(setup)
        self.predictor  = Predictor(setup)