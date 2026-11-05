"""
Core abstractions, foundational base classes, and structural contracts.

This package provides the stateless, low-level primitives for the AID framework.
It defines the abstract worker hierarchy, estimator interfaces, and structural
protocols used throughout the pipeline lifecycle.

Architectural Rule:
    Modules within `core` must remain strictly isolated from higher-level runtime
    orchestration and concrete context objects (`AID.ctx`). Dependencies flow
    one-way: `ctx` depends on `core`, never the reverse.
"""

from ..deps.types       import  Protocol
from .parameters        import  Parameters
from .registry          import  ModelRegistry
from .base              import  (
                                ModelInput,
                                ModelEnvelope,
                                ModelResult,
                                ModelBase,
                                ModelUIBase
                                )
from .datanode          import  DataSets


class StepsContextProtocol(Protocol):
    def create_inputs(self, 
        model:      str, 
        fx_choice:  str, 
        inputs:     ModelInput, 
        defaults:   ModelInput | ModelEnvelope, 
        params:     dict={}
    ) -> ModelInput | ModelEnvelope:
        """
        Create, validate, and `register inputs` for a specific pipeline worker.
        
        This is the primary entry point for setting up inputs. It resolves the final
        parameter values (merging setup, overrides, and defaults), validates the
        structure, and atomically registers the result in the `InputsRegistry`.
        
        Parameters
        ----------
        model (str) : The unique identifier for the step or model.
        fx_choice (str) : The unique identifier for the choice of internal fitter of the model.
        inputs (ModelInput) : Optional user-provided input data.
        defaults (ModelInput) : The default pipeline input schema.
        params (dict) : Free parameters allowing direct assigment to input signature. 

        Returns
        -------
        ModelInput
            The validated and registered input instance.
        """
        from ..ctx.steps import StepsContext
        StepsContext.create_inputs
    
    def register_inputs(self, model:str, inputs:ModelInput) -> None:
        """
        Register inputs for a specific pipeline worker.
        
        This is the secondary entry point to register inputs to the `InputsRegistry`.
        Aims workers / models which were fitted out of the interface scope.
        
        Parameters
        ----------
        model (str) : The unique identifier for the step or model.
        inputs (WorkerInput) : Optional user-provided input data.
        """
        from ..ctx.steps import StepsContext
        StepsContext.register_inputs

    def register_outputs(self, 
        model:  str, 
        worker: ModelUIBase | ModelBase, 
        result: ModelResult = None
    ) -> None:
        """
        Validate and `register outputs` to a specific pipeline worker inputs.
                
        Parameters
        ----------
        model : str
            The unique identifier for the step or model.
        worker : ModelUIBase | ModelBase
            Worker' Model Fitted Instance as a dynamic link to fitted outputs.
        result: ModelResult, optional
            Worker result provided directly; AIG standard fetch dynamic data by worker.fitted.
        """         
        from ..ctx.steps import StepsContext
        StepsContext.register_outputs

class WorkersContextProtocol(Protocol):
    """
    Pipeline Workers (Nodes) Dynamic IO Context.
    
    Workers are a context aware orchestrators of models in a given flow step while
    holding live reference to pipeline input-output context as io of the models.
    """    

    @property
    def io(self) -> StepsContextProtocol:
        """
        Context manager for pipeline worker inputs - outputs and configuration.
        
        Orchestrates the creation, validation, and registration of worker inputs and outputs
        for a specific pipeline step. It handles the resolution of configuration values
        (merging setup parameters, global overrides, and defaults) and ensures that once
        inputs are created, they are immediately registered and locked.        
        """
    from ..ctx.steps import WorkersContext
    WorkersContext.io
    
class ExecutionContextProtocol(Protocol):
    """
    Domain Interface and Pipeline Orchestration Layer.

    The primary API surface for end-users, providing a semantic interface to the underlying 
    AIDflow worker ecosystem. It transforms the generic execution capabilities of the `core/worker` 
    layer into domain-specific pipeline phase actions:
    - `data` - `encoding` - `training` - `selection`

    allowing users to interact with the ML pipeline using meaningful business logic rather 
    than technical configurations.
    """    
    def get_worker(self, worker: str): 
        from ..ctx.interface import ExecutionContext


class PipelineContextProtocol(Protocol):
    setup:      Parameters
    workers:    WorkersContextProtocol
    datasets:   DataSets
    flow:       ExecutionContextProtocol
    models:     list
    registry:   ModelRegistry
    structure:  Parameters.project_paths  

class ProjectContextProtocol(Protocol):
    """
    Project Context for defining, configuring AIG pipeline parameters and accessing 
    the key engine context components.
    
    This class acts as the entry point for both interactive UI sessions and 
    headless script execution. It manages the initialization of the user interface 
    (if enabled), collects configuration parameters, and delegates the creation 
    of the runtime context (`PipelineContext`).
    """    
    ctx: PipelineContextProtocol
    setup:      Parameters
    workers:    WorkersContextProtocol
    datasets:   DataSets
    flow:       ExecutionContextProtocol
    models:     list
    registry:   ModelRegistry
    structure:  Parameters.project_paths   
     
    def get_setup(self) -> Parameters: 
        """Returns the final validated pipeline Parameters as global config for steps."""          