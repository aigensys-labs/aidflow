"""
Pipeline Execution Orchestrator and Worker Lifecycle Manager.

This module implements the executive layer of AIDflow, responsible for
transforming static model definitions into active pipeline steps. It acts as
the bridge between the high-level flow configuration (defined in `interface.py`)
and the low-level execution logic handled by the Model and Fitter layers.

Key Architectural Roles
-----------------------
1. **Step Lifecycle Automation** (`WorkerTask`):
   Orchestrates the full operational cycle of a pipeline step —
   resolving dynamic inputs, instantiating the appropriate model,
   executing fit/transform, and registering the resulting artifacts
   in the global context.

2. **Dynamic Worker Registry** (`WorkerModels`):
   A frozen DictClass that serves as a "known names" registry.
   When a worker is declared in a phase's `interface.py`, its
   AIGFLOW-standard name is registered here. Other parts of the
   system (UI discovery, `WorkerPhaseBase.__repr__`) use this to
   list all available workers.

3. **Phase Base** (`WorkerPhaseBase`):
   The base class for all flow phases. Holds the `ProjectContext`
   reference, provides the registration mechanism to instantiate
   and attach workers as attributes, and registers them in the
   `ProjectWorkerRegistry` for global lookup.

4. **Contract Declaration** (`WorkerContract`):
   The single declarative point where a worker specifies its data
   inflow (defaults), outflow (dataset bindings), and verification
   (continuation checks). This decouples the "what data this step
   needs/produces" from the "how to compute it."

5. **Interface Standardization** (`WorkerTask.model`):
   Factory method ensuring all worker definitions across different
   pipeline phases adhere to a strict schema (MODEL, TITLE, CHOICE),
   enabling seamless integration with the AIDflow GUI and automated
   reporting tools.
"""

from ...deps.data       import  pDF
from ...deps.types      import  ( 
                                Generic,
                                TypeVar,
                                )
from ..structure        import  AIGMODELS
from ..parameters       import  Parameters
from ..registry         import  (
                                ModelRegistry,                             
                                ProjectModelRegistry,
                                ProjectWorkerRegistry
                                )

from ..factory          import  (
                                DictClass, 
                                Dataclass
                                )
from ..protocol         import  ProjectContextProtocol                                
from ..validation.data  import  DataValidation
from .model             import  (                                
                                FIT,
                                Validator,
                                ModelEnvelope,
                                ModelInput,
                                ModelResult,
                                ModelBase,
                                ModelUIBase
                                )
from .fitted            import  FittedOffload


@Dataclass.base
class WorkerConfig:
    """
    Static worker configuration schema.

    Declared in each phase's `interface.py` and validated at
    registration time. Defines the identity, model class, and
    optional target binding for a single pipeline step.

    Attributes
    ----------
    model : str
        AIDflow-standard model identifier. Used as 
        the registry key and attribute name on the phase.
    Model : type
        The Model class (UI or base) to instantiate.
    Input : type
        The ModelInput dataclass for this model.
    Result : type
        The ModelResult dataclass for this model.
    Target : type | None, default=None
        Secondary worker class (auto-instantiated alongside
        the primary worker). Used for target-specific variants.
    choice : str | None, default=None
        Name of the setup parameter that controls fitter
        variant selection.
        If None, the first registered fitter is used.
    title : str | None, default=None
        Display title for the UI panel.
    offload : str | list[str] | None, default=None
        Fitted artifacts to persist to disk after execution.
    """    
    model:   str
    Model:   type
    Input:   type
    Result:  type
    Target:  type = None
    choice:  str  = None
    title:   str  = None
    offload: str | list[str] = None 

@Dataclass.base
class DatasetBinding:
    """
    Declarative dataset output binding for a worker step.

    Specifies how the worker's outputs are registered in the
    pipeline's dataset context for downstream consumption.

    Attributes
    ----------
    fitted : dict[str, str] | tuple[dict[str, str], ...]
        Links from the worker's fitted result to dataset names.
        Keys are dataset names; values are result attribute
        paths (e.g., `'data'`, `'features'`).
        Supports multiple bindings via tuple of dicts.
    direct : dict[str, str] | None, default=None
        Links from the worker's model (pre-fit) to dataset names.
        Used when a dataset should reference the model object
        directly (e.g., a prediction function).
    subset : tuple[str, dict] | tuple[tuple[str, dict], ...] | None, default=None
        Creates derived datasets as subsets of existing ones.
        See `WorkerContract` docstring for full examples.
    """    
    fitted: dict[str, str] | tuple[dict[str, str], ...]
    direct: dict[str, str] = None
    subset: tuple[str, dict] | tuple[tuple[str, dict], ...] = None

@Dataclass.base
class WorkerContract:
    """
    Declarative data contract for a worker step.

    This is the single declaration point where a worker specifies
    its complete relationship with the pipeline data context:
    what it needs (defaults), what it produces (datasets), and
    what it validates (verify).

    The contract is returned by `_get_contract()` and consumed
    by the internal processor during execution. Workers MUST
    override `_get_contract` to define their pipeline binding.

    Attributes
    ----------
    defaults : ModelInput | ModelEnvelope
        Resolved input values for the worker. Either directly
        a `ModelInput` instance or a `ModelEnvelope` wrapping
        internal model configuration.
    datasets : DatasetBinding
        Output binding specification — which fitted/direct/subset
        datasets to register downstream.
    verify : str | dict[str, str] | None, default=None
        Data continuation verification specification.
        - **str**: Verify a single dataset (e.g., `'train.stable'`).
        - **dict**: Verify multiple datasets with descriptions,
          e.g., `{'train': 'train.clean', 'test': 'test.clean'}`.

    datasets
    --------
    **(A) fitted** (standard):
    - Standard route - worker result is bound to fitted property.

    Default design expects dictionary linking datasets and fitted result:
    >>> DatasetBinding(fitted = {
    ...    'features.all.stable.dist'           : self._result.features,
    ...    'features.all.dropped.stable.dist'   : self._result.dropped,
    ...    'features.num.stable.dist'           : self._result.num_features,
    ...    'features.cat.stable.dist'           : self._result.cat_features,
    ...    'stats.stable.features'              : self._stats
    ... })

    **(B) fitted + direct** (extended):
    - Standard route - worker result is bound to fitted property.
    - Extra route - worker result is bound directly on the worker.    
    - Extended design expects tuple → multiple links provided.
    
    >>> DatasetBinding(
    ...     fitted = {
    ...         'train.clean'  : self._result.data,
    ...         'dtypes.clean' : self._result.dtypes,
    ...     },
    ...     direct = {
    ...         'test.clean' : self._model_result.data
    ...     }
    ...)

    **(C) subset** (derived):
    - Creates a subset from existing datasets.
    - Tuple of source model and dictionary of datasets links to register.
    
    For instance:
        - obj = [self.fe_cleaner, self.fe_stabilizer]
        - links = { 'features.num.dropped.stable': ['vars_num', 'vars_dropped'] }
    
        1. Subset of vars_dropped in obj_1 (self.fe_stabilizer) given vars_num 
        (from obj_0 fe_cleaner) is created → [x for x in vars_dropped if x in vars_num]
        2. features.num.dropped.stable is added to dataset with value of created subset

    Default design expects tuple[str, dict] specifying datasets subsets:
    >>> source = Dataclass.schema(FeCleanerResultDual)
    ... DatasetBinding(subset = (AIGFLOW.Data.fe_cleaner, {
    ...     'features.num.dropped.stable.dist' : [source.num_features, self._result.dropped],
    ...     'features.cat.dropped.stable.dist' : [source.cat_features, self._result.dropped],
    ... }) 

    Extended design expects tuple[str, tuple] to set multiple subsets from different sources:
    >>> source1 = Dataclass.schema(FeStabilizerResult)
    ... source2 = Dataclass.schema(IxnEncoderResult)
    ... DatasetBinding(subset = (
    ...   (AIGFLOW.Data.fe_stabilizer, {
    ...     'features.all.stable.target' : [source1.features,     self._result.features.org],
    ...     'features.num.stable.target' : [source1.num_features, self._result.features.org],
    ...   }),
    ...   (AIGFLOW.Encoding.ixn_encoder, {
    ...     'features.ixn.stable.target'  : [source2.features, self._result.features.org],
    ...     'features.ixn.dropped.stable' : [source2.features, self._result.dropped.org],
    ...   })
    ... )    
    """    
    defaults: ModelInput | ModelEnvelope
    datasets: DatasetBinding
    verify: str | dict[str, str] = None

class ModelDataResult:
    """
    Dynamic path resolver for pre-fit model result attributes.

    Acts as a descriptor proxy that converts attribute access into
    standardized dataset path strings (e.g., `'_result.data'`).
    Used in `DatasetBinding.direct` to reference model-level
    outputs without requiring a fully fitted model.

    Attributes
    ----------
    data : pDF
        Placeholder for type checking (not used at runtime).

    Example
    -------
    >>> mdr = ModelDataResult()
    >>> mdr.data      # '_result.data'
    >>> mdr.features  # '_result.features'
    """
    data: pDF

    def __getattr__(self, name) -> str:
        return f'{FIT.RESULT}.{name}'

ModelResultT = TypeVar('ModelResultT', bound=ModelResult)

class WorkerBase(Generic[ModelResultT]): 
    """
    Worker Base — Context-aware flow step orchestrator.

    The primary entry point for executing a single pipeline step.
    Each worker represents one logical operation in the AIDflow
    pipeline (e.g., feature cleaning, WOE encoding, GLM training).

    Responsibilities
    ----------------
    - **Configuration**: Holds the static `WorkerConfig` (model
      identity, input/result schemas, target binding).
    - **Execution**: Delegates to `WorkerProcessor` which
      orchestrates input resolution, model fitting, and
      dataset registration.
    - **Proxy**: Once fitted, transparently forwards attribute
      access to the internal model (`worker.features` →
      `model.features`).
    - **Context Bridge**: Provides `setup`, `MODEL`, and `spec`
      accessors for the broader pipeline context.

    Usage
    -----
    Workers are instantiated by the phase (via `_register`) and
    invoked by the user or automation layer:

    >>> # In phase interface:
    >>> phase._register((Agent1Worker, Agent2Worker, ...))
    >>> # Invocation:
    >>> agent1 = ctx.workers.AIGFLOW_Training_agent1
    >>> agent1(inputs)   # or agent1() for defaults
    >>> agent1.features  # proxy to fitted model

    Parameters
    ----------
    ctx : ProjectContextProtocol
        The project context (single source of truth for setup,
        datasets, workers IO, and model registry).
    """
    config: WorkerConfig

    def __init__(self, ctx: ProjectContextProtocol):
        self.ctx  = ctx
        self.sets = ctx.datasets
        self._processor = WorkerProcessor(self, ctx)

        self._result: ModelResultT = Dataclass.schema(self.config.Result, path_type='full')

    def __call__(self, inputs: ModelInput = None, **kwargs):
        self._processor.fit(inputs, kwargs)
        return self
    
    def __getattr__(self, name):
        if not self.is_fitted:
            return
        return getattr(self.model, name)

    def __repr__(self):
        if not self.is_fitted:
            return (
                f'Worker: {self.config.model}'
                f'\n► Context process → {self.config.model}()'
                f'\n► Inputs schema   → {self.config.model}.{FIT.SPEC}'
                f'\n► Fitters schema  → {self.config.model}.fitters'
            )
        return repr(self.model)

    @property
    def MODEL(self) -> str:
        """Model ID (registry key)."""
        return self.config.model

    @property
    def model(self) -> ModelBase | ModelUIBase | None:
        """Worker model instance. Prior fit returns None."""
        return self._processor.model
    
    @property
    def setup(self) -> Parameters:
        """Flow parametrization setup."""
        return self.ctx.setup

    @property
    def fitters(self) -> DictClass:
        """
        All available fitters registered for this model.

        Returns the registry of fitter variants selectable via
        `fx_choice` at fit time.
        """
        return ModelRegistry[self.config.model]
        
    @property
    def spec(self):
        """
        Worker I/O specification.

        Unfitted: returns the default input schema (structure only).
        Fitted: returns the populated input/output record.
        """
        if self.is_fitted:            
            return getattr(self._processor.task.worker_io, self.config.model)
        return Dataclass.create_default_instance(self.config.Input)

    @property
    def is_fitted(self) -> bool:
        """True if the internal model has been fitted; False otherwise."""
        return self._processor.model is not None
    
    def get_target_worker(self) -> "WorkerBase | None":
        """
        Resolve the target worker instance for this step.

        The target worker is a secondary worker auto-instantiated
        alongside the primary (e.g., a target-specific variant).

        Resolution order:
        1. Lookup in the current phase (if already registered).
        2. Create a fresh instance if not found.

        Returns
        -------
        WorkerBase | None
            The target worker instance, or None if no target is
            configured.
        """
        target_worker: "WorkerBase | None" = self.config.Target
        tw = self.ctx.flow.get_worker(target_worker.config.model)
        if tw is not None:
            return tw
        return target_worker(self.ctx)

    def _get_contract(self, is_default: bool = True) -> WorkerContract:
        """
        Declares worker's binding contract with the pipeline data context.

        This is the single declaration point where a worker specifies:
        - **defaults** — how input fields resolve from the dataset context
        - **datasets** — which outputs to register downstream
        - **verify** — which data samples to validate post-fit

        The returned contract is consumed by the internal configurator
        during execution. Workers MUST override this method to define
        their pipeline binding.

        Parameters
        ----------
        is_default : bool = True
            Declares whether ModelEnvelope config is applicable.
            Active only for workers with ModelEnvelope defaults.

        Returns
        -------
        WorkerContract
            Declarative contract describing data inflow, outflow,
            and verification for this worker step.
            
        Notes
        -----
        This method is called once per execution (`__call__`).
        It must return a fully constructed `WorkerContract` instance.
        All data links (`self._sets.get(...)`) are resolved at call time,
        reflecting the current pipeline state.

        Example
        -------
        >>> def _get_contract(self):
        ...     return WorkerContract(
        ...         defaults = FeCleanerInput(
        ...             train_set = self._sets.get('train.raw'),
        ...             test_set  = self._sets.get('test.raw'),
        ...             dtypes    = self._sets.get('dtypes.raw'),
        ...         ),
        ...         datasets = DatasetBinding(
        ...             fitted = {
        ...                 'train.clean':          self._result.data,
        ...                 'dtypes.clean':         self._result.dtypes,
        ...                 'features.all.clean':   self._result.features,
        ...                 'stats.clean':          self._stats,
        ...             },
        ...             direct = {
        ...                 'test.clean': self._model_result.data,
        ...             }
        ...         ),
        ...         verify = {'train': 'train.clean', 'test': 'test.clean'},
        ...     )
        """
        raise NotImplementedError('AIG SAFETY ► Worker must define `_get_contract`.')        


class WorkerProcessor:
    """
    Worker Execution Processor — orchestrates the full step lifecycle.

    Created internally by `WorkerBase.__init__` and consumed via
    `WorkerBase.__call__`. Handles the sequential execution:

    1. **Validation**: Verifies the worker config satisfies the
       AIG model contract (model, Model, Input, Result types).
    2. **Registration**: Registers the worker name in `WorkerModels`
       (for discovery) and the instance in `ProjectWorkerRegistry`
       (for offload/persistence).
    3. **Execution**: Resolves the contract, creates inputs,
       fits the model (via `WorkerTask`), registers outputs.
    4. **Post-execution**: Sets dataset links, verifies data
       continuation, and offloads to disk if configured.

    This class is an implementation detail — users interact with
    `WorkerBase`, not `WorkerProcessor` directly.
    """
    model: ModelBase | ModelUIBase = None

    def __init__(self, worker: WorkerBase, ctx: ProjectContextProtocol):
        Validator.worker_base(
            worker.config.model, 
            worker.config.title, 
            worker.config.Model, 
            worker.config.Input, 
            worker.config.Result
        )
        self.worker = worker

        self._validate()
        self._register()

        self.task   = WorkerTask(worker)
        self.sets   = ctx.datasets

        self.model_result = ModelDataResult()

    @property
    def config(self):
        return self.worker.config

    @property
    def setup(self):
        return self.worker.setup
    
    @property
    def stats(self):
        """Returns model standard stats string."""
        return FIT.STATS
    
    def _register(self) -> None:
        WorkerModels.register(self.config.model)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs: ModelInput | None, kwargs: dict):
        """
        Execute the full worker step lifecycle.

        Orchestrates: contract resolution → input creation →
        model fitting → dataset registration → verification →
        offload.

        Parameters
        ----------
        inputs : ModelInput | None
            Runtime inputs. If None, defaults from the contract
            are used.
        kwargs : dict
            Dynamic parameter overrides for individual input fields.

        Returns
        -------
        WorkerProcessor
            Self (with `.model` set to the fitted model).
        """        
        if self._is_base():
              task = self.task.process_base
        else: task = self.task.process

        is_default = self.is_default(inputs, kwargs)
        contract   = self.worker._get_contract(is_default)
        
        defaults   = contract.defaults
        params, kw = self._get_params_kwargs(inputs, defaults, kwargs)
        self.model = task(inputs, defaults, params, **kw)

        self._set_datasets_links(contract)
        self._verify_flow(contract)
        self._offload_to_disk()

        return self

    def is_default(self, 
        inputs: ModelInput, 
        kwargs: dict, 
        key: str = 'train_set'
    ) -> bool:
        """
        Returns *True* when key is not present in both inputs and kwargs; otherwise *False*.

        Parameters
        ----------
        inputs : ModelInput
            The raw or partially resolved input data.
        kwargs: dict
            Dynamic indiv. input parameters with same keys as inputs.
        key : str = 'train_set'
            The uniuqe key present in inputs.
        """        
        return (
            ( inputs is None ) 
            and 
            ( kwargs.get(key) is None )
        )
    
    def _is_base(self) -> bool:
        return issubclass(self.config.Model, ModelBase)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_inputs(
        inputs:   ModelInput | None, 
        defaults: ModelInput | ModelEnvelope
    ) -> ModelInput:
        inputs_ = inputs or defaults
        if isinstance(inputs_, ModelEnvelope):
            inputs_ = inputs_.inputs
        return inputs_
    
    def _get_params_kwargs(self, 
        inputs:     ModelInput | ModelEnvelope | None, 
        defaults:   ModelInput, 
        kwargs:     dict
    ) -> tuple[dict, dict]:
        inputs_ = self._get_inputs(inputs, defaults)
        params  = {x: v for x,v in kwargs.items() if x in inputs_.keys()}
        kwargs_ = {x: v for x,v in kwargs.items() if x not in params}
        return params, kwargs_
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_datasets_subsets(self, subsets:tuple) -> None:
        source, links = subsets
        self.sets.set_from_subset(
            self.config.model, 
            [self.task.get_worker(source), self.model], 
            links
        )

    def _set_datasets_links(self, contract: WorkerContract) -> None:
        for is_fitted, links in (
            (True,  contract.datasets.fitted),
            (False, contract.datasets.direct),
        ):
            if links is None:
                continue
            links_ = links
            if isinstance(links, dict):
                links_ = (links, )
            for l in links_:
                self.sets.set_links(
                    self.config.model, self.model, links = l, is_fitted = is_fitted
                )
        subsets = contract.datasets.subset
        if subsets is None:
            return
        if not isinstance(subsets[1], tuple):
            subsets = (subsets, )
        for s in subsets:
            self._set_datasets_subsets(s)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _offload_to_disk(self) -> None:
        if self.config.offload is None:
            return
        fo = FittedOffload(self.setup.project_signature)
        fo.offload_to_disk(self.config.offload)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _verify_flow(self, contract: WorkerContract) -> None:
        """
        Workers implementing `_verify_continuation` provide flow verification
        of data continuation.
        """        
        d = contract.verify
        if d is None:
            return
        if isinstance(d, str):
            d = {'train': d}
        for s, d_ in d.items():
            DataValidation.target_continuation(
                self.sets.get(d_), 
                self.task.setup.target,
                desc = s
            )           

    def _validate(self) -> None:
        """Validates mandatory config types of `model, Model, Input, Result`."""
        if self.config.model not in AIGMODELS:
            raise ValueError(
                'AIG SAFETY ► Worker config model must from AIGMODELS:'
                f'\n► {'\n► '.join(AIGMODELS)}'
            )
        val = {
            self.config.Model: (ModelUIBase, ModelBase), 
            self.config.Input:  ModelInput, 
            self.config.Result: ModelResult,
            self.config.Target: type(self).__base__
        }
        for cls_, type_ in val.items():
            if cls_ is None:
                continue
            if not issubclass(cls_, type_):
                raise TypeError('AIG SAFETY'
                    '\n► Interface Worker config must follow AIG base signature.'
                    f'\n► Expected: {type_} \n► Worker Basis: {cls_.__base__}.'
                )   

class WorkerTask:
    """
    Orchestrator for executing a single worker step in the pipeline.
    
    Encapsulates the automation logic to run a worker. It bridges the gap
    between the static configuration defined in `WorkerModels` (via `interface.py`)
    and the dynamic execution context provided by `ProjectContext`.
    
    It handles the full lifecycle of a step:
    1. Configuring the execution environment (Setup, Steps Context).
    2. Resolving model selections and titles from the config.
    3. Instantiating and fitting the model.
    4. Registering the results.
    
    Parameters
    ----------
    flow_phase : type
        The class representing the current phase of the pipeline flow.
    config : ProjectContext
        The project configuration containing setup and steps context.
    """    
    def __init__(self, worker: WorkerBase):
        self.worker     = worker
        self.ctx        = worker.ctx
        self.setup      = self.ctx.setup
        self.worker_io  = self.ctx.workers.io
        self.p_sign     = self.setup.project_signature
    
    def _get_inputs(self, inputs: ModelInput | ModelEnvelope) -> ModelInput:
        if isinstance(inputs, ModelEnvelope):
            return inputs.inputs
        return inputs

    def _get_model(self) -> ModelUIBase | ModelBase:
        """Retrieve the UI class (executor) for a specific model."""
        return self.worker.config.Model

    def _get_model_id(self) -> str:
        """Retrieve the model str id by AIGFLOW standard."""
        return self.worker.config.model

    def _get_choice(self) -> None|str:
        """
        Resolve the specific model variant choice from the setup configuration.
        """
        prm = self.worker.config.choice
        mid = self._get_model_id()
        if not prm:
            return ModelRegistry[mid].key()
        prm = getattr(self.setup, prm) -1 
        return ModelRegistry[mid][prm]
    
    def _get_title(self) -> str:
        """Retrieve the display title for the worker's GUI panel."""
        return self.worker.config.title
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _fit_model(self, inputs: ModelInput | ModelEnvelope, kw: dict) -> ModelUIBase:
        inputs_  = self._get_inputs(inputs)
        model_ui = self._get_model()
        model_id = self._get_model_id()
        model_registry = ModelRegistry[model_id]
        
        fitted_model: ModelUIBase = model_ui(self.setup, inputs_, **kw)
        
        fitted_model.panel(
            inputs_.fx_choice,
            model_registry.keys(), 
            title = self._get_title()
        )  
        fitted_model.fit(inputs)
        return fitted_model
    
    def _set_fitted(self, 
        fitted_model: ModelUIBase | ModelBase, 
        result:       ModelResult = None,
        model:        str = None
    ):
        model_id = model or self._get_model_id()
        ProjectModelRegistry.register(fitted_model)
        self.worker_io.register_outputs(model_id, fitted_model, result)
        fitted_model._set_io( getattr(self.worker_io, model_id) )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def process(self, 
        inputs:   ModelInput, 
        defaults: ModelInput | ModelEnvelope,
        params:   dict = {},
        **kw
    ) -> ModelUIBase:
        """
        Execute the full pipeline step for a specific model with UI capabilities → `ModelUIBase`.
        
        This is the primary entry point for running a task. It orchestrates:
        1. Input Creation: Resolves final values and locks inputs in `StepsContext`.
        2. Model Fitting: Instantiates and runs the model via `_fit_model`.
        3. Output Registration: Saves the result to the context.
        4. Model Registration: Registers the model to the `ModelRegistry`
        6. State Update: Stores the fitted model on the flow phase.
        
        Parameters
        ----------
        inputs : ModelInput
            The raw or partially resolved input data.
        defaults : ModelInput
            The default schema/structure for the inputs.
        params: dict
            Allows to dynamically inject indiv. input parameters by their names.
        **kw
            Additional parameters for the model execution.
            
        Returns
        -------
        ModelUIBase
            The fitted model instance ready for downstream use.
        """         
        choice  = self._get_choice()
        model   = self._get_model_id()
        
        inputs_ = self.worker_io.create_inputs(model, choice, inputs, defaults, params)

        fitted_model = self._fit_model(inputs_, kw)
        self._set_fitted(fitted_model)
        return fitted_model
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _fit_model_base(self, inputs: ModelInput | ModelEnvelope, kw: dict) -> ModelBase:
        inputs_    = self._get_inputs(inputs)
        model_base = self._get_model()
        model_base_: ModelBase = model_base(self.setup, **kw)
        return model_base_.fit(inputs, fx_choice = inputs_.fx_choice)

    def process_base(self,
        inputs:   ModelInput, 
        defaults: ModelInput | ModelEnvelope,
        params:   dict = {},
        **kw
    ) -> ModelBase:
        """
        Execute the full pipeline step for a specific worker w/o UI capabilities → `ModelBase`.
        
        This is the primary entry point for running a task. It orchestrates:
        1. Input Creation: Resolves final values and locks inputs in `StepsContext`.
        2. Model Fitting: Instantiates and runs the model via `_fit_model`.
        3. Output Registration: Saves the result to the context.
        4. Model Registration: Registers the model to the `ModelRegistry`
        5. State Update: Stores the fitted model on the flow phase.
        
        Parameters
        ----------
        inputs : ModelInput
            The raw or partially resolved input data.
        defaults : ModelInput
            The default schema/structure for the inputs.
        params: dict
            Allows to dynamically inject indiv. input parameters by their names.
        **kw
            Additional parameters for the model execution.
            
        Returns
        -------
        ModelBase
            The fitted model instance ready for downstream use.
        """             
        fx_choice = self._get_choice()
        model     = self._get_model_id()
        inputs_   = self.worker_io.create_inputs(model, fx_choice, inputs, defaults, params)
        fitted_model = self._fit_model_base(inputs_, kw)
        self._set_fitted(fitted_model)
        return fitted_model
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def register_base(self, 
        model:        str, 
        fitted_model: ModelBase, 
        inputs:       ModelInput, 
        result:       ModelResult
    ) -> None:
        """
        Register `ModelBase` w/o interface access to the pipeline context.
        
        This is the secondary entry point to register fitted model. It orchestrates:
        1. Input Registration: Saves the inputs to the `StepsContext`.
        2. Output Registration: Saves the result to the context.
        3. Model Registration: Registers the model to the `ModelRegistry`
        4. State Update: Stores the fitted model on the flow phase.
        
        Parameters
        ----------
        model : str
            The unique identifier for the worker step.
        fitted_model : ModelBase
            Fitted instance of a model to be registered.
        inputs : ModelInput
            The raw or partially resolved input data.
        """        
        self.worker_io.register_inputs(model, inputs)
        self._set_fitted(model, fitted_model, result)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def model(model_ui:ModelUIBase | ModelBase, panel_title:str, model_choice:str = None) -> dict:
        """
        Factory method to construct a standard worker configuration dictionary.
        
        Simplifies the creation of worker definitions in `interface.py` by ensuring
        all required keys (MODEL, TITLE, CHOICE) are present and correctly mapped.
        
        Parameters
        ----------
        model_ui : ModelUIBase | ModelBase
            The class representing the model executor.
        panel_title : str
            The title to be displayed in the GUI.
        model_choice : str, optional
            The name of the strategy config parameter controlling model selection.
            
        Returns
        -------
        dict
            A dictionary conforming to the `WORKER` schema.
        """        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_worker(self, model: str|list) -> ModelBase|None | dict[str, ModelBase|None]:
        """
        Returns worker instance or populated dictionary by requested worker instances 
        if they were fitted under current project, otherwise None.

        Parameters
        ----------
        model : str | list
            The uniuqe model key or list of keys present in AIGMODELS.
        """
        worker = lambda x: getattr(self.ctx.workers, x, None)
        if isinstance(model, str):
            return worker(model)
        reg = {}
        for m in model:            
            reg[m] = worker(m)
        return reg
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class WorkerModels(DictClass):
    """
    Dynamic registry of all known worker names across the pipeline.

    Acts as a namespace where individual workers are dynamically registered
    by `interface.py` in each flow phase. Once registered, the name becomes
    a class attribute, making it discoverable via introspection (`dir()`,
    `hasattr()`, iteration).

    This class uses a **frozen DictClass** pattern: the registry is immutable
    by default, and mutations are only permitted during the brief registration
    window (when `_frozen` is temporarily set to False). This prevents
    accidental modification of the worker list after the pipeline is initialized.

    The registry is populated once during package import (when each phase's
    `interface.py` is executed) and remains stable for the lifetime of the
    application.

    Example:
        >>> WorkerModels.register('AIGFLOW.Data.loader')
        >>> 'AIGFLOW.Data.loader' in dir(WorkerModels)
        True
        >>> [w for w in WorkerModels if hasattr(WorkerModels, w)]
        ['AIGFLOW.Data.loader', 'AIGFLOW.Data.fe_cleaner', ...]
    """ 

    @classmethod
    def register(cls, model: str) -> None:
        """
        Dynamically register a worker name to the known workers registry.

        Sets the worker's AIGFLOW-standard name as a class attribute with value
        `True`. This makes the name discoverable via attribute access and
        iteration while keeping the registry lightweight (no instance data stored).

        The frozen state is temporarily unlocked during the attribute set
        operation and immediately re-locked to maintain immutability.

        Parameters:
            model (str): The model name by AIGFLOW standard (e.g., 'AIGFLOW.Data.loader').
                Must be a valid string identifier. Non-string values are silently
                ignored (defensive check for malformed registrations).

        Side Effects:
            - A class attribute with the given name is created (value: True).
            - The registry is momentarily unfrozen and re-frozen.

        Example:
            >>> WorkerModels.register('AIGFLOW.Data.fe_cleaner')
            >>> WorkerModels.AIGFLOW_Data_fe_cleaner  # attribute is set
            True
        """
        if not isinstance(model, str):
            return
        cls._frozen = False
        setattr(cls, model, True)        
        cls._frozen = True
        
class WorkerPhaseBase:
    """
    Base class for all pipeline flow phases.

    A flow phase represents a logical stage in the AIDflow pipeline
    (e.g., DataPhase, EncodingPhase, TrainingPhase). Each phase:

    - Holds a reference to the `ProjectContext` (the single source of truth
      for all pipeline state, IO, and configuration).
    - Exposes its registered workers as attributes, enabling the canonical
      access pattern: `ctx.workers.{worker_name}`.
    - Registers workers in the `ProjectWorkerRegistry` for global lookup
      (used by offload, persistence, and cross-step data access).
    - Supports **target workers** — secondary workers that are automatically
      instantiated alongside a primary worker (e.g., a "target" variant of
      the stabilizer that operates on the target variable).

    Parameters
    ----------
    ctx : ProjectContextProtocol
        The project context instance. Provides access to:
        - `ctx.setup`: Global pipeline parameters.
        - `ctx.datasets`: Data-oriented structure of worker outputs.
        - `ctx.workers`: WorkersContext for IO management.
        - `ctx.registry`: ModelRegistry for fitter lookup.                    

    Design Notes
    ------------
    - Workers are registered **once** during module import (when
        `interface.py` is executed). The instances are created
        lazily via `_make_instance()` and stored as phase attributes.
    - The phase itself is stateless with respect to data; all data
        flows through the `ProjectContext` (`ctx.datasets`,
        `ctx.workers.io`).
    - The frozen `WorkerModels` registry ensures that the set of
        available worker names is stable and discoverable.

    Init
    ----
    >>> def __init__(self, ctx: ProjectContextProtocol):        
    ... self.ctx  = ctx
    ... self.sets = ctx.datasets
    """
    
    def __repr__(self):
        workers = [w for w in WorkerModels if hasattr(self, w)]
        return f'Registred workers:\n - {f"\n - ".join(workers)}'    

    def __init__(self, ctx: ProjectContextProtocol):        
        self.ctx  = ctx
        self.sets = ctx.datasets

    def _register(self, workers: tuple[WorkerBase, ...]) -> None:
        """
        Register a batch of workers to this flow phase.

        This is the primary entry point for worker registration. Called by
        `interface.py` during module import to activate a set of workers
        under the current phase.

        For each worker in the tuple:
        1. Verifies the worker class satisfies the `WorkerBase` contract.
        2. Instantiates the worker with the phase's context.
        3. Registers the instance in `ProjectWorkerRegistry` (for offload/persistence).
        4. Sets the instance as an attribute on this phase (enabling
           `ctx.workers.{name}` access).
        5. If the worker config specifies a `Target`, registers that
           secondary worker as well.

        Parameters
        ----------
        workers : tuple[WorkerBase, ...]
            A tuple of worker classes to register. Each must be a subclass of `WorkerBase`.

        Example
        -------
        >>> # In interface.py:
        >>> phase._register((DataLoaderWorker, FeatureCleanerWorker, ...))
        >>> # After registration:
        >>> hasattr(phase, 'AIGFLOW.Data.loader')  # True
        """
        setter = WorkerPhaseSetter(self)
        setter.register(workers)

class WorkerPhaseSetter:
    """
    A flow phase setter of assigned workers.

    Parameters
    ----------
    phase : WorkerPhaseBase
        An instance of a flow phase.
    """

    def __init__(self, phase: WorkerPhaseBase):
        self.base = phase
        self.ctx  = phase.ctx

    def register(self, workers: tuple[WorkerBase, ...]) -> None:
        """Register a batch of workers to this flow phase."""
        for worker in workers:
            self._set_worker(worker)
            self._set_target_worker(worker)

    def _make_instance(self, worker: WorkerBase) -> WorkerBase:
        return worker(self.ctx)
    
    def _set_worker(self, worker: WorkerBase) -> None:
        self._verify(worker)            
        wi = self._make_instance(worker)
        ProjectWorkerRegistry.register(wi)
        setattr(self.base, worker.config.model, wi)

    def _set_target_worker(self, worker: WorkerBase) -> None:
        tw = worker.config.Target
        if tw is None:
            return
        self._set_worker(tw)

    @staticmethod
    def _verify(worker: WorkerBase) -> None:
        if issubclass(worker, WorkerBase):
            return
        raise ValueError('AIG SAFETY'
            '\n► FlowPhase must register only workers with WorkerBase signature.'
            f'\n► {worker} has not been signed by WorkerBase.'
        )  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬