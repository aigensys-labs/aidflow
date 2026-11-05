"""
Fitted Result Base - RAM Offload and Lazy Re-hydration Infrastructure.

This module defines the abstract foundation for managing large pipeline outputs
(DataFrames, Arrow Tables) across the AIDflow Worker → Model → Fitter hierarchy.

The core problem it solves:
    In a multi-step ML pipeline, intermediate results (e.g., cleaned DataFrames,
    binned features, fitted coefficient matrices) can consume significant RAM.
    Holding all of them in memory simultaneously is impractical for enterprise-scale
    datasets. This module provides a **descriptor-based virtual proxy** that:

    1. **Offloads** heavy data from RAM to disk (Arrow/Feather format) on demand.
    2. **Re-hydrates** the data transparently when the user or next pipeline
       step accesses it via normal attribute access.
    3. **Routes** the save/load operation to the correct target instance
       (Fitter or ModelRunner) based on the model's implementation specifics.

Design Principles:
    - **Transparent Access**: Consumers call `instance.result` as usual; the
      descriptor intercepts, checks disk, and returns the data.
    - **Weak References**: All instance references are held via `weakref` to
      prevent circular references and allow natural garbage collection.
    - **Per-Model Caching**: Each model in `AIGMODELS` gets its own
      `WeakKeyDictionary` cache for isolation.
    - **Pluggable Target Resolution**: The concrete `FittedIDBase` subclass
      determines *which* instance actually holds the data (Fitter vs. ModelRunner).
"""

from ....deps.common    import  (
                                weakref,
                                gc, 
                                ABC,
                                abstractmethod
                                )
from ....deps.data      import  pa, pDF
from ....deps.types     import  Any

from ...structure       import  AIGMODELS
from ...registry        import  ProjectWorkerRegistry
from ...parameters      import  Parameters
from ...factory         import  Dataclass
from ...io.io           import  FittedIO


class FittedIDBase(ABC):
    """
    Abstract routing contract for FittedResult descriptor resolution.

    Defines the interface that concrete `FittedID` implementations must fulfill
    to map a given instance (Fitter, Model, ModelUI, or ModelRunner) to:
    - The correct **setup** (global pipeline parameters).
    - The correct **model identifier** (registry key).
    - The **target instance** that actually holds the result data.
    - The **fitted result value** (or its descriptor).

    This separation allows the base `FittedResultBase` descriptor to remain
    agnostic to the specific layer hierarchy, while concrete implementations
    (e.g., `FittedID` in `fitted.py`) encode the routing logic.

    Attributes:
        instance (Any): The object on which the descriptor was accessed.
    """

    def __init__(self, instance):
        self.instance = instance

    def is_offload(self) -> bool:
        """
        Determine whether offloading to disk is enabled for this project.

        Returns:
            bool: True if the global setup has offload mode enabled.
        """        
        return self.setup.offload

    def get_descriptor(self) -> "FittedResultBase":
        """
        Retrieve the descriptor object itself (not the resolved value).

        Returns:
            FittedResultBase: The descriptor instance for programmatic inspection.
        """        
        return self._get_fitted(descriptor=True)

    def get_fitted(self) -> Any:
        """
        Retrieve the resolved fitted result value.

        Returns:
            Any: The result data (DataFrame, dict, etc.) held by the target instance.
        """        
        return self._get_fitted()
        
    @property
    @abstractmethod
    def setup(self) -> Parameters:
        """
        Resolve the pipeline Parameters for this instance.

        Returns:
            Parameters: The global pipeline configuration.

        Raises:
            TypeError: If the instance type is not recognized by the routing logic.
        """
    
    @property
    @abstractmethod
    def model(self) -> str:
        """
        Resolve the model identifier (registry key) for this instance.

        Returns:
            str: The unique model identifier (e.g., 'AIGFLOW.Data.loader').

        Raises:
            TypeError: If the instance type is not recognized by the routing logic.
        """

    @abstractmethod
    def get_target_instance(self):
        """
        Resolve the instance that actually holds the fitted result data.

        In the AIDflow architecture, the result data can reside on either:
        - The **Fitter** (when `_get_result` is implemented on the Model), or
        - The **_ModelRunner** (when the Model handles result assembly).

        Returns:
            Any: The target instance holding the result descriptor.

        Raises:
            TypeError: If routing cannot be resolved for the given instance type.
        """
    
    @abstractmethod
    def _get_fitted(self, descriptor:bool=False):
        """
        Retrieve the fitted result from the resolved target instance.

        Parameters:
            descriptor (bool): If True, return the descriptor object itself.
                If False, return the resolved value.

        Returns:
            FittedResultBase | Any: The descriptor or the result data.
        """

    @abstractmethod
    def _error(self):
        """
        Generate a descriptive error message for unsupported instance types.

        Returns:
            str: A formatted error string explaining the routing failure.
        """
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FittedIORegistry:
    """
    IO cache manager for FittedResult offload operations.

    Maintains per-model `WeakKeyDictionary` caches that map instances to
    their `FittedIO` handlers. Also tracks the "save state" (which instance
    last performed a save) to enable correct routing during load.

    This class is accessed exclusively through the descriptor protocol
    (`__get__`) and returns a bound `_FittedIOSession` for each access.

    Attributes:
        _io_cache (dict): Mapping of model ID → WeakKeyDictionary of FittedIO instances.
        _save_state (dict): Mapping of model ID → id of the instance that last saved.
    """
    __slots__ = ('_io_cache', '_save_state')

    def __init__(self):
        self._io_cache = {
            m: weakref.WeakKeyDictionary() for m in AIGMODELS
        }
        self._save_state = {
            m: None for m in AIGMODELS
        }

    def __get__(self, instance: "FittedResultBase", owner) -> "_FittedIOSession":
        return _FittedIOSession(self._io_cache, self._save_state, instance._id)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
  
class FittedResultBase:
    """
    Descriptor implementing Virtual Proxy semantics for pipeline result data.

    This is the core mechanism that allows AIDflow to manage memory for
    large intermediate results. When a pipeline step produces a heavy
    DataFrame, it can be offloaded to disk. Subsequent access via normal
    attribute access (e.g., `worker.result`) transparently re-hydrates
    the data from disk.

    Architecture:
        - **Write path** (`__set__`): Stores the result in a per-model
          WeakKeyDictionary cache and registers the IO handler.
        - **Read path** (`__get__`): Checks if data was offloaded and is
          currently None in memory. If so, loads from disk and injects
          back into the target instance's dataclass field.
        - **Offload** (`offload`): Saves data to disk, severs the in-memory
          reference, and triggers garbage collection + Arrow memory pool release.
        - **Clear** (`clear`): Removes offloaded data from disk and cache.

    The routing to the correct target instance (Fitter vs. ModelRunner) is
    delegated to the `_id` attribute, which is a concrete `FittedIDBase`
    subclass defined in the implementation modules.

    Example:
        >>> # Offload all results for a specific model
        >>> FittedResult.offload('AIGFLOW.Data.loader', project_sig)
        >>> 
        >>> # Access later - transparently loaded from disk
        >>> df = worker_instance.result.train_data
    """
    _id: FittedIDBase
    _io: FittedIORegistry

    __slots__ = ('_result_cache')

    def __init__(self):
        self._result_cache = {
            m: weakref.WeakKeyDictionary() for m in AIGMODELS
        }

    def __get__(self, instance, owner: Any = None) -> Any:        
        if instance is None:
            return self

        instance_io = self._io.bind(instance)

        # Not fitted | Cleared | Loaded stored Model
        if not instance_io.is_registred():
            return None
        
        instance_io.load()

        return self.get_result(instance)

    def __set__(self, instance, value: Any) -> None:        
        cache = self.get_model_cache(instance)
        
        if value is None: 
            cache.pop(instance, None)
            return
        
        instance_io = self._io.bind(instance)
        instance_io.register()

        cache[instance] = value
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_model_cache(self, instance) -> weakref.WeakKeyDictionary:
        """
        Retrieve the WeakKeyDictionary cache for the model associated with this instance.        
        """
        ids: FittedIDBase = self._id(instance)
        return self._result_cache[ids.model]
    
    def get_result(self, instance) -> Any | None:
        """
        Retrieve the cached result value for a given instance.

        Parameters:
            instance (Any): The object whose result is being retrieved.

        Returns:
            Any | None: The cached result data, or None if not present.
        """        
        cache = self.get_model_cache(instance)
        return cache.get(instance)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def offload(cls, model: str | list[str], project_signature: str) -> None:
        """
        Offload the fitted result data from RAM to disk.

        Saves the underlying data object (DataFrame or Dataclass containing
        DataFrames) to Arrow/Feather format on disk, then severs the
        in-memory reference and triggers garbage collection.

        This is the primary memory-reclamation mechanism in AIDflow. It is
        called automatically by the pipeline context when offload mode is
        enabled, or can be called explicitly by the user.

        Parameters:
            model (str | list[str]): The model identifier(s) to offload.
                Can be a single model string or a list of models.
            project_signature (str): The project signature used to resolve
                the model instance from the ProjectModelRegistry.

        Side Effects:
            - Data is written to disk (Arrow/Feather format).
            - In-memory DataFrame references are set to None.
            - `gc.collect()` is called to reclaim Python objects.
            - `pa.default_memory_pool().release_unused()` releases
              Arrow's internal memory buffers.
        """
        if isinstance(model, str):
            model = [model]

        for m in model:
            self, instance = cls._get_self(m, project_signature)

            if instance is None:
                continue
            
            val = self.get_result(instance)            
            if val is None:
                continue
            
            instance_io = self._io.bind(instance)
            instance_io.save(val)
    
    @classmethod
    def clear(cls, model: str, project_signature: str) -> None:
        """
        Clear offloaded data for a specific model.

        Removes the disk artifacts and in-memory cache entries associated
        with the given model. Use this to free disk space after results
        are no longer needed.

        Parameters:
            model (str): The model identifier to clear.
            project_signature (str): The project signature for registry lookup.
        """
        self, instance = cls._get_self(model, project_signature)
        if instance is None:
            return
        self._clear(instance)

    def _clear(self, instance) -> None:
        instance_io = self._io.bind(instance)
        if not instance_io.is_registred():
            return
        instance_io.clear()
        cache = self.get_model_cache(instance)
        cache.pop(instance, None)

    @classmethod
    def _get_self(cls, model: str | list[str], project_signature: str) -> tuple["FittedResultBase", Any]:
        """
        Resolve the descriptor and target instance from the registry.

        Parameters:
            model (str): The model identifier to look up.
            project_signature (str): The project signature for registry lookup.

        Returns:
            tuple: (descriptor, target_instance) or (None, None) if not found.
        """
        from ..worker import WorkerBase

        worker: WorkerBase = ProjectWorkerRegistry.get_worker(project_signature, model)
        model_instance = worker.model

        if model_instance is None: # No model registred → init model run
            return [None, None]
        
        ids: FittedIDBase = cls._id(model_instance)
        return ids.get_descriptor(), ids.get_target_instance()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class _FittedIOSession:
    """
    Bound IO session for a single FittedResult instance.

    Encapsulates all save/load/clear operations for one instance within
    a specific model's IO cache. Created fresh on each descriptor access
    via `_FittedIO.__get__`.

    The session holds:
    - A weak reference to the instance (to allow GC).
    - The model ID (for cache isolation).
    - The setup (for path resolution and offload flag).
    - The target type for offload (default: pDF).

    Attributes:
        _type (type): The data type eligible for offload (default: pDF).
    """
    _type = (pDF,)

    def __init__(self, 
        io_cache:   dict[str, weakref.WeakKeyDictionary], 
        save_state: dict[str, int|None],
        id_class:   FittedIDBase
    ):
        self._io_cache   = io_cache
        self._save_state = save_state
        self._id_cls     = id_class

    def bind(self, instance) -> "_FittedIOSession":
        """
        Bind the session to a specific instance, resolving its model and setup.

        After binding, the session is ready for save/load/clear operations
        scoped to this instance's model.

        Parameters:
            instance (Any): The target instance to bind to.

        Returns:
            _FittedIOSession: Self (for method chaining).
        """        
        ids:FittedIDBase = self._id_cls(instance)
        self._instance   = weakref.ref(instance)
        self.model       = ids.model
        self.setup       = ids.setup
        self.is_offload  = ids.is_offload()
        return self

    @property
    def instance(self):
        return self._instance() if self._instance else None

    def _get_io_cache(self) -> weakref.WeakKeyDictionary:
        return self._io_cache[self.model]
    
    def is_registred(self) -> bool:
        """
        Check if the bound instance has been registered (i.e., has been fitted).

        Returns:
            bool: True if the instance exists in the IO cache.
        """        
        cache = self._get_io_cache()
        return self.instance in cache

    def register(self) -> None:
        """
        Register a new FittedIO handler for the bound instance.

        Called during the `__set__` path to initialize the IO handler
        that will manage disk operations for this instance.
        """        
        cache = self._get_io_cache()
        cache[self.instance] = FittedIO(self.setup)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def save(self, val: Any) -> None:
        """
        Save the result data to disk and sever the in-memory reference.

        If the value is a Dataclass, iterates its fields and saves all
        fields whose type matches `_type` (default: pDF) to separate
        Arrow files. If it's a plain DataFrame, saves it directly.

        After saving, sets all matching fields on the in-memory object
        to None, releases Arrow's memory pool, and triggers GC.

        Parameters:
            val (Any): The data to save (pDF, pa.Table, or Dataclass containing these).
        """        
        if not (
            self.is_offload and
            not self._is_none(self.instance)                
        ):
            return

        io = self._get_io(self.instance)
        
        io.save(self.model, val)         
        self._update_save_state(self.instance)

        Dataclass.set_by_type(val, self._type, None)
        pa.default_memory_pool().release_unused()
        gc.collect()

    def load(self) -> Any:
        """
        Re-hydrate offloaded data from disk into the target instance.

        Only performs a disk read if:
        1. Offload mode is enabled.
        2. A save was previously performed for this model.
        3. The current in-memory value is None (was offloaded).

        The loaded data is injected back into the target instance's
        Dataclass fields, then the save state is cleared.

        Returns:
            Any | None: The loaded data, or None if no load was needed/performed.
        """        
        if not (
            self.is_offload  and
            self._is_saved() and
            self._is_none()
        ):
            return
        
        ins = self._get_saved_instance()

        io  = self._get_io(ins)
        res = io.load(self.model)
        if res is None:
            return

        val = self._get_fitted(ins)
        Dataclass.set_by_type(val, self._type, res)
        self.clear()

    def clear(self) -> None:
        """
        Remove offloaded disk artifacts and reset the save state.

        Called after a successful load to prevent redundant re-reads.
        Also callable independently to force-clear disk state.
        """        
        if not self.is_offload:
            return

        ins = self._get_saved_instance()
        if ins is None:
            return
        
        io  = self._get_io(ins)
        io.clear_model(self.model)
        self._update_save_state(ins, clear=True)

    def _get_io(self, instance) -> FittedIO:
        cache = self._get_io_cache()
        io    = cache.get(instance)

        if io is None:
            raise ValueError(
                "AIG SAFETY ► Unexpected Error ► FittedResult io is None "
                " while it musn't be."
            )
        
        return io
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _update_save_state(self, instance, clear:bool=False) -> None:
        self._save_state[self.model] = id(instance) if not clear else None

    def _get_save_state(self) -> tuple[weakref.WeakKeyDictionary, int | None]:
        cache = self._get_io_cache()
        return cache, self._save_state[self.model]

    def _get_saved_instance(self):
        """
        Resolve the instance that performed the last save operation.

        This can be either:
        - The direct instance (ModelRunner), or
        - An indirect instance (Fitter) reached via the ModelRunner.

        Returns:
            Any | None: The instance that holds the saved data.
        """        
        cache, i_id  = self._get_save_state()
        if i_id is None:
            return
        
        # Direct access → _ModelRunner | ModelBase
        if i_id == id(self.instance):
            return self.instance
        
        # Indirect access via _ModelRunner → Fitter
        return [x for x in cache if id(x) == i_id][0]
        
    def _get_fitted(self, instance):
        """
        Resolve the fitted result data object from the target instance.

        Parameters:
            instance (Any): The instance from which to retrieve the result.

        Returns:
            Any: The result data object (Dataclass or DataFrame).
        """        
        ids:FittedIDBase = self._id_cls(instance)
        return ids.get_fitted()
    
    def _is_saved(self) -> bool:
        """
        Check if a save was previously performed for this model.        
        """
        cache, i_id  = self._get_save_state()
        return i_id in [id(x) for x in cache]

    def _is_none(self, instance=None) -> bool:
        """
        Check if all offload-eligible fields on the target instance are None.        
        """
        ins = instance or self._get_saved_instance()        
        val = self._get_fitted(ins)
        
        xt = Dataclass.get_fields_type(val)
        none = []
        for x, t in xt.items():
            if t in self._type:
                none.append(val[x] is None)

        return all(none)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬