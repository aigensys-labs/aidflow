"""
State-Aware I/O Orchestrator for Managed Project Artifacts.

This module implements the persistence layer of the AIDflow
project structure. It maps logical model identifiers to their
physical storage locations and provides specialized I/O handlers
that automate the save/load lifecycle of all pipeline artifacts.

Artifact Categories
-------------------
- **Models** (`ModelIO`): Fitted ML instances (pickled,
  gzip-compressed). The primary training outputs.
- **Metas** (`MetaIO`): Lightweight statistical objects
  (diagnostics, scores, paths). Fast to load, small on disk.
- **Reports** (`ReportIO` / `DashIO`): Dashboard instances
  and visualization artifacts (pickled). Regenerable from models.
- **Features** (`FeatureIO`): The global feature map tracking
  which features were selected/transformed at each stage.
- **Data** (`DataIO`): Raw and structured data files
  (Arrow/CSV/gzip). General-purpose storage.
- **Worker Offload** (`FittedIO`): Intermediate DataFrame
  disk offload for pipeline memory management (Arrow,
  LZ4/zstd compressed).

Project Structure Enforcement
-----------------------------
All paths are resolved through `ProjectPaths`, which enforces
the mandatory AIDflow directory hierarchy:

Design Principle
----------------
**A model is never saved without its required supporting data.**
`ModelIO.save()` coordinates the atomic persistence of the
model instance, its feature list, and its statistics in a single
call — ensuring no partial state is ever left on disk.
"""

from ...deps.common         import  (
                                    Path, 
                                    shutil
                                    )
from ...deps.data           import  pa, pDF, pSe
from ...deps.plots          import  go
from ...deps.types          import  Any, Callable

from ...storage.file        import  (
                                    BaseIO, 
                                    MgzipIO, 
                                    FileIO,
                                    ArrowIO,
                                    FileUtils
                                    )
from ...storage.folder      import  Folder
from ...utils.builtins      import  Str

from ..structure            import  (
                                    SID,
                                    STRUCTURE,
                                    ProjectStructure,
                                    ProjectPaths as PP
                                    )
from ..parameters           import  (
                                    PRM,
                                    Parameters,
                                    AIGFLOW
                                    )
from ..foundation           import  (
                                    AigParam, 
                                    AigModel,
                                    AigStop
                                    )
from ..factory              import  (
                                    Dataclass,
                                    DictDataclass
                                    )
from ..interop              import  (
                                    Dict,
                                    TypeStandard
                                    )
from .temp                  import  Temp


IO = 'gzip'

class FITTED(DictDataclass):
    """Internal attribute key for fitted data identity tracking."""
    ID = 'id'

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PROJECT I/O ORCHESTRATORS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class DefaultPaths:
    """
    Base guard: verifies that project paths are initialized.

    Raises a safety error if the `Parameters` instance does not
    have `project_paths` set (i.e., the flow was not properly
    initialized with a project name and path).
    """
    def __init__(self, setup: Parameters):
        for d in ['project_paths']:
            AigStop.is_flow_init(setup, d)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ProjectPaths:
    """
    State-Aware Path Resolver — Maps Logical IDs to Physical Directories.

    The entry point for resolving the full AIDflow project
    structure on disk. Given a project name (and optional root
    path), constructs the complete directory hierarchy and
    provides access to all artifact storage locations.

    Resolution Logic
    ----------------
    1. Resolves the project root (from `path` or the default
       local scope).
    2. Verifies the project exists by checking for the setup
       file (`_setup/setup.gz`).
    3. If found: exposes `project_paths` with all standard
       structure paths.
    4. If not found: raises (or silently returns empty,
       depending on `is_raise`).

    Parameters
    ----------
    project : str
        The project name under which the AIG flow was stored.
        When no explicit path is provided, resolves within the
        local `2_Outputs` folder scope.
    path : Path | str, default=None
        Explicit root path to the project folder (the folder
        containing `0_Data`, `1_Models`, etc.). Use this
        for external locations outside the package directory.
    is_raise : bool, default=True
        If True, raises a `ValueError` when the project is
        not found. If False, `project_paths` remains None
        (enables soft-lookup patterns).
    err : str
        Error message to show, active only with is_raise=True.

    Attributes
    ----------
    project_paths : ProjectPaths
        The resolved structure containing:
        - `structure`: Complete folder hierarchy.
        - `models`: All model storage paths.
        - `metas`: All metadata storage paths.
        - `reports`: All report storage paths.
        - `root` : Path to the project root folder.
        - `setup`: Path to the project configuration file.
        - `log`: Path to the project log file, if set to 1.
    """   
    project_paths: PP = None

    def __init__(self, 
        project: str, 
        path: Path|str = None, 
        is_raise: bool = True,
        err: str = (
            'AIG WARNING ► "project" and opt. "path" ' 
            'has to be provided to obtain structure paths'
        )
    ):           
        self.is_raise = is_raise
        pp = ProjectStructure(project, predict=True, path=path).project_paths
        
        if Path( MgzipIO.suffix(pp.setup) ).exists():
            self.project_paths = pp
            return
        
        if is_raise:                 
            raise ValueError(err)
    
    def get_setup(self) -> Parameters | None:
        """
        Load and return the project's `Parameters` instance.

        Reads the saved setup file from disk and reconstructs
        the full parameter object.

        Returns
        -------
        Parameters | None
            The project configuration, or None if the project
            structure was not found.

        Raises
        ------
        Exception
            If `is_raise=True` and the setup file cannot be loaded.
        """ 
        if self.project_paths is not None:
            try: 
                return BaseIO.load(self.project_paths.setup, verbose=self.is_raise)
            except Exception as e:
                if self.is_raise: raise e
    
    def get_setup_as_dict(self) -> dict | None:
        """
        Load the project setup as a lightweight dictionary.

        Reads the CSV representation of the setup, bypassing
        full object reconstruction. Faster than `get_setup()`
        when only parameter values (not full objects) are needed.

        Returns
        -------
        dict | None
            The setup as a flat dictionary, or None.
        """
        if self.project_paths is not None:
            try: 
                return FileIO.load_csv_as_dict(
                FileUtils.ensure_extension(self.project_paths.setup, 'csv')
            )
            except Exception as e:
                if self.is_raise: raise e
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DataIO(DefaultPaths):
    """
    General-Purpose Data I/O Handler.

    Handles read/write of raw data and structured files within
    the AIDflow project. Supports Arrow (parquet/feather),
    CSV, and gzip formats.

    Path Resolution
    ---------------
    - **Absolute path**: Used as-is.
    - **Relative path with folder**: Resolved relative to CWD.
    - **Filename only**: Routed to the default project data folder.

    Save Behavior
    -------------
    Respects the global `save` flag in the project setup.
    When `save=False` (dry-run mode), no files are written —
    enabling pipeline testing without disk I/O.
    """  
    DEFAULT_PATH = STRUCTURE[SID.D]

    def __init__(self, setup: Parameters):
        super().__init__(setup)
        self.setup = setup

    @classmethod
    def _get_path(cls, path: str|Path) -> Path:
        if Path(path).exists():
            return path
        path_, file = Folder.path_file(path)
        if not path_: 
            path_ = cls.DEFAULT_PATH
        return Folder.standard( f'{path_}/{file}' )

    def is_save(self) -> bool:
        "True when project's global save flag was set to 1/True."
        return getattr(self.setup, PRM.OUTPUT.save.param, False)

    def save(self, 
        data: Any, 
        path: str|Path, 
        format: str = None, 
        async_io: bool = False, 
        verbose: bool = False
    ):
        """
        Persist data to disk using the project's standard I/O logic.

        Parameters
        ----------
        data : Any
            The data object to save (DataFrame, dict, etc.).
        path : str | Path
            Target path (resolved within the project structure).
        format : str | None, default=None
            Output format (`'gzip'` for pickle, else Arrow).
        async_io : bool, default=False
            Enable async I/O (non-blocking write).
        verbose : bool, default=False
            Enable progress display.
        """
        if path is None or not self.is_save():
            return
        path_ = self._get_path(path)
        if format == IO:
            BaseIO.save(data, path_, async_io=async_io, verbose=verbose)
        else:
            ArrowIO.save(data, path_, format=format)
    
    @classmethod  
    def load(cls,
        path: str|Path, 
        format: str = None, 
        async_io: bool = False, 
        verbose: bool = False):
        """
        Load data from disk into its original object form.

        Parameters
        ----------
        path : str | Path
            Source path (resolved within the project structure).
        format : str | None, default=None
            Source format (`'gzip'` for pickle, else Arrow).
        async_io : bool, default=False
            Enable async I/O.
        verbose : bool, default=False
            Enable progress display.

        Returns
        -------
        Any
            The loaded data object.
        """
        path_ = cls._get_path(path)
        if format == IO:
            return BaseIO.load(path_, async_io=async_io, verbose=verbose)
        return ArrowIO.load(path_, format=format)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class _ModelStandard:
    """
    Internal base for model-scoped I/O operations.

    Provides path resolution, save (with sanitization), and load 
    logic shared by `ModelIO`, `MetaIO`, and `ReportIO`.

    Sanitization Protocol
    ---------------------
    Before saving, certain attributes (`attr_to_sanitize`)
    are detached from the instance to break circular references
    or remove transient state. After the save completes, they
    are restored — ensuring the in-memory object remains
    functional.
    """
    def __init__(self, setup: Parameters):
        self.d_io = DataIO(setup)
        self.path = setup.project_paths.models # ref only → actual paths in a child class

    def _path(self, model, file_name, custom_path):
        if custom_path is not None: 
            return custom_path
        if self.path.get(model) is None:
            return None
        if file_name is None:
            file_name = model
        return self.path[model] / file_name
    
    def _sanitize_save(self, instance, attr_to_sanitize, path, async_io, verbose):
        recon = AigModel.sanitize(instance, attr_to_sanitize) 
        try: 
            self.d_io.save(instance, path, format=IO, async_io=async_io, verbose=verbose)
        finally: 
            AigModel.reconstruct(recon)

    def _save_model(self, 
        model: str, 
        instance: object, 
        file_name: str, 
        custom_path: str|Path, 
        attr_to_sanitize: str|list = None,
        async_io: bool = False,
        verbose: bool = True,        
    ):
        """
        Execute the full save: resolve path → verify → sanitize → write.

        Parameters
        ----------
        model : str
            The model identifier.
        instance : object
            The object to persist.
        file_name : str | None
            Filename (or None for default).
        custom_path : str | Path | None
            Override path.
        attr_to_sanitize : str | list | None
            Attributes to sanitize during save.
        async_io : bool, default=False
            Async I/O flag.
        verbose : bool, default=True
            Verbose flag.
        """        
        path = self._path(model, file_name, custom_path)       
        AigStop.verify_model_path(model, path)
        self._sanitize_save(instance, attr_to_sanitize, path, async_io, verbose)

    def _load_model(self, 
        model: str, 
        file_name: str, 
        custom_path: str|Path, 
        verbose: bool = True
    ):
        """
        Load a model artifact from its standard storage location.

        Parameters
        ----------
        model : str
            The model identifier.
        file_name : str | None
            Filename (or None for default).
        custom_path : str | Path | None
            Override path.
        verbose : bool, default=True
            Verbose flag.

        Returns
        -------
        object
            The loaded instance.
        """        
        path = self._path(model, file_name, custom_path)
        return self.d_io.load(path, format=IO, verbose=verbose)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
#     
class ModelIO(_ModelStandard):
    """
    Model Artifact I/O — Coordinated Save/Load of Training Outputs.

    The primary persistence handler for fitted ML models.
    Ensures that a model is **never saved without its required
    supporting data** (feature list + statistics) by
    coordinating all three in a single atomic operation.

    Save Contract
    -------------
    A single `save()` call persists:
    1. The model instance (gzip pickled)
    2. The feature list (gzip pickled)
    3. The statistics/metadata (gzip pickled)

    Load Contract
    -------------
    `load()` reconstructs only the model instance. Use `MetaIO.load()` 
    for statistics and `FeatureIO.load()` for features independently.
    """  

    def __init__(self, setup: Parameters):
        super().__init__(setup)
        self.meta_io = MetaIO(setup)
        self.fe_io   = FeatureIO(setup)
        self.path    = setup.project_paths.models
            
    def save(self, 
        model: str,             
        fit_instance: object,
        features: list = None, 
        init_fe: bool = False,
        file_name: str = None,
        custom_path: Path|str = None,
        attr_to_sanitize: str|list = None,   # attr to sanitize across Workers
        async_io: bool = False
    ) -> None:
        """
        Execute a coordinated save of model + features + stats.

        Parameters
        ----------
        model : str
            The model identifier (registry key).
        fit_instance : object
            The fitted model instance to persist.
        features : list | None, default=None
            The feature list to save (or None to skip).
        init_fe : bool, default=False
            If True, resets the feature map before saving
            (initial flow step).
        file_name : str | None, default=None
            Override filename.
        custom_path : Path | str | None, default=None
            Override save path.
        attr_to_sanitize : str | list | None, default=None
            Attributes to detach during save (circular ref
            prevention).
        async_io : bool, default=False
            Enable async I/O.
        """
        self._save_model(model, fit_instance, file_name, custom_path, attr_to_sanitize, async_io)            
        
        if features is not None:
            self.fe_io.save({model: features}, init_fe)

        stats = getattr(fit_instance, 'stats', None)
        if stats is not None:
            self.meta_io.save(model, stats, file_name, custom_path, attr_to_sanitize, True)
    
    def load(self, 
        model: str, 
        file_name: str = None,
        custom_path: Path|str=None
    ) -> object:
        """
        Reconstruct a fitted model instance from disk.

        Parameters
        ----------
        model : str
            The model identifier.
        file_name : str | None, default=None
            Override filename.
        custom_path : Path | str | None, default=None
            Override load path.

        Returns
        -------
        object
            The loaded fitted model instance.
        """ 
        return self._load_model(model, file_name, custom_path)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ReportIO(_ModelStandard):
    """
    Report Artifact I/O — Dashboard & Visualization Persistence.

    Handles save/load of dashboard instances (pickled). Reports 
    are regenerable from models — they are saved for archival 
    and audit convenience, not as the primary source of truth.
    """
    def __init__(self, setup: Parameters):
        super().__init__(setup)
        self.path = setup.project_paths.reports
            
    def save(self, 
        model: str,    
        dash_instance: object,
        file_name: str = None,
        custom_path: Path|str = None,
        attr_to_sanitize: str|list = None,
        async_io: bool = False
    ) -> None:
        """
        Persist a dashboard instance to the reports directory.

        Parameters
        ----------
        model : str
            The model identifier.
        dash_instance : object
            The dashboard instance to save.
        file_name : str | None, default=None
            Override filename.
        custom_path : Path | str | None, default=None
            Override save path.
        attr_to_sanitize : str | list | None, default=None
            Attributes to sanitize during save.
        async_io : bool, default=False
            Enable async I/O.
        """
        self._save_model(model, dash_instance, file_name, custom_path, attr_to_sanitize, async_io)            
    
    def load(self, 
        model: str, 
        file_name: str = None,
        custom_path: Path|str = None
    ) -> object:
        """
        Load a previously saved report for viewing or auditing.

        Parameters
        ----------
        model : str
            The model identifier.
        file_name : str | None, default=None
            Override filename.
        custom_path : Path | str | None, default=None
            Override load path.

        Returns
        -------
        object
            The loaded dashboard instance.
        """ 
        return self._load_model(model, file_name, custom_path)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class MetaIO(_ModelStandard):
    """
    Metadata I/O — Statistical & Diagnostic Persistence.

    Handles save/load of lightweight statistical objects
    (diagnostics, scores, optimization paths). Optimized 
    for fast loading — these objects are small and 
    frequently accessed during comparison and reporting.
    """

    def __init__(self, setup: Parameters):
        super().__init__(setup)
        self.path = setup.project_paths.metas
            
    def save(self, 
        model: str,    
        stats: object,
        file_name: str = None,
        custom_path: Path|str = None,
        attr_to_sanitize: str|list = None,
        async_io: bool = True
    ) -> None:
        """
        Persist statistical metadata to the metas directory.

        Parameters
        ----------
        model : str
            The model identifier.
        stats : object
            The statistical object to save.
        file_name : str | None, default=None
            Override filename.
        custom_path : Path | str | None, default=None
            Override save path.
        attr_to_sanitize : str | list | None, default=None
            Attributes to sanitize during save.
        async_io : bool, default=True
            Async I/O (default True — stats saves are
            non-critical and benefit from non-blocking).
        """
        self._save_model(model, stats, file_name, custom_path, attr_to_sanitize, async_io)            
    
    def load(self, 
        model: str, 
        file_name: str = None,
        custom_path: Path|str=None
    )-> object:
        """
        Load statistical metadata for a model.

        Parameters
        ----------
        model : str
            The model identifier.
        file_name : str | None, default=None
            Override filename.
        custom_path : Path | str | None, default=None
            Override load path.

        Returns
        -------
        object
            The loaded statistical object.
        """    
        return self._load_model(model, file_name, custom_path, False)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeatureIO:
    """
    Feature Map Manager — Global Feature Lineage Tracking.

    Maintains the project-level feature map: a dictionary
    recording which features were selected/transformed at each
    pipeline stage. This ensures that:

    - Downstream stages can reference upstream feature decisions.
    - The final feature set is fully auditable (which stage
      selected what, and why).
    - Inference applies the identical feature subset used
      during training.

    Update Semantics
    ----------------
    - `init_fe=True`: Resets the map (first stage in the flow).
    - `init_fe=False`: Merges new entries into the existing map.
    """

    def __init__(self, setup: Parameters):
        self.d_io = DataIO(setup)
        features  = AIGFLOW.Schema.features
        self.path = setup.project_paths.models[features] / features

    def save(self, model_features:dict, i:bool=False) -> None:
        """
        Update the global feature map with new findings.

        Parameters
        ----------
        model_features : dict
            The model → features mapping to store
            (e.g., `{'agent1': ['feat_a_WOE', 'feat_b_WOE']}`).
        i : bool, default=False
            If True, resets the map before saving (initial
            flow step — clears any previously stored features).
        """
        if not self.d_io.is_save():
            return
        mfes = self.load(i)
        mfes.update(model_features)
        self.d_io.save(mfes, self.path, format=IO)
    
    def load(self, i:bool=False) -> dict:
        """
        Retrieve the current feature map.

        Parameters
        ----------
        i : bool, default=False
            If True, returns an empty dict (reset mode —
            used before the first save in a new flow).

        Returns
        -------
        dict
            The current feature map (model → features).
        """
        if i:  mfes = {}  # safety → i (initial) of a PROJECT - saved features deletion  
        else:  mfes = self.d_io.load(self.path, format=IO)
        return mfes
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class _DashSignature:
    """
    Dashboard Content Signature — Element Validity Verification.

    Stores a content signature (hash/identifier) alongside the
    saved dashboard key attributes. Used to verify that previously 
    saved elements still match the current dashboard state —
    preventing stale element display when the dashboard code
    or configuration has changed.

    Protocol
    --------
    1. **save()**: Called after a successful dashboard save.
       Persists the current signature to disk.
    2. **is_valid_signature()**: Called before attempting to
       load saved elements. Returns True only if the stored
       signature matches the current dashboard's signature.
    
    Parameters
    ----------
    dash_instance: DashInterface
        The dashboard instance to verify.
    """
    SIGNATURE = 'report_signature'

    def __init__(self, dash_instance: type):
        self.set_signature_attrs(dash_instance)
    
    @property
    def MODEL(self) -> str:
        return self.dash.MODEL
    
    @property
    def setup(self) -> Parameters:
        return self.dash.setup

    @property
    def signature(self) -> str:
        return self.dash.dashboard_signature 

    def set_signature_attrs(self, dash_instance: type) -> None:
        """Extract the content signature from the dashboard instance."""
        from ..display import DashInterface
        self.dash: DashInterface = dash_instance        
        self.elements = self.dash._get_elements()
        self.set_signature_path()

    def set_signature_path(self) -> None:
        path = self.get_report_path()
        self.sig_path = path / self.SIGNATURE

    def get_report_path(self) -> Path:
        """Returns model report path under active project."""
        return self._get_report_path(self.setup, self.MODEL)
    
    @staticmethod
    def _get_report_path(setup: Parameters, model: str) -> Path:
        """Returns model report path under active project."""
        return setup.project_paths.reports[model]
        
    def load(self) -> str | None:
        """
        Load the stored signature from disk.

        Returns
        -------
        str | None
            The previously saved signature, or None if the
            file does not exist.
        """
        path = FileUtils.ensure_extension(self.sig_path, MgzipIO.EXT)
        if not Path(path).exists():
            return
        return BaseIO.load(self.sig_path, verbose=False)

    def save(self) -> None:
        """Persist the current signature to disk."""
        BaseIO.save(self.signature, self.sig_path)

    def is_save(self, save:bool|None) -> bool:
        """
        Returns True if either directly provided save or global setup.save is True.
        """        
        return self.dash._is_save(save)
    
    def is_valid_signature(self) -> bool:
        """
        Verify that the stored signature matches the current dashboard.

        Returns
        -------
        bool
            True if the saved signature equals the current
            dashboard signature (elements are valid for
            loading). False if they differ or no signature
            was stored.
        """        
        sig = self.load()
        return sig == self.signature

class _DashPaths:
    """
    Dashboard Path Resolver — Recursive Element Path Construction.

    Extracted from `DashIO` as a dedicated path resolution
    helper. Handles:

    - **Recursive path building**: Supports nested element
      structures (dict of dicts) at arbitrary depth.
    - **Load validity check**: Verifies all elements (including
      nested) have valid signatures and exist on disk.
    - **Temp path generation**: Collision-safe unique paths
      for elements not yet saved.
    - **File type/suffix transformation**: Uniform path
      normalization with extension enforcement.

    Element Structure
    -----------------
    A `dash_elements` dict can be:
    - **Flat**: `{'tab1': 'plot1.html', 'tab2': 'table.csv'}`
    - **Nested**: `{'tab1': {'sub_a': 'a.html', 'sub_b': 'b.html'}, 'tab2': 'c.html'}`

    Both are handled uniformly by the recursive `create_paths`
    method — the output structure mirrors the input structure.

    Parameters
    ----------
    sign : _DashSignature
        An instance of _DashSignature.
    file_type : str | None, default='html'
        File type extension.        
    """
    IS_LOAD   = '_is_active_load'

    def __init__(self, sign: _DashSignature, file_type: str | None = 'html'):
        self.sign = sign
        self.file_type = file_type

    @classmethod
    def create_paths(cls,
        path_fx:  Callable[[str], Path],
        elements: dict[str, str | dict[str, str]],
        _paths:   dict | None = None
    ) -> dict[str, Path | dict[str, Path]]:
        """
        Recursively build a paths dict from a path factory and element structure.

        Traverses the `dash_elements` dict (which may be
        flat or nested) and applies `path_fx` to each leaf
        value, preserving the structure in the output.

        Parameters
        ----------
        path_fx : Callable[[str], Path]
            A function that transforms a leaf value (suffix
            or filename) into a full `Path`. E.g.,
            `lambda s: report_dir / s` or
            `lambda s: temp.get_path(prefix=s)`.
        elements : dict[str, str | dict[str, str]]
            The element structure (flat or nested). Values
            are leaf strings; keys can have dict values for
            nesting.
        _paths : dict, default={}
            Accumulator (internal use — recursive).

        Returns
        -------
        dict[str, Path | dict[str, Path]]
            The resolved paths dict, mirroring the input
            structure.
        """
        if _paths is None:
            _paths = {}

        for k,v in elements.items():
            if isinstance(v, dict):
                _paths[k] = {}
                cls.create_paths(path_fx, v, _paths[k])
            else:
                _paths[k] = path_fx(v)
        return _paths
        
    def _is_load(self, paths: dict[str, Path | dict[str, Path]]) -> bool:
        """
        Verify whether all elements (including nested) are valid and loadable.
        """
        if any(e not in paths for e in self.sign.elements):
            return False
        if not self.sign.is_valid_signature():
            return False
        p = lambda x: all(
            Path(p).exists() for p in Dict.get_flattened_values(x)
        )
        return all(p(paths[e]) for e in self.sign.elements)

    def _get_tmp_paths(self) -> dict[str, Path | dict[str, Path]]:
        """
        Generate unique temp paths for all elements (fresh mode).
        """
        tp = Temp(self.sign.setup)
        px = self.path_standard()
        fx = lambda k: px( tp.get_path(prefix = k) )
        return self.create_paths(fx, self.sign.elements)

    def get_view_paths(self, save_paths: dict[str, Path | dict[str, Path]]
    ) -> dict[str, Path]:
        """
        Resolve element paths — load mode or temp mode.

        Decision logic:
        - If all elements are valid (signature + files exist)
          → return the saved report paths with the `IS_LOAD` flag.
        - Otherwise → generate fresh temp paths (same structure).

        Parameters
        ----------
        save_paths : dict[str, Path | dict[str, Path]]
            The resolved report paths (nested supported).

        Returns
        -------
        dict[str, Path | dict[str, Path]]
            Element → path. May include the `IS_LOAD` key
            (value True) when in load mode.
        """
        # A) Load saved elements under dash signature
        if self._is_load(save_paths):
            return {self.IS_LOAD: True} | {
                k: save_paths[k] for k in self.sign.elements
            }
        
        # B) Elements don`t exist - create temp path
        return self._get_tmp_paths()

    def path_standard(self) -> Callable[[str], Path]:
        """
        Create a path transformation function.

        Returns
        -------
        Callable[[str], Path]
            A function: `str → Path`, ready to be used as
            `path_fx` in `create_paths`.
        """        
        if not self.file_type:
            return lambda x: Path(x)
        return lambda x: Path( FileUtils.ensure_extension(x, self.file_type) )        
    
    def file_suffix(self, file_suffix: str|None) -> Callable[[str], Path]:
        """
        Create a path transformation function with type + suffix.

        Parameters
        ----------
        file_type : str | None, default=None
            The file extension to enforce (e.g., `'html'`).
            If None, no extension is added.
        file_suffix : str | None, default=None
            An additional suffix to append (e.g., `'v2'`).
            Added before the file extension. If None, no suffix is added.

        Returns
        -------
        Callable[[str], Path]
            A function: `str → Path`, ready to be used as
            `path_fx` in `create_paths`.
        """
        path_fx = self.path_standard()
        if file_suffix is None:
            return path_fx
        return lambda x: path_fx( str(x) + Str.add_prefix(file_suffix) )

    def _get_path(self, custom_path:str=None) -> Path:
        path = self.sign.get_report_path()
        path = AigParam.di(custom_path, path)
        AigStop.verify_model_path(self.sign.MODEL, path)
        return Path(path)
        
    def get_save_paths(self,               
        custom_path: str | None = None, 
        file_suffix: str | None = None,
    ) -> dict[str, Path | dict[str, Path]]:
        """
        Resolve the file paths for a set of dashboard elements.

        Parameters
        ----------
        path : str | None, default=None
            Override base path.
        file_type : str | None, default='html'
            File type extension.            
        file_suffix : str | None, default=None
            Additional suffix appended to all element paths.

        Returns
        -------
        dict[str, Path]
            Element name → resolved file path.
        """
        path = self._get_path(custom_path)
        sx = self.file_suffix(file_suffix)
        fx = lambda v: sx(path / v)        
        return self.create_paths(fx, self.sign.elements)    
    
class DashIO:
    """
    Dashboard I/O — Multi-Element Visualization Export & Validation.

    Extends `ReportIO` with support for dashboards containing
    multiple visual elements (tabs, panels, sub-views) and a
    **safety validation protocol** that prevents displaying
    stale elements.

    Element Validity Protocol
    -------------------------
    Before returning saved element paths, `DashIO` verifies:

    1. All requested elements exist in the resolved paths.
    2. The dashboard's content signature matches the stored
       signature (the dashboard hasn't changed since the
       elements were saved).
    3. All element files physically exist on disk.

    If **any** check fails, unique temp paths are generated
    instead — ensuring the caller always gets valid,
    renderable paths.

    Path Resolution Modes
    ---------------------
    - **Load mode**: Saved elements are valid → returns the
      original report paths. The returned dict includes the
      IS_LOAD flag (True) - use `is_load()` method to check.
    - **Temp mode**: Elements are invalid/missing → returns
      unique temp paths. No IS_LOAD flag.

    Save Behavior
    -------------
    - `save=True`: Persists the dashboard instance +
      signature to the reports directory. Returns resolved
      report paths.
    - `save=False`: Does not write. Checks element validity
      and returns either saved paths (load mode) or temp
      paths (fresh mode).
    """

    def __init__(self, setup: Parameters):                
        self.setup = setup
        self.io = ReportIO(setup)
        
    def get_report_path(self, model:str) -> Path:
        """Returns model report path under active project."""
        return _DashSignature._get_report_path(self.setup, model)

    def get_custom_selection_path(self, model:str) -> Path:
        """
        Returns model report path of `user custom selection` io ops
        under active project.
        """
        return self.get_report_path(model) / 'custom_selection'
    
    def save(self, 
        dash_instance: type, 
        save: bool = None, 
        custom_path: Path | str = None,
        report_name: str = None,
        file_type: str | None = 'html',
        file_suffix: str = None,
        attr_to_sanitize: list | str = None,
        async_io: bool = False,
        save_signature : bool = False
    ) -> dict[str, Path | dict[str, Path]]:
        """
        Save the dashboard and resolve element paths with validity check.

        **Save mode** (`save=True`):
        1. Persists the dashboard instance to the reports directory.
        2. Saves the content signature.
        3. Returns the resolved report paths.

        **No-save mode** (`save=False`):
        1. Checks if saved elements are valid (signature + files exist).
        2. If valid → returns saved report paths (load mode).
        3. If invalid → returns fresh temp paths.

        Parameters
        ----------
        dash_instance : DashInterface
            The dashboard instance to persist.
        save : bool | None, default=None
            Force save on/off. If None, uses the project
            setup's save flag.
        custom_path : Path | str | None, default=None
            Override save path.
        report_name : str, default=None
            Optional file name for repor instance.
            None resolves to AIG model ID.
        file_type : str | None, default='html'
            File type extension.
        file_suffix : str | None, default=None
            Additional filename suffix.
        attr_to_sanitize : list | str | None, default=None
            Attributes to sanitize during save.
        async_io : bool, default=False
            Enable async I/O.
        save_signature : bool, default = False
            Optionally saves report signature to an extra file
            for fast verification of loadable elements.

        Returns
        -------
        dict[str, Path | dict[str, Path]]
            Element name → resolved path (or temp path if save=False).
            Structure follows `dash_elements structure`.
        """
        sign  = _DashSignature(dash_instance)
        paths = _DashPaths(sign, file_type)

        save_paths = paths.get_save_paths(custom_path, file_suffix)

        if not sign.is_save(save):
            return paths.get_view_paths(save_paths)
                
        self.io.save(
            sign.MODEL, dash_instance, 
            custom_path = custom_path, file_name = report_name, 
            attr_to_sanitize = attr_to_sanitize, async_io = async_io
        )
        if save_signature:
            sign.save()
        
        return save_paths
    
    def save_file(self,
        obj: Any, 
        path: Path | str, 
        async_io: bool = True, 
        verbose: bool = True
    ) -> None:
        """Dashboard `BaseIO save` utility for extra files."""
        DataIO(self.setup).save(
            obj, path=path, format=IO, async_io=async_io, verbose=verbose
        )

    def load(self, 
        model: str = None, 
        file_name: str = None, 
        custom_path: Path|str = None
    ) -> Any | None:
        """
        Load a previously saved dashboard artifact.

        Parameters
        ----------
        model : str | None, default=None
            The model identifier.
        file_name : str | None, default=None
            Override filename.
        custom_path : Path | str | None, default=None
            Override load path.

        Returns
        -------
        object | None
            The loaded dashboard instance, or None if the
            file does not exist (prevents execution crashes).
        """
        if custom_path is None:
            return
        path = FileUtils.ensure_extension(custom_path, MgzipIO.EXT)
        if not Path(path).exists():
            return 
        return self.io.load(model, file_name=file_name, custom_path=path)

    @staticmethod
    def is_load(paths: dict[str, Path | dict[str, Path]]) -> bool:
        """
        Check whether a paths dict is in 'load' mode.

        Parameters
        ----------
        paths : dict[str, Path]
            The paths dict returned by `save()` or `_get_view_paths()`.

        Returns
        -------
        bool
            True if the dict contains the `IS_LOAD` flag
            (elements were previously saved and are valid for loading).
        """
        return _DashPaths.IS_LOAD in paths

    @classmethod
    def resolve_fig_path(cls, 
        fig_fx: Callable[[], go.Figure], 
        paths: dict[str, Path | dict[str, Path]], 
        element: str
    ) -> tuple[go.Figure | Path | dict[str, Path], Path | dict[str, Path]]:
        """
        Resolve the figure object and its save path for a dashboard element.

        Dual-mode resolution based on the paths dict state:

        - **Load mode** (elements previously saved + signature valid): 
          Returns the saved HTML path for both returned objects.
        - **Save / Temp mode** (elements not yet saved): Returns the
          live Plotly Figure (produced by `fig_fx()`) and the path
          where it will be persisted.

        Parameters
        ----------
        fig_fx : Callable
            `Only invoked in save mode`.
            A zero-argument callable that produces the Plotly Figure e.g.:
            - lambda: get_figure() → fig to save
        paths : dict[str, Path | dict[str, Path]]
            The paths dict from `save()`.
            Includes the `IS_LOAD` flag when in the Load mode.
        element : str
            The element name to resolve.

        Returns
        -------
        tuple[go.Figure | (Path | dict[str, Path]), Path | dict[str, Path]]
            - In load mode: `(report_path, report_path)` —
              the saved HTML file serves as both the figure and the path.
            - In save mode: `(fig, report_path)` — the fig_fx() output, 
              and the path for persistence.

        Raises
        ------
        ValueError
            If `element` is not present in `paths`.
        """        
        report_path = cls._resolve_element(element, paths)
        if cls.is_load(paths):
            return report_path, report_path
        return fig_fx(), report_path

    @classmethod
    def resolve_path(cls, 
        paths: dict[str, Path | dict[str, Path]], 
        element: str
    ) -> tuple[bool, Path | dict[str, Path]]:
        """
        Resolve dash state `LOAD / SAVE` and the load / save path for 
        a dashboard element.

        Dual-mode resolution based on the paths dict state:

        - **Load mode** (elements previously saved + signature valid): 
          Returns the `(True, report_path)` - can be safely loaded.
        - **Save / Temp mode** (elements not yet saved): Returns the
          `(False, report_path)` - path where it will be persisted.

        Parameters
        ----------
        paths : dict[str, Path | dict[str, Path]]
            The paths dict from `save()`.
            Includes the `IS_LOAD` flag when in the Load mode.
        element : str
            The element name to resolve.

        Returns
        -------
        tuple[bool, Path | dict[str, Path]]
            - In load mode: `(True, report_path)`.
            - In save mode: `(False, report_path)`.

        Raises
        ------
        ValueError
            If `element` is not present in `paths`.
        """        
        report_path = cls._resolve_element(element, paths)
        if cls.is_load(paths):
            return True, report_path
        return False, report_path
        
    def _resolve_element(element: str, paths: dict[str, Path | dict]) -> Path | dict:
        if element in paths:
            return paths[element]
        raise ValueError(
            'AIG SAFETY ► Dash element is not present in paths.'
        )        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FittedIO:
    """
    Worker Output Disk Offload — Pipeline Memory Management.

    Manages the disk offload/rehydration of intermediate worker 
    outputs (DataFrames) to prevent memory pressure during long 
    pipeline executions. Each instance is scoped to a specific 
    project and stores data under `project_root/_offload/`.

    Use Case
    --------
    When a pipeline step produces a large intermediate DataFrame
    that is only needed by the next step (not persisted long-term),
    `FittedIO` offloads it to disk (Arrow format, compressed)
    and rehydrates it on demand. This keeps peak RAM usage
    bounded regardless of intermediate data size.

    Data Format
    -----------
    - **Serialization**: Apache Arrow (feather v2).
    - **Compression**: zstd (default) — fast with good ratio for tabular 
      data. Configurable per save.
    - **Categorical handling**: Category codes are flattened to integers 
      on save and restored on load (preserving the original category mapping 
      via an internal schema cache).

    File Naming
    -----------
    Files are named `{model_id}_{index}.arrow` under `_offload/`. 
    The index increments for multi-field Dataclass saves 
    (each DataFrame field gets its own file).

    Lifecycle
    ---------
    1. **save**: Offload data to disk (resets internal path registry).
    2. **load**: Rehydrate from disk (returns DataFrame or list of DataFrames).
    3. **clear_model**: Remove specific model's files after successful rehydration.
    4. **clear**: Remove the entire `_offload` directory (project reset).

    Parameters
    ----------
    setup : Parameters
        The project parameter context (provides the root path).
    """
    FOLDER = '_offload'

    def __init__(self, setup:Parameters):
        self._path  = self._get_tmp_path(setup)
        self._saved_path: dict[str, str] = {}
        self._schema: dict[str|None, dict[str, dict]] = {None: {}}        

    def _get_tmp_path(self, setup:Parameters) -> Path:
        return setup.project_paths.root / self.FOLDER
    
    @staticmethod
    def _suffix(path) -> str:
        return FileUtils.ensure_extension(path, 'arrow')
    
    def _get_path(self, model:str, is_save:bool=False) -> str:
        """Generate the next available file path for a model."""
        i = len(self._saved_path)
        p = lambda i:self._suffix(self._path / f'{model}_{i}')
        if not is_save:
            return p(i)
        while Path(p(i)).exists():
            i+=1
        return p(i)

    def _standardize(self, data:pDF, id_:int, is_save:bool=False) -> pDF:
        """
        Handle categorical dtype serialization/deserialization.

        On save: Flattens category codes to integers and caches
        the mapping in `self._schema`.

        On load: Restores category codes from the cached mapping.        
        """
        if not is_save:
            if id_ not in self._schema:
                return data
            
            for x, s in self._schema[id_].items():
                dc: pSe = TypeStandard.any_to_cat(data[x])
                data[x] = dc.cat.rename_categories({i:c for c,i in s.items()})
            return data
        
        cat = TypeStandard.get_dtypes_cat(data).index.to_list()
        
        if not cat:
            return data
        
        self._schema[id_] = {}
        s = self._schema[id_]

        d = {}
        for x in cat:
            dc: pSe = data[x]
            s[x] = {c:i for i,c in enumerate(dc.cat.categories)}        
            d[x] = dc.cat.rename_categories(s[x])

        d = pDF(d)

        col = data.columns
        res = [x for x in col if x not in cat]
        d[res] = data[res]

        return d[col]
    
    def _save(self, data: pDF | pa.Table, path: str, compression: str):
        id_ = id(data)
        if isinstance(data, pDF):
            id_  = id_ if (FITTED.ID not in data.attrs) else data.attrs[FITTED.ID]
            data = self._standardize(data, id_, True)
            data = pa.Table.from_pandas(data, preserve_index = True)
        ArrowIO.to_feather(data, path, compression = compression, compression_level = 3)
        self._saved_path[id_] = path

    def save(self, 
        model: str,
        data:  DictDataclass | pDF | pa.Table,
        offload_type: type = pDF,
        compression:  str  = "zstd"
    ) -> None:
        """
        Offload intermediate worker output to disk.

        If the data is a plain DataFrame/Arrow Table, saves it
        as a single file. If it's a Dataclass, iterates its
        typed fields and saves all fields matching
        `offload_type` as separate Arrow files.

        Parameters
        ----------
        model : str
            The model identifier (used for path generation).
        data : DictDataclass | pDF | pa.Table
            The data to offload.
        offload_type : type, default=pDF
            The field type to target for offloading (when
            data is a Dataclass).
        compression : str, default='zstd'
            Compression algorithm (`'zstd'`, `'lz4'`,
            `'uncompressed'`).

        Side Effects
        ------------
        - Creates `.arrow` files under `_offload/`.
        - Resets the internal path registry (fresh save).
        """
        self._saved_path = {}
        
        save = lambda d: self._save(
            d, self._get_path(model, True), compression
        )
        if not Dataclass.is_dataclass(data):            
            save(data)
            return

        schema = Dataclass.get_fields_type(data)

        for x, t in schema.items():
            if issubclass(t, offload_type):
                save(data[x])

    def _load(self, path: str, id_: int = None) -> pDF | None:
        if not Path(path).exists():
            return
        table = ArrowIO.read_feather(path)
        data  = table.to_pandas()
        if id_:        
            data.attrs[FITTED.ID] = id_
        return self._standardize(data, id_)
    
    def load(self, model: str) -> pDF | list[pDF] | None:
        """
        Rehydrate worker output from disk.

        Parameters
        ----------
        model : str
            The model identifier used for path resolution.

        Returns
        -------
        pDF | list[pDF] | None
            - Single DataFrame (if one file was saved).
            - List of DataFrames (if a Dataclass with multiple
              fields was saved).
            - None (if no files were found).
        """
        if not self._saved_path:
            path = self._get_path(model)
            return self._load(path)

        res = []
        for i,p in self._saved_path.items():
            res.append( self._load(p, i) )

        if all(x is None for x in res):
            return
        
        return res

    def clear_model(self, model: str) -> None:
        """
        Remove a specific model's offloaded files from disk.

        Called after a successful rehydration to prevent stale
        data accumulation.

        Parameters
        ----------
        model : str
            The model identifier whose files should be removed.
        """
        paths = self._saved_path or {0: self._get_path(model)}
        for p in paths.values():
            Path(p).unlink(missing_ok=True)

    def clear(self) -> None:
        """
        Remove the entire project's `_offload` directory.

        Performs a recursive delete of all offloaded data.
        Use with caution — this removes all offloaded results
        across all models for this project.

        Side Effects
        ------------
        - Recursively deletes `_offload/` and all contents.
        - Silently ignores errors (directory already removed).
        """
        shutil.rmtree(self._path, ignore_errors=True)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬