"""
Ephemeral State & Temporary Filesystem Orchestration.

This module provides a dual-purpose system for handling short-lived
data within the AIDflow ecosystem.

In-Memory Registry (TempRegistry)
---------------------------------
A decoupled global state service (GSS) that enables communication
between disparate pipeline layers. Objects (file paths, references,
identifiers) are registered under a project signature and retrieved
by downstream consumers — without requiring direct references or
circular imports between layers.

Use case: A fitter registers a report path during `fit()`;
a later display layer retrieves it via the project signature
without knowing which fitter produced it.

Physical Temp Storage (Temp)
----------------------------
Manages a dedicated scratchpad directory (`_previews`) within
the project output folder for short-lived artifacts:

- Static HTML overviews (agent performance reports).
- Intermediate visualization buffers.
- Ephemeral report fragments (generated on-demand, not part of
  the permanent project structure).

Files are named with unique paths (timestamp + UUID) to prevent
collisions between concurrent processes. The entire temp folder
is cleared on demand (`clear()`) to keep the project output
structure clean.

Design Principle
----------------
Temporary artifacts live in `_previews`. This ensures that:
- The project structure remains stable and auditable.
- Temporary reports can be regenerated at any time.
- Cleanup is a single atomic operation (remove the folder).
"""

from ...deps.common     import shutil, Path, threading
from ...deps.types      import Any

from ...storage.folder  import Folder
from ...storage.file    import FileUtils
from ...utils.builtins  import Str, Misc

from ..structure        import SID, STRUCTURE


class TempRegistry:
    """
    In-Memory Global State Service (GSS) — Temporal Object Registry.

    Enables decoupled communication between AIDflow pipeline layers
    by allowing objects to be registered and retrieved via project
    signatures. Acts as a transient key-value store scoped to a
    single project execution.

    Storage Layout
    --------------
    `{project_signature: {obj_id: value}}`

    Two levels of namespacing ensure that objects from different
    pipeline stages (or different projects running in the same
    process) never collide.

    Lifecycle
    ---------
    - **register**: Store a value (typically a file path or
      reference) under a signature + ID pair.
    - **get**: Retrieve without removing (read-only access).
    - **pop**: Retrieve and remove in one operation (consume-once
      semantics).
    - **erase**: Remove without retrieving (cleanup).

    Examples
    --------
    >>> # Producer (fitter layer):
    >>> TempRegistry.register('project_sig', 'report_path', '/tmp/report.html')
    >>>
    >>> # Consumer (display layer):
    >>> path = TempRegistry.get('project_sig', 'report_path')
    """
    _storage: dict[str, dict[str, str]] = {}
    _lock  = threading.Lock()

    @classmethod
    def register(cls, project_sign: str, obj_id: str, temp_object: Any) -> None:
        """
        Register a temporal object under the project signature.

        Parameters
        ----------
        project_sign : str
            The project signature (unique project identifier).
        obj_id : str
            The object identifier within this project
            (e.g., `'report_path'`, `'agent_1_overview'`).
        temp_object : Any
            The value to store (typically a path string or reference).
        """       
        with cls._lock:
            if project_sign not in cls._storage:
                cls._storage[project_sign] = {}
            
            cls._storage[project_sign][obj_id] = temp_object

    @classmethod
    def get(cls, project_sign: str, obj_id: str) -> Any | None:
        """
        Retrieve a temporal object (read-only, does not remove).

        Parameters
        ----------
        project_sign : str
            The project signature.
        obj_id : str
            The object identifier.

        Returns
        -------
        Any | None
            The stored value, or None if not found.
        """
        with cls._lock:
            project_bucket = cls._storage.get(project_sign)
            if project_bucket:
                return project_bucket.get(obj_id)
            return None

    @classmethod
    def pop(cls, project_sign: str, obj_id: str) -> Any | None:
        """
        Retrieve and remove a temporal object (consume-once).

        Parameters
        ----------
        project_sign : str
            The project signature.
        obj_id : str
            The object identifier.

        Returns
        -------
        Any | None
            The stored value (now removed), or None if not found.
        """
        with cls._lock:
            project_bucket = cls._storage.get(project_sign)
            if project_bucket:
                return project_bucket.pop(obj_id, None)
            return None

    @classmethod
    def erase(cls, project_sign: str, obj_id: str) -> None:
        """
        Remove a temporal object from the registry.

        Parameters
        ----------
        project_sign : str
            The project signature.
        obj_id : str
            The object identifier to remove.
        """
        with cls._lock:
            project_bucket = cls._storage.get(project_sign)
            if project_bucket:
                project_bucket.pop(obj_id, None)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Temp:
    """
    Temporary Filesystem Manager — Project Scratchpad Directory.

    Manages the `_previews` directory within the project output
    folder. Creates unique file paths for short-lived artifacts
    (HTML reports, visualization buffers) and provides cleanup
    utilities.

    When to Use
    -----------
    Temporary files are created **only when reporting is
    requested**. Reports can be regenerated at any time from
    stored models via:

    `ctx.report_io.load(model).show()`

    Naming Convention
    -----------------
    Each file is named with a unique combination of:
    - **prefix**: Human-readable identifier (e.g., `'Agent_1_Overview'`).
    - **timestamp + UUID**: Collision-safe uniqueness (`1712345678_a1b2c3d4`).
    - **suffix** (optional): Additional qualifier.
    - **extension**: File type (e.g., `'.html'`).

    Example path:
    `project_root/_previews/Agent_1_Overview/1712345678_a1b2c3d4.html`

    Cleanup
    -------
    - `clear()`: Remove the project-specific temp folder.
    - `clear_root()`: Remove the default (shared) temp folder.

    Parameters
    ----------
    setup : Parameters
        The project's parameter context (provides the root path).
    """

    FOLDER  = '_previews'
    DEFAULT = Folder.standard(STRUCTURE[SID.O]) / FOLDER

    def __init__(self, setup: type):
        self._path = self._get_tmp_path(setup)

    @classmethod
    def _get_tmp_path(cls, setup: type) -> Path:
        from ..parameters import Parameters
        s: Parameters = setup
        return s.project_paths.root / cls.FOLDER
    
    @classmethod
    def _get_gen_path(cls, signature_path: str | Path | None) -> Path | None:
        if isinstance(signature_path, Path):
            return signature_path
        if not signature_path:
            return cls.DEFAULT
        from ..registry import ProjectModelRegistry
        setup = ProjectModelRegistry.get_setup(signature_path)
        return cls._get_tmp_path(setup)
    
    @classmethod
    def gen_unique_path(cls, 
        prefix: str = None, 
        suffix: str = None, 
        file_type: str = None
    ) -> str | Path:
        """
        Generate a unique filename (without folder).

        Combines prefix, timestamp+UUID, suffix, and extension
        into a collision-safe filename.

        Parameters
        ----------
        prefix : str | None, default=None
            Human-readable identifier (e.g., `'Agent_1_Overview'`).
        suffix : str | None, default=None
            Additional qualifier appended after the UUID.
        file_type : str | None, default=None
            File extension (e.g., `'html'`, `'json'`).

        Returns
        -------
        str
            The unique filename (e.g., `'Agent_1_Overview_1712345678_a1b2c3d4.html'`).
        """
        name = f'{Misc.timestamp():.0f}_{Misc.uuid(8)}'
        path = Str.join(prefix or '', name, suffix or '')
        if file_type: 
            path = FileUtils.ensure_extension(path, file_type)
        return path
    
    @classmethod
    def gen_path(cls, 
        prefix: str = None, 
        suffix: str = None, 
        file_type: str = None, 
        folder: str | Path = None,
        project_signature: str = None
    ) -> Path:
        """
        Generate a unique full path under the project's temp folder.

        Creates the directory structure if it does not exist.

        Parameters
        ----------
        prefix : str | None, default=None
            Human-readable file identifier.
        suffix : str | None, default=None
            Additional qualifier.
        file_type : str | None, default=None
            File extension.
        folder : str | Path | None, default=None
            Subfolder within the temp directory (e.g.,
            `'Agent_1_Overview'`).
        project_signature : str | None, default=None
            Project signature for path resolution. If None,
            uses the DEFAULT (shared) temp path.

        Returns
        -------
        Path
            The full unique path (directory auto-created).
        """
        path = cls._get_gen_path(project_signature)
        Folder.create(path)
        f = cls.gen_unique_path(prefix, suffix, file_type)        
        if not folder:
            return Folder.standard(path / f)
        return Folder.standard(path / folder / f)

    def get_path(self,
        prefix: str = None, 
        suffix: str = None, 
        file_type: str = None, 
        folder: str | Path = None
    ) -> Path:
        """
        Generate a unique path under this instance's temp folder.

        Instance-level convenience (uses the path resolved at
        initialization from `setup`).

        Parameters
        ----------
        prefix : str | None, default=None
            Human-readable file identifier.
        suffix : str | None, default=None
            Additional qualifier.
        file_type : str | None, default=None
            File extension.
        folder : str | Path | None, default=None
            Subfolder within the temp directory.

        Returns
        -------
        Path
            The full unique path.
        """
        return self.gen_path(prefix, suffix, file_type, folder, self._path)
    
    def clear(self) -> None:
        """
        Remove this project's temp folder and all its contents.

        Safe to call even if the folder does not exist
        (errors are silently ignored).
        """
        shutil.rmtree(self._path, ignore_errors=True)

    @classmethod
    def clear_root(cls):
        """
        Remove the default (shared) temp folder.

        Use with caution — this affects all projects sharing
        the default path.
        """        
        shutil.rmtree(cls.DEFAULT, ignore_errors=True)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬