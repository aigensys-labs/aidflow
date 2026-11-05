"""
State-Aware Reporting Engine and Visualization Framework for AIDflow.

This module implements the standardized reporting layer used to communicate 
model performance, feature importance, and pipeline stability. It transforms 
raw execution results into high-fidelity, interactive dashboards through a 
decoupled architecture that separates data computation from visual rendering.

Key architectural capabilities:
1. Asynchronous Dashboard Synthesis: Utilizes `DashBackground` to offload heavy 
   visual computations to background processes, ensuring the Jupyter interface 
   remains responsive during complex report generation.

2. Polished Visual Standards: Enforces a unified design language across all reports 
   via `TableStyle` and integrated Plotly serialization, allowing for consistent 
   branding and readability across different model types.

3. Provenance-Linked Reporting: Integrates directly with the Model's state to 
   provide time-stamped, reproducible snapshots of performance, enabling a 
   complete audit trail from configuration to final visual output.

4. Polymorphic Display Modes: Supports multiple output targets through `DashInterface`, 
   allowing reports to be rendered as interactive Jupyter widgets, persisted as 
   standalone HTML files via `DashPlotlyLink`, or exported for external reporting.
"""

from ...deps.common             import  (
                                        Path,
                                        threading,
                                        Future,
                                        contextmanager
                                        )
from ...deps.plots              import  go
from ...deps.display            import  display
from ...deps.data               import  pd_Styler
from ...deps.types              import  (
                                        Any,
                                        Literal,
                                        FuncType
                                        )

from ...storage.export          import  (
                                        Export, 
                                        FileUtils
                                        )
from ...display.style           import  TableStyle
from ...display.verbosity       import  Verbosity

from ..parameters               import  Parameters
from ..foundation               import  (
                                        AigParam, 
                                        AigModel, 
                                        AigStop
                                        )
from ..execution                import  BackgroundProcess
from ..io                       import  Temp
from ..interop                  import  Dict
from ..logging                  import  (
                                        LOG, 
                                        LogSignature
                                        )
from ..accessor                 import  (
                                        UI, 
                                        Format
                                        )

class DashInterface:
    """
    Base interface defining the lifecycle and behavior of an AIDflow Dashboard.

    Establishes the mandatory protocol for all dashboard/report types,
    ensuring consistent build, save, and display behavior regardless
    of complexity (simple tables, stats summaries, or interactive
    Plotly views).

    Lifecycle
    ---------
    1. **build()**: Synthesize visual components from the model's
       fitted state. The primary engine — called internally by
       `show()` or directly with `save=True` for file output.

    2. **show()**: Render the completed dashboard to the active
       output cell. Manages verbosity gating and appends a project
       timestamp for provenance tracking.

    3. **save()**: Persist the dashboard as a static file
       (HTML/plotly) for archival or external reporting.

    4. **summary()**: Print the final text summary (feature counts,
       verdicts) & log it — the human-readable conclusion of the stage.

    Verbosity Protocol
    ------------------
    Dashboards respect the `Verbosity` level:
    - `QUIET`: No display, no file output.
    - `INFO`: Text summary only (no visual rendering).
    - `GUI`: Full visual rendering (Plotly/HTML) + summary.

    State Snapshotting
    ------------------
    The `_snap_init_state` / `_clone_init_state` pair enables
    the **update workflow**: a dashboard can be "cloned" back to
    its initial state and re-built with new parameters (e.g.,
    threshold tuning) without re-fitting the underlying model.

    Parameters
    ----------
    setup : Parameters
        Project parametrization.
    verbose : Verbosity | int | str | bool
        Display level. Controls whether the dashboard renders
        visually, prints text only, or remains silent.
    """

    def __init__(self, fitter: type, verbose: Verbosity|int|str|bool):
        from ..base import FitterBase
        fitter: FitterBase = fitter
        self._setup     = fitter.setup
        self._model     = fitter.MODEL
        self._signature = LogSignature.get()
        self._timestamp = AigModel.timestamp()
        self.verbose    = Verbosity.normalize(verbose)
        self.elements   = {}

    def save(self, save=None, custom_path=None, **kwargs):        
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
        dash_instance : object
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
        save_signature : bool, default = True
            Optionally, saves report signature for loadable elements.

        Returns
        -------
        dict[str, Path | dict[str, Path]]
            Element name → resolved path (or temp path if save=False).
            Structure follows `elements structure` (nested supported).
        """
        from ..io import DashIO
        return DashIO(self.setup).save(self, save, custom_path, **kwargs)

    def summary(self, _is_view=True):
        """
        Shows the final report summary.

        The human-readable conclusion of the stage — typically
        showing feature counts (in/out), verdicts, and key
        statistics. Logs to the project logger via the
        `ReportLogger` mechanism.
        """
        raise NotImplementedError(
            'AIG SAFETY ► Dashboard must implement `summary()`.'
        )
        self._final_summary()

    @AigParam.with_setattr_runtime(verbose=None)   
    def build(self, save=None, custom_path=None, **kwargs):
        """
        Synthesize the dashboard's visual components from the model's state.

        The primary engine of the report. Processes raw model
        output into visual elements (tables, Plotly figures,
        HTML). Can be configured to simultaneously persist the
        result as a file for external reporting.

        Parameters
        ----------
        save : bool | None, default=None
            If True, also save the output to file.
        custom_path : str | None, default=None
            Custom save path (when `save=True`).
        **kwargs : dict
            Forwarded to the concrete build implementation.
        """
        raise NotImplementedError(
            'AIG SAFETY ► Dashboard must implement `build()`.'
        )

    def show(self):
        """
        Render the completed dashboard to the active output cell.

        Triggers the internal `_show` mechanism, which manages
        verbosity settings (temporary escalation to GUI level),
        appends a project timestamp for provenance tracking,
        and logs the output via `ReportLogger`.
        """
        return self._show()

    def _show(self, 
        verbose: bool|Verbosity = None, 
        with_signature: bool = True, 
        **kwargs
    ):
        """
        Internal show — render with verbosity escalation.

        Temporarily sets the verbosity to GUI level (if not
        already), builds the dashboard for display, and prints
        the project timestamp. Used by `show()` and
        `_show_without_ts()`.

        Parameters
        ----------
        verbose : bool | Verbosity | None, default=None
            Override verbosity for this call. If None, defaults
            to GUI level.
        with_signature : bool, default=True
            Append the signature after rendering.
        """
        with self._temporary_verbose(verbose):
            if UI.is_html():
                self.build(save=False, _is_view=True, **kwargs)                
            else:
                self.summary(True)
            if with_signature:
                print(self._get_signatures())
            
    def _show_without_sig(self):
        """
        Show dashboard without signature — for delayed/secondary
        views where the signature would be redundant.
        """
        self._show(with_signature = False)

    @staticmethod
    def _is_view(kwargs: dict | bool | None) -> bool:
        """
        Returns True if when called from `show()`
        (enforces GUI-level rendering regardless of the
        current verbosity setting).        
        """
        if not kwargs:
            return False
        if isinstance(kwargs, bool):
            return kwargs
        return bool( kwargs.get('_is_view') )
    
    def _get_signatures(self) -> str:
        """Returns project- dashboard signature."""
        return (
            f'Report Saved at: {self._timestamp} | '
            f'Project Signature: {self.project_signature} | '
            f'Report Signature: {self._signature}'
        )
    
    @property
    def dashboard_signature(self) -> str:
        """Returns dashboard signature."""
        return self._signature
    
    @property
    def project_signature(self) -> str:
        """Project signature of fitted Model-Fitter."""
        return self._setup.project_signature
    
    @property
    def setup(self) -> Parameters:
        """
        Project setup of fitted Model-Fitter under project signature.
        """
        return self._setup

    @property
    def MODEL(self):
        """Returns AIG model ID."""
        return self._model
    
    @property
    def _is_not_verbose(self) -> bool:
        """True if verbosity is QUIET (no display at all)."""
        return self.verbose == Verbosity.QUIET

    @property
    def _is_info_only(self) -> bool:
        """True if verbosity is INFO (text only)."""
        return self.verbose == Verbosity.INFO

    @property
    def _is_gui(self) -> bool:
        """True if verbosity is at GUI level (full visual rendering)."""
        return self.verbose == Verbosity.GUI
            
    def _is_any(self, *conds):
        """Returns True if any condition is truthy."""
        return any(conds)

    def _is_save(self, save:bool|None) -> bool:
        """
        Returns True if either directly provided save or global setup.save is True.
        """
        return AigParam.di(save, self.setup.save)
    
    def _set_elements(self, elements: dict) -> None:
        """Sets schema of loadable dash elements."""
        if isinstance(elements, dict):
            self.elements = elements
            return
        raise ValueError('AIG SAFETY ► elements schema must be a dict type.')

    def _get_elements(self) -> dict:
        """Returns schema of loadable dash elements."""
        return self.elements
    
    def _snap_init_state(self, **kwargs):
        """
        Snapshot the current instance state for later cloning.

        Captures `self.__dict__` (plus any additional kwargs)
        into `self._init_state`. Required before the first
        `build()` call to enable the update workflow.

        Parameters
        ----------
        **kwargs : dict
            Additional state entries to include in the snapshot
            (e.g., data references not stored on `self`).
        """        
        self._init_state = Dict.deep_copy(self.__dict__) | kwargs
        
    def _clone_init_state(self, **kwargs):
        """
        Clone a new instance from the initial state snapshot.

        Produces a fresh instance of the same class with the
        state captured by `_snap_init_state`, optionally
        overridden by `kwargs`. Enables re-building the
        dashboard with new parameters (e.g., updated threshold)
        without re-fitting the model.

        Parameters
        ----------
        **kwargs : dict
            State overrides to apply on top of the snapshot.

        Returns
        -------
        same class instance
            A new instance with the initial state + overrides.
        """        
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self._init_state | kwargs)
        return new
    
    @staticmethod
    def _plotly_serialize(
        d: go.Figure|dict, 
        to_io_or_from_io: bool|Literal[0,1]=0
    ) -> go.Figure:
        """
        Serialize or deserialize a Plotly figure for storage.

        Parameters
        ----------
        d : go.Figure | dict | None
            The figure (or its dict representation) to convert.
        to_io_or_from_io : int, default=0
            - `0`: Figure → dict (serialize for storage).
            - `1`: dict → Figure (deserialize from storage).

        Returns
        -------
        dict | go.Figure | None
            The converted representation (or None if input is None).
        """        
        if to_io_or_from_io == 0:
            if d is None: 
                return None
            return d.to_dict()
        if d is None: 
            return None
        return go.Figure(d)
    
    @contextmanager
    def _temporary_verbose(self, verbose: bool | Verbosity = None):
        org = self.verbose
        self.verbose = verbose or Verbosity.GUI
        try:
            yield
        finally:
            self.verbose = org

    def _info_highlight(self, 
        message, 
        center:    bool = False,
        background: str = "#FF020215",
        log_only:  bool = False,
        is_view:   dict | bool = None,
        **kwargs
    ) -> None:
        """
        Display and log highlighted info banner (styled title block).

        Renders a formatted message with background color,
        centering, and optional gradient. Logs via
        `ReportLogger` when a logger reference is present.

        Parameters
        ----------
        message : str
            The text to display.
        center : bool, default=False
            Center-align the text.
        log_only : bool, default=False
            Log only, not shown in UI.            
        is_view : dict | bool, default=False
            Report view, no active log.            
        **kwargs : dict
            Forwarded to `Format.title`. Style tweaks / overrides. 
        """        
        DashReportLog(
            verbose  = self.verbose, 
            log_only = log_only, 
            is_view  = self._is_view(is_view),
            **kwargs
        ).info_highlight(message, center=center, background=background)
    
    def _final_summary(self,
        num: list|int = None, 
        cat: list|int = None,
        out: list|int = None,
        m_base:    str|FuncType = None, 
        m_all_in:  str = '',
        m_all_out: str = '',
        m:         str|FuncType = None,
        b:         str = None,
        margin_t:  int = 24,
        log_only:  bool = False,
        is_view:   dict | bool = None,
        **kwargs
    ) -> None:
        """
        Render the final summary banner — the conclusion of the stage.

        Displays a color-coded status banner summarizing the feature 
        outcome (all in / all out / partial) with appropriate messaging. 
        Also serves as the **logging entry point**: the banner text is 
        captured by the and written to the project log.

        Color Logic
        -----------
        - **Blue** (`#DBE1F1`): Standard outcome (some features
          in, some out) or custom message provided.
        - **Pink** (`#E79FB8`): Warning — ALL features were eliminated.
        - **Green** (`#7DEDC4`): All features passed — none were removed.

        Parameters
        ----------
        num : list | int | None
            Numeric features (list of names or count).
        cat : list | int | None
            Categorical features (list of names or count).
        out : list | int | None
            Output (selected) features (list of names or count).
        m_base : str | callable | None
            Default banner message (used when partial outcome).
        m_all_in : str
            Suffix for the "all features passed" message.
        m_all_out : str
            Suffix for the "all features eliminated" warning.
        m : str | callable | None
            Override message (bypasses the auto-logic).
        b : str | None
            Override background color (hex).
        margin_t : int, default=24
            Top margin for GUI rendering (0 when not GUI).
        log_only : bool, default=False
            Log only, not shown in UI.
        is_view : dict | bool, default=False
            Report view, no active log.
        **kwargs : dict
            Forwarded to `Format.title`. Style tweaks / overrides. 
        """    
        DashReportLog(
            verbose  = self.verbose,
            log_only = log_only,
            is_view  = self._is_view(is_view),
            margin_t = margin_t,
            **kwargs
        ).final_summary(
            num = num, 
            cat = cat,
            out = out,
            m_base = m_base, 
            m_all_in = m_all_in,
            m_all_out = m_all_out,
            m = m,
            b = b  
        )

class DashReportLog:
    """
    Render the final summary banner — the conclusion of the stage.

    Displays a color-coded status banner summarizing the feature 
    outcome (all in / all out / partial) with appropriate messaging. 
    Also serves as the **logging entry point**: the banner text is 
    captured by the and written to the project log.

    Color Logic
    -----------
    - **Blue** (`#DBE1F1`): Standard outcome (some features
        in, some out) or custom message provided.
    - **Pink** (`#E79FB8`): Warning — ALL features were eliminated.
    - **Green** (`#7DEDC4`): All features passed — none were removed.

    Parameters
    ----------
    verbose : Verbosity | int | str | bool
        Display level. Controls whether the dashboard renders
        visually, prints text only, or remains silent.        
    log_only : bool, default=False
        Log only, not shown in UI.        
    margin_t : int, default=24
        Top margin for GUI rendering (0 when not GUI).
    **kwargs : dict
        Forwarded to `Format.title`. Style tweaks / overrides. 
    """

    def __init__(self,
        verbose: bool | Verbosity = True, 
        log_only:  bool = False,
        is_view:   bool = False,
        margin_t:  int  = 24,
        **kwargs
    ):
        self.verbose  = verbose or is_view
        self.log_only = log_only
        self.is_view  = is_view
        self.margin_t = margin_t
        self.title_kw = kwargs

        # keep console output light
        self._div = None if not UI.is_html() else 'solid'

    @property
    def is_gui(self) -> bool:
        """True if verbosity is at GUI level (full visual rendering)."""
        return self.verbose == Verbosity.GUI
        
    def final_summary(self,
        num: list|int = None, 
        cat: list|int = None,
        out: list|int = None,
        m_base:    str|FuncType = None, 
        m_all_in:  str = '',
        m_all_out: str = '',
        m:         str|FuncType = None,
        b:         str = None,        
    ) -> None:
        """
        Render the final summary banner — the conclusion of the stage.

        Displays a color-coded status banner summarizing the feature 
        outcome (all in / all out / partial) with appropriate messaging. 
        Also serves as the **logging entry point**: the banner text is 
        captured by the and written to the project log.

        Color Logic
        -----------
        - **Blue** (`#DBE1F1`): Standard outcome (some features
          in, some out) or custom message provided.
        - **Pink** (`#E79FB8`): Warning — ALL features were eliminated.
        - **Green** (`#7DEDC4`): All features passed — none were removed.

        Parameters
        ----------
        num : list | int | None
            Numeric features (list of names or count).
        cat : list | int | None
            Categorical features (list of names or count).
        out : list | int | None
            Output (selected) features (list of names or count).
        m_base : str | callable | None
            Default banner message (used when partial outcome).
        m_all_in : str
            Suffix for the "all features passed" message.
        m_all_out : str
            Suffix for the "all features eliminated" warning.
        m : str | callable | None
            Override message (bypasses the auto-logic).
        b : str | None
            Override background color (hex).
        """    

        def lx(nco):
            if nco is None:
                return 1
            if isinstance(nco, int):
                return nco
            return len(nco)
                
        ln = lx(num)        
        lc = lx(cat)        
        lo = lx(out)        
        
        if m is not None:
            if b is None:
                b  = "#DBE1F1"
                b2 = "#ABB7E7"
            else:
                b2 = b
        
        elif not (ln + lc):
            b  = "#E79FB8"
            b2 = "#E79FB8"
            m  = f'► WARNING: All Features were Eliminated {m_all_out}'
                        
        elif not lo:
            b  = "#7DEDC4"
            b2 = "#44F4B3"
            m  = f'► All Features are Considered as Stable {m_all_in}'
            f'{' ' if m_all_in else ''}on TRAIN ► No Features were removed from ML Data'
        
        else:
            b  = "#DBE1F1"
            b2 = "#ABB7E7"
            m  = m_base
        
        style_kw = self._align_kw()

        Format.title(
            m,
            show  = self._is_show(),
            style = Format.StyleTitle(
                color         = '#000000',
                color2        = b2,
                background    = b,
                gradient_kind = 'right', 
                margin_t      = self.margin_t if self.is_gui else 0,
                center        = False, 
                div           = self._div,
                **style_kw
        ), **self.title_kw)
    
    def info_highlight(self, message, center:bool=False, background:str = "#FF020223"
    ) -> None:
        """
        Display a highlighted info banner (styled title block).

        Renders a formatted message with background color,
        centering, and optional gradient. Logs via
        `ReportLogger` when a logger reference is present.

        Parameters
        ----------
        message : str
            The text to display.
        center : bool, default=False
            Center-align the text.
        **kwargs : dict
            Forwarded to `Format.title`. Style tweaks / overrides. 
        """        
        style_kw = self._align_kw()
        Format.title(
            message,
            show  = self._is_show(),
            style = Format.StyleStats(
                center      = center, 
                color2      = background,
                background  = background, 
                div         = self._div,
                **style_kw
        ), **self.title_kw)

    def _is_show(self) -> bool:
        return False if self.log_only else self.verbose

    def _align_kw(self) -> dict:
        kw = self._align_kwargs_style(self.title_kw)
        if not self.is_view: # logging active
            self.title_kw[LOG.R] = self
        return kw
        
    @staticmethod
    def _align_kwargs_style(kwargs:dict) -> dict:
        p = AigParam.get_definition(Format.Style)
        return {k:kwargs.pop(k) for k in kwargs.copy() if k in p}
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class DashMixin:
    """
    Behavioral mixin providing automated dashboard lifecycle management.

    Allows any Model implementation to seamlessly integrate reporting
    capabilities without boilerplate logic for initialization, background
    offloading, or state access.

    Two execution modes are supported:
    - **Synchronous** (default): The dashboard is built in the
      current thread after the model step completes. Suitable
      for small dashboards or when the user explicitly wants
      to see the result before proceeding.
    - **Asynchronous** (`build_at_background=True`): The dashboard
      is built in a background thread. The pipeline immediately
      continues; the dashboard becomes available (via `.dashboard`)
      once the background process completes. A `threading.Event`
      provides synchronization for code that needs to wait.

    Save vs. Build
    --------------
    - **build** (interactive): Renders the dashboard in the
      active UI environment (Jupyter, Dash app, etc.).
    - **save** (static): Generates a static artifact dashboard
      for archival or reporting. Triggered when `setup.save` is
      active and no verbosity is requested.
    """
    
    _dashboard: DashInterface = None
        
    def _create_dashboard(self, 
        DashClass: DashInterface, 
        *args,
        build_dash_after_init: bool = None,
        **kwargs
    ) -> DashInterface | None:
        """
        Instantiate and build/save the dashboard.

        Determines the execution mode (build vs. save) based on
        the `verbose` setting and `setup.save`, then creates and
        triggers the dashboard.
        """
        from ..base import FitterBase
        instance: FitterBase = self
        
        verbose  = kwargs.get('verbose', instance.verbose)
        is_build = AigParam.di(build_dash_after_init, verbose)
        setup    = instance.setup
        is_save  = setup.save
        is_log   = setup.logging

        if not any([is_build, is_save, is_log]):
            return
        
        dashboard: DashInterface = DashClass(self, *args, **kwargs)

        if is_build:
            dashboard.build()
        elif is_save:
            dashboard.save()

        return dashboard
    
    def _callback_dashboard(self, dash: DashInterface) -> None:
        self._dashboard: DashInterface = dash

    def _init_dashboard(self, 
        DashClass: DashInterface, 
        *args,
        build_dash_after_init: bool = None,
        build_at_background:   bool = False,
        with_background_sync:  threading.Event = None,
        **kwargs
    ) -> None:
        """
        Initialize the dashboard after the primary model step completes.

        Orchestrates dashboard creation with full control over
        execution mode (sync/async), parameter injection, and
        background synchronization.

        Execution Flow
        --------------
        1. **Synchronous path** (default): Creates and builds
           the dashboard in the current thread. Blocks until
           complete.
        2. **Asynchronous path** (`build_at_background=True`):
           Offloads dashboard creation to a background thread.
           The pipeline immediately continues. The dashboard
           becomes available via `.dashboard` once complete.

        Parameters
        ----------
        DashClass : DashInterface
            The dashboard class to instantiate.
        build_dash_after_init : bool | None, default=None
            - **None**: Inherit from `verbose`.
            - **True**: Force interactive build after init.
            - **False**: Static save only (no interactive build).
        build_at_background : bool, default=False
            - **False**: Build in the current thread (blocking).
            - **True**: Offload to a background thread
              (non-blocking). The pipeline continues
              immediately; the dashboard is available once
              the thread completes.
        with_background_sync : threading.Event | None, default=None
            When `build_at_background=True`, a `threading.Event`
            that is set once the dashboard is fully constructed.
            Useful for code that must wait for the dashboard
            before proceeding (e.g., screenshot capture).
        *args, **kwargs :
            Additional parameters forwarded to the dashboard
            constructor.
        """

        # Collect any attributes that the dashboard class expects but are not in kwargs
        dash_params = AigParam.get_definition(DashClass)
        if not dash_params:
            raise AigStop.custom(
                'AIG auto fitter-dashboard construct is only possible when Dash class '
                'accepts fitter instance as the first argument.'
            )

        # Instantiate and build dashboard
        after_init = dict(
            build_dash_after_init = build_dash_after_init,
        )
        if not build_at_background:
            self._dashboard = self._create_dashboard(DashClass, *args, **after_init, **kwargs)
            return 

        BackgroundProcess(
            callback = self._callback_dashboard,
            new_process = False,
            daemon = False,
            waiting_event = with_background_sync
        ).run(
            DashMixin._create_dashboard, self, DashClass, *args, **after_init, **kwargs
        )

    @property
    def dashboard(self) -> DashInterface:
        """
        The instantiated dashboard instance.

        Safe access with a clear error if the dashboard has not
        yet been created (i.e., the primary fit step has not
        been executed, or the background thread has not
        completed).

        Raises
        ------
        AigStop
            If accessed before the parent worker has executed
            its primary fit/process step.

        Notes
        -----
        When `build_at_background=True`, this property may raise
        if accessed before the background thread completes.
        Use `with_background_sync` (threading.Event) to
        synchronize if sequential access is required.
        """
        if not self._dashboard is None:
            return self._dashboard
        
        raise AigStop.custom(
            "Dashboard not available. Please run the main step (fit/process) first "
            "with Verbosity > 0 or Logging = 1."
        )
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class DashBackground:
    """
    Synchronization primitives for background dashboard construction.

    Provides a two-stage synchronization protocol for dashboards
    that are built in a background thread. The pipeline must
    wait for the dashboard to reach specific readiness states
    before consuming its output (e.g., for snapshots, reports,
    or interactive updates).

    Two-Stage Synchronization
    -------------------------
    Dashboard construction is split into two logical stages:

    1. **Stage 1 — Data Generation** (`.init`):
       The dashboard generates its hard data (metrics, tables,
       feature lists). This is the computationally heavy step.
       Once complete, the data is available for `.build`.

    2. **Stage 2 — View Creation** (`.build`):
       The dashboard renders its visual components (charts,
       panels, layout). Once complete, the view is available
       for user interaction or snapshot capture.

    Each stage has an independent readiness flag. Code can
    check or wait for either stage independently:

    >>> # Wait for data generation before building:
    >>> dash.wait_for_stage(1)
    >>>
    >>> # Wait for full completion before snapshot:
    >>> dash.wait_for_stage(2)
    >>> dash.take_snapshot()

    State Sanitization
    ------------------
    When the dashboard state is persisted (e.g., for project
    save/restore), the sync events are replaced with `True`
    flags (via `get_sync_to_snap`). On restore, `is_sync_ready`
    immediately returns True — the dashboard is considered fully
    constructed without waiting.

    Parameters
    ----------
    (None — this is a mixin/base; no constructor args.)
    """   

    @staticmethod
    def set_background_sync(fitter: type) -> BackgroundProcess.Event:
        """
        Create and attach a background synchronization event to the fitter.

        The event is stored as an attribute on the fitter instance
        (keyed by `FIT.SYNC`), enabling the dashboard mixin to
        reference it during `_init_dashboard` with
        `build_at_background=True`.

        Parameters
        ----------
        fitter : FitterBase
            The fitter instance to attach the event to.

        Returns
        -------
        BackgroundProcess.Event
            The threading event (set when dashboard construction
            completes). Pass this as `with_background_sync` to
            `_init_dashboard`.
        """        
        from ..base import FIT
        event = BackgroundProcess.Event()
        setattr(fitter, FIT.SYNC, event)
        return event

    @staticmethod
    def get_background_sync(fitter: type) -> BackgroundProcess.Event | None:
        """
        Retrieve the background synchronization event from the fitter.

        Returns
        -------
        BackgroundProcess.Event | None
            The event if previously set via `set_sync_task`;
            None otherwise.
        """        
        from ..base import FIT
        return getattr(fitter, FIT.SYNC, None)
    
    @classmethod
    def run_with_background_sync(cls, 
        fx: FuncType, 
        fitter: type, 
        message : str = 'Report build in progress',
        show_timer: bool = True,
        verbose: bool = True,
        daemon: bool = True,
        args: tuple = (),
        **kwargs
    ) -> None:
        """
        Execute a function with background synchronization on a fitter's event.

        Retrieves the `threading.Event` previously attached to the
        fitter (via `set_sync_task`) and waits for it before executing 
        `fx`. If no event is attached (sanitized / restored state), 
        executes `fx` immediately.

        Use this for operations that must wait for the background
        dashboard to finish before proceeding (e.g., generating
        a report snapshot, exporting a static view), while
        displaying a progress indicator to the user during the
        wait.

        Parameters
        ----------
        fx : callable
            The function to execute once the sync event is set.
            Receives `(args, kwargs)` as a single tuple
            (BackgroundProcess protocol).
        fitter : FitterBase
            The fitter instance from which the sync event is
            retrieved (must have been previously set via
            `set_sync_task`).
        message : str, default='Report build in progress'
            Status message displayed while waiting for the
            background thread to complete.
        show_timer : bool, default=True
            Display an elapsed-time counter while waiting.
        verbose : bool, default=True
            Enable status display (message + timer). If False,
            waits silently.
        daemon : bool = True
            If True, the thread is killed when the main thread exits.
            If False, the interpreter waits for the thread to complete
            before shutting down. Use False for threads doing final
            saves or critical cleanup.            
        args : tuple = ()
            Arguments forwarded to `fx`.
        **kwargs :
            Arguments forwarded to `fx`.

        Notes
        -----
        **Sanitized State**: If the fitter has no sync event
        (e.g., restored from a project snapshot where the
        dashboard is pre-built), the function executes
        immediately without waiting.
        """        
        sync = cls.get_background_sync(fitter)

        # sync=None (sanitized) → show stored report | is_set → ready
        if sync is None or sync.is_set(): 
            fx((args, kwargs))
            return
        
        BackgroundProcess(
            fx, 
            waiting_event = sync,
            daemon        = daemon,
            message       = message,
            show_timer    = show_timer,
            verbose       = verbose,
        ).run(sync, *args, **kwargs)        

    def set_sync_wait(self) -> None:
        """
        1. wait for .init  to finish: dash hard data gen → consumed by .build
        2. wait for .build to finish: dash view creation → consumed by ._update_vars_out_user
        """     
        self._background_sync1 = BackgroundProcess()
        self._background_sync2 = BackgroundProcess()        

    def get_sync_to_snap(self):
        """Sanitize state to save dash state for reporting."""
        new = self.__class__()
        new._background_sync1 = True
        new._background_sync2 = True        
        return new
    
    def get_sync_wait(self, back_sync:int) -> BackgroundProcess | bool:
        """
        Retrieve the synchronization barrier for a specific stage.

        Parameters
        ----------
        back_sync : int
            - **1**: Stage 1 barrier (data generation).
            - **2**: Stage 2 barrier (view creation).

        Returns
        -------
        BackgroundProcess | bool
            The live barrier (for waiting) or `True` (if
            sanitized from a snapshot).
        """        
        if back_sync==1: 
            return self._background_sync1
        return self._background_sync2

    def is_sync_ready(self, back_sync:int) -> bool:
        """
        Check whether a synchronization stage has completed.

        Parameters
        ----------
        back_sync : int
            - **1**: Data generation complete?
            - **2**: View creation complete?

        Returns
        -------
        bool
            True if the stage is ready (immediate execution
            possible). False if the background thread is still
            working (must wait).
        """        
        sync = self.get_sync_wait(back_sync)
        if isinstance(sync, bool): 
            return True # → snap saved dash → bool "True"
        if sync.is_set():
            return True # → current run event True
        return False    # → thread.Event → dash async creation
    
    def set_sync_ready(self, back_sync:int) -> None:
        """
        Mark a synchronization stage as complete.

        Called by the background thread when the stage finishes.
        Subsequent calls to `is_sync_ready` will return True.

        Idempotent: if already ready, this is a no-op.

        Parameters
        ----------
        back_sync : int
            - **1**: Mark data generation as complete.
            - **2**: Mark view creation as complete.
        """        
        if self.is_sync_ready(back_sync):
            return
        sync = self.get_sync_wait(back_sync)
        sync.set_created()
    
    def run_with_sync(self, fx: FuncType, back_sync:int, *args, **kwargs) -> None:
        """
        Execute a function with synchronization guarantee.

        If the target stage is already ready, executes `fx`
        immediately. Otherwise, spawns a background process that
        waits for the barrier before executing `fx`.

        Use this for fire-and-forget operations that depend on
        dashboard readiness (e.g., updating a UI panel after
        the dashboard has rendered).

        Parameters
        ----------
        fx : callable
            The function to execute. Receives `(args, kwargs)`
            as a single tuple (BackgroundProcess protocol).
        back_sync : int
            - **1**: Wait for data generation.
            - **2**: Wait for view creation.
        *args, **kwargs :
            Arguments forwarded to `fx`.
        """           
        if self.is_sync_ready(back_sync):
            fx((args, kwargs)) # → (a,k): output design of ProcessBackground.waiting_helper
            return
        
        wait_until_created = self.get_sync_wait(back_sync).wait_event       
        BackgroundProcess(
            fx, waiting_event = wait_until_created,
        ).run(wait_until_created, *args, **kwargs)

    def run_with_result(self, fx: FuncType, back_sync:int, *args, **kwargs) -> Any:
        """
        Execute a function with synchronization and return its result.

        Like `run_with_sync`, but **blocks the calling thread**
        until `fx` completes and returns a value. Use this when
        the result is needed immediately (e.g., extracting
        dashboard metrics for a report).

        If the target stage is already ready, executes `fx`
        synchronously and returns immediately.

        Parameters
        ----------
        fx : callable
            The function to execute. Receives `(args, kwargs)`
            as a single tuple (BackgroundProcess protocol).
        back_sync : int
            - **1**: Wait for data generation.
            - **2**: Wait for view creation.
        *args, **kwargs :
            Arguments forwarded to `fx`.

        Returns
        -------
        Any
            The return value of `fx`.

        Raises
        ------
        Exception
            Any exception raised by `fx` is propagated to the
            calling thread via the Future.
        """
        if self.is_sync_ready(back_sync):
            return fx((args, kwargs))
                            
        wait_until_created = self.get_sync_wait(back_sync).wait_event
        fut = Future()

        def wrap(*aw, **kw):
            wait_until_created.wait()
            try:
                res = fx((aw, kw))
                fut.set_result(res)
            except Exception as e:
                fut.set_exception(e)

        BackgroundProcess().run(wrap, *args, **kwargs)
        return fut.result()
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class DashPlotlyLink:
    """
    Plotly-to-HTML Bridge — Interactive Plot Persistence & Display.

    Serializes a Plotly figure to a standalone HTML file and
    provides a uniform output interface for dashboards. The
    output can be either:

    - **The raw Plotly Figure** (in-memory, for direct display
      or programmatic access).
    - **A clickable HTML link** (persisted to disk, viewable
      without a Python kernel).

    Input Modes
    -----------
    - **`go.Figure`**: An in-memory Plotly figure. It is
      serialized to HTML (link mode) or returned as-is
      (figure mode).
    - **`Path`**: A path to an already-saved HTML file. Used
      directly for link resolution (no re-serialization).

    Output Modes
    ------------
    - `set_fig_not_dash=True`: Returns the raw Plotly Figure.
      No file I/O occurs. Use when the figure will be displayed
      directly or passed to another component.

    - `set_fig_not_dash=False` (default): Produces a link artifact:
      - `dict` — when `make_clickable=False`.
      - `pd.Styler` — when `make_clickable=True`.

    Path Resolution
    ---------------
    - **Explicit**: When `report_path` is provided, the HTML
      is saved to (or loaded from) that exact location.
    - **Project-scoped**: When `report_path` is None and
      `signature` is a Parameters instance, a unique path is
      generated under the project's temp directory.
    - **Signature-scoped**: When `signature` is a project
      signature string, resolved via the project registry.

    Parameters
    ----------
    figure_path : go.Figure | Path
        - `go.Figure` → the Plotly figure to process.
        - `Path` → path to an already-saved HTML file
          (skips serialization, used directly for the link).
    report_path : str | None, default=None
        Explicit target path for the HTML file. When
        `figure_path` is a `Path`, this is ignored (the
        figure_path itself provides the location).
    signature : Parameters | str, default=None
        Project context for path resolution. Can be a full
        `Parameters` instance or a project signature string.
    link_name : str | None, default=None
        The display name for the link (used in the clickable
        table when `make_clickable=True`).
    dash_name : str, default='Dashboard'
        The dashboard-level name (parent link grouping).
    tab_name : str, default='Link'
        The tab/element name within the dashboard.
    make_clickable : bool, default=True
        When in link mode: if True, wraps the link in a styled
        clickable table (`pd.Styler`). If False, returns a
        plain dict.
    hide_columns : bool, default=False
        When `make_clickable=True`: hide the path column
        (show only the link name).
    set_fig_not_dash : bool, default=False
        If True, the output is the raw Figure. If False, the
        figure is exported to HTML and the output is a link
        or clickable table.        
    show : int | bool, default=True
        If truthy, immediately displays the output to the
        active cell upon construction.
    """

    def __init__(self, *,
        figure_path: go.Figure | Path,
        report_path:        str  = None,
        signature:Parameters|str = None,
        link_name:          str  = None,
        dash_name:          str  = 'Dashboard',
        tab_name:           str  = 'Link',
        make_clickable:     bool = True,
        hide_columns:       bool = False,
        set_fig_not_dash:   bool = False,
        show:           int|bool = True,
    ):
        self.report_path      = report_path
        self.signature        = signature
        self.link_name        = link_name
        self.dash_name        = dash_name
        self.tab_name         = tab_name
        self.make_clickable   = make_clickable
        self.hide_columns     = hide_columns
        self.set_fig_not_dash = set_fig_not_dash
        self.verbose          = show

        self.save(figure_path)

    def save(self, fig: go.Figure | Path) -> None: 
        """
        Serialize the Plotly figure and produce the output object.

        In **figure mode** (`set_fig_not_dash=True`):
        Sets `self.output` to the raw Figure. No file I/O.

        In **link mode** (`set_fig_not_dash=False`):
        Exports the figure to HTML, resolves the target path
        (report or temp), and constructs the output as either
        a plain dict or a clickable table.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to process.

        Raises
        ------
        AigStop.custom
            If `fig` is None.
        """      
        if fig is None:
            AigStop.custom(
                "AIG SAFETY → DashPlotlyLink → Figure is None. Nothing to save or display."
            )

        if self.set_fig_not_dash: 
            self._set_show_output(fig)
            return
        
        path  = self._make_fig_path(fig)
        links = {self.tab_name: path}

        if self.link_name is not None:
            links = {self.dash_name: self.link_name} | links

        dash = links
        if self.make_clickable:
            dash = TableStyle.clickable(
                [links], self.tab_name, hide_columns=self.hide_columns, show=False
            )
        self._set_show_output(dash)
    
    def get_output(self) -> dict | pd_Styler | go.Figure:
        """
        Retrieve the processed output object.

        Returns
        -------
        go.Figure
            When `set_fig_not_dash=True` — the raw Plotly Figure.
        dict
            When `set_fig_not_dash=False` and
            `make_clickable=False` — a mapping of `{name: path}`.
        pd.Styler
            When `set_fig_not_dash=False` and
            `make_clickable=True` — a styled clickable table with the link.
        """
        return self.output

    def _set_show_output(self, fig_dash) -> None:
        self.output = fig_dash
        if self.verbose: 
            display(fig_dash)
        
    def _make_fig_path(self, fig: go.Figure | Path) -> Path:
        if isinstance(fig, Path):
            return FileUtils.ensure_extension(fig, 'html')
        return Export.plt_as_html(fig, self._route())
    
    def _route(self) -> Path:
        if self.report_path:
            return self.report_path
        
        kw = dict(prefix = 'Dash', file_type = 'html')
        if isinstance(self.signature, Parameters):
            return Temp(self.signature).get_path(**kw)
        
        return Temp.gen_path(**kw, project_signature=self.signature)
# ─────────────────────────────────────────────────────────────────────────────────────────────────