"""
Global Environment Orchestrator and Runtime Configuration Manager.

This module serves as the infrastructure layer of AIDflow, responsible
for detecting the host platform and initializing the global runtime
environment. It ensures that the library's behavior—specifically
regarding visual output and computational parallelism—is consistent
across different execution contexts (Terminal, Jupyter Notebooks,
Cloud IDEs).

Key Architectural Responsibilities
---------------------------------
1. **Interface Abstraction**: Dynamically switches between ANSI
   and HTML rendering engines based on the detected UI context.
2. **Backend Parallelism**: Provides a unified interface for
   selecting the computational backend (Ray, Multiprocessing, or
   single-threaded) — scaling from local to distributed without
   code changes.
3. **Theme & Styling**: Manages global visual themes and CSS
   injection for HTML outputs.
4. **Singleton Configuration**: Maintains a centralized ``Config``
   registry that tracks platform metadata and environment
   settings — the single source of truth for the library's
   operational mode.

By decoupling "how it runs" (infrastructure) from "what it does"
(ML logic), this module allows AIDflow to be platform-agnostic
and environment-aware.
"""

from ...deps.common             import  (
                                        platform, 
                                        sys, 
                                        )
from ...deps.display            import  HTML
from ...deps.types              import  Literal

from ...utils.platform          import  Platform
from ...utils.ignore            import  Warnings
from ...display.css             import  (
                                        CSS, 
                                        CssStyle,
                                        )
from ...display.format          import  (
                                        FormatUI, 
                                        FormatHTML, 
                                        FormatANSI
                                        )
from ...display.rgb             import  (
                                        RgbUI, 
                                        RgbANSI, 
                                        RgbHTML,
                                        )
from ...display.spinner         import  (
                                        SpinnerUI,
                                        SpinnerANSI,
                                        SpinnerHTML,
                                        )
from ..factory                  import  (
                                        DictClass, 
                                        Dataclass
                                        )
from ..config                   import  THEME
from .parallel                  import  (
                                        Parallel, 
                                        ParallelRay, 
                                        ParallelMulti
                                        )


if (3, 12) > (sys.version_info.major, sys.version_info.minor):
    raise RuntimeError(
        "This package requires Python 3.12+, "
        "current system operates on "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# GLOBAL SETTINGS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class UI(DictClass):
    """UI environment ANSI | HTML."""
    ANSI = 'ansi'
    HTML = 'html'

class PARALLEL(DictClass):
    """Parallel backend RAY | MULTI | OFF."""
    RAY   = 'ray'
    MULTI = 'multi'
    OFF   = 'off'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@Dataclass.base
class Config:
    """
    Global runtime configuration state.

    Immutable snapshot of the library's operational mode at a
    given point in time. Holds the rendering engine, theme,
    parallel backend, and platform metadata.

    Attributes
    ----------
    platform : str
        Operating system identifier (e.g., `'linux'`, `'darwin'`).
    python_version : str
        Python version string (e.g., `'3.12.4'`).
    ui : str
        Rendering engine: `UI.ANSI` or `UI.HTML`.
    theme : str
        Active visual theme identifier.
    ui_css : dict[str, HTML]
        CSS style definitions for HTML rendering (None for ANSI).
    rgb : RgbHTML | RgbANSI
        Color formatting engine (HTML-safe or ANSI escape codes).
    formatter : FormatHTML | FormatANSI
        Text/table formatting engine.
    parallel : ParallelRay | ParallelMulti
        Active parallel execution backend.
    """
    platform        : str
    python_version  : str
    ui              : str
    theme           : str    
    ui_css          : dict[str, HTML]
    rgb             : RgbHTML | RgbANSI
    formatter       : FormatHTML | FormatANSI
    spinner         : SpinnerHTML | SpinnerANSI
    parallel        : ParallelRay | ParallelMulti
    
    def get_css_id(self, 
        name: Literal['panel', 'tuning', 'button', 'step', 'line']
    ) -> str:
        """
        Resolve a semantic CSS class name to its concrete
        definition from the active theme.

        Parameters
        ----------
        name : str
            Semantic identifier: `'panel'`, `'tuning'`,
            `'button'`, `'step'`, or `'line'`.

        Returns
        -------
        str
            The CSS class string for the current theme.
        """        
        m = {
            'panel':    CSS.PANEL,
            'tuning':   CSS.TUNING,
            'button':   CSS.BUTTON,
            'graph':    CSS.GRAPH,
            'step':     CSS.STEP,
            'line':     CSS.LINE
        }
        return m[name]
    
    def update_theme(self, theme:THEME) -> "Config":
        """
        Return a new Config with the theme changed.

        Parameters
        ----------
        theme : THEME
            The new theme to apply.

        Returns
        -------
        Config
            A new Config instance with the updated theme.
        """
        prm = self.to_dict() | dict(theme = theme)
        return Config(**prm)
    
    def update_ui(self, ui:UI) -> "Config":
        """
        Return a new Config with the rendering engine changed.

        Parameters
        ----------
        ui : UI
            The new UI mode (`UI.ANSI` or `UI.HTML`).

        Returns
        -------
        Config
            A new Config instance with the updated UI mode.
        """        
        prm = self.to_dict() | dict(ui = ui)
        return Config(**prm)
        
    def is_html(self) -> bool:
        """Check whether the active rendering engine is HTML."""
        return self.ui == UI.HTML
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# GLOBAL CONFIG MANAGER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ConfigManager:
    """
    Global configuration registry (singleton).

    Maintains a named registry of `Config` instances and
    tracks which one is the active default for the pipeline.
    All AIDflow pipeline steps resolve their rendering engine,
    theme, and parallel backend through this manager.

    The manager is initialized with a default ANSI/Multi
    configuration for worker threads (no HTML, no Ray).
    """
    def __init__(self):
        self._registry: dict[str, Config] = {}
        self._def_name = 'ansi_multi'
        self.build(self._def_name, UI.ANSI, PARALLEL.MULTI, THEME.DARK)  
        # default ANSI config for workers
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_css(ui:str, theme:str|None) -> HTML | None:
        """Generate CSS style definitions for HTML rendering."""        
        if ui != UI.HTML:
            return None
        return CssStyle(theme).ui()

    @staticmethod
    def _get_ui_backend() -> str:
        """Auto-detect: HTML for Jupyter, ANSI for terminal."""
        if Platform.is_in_jupyter():
            return UI.HTML
        return UI.ANSI

    @staticmethod
    def _get_parallel_backend() -> str:
        """Auto-detect: RAY or MULTIPROCESS."""
        if Platform.is_ray_active():
            return PARALLEL.RAY
        return PARALLEL.MULTI

    @staticmethod
    def _normalize_parallel(parallel:str) -> str|None:
        if parallel == PARALLEL.OFF:
            return None
        return parallel
    
    @staticmethod
    def _get_default_theme(ui:str) -> str | None:
        if ui != UI.HTML:
            return None
        return THEME.METAL
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def build(self, 
        name:     str, 
        ui:       UI = UI.ANSI, 
        parallel: PARALLEL = PARALLEL.MULTI,
        theme:    THEME = THEME.METAL,
    ) -> Config:
        """
        Build a new Config and register it under the given name.

        Resolves the rendering engine, CSS theme, color
        formatter, and parallel backend into a unified Config
        instance. Also performs one-time platform
        initialization (cross-platform save/load compatibility,
        warning suppression).

        Parameters
        ----------
        name : str
            Registry key for this configuration.
        ui : UI, default=UI.ANSI
            Rendering engine.
        parallel : PARALLEL, default=PARALLEL.RAY
            Parallel execution backend.
        theme : THEME, default=THEME.METAL
            Visual theme.

        Returns
        -------
        Config
            The newly built configuration.
        """
        Platform.cross_platform_save_load()
        Platform.joblib_dependecy_workaround()
        Warnings.local_redundant()
        
        css         = self._apply_css(ui, theme)
        rgb         = RgbUI.get_formatter(ui)
        formatter   = FormatUI.get_formatter(ui)
        spinner     = SpinnerUI.get_spinner(ui)
        parallel_   = self._normalize_parallel(parallel)
        parallel_   = Parallel.get_backend(parallel_)

        config = Config(
            platform        = platform.system().lower(),
            python_version  = sys.version.split()[0],
            ui              = ui,
            theme           = theme,
            ui_css          = css,
            rgb             = rgb,
            formatter       = formatter,
            spinner         = spinner,
            parallel        = parallel_,
        )
        self._registry[name] = config
        return config
    
    def update(self, name:str, config:Config) -> None:
        """
        Replace an existing config in the registry.

        Parameters
        ----------
        name : str
            Registry key to update.
        config : Config
            The new configuration to store.
        """
        self._registry[name] = config

    def set_default(self, name:str) -> Config:
        """
        Set the active default config for all pipeline steps.

        Parameters
        ----------
        name : str
            Registry key to set as default.
        """
        self._def_name = name        

    def get(self, name: str) -> Config:
        """
        Retrieve a config by name. If not found, builds a
        default ANSI configuration under that name.

        Parameters
        ----------
        name : str
            Registry key to look up.

        Returns
        -------
        Config
            The configuration for the given name.
        """
        if name not in self._registry:
            return self.build(name)
        return self._registry[name]
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @property
    def default(self) -> Config:
        """The currently active default configuration."""
        return self.get(self._def_name)

    @property
    def rgb(self):
        """The active color formatting engine."""
        return self.default.rgb

    @property
    def formatter(self):
        """The active text/table formatting engine."""
        return self.default.formatter

    @property
    def spinner(self):
        """The active waiting spinner engine."""
        return self.default.spinner
    
    @property
    def parallel(self):
        """The active parallel execution backend."""
        return self.default.parallel
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
def set_backend(
    config:   ConfigManager, 
    name:     str = 'env-config', 
    parallel: PARALLEL = None,
    ui:       UI = None,
    theme:    THEME = None,
) -> Config:
    """
    Build a new configuration and set it as the pipeline default.

    This is the primary entry point for configuring the AIDflow
    runtime environment at import time. Typically called once
    per execution context (e.g., in a notebook cell or script
    header).

    Parameters
    ----------
    config : ConfigManager
        The global config manager instance.
    name : str, default='env-config'
        Registry name for this configuration.
    parallel : PARALLEL, default=None
        Auto-detects parallel environment when default None is set.
        Parallel execution backend: 
        - PARALLEL.RAY   → preferred 
        - PARALLEL.MULTI → fallback
        - PARALLEL.OFF   → off
    ui : UI, default=None
        Auto-detects ui environment when default None is set.
        Rendering engine:
        - UI.HTML → notebooks
        - UI.ANSI → terminal
    theme : THEME, default=None
        Visual theme. Active only with UI.HTML inside HTML env.

    Returns
    -------
    Config
        The active configuration (for programmatic access).
    """ 
    ui = ui or config._get_ui_backend()
    pl = parallel or config._get_parallel_backend()
    th = theme or config._get_default_theme(ui)

    config.set_default(name)
    config.build(name, ui, pl, th)

    if Platform.is_main():
        print( RgbANSI.bold_green('AIDflow library imported √') )

    return config.get(name)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

config_manager = ConfigManager()