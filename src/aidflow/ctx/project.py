"""
Project Context — Session Initialization & Configuration.

This module implements `ProjectContext`, the entry point for
starting an AIDflow project. It manages the configuration
lifecycle:

genesis (identity + data) → params (hyperparameters) → flow (execution)

Dual-Mode Initialization
------------------------
- **HTML mode**: Launches interactive panels
  (`ProjectGenesis`, `PipelineConfigurator`) for visual
  parameter tuning in a notebook or browser.
- **ANSI mode**: Accepts dictionary-based arguments for
  scripted/headless execution.

Both modes produce the same final state — a fully resolved
`Parameters` object and an active `PipelineContext`.

Configuration Immutability
--------------------------
Once initialized, the project setup is locked via a read-only
proxy. Direct mutation of pipeline parameters after
initialization is prevented, ensuring runtime consistency
across all workers.

Interface Delegation
--------------------
`ProjectContext` is the single object the user interacts
with. All operational calls (flow execution, worker access,
state queries) are delegated to the underlying
`PipelineContext` — the user never touches the internal
layers directly.
"""

from ..deps.types           import  Literal
from ..core.parameters      import  (
                                    PRM,
                                    Parameters, 
                                    Setup, 
                                    )
from ..core.config          import  (
                                    ProjectGenesis, 
                                    PipelineConfigurator
                                    )
from ..core.accessor        import  UI
from ..core.io              import  ProjectIOCleaner
from ..models               import  (
                                    fitters,
                                    models,
                                    dashboards
                                    )
from .pipeline              import  PipelineContext


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PROJECT CONTEXT - BOOTLOADER AND CONFIG GATEWAY
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
SETUP = '_setup'

def _getattribute(instance:type):
    return object.__getattribute__(instance, SETUP) 

class _SetupProxy(Parameters):
    """
    A transparent proxy object that delegates attribute access to an internal 
    `Parameters` instance while respecting dynamic updates of Parameters in the parent.
    
    Parameters
    ----------
    context : ProjectContext
        The parent instance ProjectContext that holds the actual `_setup` attribute.            
    """    
    def __init__(self, context:"ProjectContext"):
        setattr(self, SETUP, context)
    
    def __getattribute__(self, name):
        context:"ProjectContext" = _getattribute(self)
        setup:Parameters = _getattribute(context)    
        if name == SETUP:
            return setup
        return getattr(setup, name)
    
    def __repr__(self) -> str:
        return repr(getattr(self, SETUP))
    
class ProjectSetup:
    """
    A Descriptor that implements a lazy-loaded, read-only cache for the `_SetupProxy` instance.
    """
    def __init__(self):
        self.cache_name = None

    def __set_name__(self, owner:"ProjectContext", name:str) -> None:
        self.cache_name = f'{name}_cache'
        
    def __get__(self, instance, owner) -> None|_SetupProxy:
        if instance is None:
            return self
        
        cached = self._get_cache(instance)
        if self._is_same(instance, cached):
            return cached
        
        self._set_cache(instance)
        return self._get_cache(instance)
    
    def __set__(self, instance, owner) -> None:
        raise AttributeError('AIG SAFETY ► Defined Project Context is read only.')
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_cache(self, instance:"ProjectContext") -> None|_SetupProxy:
        return instance.__dict__.get(self.cache_name)
    
    def _set_cache(self, instance:"ProjectContext"):
        object.__setattr__(instance, self.cache_name, _SetupProxy(instance))        

    @staticmethod
    def _is_same(instance:"ProjectContext", cached: None|_SetupProxy) -> bool:
        if cached is None:
            return False
        return _getattribute(instance) is _getattribute(cached)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ProjectContext:
    """
    Project Context for defining, configuring AIG pipeline parameters and accessing 
    the key engine context components.
    
    This class acts as the entry point for both interactive UI sessions and 
    headless script execution. It manages the initialization of the user interface 
    (if enabled), collects configuration parameters, and delegates the creation 
    of the runtime context (`PipelineContext`).
    
    It supports a dual-mode architecture:
    1. **UI Mode (HTML)**: Launches `ProjectGenesis` and `PipelineConfigurator` 
       web panels for interactive parameter tuning.
    2. **Script Mode**: Uses dictionary-based arguments passed directly to the setup.

    Parameters
    ----------
    config : Config
        The global configuration object containing UI mode, theme, and other settings.

    css_id : str, default='v1'
        A unique ID suffix used to generate CSS panel identifiers 
        (e.g., 'gateway-v1', 'config-v1').

    Methods
    -------
    genesis(theme, kind, **kwargs)
        Initializes the interactive gateway or sets up script-mode parameters.

    params(theme, **kwargs)
        Launches the configuration UI or applies script-mode parameters, then 
        initializes the runtime context.

    get_setup()
        Returns the final merged `Parameters` object used for training.

    Examples
    --------    
    Interactive usage:

    ```python
    ctx = ProjectContext()
    ctx.genesis(kind='v')  # Launches gateway in vertical style
    ctx.params()           # Launches config panel
    ```

    Script usage:

    ```python
    ctx = ProjectContext()
    ctx.genesis(
        project      = 'test',
        input_data   = 'test.csv',
        target       = 'target',
        yy_mm        = 'decision_date',
        yy_mm_format = '%Y-%m-%d'
    )
    ctx.params(
        min_bin_share = 10,
        gb_strategy   = 0, 
        gb_max_leaves = 50
    )
    ```
    """    
    setup = ProjectSetup()
    ctx = None
    
    def __init__(self, css_id='v1'):
        self.config = UI
        self.theme  = self.config.theme
        self.css_id = css_id
    
    def __getattr__(self, name:str):
        return getattr(self.ctx, name, None)

    def __repr__(self):
        return PipelineContext.help()
    
    @staticmethod
    def _adjust_ansi_kw(kwargs:dict) -> dict:
        verbose   = PRM.OUTPUT.verbose
        v_kwargs  = kwargs.get(verbose.param, verbose.value)
        v_allowed = dict(verbose = min(int(v_kwargs), 1))
        return kwargs | v_allowed
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def genesis(self, theme:str=None, kind:Literal['v','h']='v', **kwargs) -> None:
        """
        Initializes the gateway layer parameters.
        
        In the UI mode, this launches the `ProjectGenesis` web panel to handle
        initial data loading or model selection. In script mode, it simply
        captures the provided `kwargs` for later use.

        Parameters
        ----------
        theme : str, optional
            Custom theme name for the UI. Defaults to `self.theme`.

        kind : {'v', 'h'}, default='h'
            Layout orientation for the gateway: 'v' for vertical, 'h' for horizontal.

        **kwargs : dict
            Configuration arguments passed in a script mode.
        """        
        if self.config.is_html():
            self.gen = ProjectGenesis(
                theme or self.theme, 
                kind  = kind, 
                setup = kwargs,
                panel_css_id = f'gateway-{self.css_id}'
            )
            self.gen.show()        
            self.kw_gen = lambda: self.gen.get_config()
        else:
            self.kw_gen = lambda: kwargs
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def params(self, theme:str=None, **kwargs) -> None:
        """
        Launches the configuration UI interface or applies parameters directly.
        
        This step merges the base gateway configuration with the specific
        hyperparameters or pipeline settings.

        The direct parametrization by `kwargs` is passed to `Parameters` for
        validation and default values. The created Parameters instance is
        set as the global configuration to the context scope as `setup`.
        
        After processing, it acts as a transparent proxy to the `PipelineContext`
        which acts as Centralized Pipeline Context and Global State Coordinator.

        Parameters
        ----------
        theme : str, optional
            Custom theme name for the configuration panel.

        **kwargs : dict 
            Dictionary of parameters to apply in a script mode.
        """        
        if self.config.is_html():
            self.cfg = PipelineConfigurator(
                theme or self.theme,
                panel_css_id = f'config-{self.css_id}',
                callback = self._set_setup,
                setup = kwargs
            )
            self.cfg.show()
        else:
            kw = self._adjust_ansi_kw(kwargs)
            self._set_setup(kw)

        self._set_context()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_setup(self, kw_cfg:dict) -> None:        
        setup: Parameters = Setup(
            **self.kw_gen(),
            **kw_cfg
        ).get_setup()
        setattr(self, SETUP, setup)

    def _set_context(self) -> None:
        self.ctx = PipelineContext(self.setup)
        ProjectIOCleaner(self.setup).clear()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_setup(self) -> Parameters:
        """
        Returns the final validated pipeline Parameters as global config scope for 
        the pipeline steps.        
        """        
        return self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬