"""
Champion Agent Report — Final Model Governance Dashboard.

This module implements the comprehensive reporting interface for
the champion (selected) model agent. It bridges the model's raw
statistical output and the visualization layer, producing a
multi-section dashboard that serves as the primary governance
artifact for model approval.

Report Sections
---------------
1. **Model Performance**: Gini, IV, KS metrics per sample
   (train/valid/test) — the headline scorecard.
2. **Probability Cutoffs**: Confusion matrix at the optimal
   threshold on the test sample.
3. **Model Calibration**: Predicted vs. actual probability
   alignment (reliability diagram).
4. **PSI Overview**: Population Stability Index per feature —
   temporal drift detection.
5. **Model Summary**: Aggregate statistics (feature counts,
   coefficient ranges).
6. **Feature Correlation**: Correlation matrix on the test
   sample.
7. **Feature Importance**: Per-feature contribution ranking.
8. **Feature Predictive Strength**: Individual feature
   discriminative power (per-feature Gini/IV).
9. **Feature Temporal Stability**: Per-feature stability
   scores over time.
10. **Final Summary**: Color-coded verdict banner (logged
    via `ReportLogger`).
"""

from ....deps.common            import  Path
from ....deps.data              import  pDF
from ....core.parameters        import  (
                                        Parameters, 
                                        AIGFLOW
                                        )
from ....core.registry          import  (
                                        DashRegister,
                                        ProjectModelRegistry
                                        )
from ....core.foundation        import  AigParam
from ....core.io                import  DashIO
from ....core.display           import  (
                                        DashInterface, 
                                        DashBackground
                                        )
from ....core.accessor          import  UI, Rgb, Format

from .champion_predictor        import  (
                                        REG,
                                        Fitter,
                                        AgentPredictor
                                        )
from .stats                     import  (
                                        SMPL,
                                        DashStats,
                                        DashStatsCreator,
                                        ModelPerformance,
                                        ModelCalibration,
                                        ModelStability,
                                        FeaturesStability,
                                        ModelSummary,
                                        FeaturesCorrelation,
                                        FeaturesPredictiveness
                                        )

MODEL = AIGFLOW.Selection.champion_agent
DREG  = DashRegister(MODEL)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → PRODUCTION AGENT DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬   
@DREG.register(REG.AIG)
class AgentPredictorDash(DashInterface):
    """
    Champion Agent Dashboard — Full Model Governance Report.

    Implements the complete reporting protocol for the
    selected (champion) model agent. Produces a multi-section
    dashboard covering performance, calibration, stability,
    feature analysis, and a final verdict summary.

    Report Sections (in display order)
    ----------------------------------
    1. Model Performance Overview (Gini, IV, KS per sample).
    2. Probability Cutoffs on Test (confusion matrix at
       optimal threshold).
    3. Model Calibration Overview (reliability diagram).
    4. PSI Overview (feature-level temporal stability).
    5. Model Summary Statistics (aggregate metrics).
    6. Feature Correlation on Test (correlation matrix).
    7. Feature Importance (sensitivity ranking).
    8. Feature Predictive Strength (per-feature Gini/IV).
    9. Feature Temporal Stability (per-feature over time).
    10. Final Summary (color-coded verdict + logging).

    Verbosity Behavior
    ------------------
    - `QUIET`: No display, no save.
    - `INFO`: Text summary only (final verdict banner).
    - `GUI`: Full visual rendering of all sections.

    Parameters
    ----------
    fitter : Fitter
        The fitted champion agent fitter (source of all
        statistics).
    train_set : pDF
        The training data (for statistical computation).
    test_set : pDF
        The test data (for OOS statistics).
    verbose : int | bool, default=True
        Display level.
    """  
    
    def __init__(self, 
        fitter:     Fitter,
        train_set:  pDF,
        test_set:   pDF,
        verbose:    int|bool = True,
    ):
        super().__init__(fitter, verbose)

        s = DashStatsCreator(fitter, self.setup)
        self.stats = s.create(train_set, test_set)

        self._gb = '' if not self.stats.is_gbm else f' | BY WOE VIEW'
        self._sa = 'Train' if not self.stats.is_gbm else 'Valid'

        self._set_elements({
            'summary'    : 'correlation_matrix',
            'psi'        : 'dash_grid_fe_psi',
            'stability'  : self._create_elements(),
            'predictive' : 'feature_predictivness'
        })
    
    def _create_elements(self):    
        return {
            tx: f'dash_grid_stability_{tx.split(' ')[0]}'.lower()
            for tx in self.stats.samples
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def performance(self):
        if not self._is_gui:
            return
        DashUtils.title('MODEL PERFORMANCE OVERVIEW', style = dict(t=0)) 
        p = ModelPerformance(self.stats.diagnostics)
        p.graph()
        print()
        p.tables()
    # ─────────────────────────────────────────────────────────────────────────────────────────────        
    
    def confusion_mx(self):
        if not self._is_gui:
            return        
        o = f'OPTIMAL CUTOFF {self.stats.optim_cutoff:.4f}'
        DashUtils.title('PROBABILITY CUTOFFS ON TEST SAMPLE | ' + o)        
        self.stats.confusion_mx.show()
    # ─────────────────────────────────────────────────────────────────────────────────────────────        
    
    def calibration(self):
        if not self._is_gui:
            return        
        DashUtils.title('MODEL CALIBRATION OVERVIEW')
        c = ModelCalibration(self.stats.calibration)
        c.graph()
        print()
        c.table()
    # ─────────────────────────────────────────────────────────────────────────────────────────────       

    def psi(self, utils: "DashUtils"):
        title = f'PSI OVERVIEW → CALIBRATED BY ' + utils.get_sample()
        path, is_load, is_save = utils.resolve_state('psi', title)
        if not path:
            return
        s = ModelStability(self.stats.psi, self._sa, path, is_load, is_save)
        s.dash()
    # ─────────────────────────────────────────────────────────────────────────────────────────────       

    def model_summary(self):
        if not self._is_gui:
            return
        DashUtils.title('MODEL SUMMARY STATISTICS', style = dict(b=30))   
        ModelSummary.show(self.stats.summary)

    def correlation(self, utils: "DashUtils"):
        title = 'FEATURE CORRELATION ON TEST SAMPLE'
        path, is_load, is_save = utils.resolve_state('summary', title, dict(t=30, b=10))
        if not path:
            return
        c = FeaturesCorrelation(self.stats.correlation, path, is_load, is_save)
        c.graph()
        c.table()        
    # ───────────────────────────────────────────────────────────────────────────────────────────── 

    def fe_importance(self):
        if not self._is_gui:
            return
        DashUtils.title('MODEL SENSITIVITY TO FEATURE DRIVERS ON TEST SAMPLE')
        self.stats.importance[SMPL.TE].show_dash()
    # ───────────────────────────────────────────────────────────────────────────────────────────── 
    
    def fe_predictive(self, utils: "DashUtils"):
        title = 'FEATURE PREDICTIVE STRENGTH OVERVIEW' + utils.get_gb_woe()
        path, is_load, is_save = utils.resolve_state('predictive', title)
        if not path:
            return        
        w = FeaturesPredictiveness(self.stats.predictive, path, is_load, is_save)
        w.grid()
        w.table()
    # ───────────────────────────────────────────────────────────────────────────────────────────── 

    def fe_stability(self, utils: "DashUtils"):
        title = 'FEATURE TEMPORAL PREDICTIVE STABILITY OVERVIEW' + utils.get_gb_woe()
        path, is_load, is_save = utils.resolve_state('stability', title)
        if not path:
            return        
        s = FeaturesStability(self.stats.stability, path, is_load, is_save)
        s.grid()
        s.table()
    # ───────────────────────────────────────────────────────────────────────────────────────────── 

    def _get_result(self):
        d = self.stats.diagnostics
        r = lambda x: f': Gini: {d[x].gini:.3} | IV: {d[x].IV:.3f} | KS: {d[x].KS:.3f}'
        c = [Rgb.bold_cyan, Rgb.bold_purple]
        t = [f'{c[i](x)} {r(x)}' for i, x in enumerate(d)]
        return ' → '.join(t)

    def summary(self, _is_view=True):
        d = '\n' if not UI.is_html() else ''
        m = ' '.join([
            Rgb.bold(f'► Champion Agent → {len(self.stats.features)} Selected Features'),
            f'{d}► Performance Overview →',
            self._get_result()
        ])
        self._final_summary(m=m, is_view=_is_view)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @AigParam.with_setattr_runtime(verbose=None)    
    def build(self, save=None, custom_path=None, **kwargs):
        """Construct dashboard with option to save it to default project path or custom path."""
        paths = self.save(save, custom_path, save_signature = True)
        utils = DashUtils(self, save, paths)
        self.performance()
        self.confusion_mx()
        self.calibration()
        self.psi(utils)        
        self.model_summary()        
        self.correlation(utils)    
        self.fe_importance()        
        self.fe_predictive(utils)
        self.fe_stability(utils)
        self.summary(kwargs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashUtils:
    """
    Dashboard Rendering Utilities — State Resolution & Display Helpers.

    Provides a uniform interface for:
    - **Path resolution**: Determines whether a dashboard
      element should be loaded (previously saved + valid
      signature) or rendered fresh (temp mode).
    - **Title display**: Consistent section header formatting
      with configurable margins.
    - **Context flags**: Derives WOE-view context and sample
      name from the dashboard's statistics.

    The `resolve_state` method is the key method — it
    encapsulates the full decision logic for a given element:
    whether to display, whether to load from saved files or
    render fresh, and which path to use.

    Parameters
    ----------
    dash_instance : DashInterface
        The parent dashboard (provides stats, verbosity,
        and save flag).
    save : bool | None
        The effective save flag (resolved from the dashboard).
    paths : dict[str, Path]
        The resolved element paths (from `DashIO.save()`).
    """
    def __init__(self, 
        dash_instance: DashInterface, 
        save: bool|None, 
        paths: dict[str, Path]
    ):
        self.dash = dash_instance        
        self.stats: DashStats = self.dash.stats
        self.is_save = dash_instance._is_save(save)
        self.paths = paths

    def get_gb_woe(self) -> str:
        """Return the WOE-view context suffix."""        
        return '' if not self.stats.is_gbm else f' | BY WOE VIEW'

    def get_sample(self) -> str:
        """Return the primary sample name (uppercased)."""        
        return self.stats.samples[0].upper()

    @classmethod
    def title(cls, title:str, style:dict=None) -> None:
        """
        Display a section title header.

        Parameters
        ----------
        title : str
            The section title text.
        style : dict | None, default=None
            Style overrides         
            - t : int, default=40 - Top margin.
            - b : int, default=25 - Bottom margin.
        """        
        style_ = cls.style(**(style or {}))
        Format.title(title, style = style_)

    @staticmethod
    def style(t:int=40, b:int=25):
        """
        Create a standard title style.

        Parameters
        ----------
        t : int, default=40 - Top margin.
        b : int, default=25 - Bottom margin.

        Returns
        -------
        Format.StyleStats
            The style object.
        """        
        return Format.StyleStats(margin_t=t, margin_b=b)
               
    def resolve_state(self, element: str, title:str, style:dict=None
    ) -> tuple[Path | None, bool | None, bool | None]:
        """
        Resolve the display state for a dashboard element.

        The key decision method. Determines:
        1. Whether to display at all (verbosity check).
        2. Whether to show the section title (GUI mode).
        3. Whether to load from saved file or render fresh.
        4. The path to use.

        Parameters
        ----------
        element : str
            The element name (key in the paths dict).
        title : str
            The section title to display (GUI mode only).
        style : dict | None, default=None
            Title style overrides.

        Returns
        -------
        tuple[Path | None, bool | None, bool | None]
            - `[None]*3`: No action → exit
            - `(path, is_load, is_save)`: The resolved path,
            load flag, and save flag.
        """        
        is_na, is_save = self._is_na_is_save()
        if is_na:
            return [None]*3
        
        if self.dash._is_gui:            
            style_ = self.style(**(style or {}))
            Format.title(title, style=style_)

        is_load, path = self._resolve_path(self.paths, element)
        
        return path, is_load, is_save
    
    @staticmethod
    def _resolve_path(paths: dict, element: str):
        """Resolve the path and load flag for a single element."""
        return DashIO.resolve_path(paths, element)

    def _is_na_is_save(self) -> tuple[bool, bool]:
        """Determine no-action and save-only modes."""        
        is_save_only = self.is_save and not self.dash.verbose
        is_no_action = not (self.dash._is_gui or is_save_only)
        return is_no_action, is_save_only    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class AgentPredictorDashWaiter:
    """
    Final Model Governance Report — Async-Aware Display.

    Triggers the visualization of the champion agent's
    dashboard. Handles the async case: if the report was
    build as a background process (still building), this
    class waits for completion before rendering.

    Use Case
    --------
    When the report was created with background execution, 
    it must wait for the build to finish before displaying 
    results. This class encapsulates that synchronization 
    logic.

    Display Modes
    -------------
    - **Sync** (`sync=None`): The model is already fully
      fitted → show the stored report immediately.
    - **Async** (`sync=Event`): The model is still
      building → launch a background process that waits
      for the event, then shows the report.

    Parameters
    ----------
    setup : Parameters
        The AIG flow parameter context (provides project
        signature, verbosity, and model registry access).
    """
    def __init__(self, setup: Parameters):
        self.setup = setup

    def _get_model(self) -> AgentPredictor:
        return ProjectModelRegistry.get_model(self.setup.project_signature, MODEL)

    def _show(self, ar_kw: tuple[tuple[AgentPredictor, bool|int], dict]):
        (model, verbose), _ = ar_kw 
        model.report._show(
            verbose = verbose,
            with_signature = False
        )

    def show(self, verbose: bool|int = None) -> None:
        """
        Trigger the visualization of the Champion Agent report.

        Resolves the display mode:
        - If the report is ready → immediate display.
        - If the report is still building (sync=Event) →
          background wait + display.

        Parameters
        ----------
        verbose : bool | int | None, default=None
            Verbosity override. If None, uses the flow
            global verbosity level.
        """
        verbose_ = AigParam.di(verbose, self.setup.verbose)
        
        if not verbose_:
            return
        
        model = self._get_model()
        args  = (model, verbose_)

        message = 'Agent Report Build in Progress'
        DashBackground.run_with_background_sync(
            fx      = self._show, 
            fitter  = model.fx.fitter, 
            message = message, 
            args    = args,
            daemon  = False, 
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬