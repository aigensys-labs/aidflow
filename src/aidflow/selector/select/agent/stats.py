"""
Champion Agent Report — Final Production Assessment.

This module consolidates all performance metrics, stability analyses, and
visual diagnostics into a single, regulator-ready report for the production
Champion model. It serves as the definitive source of truth for:

- **Regulatory Compliance**: Provides auditable evidence of model
  performance (Gini, KS, IV), calibration accuracy, and population
  stability (PSI) across Train, Valid, and Test samples.
- **Analytical Assessment**: Delivers deep-dive feature-level analyses
  including importance ranking (SHAP/Permutation), WOE predictiveness,
  inter-feature correlation, and temporal stability checks.
- **Decision Support**: Surfaces the optimal probability cutoff with
  surrounding sensitivity analysis (FPR/TPR tradeoff) to inform
  deployment thresholds.

The report is structured as a two-phase system:

1. **Computation Phase** (`DashStatsCreator`): Aggregates all metrics
   from the fitted model and raw datasets into a typed `DashStats`
   container.
2. **Presentation Phase** (Individual Report Classes): Renders the
   pre-computed statistics into interactive Plotly figures, styled HTML
   tables, and combined PDF-ready reports.

Integration:
    This module is the terminal consumer of the AIDflow pipeline. It
    reads from the `ProjectContext` [13], accesses fitted models via
    the `ModelRegistry` [11], and leverages the Worker-Model-Fitter
    architecture to retrieve all trained components [9].
"""

from ....deps.common                            import  Path
from ....deps.data                              import  np, pDF, pSe
from ....deps.plots                             import  go
from ....deps.data_view                         import  itables_show
from ....deps.types                             import  Literal

from ....core.parameters                        import  (
                                                        Parameters, 
                                                        AIGFLOW
                                                        )
from ....core.registry                          import  ProjectModelRegistry
from ....core.factory                           import  (
                                                        Dataclass,
                                                        DictClass
                                                        )
from ....core.display                           import  (
                                                        Display,
                                                        DashPlotlyLink
                                                        )
from ....core.interop                           import  Pd
from ....core.standard                          import  VarStd

from ....ml.sampling                            import  Subsampling
from ....ml.features                            import  Correlation
from ....ml.features.woe                        import  (
                                                        OV,
                                                        WoEOverview
                                                        )
from ....ml.metrics                             import  (
                                                        LBL as LBL_D,
                                                        DiagnosticsResult,
                                                        SeparationDiagnostics, 
                                                        GiniLiftPlot,
                                                        ConfusionMatrixPlot
                                                        )
from ....ml.models.calibration                  import  (
                                                        LBL as LBL_C,
                                                        CalibrationAssessment,
                                                        CalibrationPlot
                                                        )
from ....ml.models.psi                          import  (
                                                        PopulationStability,
                                                        PopulationStabilityPlot,
                                                        PopulationStabilityResult
                                                        )
from ....ml.features.stability                  import  (
                                                        LBL  as LBL_S,
                                                        STAB as STAB_S,
                                                        DASH as DASH_S,
                                                        FeConsistency,
                                                        FEConsistencyPlot
                                                        )

from ....data.encode.woe_stabilizer             import  (
                                                        FitParams    as WOE_FitParams,
                                                        FitParamsCfg as WOE_FitParamsCfg,
                                                        WoeStabilizer
                                                        )
from ....learner.features.importance            import  (
                                                        FeatureImportance,
                                                        FeatureImportanceResult
                                                        )

from ....storage.export                         import  Export
from ....display.style                          import  (
                                                        TableStyle, 
                                                        ItablesStyle
                                                        )

from .champion_predictor                        import  Fitter
from .fitter                                    import  (
                                                        FitParams, 
                                                        FitDashLink,
                                                        PredictorFitter
                                                        )


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class SMPL(DictClass):
    """Sample label - train, valid, test - constants for report sections."""
    TR = 'Train Sample'
    VA = 'Valid Sample'
    TE = 'Test Sample'

class PRF(DictClass):
    """Performance metrics column constants."""
    G  = 'Gini'
    IV = 'IV'
    KS = 'KS'
    PB = 'ProbaBIN'
    T  = 'E(Target)'
    L  = 'E(Lift)'
    W  = 'Weight'
    C  = 'Count'

class CON(DictClass):
    """Confusion Matrix / Cutoff analysis column constants."""
    T   = 'E(T|P>Cutoff)'
    TR  = 'E(T|P<Cutoff)'
    PC  = 'Prob. Cutoff'
    CS  = 'Popul. Cut'
    FPR = 'FPR'
    TPR = 'TPR'

class CAL(DictClass):
    """Calibration assessment column constants."""
    PB = 'ProbaBIN'
    P  = 'E(Proba)'
    T  = 'E(Target)'
    W  = 'Weight'
    C  = 'Count'

class STB(DictClass):
    """Feature stability column constants."""
    F  = 'Feature'
    S  = 'Stability'
    IS = 'Stable'
    D  = 'DriftΔ'

@Dataclass.base
class DashStats:
    """
    Typed container for all Champion Report statistics.

    Serves as the single structured result object produced by
    `DashStatsCreator.create()`. Contains all pre-computed metrics,
    distributions, and analysis results needed by the presentation layer.

    Attributes:
        is_gbm (bool): Whether the champion model is a GBM (affects
            which sample is used for diagnostics and importance methods).
        samples (list): Active samples with respect to GBM / GLM.
            GLM[train, test] / GBM[valid, test]
        features (list): Ordered list of model features.
        diagnostics (dict): Separation diagnostics per sample
            (Gini, KS, IV, Lift distribution, PR curve).
        optim_cutoff (float): The optimal probability cutoff for
            classification decisions.
        confusion_mx (go.Figure): Pre-rendered confusion matrix plot.
        calibration (dict): Per-sample calibration bin tables.
        psi (PopulationStabilityResult): PSI analysis results
            (model-level and feature-level).
        summary (object): Model coefficient/weight summary table.
        correlation (pDF): Feature correlation matrix.
        importance (dict): Per-sample feature importance results.
        predictive (dict): Per-sample WOE bin predictiveness tables.
        stability (dict): Per-sample feature temporal stability results.
    """    
    is_gbm:         bool
    samples:        list[str]
    features:       list[str]
    diagnostics:    dict[str, DiagnosticsResult]
    optim_cutoff:   float
    confusion_mx:   go.Figure
    calibration:    dict[str, pDF]
    psi:            PopulationStabilityResult
    summary:        object
    correlation:    pDF
    importance:     dict[str, FeatureImportance]
    predictive:     dict[str, pDF]
    stability:      dict[str, dict]

@Dataclass.base
class DashData:
    """Internal container for raw training and test datasets."""
    train: pDF
    test:  pDF
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class StatsUtils:
    """
    Static utility helpers for Champion Report sample handling.

    Provides small, reusable functions for sample name resolution,
    feature alignment (WOE mapping for GBM), and temporary path
    generation for report artifacts.
    """

    @staticmethod
    def smpl_tv(is_gbm:bool) -> str:
        """Resolve the primary sample label based on model type."""
        return SMPL.VA if is_gbm else SMPL.TR
    
    @staticmethod
    def align_fes(fes: list, is_gbm:bool) -> list:
        """Align feature names to their WOE-encoded equivalents for GBM models."""
        if not is_gbm:
            return fes
        return VarStd.var_to_woe(fes)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashStatsCreator:
    """
    Orchestrator for all Champion Report metric computations.

    This is the primary computation engine that transforms raw datasets
    and a fitted Champion model into a complete `DashStats` container.
    It coordinates the execution of all sub-analyses:

    1. **Prediction**: Generates probability scores for both samples.
    2. **Diagnostics**: Computes Gini, KS, IV, Lift, and PR curves.
    3. **Cutoff**: Determines the optimal probability threshold.
    4. **Confusion Matrix**: Evaluates classification accuracy at the
       optimal cutoff with sensitivity analysis.
    5. **Calibration**: Assesses probability reliability per bin.
    6. **Stability (PSI)**: Measures population drift at model and
       feature levels.
    7. **Summary**: Extracts model coefficients/weights.
    8. **Correlation**: Computes inter-feature correlation matrix.
    9. **Importance**: Ranks features by predictive contribution.
    10. **Predictiveness**: Evaluates WOE bin-level target separation.
    11. **Feature Stability**: Checks temporal consistency of features.

    Parameters:
        fitter (Fitter): The fitted Champion Fitter instance from the
            pipeline. Contains the trained model, diagnostics, and
            train/validation split indices.
        setup (Parameters): The global project parameters including
            bin count, alpha level, and project signature.

    Example:
        >>> creator = DashStatsCreator(fitter=champion_fitter, setup=project_setup)
        >>> stats = creator.create(train_set=train_df, test_set=oot_df)
        >>> stats.diagnostics  # Per-sample Gini, KS, IV, Lift
        >>> stats.stability    # PSI results
    """

    def __init__(self, fitter: Fitter, setup: Parameters):
        model: PredictorFitter = fitter.model.model
        params: FitParams = model.params
        dash: FitDashLink = model.dash
        
        self.target  = params.target
        self.is_gbm  = model.is_gbm   # Diag. data → GBM valid part | GLM full train        
        self.diag    = fitter.stats.diagnostics
        self.idx     = dash.train_val_idx

        self.setup    = setup
        self.fitter   = fitter
        self.features = model.features
        
        self.project_sig = setup.project_signature

        self.aligned_fes = StatsUtils.align_fes(self.features, self.is_gbm)
    # ─────────────────────────────────────────────────────────────────────────────────────────────        

    def _active_smpl(self) -> list[str]:
        """Returns list of active samples with respect to GLM / GBM sample use."""
        tv = StatsUtils.smpl_tv(self.is_gbm)
        return [tv, SMPL.TE]
    
    def _smpl_def(self, *smpl) -> dict:
        """Build the standard sample dictionary with resolved labels."""
        lbl = self._active_smpl()
        return {
            lbl[0]: smpl[0],
            lbl[1]: smpl[1]
        }
    
    def _samples(self, data: DashData) -> dict[str, pDF]:
        """Resolve the two primary samples from the data container."""
        return self._smpl_def(
            data.train.loc[self.idx], data.test
        )

    def _subsamples(self, data: DashData) -> dict[str, pDF]:
        """Resolve the two primary samples from the data container with subsampling."""
        d = self._samples(data)
        r = {}
        for k, s in d.items():
            r[k] = Subsampling(s, s[self.target]).univar()[0]
        return r

    def _predict(self, data: DashData) -> tuple[dict[str, pSe], dict[str, pSe]]:
        """Generate probability predictions for both samples."""
        smpl = self._samples(data)
        y, p = {}, {}
        for s, d in smpl.items():
            y[s] = d[self.target]
            p[s] = pSe(self.fitter.predict_proba(d)[:, 1], index=d.index)
        return y, p
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
        
    def _diagnostics(self, y_true: pSe, y_pred: pSe) -> dict[str, DiagnosticsResult]:
        """
        Compute separation diagnostics for the test sample.

        Combines the pre-computed training diagnostics with fresh
        test-sample evaluation.        
        """
        dg = SeparationDiagnostics(lift_bins = 10).evaluate(y_true, y_pred)
        return self._smpl_def(self.diag, dg)

    @staticmethod
    def _get_optim_cutoff(d: dict[str, DiagnosticsResult]) -> float:
        """Extract the optimal probability cutoff from the test PR curve."""
        return SeparationDiagnostics.get_pr_optim_cutoff(d[SMPL.TE].PR)

    @staticmethod
    def _confusion_mx(
        y_true: pSe, y_pred: pSe, diag: dict[str, DiagnosticsResult], optim_cut: float
    ) -> go.Figure:
        """Create the confusion matrix and cutoff sensitivity visualization."""
        return ConfusionMatrix(optim_cut).create(y_true, y_pred, diag[SMPL.TE])

    def _calibration(self, target: dict[str, pSe], pred: dict[str, pSe]) -> dict[str, pDF]:
        """Compute probability calibration tables for both samples."""
        return ModelCalibration.process(self.setup, target, pred)

    def _psi(self, data: DashData, pred: dict[str, pSe]) -> PopulationStabilityResult:
        """Compute model and feature level PSI stability analysis."""
        s = PopulationStability(self.fitter.model.model, self.target, self.setup.select_bins)
        return s.create(*self._subsamples(data).values(), *pred.values())

    def _summary(self) -> pDF:
        """Generate the model coefficient/weight summary table."""
        s = ModelSummary(self.fitter)
        return s.get()

    def _correlation(self, data: DashData) -> pDF:
        """Compute the inter-feature correlation matrix."""
        c = FeaturesCorrData(self.features)
        return c.create(data.test)

    def _importance(self, data: DashData, diag: dict[str, DiagnosticsResult]
    ) -> dict[str, FeatureImportanceResult]:
        """Compute per-sample feature importance rankings."""
        i = FeaturesImportance(self.fitter, self.target, self.features, diag, self.is_gbm)
        return i.fit( self._samples(data), data.train )

    def _predictive(self, data: DashData) -> dict[str, pDF]:
        """Compute WOE bin-level predictiveness tables."""
        w = WoEOverview(self.target, self.aligned_fes, self.setup.select_alpha)
        return w.create( self._samples(data) )

    def _stability(self, data: DashData) -> dict[str, dict]:
        """Compute temporal feature stability (consistency) per sample."""
        s = FeaturesStabilityPlot(self.target, self.aligned_fes, self.project_sig)
        return s.create( self._samples(data) )    
    # ─────────────────────────────────────────────────────────────────────────────────────────────    

    def create(self, train_set:pDF, test_set:pDF) -> DashStats:
        """
        Execute the full Champion Report computation pipeline.

        This is the primary entry point. It runs all 11 sub-analyses
        in sequence and packages the results into a single typed
        `DashStats` container ready for presentation.

        Parameters:
            train_set (pDF): The training (or validation) dataset
                containing all model features and the target column.
            test_set (pDF): The out-of-time test dataset with the
                same schema as `train_set`.

        Returns:
            DashStats: The complete, pre-computed statistics container
                for the Champion Report. All fields are populated and
                ready for the presentation-layer classes to render.

        Example:
            >>> stats = creator.create(train_df, oot_df)
            >>> stats.diagnostics['Test Sample'].gini   # 0.58
            >>> stats.optim_cutoff                      # 0.342
            >>> stats.stability.model_psi[0]            # {'Vol_PSI': 0.06, ...}
        """        
        dd    = DashData(train = train_set, test = test_set)
        y, p  = self._predict(dd)
        yp_te = y[SMPL.TE], p[SMPL.TE]

        dg = self._diagnostics(*yp_te)
        oc = self._get_optim_cutoff(dg)
        cx = self._confusion_mx(*yp_te, dg, oc)
        ca = self._calibration(y, p)
        st = self._psi(dd, p)
        su = self._summary()
        co = self._correlation(dd)
        im = self._importance(dd, dg)
        bn = self._predictive(dd)
        fs = self._stability(dd)

        return DashStats(
            is_gbm          = self.is_gbm,
            samples         = self._active_smpl(),
            features        = self.features,
            diagnostics     = dg,
            optim_cutoff    = oc,
            confusion_mx    = cx,
            calibration     = ca,
            psi             = st,
            summary         = su,
            correlation     = co,
            importance      = im,
            predictive      = bn,
            stability       = fs,
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
        
class ModelPerformance:
    """
    Model separation performance report renderer.

    Displays the Gini, KS, and IV metrics alongside the Lift
    distribution chart and per-bin performance tables for each
    sample (Train/Valid and Test).

    This is the primary "is the model good?" section of the Champion
    Report, providing regulators with the standard separation metrics.

    Parameters:
        diag (dict): Per-sample `DiagnosticsResult` objects from
            `DashStats.diagnostics`.

    Example:
        >>> perf = ModelPerformance(stats.diagnostics)
        >>> perf.graph()   # Renders Gini + Lift dual chart
        >>> perf.tables()  # Renders metrics table + per-bin lift tables
    """

    def __init__(self, diag: dict[str, DiagnosticsResult]):
        self.diag = diag

    def graph(self) -> None:
        """
        Render the Gini and Lift performance chart.

        Displays a dual-panel figure showing the cumulative Gini curve
        and the per-bin lift distribution for both samples, with
        sample labels in the chart titles.

        The figure is rendered inline (e.g., in a Jupyter notebook or
        Dash report).
        """        
        t = GiniLiftPlot.title_helper(self.diag)
        titles = [
            f'Gini Performance → {t[LBL_D.G]}',
            f'Lift Performance → {t[LBL_D.L]}'
        ]
        colors = ['rgb( 33, 115, 115)', 'rgb(165,  75, 155)']
        GiniLiftPlot(self.diag, plot_titles=titles, theme=colors).create()

    def _perf(self, perf: dict) -> pDF:
        """Format the summary metrics into a styled comparison table."""
        p = pDF(perf, index=list(self.diag)).T
        return Pd.style(p, format = dict(precision = 4))
    
    @staticmethod
    def _lift(lift: pDF) -> pDF:
        """Format a single sample's lift distribution into a styled table."""
        f = '{:.4f}'
        p = '{:.2%}'
        l = (
            lift.reset_index()
            .rename(columns={
                LBL_D.P: PRF.PB,
                LBL_D.T: PRF.T,
                LBL_D.L: PRF.L,
                LBL_D.W: PRF.W,
                LBL_D.C: PRF.C,
            })
        )
        return Pd.style(
            l, 
            hide_axis   = 0,
            format_cols = {PRF.T: p, PRF.L: f, PRF.W: p}
        )
    
    def tables(self) -> None:
        """
        Render all performance tables: summary metrics + per-bin lift.

        Produces a combined HTML report containing:
        1. A summary table with Gini, IV, and KS for each sample.
        2. A per-bin lift table for each sample showing probability bin,
           expected target rate, lift, weight, and count.

        The tables are rendered inline with center-aligned titles.
        """        
        perf, lift = [], []
        for d in self.diag.values():
            perf.append({PRF.G: d.gini, PRF.IV: d.IV, PRF.KS: d.KS})
            lift.append( self._lift(d.lift) )

        TableStyle.combine_pdf_to_html(
            pdf_tables      = [self._perf(perf), *lift], 
            titles          = ['Metrics', *list(self.diag)], 
            tables_per_block= 1, 
            h_align         = 'center', 
            show            = True
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ConfusionMatrix:
    """
    Classification accuracy and cutoff sensitivity report.

    Renders the confusion matrix at the optimal probability cutoff,
    accompanied by a sensitivity table showing the FPR/TPR tradeoff
    across cutoffs in the vicinity of the optimal threshold.

    This section answers: "If we deploy at this threshold, what is
    our precision/recall, and how sensitive is performance to small
    threshold shifts?"

    Parameters:
        optim_cutoff (float): The optimal probability threshold
            derived from the PR curve analysis.

    Example:
        >>> cm = ConfusionMatrix(optim_cutoff=0.342)
        >>> fig = cm.create(y_true, y_pred, test_diagnostics)
    """

    def __init__(self, optim_cutoff: float):
        self.optim_cut = optim_cutoff
    
    @staticmethod
    def _align_data(diag: DiagnosticsResult) -> pDF:
        """Extract and align the PR curve data for the cutoff table."""
        pr = diag.PR

        ipr:pDF = pr[LBL_D.IPR].copy()
        ipr[CON.FPR] = pr[LBL_D.FPR]
        ipr[CON.TPR] = pr[LBL_D.TPR]
        ipr = (
            ipr
            .rename(columns={
                LBL_D.PC : CON.PC, 
                LBL_D.CS : CON.CS, 
                LBL_D.T  : CON.T, 
                LBL_D.TR : CON.TR
            })
        )
        return ipr[[CON.PC, CON.TR, CON.T, CON.CS, CON.FPR, CON.TPR]]
    
    @classmethod
    def _table_data(cls, diag: DiagnosticsResult) -> tuple[dict, int]:
        """
        Prepare the cutoff sensitivity table data.

        Selects cutoffs at ±10 percentile intervals around the optimal
        cutoff to show how performance degrades as the threshold moves.        
        """
        
        ipr = cls._align_data(diag)
        oc  = SeparationDiagnostics.get_pr_optim_cutoff(diag.PR)
        ix  = int(ipr.loc[ipr[CON.PC]==oc].index.values[0])
        idx = list(ipr.sort_index().index)[:-1]
        
        i = idx.index(ix)   
        l = list(range(i, -1, -10)) + list(range(i + 10, len(idx), 10))
        l = sorted(l)
            
        return ipr.map("{:.2%}".format).loc[l].to_dict('list'), l.index(ix)

    def create(self, y_true: pSe, y_pred: pSe, diag: DiagnosticsResult) -> go.Figure:
        """
        Render the complete confusion matrix visualization.

        Produces a dual-panel figure:
        1. **Confusion Matrix**: The 2×2 matrix at the optimal cutoff.
        2. **Cutoff Sensitivity**: A table showing FPR/TPR at cutoffs
           spaced 10 percentile points around the optimal threshold.

        Parameters:
            y_true (pSe): Actual target values for the test sample.
            y_pred (pSe): Predicted probabilities for the test sample.
            diag (DiagnosticsResult): Test-sample diagnostics containing
                the PR curve data.

        Returns:
            go.Figure: The complete confusion matrix figure with
                embedded sensitivity table.
        """        
        po  = (y_pred > self.optim_cut).astype(int)
        t,i = self._table_data(diag)
        st  = [
            f'Confusion Matrix | Cutoff {self.optim_cut} (continuous)',
            'Cutoffs Impact Around Optimal Cutoff (percentile)'
        ]
        return ConfusionMatrixPlot(plot_titles=st).create(y_true, po, table_data=t, row_highlight=i) 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ModelCalibration:
    """
    Probability calibration assessment report.

    Evaluates whether the model's predicted probabilities are reliable:
    if the model predicts 70% probability, do approximately 70% of
    those cases actually convert?

    This is a critical regulatory requirement in many jurisdictions
    (e.g., fair lending, credit scoring) where probability outputs
    are used for decision-making.

    Parameters:
        stats (dict): Per-sample calibration bin dataframes from
            `DashStats.calibration`.

    Example:
        >>> cal = ModelCalibration(stats.calibration)
        >>> cal.graph()  # Renders calibration curve (reliability diagram)
        >>> cal.table()  # Renders per-bin calibration tables
    """

    def __init__(self, stats: dict[str, pDF]):
        self.stats = stats        
        self._smpl = [x for x in stats if x != LBL_C.A]

    @staticmethod
    def process(
        setup:  Parameters, 
        target: dict[str, pSe], 
        pred:   dict[str, pSe]
    ) -> dict[str, pDF]:
        """
        Compute calibration assessment for all samples.

        Parameters:
            setup (Parameters): Project configuration (alpha, bins).
            target (dict): Per-sample actual target series.
            pred (dict): Per-sample predicted probability series.

        Returns:
            dict: Mapping of sample label → calibration bin dataframe
                containing expected probability, actual rate, weight,
        """
        ca  = CalibrationAssessment(setup.select_alpha, setup.select_bins, kind='uniform')        
        cal = {}
        for s, t in target.items():
            cal = cal | ca.process(t, {s: pred[s]}) 
        return cal
    
    def graph(self) -> None:
        """
        Render the calibration reliability diagram.

        Displays a dual-panel figure showing the predicted vs. actual
        probability per bin for both samples. Points falling on the
        diagonal indicate perfect calibration.
        """        
        return CalibrationPlot(self.stats, graph_titles = self._smpl).create()

    @staticmethod
    def _style(cal:pDF) -> pDF:
        """Format a single sample's calibration table for display."""
        f = '{:.4f}'
        p = '{:.2%}'
        c = cal.rename(columns={
            LBL_C.PB: CAL.PB,
            LBL_C.P:  CAL.P,
            LBL_C.T:  CAL.T,
            LBL_C.C:  CAL.C,
            LBL_C.W:  CAL.W,
        })[[CAL.PB, CAL.P, CAL.T, LBL_C.D, CAL.W, CAL.C, LBL_C.V]]
        return Pd.style(
            c, 
            hide_axis   = 0,
            format_cols = {CAL.P: p, CAL.T: p, LBL_C.D: f, CAL.W: p},
            color       = {'subset': LBL_C.V}
        )
        
    def table(self) -> None:
        """
        Render the per-sample calibration tables.

        Produces a combined HTML report with one calibration table
        per sample, showing probability bin, expected probability,
        actual target rate, deviation, weight, and count.
        """        
        cal = []
        for s in self._smpl:
            d = self.stats[s]
            cal.append(self._style(d))

        TableStyle.combine_pdf_to_html(
            pdf_tables      = cal, 
            titles          = self._smpl, 
            tables_per_block= 1, 
            h_align         = 'center', 
            show            = True
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ModelStability:
    """
    Population Stability Index (PSI) report renderer.

    Wraps the pre-computed `PopulationStabilityResult` and renders
    the complete PSI analysis: aggregated summary, model prediction
    distribution chart, feature-level PSI grid, and stability overview
    table.

    This section answers: "Is the model's input distribution and
    predictive behavior stable between the reference and production
    populations?"

    Parameters:
        stability (dict): The stability data from `DashStats.stability`.
        expected (str): The label for the reference sample
            (e.g., 'Train' or 'Valid').
        report_path (Path): Optional path for saving report artifacts.
        is_load (bool): Indicator whether previously created HTML 
            artifacts can be used.
        is_save_only (bool): Set to True to save static HTML object 
            and exit (is_load is ignored).

    Example:
        >>> stab = ModelStability(
        ...     stability = stats.stability,
        ...     expected  = 'Valid'
        ... )
        >>> stab.dash()  # Renders full PSI report
    """

    def __init__(self, 
        stability: dict[str, dict], 
        expected: str, 
        report_path: Path = None,
        is_load: bool = False,
        is_save_only : bool = False
    ):
        self.stab  = stability
        self.exp   = expected
        self.path  = report_path
        self.is_load  = is_load
        self.is_save_only  = is_save_only

    def dash(self) -> None:
        """
        Render the complete PSI report in sequence.

        Executes all PSI visualization components:
        1. Aggregated PSI summary table (Model + Features)
        2. Model prediction stability chart
        3. Feature PSI grid (per-feature distribution + lift)
        4. Feature overview table with stability flags
        """
        psi = PopulationStabilityPlot(
            stability   = self.stab, 
            report_path = self.path, 
            trace_names = [self.exp, 'Test']
        )

        if self.is_save_only:
            psi.fe_grid(False)
            return
        
        psi.dash(self.path if self.is_load else None)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ModelSummary:
    """
    Model coefficient/weight summary renderer.

    Displays the fitted model's parameters in a readable format:
    - **Linear models**: Coefficients, intercept, and standard errors.
    - **GBM models**: Aggregated SHAP-based feature weights.

    This provides auditors with the "what did the model learn?"
    transparency required for regulatory review.

    Parameters:
        fitter (Fitter): The fitted Champion Fitter instance.

    Example:
        >>> summary = ModelSummary(champion_fitter)
        >>> df = summary.get()     # Returns the styled table
        >>> summary.show(df)       # Renders inline
    """

    def __init__(self, fitter: Fitter):
        self.model: PredictorFitter = fitter.model.model
    
    def get(self) -> pDF:
        """
        Retrieve the model summary table.

        Returns:
            pDF: A styled dataframe of model parameters (coefficients
                or weights) with feature names as the index.
        """        
        return self.model.summary(show=False)

    @staticmethod
    def show(stats: pDF) -> None:
        """
        Render the model summary table inline.

        Parameters:
            stats (pDF): The pre-formatted summary dataframe.
        """        
        Display.construct(stats)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeaturesCorrData:
    """
    Feature correlation matrix computation.

    Computes the Pearson correlation matrix for all model features
    on the test sample. Used to identify multicollinearity concerns
    and provide context for feature importance interpretation.

    Parameters:
        features (list): The ordered list of model feature names.

    Example:
        >>> corr = FeaturesCorrData(model.features)
        >>> matrix = corr.create(test_df)  # Correlation matrix
    """

    def __init__(self, features: list):
        self.features = features

    def create(self, data: pDF) -> pDF:
        """
        Compute the correlation matrix.

        Parameters:
            data (pDF): Dataset containing the feature columns.

        Returns:
            pDF: A symmetric correlation matrix (float32, 6 decimal
                places) with features as both rows and columns.
        """        
        return data[self.features].corr(numeric_only=True).astype(np.float32).round(6)

class FeaturesCorrelation:
    """
    Feature correlation visualization and reporting.

    Renders the correlation matrix as an interactive heatmap and
    identifies high-correlation feature pairs (above a configurable
    threshold) for analyst review.

    Parameters:
        corr_mx (pDF): The pre-computed correlation matrix.
        report_path (Path): Path for saving the heatmap artifact.
        is_load (bool): Indicator whether previously created HTML 
            artifacts can be used.
        is_save_only (bool): Set to True to save static HTML object 
            and exit (is_load is ignored).

    Example:
        >>> corr_report = FeaturesCorrelation(stats.correlation, report_path)
        >>> corr_report.graph()  # Renders heatmap
        >>> corr_report.table()  # Lists high-correlation pairs
    """

    def __init__(self, 
        corr_mx: dict[str, pDF], 
        report_path: Path = None, 
        is_load: bool = False,
        is_save_only : bool = False
    ):
        self.c_mx = corr_mx
        self.path = report_path
        self.is_load = is_load
        self.is_save_only = is_save_only

    @staticmethod
    def _heatmap(correlation_mx: pDF) -> go.Figure:
        """Generate the upper-triangle correlation heatmap figure."""
        l = dict(
            xaxis  = dict(tickfont = dict(size=12)),
            yaxis  = dict(tickfont = dict(size=12))
        )
        upper = Correlation.upper(correlation_mx)
        return Correlation.heatmap(upper = upper, update_layout = l)

    def _save(self, fig: go.Figure | Path) -> None:
        """Save the heatmap and embed a clickable link in the report."""
        link = DashPlotlyLink(
            figure_path  = fig,
            report_path  = self.path,
            link_name    = 'Correlation Heatmap',
            hide_columns = True,
            show         = False
        ).get_output()
        
        if self.is_save_only:
            return
        TableStyle.combine_pdf_tables(link)

    def graph(self) -> None:
        """
        Render the correlation heatmap and save it to the report.

        Displays an upper-triangle heatmap of all feature correlations,
        color-scaled from -1 (strong negative) to +1 (strong positive).
        """
        g = self.path
        if not self.is_load:
            g = self._heatmap(self.c_mx)
        self._save(g)

    def table(self) -> None:
        """
        List high-correlation feature pairs for analyst review.

        Displays a table of feature pairs with correlation above 0.99,
        flagging potential multicollinearity issues.
        """
        if self.is_save_only:
            return
        print()
        Correlation.show(self.c_mx, max_corr = 0.99)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeaturesImportance:
    """
    Feature importance ranking computation.

    Computes feature importance using model-appropriate methods:
    - **GBM models**: Permutation importance + SHAP values.
    - **Linear models**: Ablation (coefficient magnitude) based ranking.

    Results are ordered by training importance to provide a consistent
    reference for cross-sample comparison.

    Parameters:
        fitter (Fitter): The fitted Champion Fitter.
        target (str): Target column name.
        features (list): Model feature names.
        diagnostics (dict): Per-sample diagnostics (for Gini context).
        is_gbm (bool): Whether the model is a GBM.

    Example:
        >>> imp = FeaturesImportance(
        ...     fitter=champion_fitter,
        ...     target='y',
        ...     features=model.features,
        ...     diagnostics=stats.diagnostics,
        ...     is_gbm=True
        ... )
        >>> results = imp.fit({'Valid Sample': val_df, 'Test Sample': oot_df})
    """

    def __init__(self, 
        fitter: Fitter, 
        target: str, 
        features: list, 
        diagnostics: dict[str, DiagnosticsResult], 
        is_gbm: bool
    ):
        self.model: PredictorFitter  = fitter.model.model
        self.target = target
        self.fes    = features
        self.order  = self.model.get_ordered_fes()
        self.diag   = diagnostics
        self.is_gbm = is_gbm
        
        self._fi = FeatureImportance(
            model = self.model.model,  
            dash_title = dict(
                table = None,
                main  = None,
                shap  = None
            ),
            verbose  = False,
            shap_fig = False
        )
    
    @staticmethod
    def _title(tx:str) -> dict:
        return dict(
            table = f'Feature Effect by Method on {tx} | Ordered by Training Importance'
        )
    
    def _fit(self, tx: str, data: pDF, ablation_train: tuple[pDF, pSe]) -> FeatureImportance:
        """Compute importance for a single sample."""
        t = self._title(tx)
        self._fi.update_dash_title(t)

        return self._fi.fit(
            X = data[self.fes], 
            y = data[self.target], 
            ablation_train = ablation_train,
            features_order = self.order, 
        )
    
    def _get_train_sample(self, train_valid:pDF, train_data: pDF) -> tuple[pDF, pSe]:
        # 1. GLM uses the whole train in the final prod. fit
        if train_valid.index.equals(train_data.index):
            train = train_data
        # 2. GBM standard train-valid split
        else:
            idx = train_valid.index
            train = train_data[~train_data.index.isin(idx)]
            
        return (
            train[self.fes],
            train[self.target]
        )
    
    def fit(self, datasets: dict[str,pDF], train_data: pDF):
        """
        Compute feature importance across all samples.

        Parameters:
            datasets (dict) : Evaluation samples - mapping of sample label → dataset.
            train_data (pDF) : Full training data.

        Returns:
            dict: Mapping of sample label → `FeatureImportanceResult`.
        """        
        im = {}
        tr = StatsUtils.smpl_tv(self.is_gbm)
        t  = self._get_train_sample(datasets[tr], train_data)
        for tx, d in datasets.items():             
            im[tx] = self._fit(tx, d, t)
        return im
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeaturesPredictiveness:
    """
    WOE bin-level predictiveness report renderer.

    Displays the target separation (lift) within each WOE bin for
    all features, providing a granular view of how each feature
    contributes to class discrimination.

    This is the "evidence" section for regulators: it shows exactly
    how each variable separates good from bad customers at every
    probability level.

    Parameters:
        binning (dict): Pre-computed WOE overview data containing
            per-feature bin tables and a summary table.
        report_path (Path): Path for saving the detailed grid artifact.
        is_load (bool): Indicator whether previously created HTML 
            artifacts can be used.
        is_save_only (bool): Set to True to save static HTML object 
            and exit (is_load is ignored).

    Example:
        >>> pred = FeaturesPredictiveness(stats.predictive, report_path)
        >>> pred.table()  # Renders summary table with stability flags
        >>> pred.grid()   # Saves full per-feature bin grid
    """

    def __init__(self,         
        binning: dict[str, dict[str, pDF]], 
        report_path: Path = None,
        is_load: bool = False,
        is_save_only : bool = False
    ):
        self.bins = binning
        self.path = report_path
        self.is_load = is_load
        self.is_save_only = is_save_only

    @staticmethod
    def _style(tbl:pDF, hide_axis:int=None) -> pDF:
        """Apply consistent formatting to a WOE bin table."""
        f = '{:.4f}'
        i = '{:.0f}'
        x = [x for x in tbl.columns if x not in [OV.FE, OV.IS]]
        c = [x for x in tbl.columns if OV.C('') in x]
        return Pd.style(
            tbl, 
            hide_axis   = hide_axis,
            format_cols = {c: f for c in x} | {x: i for x in c},
            color       = {'subset': OV.IS, 'color': '#B50064;'}
        )

    @staticmethod
    def _combine(tables: list, titles: list = None, show: bool = True) -> str | None:
        """Combine styled tables into a single HTML block."""
        return TableStyle.combine_pdf_to_html(
            pdf_tables       = tables, 
            titles           = titles,
            tables_per_block = 1, 
            p_tables_per_row = 1,
            h_cols_per_row   = 1,
            h_align          = 'center', 
            show             = show
        )

    def _save(self, html_path: str | Path) -> None:
        """Save the detailed feature grid as an HTML file and embed a link."""
        path = html_path
        if not self.is_load:
            html = TableStyle.format_html([html_path])
            path = Export.html(html, self.path)

        if self.is_save_only:
            return
        
        link = TableStyle.clickable([path], show=False)
        TableStyle.combine_pdf_to_html(
            [link], ['Feature Detail'], tables_per_block=1, h_align='center', show = True
        )

    def table(self, kind: Literal['itables', 'pd'] = 'itables') -> None:
        """
        Render the feature predictiveness summary table.

        Displays a single table listing all features with their overall
        IV score, number of bins, and a stability indicator
        (color-coded: green for stable, red for unstable).

        Parameters:
            kind (str): Rendering engine. 'itables' for interactive
                DataTables with boolean color coding; 'pd' for static
                pandas styling.
        """
        if self.is_save_only:
            return
        
        if kind == 'itables':
            js = ItablesStyle.bool_color(OV.IS, ['#B50064', '#2BCF91'])
            cd = [{"targets": -1, "createdCell": js}]
            itables_show(self._style(self.bins[OV.SUM]), columnDefs=cd)
        else:
            self._combine([self._style(self.bins[OV.SUM])])

    def grid(self) -> None:
        """
        Generate and save the full per-feature WOE bin grid.

        Creates a detailed HTML report containing one table per feature,
        showing the WOE value, target rate, lift, count, and weight
        for each bin. Saves the file and embeds a clickable link.
        """
        if self.is_load:
            self._save(self.path)
            return 
        
        fes = self.bins[OV.DATA]
        g = []
        for d in fes.values():
            d_ = d.copy()
            d_.index.name = 'WOE'
            g.append(self._style(d_, hide_axis = None))

        t = self._combine(g, list(fes), show=False)
        self._save(t)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeaturesStabilityPlot:
    """
    Feature temporal stability computation.

    Evaluates whether each feature's WOE distribution remains
    statistically consistent over time using hypothesis testing
    (chi-squared or t-test) across time units.

    Uses the same stability parameters (alpha, cutoff, time unit)
    that were configured in the WOE Stabilizer step of the pipeline,
    ensuring consistency between training-time and report-time
    assessments.

    Parameters:
        target (str): Target column name.
        woe_features (list): WOE-encoded feature names.
        project_sig (str): The project signature for registry lookups.

    Example:
        >>> stab = FeaturesStabilityPlot(target='y', woe_fes=features, sig=project_sig)
        >>> results = stab.create({'Valid Sample': val_df, 'Test Sample': oot_df})
    """

    def __init__(self, target:str, woe_features:list, project_sig: str):
        self.target = target
        self.fes    = woe_features
        self.sig    = project_sig

        self._p = Dataclass.schema(WOE_FitParams)
        self._c = Dataclass.schema(WOE_FitParamsCfg)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def _get_stability_prm(self) -> WOE_FitParams:
        """
        Retrieve the stability parameters from the fitted WOE Stabilizer.

        Ensures the report uses the exact same thresholds and
        time-unit configuration as the pipeline's WOE stabilization
        step.        
        """
        m: WoeStabilizer = ProjectModelRegistry.get_model(self.sig, AIGFLOW.Encoding.woe_stabilizer)
        p = m.get_all_params()        
        return {
            self._p.stability_cutoff:   p[self._p.stability_cutoff],
            self._p.alpha:              p[self._p.alpha],
            self._c.time_unit:          p[self._c.time_unit],
            self._c.time_var:           p[self._c.time_var],
            self._c.max_time_units:     p[self._c.max_time_units]
        }
    
    def _fe_consistency(self) -> FeConsistency:
        """Instantiate the feature consistency tester with pipeline params."""
        prm = self._get_stability_prm()        
        return FeConsistency( 
            self.fes, self.target, 
            time_var       = prm[self._c.time_var], 
            c_test_th      = prm[self._p.stability_cutoff], 
            t_test_alpha   = prm[self._p.alpha],
            time_unit      = prm[self._c.time_unit], 
            max_time_units = prm[self._c.max_time_units]
        )
    
    def _create_stats(self, data: pDF) -> dict:
        """Execute the consistency test on a single sample."""
        c = self._fe_consistency()
        return c.test(data, verbose=False)

    def _create_fig(self, tx:str, data: dict) -> go.Figure:
        """Generate the stability grid figure for a single sample."""
        return FEConsistencyPlot.dash(
            test_data   = data,
            features    = self.fes,
            verbose     = False,
            grid_title  = f'{tx} → Features Stability'.upper(),
            fig_layout  = FEConsistencyPlot.FigLayout(
                graph_height = 350, 
                graph_width  = 500, 
                horizontal_spacing = 0.15,
                vertical_spacing   = 0.15,                
            )
        )
    
    def create(self, datasets: dict[str,pDF]) -> dict[str, dict]:
        """
        Compute feature stability across all samples.

        Parameters:
            datasets (dict): Mapping of sample label → dataset.

        Returns:
            dict: Mapping of sample label → {'data': test_stats,
                'fig': Plotly figure}.
        """        
        s, f = {}, {}
        for tx, d in datasets.items():
            s = self._create_stats(d)
            f[tx] = self._create_fig(tx, s)
        return f
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class FeaturesStability:
    """
    Feature temporal stability report renderer.

    Displays the cross-sample feature stability comparison:
    - **Grid view**: Interactive Plotly figures showing per-feature
      consistency test results for each sample.
    - **Table view**: A consolidated table showing each feature's
      stability score per sample, the drift delta, and an overall
      stable/unstable flag.

    This section answers: "Are the model's input features behaving
    consistently over time, or has drift occurred that may invalidate
    the model?"

    Parameters:
        stability (dict): Pre-computed stability results from
            `DashStats.fe_stability`.
        report_path (dict[str, Path]): Train-Test Path for saving stability 
            figure artifacts.
        is_load (bool): Indicator whether previously created HTML 
            artifacts can be used.
        is_save_only (bool): Set to True to save static HTML object 
            and exit (is_load is ignored).

    Example:
        >>> fs = FeaturesStability(stats.fe_stability, report_path)
        >>> fs.grid()   # Renders per-sample stability figure grid
        >>> fs.table()  # Renders cross-sample stability comparison table
    """

    def __init__(self, 
        stability: dict[str, dict], 
        report_path: dict[str, Path] = None, 
        is_load: bool = False,
        is_save_only: bool = False
    ):
        self.stab = stability
        self.path = report_path
        self.is_load = is_load
        self.is_save_only = is_save_only
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _path(self, tx: str) -> str|None:
        """Generate the artifact path for a specific sample's stability figure."""
        if self.path is None:
            return
        return self.path[tx]
        
    def _save(self, fig: go.Figure, tx: str) -> pDF:
        """Save a stability figure and return a clickable link."""
        return DashPlotlyLink(
            figure_path     = self._path(tx) if self.is_load else fig, 
            report_path     = self._path(tx),
            make_clickable  = True,
            hide_columns    = True,
            show            = False
        ).get_output()

    def _dash(self) -> list[pDF]:
        """Prepare all stability figures as clickable links."""
        links = []
        for t, f in self.stab.items():
            links.append( self._save(f[DASH_S.FIG], t) )
        return links
    
    def grid(self):
        """
        Render the per-sample stability figure grid.

        Displays each sample's stability figure as a clickable link
        in a two-column layout, allowing reviewers to open the full
        interactive chart for detailed inspection.
        """        
        links = self._dash()
        if self.is_save_only:
            return
        
        return TableStyle.combine_pdf_to_html(
            links, 
            list(self.stab), 
            tables_per_block = 1,
            h_cols_per_row   = 2,             
            h_align          = 'center', 
            show             = True
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _smpl_names(self) -> dict[str, str]:
        """Shorten sample labels for table column headers."""
        return {x: x.split(' ')[0] for x in self.stab}
    
    def _create_data(self) -> dict[str, pDF]:
        """
        Assemble the cross-sample stability comparison table.

        Creates a unified table where each row is a feature and
        columns show: stability score per sample, drift delta
        (difference between samples), and overall stable/unstable flag.        
        """
        tbl = {}
        for tx, d in self.stab.items():
            dt:dict = d[DASH_S.DATA]
            v = []
            for fe, vals in dt.items():
                v.append({
                    STB.F:  fe, 
                    STB.S:  vals[LBL_S.VALUE], 
                    STB.IS: vals[LBL_S.IS]
                })
            tbl[tx] = pDF(v)
        
        smpl = list(self.stab)
        sn = self._smpl_names().items()
        d = lambda i,x: tbl[smpl[i]][x] 
        return pDF(
            {STB.F:  tbl[tx][STB.F]             } | 
            {n:      tbl[s][STB.S] for s,n in sn} |
            {STB.D:  d(1, STB.S)  - d(0, STB.S) } |
            {STB.IS: 
                (d(0, STB.IS)==STAB_S.Y) & 
                (d(1, STB.IS).isin([STAB_S.Y, STAB_S.NA])) }
        )

    def _style(self, tbl:pDF, hide_axis:int=None) -> pDF:
        """Apply formatting and color coding to the stability table."""
        f = '{:.4f}'
        p = '{:.2%}'
        n = list(self._smpl_names().values())
        return Pd.style(
            tbl, 
            hide_axis   = hide_axis,
            format_cols = {s: p for s in self.stab} | {x: f for x in [STB.D] + n},
            color       = {'subset': STB.IS, 'color': ['#B50064;', None]}
        )
        
    def table(self, kind: Literal['itables', 'pd'] = 'itables') -> None:
        """
        Render the cross-sample feature stability comparison table.

        Displays a single consolidated table with:
        - Each feature as a row
        - Stability score per sample (e.g., Valid, Test)
        - Drift delta (Test score - Valid score)
        - Overall stability flag (green if stable in both, red otherwise)

        Parameters:
            kind (str): Rendering engine. 'itables' for interactive
                DataTables with boolean color coding; 'pd' for static
                pandas styling.
        """
        if self.is_save_only:
            return
        
        tbl = self._create_data()
        
        if kind == 'itables':
            js = ItablesStyle.bool_color(STB.IS)
            cd = [{"targets": -1, "createdCell": js}]
            itables_show(self._style(tbl), columnDefs=cd)
            return
        
        TableStyle.combine_pdf_to_html(
            pdf_tables       = [self._style(tbl)], 
            tables_per_block = 1, 
            h_align          = 'center', 
            show             = True
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬