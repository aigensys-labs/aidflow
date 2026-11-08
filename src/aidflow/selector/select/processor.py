"""
Agent Selector — Final Production Deployment Layer.

This module implements the final decision and deployment stage of the
AIDflow pipeline. It sits above all trained and compared agents,
allowing the user to:

1. **Choose** which agent to deploy as the production (champion) model.
2. **Optionally select** a subset of that agent's features.
3. **Inspect** comprehensive diagnostics supporting the decision:
   - WOE feature behavior on test data (per agent).
   - SHAP value drift (train vs. test) per feature.
   - Permutation importance and significance.
   - Gini path (incremental value of each feature in the model).
   - Performance comparison (training version vs. champion version).
4. **Generate** a static HTML overview for documentation/archival.
5. **Produce** the final `AgentPredictor` — a self-contained,
   deployable prediction interface.

Champion vs. Training Agent
---------------------------
- **Training Agent**: The original trained model (as produced by
  the training stage). Its features and parameters are fixed.
- **Champion Agent**: The deployed version, which may use a subset
  of the training agent's features (selected via `agent_choice`
  as a dict). The champion is the model that goes to production.

The selector fits the champion on the training data and evaluates
it on the test data, providing a side-by-side comparison.

Feature Selection within an Agent
---------------------------------
The `agent_choice` parameter supports two modes:
- **int**: Use the agent with all its selected features.
- **dict[int, list]**: Use the agent but only a subset of its
  features (e.g., `{1: ['rev_WOE', 'ten_WOE']}`). This allows
  the user to manually curate the final feature set based on the
  diagnostic insights (SHAP drift, significance, etc.).
"""

from ...deps.data                   import  pDF, pSe, np
from ...deps.plots                  import  go
from ...deps.types                  import  NDArray

from ...core.base                   import  FitterModelBase
from ...core.factory                import  (
                                            Dataclass, 
                                            DictClass
                                            )
from ...core.foundation             import  AigParam
from ...core.base.worker            import  WorkerBase
from ...core.base.adapter           import  ModelAdapterBase
from ...core.io                     import  (
                                            TempRegistry,
                                            Temp
                                            )
from ...core.interop                import  Pd
from ...core.standard               import  VarStd
from ...core.accessor               import  (
                                            Format, 
                                            Spinner
                                            )

from ...ml.features                 import  WoEPlot
from ...ml.tests                    import  TestSampleDiff
from ...ml.metrics                  import  (
                                            LBL as LBL_D,
                                            DiagnosticsResult,
                                            SeparationDiagnostics,
                                            GiniLiftPlot
                                            )
from ...learner.features.pruning    import  (
                                            GINI as GINI_G,
                                            GiniPath,
                                            )
from ...learner.features.importance import  (
                                            LBL as LBL_I,
                                            MTD as MTD_I
                                            )

from ...storage.export              import  Export

from ...display.style               import  TableStyle

from ..base.processor               import  (
                                            AGENTS,
                                            FitterSelectionBase
                                            )
from ..predict.adapter              import  PredictorAdapter                                            
from ..compare.processor            import  (
                                            STAT as STAT_C,
                                            RANK as RANK_C,
                                            ComparatorStats
                                            )

from .agent                         import  (
                                            AgentPredictorInput,
                                            AgentPredictorResultDual,
                                            AgentPredictor
                                            )


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENT CHOICE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ID(DictClass):
    """Static view file identifiers for HTML overview."""
    P = lambda x: f'Static_View_of_Agent_{x}_Performance'

class LBL(DictClass):
    """Labels for performance metrics, drift indicators, and display."""
    N  = lambda x: f'Agent {x}'
    TD = 'TrainΔ'
    VD = 'ValidΔ'
    V  = 'Valid'
    T  = 'Test'
    D  = 'DriftΔ'
    S  = 'SignificantΔ'
    P  = 'PermutationΔ'
    GT = 'Gini'
    IV = 'IV'
    KS = 'KS'

class STAT(DictClass):
    """Store keys for per-agent diagnostic data."""
    FS = 'fe_selected'
    FO = 'fe_ordered'
    FD = 'fe_description'
    GP = 'gini_path'
    SH = 'shap'
    PM = 'permutation'
    PF = 'performance'
    SO = 'score_order'

class DIAG(DictClass):
    """Diagnostic comparison labels."""
    T = 'Training Agent'
    C = 'Champion Agent'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@Dataclass.base
class SelectorParamsBase:
    """
    Selector configuration — analysis and output settings.

    Attributes
    ----------
    alpha : float, default=0.05
        Significance level for SHAP drift testing
        (Welch's t-test on SHAP means).
    proba_bins : int, default=10
        Number of bins for the lift/diagnostic tables.
    agent_choice : int | dict[int, list], default=1
        Agent selection:
        - **int**: Use that agent with all its features.
        - **dict**: `{agent_key: [feature1, feature2, ...]}` —
          use that agent with only the listed features.
    create_html_overview : bool, default=False
        If True, generate and save a static HTML overview of
        each agent's performance (for documentation/archival).
    """    
    alpha: float = 0.05
    proba_bins: int = 10
    agent_choice: int | dict[int, list] = 1
    create_html_overview: bool = False

@Dataclass.base(update_specs=dict(slots=False))
class SelectorParamsCfg:
    """
    Selector configuration — pipeline component references.

    Attributes
    ----------
    predictors : PredictorAdapter
        Adapter holding all trained agents.
    fe_cleaner : ModelAdapterBase
        The fitted feature cleaner (for preprocessing).
    woe_encoder : ModelAdapterBase
        The fitted WOE encoder (for preprocessing + plots).
    comparator : ModelAdapterBase
        The fitted comparator (for performance reference).
    target : str
        Target column name.
    verbose : bool | int
        Verbosity to provide waiting counter.
    predictor : ModelAdapterBase | AgentPredictor, default=None
        A pre-fitted agent predictor (if the user has already
        selected and fitted the champion). If None, the selector
        will fit one during `process`.
    preprocess : bool | None, default=None
        If True, apply the full encoding pipeline to input data.
    """    
    predictors:  PredictorAdapter
    fe_cleaner:  ModelAdapterBase    
    woe_encoder: ModelAdapterBase    
    comparator:  ModelAdapterBase    
    target:      str
    verbose:     bool | int = True
    predictor:   ModelAdapterBase | AgentPredictor = None
    preprocess:  bool|None = None

@Dataclass.base
class SelectorParams(SelectorParamsBase, SelectorParamsCfg):
    """Woker params input design."""

@Dataclass.base
class SelectorInput:
    """
    Selector input specification.

    Attributes
    ----------
    test_set : pDF
        OOT/test data for evaluation and diagnostics.
        Must contain the target column.
    train_set : pDF | None, default=None
        Training data (for SHAP drift comparison and champion
        fitting). If None, only test diagnostics are produced.
    agents : tuple | list | None
        Agents to include in the analysis. Default: all.
    """
    test_set:   pDF 
    train_set:  pDF = None
    agents:     tuple|list|None = tuple(AGENTS)

@Dataclass.base
class SelectorResult:
    """
    Selector output — the production deployment package.

    Attributes
    ----------
    agent_predictor : AgentPredictor
        The fitted champion agent predictor — a self-contained,
        deployable prediction interface. Use `.transform(data)`
        for production scoring.
    performance : dict
        Per-agent performance comparison (training vs. champion).
    shap : dict
        Per-agent SHAP drift analysis (train vs. test per feature).
    """    
    agent_predictor: AgentPredictor
    performance:     dict
    shap:            dict

@Dataclass.base
class SelectorStats:
    """
    Full selector diagnostic output.

    Attributes
    ----------
    summary : dict[int, dict]
        Per-agent diagnostic bundle: selected features, ordered
        features, per-feature descriptions, Gini path, SHAP drift,
        permutation importance, and performance comparison.
    diagnostics : dict[str, DiagnosticsResult]
        Side-by-side diagnostic results for the training agent
        (`DIAG.T`) and the champion agent (`DIAG.C`).
    """    
    summary:     dict[int, dict]
    diagnostics: dict[str, DiagnosticsResult]
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PredictorBase:
    """
    Base class for selector-level prediction utilities.

    Provides shared access to the predictor adapter, agent
    resolution, and feature extraction with safety checks.

    Parameters
    ----------
    predictors : PredictorAdapter
        Adapter holding trained agents.
    preprocess : bool, default=False
        Whether to apply preprocessing when predicting.
    """
    _store: dict[int, dict] = {}

    def __init__(self, predictors: PredictorAdapter, preprocess:bool=False):
        self.predictors  = predictors
        self.preprocess  = preprocess

    def _norm_idx(self, a:str|int) -> int:
        """Normalize an agent identifier to its integer key."""
        if isinstance(a, int):
            return a
        return {a:i for i,a in AGENTS.items()}[a]
        
    def _get_fes(self, a:int) -> tuple[list, list]:
        """Returns tuple of Agent's [0] Selected Features and [1] Selected Ordered Features."""
        model = self.predictors.get_agent(a)
        fes   = model.state.features.selected
        order = [x for x in model.fitter.get_ordered_fes() if x in fes]
        
        if len(fes) != len(order):
            raise ValueError(
            f'AIG SAFETY ► Unexpected mismatch between selected vs. ordered Features in Agent {a}'
            )
        
        return fes, order

    def _get_stored(self, method:str, a:int=None) -> pDF | tuple | None:
        """Retrieve a cached diagnostic result."""
        if a is None:
            return self._store[method]
        return self._store[method].get(a)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class WOEEncoder(PredictorBase):
    """
    WOE Data and Plot Provider for Selector Diagnostics.

    Creates WOE-encoded data for the test set (enabling feature
    behavior visualization) and generates interactive WOE dash
    plots per agent — showing the bin distribution and target
    rate on test data for each significant feature.

    Parameters
    ----------
    predictors : PredictorAdapter
        Adapter holding trained agents.
    woe_encoder : ModelAdapterBase
        The fitted WOE encoder (from the encoding stage).
    target : str
        Target column name.
    preprocess : bool, default=True
        Whether to apply the full encoding pipeline to input data.
    """    
    def __init__(self, 
        predictors:  PredictorAdapter, 
        woe_encoder: ModelAdapterBase, 
        target:      str, 
        preprocess:  bool = True
    ):
        super().__init__(predictors, preprocess)
        self.woe_encoder = woe_encoder
        self.target      = target
        self._set_fes()

    def _set_fes(self) -> None:
        """
        Resolve the feature lists for WOE encoding and plotting.

        For GLM agents: uses the original (pre-WOE) feature names.
        For GBM agents: uses the WOE-encoded feature names directly.

        Sets:
        - `self._fes_order`: Per-agent feature list in importance
          order (for plot column ordering).
        - `self._fes_org`: Union of all original feature names
          (for the WOE encoder input).
        """        
        self._fes_order = {}
        org = []

        for a in self.predictors.predictors:
            fes, order = self._get_fes(a)
            a_ = self._norm_idx(a)
            self._fes_order[a_] = order

            if self.predictors.is_gbm(a):
                org.extend(fes)
            else:
                org.extend(VarStd.fe_to_var(fes))

        self._fes_org = list(set(org))                
    # ─────────────────────────────────────────────────────────────────────────────────────────────
       
    def create_ml_data(self, data:pDF) -> pDF:
        """
        Create WOE-encoded data for the test set.

        Applies the trained WOE encoder to the specified features
        and appends the results to the input data.

        Parameters
        ----------
        data : pDF
            Raw or pre-processed test data.

        Returns
        -------
        pDF
            Input data + WOE-encoded feature columns.
        """
        w = self.woe_encoder.predict(
            data, subset = self._fes_org, preprocess = self.preprocess
        )
        d = data.copy()
        d[w.columns] = w
        return d

    def create_plots(self, 
        ml_data:   pDF, 
        agents:    tuple[int, ...] = tuple(AGENTS),
        add_title: bool = False
    ) -> dict[int, go.Figure]:
        """
        Create interactive WOE dash plots for the specified agents.

        Each plot shows the bin distribution and target rate for
        each significant feature, ordered by training importance.
        Provides a visual check of feature behavior on test data.

        Parameters
        ----------
        ml_data : pDF
            WOE-encoded test data (from `create_ml_data`).
        agents : tuple[int, ...]
            Agent keys to plot.
        add_title : bool, default=False
            Whether to include a descriptive title in the plot.

        Returns
        -------
        dict[int, go.Figure]
            Per-agent WOE dash figures.
        """
        title = lambda a: (f'{LBL.N(a)}'
            '→ Test Data WOE View to Significant Features Behavior '
            '→ Ordered by Training Feature Importance'
        )
        update_layout = {'margin': dict(t=40, r=0, l=0, b=0)} if not add_title else {}

        d = {}
        for a in agents:
            d[a] = WoEPlot(
                encoder = self.woe_encoder,
                apply_data_quality = False,
                apply_woe_encoding = False
            ).dash(
                data = ml_data,
                target = self.target,
                features = self._fes_order[a],
                grid_cols = 3,
                grid_title = title(a) if add_title else None, 
                update_layout = update_layout,
                verbose = False
            )['fig']
        return d
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PredictorStats(PredictorBase):
    """
    Per-Agent Diagnostic Stats Provider for Selector.

    Computes and caches the full diagnostic bundle for each agent:
    - **Performance**: Gini/IV/KS comparison (training vs. test).
    - **Gini Path**: Incremental Gini contribution of each feature
      (in importance order) — shows which features add value.
    - **SHAP Drift**: Train vs. test SHAP mean comparison per
      feature, with significance testing (Welch's t-test).
      Highlights features whose behavior changed between train
      and test (potential instability).
    - **Permutation Importance**: Per-feature Gini drop when
      permuted, with significance flags.

    All results are cached in `self._store` for efficient
    repeated access.

    Parameters
    ----------
    predictors : PredictorAdapter
        Adapter holding trained agents.
    comparator : ModelAdapterBase
        The fitted comparator (for performance reference).
    alpha : float, default=0.05
        Significance level for SHAP drift testing.
    """

    def __init__(self, 
        predictors: PredictorAdapter, 
        comparator: ModelAdapterBase, 
        alpha: float = 0.05
    ):
        super().__init__(predictors)
        self.compare = comparator
        self.alpha   = alpha

        self._store  = {
            STAT.GP: {},
            STAT.PM: {},
            STAT.SH: {},
            STAT.PF: {},
            STAT.SO: {},
        }   
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def get_stats(self, 
        test_ml:  pDF, 
        train_ml: pDF = None, 
        agents:   tuple[int, ...] = tuple(AGENTS)
    ) -> dict[int, dict]:
        """
        Compute the full diagnostic bundle for all specified agents.

        Parameters
        ----------
        test_ml : pDF
            WOE-encoded test data.
        train_ml : pDF | None
            WOE-encoded training data (for SHAP drift).
        agents : tuple[int, ...]
            Agent keys to analyze.

        Returns
        -------
        dict[int, dict]
            Per-agent diagnostic bundle:
            - `STAT.FS`: Selected features.
            - `STAT.FO`: Features in importance order.
            - `STAT.FD`: Per-feature descriptions.
            - `STAT.GP`: Gini path table.
            - `STAT.SH`: SHAP drift table.
            - `STAT.PM`: Permutation importance table.
            - `STAT.PF`: Performance comparison table.
        """
        stats  = {}
        for a in AGENTS:
            if a not in agents:
                continue

            fes, fe_order = self._get_fes(a)
            g_p  = self._get_gini_path(a, fe_order)
            shap = self._compare_shap(a, train_ml, test_ml, fes, fe_order)
            perm = self._get_permutation(a, fe_order)
            perf = self._get_performance(a)
            
            fe_def = self._get_fes_def(fe_order, g_p, shap, perm)

            stats[a] = {
                STAT.FS: fes,
                STAT.FO: fe_order,
                STAT.FD: fe_def,
                STAT.GP: g_p,
                STAT.SH: shap,
                STAT.PM: perm,
                STAT.PF: perf
            }

        return stats
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    def _get_performance(self, a:int) -> dict:
        """
        Get the performance comparison (training vs. test) for an agent.

        Compares the agent's validation metrics (from training)
        against its test metrics, computing the drift (Test - Valid).

        Returns
        -------
        pDF
            Table with rows: Gini, IV, KS. Columns: Valid, Test, DriftΔ.
        """        
        s = self._get_stored(STAT.PF, a)
        if s is not None:
            return s
        
        m = self.predictors.get_agent(a)
        d:DiagnosticsResult = m.stats.diagnostics
                
        test:ComparatorStats = self.compare.stats
        p = test.summary[RANK_C.PE].loc[a]
        p = pDF({
            LBL.V: [d.gini, d.IV, d.KS],
            LBL.T: [p[STAT_C.GT], p[STAT_C.IV], p[STAT_C.KS]]
        })
        p.index  = [LBL.GT, LBL.IV, LBL.KS]
        p[LBL.D] = p[LBL.T] - p[LBL.V]
        
        self._store[STAT.PF][a] = p
        return self._get_stored(STAT.PF, a)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_gini_path(self, a:int, fe_order:list) -> pDF:
        """
        Get the Gini path (incremental Gini per feature) for an agent.

        The Gini path shows how much Gini each feature adds to the
        model, in importance order. A feature with a small
        incremental Gini contributes little beyond the preceding
        features.

        Returns
        -------
        pDF
            One row per feature (in importance order) with
            TrainΔ and ValidΔ incremental Gini values.
        """        
        s = self._get_stored(STAT.GP, a)
        if s is not None:
            return s
        
        model = self.predictors.get_agent(a)
        gp: GiniPath = model.stats.gini_path
        
        tv  = gp.train_valid.loc[fe_order].copy()
        tv  = tv.rename(columns={GINI_G.T: LBL.TD, GINI_G.V: LBL.VD})
        tv1 = tv.diff(1)
        tv1.iloc[0] = tv.iloc[0]
        self._store[STAT.GP][a] = tv1

        return self._get_stored(STAT.GP, a)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_permutation(self, a:int, fe_order:list) -> pDF:
        """
        Get the permutation importance for an agent.

        Per-feature Gini drop when that feature's values are
        randomly permuted. With significance flags indicating
        whether the drop is statistically meaningful.

        Returns
        -------
        pDF
            One row per feature (in importance order) with
            PermutationΔ value and significance flag.
        """        
        s = self._get_stored(STAT.PM, a)
        if s is not None:
            return s
        
        model = self.predictors.get_agent(a)
        fi = model.report.feature_importance.method_results
        p:pDF = fi[MTD_I.P].set_index(LBL_I.F)
        p = p.loc[fe_order, [LBL_I.P, LBL_I.IS]].rename(columns={LBL_I.P: LBL.P})

        self._store[STAT.PM][a] = p
        return self._get_stored(STAT.PM, a)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _shap_stats(shap_array:NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """
        Compute per-feature SHAP summary statistics.

        Parameters
        ----------
        shap_array : NDArray
            SHAP values of shape (n_samples, n_features).

        Returns
        -------
        tuple
            (mean_abs_shap, std_shap, n_samples)
        """        
        m = np.abs(shap_array).mean(axis=0)
        s = shap_array.std(ddof=1, axis=0)
        n = np.array([shap_array.shape[0]]*shap_array.shape[1])
        return m, s, n

    def _store_train_shap(self, a:int, train_ml:pDF|None, fes_orig:list):
        """
        Compute and cache the training set SHAP statistics.

        If no training data is provided, stores NaN placeholders.

        Parameters
        ----------
        a : int
            Agent key.
        train_ml : pDF | None
            Training data (WOE-encoded).
        fes_orig : list
            Original feature names for SHAP computation.
        """        
        if self._get_stored(STAT.SH, a) is not None:
            return
        if train_ml is None:
            nan = np.array([np.nan]*len(fes_orig))
            self._store[STAT.SH][a] = (nan, nan, nan)
            return
        model = self.predictors.get_agent(a)
        v = train_ml.loc[model.fitter.dash.val_idx, fes_orig]
        v_shap = model.fitter.shap_values(v)
        self._store[STAT.SH][a] = self._shap_stats(v_shap)

    def _compare_shap(self, 
        a:          int, 
        train_ml:   pDF, 
        test_ml:    pDF, 
        fes_orig:   list, 
        fes_order:  list
    ) -> pDF:
        """
        Compute SHAP drift analysis: train vs. test per feature.

        Compares the mean absolute SHAP values between training
        and test data for each feature. If the difference is
        statistically significant (Welch's t-test), it indicates
        the feature's contribution has changed — a potential
        instability flag for production.

        Parameters
        ----------
        a : int
            Agent key.
        train_ml : pDF
            Training data.
        test_ml : pDF
            Test data.
        fes_orig : list
            Original feature names.
        fes_order : list
            Features in importance order.

        Returns
        -------
        pDF
            Per-feature SHAP comparison: Valid, Test, DriftΔ,
            and SignificantΔ flag.
        """        
        self._store_train_shap(a, train_ml, fes_orig)        
        model = self.predictors.get_agent(a)

        t_shap = model.fitter.shap_values(test_ml[fes_orig])
        tm, ts, tn = self._shap_stats(t_shap)
        vm, vs, vn = self._get_stored(STAT.SH, a)

        s = pDF(index = fes_orig)
        s[LBL.V] = vm
        s[LBL.T] = tm
        s[LBL.D] = s[LBL.T] - s[LBL.V]
        s[LBL.S] = TestSampleDiff(self.alpha).welch_mx(
            m1 = tm, n1 = tn, s1 = ts, 
            m2 = vm, n2 = vn, s2 = vs
        )
        s = s.loc[fes_order]

        if np.isnan(vm).sum() == len(fes_orig):
            return s[[LBL.T]]

        return s
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _get_fes_def(fe_order:list, gini_path:pDF, shap:pDF, permutation:pDF) -> dict[str, dict]:
        """
        Assemble per-feature description strings for display.

        For each feature (in importance order), provides:
        - Gini path incremental value.
        - Permutation importance value.
        - SHAP valid/test values.

        Parameters
        ----------
        fe_order : list
            Features in importance order.
        gini_path : pDF
            Gini path table.
        shap : pDF
            SHAP drift table.
        permutation : pDF
            Permutation importance table.

        Returns
        -------
        dict[str, dict]
            {feature_name: {metric: formatted_value}}
        """        
        defs = {}
        gp = gini_path.map(lambda x: f"\u00A0{x:.3f}" if x >= 0 else f"{x:.3f}")
        pm = permutation.map(lambda x: f"\u00A0{x:.3f}" if x >= 0 else f"{x:.3f}")
        sh = shap.map("{:.3f}".format)
        for fe in fe_order:
            defs[fe] = {
                STAT.GP: gp.loc[fe, LBL.VD],
                STAT.PM: pm.loc[fe, LBL.P],
                STAT.SH: sh.loc[fe, [LBL.V, LBL.T]].to_dict(),
            }
        return defs
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class AgentChoicePredictor(PredictorBase):
    """
    Champion Agent Fitting and Diagnostics.

    Resolves the user's agent choice (int or dict), fits the champion
    agent predictor on training data, and produces side-by-side
    diagnostics comparing:

    - **Training Agent** (DIAG.T): The original agent as trained
    (all selected features, trained version).
    - **Champion Agent** (DIAG.C): The deployed version (possibly a
    feature subset, refit on training data).

    This comparison shows the impact of the user's selection decision
    (feature subset) on model performance — enabling informed deployment.

    Parameters
    ----------
    predictors : PredictorAdapter
        Adapter holding trained agents.
    agent_choice : int | dict[int, list]
        - **int**: Use that agent with all its features.
        - **dict**: `{agent_key: [feat1, feat2, ...]}` — use that
        agent with only the listed features.
    target : str
        Target column name.
    predictor : ModelAdapterBase | AgentPredictor, default=None
        A pre-fitted agent predictor (if already created). If None,
        one is created and fitted during `process`.
    bins : int, default=10
        Number of bins for lift/diagnostic tables.
    """
    _test_id: int = None
    
    def __init__(self, 
        predictors:   PredictorAdapter, 
        agent_choice: int | dict[int, list],
        target:       str,
        predictor:    ModelAdapterBase | AgentPredictor = None,
        bins:         int = 10
    ):
        super().__init__(predictors)        
        self.agent_choice = agent_choice
        self.target       = target
        self.bins         = bins
        
        self.agent_predictor = predictor or AgentPredictor(self._get_setup())
        self._agent_worker   = predictor
        
        self._store = {
            STAT.PF: {}
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def _is_predictor_model(self) -> bool:
        """AgentPredictor (model) or AgentPredictorWorker (worker)."""
        return isinstance(self.agent_predictor, AgentPredictor)
    
    def get_agent_predictor(self) -> AgentPredictor:
        """Returns agent predictor."""
        if self._is_predictor_model():
            return self.agent_predictor
        w: WorkerBase = self.agent_predictor.model
        return w.model.model

    def _get_agent_predictor_inputs(self, train_ml:pDF, test_ml:pDF) -> AgentPredictorInput:
        return AgentPredictorInput(
            train_set    = train_ml,
            test_set     = test_ml,
            features     = self._get_selected_fes(),
            agent_choice = self._get_selected_agent(),
            target       = self.target
        )       

    def _fit_agent_predictor_model(self, train_ml:pDF, test_ml:pDF) -> None:
        """
        Fit the champion agent predictor on training data.

        Resolves the selected features and agent, then fits the
        `AgentPredictor` model. Handles both the worker API (process)
        and the model API (fit) depending on the predictor state.

        Parameters
        ----------
        train_ml : pDF
            Training data (WOE-encoded).
        test_ml : pDF
            Test data (WOE-encoded) — used for dual evaluation.
        """
        inputs = self._get_agent_predictor_inputs(train_ml, test_ml)
        predictor = self.agent_predictor # worker process api
        if self._is_predictor_model():   # model fit api         
            predictor = predictor.fit    
        predictor(inputs)
    
    def _transform(self, test_ml:pDF, is_inference:bool) -> NDArray:
        """Returns agent test data predicton."""
        m = self.agent_predictor
        f = self._get_selected_fes()

        if is_inference or self._is_predictor_model():
            return m.transform(test_ml[f]).data.squeeze()  

        res: AgentPredictorResultDual = m.fitted
        return res.test_data.squeeze()
    # ─────────────────────────────────────────────────────────────────────────────────────────────   
    
    def _get_setup(self):
        m = self.predictors.get_agent(next(iter(AGENTS)))
        return m.model.setup
    
    def _get_selected_agent(self) -> int:
        if isinstance(self.agent_choice, int):
            return self.agent_choice
        return next(iter(self.agent_choice))
    
    def _get_agent_orig_model(self):
        """Agent Model from Training Phase."""
        a = self._get_selected_agent()
        return self.predictors.get_agent(a)

    def _get_orig_fes(self) -> list:
        """Agent's Selected Features in orig. data order."""
        a = self._get_selected_agent()
        return self._get_fes(a)[0]
        
    def _get_selected_fes(self) -> list:
        """Agent's User Selected Features in orig. data order."""
        fes = self._get_orig_fes() 
        if isinstance(self.agent_choice, int):
            return fes
        a = self._get_selected_agent()
        return [x for x in fes if x in self.agent_choice[a]] # to maintain orig. order
    
    def _diagnostics(self, y: pSe, p: pSe) -> DiagnosticsResult:
        return SeparationDiagnostics(lift_bins=self.bins).evaluate(y, p)
        
    def _predict_diag(self, test_ml:pDF, train_ml:pDF=None, is_champion_agent:bool = False
    ) -> DiagnosticsResult:
        if not is_champion_agent:
            m = self._get_agent_orig_model()
            f = self._get_orig_fes()
            p = m.transform(test_ml[f]).data.squeeze()
        else:
            p = self._transform(test_ml, train_ml is None)
        return self._diagnostics(test_ml[self.target], p)
    # ─────────────────────────────────────────────────────────────────────────────────────────────         
        
    def create_diagnostics(self, test_ml:pDF, train_ml:pDF=None) -> dict:
        """
        Create side-by-side diagnostics for the Training vs Champion agent.

        Computes:
        - **DIAG.T** (Training Agent): The original agent's test
        performance (all selected features).
        - **DIAG.C** (Champion Agent): The user-selected version's
        test performance (possibly a feature subset).

        Results are cached (keyed by test data identity) to avoid
        recomputation on repeated calls.

        Parameters
        ----------
        test_ml : pDF
            Test data (WOE-encoded).
        train_ml : pDF | None
            Training data (for inference mode detection).

        Returns
        -------
        dict
            {DIAG.T: DiagnosticsResult, DIAG.C: DiagnosticsResult}
        """
        a = self._get_selected_agent()
        
        if (
            self._get_stored(STAT.PF, a) is None or 
            id(test_ml) != self._test_id
        ):
            self._test_id = id(test_ml)
            self._store[STAT.PF][a] = self._predict_diag(test_ml)
        
        o = self._get_stored(STAT.PF, a)
        r = self._predict_diag(test_ml, train_ml, True)

        return {
            DIAG.T: o, 
            DIAG.C: r
        }
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
   
class Fitter(FitterSelectionBase[SelectorInput, SelectorParams]):
    """
    Selector Fitter — orchestrates the full selection and deployment pipeline.

    Coordinates the three selector components:

    1. **WOEEncoder**: Creates WOE-encoded test data and plots.
    2. **PredictorStats**: Computes per-agent diagnostic bundles.
    3. **AgentChoicePredictor**: Fits the champion and creates
    training-vs-champion diagnostics.

    The pipeline:
    test data → WOE encode → per-agent diagnostics → champion fit
    → training vs champion diagnostics → (optional) HTML overview
    → return the fitted AgentPredictor.

    Parameters
    ----------
    params : SelectorParams
        Full selector configuration (analysis settings + pipeline refs).
    """
    
    SelectionInput:  type = SelectorInput
    agents_condition: int = 1

    def __init__(self, params: SelectorParams):
        super().__init__(params)
        
        self.woe_encoder = WOEEncoder(
            predictors  = self.prm.predictors,
            woe_encoder = self.prm.woe_encoder,
            target      = self.prm.target,
            preprocess  = self.prm.preprocess
        )
        self.predictor_stats = PredictorStats(
            predictors  = self.prm.predictors,
            comparator  = self.prm.comparator,
            alpha       = self.prm.alpha
        )        
        self.agent_predictor = AgentChoicePredictor(
            predictors  = self.prm.predictors,
            agent_choice= self.prm.agent_choice,
            predictor   = self.prm.predictor,
            target      = self.prm.target,
            bins        = self.prm.proba_bins
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    def _save_overview(self, overview: dict[int, str]):
        """Saves Static Prformance Overview to AIG temp folder."""
        for a, html in overview.items():
            m = self.prm.predictors.get_agent(a)
            s = m.get_project_signature()
            i = ID.P(a)
            p = Temp.gen_path(i, file_type='html', project_signature=s)
            TempRegistry.register(s, i, p)
            Export.html(html, p)
    # ─────────────────────────────────────────────────────────────────────────────────────────────    

    def process(self, inputs: SelectorInput | pDF, **kwargs) -> AgentPredictor:
        """
        Execute the full selector pipeline.

        Steps:
        1. Resolve input (test_set, train_set, agents).
        2. Create WOE-encoded test data.
        3. Compute per-agent diagnostic bundles.
        4. Fit the champion agent predictor (if training data provided).
        5. Create training-vs-champion diagnostics.
        6. Optionally generate and save HTML overviews.
        7. Return the fitted `AgentPredictor`.

        Parameters
        ----------
        inputs : SelectorInput | pDF
            Test data (required) + training data (optional) + agent filter.
        kwargs : dict
            Forwarded to input resolution.

        Returns
        -------
        AgentPredictor
            The fitted, deployable champion agent predictor.
            Use `.transform(data)` for production scoring.
        """

        inputs_  = self.get_inputs(inputs, kwargs)
        data_ml  = self.woe_encoder.create_ml_data(inputs_.test_set)
        train_ml = inputs_.train_set

        summary = self.predictor_stats.get_stats(data_ml, train_ml, self.agents)

        if train_ml is not None:
            self.agent_predictor._fit_agent_predictor_model(train_ml, data_ml)

        diag = self.agent_predictor.create_diagnostics(data_ml, train_ml)

        # Static Temp Overview
        if self.prm.create_html_overview:
            woe_fig  = self.woe_encoder.create_plots(data_ml, self.agents)
            overview = SelectorOverview(woe_fig, summary).create()
            self._save_overview(overview)

        self.stats = SelectorStats(
            summary = summary,
            diagnostics = diag
        )

        return self.agent_predictor.get_agent_predictor()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SelectorProcessor(FitterModelBase[Fitter]):
    """
    Agent Selector — Final Production Deployment Layer.

    Public API for the final stage of the AIDflow pipeline. Selects
    the champion agent from trained candidates, optionally curates
    its feature subset, fits the production predictor, and produces
    the full diagnostic package supporting the deployment decision.

    Decision Support
    ----------------
    The selector provides comprehensive diagnostics to inform the
    deployment decision:
    - **Performance comparison**: Training vs Champion (impact of
    feature selection).
    - **SHAP drift**: Which features changed behavior between train
    and test (instability flag).
    - **Gini path**: Incremental value of each feature (redundancy
    detection).
    - **Permutation importance**: Feature robustness.
    - **WOE plots**: Visual verification of feature behavior on test.
    - **HTML overview**: Archival documentation of the decision.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for SHAP drift testing.
    proba_bins : int, default=10
        Number of bins for lift/diagnostic tables.
    agent_choice : int | dict[int, list], default=1
        - **int**: Use that agent with all its features.
        - **dict**: `{agent_key: [feat1, feat2, ...]}` — feature
        subset selection (e.g., `{1: ['rev_WOE', 'ten_WOE']}`).
    create_html_overview : bool, default=False
        Generate and save static HTML overviews for archival.
    preprocess : bool | None, default=None
        If True, apply the full encoding pipeline to input data.

    Notes
    -----
    **Champion vs Training Agent**: The "Training Agent" is the
    original as trained. The "Champion Agent" is the deployed
    version (possibly with a feature subset). The selector fits the
    champion on training data and evaluates on test, providing a
    fair side-by-side comparison.

    **Feature Subset Selection**: Use `agent_choice` as a dict to
    manually curate the final feature set based on the diagnostic
    insights (drop features with high SHAP drift or low Gini
    contribution).

    **Self-Contained Output**: The returned `AgentPredictor` is
    a standalone object — it holds all encoding maps, model
    parameters, and feature names needed for production inference
    without referencing the pipeline.

    Examples
    --------
    >>> from aidflow.fitters import SelectorProcessor
    >>> selector = SelectorProcessor(
    ...     predictors=predictors,
    ...     fe_cleaner=fe_cleaner,
    ...     woe_encoder=woe_encoder,
    ...     comparator=comparator,
    ...     target='is_churn',
    ...     agent_choice=1,
    ... )
    >>> result = selector.process(test_data, train_set=train_data)
    >>> result.agent_predictor  # Deployable predictor
    >>> result.performance      # Per-agent perf comparison
    >>> result.shap             # Per-agent SHAP drift

    >>> # With feature subset:
    >>> selector = SelectorProcessor(
    ...     ...,
    ...     agent_choice={1: ['revenue_WOE', 'tenure_WOE', 'segment_WOE']},
    ... )
    >>> result = selector.process(test_data, train_set=train_data)
    """

    def __init__(self, *args, params: dict|SelectorParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(SelectorParams, params, args, kwargs)
        self._fitter = Fitter(self._params)
        
    def _get_stat(self, req:str) -> dict:
        stat = {}
        stats: SelectorStats = self.stats
        for a, s in stats.summary.items():
            stat[a] = s[req]
        return stat
    
    def process(self, inputs: SelectorInput | pDF, **kwargs) -> SelectorResult:
        """
        Execute the selector: fit the champion and return the deployment package.

        Parameters
        ----------
        inputs : SelectorInput | pDF
            - **pDF**: Test data only (diagnostics without champion fit).
            - **SelectorInput**: Test data + optional training data + agent filter.
        kwargs : dict
            Forwarded to input resolution.

        Returns
        -------
        SelectorResult
            - `agent_predictor`: The fitted, deployable champion predictor.
            - `performance`: Per-agent performance comparison tables.
            - `shap`: Per-agent SHAP drift tables.
        """
        t = 'Agent Selector compute in progress'
        with Spinner(t, show_timer=True).progress(self._params.verbose):
            agent = self._fitter.process(inputs, **kwargs)
        return SelectorResult(
            agent_predictor = agent,
            performance     = self._get_stat(STAT.PF),
            shap            = self._get_stat(STAT.SH),
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SelectorOverview:
    """
    Static HTML Overview of Agent Performance.

    Combines the diagnostic tables (SHAP drift, Gini path, performance)
    and the WOE dash plot into a single self-contained HTML page per
    agent. Designed for documentation, archival, and stakeholder
    communication.

    Each agent's overview contains:
    1. **SHAP Value Drift** table: Train vs Test mean SHAP per feature,
    with significance flags.
    2. **Gini Added Value** table: Incremental Gini per feature (in
    importance order) + permutation importance.
    3. **Model Performance** table: Gini/IV/KS comparison.
    4. **WOE Plot**: Interactive dash showing bin distribution and
    target rate on test data.

    Parameters
    ----------
    woe_figs : dict[int, go.Figure]
        Per-agent WOE dash figures.
    stats : dict[int, dict[str, pDF|dict]]
        Per-agent diagnostic bundles (from `PredictorStats`).
    """
    def __init__(self, 
        woe_figs: dict[int, go.Figure], 
        stats: dict[int, dict[str, pDF|dict]]
    ):
        self.figs  = woe_figs
        self.stats = stats
    
    def _stats_to_html(self, a:int) -> str:
        stat = self.stats[a]
        
        g = stat[STAT.GP].drop(LBL.TD, axis=1).copy()
        p = stat[STAT.PM]
        g[p.columns] = p

        g = Pd.style(
            g, hide_axis = 0, 
            format = dict(precision=4),
            color  = dict(subset = LBL_I.IS, color = ['#B50064', None])
        )
        
        s = stat[STAT.SH]                
        c = dict(subset = LBL.S) if LBL.S in s.columns else None
        s = Pd.style(
            s,  
            format = dict(precision=3), color = c
        )

        o = Pd.style(
            stat[STAT.PF],
            format = dict(precision=4)
        )

        return TableStyle.combine_pdf_to_html(
            [s, g, o], 
            [
                'Feature Shap Value Drift', 
                'Feature Gini Added Value to the Model',
                'Model Performance '
            ], 
            tables_per_block = 2, 
            h_align = 'left', 
            show = False
        )

    @staticmethod
    def _html(a:int, stats:str, fig:str) -> str:
        t1 = (
            f'Agent {a} → Model Sensitivity Overview → '
            'Ordered by Training Feature Importance'
        ).upper()        
        t2 = f'Agent {a} → WOE View to Features Behavior on Test Data'.upper()
        s  = lambda x: Format.StyleStats(margin_t=0, margin_b=x, to_html=False)

        el = [
            f'{Format.title(t1, style=s(25), show=False)} {stats}',
            f'{Format.title(t2, style=s(10), show=False)} {fig}'
        ]
        return TableStyle.format_html(el)
    
    def create(self) -> dict[int, str]:
        """Returns HTML string of Agents Performance on Validation and Test Samples"""
        html = {}
        for a, fig in self.figs.items():
            s = self._stats_to_html(a)
            f = fig.to_html(full_html=False, include_plotlyjs='cdn')
            html[a] = self._html(a, s, f)
        return html
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SelectorGiniPlot:
    """
    Gini and Lift Plot — Training vs Champion Agent Comparison.

    Creates a dual-panel plot showing:
    - **Gini curve**: Cumulative Gini as a function of the predicted
    probability quantile (AUC-like behavior).
    - **Lift curve**: Cumulative lift at each probability bin.

    Both curves are overlaid for the Training Agent (full feature set)
    and the Champion Agent (user-selected subset), enabling a visual
    assessment of the impact of the selection decision.

    Parameters
    ----------
    diagnostics : dict[str, DiagnosticsResult]
        Side-by-side diagnostics: {DIAG.T: training, DIAG.C: champion}.
    update_layout : dict, default=None
        Additional Plotly layout overrides.
    """   
    def __init__(self, 
        diagnostics:   dict[str, DiagnosticsResult], 
        update_layout: dict = None
    ):
        t = GiniLiftPlot.title_helper(diagnostics)
        p = 'Test Performance'
        titles = [
            f'Gini {p} → {t[LBL_D.G]}',
            f'Lift {p} → {t[LBL_D.L]}'
        ]
        # Champion: 'rgb( 60,  95, 140)' | 'rgb(105, 120, 150)'
        #           'rgb( 90, 145, 170)' | 'rgb( 15,  90, 185)'
        colors = ['rgb(150, 155, 170)', 'rgb( 90, 145, 170)']
        self._gini_plot = GiniLiftPlot(
            diagnostics, plot_titles=titles, update_layout=update_layout, theme=colors
        )
    
    def create(self, show:bool = True):
        """
        Render the Gini and Lift comparison plot.

        Returns
        -------
        go.Figure
            Dual-panel figure (Gini + Lift) with both agents overlaid.
        """
        return self._gini_plot.create(show)        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬