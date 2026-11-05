"""
Population Stability Index (PSI) — Model & Feature Drift Monitoring.

This module provides Population Stability Index computation for both 
model-level and feature-level drift detection. It supports a dual analysis paradigm:

* **Volume PSI** — Measures distributional shift of model predictions and feature 
  values between the expected (training) and actual (production/evaluation)
  populations.

* **Lift PSI** — Measures the shift in *risk contribution* by weighting each
  bin's distribution change by its target (bad-rate) lift, ensuring that
  drift in low-risk segments does not mask drift in high-risk segments.
"""

from ...deps.common         import  Path
from ...deps.data           import  (
                                    np,  pd, 
                                    pDF, pSe, 
                                    pd_Styler
                                    )
from ...deps.plots          import  (
                                    go, 
                                    make_subplots
                                    )
from ...deps.data_view      import  itables_show
from ...deps.types          import  (
                                    Literal, 
                                    FuncType
                                    )

from ...core.factory        import  (
                                    Dataclass,
                                    DictClass
                                    )
from ...core.display        import  (
                                    Display,
                                    DashPlotlyLink
                                    )
from ...core.interop        import  Pd
from ...core.standard       import  TypeDef

from ...display.plots       import  Plot
from ...display.style       import  (
                                    TableStyle, 
                                    ItablesStyle
                                    )
from ...utils.builtins      import  Str

from .models                import  LGB, XGB, Logit, Lasso, ElasticNet, LogReg


type InputModel = Logit | Lasso | ElasticNet | LogReg | LGB | XGB


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class PSI(DictClass):
    F  = 'Feature'
    W  = 'Weight'
    A  = 'Actual'
    E  = 'Expected'
    SE = f'Share|{E}'
    SA = f'Share|{A}'
    LE = f'Lift|{E}'
    LA = f'Lift|{A}'    
    NL = lambda x: x.split('|')[1]
    VU = 'Vol_PSI'
    VW = 'W(Vol_PSI)'
    LU = 'Lift_PSI'
    LW = 'W(Lift_PSI)'
    VS = 'Vol_Status'
    LS = 'Lift_Status'
    IS = 'Stable'
    PF = 'Features'
    PM = 'Model'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# POPULATION STABILITY INDEX
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class PopulationStabilityResult:
    model_psi:    tuple
    feature_psi:  tuple

class PopulationStability:
    """
    Core PSI computation engine for model and feature population drift.

    Computes Population Stability Index metrics for both the overall model
    (prediction distribution) and individual features. Supports two PSI
    dimensions:

    - **Volume PSI**: Distribution shift in raw feature/prediction values.
    - **Lift PSI**: Distribution shift in target impact per bin, weighted
      by model feature importance.

    The engine is model-agnostic: it automatically detects whether the
    provided model is a GBM (using SHAP weights) or a linear model
    (using coefficient weights).

    Parameters:
        model (InputModel): A fitted ML model with *features* and *shap_values* API.
        target (str): Column name of the target variable in the dataframes.
        bins (int): Number of quantile bins for numeric feature discretization.
        eps (float): Small constant to prevent division by zero in the
            PSI logarithmic calculation.

    Example:
        >>> from aidflow.ml.psi import PopulationStability
        >>> psi = PopulationStability(model=fitted_model, target='y', bins=10)
        >>> result = psi.create(
        ...     data_expected = train_df,
        ...     data_actual   = test_df,
        ...     pred_expected = train_pred,
        ...     pred_actual   = test_pred
        ... )
        >>> result.model_psi[0]  # {'Vol_PSI': 0.05, 'Lift_PSI': 0.12}
    """
    def __init__(self, 
        model:   InputModel,         
        target:  str,
        bins:    int   = 10,
        eps:     float = 1e-6,        
    ):
        self.model  = model
        self.target = target
        self.bins   = bins
        self.eps    = eps

        self.is_gbm = hasattr(model, '_Booster')
        self.fes    = self.model.features

        self._t_mean = {}

        self._verify()
    # ─────────────────────────────────────────────────────────────────────────────────────────────        

    def _verify(self):
        if self.fes is None:
            raise ValueError(
                'AIG SAFETY ► Provided model must be fitted with populated features. '
                'Provided model features are None.'
            )
        if not hasattr(self.model, 'features'):
            raise TypeError(
                'AIG SAFETY ► Unsupported model type → `features` property is not present.'
            )
        if not hasattr(self.model, 'shap_values'):
            raise TypeError(
                'AIG SAFETY ► Unsupported model type → `shap_values` method is not present.'                
            )
    # ─────────────────────────────────────────────────────────────────────────────────────────────        
    
    @staticmethod
    def _get_bins(bins:pSe|np.ndarray) -> list:
        if isinstance(bins, pSe):
            bins:pd.Categorical = bins.cat
        return sorted([x.left for x in bins.categories] + [np.inf])

    def _get_fe_se(self, d:pDF, fe:str, bins:pSe|np.ndarray=None) -> pSe:
        """
        Extract and optionally bin a single feature column.

        For GBM models with numeric features, applies quantile binning.
        For linear models returns the raw series.
        """        
        x = d[fe]
        if not self.is_gbm:
            return x
        if x.dtype in TypeDef.numeric:
            if bins is not None:
                return Pd.cut(x, bins = self._get_bins(bins), duplicates='drop')
            return Pd.qcut(x, self.bins, duplicates='drop')
        return x

    def _get_weight(self, d:pDF) -> pSe:
        """
        Compute feature importance weights from the fitted model.

        For GBM models, uses the mean absolute SHAP value across the
        provided dataset. For linear models, uses the raw coefficients.
        """        
        if self.is_gbm:
            s = self.model.shap_values(d[self.fes])
            c = np.abs(s).mean(axis=0)
        else:
            c = self.model.coef_
        return pSe(c, index = self.fes)

    def _get_weight_lbl(self):
        """Generate the column label for the weight based on model type."""
        if self.is_gbm:
            l = 'SHAP'
        else:
            l = 'Coef'
        return f'{PSI.W} ({l})'    
    # ─────────────────────────────────────────────────────────────────────────────────────────────        
        
    def _get_target_mean(self, d:pDF|pSe, e_a:str) -> float:
        """Retrieve (or compute and cache) the mean target value for a sample."""
        m = self._t_mean.get(e_a)
        if m is None:
            if isinstance(d, pDF):
                d = d[self.target]
            m = d.mean()
            self._t_mean[e_a] = m if m else np.nan
        return self._t_mean[e_a]
                
    def _psi(self, expected:pSe, actual:pSe) -> pSe:
        """
        Compute the raw PSI values for aligned distributions.

        Applies the standard PSI formula: sum((actual - expected) * ln(actual / expected)).        
        """
        return (actual - expected) * np.log(actual.clip(self.eps) / expected.clip(self.eps))    

    def _lift_bd(self, target:pSe, fe:pSe, e_a:str) -> tuple[pSe, pSe]:
        """Compute lift and bad-rate distributions per bin."""        
        m = target.groupby(fe).agg(['mean', 'count', 'sum'])
        if isinstance(m.columns, pd.MultiIndex):
            m.columns = m.columns.get_level_values(1)
        m = m.where(m['count'] > 30, np.nan)
        return (
            m['mean'] / self._get_target_mean(target, e_a), # lift → to stats - graphs
            m['sum'] / m['sum'].sum()                       # bad dist → directly plugable to _psi()
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────        
    
    @staticmethod
    def _dist(d:pSe) -> pSe:
        """Compute the normalized frequency distribution of a series."""
        return d.value_counts(normalize=True, sort=False).sort_index()

    @staticmethod
    def _align_fe_data(d0:pSe, d1:pSe) -> tuple[pSe, pSe]:
        """
        Align two distributions to a common index for element-wise comparison.

        Missing categories in either distribution are filled with zero.
        """        
        idx = d0.index.union(d1.index)
        d0_ = d0.reindex(idx, fill_value=0)
        d1_ = d1.reindex(idx, fill_value=0)
        return d0_, d1_
    
    @staticmethod
    def _concat(d0:pSe, d1:pSe, l0:pSe, l1:pSe) -> pDF:
        """
        Assemble a per-bin comparison dataframe from distribution and lift series.

        Parameters:
            d0 (pSe): Expected share distribution.
            d1 (pSe): Actual share distribution.
            l0 (pSe): Expected lift values.
            l1 (pSe): Actual lift values.

        Returns:
            A dataframe with columns [Share|Expected, Share|Actual, 
            Lift|Expected, Lift|Actual].
        """        
        d = Pd.concat([d0, d1, l0, l1], axis=1)
        d.columns = [PSI.SE, PSI.SA, PSI.LE, PSI.LA]
        return d

    @staticmethod
    def _status(psi: pSe) -> pSe:
        """
        Classify PSI values into stability status categories.

        Thresholds:
            - Stable:   PSI < 0.10
            - Warning:  0.10 ≤ PSI < 0.25
            - Critical: PSI ≥ 0.25
        """        
        return Pd.cut(
            psi,
            bins   = [-np.inf, 0.10, 0.25, np.inf],
            labels = ["Stable (<0.10)", "Warning (0.10-0.25)", "Critical (>=0.25)"],
            duplicates = 'drop'
        )

    @staticmethod
    def _result_psi(vol:float, lift:float) -> dict[str, float]:
        """Package volume and lift PSI into the standard result dictionary."""        
        return {PSI.VU: vol, PSI.LU: lift}
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _fe_dist(self, expected:pSe, actual:pSe) -> tuple[pSe, pSe]:
        """Compute and align normalized distributions for a single feature."""
        d0 = self._dist(expected)
        d1 = self._dist(actual) 
        return self._align_fe_data(d0, d1)

    def _fe_lift(self, expected:pSe, actual:pSe, target_e:pSe, target_a:pSe, 
    ) -> tuple[tuple[pSe, pSe], tuple[pSe, pSe]]:
        """Compute aligned lift and bad-rate distributions for a single feature."""
        l0, b0 = self._lift_bd(target_e, expected, PSI.LE)
        l1, b1 = self._lift_bd(target_a,  actual,  PSI.LA)
        return (
            self._align_fe_data(l0, l1), 
            self._align_fe_data(b0, b1)
        )

    def _add_weighted_psi(self, psi:pDF) -> None:
        """
        Compute model-weighted PSI and assign stability status labels.

        Multiplies each PSI value by the normalized model weight (SHAP or Coef)
        to produce a weighted PSI that reflects the feature's contribution
        to overall model output.
        """
        w = self._get_weight_lbl()
        weight = psi[w] / psi[w].sum()

        for v_l in ['volume','lift']:
            if v_l == 'volume':
                  w, v, s = PSI.VW, PSI.VU, PSI.VS
            else: w, v, s = PSI.LW, PSI.LU, PSI.LS

            psi[w] = psi[v] * weight
            psi[s] = self._status(psi[w])

    def feature_psi(self, expected:pDF, actual:pDF) -> tuple[dict[str, float], pDF, dict[str, pDF]]:
        """
        Compute per-feature Population Stability Index.

        Evaluates each feature individually, measuring both the distribution
        shift (Volume PSI) and the target-impact shift (Lift PSI) between
        the expected and actual populations. Results are weighted by model
        importance (SHAP for GBM, Coef for linear).

        Parameters:
            expected (pDF): The reference dataset (e.g., training population).
                Must contain all model features and the target column.
            actual (pDF): The comparison dataset (e.g., new/test population).
                Must contain all model features and the target column.

        Returns:
            tuple[dict, pDF, dict]:
                - dict: Aggregated summary — {'Vol_PSI': float, 'Lift_PSI': float}.
                - pDF: Per-feature PSI table with columns including raw PSI,
                  weighted PSI, status labels, and overall stability flag.
                - dict: Mapping of feature name → per-bin distribution
                  comparison dataframe (for visualization).

        Example:
            >>> summary, table, dists = engine.feature_psi(train_df, test_df)
            >>> table  # Feature-level PSI scores with stability status
            >>> dists['age']  # Per-bin share and lift comparison for 'age'
        """
        dist, psi = {}, []

        for fe in self.fes:
            e = self._get_fe_se(expected, fe)
            a = self._get_fe_se(actual, fe, e.unique())

            d0,  d1  = self._fe_dist(e, a)
            l01, b01 = self._fe_lift(e, a, expected[self.target], actual[self.target])

            dist[fe] = self._concat(d0, d1, *l01)

            psi_v = self._psi(d0, d1)
            psi_b = self._psi(*b01)
            psi.append({PSI.F: fe, PSI.VU: psi_v.sum(axis=0), PSI.LU: psi_b.sum(axis=0)})

        w = self._get_weight_lbl()

        psi = pDF(psi).set_index(PSI.F)
        psi[w] = self._get_weight(expected)

        self._add_weighted_psi(psi)

        psi[PSI.IS] = ( psi[PSI.VW] < 0.1 ) & ( psi[PSI.LW] < 0.1 )

        columns  = [w, PSI.VU, PSI.LU, PSI.VW, PSI.LW, PSI.VS, PSI.LS, PSI.IS]
        fe_psi   = psi[columns].reset_index()
        data_psi = psi[PSI.VW].sum()
        lift_psi = psi[PSI.LW].sum()

        return self._result_psi(data_psi, lift_psi), fe_psi, dist 
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _align_data(d:pSe|np.ndarray) -> pSe:        
        """Ensure input data is a pandas Series."""
        if isinstance(d, pSe):
            return d
        return pSe(d)
    
    def model_psi(self, 
        pred_expected:   pSe|np.ndarray, 
        pred_actual:     pSe|np.ndarray, 
        target_expected: pSe|np.ndarray, 
        target_actual:   pSe|np.ndarray  
    ) -> tuple[dict[str, float], pDF]:
        """
        Compute model-level Population Stability Index.

        Measures the shift in the model's prediction distribution and the
        corresponding target-impact (lift) distribution between two populations.
        Predictions are binned using quantile cuts derived from the expected
        sample to ensure consistent bin boundaries.

        Parameters:
            pred_expected (pSe | np.ndarray): Model predictions for the
                reference (expected) population.
            pred_actual (pSe | np.ndarray): Model predictions for the
                comparison (actual) population.
            target_expected (pSe | np.ndarray): Actual target values for
                the expected population.
            target_actual (pSe | np.ndarray): Actual target values for
                the comparison population.

        Returns:
            tuple[dict, pDF]:
                - dict: Aggregated summary — {'Vol_PSI': float, 'Lift_PSI': float}.
                - pDF: Per-bin distribution comparison with columns
                  [Share|Expected, Share|Actual, Lift|Expected, Lift|Actual].

        Example:
            >>> summary, bin_table = engine.model_psi(
            ...     pred_expected   = train_pred,
            ...     pred_actual     = test_pred,
            ...     target_expected = train_target,
            ...     target_actual   = test_target
            ... )
            >>> summary  # {'Vol_PSI': 0.08, 'Lift_PSI': 0.15}
        """
        m0 = Pd.qcut(self._align_data(pred_expected), self.bins, duplicates = 'drop')
        m1 = Pd.cut(self._align_data(pred_actual), bins = self._get_bins(m0), duplicates = 'drop')

        m0_ = self._dist(m0)
        m1_ = self._dist(m1)

        l0, b0 = self._lift_bd(self._align_data(target_expected), m0, PSI.LE)
        l1, b1 = self._lift_bd(self._align_data(target_actual),   m1, PSI.LA)

        dist  = self._concat(m0_, m1_, l0, l1)
        m_psi =  self._psi(m0_, m1_).sum()
        l_psi =  self._psi(b0, b1).sum()
        
        return self._result_psi(m_psi, l_psi), dist
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _align_target(self, expected:pDF, actual:pDF) -> tuple[pSe, pSe]:
        """Extract target columns from both datasets."""
        return expected[self.target], actual[self.target]
    
    def create(self, 
        data_expected: pDF, 
        data_actual:   pDF, 
        pred_expected: pSe,
        pred_actual:   pSe
    ) -> PopulationStabilityResult:
        """
        Execute the full PSI analysis for both model and feature levels.

        This is the primary entry point for PSI computation. It runs
        `model_psi` and `feature_psi` in sequence and packages
        the results into a single `PopulationStabilityResult` object.

        Parameters:
            data_expected (pDF): Reference population dataframe containing
                all model features and the target column.
            data_actual (pDF): Comparison population dataframe with the
                same schema as `data_expected`.
            pred_expected (pSe): Model predictions for the expected population.
            pred_actual (pSe): Model predictions for the actual population.

        Returns:
            PopulationStabilityResult: A structured result containing:
                - `.model_psi`:   (summary_dict, bin_comparison_df)
                - `.feature_psi`: (summary_dict, feature_table_df, per_feature_dists)

        Example:
            >>> result = engine.create(
            ...     data_expected = train_df,
            ...     data_actual   = production_df,
            ...     pred_expected = train_predictions,
            ...     pred_actual   = production_predictions
            ... )
            >>> result.model_psi[0]
            {'Vol_PSI': 0.06, 'Lift_PSI': 0.11}
            >>> result.feature_psi[1].head()  # Per-feature scores
        """        
        te, ta = self._align_target(data_expected, data_actual) 
        return PopulationStabilityResult(
            model_psi   = self.model_psi(pred_expected, pred_actual, te, ta),
            feature_psi = self.feature_psi(data_expected, data_actual)
        )    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PopulationStabilityPlot:
    """
    Interactive visualization engine for PSI results.

    Renders Plotly-based charts and styled HTML tables for PSI analysis
    output. Supports three primary visualizations:

    - **Model PSI Graph**: Dual-axis bar+line chart showing prediction
      bin distribution shift and lift change.
    - **Feature PSI Grid**: A subplots grid where each feature gets its
      own distribution+lift comparison chart.
    - **Feature Overview Table**: A styled HTML table (via itables or
      pandas) with color-coded stability indicators.

    All visualizations respect the AIDflow display system, integrating
    with both Jupyter notebooks and Dash-based report generation.

    Parameters:
        stability (PopulationStabilityResult): The computed PSI results
            from `PopulationStability.create()`.
        report_path (str): Optional file path prefix for saving generated
            plot artifacts (used in report mode).
        trace_names (list|tuple): Two-element array of display names for the
            Expected and Actual samples (e.g., ['Training', 'Production']).
        theme (list|tuple): Two-element array of display color theme for the
            Expected and Actual samples (e.g., ['rgb( 33, 115, 115)', 'rgb(165,  75, 155)']).

    Example:
        >>> plotter = PopulationStabilityPlot(
        ...     stability   = result,
        ...     trace_names = ['Training', 'Q3 Production']
        ... )
        >>> plotter.dash()  # Renders full report: PSI table + model graph + feature grid
    """

    def __init__(self, 
        stability:   PopulationStabilityResult, 
        report_path: str = None,
        trace_names: tuple = ('Expected', 'Actual'),
        theme:       tuple = ('rgb( 33, 115, 115)', 'rgb(165,  75, 155)')
    ):
        self.stab  = stability
        self.path  = report_path
        self.trace = dict(zip((PSI.E, PSI.A), trace_names))
        self.theme = dict(zip((PSI.E, PSI.A), theme))

    @staticmethod
    def _get_name(e_a:str) -> str:                
        """Extract the sample name from a composite column label."""
        return PSI.NL(e_a)
    
    def _get_color(self, e_a:str) -> str:
        """Retrieve the assigned color for a sample."""
        return self.theme[self._get_name(e_a)]
    
    def _trace_name(self, e_a: str, ext:str='') -> str:
        """Build a human-readable trace name for chart legends."""
        s = self.trace[self._get_name(e_a)]
        if not ext:
            return s        
        return ext +' '+s

    @staticmethod
    def _x_axis(ax:pSe) -> list:
        """Data format of x axis."""
        ax = ax.index.astype(str)
        return [Str.max_len(x, 25, 20) for x in ax]
    
    @staticmethod
    def _x_axis_angle(ax:pSe, is_fe:bool) -> dict:
        if not is_fe:
            return {}
        x = ax.index.astype(str).str.len().max()
        if x < 15:
            return {}
        return dict(tickangle=35)
    
    @staticmethod
    def _desc_name(is_fe:bool) -> str:
        """Determine the x-axis label based on encoding type."""
        return 'Bin' if is_fe else 'E(P)'
        
    def _hist_trace(self, data:pDF, e_a:str, is_fe:bool) -> go.Bar:
        """Create a Plotly bar trace for share distribution."""
        t = self._desc_name(is_fe)
        return go.Bar(
            y = data[e_a],
            x = self._x_axis(data[e_a]),
            name          = self._trace_name(e_a, 'Share'),
            marker_color  = self._get_color(e_a),
            opacity       = 0.35,
            hovertemplate = '<br>Share: %{y:.2%}'+f'<br>{t}: '+'%{x}',
            zorder = 1,
        )

    def _lift_trace(self, data:pDF, e_a:str, is_fe:bool) -> go.Scatter:
        """Create a Plotly scatter trace for lift values."""
        t = self._desc_name(is_fe)
        return go.Scatter(
            y = data[e_a],
            x = self._x_axis(data[e_a]), 
            name = self._trace_name(e_a, 'Lift'),
            line = dict(
                color = self._get_color(e_a),
                width = 2,
            ),
            hovertemplate = '<br>E(Lift): %{y:.3f}'+f'<br>{t}: '+'%{x}',
            zorder = 0,
        )

    def _model_title(self, font:FuncType) -> dict:
        """Title of the main model grpah."""
        return dict(
            text    = f'Model Prediction Stability → Bins by {self._trace_name(PSI.SE)} Sample',
            x       = 0.5, 
            xanchor = 'center', 
            yanchor = 'top',
            font    = font(12)
        )

    def _graph(self, data: pDF, is_fe:bool=True) -> go.Figure:
        """
        Assemble a dual-axis (bar + line) Plotly figure for one PSI comparison.

        Renders share distributions as bars (primary y-axis) and lift
        values as lines (secondary y-axis) over common bin categories.
        """        
        fig = make_subplots(specs=[[{"secondary_y": True}]]) 

        for s in [PSI.SE, PSI.SA]:
            fig.add_trace(self._hist_trace(data, s, is_fe))

        for l in [PSI.LE, PSI.LA]:
            fig.add_trace(self._lift_trace(data, l, is_fe), secondary_y=True)

        font  = lambda s=11: dict(family="Arial", size=s, color="black")
        title = lambda t: dict(text=t, font=font())
        axis  = lambda t: dict(title=title(t))

        fig.update_layout(
            height = 400,
            width  = 450 if is_fe else None,
            margin = dict(t=40, b=10, l=0, r=0),

            font  = font(),
            title = None if is_fe else self._model_title(font),

            xaxis  = axis(None if is_fe else 'Probability Bin') | self._x_axis_angle(data[s], is_fe),
            yaxis  = axis('E(Share)'),
            yaxis2 = axis('E(Lift)'),
            
            legend = dict(
                orientation = "h", 
                yanchor     = "bottom", 
                xanchor     = "center", 
                x = 0.5,
                y = -0.3, 
            ),
            
            bargap      = 0.35,
            bargroupgap = 0.1,
        )

        return fig
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _combine(tables: list, titles: list = None, show: bool = True) -> str | None:
        """Combine styled dataframes into a single HTML report block."""        
        return TableStyle.combine_pdf_to_html(
            pdf_tables       = tables, 
            titles           = titles,
            tables_per_block = 1, 
            p_tables_per_row = 1,
            h_cols_per_row   = 1,
            h_align          = 'center', 
            show             = show
        )

    @staticmethod
    def _style(d: pDF, hide_axis:int=None) -> pDF:
        """Apply consistent formatting to a PSI results table."""
        f = '{:.4f}'
        x = [x for x in d.columns if x not in [PSI.F, PSI.VS, PSI.LS, PSI.IS]]
        w = [x for x in d.columns if PSI.W+' (' in x]
        return Pd.style(
            d, 
            hide_axis   = hide_axis,
            format_cols = {c: f for c in x},
            color       = {'subset': PSI.IS, 'color': ['#B50064;', None]},
            # properties  = {'subset': [PSI.VS, PSI.LS] + w, 'width':'150px'}
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _fes(self) -> list:
        """Retrieve the list of feature names from the computed results."""
        return self.stab.feature_psi[1][PSI.F].to_list()
    
    def _fe_graph(self, x:str) -> go.Figure:
        """Generate the Plotly figure for a single feature's PSI comparison."""
        s = self.stab.feature_psi[2]
        return self._graph(s[x])    
    
    def _fe_title(self, x:str) -> str:
        """Generate the subplot title for a feature, including its status."""
        s:pDF = self.stab.feature_psi[1]
        f = s.loc[s[PSI.F]==x].iloc[0]
        return f'{x} → {f[PSI.VS]} | {round(f[PSI.VW], 4)}'

    def _grid_title(self) -> str:
        """Generate the overall title for the feature PSI grid."""
        return (
            'Feature Population Stability → '
            f'{self._trace_name(PSI.SE)} vs. {self._trace_name(PSI.SA)}'.upper()
        )
    
    def _grid(self) -> go.Figure:
        """
        Assemble the full feature PSI subplots grid.

        Creates a multi-panel figure where each feature gets its own
        distribution+lift comparison chart, arranged in a responsive grid.
        """
        return Plot.create_subplots(
            self._fes(),
            self._fe_graph,
            self._fe_title,
            height = 300,
            width  = 450,
            secondary_y = True,
            horizontal_spacing = 0.2,
            vertical_spacing   = 0.35,
            grid_title = self._grid_title(),
            update_layout = {'margin': dict(t=70, b=50, l=0, r=0)},
            desc_axes_by_first_only = True
        )

    def _save(self, fig: go.Figure | Path) -> str:
        """Save a Plotly figure and return an embeddable HTML link."""
        link = DashPlotlyLink(
            figure_path   = fig, 
            report_path   = self.path,
            hide_columns  = True,
            show          = False
        ).get_output()
        return TableStyle.combine_pdf_to_html(
            [link], ['Feature PSI'], h_align='center', show=False
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _psi(self) -> pDF:
        """
        Assemble the aggregated PSI summary table.

        Combines model-level and feature-level PSI scores into a
        two-row summary table with status classifications.
        """
        m = self.stab.model_psi[0]
        d = self.stab.feature_psi[0]

        p = pDF({PSI.VU: [m[PSI.VU], d[PSI.VU]], PSI.LU: [m[PSI.LU], d[PSI.LU]]})

        p[PSI.VS] = PopulationStability._status(p[PSI.VU])
        p[PSI.LS] = PopulationStability._status(p[PSI.LU])

        t = lambda x: x.split('_')[0]
        p.index = [t(PSI.PM), t(PSI.PF)]

        return p        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def psi(self, show:bool=True) -> str|None:
        """
        Display or return the aggregated PSI summary table.

        Renders a compact two-row table showing model-level and
        feature-level PSI scores with their stability status.

        Parameters:
            show (bool): If True, renders the table inline (e.g., in Jupyter).
                If False, returns the HTML string for embedding.

        Returns:
            str | None: HTML string if ``show`` is False, otherwise None.
        """        
        p = self._psi()
        t = TableStyle.combine_pdf_to_html(
            pdf_tables  = [Pd.style(p, format_cols = {x:'{:.4f}' for x in [PSI.VU, PSI.LU]})],
            titles      = ['Aggregated Result'],
            h_align     = 'center',
            show        = False
        )   
        if not show:
            return t
        Display.construct(t)     

    def model_graph(self, show:bool=True) ->  go.Figure|None:
        """
        Display or return the model-level PSI distribution chart.

        Renders a dual-axis chart showing the prediction bin distribution
        shift (bars) and target lift change (lines) between the two samples.

        Parameters:
            show (bool): If True, displays the figure. If False, returns it.

        Returns:
            go.Figure | None: The Plotly figure if ``show`` is False.
        """        
        sta = self.stab.model_psi
        fig = self._graph(sta[1], False)
        if not show:
            return fig
        fig.show()

    def fe_grid(self, show:bool=True, load_path:Path=None) -> str|None:
        """
        Display or return the full feature PSI grid.

        Renders a multi-panel grid where each feature has its own
        distribution+lift comparison chart, with per-feature status
        labels in the subplot titles.

        Parameters:
            show (bool): If True, renders the grid inline.
                If False, returns the HTML string for report embedding.
            load_path (Path): Optionally, providing a path for previously 
            created grid HTML file will skip graphs creation and use provided 
            link instead.

        Returns:
            str | None: HTML string if ``show`` is False, otherwise None.
        """        
        f = load_path
        if not f:
            f = self._grid()

        g = self._save(f)  
        if not show:
            return g
        Display.construct(g)

    def fe_overview(self, 
        kind: Literal['itables', 'pd'] = 'itables', 
        show: bool = True
    ) -> pd_Styler|None:
        """
        Display or return the feature-level PSI overview table.

        Renders a styled table of all features with their raw and weighted
        PSI scores, stability status, and an overall stability flag
        (color-coded: green for stable, red for unstable).

        Parameters:
            kind (str): Rendering engine. 'itables' for interactive
                DataTables (Jupyter), 'pd' for static pandas styling.
            show (bool): If True, renders inline. If False, returns the
                styled object for programmatic use.

        Returns:
            pd_Styler | None: The styled DataFrame if ``show`` is False.
        """        
        d: pDF = self._style( self.stab.feature_psi[1] )
        if not show:
            return d
        if kind == 'itables':
            js  = ItablesStyle.bool_color(PSI.IS, ['#2BCF91', '#B50064'])
            cd  = [{"targets": -1, "createdCell": js}]
            css = "thead th { font-size: 10px !important; }"
            itables_show(self._style(d), columnDefs=cd, update_css=css)
        else:
            self._combine([self._style(d)])

    def dash(self, grid_link: Path = None) -> None:
        """
        Render the complete PSI report in sequence.

        Executes all visualization components in a fixed order:
        1. Aggregated PSI summary table
        2. Model-level prediction stability chart
        3. Feature-level PSI grid
        4. Feature overview table

        This is the "one-shot" method for displaying the full PSI
        analysis in a single call (e.g., in a Jupyter notebook cell).

        Parameters
        ----------
        grid_link : Path = None
            Optionally, providing a path for previously created grid HTML
            file will skip graphs creation and use provided link instead.
        """
        self.psi()
        print()
        self.model_graph()
        print()
        self.fe_grid(load_path = grid_link)
        self.fe_overview()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬