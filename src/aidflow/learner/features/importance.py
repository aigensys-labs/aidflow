"""
Feature Importance — Multi-Perspective Attribution Engine.

This module implements a comprehensive framework for assessing
feature impact through three complementary methodologies:

1. **Ablation**: Measures the residual performance change (ΔGini)
   when a feature is removed. For GLM models, uses an analytic
   Schur-complement solver (O(1) per feature, no refit). For GBM
   models, uses resampling (O(p) refits). Significance is
   assessed via the DeLong test.

2. **Permutation Importance**: Assesses sensitivity to
   information destruction. Shuffles each feature's values and
   measures the resulting performance degradation. Model-agnostic —
   works with any estimator.

3. **SHAP (Shapley Additive exPlanations)**: Quantifies each
   feature's marginal contribution to the prediction relative to
   the model baseline. Uses the model's internal structure when
   available (tree-based: 100x faster; linear: structured
   decomposition).

Consensus Metrics
-----------------
- **Consensus rank** (median): The central tendency across
  methods — the "most likely" importance ordering.
- **Consistency score** (0–1): How much the methods agree on
  the relative ordering. 1.0 = perfect agreement; 0.0 =
  complete disagreement.
- **Pairwise Kendall τ**: Rank correlation between each pair
  of methods. Low values indicate collinearity, proxy features,
  or interactions that different methods resolve differently.
- **Kendall W** (0–1): Multi-rater agreement across all methods
  simultaneously.
"""

from ...deps.data           import  pDF, pSe, np
from ...deps.data_view      import  itables_html
from ...deps.ml.glm         import  LogisticRegression
from ...deps.ml.diag        import  (
                                    Explainer,
                                    TreeExplainer,
                                    LinearExplainer,
                                    permutation_importance,
                                    )
from ...deps.ml.stats       import  kendalltau
from ...deps.display        import  display
from ...deps.types          import  (
                                    Literal, 
                                    NDArray, 
                                    )

from ...core.factory        import  (
                                    Dataclass, 
                                    DictClass
                                    )
from ...core.interop        import  Pd
from ...core.display        import  Display
from ...core.accessor       import  Format

from ...ml.models           import  (
                                    ModelUtils, 
                                    ModelView
                                    )
from ...ml.models.ablation  import  (
                                    LBL as LBL_FA,
                                    FeatureAblation as _FeatureAblation
                                    )
from ...ml.tests            import  TestSampleDiff
from ...ml.sampling         import  DataUtils

from ...utils.builtins      import  List
from ...display.style       import  (
                                    TableStyle, 
                                    ItablesStyle
                                    )

from .estimator             import  FeatureEstimator
from .sampling              import  SamplingMixin


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    F  = 'Feature'
    A  = 'Ablation_ΔGini'
    P  = 'Permutation_ΔGini'
    S  = 'SHAP_ΔLog_of_odds'
    IS = 'SignificantΔ'
    C  = 'Consensus'
    CS = 'Consistency'
    STD= lambda x: f's({x})'

class MTD(DictClass):
    A = 'ablation'
    P = 'permutation'
    S = 'shap'

class VIEW(DictClass):
    AS = 'assessment_shap_fig'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FEATURE BASE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FeatureUtils:
    
    @staticmethod
    def is_agg_shap_fig(kwargs:dict) -> bool:
        """FeatureImportance run indicator."""        
        return kwargs.get(VIEW.AS, False)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeatureBase:
    """
    Base class for single-method feature importance assessment.

    Provides the common contract for all importance methods:
    model resolution, feature detection, result storage, and display formatting. 
    Subclasses implement the `_fit` method with their specific computation.
    
    Parameters
    -----------
    model : LogisticRegression = None
        Fitted model with standard .fit()/.predict_proba() api can be provided.
        Defaults to LogisticRegression when None.

    random_state : int   = 42
        Seed for reproducibility.
    """

    def __init__(self,
        model:          LogisticRegression = None,
        random_state:   int   = 42,
        kwargs:         dict  = None
    ):
        self.model   = model
        self.random  = random_state
        self._set_init()

        self._is_agg = FeatureUtils.is_agg_shap_fig(kwargs or {})
        self._est = FeatureEstimator(model, random_state=random_state)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @property
    def features(self):
        """Returns Feature effect result."""
        return self.result
        
    def show_result(self) -> None:
        """Display the formatted importance table."""
        self._show_result()

    def _show_result(self, format:dict=None) -> None:
        display( Pd.style(
            self.result, 
            format_cols = format, 
            properties  = {'subset':LBL.F, 'width':'400px', 'text-align':'left'}
        ))        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_init(self) -> None:
        self.result = []
        self.fig = None

    def _set_fes(self, X:pDF) -> None:
        self._fes = DataUtils.get_fes(X)

    def _set_figure(self, fig) -> None:
        self.fig = fig

    def _set_result(self, result:dict|pDF, sort_by:str=None, ascending:bool=True) -> None:        
        if not sort_by:
            self.result = result
            return
        self.result = pDF(result).sort_values(sort_by, ascending=ascending).reset_index(drop=True)    

    def _is_fitted(self) -> bool:
        if self.model is None:
            return False
        if self._est.get_model() is not self.model:
            return False
        p = lambda x: getattr(self.model, x, None) is not None
        return (
            p('coef_') or p('_Booster')
        )  
    
    def _set_model(self, X:pDF, y:pSe) -> None:
        if self._is_fitted():
            return
        self.model = self._fit_model(X, y)  

    def _fit_model(self, X:pDF, y:pSe):
        m = self._est.get_estimator_instance(X)
        return m.fit(X,y)
  
    def _fit(self, X:pDF, y:pSe):
        raise NotImplemented('Worker class must implement "_fit()" method.')
    
    def fit(self, X:pDF, y:pSe, *args, **kwargs):
        """
        Execute the feature importance assessment.
        Resolves the model, detects features, and stores the sorted result.

        Parameters
        ----------
        X : pDF
            Feature matrix (evaluation sample).
        y : pSe
            Target vector (evaluation sample).

        Returns
        -------
        self
            Fitted instance with `.result` set.
        """
        self._set_init()
        self._set_fes(X)
        self._set_model(X, y)
        res = self._fit(X, y, *args, **kwargs)
        self._set_result(**res)
        return self     
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FEATURE ASSESSMENT
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FeatureAblation(FeatureBase):
    """
    Feature Importance via Ablation (Residual Change).

    Measures the individual contribution of each feature by
    quantifying the performance degradation (ΔGini) when the
    feature is removed from the model.

    Methodology
    -----------
    - **GLM models**: Analytic solver using the Schur complement
      (deletion formula) + Newton-Raphson correction. O(1) per
      feature — no refit required.
    - **GBM models**: Resampling solver — refits the model on
      (p-1) features. O(p) refits — computationally expensive
      but statistically exact.

    Significance
    ------------
    The statistical significance of each feature's Gini drop is
    assessed using the **DeLong test**, which correctly accounts
    for the correlation between full-model and ablated-model
    predictions on the same evaluation observations.

    Re-evaluation Protocol
    ----------------------
    After the initial `fit`, the internal solver instance is
    cached. Subsequent calls with a different evaluation set can
    reuse this instance (via `get_fa_instance()`) to obtain
    near-zero-cost re-evaluation (prediction only, no
    recompute/refit).

    Parameters
    ----------
    model : LogisticRegression | None, default=None
        Fitted model with standard `.fit()`/`.predict_proba()`
        API. If None, a default LogisticRegression is fitted
        internally.
    alpha : float, default=0.05
        Significance level. Features with p < alpha are flagged
        as "significant" (`SignificantΔ` column).
    random_state : int, default=42
        Seed for reproducibility.
    """

    def __init__(self, 
        model:          LogisticRegression = None,
        alpha:          float = 0.05,
        random_state:   int = 42,

        **kwargs
    ):
        super().__init__(model, random_state, kwargs)
        self.alpha = alpha
        self._fa = None
    
    def get_fa_instance(self) -> None | _FeatureAblation:
        """
        Retrieve the cached ablation solver for fast re-evaluation.

        After the initial `fit`, this solver holds all
        expensive computational state:

        - **GLM**: Reduced beta vectors (Schur complement +
          Newton steps) for each feature.
        - **GBM**: Fitted estimators (p models, one per ablated
          feature).

        Passing this instance to a subsequent `fit` call
        enables near-zero-cost re-evaluation on a different
        evaluation set (prediction only, no recompute).

        Returns
        -------
        FeatureAblation | None
            The cached solver instance, or None if `fit` has
            not been called yet.
        """
        return self._fa
        
    def _get_result(self,  
        X:pDF, y:pSe, train_setup: tuple[pDF, pSe] | _FeatureAblation
    ) -> pDF:
        eval_set = ([X,y],)

        if isinstance(train_setup, _FeatureAblation):
            self._fa = train_setup
            return self._fa.update(eval_set)
        
        self._fa = _FeatureAblation(self.model, self.alpha)
        return self._fa.evaluate(*train_setup, eval_set = eval_set)

    def _fit(self, 
        X:pDF, y:pSe, train_setup: tuple[pDF, pSe] | _FeatureAblation
    ) -> dict:
        """
        Compute ablation importance for all features.

        Parameters
        ----------
        X : pDF
            Evaluation feature matrix (validation/test sample).
        y : pSe
            Evaluation target (validation/test sample).
        train_setup : tuple[pDF, pSe] | FeatureAblation
            Two inputs are supported to minimize ablation result:
            - **Mode 1**: A tuple `(X_train, y_train)` - the
                training data used in the original model fit. Required
                for the initial assessment.
            - **Mode 2**: A `FeatureAblation` instance obtained
                from `get_fa_instance()`. Enables fast re-evaluation
                on a different eval set without recomputation.
        """
        res = self._get_result(X, y, train_setup)
        return dict(
            result = res.rename(
                columns = {LBL_FA.A: LBL.A, LBL_FA.IS: LBL.IS}
            )
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeatureAttributionSHAP(FeatureBase):
    """
    Feature Importance via SHAP (Shapley Additive exPlanations).

    Quantifies each feature's marginal contribution to the
    prediction relative to the model baseline. Uses the model's
    internal structure when available:

    - **Tree-based models** (LightGBM, XGBoost): Exploits the
      tree structure for exact Shapley values (100x faster than
      Monte Carlo sampling).
    - **Linear models** (Logistic Regression): Structured
      decomposition based on feature values.
    - **Generic models**: Fallback to generic kernel explainer.

    The result is the mean absolute SHAP value per feature —
    a model-specific measure of predictive contribution.

    Parameters
    ----------
    model : LogisticRegression | None, default=None
        Fitted model. If None, a default LogisticRegression is
        fitted internally.
    random_state : int, default=42
        Seed for reproducibility.
    verbose : bool, default=True
        Controls output display.
    """
    def __init__(self, 
        model: LogisticRegression = None, 
        random_state: int = 42, 
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(model, random_state, kwargs)
        self.verbose = verbose

    def _plot(self, X:pDF, shap_vals:NDArray) -> None:
        """Summary plot (Beeswarm is richer than bar)."""
        if not ( self.verbose or self._is_agg ):
            return
        fig = ModelView.shap_plot(X, shap_vals, self._fes, show=False)
        if self._is_agg:
            self._set_figure(fig)
            return
        display(fig)

    def _shap_vals(self, X:pDF) -> NDArray:
        """
        The shap's internal auto choice of proper Explainer works well.
         - respects directly implemented .shap_values()
         - tree-based with structure by the tree (100x faster)
         - logit-based with structure by X
        """
        if hasattr(self.model, 'shap_values'):
            return self.model.shap_values(X)
        
        try:    explainer:TreeExplainer   = Explainer(self.model)
        except: explainer:LinearExplainer = Explainer(self.model,X)
        return  explainer.shap_values(X)

    @staticmethod
    def _verify_binary(shap_vals:NDArray) -> None:
        if len(shap_vals.shape) != 2:
            raise TypeError('AIG SAFETY ► Binary Target variable is expected.')
    
    def _fit(self, X:pDF, _) -> dict:
        """
        Compute mean absolute SHAP values per feature.

        Parameters
        ----------
        X : pDF
            Feature matrix for SHAP computation.
        _ : pSe
            Target (unused — SHAP is model-specific, not
            target-dependent).
        """
        shap_vals = self._shap_vals(X)
        self._verify_binary(shap_vals)
        means = np.abs(shap_vals).mean(axis=0)  # mean absolute SHAP value per feature
        
        self._plot(X, shap_vals)

        return dict(
            result = {
                LBL.F:  self._fes,
                LBL.S:  means,
            },
            sort_by = LBL.S,
            ascending = False
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeaturePermutation(FeatureBase):
    """
    Feature Importance via Permutation (Information Destruction).

    Measures the performance degradation when each feature's
    values are randomly shuffled, destroying its relationship
    with the target. Model-agnostic — works with any estimator.

    Methodology
    -----------
    For each feature:
    1. Shuffle the feature's values (break the signal).
    2. Measure the resulting AUC degradation (ΔGini).
    3. Repeat `n_repeats` times for a stable estimate.
    4. Statistical test (t-test) determines whether the effect
       is significantly different from zero.

    The result is the average ΔGini per feature, with a
    significance flag. Larger ΔGini = more important feature.

    Parameters
    ----------
    model : LogisticRegression | None, default=None
        Fitted model. If None, a default LogisticRegression is
        fitted internally.
    alpha : float, default=0.05
        Significance level for the t-test. Features with
        p < alpha are flagged as "significant."
    n_repeats : int, default=5
        Number of permutation rounds per feature. Higher =
        more stable estimates at the cost of computation.
    random_state : int, default=42
        Seed for reproducibility (permutation shuffling).
    verbose : bool, default=True
        Controls output display.
    """

    def __init__(self, 
        model: LogisticRegression = None, 
        alpha : int = 0.05,
        n_repeats: int = 5,
        random_state: int = 42, 
        **kwargs
    ):
        super().__init__(model, random_state, kwargs)
        self.alpha = alpha
        self.n_repeats = n_repeats

    def _n_jobs(self) -> int:
        """
        NOTE This has a potential to kill silently os threading.
        e.g. For trees setting it to -1 → silent crawling death for OS.
        """
        if ModelUtils.is_tree(self.model):       return  1 # explicit note ▲
        if not ModelUtils.is_n_jobs(self.model): return -1 # safe as no spawn of n_jobs possible
        return 1
    
    def _significance(self, means:NDArray, stds:NDArray, n_samples:int) -> NDArray:
        sig = TestSampleDiff(self.alpha).t_test(means=means, n_samples=n_samples, stds=stds)
        return sig & (means < 0)
    
    def _fit(self, X:pDF, y:pSe) -> dict:
        """
        Compute permutation importance for all features.

        Parameters
        ----------
        X : pDF
            Feature matrix (evaluation sample).
        y : pSe
            Target vector (evaluation sample).
        """
        perm = permutation_importance(
            self.model, X, y, 
            n_repeats    = self.n_repeats, 
            random_state = self.random, 
            scoring      = 'roc_auc',
            n_jobs       = 1#self._n_jobs()
        )
        means = -2*perm.importances_mean # avg δGini 
        stds  =  2*perm.importances_std  # std δGini 
        sig   = self._significance(means, stds, self.n_repeats)
        lbl_s = LBL.STD(LBL.P)

        return dict(
            result = {
                LBL.F:  self._fes,
                LBL.P:  means,  
                lbl_s:  stds,   
                LBL.IS: sig
            },
            sort_by = LBL.P,
            ascending = True
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FEATURES ASSESSMENTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class FeatureImportanceResult:
    """
    Feature importance consensus result.

    The aggregated output of the multi-method importance
    assessment. Contains per-method rankings, pairwise
    agreement metrics, and the overall consensus.

    Attributes
    ----------
    ranks : pDF
        Per-feature importance values from each method, plus
        the consensus rank (median) and consistency score.
    pairwise_ranks : pDF
        Pairwise Kendall τ correlation matrix between methods.
        Low values indicate features where the methods
        disagree (collinearity, proxies, interactions).
    kendall_w : float
        Multi-rater agreement score (0–1). 1.0 = all methods
        rank features identically; 0.0 = complete
        disagreement.
    method_results : dict
        Raw per-method results (DataFrames keyed by method
        name: `'ablation'`, `'permutation'`, `'shap'`).
    """    
    ranks:          pDF 
    pairwise_ranks: pDF
    kendall_w:      float
    method_results: dict
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ImportanceFitter:
    """
    Multi-Method Feature Importance Execution Engine.

    Orchestrates the execution of all selected importance
    methods (ablation, permutation, SHAP) on the same
    evaluation sample, collects per-feature importance values,
    and produces a rank matrix for consensus aggregation.

    Supported Methods
    -----------------
    - **Ablation** (`'ablation'`): Residual ΔGini when a
      feature is removed. Requires training data for the
      initial computation; subsequent re-evaluations use the
      cached solver (near-zero cost).
    - **Permutation** (`'permutation'`): ΔGini when a
      feature is shuffled. Model-agnostic.
    - **SHAP** (`'shap'`): Marginal contribution relative to
      the model baseline. Model-specific.

    Parameters
    ----------
    model : LogisticRegression | None, default=None
        Fitted model to assess. If None, a default
        LogisticRegression is fitted internally.
    methods : list | None, default=None
        Methods to include. Default: all three
        (`['ablation', 'permutation', 'shap']`).
    params : dict | None, default=None
        Global parameters for all methods (`alpha`,
        `n_repeats`, `random_state`).
    shap_fig : bool, default=True
        Generate SHAP summary figure (when SHAP is among the
        methods).
    """

    implemented_methods = {
        MTD.A:  dict( m=FeatureAblation,        c=LBL.A, a=True  ),
        MTD.P:  dict( m=FeaturePermutation,     c=LBL.P, a=True  ), 
        MTD.S:  dict( m=FeatureAttributionSHAP, c=LBL.S, a=False )
    }               # m = method class, c = sorting col, a=ascending

    def __init__(self, 
        model:     LogisticRegression = None,  
        methods:   list|Literal['ablation', 'permutation', 'shap'] = None, 
        params:    dict  = None,
        shap_fig:  bool  = True,
    ):
        self.model      = model
        self.shap_fig   = shap_fig

        self._set_params(params)
        self._set_methods(methods)

        self.method_result = {}        
        self.method_figure = {}
        self._fa_instance  = None  

    def _set_params(self, params:dict) -> None:
        """Merge user parameters with defaults."""        
        default = dict(
            model        = self.model,
            alpha        = 0.05,
            n_repeats    = 5,
            random_state = 42,
            verbose      = False,
        )                   
        default[VIEW.AS] = self.shap_fig
        self.params = default | (params or {})
    
    def _set_methods(self, methods):
        """
        Resolve and validate the method list.

        If None, defaults to all implemented methods. Validates
        that all requested methods are supported.

        Raises
        ------
        ValueError
            If any requested method is not in the implemented
            methods registry.
        """        
        im = list(self.implemented_methods)
        if methods is None:
            self.methods = im
        else: 
            self.methods = List.ensure(methods)
        self.methods = list(self.methods)
        if not all(m in im for m in self.methods):
            raise ValueError(f"Unknown method(s) in: {self.methods} → implemented: {im}")

    def _get_method(self, method:str) -> tuple[FeatureBase, str, str]:
        """
        Resolve a method name to its implementation.

        Returns
        -------
        tuple
            (method_class, sort_column, ascending_flag)
        """        
        m_c = self.implemented_methods[method]
        return m_c['m'], m_c['c'], m_c['a']
    
    def _run_method(self, 
        method:str, 
        data: tuple[pDF, pSe] | tuple[pDF, pSe, tuple[pDF,pSe]]
    ) -> tuple[pDF,str,bool]:
        """
        Execute a single importance method and return its result.

        Instantiates the method class with the global params,
        calls `.fit(*data)`, and captures the result DataFrame
        and any figure.

        Parameters
        ----------
        method : str
            Method identifier (`'ablation'`, `'permutation'`,
            or `'shap'`).
        data : tuple
            Method-specific input data (eval set, and for
            ablation: train set or cached solver).

        Returns
        -------
        tuple
            (result_df, sort_column, ascending_flag, figure)
        """        
        m,c,a = self._get_method(method)
        mf: FeatureBase = m(**self.params).fit(*data)
        df  = mf.result
        fig = mf.fig
        self._set_ablation(mf)
        return df, c, a, fig
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_ablation(self, mf:FeatureBase) -> None:
        """
        Cache the ablation solver instance for fast re-evaluation.

        If the method is ablation, stores its internal solver
        so subsequent calls can reuse it (prediction-only, no
        recompute).
        """        
        if not isinstance(mf, FeatureAblation):
            return
        self._fa_instance = mf.get_fa_instance()

    def _get_data(self,
        method: str, X:pDF, y:pSe, ablation_train: None | tuple[pDF, pSe] | _FeatureAblation
    ) -> tuple[pDF, pSe]:
        """
        Prepare the method-specific input data.

        For ablation, ensures the training pair is available
        (or the cached solver instance for fast re-evaluation).
        For other methods, returns the eval set directly.

        Parameters
        ----------
        method : str
            Method identifier.
        X : pDF
            Evaluation feature matrix.
        y : pSe
            Evaluation target.
        ablation_train : None | tuple[pDF, pSe] | _FeatureAblation
            Training data or cached solver (ablation only).

        Returns
        -------
        tuple
            Method-specific data tuple.
        """
        if method != MTD.A:
            return X, y
        
        if ablation_train is None or len(ablation_train) != 2:
            raise ValueError(
                'AIG SAFETY ► When ablation method is selected, training pair (X,y) '
                'must be provided.'
            )

        # provide fast route to ablt result
        if self._fa_instance is not None:
            ablation_train = self._fa_instance

        return X, y, ablation_train # eval_set X,y + train (X,y) | fitted FA    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def process(self, 
        X:pDF, y:pSe, ablation_train: None | tuple[pDF, pSe] | _FeatureAblation
    ) -> pDF:
        """
        Execute all selected methods and collect per-feature ranks.

        Iterates over the configured methods, runs each on the
        provided data, and assembles the result into a rank
        matrix (one column per method, rows are features).

        Parameters
        ----------
        X : pDF
            Evaluation feature matrix.
        y : pSe
            Evaluation target.
        ablation_train : None | tuple[pDF, pSe] | _FeatureAblation
            Training data or cached solver for ablation.
            Required when ablation is among the methods.

        Returns
        -------
        pDF
            Per-feature rank matrix (one column per method,
            values are ordinal ranks: 1 = most important).
        """
        ranks = pDF(index=X.columns)
        for method in self.methods:
            d = self._get_data(method, X, y, ablation_train)
            df, col, order, fig = self._run_method(method, d)
            self.method_result[method] = df
            self.method_figure[method] = fig
            df = df.set_index(LBL.F)
            ranks[method] = df[col].rank(ascending=order, method='average')
        return ranks
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ImportanceRanking:
    """
    Multi-Method Rank Aggregation & Consensus Scoring.

    Provides the post-processing pipeline for feature importance
    rank matrices: top-k subsetting, consensus computation,
    pairwise agreement metrics, and global multi-rater agreement.

    The processing pipeline:
    1. **Subset**: Restrict to the top-k features (optional).
    2. **Consistency**: Compute the consensus rank (median) and
       consistency score (0–1) per feature.
    3. **Kendall τ**: Pairwise rank correlation between methods.
    4. **Kendall W**: Global multi-rater agreement score.

    Parameters
    ----------
    methods : list
        The list of method names `present in the rank matrix`
        (e.g., `['ablation', 'permutation', 'shap']`).
        Required for the consistency computation.
    top_k : int | None, default=None
        If set, restrict the output to the top-k features
        (union across all methods). If None, all features are
        retained.
    """

    def __init__(self,
        methods:   list|Literal['ablation', 'permutation', 'shap'],
        top_k:     int = None,                  
    ):
        self.methods = methods
        self.top_k   = top_k

    def subset(self, ranks:pDF) -> pDF:
        """
        Restrict to the top-k features (by consensus rank).

        Parameters
        ----------
        ranks : pDF
            Full rank matrix.

        Returns
        -------
        pDF
            Subset containing only the top-k features (union
            across methods).
        """
        if self.top_k is None:
           return ranks
        top_features = set()
        for col in ranks.columns:
            top_features.update(ranks[col].nsmallest(self.top_k).index)
        return ranks.loc[top_features]
    
    def consistency(self, ranks:pDF) -> pDF:
        """
        Compute consensus rank and consistency score.

        The consensus is the median rank across methods. The
        consistency score measures how tightly the methods
        agree: 1.0 = perfect agreement, 0.0 = maximum
        disagreement. Normalized to (0, 1) scale.

        Parameters
        ----------
        ranks : pDF
            Per-method rank matrix.

        Returns
        -------
        pDF
            Rank matrix with `Consensus` and `Consistency`
            columns added, sorted by consensus.
        """
        ranks[LBL.C] = ranks.median(axis=1)
        scale_to_0_1 = (len(ranks) - 1)**2
        mse_per_fe   = (ranks[self.methods].sub(ranks[LBL.C], axis=0)**2).mean(axis=1)
        ranks[LBL.CS] = 1 - (mse_per_fe / scale_to_0_1).round(3)
        return ranks.sort_values([LBL.C, self.methods[0]])
    
    def kendalltau(self, ranks:pDF) -> pDF:
        """
        Compute pairwise Kendall τ between all methods.

        Low τ between two methods indicates features where they
        disagree on ordering. Common causes: collinearity,
        proxy features, interactions, or method-specific
        sensitivities.

        Parameters
        ----------
        ranks : pDF
            Per-method rank matrix.

        Returns
        -------
        pDF
            Symmetric pairwise τ matrix (methods × methods).
        """
        pairwise = pDF(index=self.methods, columns=self.methods, dtype=float)
        for m1 in self.methods:
            for m2 in self.methods:                
                pairwise.loc[m1, m2] = kendalltau(ranks[m1], ranks[m2])[0]
        return pairwise.round(3)
    
    def kendall_w(self, ranks:pDF) -> float:
        """
        Compute Kendall W — multi-rater agreement across all methods.

        Generalizes pairwise τ to the group level. Measures
        whether all methods agree on the overall ranking.
        Normalized to (0, 1): 1.0 = perfect agreement, 0.0 =
        no agreement beyond chance.

        Parameters
        ----------
        ranks : pDF
            Per-method rank matrix.

        Returns
        -------
        float
            Agreement score (0–1).
        """
        w = ranks[self.methods].var(axis=1).mean()
        w = 1 - w / (ranks[self.methods].max().max() - ranks[self.methods].min().min())
        return round(w,3)

    def process(self, ranks:pDF) -> tuple[pDF, pDF, float]:
        """
        Execute the full rank aggregation pipeline.

        Applies the sequence: subset → consistency → pairwise
        Kendall τ → Kendall W.

        Parameters
        ----------
        ranks : pDF
            Per-method rank matrix (one column per method,
            rows are features). Values are ordinal ranks
            (1 = most important).

        Returns
        -------
        tuple[pDF, pDF, float]
            - **ranks**: Processed rank matrix with consensus
              and consistency columns, sorted by consensus.
            - **pairwise**: Pairwise Kendall τ matrix.
            - **kendall_w**: Global agreement score (0–1).
        """
        ranks_    = self.subset(ranks)
        ranks_    = self.consistency(ranks_)
        pairwise  = self.kendalltau(ranks_)
        kendall_w = self.kendall_w(ranks_)

        return ranks_, pairwise, kendall_w
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class FeatureImportance(SamplingMixin):
    """
    Multi-Method Feature Importance Consensus Engine.

    Aggregates multiple feature importance assessments into a
    single consensus ranking with agreement metrics. Provides
    a holistic view of feature utility that is more robust than
    any single metric alone.

    Supported Methods
    -----------------
    - **Ablation**: Residual ΔGini when a feature is removed
      (model-specific: analytic for GLM, resampling for GBM).
    - **Permutation**: ΔGini when a feature is shuffled
      (model-agnostic).
    - **SHAP**: Marginal contribution to the prediction
      relative to the model baseline (model-specific).

    Consensus Metrics
    -----------------
    - **Consensus rank** (median across methods): The central
      tendency — the "most likely" importance ordering.
    - **Consistency score** (0–1): How much the methods agree
      on relative ordering.
    - **Pairwise Kendall τ**: Agreement between each pair of
      methods. Low values flag features where collinearity,
      proxies, or interactions cause method-specific
      disagreement.
    - **Kendall W** (0–1): Global agreement across all methods.

    Parameters
    ----------
    model : LogisticRegression | None, default=None
        Fitted model to assess. If None, a default
        LogisticRegression is fitted internally.
    top_k : int | None, default=None
        If set, limit the consensus table to the top-k features
        (by consensus rank).
    methods : list | None, default=None
        Methods to include. Default: all three
        (`['ablation', 'permutation', 'shap']`).
    params : dict | None, default=None
        Global parameters for all methods (`alpha`,
        `n_repeats`, `random_state`). See Notes for the
        full parameter reference.
    verbose : bool, default=True
        Enable output display.
    shap_fig : bool, default=True
        Generate SHAP summary figure (when SHAP is among the
        methods).
    dash_type : str, default='main'
        Display density: `'main'` (overview table),
        `'full'` (overview + per-method tables + pairwise
        correlation), `'fig'` (SHAP figure only).
    dash_title : dict, default=dict()
        Custom display titles.

    Notes
    -----
    **Parameter Reference** (`params` dict):

    - `alpha` (0.05): Significance threshold. For ablation:
      DeLong test p-value cutoff. For permutation: t-test
      p-value cutoff.
    - `n_repeats` (5): Permutation shuffle rounds per feature.
    - `random_state` (42): Seed for reproducibility.

    **Ablation Re-evaluation**: After the initial `fit`, the
    internal ablation solver is cached. Subsequent calls with a
    different evaluation set achieve near-zero-cost re-evaluation
    (prediction only, no recompute).

    **Subsampling**: For large datasets, a stratified subsample
    is used for the importance computation (sensitivity-aware
    sample size selection).

    **Interpretation of Disagreement**: Low pairwise Kendall τ
    between methods is *informative* — it flags features where
    the importance depends on the assessment lens. Common causes:
    collinearity (ablation vs. permutation), proxy features
    (SHAP vs. ablation), or interactions (permutation vs. SHAP).
    """

    def __init__(self, 
        model:     LogisticRegression = None,  
        top_k:     int = None, 
        methods:   list|Literal['ablation', 'permutation', 'shap'] = None, 
        params:    dict  = None,
        verbose:   bool  = True,
        shap_fig:  bool  = True,
        dash_type: Literal['main','full','fig'] = 'main',
        dash_title: dict | None = None
    ):
        self.fitter  = ImportanceFitter(model, methods, params, shap_fig)
        self.ranking = ImportanceRanking(self.fitter.methods, top_k)

        self.verbose    = verbose
        self.dash_type  = dash_type
        self.dash_title = dash_title

        self.dash = []
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    @property
    def result(self) -> FeatureImportanceResult:
        """
        The consensus assessment result.

        Returns
        -------
        FeatureImportanceResult
            Contains:
            - `ranks`: Per-feature importance + consensus +
              consistency.
            - `pairwise_ranks`: Kendall τ between methods.
            - `kendall_w`: Global agreement score.
            - `method_results`: Raw per-method DataFrames.
        """
        return self._result
    # ─────────────────────────────────────────────────────────────────────────────────────────────    

    def show_dash(self) -> None:
        """Display the assessment output once FeatureImportance was fitted."""
        self.dash.show() 
        
    def update_dash_title(self, title: dict) -> None:
        """
        Update the display title.

        Useful when multiple evaluation sets are provided in
        subsequent runs — the title differentiates them.
        """
        self.dash_title = self.dash_title | title

    def _create_dash(self, extras:pSe|pDF=None) -> None:
        """
        Construct the display output.

        Parameters
        ----------
        extras : pSe | pDF | None
            Additional per-feature columns to include in the
            overview table (e.g., stability score, WOE
            direction). Must have an index matching the
            features.
        """
        self.dash = FeatureImportanceDash(
            im_result   = self._result,
            methods     = self.fitter.methods,
            results     = self.fitter.method_result,
            figures     = self.fitter.method_figure,
            dash_type   = self.dash_type,
            dash_title  = self.dash_title
        )
        self.dash.create(extras)
        if not self.verbose:
            return
        self.show_dash() 
    # ─────────────────────────────────────────────────────────────────────────────────────────────    

    @staticmethod
    def _features_order(ranks:pDF, features_order:list) -> pDF:
        """
        Order features by custom order.

        If `features_order` is provided, reorders the table
        to match. Otherwise, sorts by consensus rank (most
        important first).
        """
        if features_order is None:
            return ranks.sort_values(by=LBL.C)
        return ranks.loc[features_order]

    def _get_ablation_subsample(self, train_set: tuple[pDF, pSe] | None
    ) -> tuple[pDF, pSe] | None:
        """
        Subsample ablation training set according to sensitivity subsampling setup.
        """
        if train_set is None:
            return None        
        return self._get_subsample(*train_set, self.fitter.params)
    
    def fit(self, 
        X: pDF, 
        y: pSe, 
        ablation_train: tuple[pDF, pSe] = None,
        features_order: list = None, 
        dash_extras: pSe|pDF = None 
    ):
        """
        Compute feature importance consensus across all methods.

        Executes the full multi-method assessment pipeline:
        subsample → run methods → rank → consensus → agreement
        metrics → display.

        Parameters
        ----------
        X : pDF
            Evaluation feature matrix (validation/test sample).
        y : pSe
            Evaluation target (validation/test sample).
        ablation_train : tuple[pDF, pSe] | None, default=None
            Training data pair `(X_train, y_train)` used in
            the original model fit. Required when ablation is
            among the selected methods. On subsequent calls,
            the cached ablation solver enables near-zero-cost
            re-evaluation.
        features_order : list | None, default=None
            Custom feature ordering for the final table. If
            None, sorted by consensus rank (most important
            first).
        dash_extras : pSe | pDF | None, default=None
            Additional per-feature columns for the overview
            table. Must have an index matching the features.

        Returns
        -------
        self
            Fitted instance with `.result` (consensus package)
            set.
        """
        X_,y_ = self._get_subsample(X, y, self.fitter.params)
        abl_t = self._get_ablation_subsample(ablation_train)
        ranks = self.fitter.process(X_, y_, abl_t)        
        ranks, pairwise, kendall_w = self.ranking.process(ranks)
        
        self._result = FeatureImportanceResult(
            ranks           = self._features_order(ranks, features_order),
            pairwise_ranks  = pairwise,
            kendall_w       = kendall_w,
            method_results  = self.fitter.method_result
        )
        self._create_dash(dash_extras)
        return self
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeatureImportanceDash:
    """
    Feature Importance Display Formatter.

    Internal display layer that constructs the formatted
    output (tables, figures) from the assessment results.
    """

    def __init__(self, 
        im_result:  FeatureImportanceResult,         
        methods:    list, 
        results:    dict,
        figures:    dict,
        dash_type:  Literal['main','full','fig'] = 'main',
        dash_title: dict | None = None
    ):
        self.im_result  = im_result
        self.methods    = methods
        self.results    = results       
        self.figures    = figures

        self.dash_type  = dash_type
        self.dash_title = dash_title or dict(
            main  = 'Feature Effect by Method',
            shap  = 'Feature Shap Effect on Model Output',
            table = 'Feature Effect by Method',
        )

        self.dash = []  

    def _table(self, tables:list, titles:list, t:int=1) -> None:
        self.dash.append( 
            TableStyle.combine_pdf_to_html(
            tables, titles, 
            tables_per_block = t,
            h_cols_per_row   = 1, 
            h_gap_horizontal = 20,
            show = False
        ))
    
    def _title(self, t:str) -> None:
        t_ = t
        if t in ['main', 'shap', 'table']:
            t_:str = self.dash_title.get(t)
        if not t_:
            self.dash.append('<br>')
            return    
        self.dash.append(
            Format.title(t_.upper(), style=Format.StyleStats(), show=False)
        )

    @staticmethod
    def _add_extras(effect:pDF, extras:pSe|pDF = None) -> list:
        if extras is None:
            return []
        if not isinstance(extras, (pDF, pSe)):
            print('AIG SAFETY ► Provided extras must be pDF|pSe type ► skipping.')
            return []
        fes = [x for x in effect.index if x not in extras.index]
        if fes:
            print(
            'AIG SAFETY ► Provided extras for Importance overview does not '
            f'align ► skipping join\nFollowing index values are missing:\n{fes}'
            )
            return []
        
        if isinstance(extras, pDF):
            c = extras.columns.to_list()
        else: 
            c = extras.name or 'Extras'
        effect[c] = extras.loc[effect.index]
        return List.ensure(c)

    def _data(self, extras:pSe|pDF = None) -> pDF:        
        result = self.im_result.method_results
        effect = pDF(index=self.im_result.ranks.index)
        sig    = pSe(False, index = effect.index)

        for d in result.values():
            d = d.set_index(LBL.F)
            effect[d.columns] = d
            if LBL.IS in d.columns:
                sig |= d[LBL.IS]

        ext = self._add_extras(effect, extras)
        
        if any([x in result for x in (MTD.A, MTD.P)]):
            effect[LBL.IS] = sig

        keep = [LBL.IS, LBL.A, LBL.P, LBL.S] + ext

        effect = (
            effect[[x for x in keep if x in effect.columns]]
            .reset_index(names=LBL.F)
        )
        effect[LBL.F] = Pd.str_max_len(effect[LBL.F], 35)
        sig_highlight = {'subset':LBL.IS, 'color':['#B50064', '#1DA889']} 

        return Pd.style(effect, 
            format = {'precision':3},
            color  = sig_highlight if LBL.IS in effect.columns else None,
            # properties  = {'subset':LBL.F, 'max-width':'270px', 'text-align':'left'}
        )

    def _main(self, extras: pSe|pDF = None) -> None:
        if self.dash_type == 'fig':
            return
        self._title('main')
        data  = self._data(extras)    
        form  = [{"targets": "_all", "createdCell": ItablesStyle.bool_color(LBL.IS)}]
        css   = "thead th { font-size: 10.5px !important; }"
        html  = itables_html(data, columnDefs=form, update_css=css)
        self.dash.append(html)

    def _full(self) -> None:
        if self.dash_type != 'full':
            return
        if len(self.methods) > 1:
            schema = Dataclass.schema(FeatureImportanceResult)
            self._title('Feature Ranking Consistency')
            self._table(
                [self.im_result[r] for r in (schema.ranks, schema.pairwise_ranks)], 
                ['Feature Ranking by Method', 'Feature Order Consistency by Kendall τ'], 
                2
            )
        self._title('table')
        self._table(
            list( self.results.values() ), 
            self.methods
        )            

    def _fig(self) -> None:
        if self.figures.get(MTD.S) is not None:  
            self._title('shap')
            self.dash.append(self.figures[MTD.S])

    def create(self, extras: pSe|pDF = None) -> None:
        """
        Construct the full display output.

        Parameters
        ----------
        extras : pSe | pDF | None
            Additional per-feature columns for the overview.
        """
        self._main(extras)
        self._full()
        self._fig()        

    def show(self) -> None:
        """Render and display the formatted output."""
        Display.construct(self.dash)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬