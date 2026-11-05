"""
Binary GLM Training with Two-Stage Feature Selection.

This module implements the GLM (Generalized Linear Model) training stage
of the pipeline. It takes WOE-encoded features and a binary target,
performs automated feature selection via a two-stage process, and fits
a final Logistic Regression model for inference.

Two-Stage Feature Selection
---------------------------
**Stage 1 — Regularization Search (`LinearSelectorCV`)**:
Performs cross-validated regularization (L1/L2/ElasticNet/PLS) to
identify a candidate set of features that are statistically significant
against the target. The regularization path is explored, and features
that survive the sparsity threshold are retained.

**Stage 2 — Shadow Pruning (`FeaturesShadowPruner`)**:
Tests the stability of each candidate feature by comparing its
contribution against "shadow" features (random permutations of the
target). Features whose importance is not significantly above the
shadow baseline are removed. This removes features that appear
significant due to overfitting or spurious correlation, while
preserving genuinely informative features.

**Final Model**:
A standard Logistic Regression is fit on the surviving features using
a stratified train/validation split. The validation split is used
for diagnostic metrics (Gini, lift, KS) — ensuring fair comparison
with tree-based models that use the same split protocol.

Why WOE + GLM?
--------------
- **Interpretability**: Coefficients directly map to business risk
  (sign = direction, magnitude = strength).
- **Regulatory**: Linear models are auditable and explainable by
  construction (no black-box interactions).
- **Stability**: WOE features are approximately linear, so the
  linear model assumption is well-satisfied.
- **Minimal**: The two-stage selection produces a small, stable
  feature set suitable for scorecard deployment.

Solvers
-------
- **Sparse (stepwise)**: Sequential addition/removal by significance.
  Most interpretable, smallest model.
- **Lasso (L1)**: Simultaneous coefficient shrinkage and selection.
  Handles multicollinearity by picking one from a correlated group.
- **ElasticNet (L1+L2)**: Combines Lasso selection with Ridge
  stability. Handles correlated feature groups more robustly.
- **Logistic Regression (unpenalized)**: Standard MLE, no
  regularization. Use when the feature set is already curated.
- **PLS (Partial Least Squares)**: Maximizes covariance between
  features and target. Best for high-dimensional, correlated inputs.
"""

from ....deps.data              import  np, pd, pDF, pSe
from ....deps.types             import  NDArray

from ....core.base              import  (
                                        FitterTrainBase, 
                                        FitterModelBase
                                        )
from ....core.factory           import  Dataclass
from ....core.foundation        import  AigParam
from ....core.accessor          import  Spinner

from ....ml.models              import  Logit
from ....ml.sampling            import  Sampling
from ....ml.metrics             import  (
                                        DiagnosticsResult,
                                        SeparationDiagnostics
                                        )
from ....display.verbosity      import  Verbosity

from ...features.selection      import  (
                                        LS, 
                                        LinearSelectorCV
                                        )
from ...features.pruning        import  (
                                        GiniPath,
                                        FeaturesShadowPruner
                                        )


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONFIG
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
GLM = LinearSelectorCV.implemented_estimators

@Dataclass.base
class EstimatorCfg:
    """
    Estimator backend configuration.

    Attributes
    ----------
    solver : str
        Feature selection solver: 'S' (sparse/stepwise), 'L' (Lasso),
        'E' (ElasticNet), 'R' (unpenalized Logit), 'P' (PLS).
    config : dict
        Additional solver-specific hyperparameters (e.g., l1_ratio
        for ElasticNet, n_components for PLS).
    """ 
    solver:  str = LS.S
    config: dict = Dataclass.field({})
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LEARN → GLM
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class FitParamsBase:
    """
    Feature selection and model fitting thresholds.

    Attributes
    ----------
    alpha : float, default=0.05
        Statistical significance level for feature selection.
        Controls the sparsity threshold in the regularization path —
        features that do not contribute significantly (p > alpha)
        to the model are excluded.
    eps : float, default=1e-3
        Minimum coefficient magnitude for a feature to be considered
        "selected." Features with coefficients below this threshold
        are treated as zero (inactive). Controls the number of
        regularization path points explored.
    tol : float, default=1e-4
        Convergence tolerance for the optimization solver.
        Lower values produce more precise coefficients at the cost
        of longer computation time.
    shadow_quantile : float, default=0.9
        Quantile of the shadow (permutation) importance distribution
        used as the pruning threshold. A feature must exceed the
        90th percentile of shadow importances to survive pruning.
        Higher = stricter pruning (fewer features retained).
    cv : int, default=3
        Number of cross-validation folds for the regularization
        search and shadow pruning. Higher = more stable estimates
        at the cost of computation time.
    """     
    alpha:           float = 0.05
    eps:             float = 1e-3
    tol:             float = 1e-4
    shadow_quantile: float = 0.9
    cv:                int = 3

@Dataclass.base(update_specs=dict(slots=False))
class FitParamsCfg:
    """
    Pipeline configuration.

    Attributes
    ----------
    target : str
        Target column name (binary: 0/1).
    val_size : float, default=0.33
        Fraction of data reserved for validation (used for
        diagnostic metrics only — feature selection uses CV,
        not this split).
    n_permutations : int, default=3
        Number of shadow (permutation) samples per feature in
        the pruning stage. Higher = more stable shadow baseline
        at the cost of computation time.
    random_state : int, default=42
        Seed for reproducibility (CV splits, sampling, permutations).
    verbose : bool | Verbosity, default=True
        Enable progress display and diagnostics.
    """
    target:         str
    val_size:       float = 0.33
    n_permutations: int = 3
    random_state:   int = 42
    verbose:        bool|Verbosity = True

@Dataclass.base
class FitParams(FitParamsBase, FitParamsCfg):
    """Fitter params input design."""

@Dataclass.base
class FitResult:
    """
    Feature selection result.

    Attributes
    ----------
    features_selected : list
        Features that survived both selection stages.
    features_dropped : list
        Features removed during selection (with implicit reason
        from the stage at which they were dropped).
    """
    features_selected: list
    features_dropped:  list

@Dataclass.base
class FitStats:
    """
    Diagnostic statistics for user-facing display.

    Attributes
    ----------
    selector_path : dict
        Regularization path from Stage 1 (coefficients at each
        alpha level, Gini at each point).
    pruner_path : dict
        Shadow pruning path from Stage 2 (importance rankings,
        shadow baseline, drop decisions).
    gini_path : GiniPath
        Combined Gini trajectory across both stages.
    diagnostics : DiagnosticsResult
        Validation metrics (Gini, AUC, KS, lift table) on the
        held-out validation set.
    """
    selector_path: dict
    pruner_path:   dict
    gini_path:     GiniPath
    diagnostics:   DiagnosticsResult = None

@Dataclass.base
class FitDashLink:
    """
    Reference bundle for the user-facing display layer.

    Provides the display component with direct access to the
    fitted model, pruning diagnostics, and data references
    without requiring separate lookups.

    Attributes
    ----------
    pruner : FeaturesShadowPruner
        Fitted pruner (for importance plots and shadow baseline).
    model : Logit
        Final fitted Logistic Regression.
    target : str
        Target column name.
    features : list
        Selected feature names.
    total_fes : int
        Total features considered (before selection).
    val_idx : pd.Index
        Validation set row indices (for metric display).
    """
    pruner:    FeaturesShadowPruner
    model:     Logit
    target:    str
    features:  list
    total_fes: int
    val_idx:   pd.Index
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class FitWorker:
    """
    Two-Stage Feature Selection Pipeline — Worker.

    Orchestrates the sequential execution of:

    1. **Stage 1 — Regularization Search** (`LinearSelectorCV`):
       Cross-validated regularization path exploration. Identifies
       candidate features that are statistically significant. The
       sparsity of the solution (controlled by `alpha` and
       `eps`) determines how many features survive.

    2. **Stage 2 — Shadow Pruning** (`FeaturesShadowPruner`):
       Stability testing via target permutation. Each candidate
       feature's importance is compared against a shadow baseline
       (importance when the target is randomly shuffled). Features
       that do not significantly exceed the shadow baseline are
       removed. This eliminates spurious correlations and
       overfit features.

    The result is a refined, stable feature subset with full
    optimization path diagnostics.

    Parameters
    ----------
    params : FitParams
        Selection thresholds and pipeline configuration.
    config : EstimatorCfg
        Solver backend configuration.
    """
    selector = None
    pruner   = None
    result   = None

    def __init__(self, params:FitParams, config:EstimatorCfg = None):
        self.prm     = params
        self.config  = config or EstimatorCfg()
        self.target  = params.target
        self.verbose = params.verbose        
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def _get_n_alphas(self) -> int:
        """
        Determine the number of regularization path points.

        Scales with the `eps` threshold: smaller eps means a finer
        grid is needed to resolve small coefficients.

        Returns
        -------
        int
            Number of alpha points to explore (10–500).
        """
        multiplier = np.log10(1 / self.prm.eps) / 3.0   # 1e-3 is 3 orders of magnitude
        return int(100 * np.clip(multiplier, 1, 5))

    def _get_setup(self) -> dict:
        """
        Assemble the regularization solver constructor arguments.

        Returns
        -------
        dict
            Merged configuration: solver type, alpha, eps, tol,
            path points, random state, and user overrides.
        """        
        default = dict(
            alphas       = self._get_n_alphas(),
            random_state = self.prm.random_state,
            verbose      = False
        )
        prm = dict(
            solver = self.config.solver,
            alpha  = self.prm.alpha,
            eps    = self.prm.eps,
            tol    = self.prm.tol,
        )
        return default | self.config.config | prm
        
    def fit_selector(self, data:pDF, fes:list) -> None:
        """
        Execute Stage 1: Regularization-based feature selection.

        Fits a cross-validated regularized model on the candidate
        features and identifies those with non-zero coefficients
        at the selected sparsity level.

        Parameters
        ----------
        data : pDF
            Dataset containing features and target.
        fes : list
            Candidate feature names.
        """      
        self.selector = LinearSelectorCV(**self._get_setup())
        self.selector.fit(data[fes], data[self.target])

    def _get_selector_result(self) -> list:
        """
        Retrieve the feature list selected by Stage 1.

        Returns
        -------
        list
            Feature names with non-zero coefficients.

        Raises
        ------
        ValueError
            If `fit_selector` has not been called yet.
        """        
        if self.selector is None:
            raise ValueError("Fit selector .fit_selector() prior calling results.")
        return self.selector.result
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def _get_n_shadow_samples(self) -> int:
        """
        Determine the number of shadow permutation samples.

        Scales with CV folds: more folds means fewer independent
        shadow samples are needed (each fold already provides
        variance reduction).

        Returns
        -------
        int
            Number of permutation samples (2–10).
        """
        x = max(2, int(20 / self.prm.cv))
        return np.clip( self.prm.n_permutations, 2, x)
                
    def fit_pruner(self, data:pDF) -> None:
        """
        Execute Stage 2: Shadow feature pruning.

        Tests the stability of Stage-1 features by comparing their
        model importance against a shadow baseline (importance when
        target is randomly permuted). Features not significantly
        above the baseline are removed.

        Parameters
        ----------
        data : pDF
            Full dataset (used for CV splits and shadow generation).
        """            
        fes = self._get_selector_result()
        self.pruner = FeaturesShadowPruner(
            model           = self.selector,
            n_permutations  = self._get_n_shadow_samples(),
            cv              = self.prm.cv,
            val_size        = self.prm.val_size,
            random_state    = self.prm.random_state,
            verbose         = False
        ).fit(
            data[fes], data[self.target], 
            shadow_quantile = self.prm.shadow_quantile
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def _get_result(self, fes:list) -> tuple[FitResult, FitStats]:
        """
        Package the selection results and diagnostics.

        Parameters
        ----------
        fes : list
            Original candidate features (for computing the dropped list).

        Returns
        -------
        tuple[FitResult, FitStats]
            Selected/dropped features + optimization paths.
        """        
        return (
            FitResult(
                features_selected = self.pruner.result,
                features_dropped  = [x for x in fes if x not in self.pruner.result],
            ),
            FitStats(
                selector_path = self.selector.optim_path,
                pruner_path   = self.pruner.optim_path,
                gini_path     = self.pruner.gini_path,            
            ))
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def pipeline(self, data:pDF, fes:list) -> None:     
        """
        Execute the full two-stage feature selection pipeline.

        Stage 1 (regularization) → Stage 2 (shadow pruning).

        Parameters
        ----------
        data : pDF
            Full dataset.
        fes : list
            Candidate feature names.
        """
        self.fit_selector(data, fes)
        self.fit_pruner(data)
        
    def process(self, data:pDF, fes:list) -> tuple[FitResult, FitStats, FeaturesShadowPruner]:
        """
        Entry point: execute the full selection pipeline.

        Parameters
        ----------
        data : pDF
            Full dataset.
        fes : list
            Candidate feature names.

        Returns
        -------
        tuple[FitResult, FitStats, FeaturesShadowPruner]
            Selection result, diagnostics, and the fitted pruner
            (for importance ordering).
        """
        self.pipeline(data, fes)
        result, stats = self._get_result(fes)
        return result, stats, self.pruner
# ─────────────────────────────────────────────────────────────────────────────────────────────────
   
class Fitter(FitterTrainBase):
    """
    GLM Training — execution backend.

    Orchestrates the full training lifecycle:

    1. **Feature Selection**: Two-stage pipeline (regularization +
       shadow pruning) to identify a stable, significant feature set.
    2. **Model Fitting**: Logistic Regression on the selected features
       (train split only).
    3. **Diagnostics**: Validation metrics (Gini, AUC, lift) on the
       held-out set for fair comparison with tree-based models.
    4. **Importance**: Feature ordering by contribution to the
       model (from the shadow pruner).

    Parameters
    ----------
    params : FitParams
        Selection thresholds and pipeline configuration.
    config : EstimatorCfg
        Solver backend configuration.
    """
    _model: Logit = None
    
    def __init__(self, params:FitParams, config:EstimatorCfg=None):
        self.prm = params 
        self.config = config

    def _train_valid_split(self, data:pDF) -> tuple[pDF, pDF]:
        """
        Stratified train/validation split for the final model fit.

        Feature selection uses cross-validation (not this split),
        so the validation set is truly held-out for the final
        logistic regression. This ensures diagnostic metrics are
        comparable with tree-based models that use the same
        train/valid protocol.

        Parameters
        ----------
        data : pDF
            Full dataset.

        Returns
        -------
        tuple[pDF, pDF]
            (train_data, validation_data)
        """
        return Sampling(
            sample_size  = self.prm.val_size, 
            stratify     = data[self.prm.target], 
            random_state = self.prm.random_state, 
            shuffle      = True
        ).train_valid_split(data)        
                
    def _fit(self, X:pDF, y:pSe) -> None:
        self._model = Logit().fit(X, y)  

    def _valid_diagnostics(self, d_val:pDF) -> DiagnosticsResult:
        """
        Compute validation metrics on the held-out set.

        Parameters
        ----------
        d_val : pDF
            Validation data.

        Returns
        -------
        DiagnosticsResult
            Gini, AUC, KS, and lift table on the validation set.
        """            
        t = d_val[self.prm.target]
        p = self.predict_proba(d_val)[:, 1]
        return SeparationDiagnostics(lift_bins=10).evaluate(t, p)
        
    def fit(self, data:pDF, fes:list):
        """
        Execute the full GLM training pipeline.

        Steps:
        1. Two-stage feature selection (regularization + shadow pruning).
        2. Stratified train/validation split.
        3. Fit Logistic Regression on selected features (train only).
        4. Compute validation diagnostics.
        5. Order features by importance.

        Parameters
        ----------
        data : pDF
            Training data (target + WOE features).
        fes : list
            WOE feature names to consider.

        Returns
        -------
        self
            Fitted instance with `features`, `stats`, and
            `ordered_features` set.
        """
        worker = FitWorker(self.prm, self.config)
        self.result, stats, pruner = worker.process(data, fes)
        
        d_train, d_val = self._train_valid_split(data)

        self.features = self.result.features_selected
        self._fit(d_train[self.features], d_train[self.prm.target])

        self.stats = FitStats(
            **(dict(stats) | dict(diagnostics = self._valid_diagnostics(d_val)))
        )
        self.ordered_features = pruner.get_ordered_importance_fes(self.features)

        self.dash = FitDashLink(
            pruner    = pruner,
            model     = self._model,
            target    = self.prm.target,
            features  = self.features,
            total_fes = len(fes),
            val_idx   = d_val.index
        )
        
        return self  
        
    def predict_proba(self, data:pDF) -> NDArray:
        """
        Predict event probabilities using the fitted GLM.

        Parameters
        ----------
        data : pDF
            Input data containing the selected features.

        Returns
        -------
        NDArray
            2-D array of shape (n_samples, 2) with P(class=0)
            and P(class=1).
        """
        return self.model.predict_proba(data[self.features])
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LEARN → GLM
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class BinaryGLM(FitterModelBase[Fitter]):
    """
    Binary GLM — Base Class for Logistic Regression with Feature Selection.

    Trains a Logistic Regression model on WOE-encoded features with
    automated two-stage feature selection (regularization + shadow
    pruning). The final model is a standard, interpretable linear
    model suitable for scorecard deployment.

    Parameters
    ----------
    target : str
        Target column name (binary: 0/1).
    alpha : float, default=0.05
        Statistical significance level for feature selection.
    eps : float, default=1e-3
        Minimum coefficient threshold for feature inclusion.
    tol : float, default=1e-4
        Solver convergence tolerance.
    shadow_quantile : float, default=0.9
        Shadow pruning threshold (90th percentile).
    cv : int, default=3
        Cross-validation folds.
    val_size : float, default=0.33
        Validation fraction (for diagnostics only).
    n_permutations : int, default=3
        Shadow permutation samples per feature.
    random_state : int, default=42
        Seed for reproducibility.
    verbose : bool | Verbosity, default=True
        Enable progress and diagnostics.

    Notes
    -----
    **Interpretability**: The final model is a standard logistic
    regression. Coefficients directly represent the log-odds
    contribution of each WOE feature — ideal for regulatory
    environments requiring explainability.

    **Stability**: The two-stage selection (regularization + shadow
    pruning) produces a small, stable feature set that is robust to
    sample variation. This is critical for production scorecards
    where feature availability may fluctuate.

    **WOE Alignment**: WOE features are approximately linear in
    log-odds space, so the linear model assumption is well-satisfied.

    Examples
    --------
    >>> from aidflow.fitters import SparseBinaryGLM
    >>> model = SparseBinaryGLM(target='is_churn', alpha=0.05)
    >>> model.fit(data_woe)
    >>> probs = model.predict_proba(data_new) 
    >>> model.get_ordered_fes()  # ['tenure_WOE', 'revenue_WOE', ...]
    """
    _solver: str = None

    def __init__(self, *args, params: dict|FitParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(FitParams, params, args, kwargs)
        self._config = self.get_estimator_config(self._params, kwargs)
    
    def _get_fes(self, data:pDF, fes:list|None) -> list:
        if fes is not None:
            return fes
        return [x for x in data.columns if x!=self._params.target]
        
    def fit(self, data:pDF, features:list|None = None):        
        """
        Fit the GLM: feature selection + logistic regression.

        Parameters
        ----------
        data : pDF
            Training data (target + WOE features).
        features : list | None, default=None
            Specific WOE features to consider. If None, all
            non-target columns are used.

        Returns
        -------
        self
            Fitted instance with `predict_proba` and
            `get_ordered_fes` available.
        """
        fes = self._get_fes(data, features)
        cfg = EstimatorCfg(solver = self._solver, config = self._config)
        self._fitter = Fitter(self._params, cfg)
        
        title = f'GLM {self._solver.upper()} training in progress'
        with Spinner(title, show_timer=True).progress(self._params.verbose):
            self._fitter.fit(data, fes)
            
        return self
    
    def predict_proba(self, data:pDF) -> NDArray:
        """
        Predict event probabilities.

        Parameters
        ----------
        data : pDF
            Input data containing the selected WOE features.

        Returns
        -------
        NDArray
            2-D array of shape (n_samples, 2): [P(class=0), P(class=1)].
        """
        return self.fitter.predict_proba(data)
    
    def get_ordered_fes(self) -> list:
        """
        Return the selected features in descending importance order.

        Importance is determined by the shadow pruning stage
        (contribution relative to random noise).

        Returns
        -------
        list
            Feature names, most important first.
        """
        return self.fitter.ordered_features
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SparseBinaryGLM(BinaryGLM):
    """
    Binary GLM — Sparse (Stepwise) Feature Selection.

    Uses sequential stepwise selection (forward addition / backward
    removal) based on statistical significance. Produces the
    smallest possible model that maintains predictive power.

    Best For
    --------
    - Regulatory environments requiring maximum interpretability.
    - Scorecards with strict feature count limits.
    - Scenarios where each feature must be individually justified.

    Trade-off: May miss features that are only significant in
    combination.

    Parameters
    ----------
    target : str
        Target column name.
    alpha : float, default=0.05
        Significance level for stepwise decisions.
    shadow_quantile : float, default=0.9
        Shadow pruning threshold.
    verbose : bool | Verbosity, default=True
        Enable progress display.

    Examples
    --------
    >>> model = SparseBinaryGLM(target='is_churn', alpha=0.01)
    >>> model.fit(data_woe)
    >>> model.get_ordered_fes()  # ['tenure_WOE', 'score_WOE']
    """
    _solver = LS.S
    _estimator = GLM[_solver]

class LassoBinaryGLM(BinaryGLM):
    """
    Binary GLM — Lasso (L1) Feature Selection.

    Uses L1 regularization to simultaneously shrink and select
    features. Features with the weakest signal are driven to
    exactly zero coefficient, effectively removing them.

    Best For
    --------
    - High-dimensional feature spaces with multicollinearity.
    - Scenarios where a compact, sparse model is desired.
    - Lasso picks ONE feature from a correlated group (vs.
      ElasticNet which keeps all of them).

    Trade-off: With highly correlated features, Lasso may
    arbitrarily select one and discard the others, potentially
    losing information.

    Parameters
    ----------
    target : str
        Target column name.
    alpha : float, default=0.05
        Significance level.
    eps : float, default=1e-3
        Minimum coefficient threshold.
    shadow_quantile : float, default=0.9
        Shadow pruning threshold.
    verbose : bool | Verbosity, default=True
        Enable progress display.

    Examples
    --------
    >>> model = LassoBinaryGLM(target='is_churn', alpha=0.05)
    >>> model.fit(data_woe)
    >>> model.get_ordered_fes()  # ['revenue_WOE', 'tenure_WOE']
    """
    _solver = LS.L
    _estimator = GLM[_solver]

class ElasticBinaryGLM(BinaryGLM):
    """
    Binary GLM — ElasticNet (L1 + L2) Feature Selection.

    Combines L1 (Lasso) selection with L2 (Ridge) stabilization.
    The L1 penalty shrinks weak features to zero; the L2 penalty
    stabilizes coefficients of correlated features, preventing
    the arbitrary selection behavior of pure Lasso.

    Best For
    --------
    - Feature groups with high correlation (e.g., multiple
      revenue-based WOE features).
    - Scenarios where you want ALL relevant features from a
      correlated group (not just one).
    - Generally the most robust default for production scorecards.

    Trade-off: Slightly more features retained than Lasso (the
      L2 component prevents full shrinkage to zero for
      correlated features). Shadow pruning compensates by
      removing the truly redundant ones.

    Parameters
    ----------
    target : str
        Target column name.
    alpha : float, default=0.05
        Significance level.
    eps : float, default=1e-3
        Minimum coefficient threshold.
    shadow_quantile : float, default=0.9
        Shadow pruning threshold.
    verbose : bool | Verbosity, default=True
        Enable progress display.

    Notes
    -----
    The L1/L2 mixture ratio (`l1_ratio`) can be passed via
    `config`:

    >>> model = ElasticBinaryGLM(target='churn', config={'l1_ratio': 0.5})

    Examples
    --------
    >>> model = ElasticBinaryGLM(target='is_churn')
    >>> model.fit(data_woe)
    >>> model.get_ordered_fes()
    """
    _solver = LS.E
    _estimator = GLM[_solver]

class LogRegBinaryGLM(BinaryGLM):
    """
    Binary GLM — Standard Logistic Regression (No Regularization in Selection).

    Uses unregularized Logistic Regression for the selection stage.
    All features are retained in the candidate set, and the shadow
    pruning stage is the sole mechanism for feature elimination.

    Best For
    --------
    - Small feature sets (≤20) where regularization may be
    unnecessarily restrictive.
    - Scenarios where the user wants maximum control via the
    shadow pruning threshold (`shadow_quantile`).
    - Baseline comparison against regularized variants.

    Parameters
    ----------
    target : str
        Binary target column name.
    shadow_quantile : float, default=0.9
        Shadow pruning quantile (primary selection mechanism here).
    cv : int, default=3
        Cross-validation folds (for shadow evaluation).
    verbose : bool | Verbosity, default=True
        Enable progress display.

    Examples
    --------
    >>> from aidflow.fitters import LogRegBinaryGLM
    >>> model = LogRegBinaryGLM(target='is_churn', shadow_quantile=0.95)
    >>> model.fit(data_woe)
    >>> model.predict_proba(new_data)
    """
    _solver = LS.R
    _estimator = GLM[_solver]


class PLSBinaryGLM(BinaryGLM):
    """
    Binary GLM — Partial Least Squares (PLS) Selection.

    Uses PLS to identify latent components that maximize covariance
    with the target, then selects features contributing most to
    those components. Handles high multicollinearity gracefully.

    Best For
    --------
    - Highly collinear or redundant feature sets.
    - Scenarios where features form natural "groups" or
    "directions" of variation.
    - When the signal is distributed across many correlated
    features rather than concentrated in a few.

    Parameters
    ----------
    target : str
        Binary target column name.
    alpha : float, default=0.05
        Significance level.
    shadow_quantile : float, default=0.9
        Shadow pruning quantile.
    cv : int, default=3
        Cross-validation folds.
    verbose : bool | Verbosity, default=True
        Enable progress display.

    Examples
    --------
    >>> from aidflow.fitters import PLSBinaryGLM
    >>> model = PLSBinaryGLM(target='is_churn')
    >>> model.fit(data_woe)
    >>> model.predict_proba(new_data) 
    """
    _solver = LS.P
    _estimator = GLM[_solver]
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬