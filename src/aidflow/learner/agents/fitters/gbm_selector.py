"""
GBM (Gradient Boosting Machine) Training with Automated Feature Selection.

This module implements the GBM training stage of the AIDflow pipeline,
responsible for fitting a tree-based ensemble model (LightGBM or XGBoost)
on WOE-encoded features with a two-stage automated feature selection pipeline.

Two-Stage Feature Selection
---------------------------
The feature selection pipeline operates in two sequential stages:

1. **Tree-Based Selection (TreeSelectorGB)**
   A gradient boosting model is trained with early stopping on a
   validation set. Features that contribute to the model's predictive
   power (as determined by the tree splits and validation metric)
   are identified as candidates. This stage captures non-linear
   signal and interactions that a linear model would miss.

2. **Shadow Pruning (FeaturesShadowPruner)**
   From the candidate set, features are compared against "shadow"
   features (random permutations). If a real feature's contribution
   is not significantly better than random noise, it is pruned.
   This eliminates features that appear useful due to overfitting
   or chance, reducing variance while preserving true signal.

The result is a minimal, stable feature set optimized for OOT
performance in a tree-based context.

Key Differences from GLM Training
---------------------------------
- **Final model**: GBM (LightGBM/XGBoost) — captures non-linear
  effects and interactions without explicit feature engineering.
- **Selection mechanism**: Tree-based (split importance + early
  stopping) rather than coefficient-based (regularization path).
- **Categorical support**: Native categorical feature handling
  (no one-hot required).
- **Early stopping**: Train/validation split is used during
  selection for early stopping (not just post-hoc diagnostics).
- **Overfitting risk**: Higher than GLM (trees can memorize noise).
  The two-stage selection + shadow pruning mitigates this.

Train/Validation Split
----------------------
Applied before selection (unlike GLM, where selection uses CV).
The split serves dual purpose:
- Early stopping during the tree-based selection phase.
- Final model training with early stopping.
- Validation diagnostics for fair comparison.
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

from ....ml.models              import  LGB, XGB
from ....ml.sampling            import  Sampling
from ....ml.metrics             import  (
                                        DiagnosticsResult,
                                        SeparationDiagnostics
                                        )
from ....display.verbosity      import  Verbosity

from ...features.selection      import  (
                                        TS, 
                                        TreeSelectorGB
                                        )
from ...features.pruning        import  (
                                        GiniPath,
                                        FeaturesShadowPruner
                                        )


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONFIG
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
GBM = TreeSelectorGB.implemented_estimators

@Dataclass.base
class EstimatorCfg:
    """
    GBM solver configuration.

    Attributes
    ----------
    solver : str, default='L' (LightGBM)
        Tree-based engine identifier. Options:
        - 'L' : LightGBM (leaf-wise growth, fast)
        - 'X' : XGBoost (depth-wise growth, robust)
    config : dict, default={}
        Additional solver-specific hyperparameters
        (e.g., `num_leaves`, `max_depth`, `colsample_bytree`).
    """
    solver:  str = TS.L
    config: dict = Dataclass.field({})
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LEARN → GBM
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class FitParamsBase:
    """
    GBM tree complexity and selection settings.

    Attributes
    ----------
    max_leaves : int, default=31
        Maximum number of leaves per tree. Controls tree complexity —
        higher values allow more complex splits (more signal, more
        overfitting risk). 31 is the LightGBM default.
    learning_rate : float, default=0.1
        Shrinkage factor. Each tree's contribution is multiplied by
        this factor. Lower values = more trees needed but less
        overfitting per tree.
    n_estimators : int, default=100
        Maximum number of boosting rounds. Actual number is determined
        by early stopping (whichever comes first).
    min_child_weight : float, default=0.1
        Minimum sum of instance weights (hessian) required in a leaf.
        Prevents overfitting on small leaf populations.
    shadow_quantile : float, default=0.9
        Quantile threshold for shadow feature comparison.
        A real feature is kept only if its performance exceeds this
        quantile of the shadow (permutation) distribution.
    cv : int, default=3
        Number of cross-validation folds for shadow evaluation.
    """  
    max_leaves:         int = 31
    learning_rate:    float = 0.1
    n_estimators:       int = 100
    min_child_weight: float = 0.1
    shadow_quantile:  float = 0.9
    cv:                 int = 3

@Dataclass.base(update_specs=dict(slots=False))
class FitParamsCfg:
    """
    Pipeline configuration.

    Attributes
    ----------
    target : str
        Binary target column name (0/1).
    val_size : float, default=0.33
        Fraction of data held out for validation (early stopping
        + diagnostics).
    n_permutations : int, default=3
        Number of shadow (permutation) samples per feature.
        More permutations = more reliable noise distribution.
    random_state : int, default=42
        Seed for reproducibility.
    verbose : bool | Verbosity, default=True
        Enable progress display.
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
        Features that passed both selection stages (final model features).
    features_dropped : list
        Features that were eliminated.
    """
    features_selected: list
    features_dropped:  list

@Dataclass.base
class FitStats:
    """
    Diagnostic paths for the feature selection process.

    Attributes
    ----------
    selector_path : dict
        Tree-based selection path: feature importances, split
        frequencies, and metrics at each boosting round.
    pruner_path : dict
        Shadow pruning path: per-feature importance scores,
        shadow comparisons, and pruning decisions.
    gini_path : GiniPath
        Gini/AUC evolution as features are added/removed.
    diagnostics : DiagnosticsResult
        Validation set metrics (AUC, Gini, lift table) from
        the final fitted model.
    """
    selector_path: dict
    pruner_path:   dict
    gini_path:     GiniPath
    diagnostics:   DiagnosticsResult = None

@Dataclass.base
class FitDashLink:
    """
    Diagnostic bundle for user-facing display (Dash/UI connection).

    Contains all objects needed to render the feature selection
    process, model importances, and validation metrics.

    Attributes
    ----------
    pruner : FeaturesShadowPruner
        The fitted pruner (for shadow comparison display).
    model : LGB | XGB
        The final fitted GBM model.
    target : str
        Target column name.
    features : list
        Selected feature names.
    total_fes : int
        Total number of input features (before selection).
    val_idx : pd.Index
        Validation set row indices (for metric computation).
    """
    pruner:    FeaturesShadowPruner
    model:     LGB | XGB
    target:    str
    features:  list
    total_fes: int
    val_idx:   pd.Index
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class FitWorker:
    """
    Two-Stage Feature Selection Worker (Tree-Based).

    Orchestrates the sequential execution of:

    1. **TreeSelectorGB** (Stage 1): Trains a gradient boosting model
       with early stopping to identify features that contribute to
       predictive power. Captures non-linear signal and interactions.

    2. **FeaturesShadowPruner** (Stage 2): Compares each candidate
       feature against random permutations (shadow features). Features
       whose contribution is not significantly better than noise are
       pruned. This eliminates overfitting artifacts from Stage 1.

    Unlike the GLM worker (which uses CV-based regularization search),
    the GBM worker uses a train/validation split with early stopping
    for both selection and final model fitting.

    Parameters
    ----------
    params : FitParams
        Tree complexity and selection thresholds.
    config : EstimatorCfg
        Engine choice (LightGBM/XGBoost) and hyperparameters.
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
    
    def _get_setup(self) -> dict:
        """
        Assemble the tree selector constructor arguments.

        Merges: engine choice, tree complexity parameters,
        and any solver-specific hyperparameters.

        Returns
        -------
        dict
            Complete parameter set for `TreeSelectorGB`.
        """        
        default = dict(
            random_state = self.prm.random_state,
            verbose      = False
        )
        prm = dict(
            solver           = self.config.solver,
            max_leaves       = self.prm.max_leaves,
            learning_rate    = self.prm.learning_rate,
            n_estimators     = self.prm.n_estimators,
            min_child_weight = self.prm.min_child_weight,
        )
        return default | self.config.config | prm    

    def _train_valid_split(self, data:pDF, num:list, cat:list = None
    ) -> tuple[ tuple[pDF,pSe], tuple[pDF,pSe] ]:
        """
        Stratified train/validation split for selection and fitting.

        Unlike GLM (where selection uses CV), GBM selection uses
        early stopping on a validation set. This split serves both
        the selection stage and the final model fit.

        Parameters
        ----------
        data : pDF
            Full dataset.
        num : list
            Numerical feature names.
        cat : list
            Categorical feature names.

        Returns
        -------
        tuple
            ((X_train, y_train), (X_valid, y_valid))
        """        
        return Sampling(
            sample_size  = self.prm.val_size, 
            stratify     = data[self.prm.target], 
            random_state = self.prm.random_state, 
            shuffle      = True
        ).train_valid_split(data[num + (cat or [])], data[self.target])   

    def _get_n_permutations(self) -> int:
        """
        Determine shadow permutation count for the selection stage.

        Scales with CV folds to balance computation cost against
        reliability of the noise distribution.

        Returns
        -------
        int
            Number of shadow samples (clipped to [2, 10]).
        """
        return np.clip(int(self.prm.n_permutations * self.prm.cv/1.5), 2, 10)
    
    def fit_selector(self, data:pDF, num:list, cat:list) -> None:
        """
        Stage 1: Tree-based feature selection with early stopping.

        Trains a GBM on the train set, monitors the validation metric,
        and identifies features that contribute to predictive power
        before overfitting begins.

        Parameters
        ----------
        data : pDF
            Full dataset (features + target).
        num : list
            Numerical feature names.
        cat : list
            Categorical feature names.
        """
        Xy_train, Xy_valid = self._train_valid_split(data, num, cat)
        self.selector = TreeSelectorGB(**self._get_setup(), categorical_feature=cat)
        self.selector.fit(
            *Xy_train,
            shadow_quantile = self.prm.shadow_quantile,
            n_permutations  = self._get_n_permutations(),
            eval_set = [Xy_valid]
        )

    def _get_selector_result(self) -> list:
        """
        Retrieve the selected features from Stage 1.

        Returns
        -------
        list
            Feature names identified by the tree-based selection.

        Raises
        ------
        ValueError
            If `fit_selector` has not been called.
        """        
        if self.selector is None:
            raise ValueError("Fit selector .fit_selector() prior calling results.")
        return self.selector.result
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def _get_n_shadow_samples(self) -> int:
        """
        Determine shadow permutation count for the pruning stage.

        Returns
        -------
        int
            Number of shadow samples (clipped to [2, max]).
        """
        x = max(2, int(20 / self.prm.cv))
        return np.clip(self.prm.n_permutations, 2, x)
                    
    def fit_pruner(self, data:pDF) -> None:
        """
        Stage 2: Shadow feature pruning.

        Compares each Stage 1 candidate against random permutations.
        Features not significantly better than noise are removed.

        Parameters
        ----------
        data : pDF
            Full dataset.
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
        Assemble the final result and stats from both stages.

        Parameters
        ----------
        fes : list
            Original feature list (for computing the dropped set).

        Returns
        -------
        tuple[FitResult, FitStats]
            Selected/dropped features + diagnostic paths.
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

    def pipeline(self, data:pDF, num:list, cat:list) -> None:     
        """
        Execute the full two-stage feature selection pipeline.

        Stage 1 (Tree Selector) → Stage 2 (Shadow Pruner).

        Parameters
        ----------
        data : pDF
            Full dataset.
        num : list
            Numerical feature names.
        cat : list
            Categorical feature names.
        """
        self.fit_selector(data, num, cat)
        self.fit_pruner(data)        
        
    def process(self, data:pDF, num:list, cat:list) -> tuple[FitResult, FitStats, FeaturesShadowPruner]:
        """
        Entry point: execute the selection pipeline and return all results.

        Parameters
        ----------
        data : pDF
            Full dataset.
        num : list
            Numerical feature names.
        cat : list
            Categorical feature names.

        Returns
        -------
        tuple[FitResult, FitStats, FeaturesShadowPruner]
            Selection result, stats, and the fitted pruner.
        """
        self.pipeline(data, num, cat)
        result, stats = self._get_result(num + cat)
        return result, stats, self.pruner
# ─────────────────────────────────────────────────────────────────────────────────────────────────
   
class Fitter(FitterTrainBase):
    """
    GBM Training — execution backend.

    Orchestrates the full training pipeline:

    1. **Feature Selection** (two stages):
       - Tree-based selection with early stopping → candidates.
       - Shadow pruning → final minimal feature set.

    2. **Final Model Fit**: GBM (LightGBM/XGBoost) on train set with
       early stopping on validation set.

    3. **Validation Diagnostics**: AUC, Gini, lift table on the
       validation set.

    4. **Importance Ordering**: Features ranked by contribution.

    Unlike the GLM fitter (which always produces a Logistic Regression),
    the GBM fitter produces a tree-based ensemble — capturing
    non-linear effects and feature interactions natively.

    Parameters
    ----------
    params : FitParams
        Tree complexity and selection thresholds.
    config : EstimatorCfg
        Engine choice and hyperparameters.
    """
    _model: LGB | XGB = None
    
    def __init__(self, params:FitParams, config:EstimatorCfg=None):
        self.prm = params 
        self.config = config or EstimatorCfg()
    
    def _get_result_c(self, cat:list) -> tuple[list, list]:
        return [x for x in self.features if x in cat]
    
    def _fit(self, worker:FitWorker, data:pDF, cat:list) -> pd.Index:        
        Xy_train, Xy_valid = worker._train_valid_split(data, self.features)

        e: LGB|XGB = GBM[self.config.solver]
        s = self.get_estimator_shared_setup(e, self.prm) | dict(verbose=-1)

        self._model = e(**s, categorical_feature = self._get_result_c(cat))
        self._model.fit(*Xy_train, eval_set = [Xy_valid])  

        return Xy_valid[0].index

    def _valid_diagnostics(self, d_val:pDF) -> DiagnosticsResult:
            t = d_val[self.prm.target]
            p = self.predict_proba(d_val)[:, 1]
            return SeparationDiagnostics(lift_bins=10).evaluate(t, p)
    
    def fit(self, data:pDF, num:list, cat:list):
        """
        Execute the full GBM training pipeline.

        Steps:
        1. Two-stage feature selection (tree selector + shadow pruner).
        2. Fit final GBM with early stopping.
        3. Compute validation diagnostics.
        4. Order features by importance.

        Parameters
        ----------
        data : pDF
            Full dataset (WOE features + optional categorical + target).
        num : list
            Numerical WOE feature names.
        cat : list
            Categorical feature names.

        Returns
        -------
        self
            Fitted instance with `model`, `features`, `stats`,
            and `ordered_features` set.
        """
        worker = FitWorker(self.prm, self.config)
        self.result, stats, pruner = worker.process(data, num, cat)
        
        self.features = self.result.features_selected
        val_idx = self._fit(worker, data, cat)
        
        self.stats = FitStats(
            **(dict(stats) | dict(diagnostics = self._valid_diagnostics(data.loc[val_idx])))
        )        
        self.ordered_features = pruner.get_ordered_importance_fes(self.features)

        self.dash = FitDashLink(
            pruner    = pruner,
            model     = self._model,
            target    = self.prm.target,
            features  = self.features,
            total_fes = len(num)+len(cat),
            val_idx   = val_idx
        )

        return self  
        
    def predict_proba(self, data:pDF) -> NDArray:
        """
        Predict event probabilities using the fitted GBM model.

        Parameters
        ----------
        data : pDF
            New data containing the selected features.

        Returns
        -------
        NDArray
            2-D array of shape (n_samples, 2): [P(0), P(1)] per row.
        """
        return self.model.predict_proba(data[self.features])
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LEARN → GBM
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class BinaryGBM(FitterModelBase[Fitter]):
    """
    Binary GBM — Base Class for Tree-Based Feature Selection + GBM Training.

    Public API for the GBM training stage. Configures the tree-based
    feature selection engine and orchestrates the full pipeline:
    selection → pruning → GBM fit → diagnostics.

    Pipeline Position
    -----------------
    Final training stage (alternative to GLM). Input: WOE-encoded
    features (+ optional categorical) + binary target.

    GBM vs GLM — When to Use
    ------------------------
    - **GBM**: Non-linear effects, feature interactions, categorical
      features, higher predictive ceiling. Best when the relationship
      between features and target is not purely linear.
    - **GLM**: Maximum interpretability, linear scoring, regulatory
      transparency. Best when a linear scorecard is required.

    Both use the same two-stage selection + shadow pruning, ensuring
    a stable, minimal feature set regardless of the final model class.

    Parameters
    ----------
    target : str
        Binary target column name (0/1).
    max_leaves : int, default=31
        Maximum leaves per tree (complexity control).
    learning_rate : float, default=0.1
        Shrinkage factor per boosting round.
    n_estimators : int, default=100
        Maximum boosting rounds (early stopping may stop earlier).
    min_child_weight : float, default=0.1
        Minimum weight in a leaf (overfitting control).
    shadow_quantile : float, default=0.9
        Shadow pruning threshold (higher = more conservative).
    cv : int, default=3
        Cross-validation folds for shadow evaluation.
    val_size : float, default=0.33
        Validation fraction (early stopping + diagnostics).
    n_permutations : int, default=3
        Shadow permutation count.
    random_state : int, default=42
        Seed for reproducibility.
    verbose : bool | Verbosity, default=True
        Enable progress display.

    Notes
    -----
    **Why Two-Stage Selection for GBM?**
    GBM trees can memorize training noise, especially in the early
    boosting rounds. Stage 1 (tree selection) captures the signal;
    Stage 2 (shadow pruning) removes the noise. Together they produce
    a feature set that generalizes well to OOT data.

    **Categorical Features**: Unlike GLM (which requires WOE encoding
    for all features), GBM natively handles categorical features
    without encoding. Pass them via `cat_features`.

    **Early Stopping**: The validation set is used during both
    selection (to identify when to stop boosting) and final fitting
    (to determine the optimal number of trees). This prevents
    overfitting more effectively than a fixed `n_estimators`.

    Examples
    --------
    >>> from aidflow.models import LightBinaryGBM
    """
    _solver: str = None

    def __init__(self, *args, params: dict|FitParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(FitParams, params, args, kwargs)
        self._config = self.get_estimator_config(self._params, kwargs)
    
    def _get_fes(self, data:pDF, num:list|None, cat:list|None) -> list:
        if num is not None or cat is not None:
            return num or [], cat or []
        return [x for x in data.columns if x!=self._params.target], [] # default all num, no cat
        
    def fit(self, data:pDF, num_features:list = None, cat_features:list = None):
        """
        Fit the GBM: feature selection + tree-based training.

        Executes the full pipeline:
        1. Two-stage feature selection (tree selector + shadow pruning).
        2. GBM fit with early stopping.
        3. Validation diagnostics.

        Parameters
        ----------
        data : pDF
            WOE-encoded features + optional categorical + target.
        num_features : list, default=None
            Numerical WOE features to select from. If None, all
            non-target columns are used as numerical.
        cat_features : list, default=None
            Categorical features (natively handled by the GBM).
            If None, no categorical features are used.

        Returns
        -------
        self
            Fitted instance ready for `predict_proba`.
        """       
        num, cat = self._get_fes(data, num_features, cat_features)
        cfg = EstimatorCfg(solver = self._solver, config = self._config)
        self._fitter = Fitter(self._params, cfg)

        title = f'GBM {self._solver.upper()} training in progress'
        with Spinner(title, show_timer=True).progress(self._params.verbose):        
            self._fitter.fit(data, num, cat)
            
        return self
    
    def predict_proba(self, data:pDF) -> NDArray:
        """
        Predict event probabilities.

        Parameters
        ----------
        data : pDF
            New data containing the selected features.

        Returns
        -------
        NDArray
            2-D array of shape (n_samples, 2): [P(0), P(1)] per row.
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

class LightBinaryGBM(BinaryGBM):
    """
    Binary GBM — LightGBM Engine.

    Uses LightGBM (leaf-wise tree growth) for both feature selection
    and final model training.

    Characteristics
    ---------------
    - **Leaf-wise growth**: Splits the leaf with the highest loss
      reduction. Produces deeper, more specialized trees.
    - **Fast training**: Hist-based splitting, parallel feature
      evaluation. Best for large datasets.
    - **Higher overfitting risk**: Leaf-wise growth can memorize
      noise. Mitigated by `max_leaves` and `min_child_weight`.
    - **Categorical support**: Native categorical feature handling.

    Best For
    --------
    - Large datasets (100k+ rows) where training speed matters.
    - Feature sets with clear signal (fewer noise features).
    - Scenarios where maximum predictive accuracy is the priority.

    Parameters
    ----------
    target : str
        Binary target column name.
    max_leaves : int, default=31
        Maximum leaves per tree (primary complexity control).
    learning_rate : float, default=0.1
        Shrinkage factor.
    n_estimators : int, default=100
        Maximum boosting rounds.
    min_child_weight : float, default=0.1
        Minimum leaf weight (overfitting control).
    shadow_quantile : float, default=0.9
        Shadow pruning threshold.
    verbose : bool | Verbosity, default=True
        Enable progress display.

    Examples
    --------
    >>> from aidflow.fitters import LightBinaryGBM
    >>> model = LightBinaryGBM(target='is_churn', max_leaves=15)
    >>> model.fit(data, num_features=['rev_WOE', 'ten_WOE'],
    ...           cat_features=['segment'])
    >>> model.predict_proba(new_data)
    """
    _solver = TS.L
    _estimator = GBM[_solver]

class ExtremeBinaryGBM(BinaryGBM):
    """
    Binary GBM — XGBoost Engine.

    Uses XGBoost (depth-wise tree growth) for both feature selection
    and final model training.

    Characteristics
    ---------------
    - **Depth-wise growth**: Splits all leaves at a given level
      before going deeper. Produces more balanced trees.
    - **Lower overfitting risk**: Depth-wise growth is more
      conservative than leaf-wise. Better for noisy data.
    - **Robust defaults**: XGBoost's regularization (`lambda`,
      `alpha`, `subsample`) provides additional stability.
    - **Categorical support**: Native categorical feature handling.

    Best For
    --------
    - Noisy datasets where overfitting is a concern.
    - Scenarios where model stability (low variance) is preferred
    over maximum training accuracy.
    - Smaller datasets where the speed advantage of LightGBM
    is less critical.

    Parameters
    ----------
    target : str
        Binary target column name.
    max_leaves : int, default=31
        Maximum leaves per tree.
    learning_rate : float, default=0.1
        Shrinkage factor.
    n_estimators : int, default=100
        Maximum boosting rounds.
    min_child_weight : float, default=0.1
        Minimum leaf weight.
    shadow_quantile : float, default=0.9
        Shadow pruning threshold.
    verbose : bool | Verbosity, default=True
        Enable progress display.

    Examples
    --------
    >>> from aidflow.fitters import ExtremeBinaryGBM
    >>> model = ExtremeBinaryGBM(target='is_default')
    >>> model.fit(data, num_features=['score_WOE', 'util_WOE'])
    >>> model.predict_proba(new_data)
    """
    _solver = TS.X
    _estimator = GBM[_solver]
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬