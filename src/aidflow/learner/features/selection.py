"""
Feature Selection — Regularization & Shadow Importance Engines.

This module implements the feature selection stage for both linear
(GLM) and non-linear (GBM) models. It provides two complementary
selection engines:

1. **Linear Selector** (GLM): Exploits the linearity of WOE
   features in log-odds space. Uses regularization path
   exploration (Lasso, ElasticNet, PLS, or stepwise) to identify
   features with statistically significant coefficients. Fast,
   interpretable, and produces a parsimonious feature set.

2. **Tree Selector** (GBM): Uses shadow-importance analysis with
   LightGBM/XGBoost. For each iteration, shadow copies of
   features are appended to the training matrix. The model's
   feature importances (gain) for the original features are
   compared against the shadow baseline. Features whose gain
   does not exceed a high quantile of the shadow distribution
   are pruned.

Dual-Track Design
-----------------
The two tracks serve different model families:

- **GLM agents** consume the Linear Selector's output: a sparse
  set of features with non-zero coefficients, ready for final
  Logistic Regression fitting.
- **GBM agents** consume the Tree Selector's output: a pruned
  feature set that passed the shadow-importance test, ready for
  final gradient boosting fitting.

Both selectors produce a **stable, reproducible** feature subset
that is robust to sample variation — critical for production
scorecards where feature availability may fluctuate.
"""

from ...deps.data               import  np, pDF, pSe
from ...deps.display            import  display
from ...deps.types              import  (
                                        Any,
                                        Literal, 
                                        NDArray, 
                                        ArrayLike
                                        )
from ...core.factory            import  (
                                        DictClass, 
                                        Dataclass
                                        )
from ...ml.models               import  (
                                        Logit,
                                        LogReg,
                                        Lasso, 
                                        ElasticNet,
                                        XGB,
                                        LGB,
                                        SparseSelector,
                                        PLSSelector,
                                        )                                        
from ...ml.models.prediction    import  PredictSelectionMixin
from ...ml.sampling             import  (
                                        Subsampling, 
                                        DataUtils
                                        )


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LS(DictClass):
    """Linear Selector: sparse | lasso | elastic | logreg."""
    S = 'sparse'
    P = 'pls'
    L = 'lasso'
    E = 'elastic'
    R = 'logreg'

class TS(DictClass):
    """Tree Selector: xgb | lgb."""
    X = 'xgb'
    L = 'lgb'

class TRES(DictClass):
    """Tree Results."""
    F = 'final_model'
    I = 'feature_importance'
    M = 'Average'
    C = 'Choice'
    Y = "✅"
    N = "❌"

@Dataclass.base
class TreeSelectorStep:
    """
    Single iteration result for the shadow-importance selection.

    Attributes
    ----------
    importance : pDF
        Per-feature importance values (original features only).
    shadow_th : float
        The shadow importance threshold for this iteration
        (quantile of the shadow feature importances).
    org_min : float
        Minimum importance among the original features.
    iter_i : int | str
        Iteration identifier (integer index, or `'final_model'`
        for the aggregated result).
    """    
    importance:   pDF
    shadow_th:  float
    org_min:    float
    iter_i:   int|str

    def __len__(self) -> int:
        c = TRES.M if TRES.M in self.importance.columns else TRES.I
        return sum(self.importance[c] > self.shadow_th)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOG-REG FEATURES SELECTION BY MODEL'S RESIDUAL CHANGE ON CV SAMPLES
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class SelectionBase:
    """
    Base class for feature selection engines.

    Provides the common contract: solver registration, setup
    management, subsampling, optimization path tracking, and
    estimator instantiation. 

    Parameters
    ----------
    verbose : bool, default=False
        Enable progress output during the selection process.
    """    
    implemented_estimators: dict

    def __init__(self, verbose:bool=False):
        self.verbose = verbose
        self._set_init()

    def _set_init(self):
        self.result      = []
        self.optim_path  = {}    

    def _get_solver(self, solver:str) -> Any:
        return self.implemented_estimators[solver]        
    
    def _set_setup(self, solver:str, setup:dict) -> None:
        self.solver = solver
        self.setup  = {'estimator': self._get_solver(solver), 'setup': setup}

    def _subsample(self, X: pDF, y: pSe, is_gbm:bool=False) -> tuple[pDF, pSe]:
        rs = self.get_setup().get('random_state', 42)
        ss = Subsampling(X, y, rs)
        if is_gbm:
            return ss.gbm()
        return ss.glm()
        
    def _update_optim_path(self, x: ArrayLike | TreeSelectorStep, i:int, info:str = None):
        self.optim_path[i] = x
        default = f'Iteration {i} ► Features Selected: {len(x)}'
        self._info(default if not info else info)

    def _info(self, info:str):
        if not self.verbose:
            return
        print(info)

    def get_estimator(self) -> Any:
        return self.setup['estimator']
    
    def get_setup(self) -> dict:
        return self.setup['setup']
    
    def get_estimator_instance(self, **kwargs) -> Any:
        e = self.get_estimator()
        s = self.get_setup() | kwargs
        return e(**s)     
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

   


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOG-REG FEATURES SELECTION → ADJ. LASSO/ELASTIC CV APPROACH GIVEN WOE+LR MATH
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class PredictLinearBase(PredictSelectionMixin):
    """Linear prediction mixin — fits the final Logit on selected features."""

    model:Logit = None

    def _set_predict_model(self, X:pDF|NDArray, y:pSe) -> None:
        if not len(self.result):
            return
        self.model = Logit().fit(X, y)

    @property
    def params(self):
        """Returns chosen significant features with coefficients."""
        return self.model.params_
    
    @property
    def summary(self) -> None:
        """Shows statistical summary of the final model."""
        display(self.model.summary_)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class LinearSelectorBase(SelectionBase):
    """
    Linear feature selection base — regularization-path driven.

    Implements the iterative coefficient-shrinkage selection for
    linear models. Exploits the WOE+GLM mathematical alignment:
    WOE features are approximately linear in log-odds space, so
    regularization (L1/L2) effectively performs feature selection
    by shrinking non-informative coefficients to zero.

    Solver Strategies
    -----------------
    - **sparse**: Sequential stepwise selection. Adds/removes
      features by statistical significance. Produces the
      smallest model.
    - **pls**: Partial Least Squares. Maximizes covariance
      between features and target. Best for high-dimensional,
      correlated inputs.
    - **lasso / elastic / logreg**: Iterative fit on the
      shrinking feature subset. Features with coefficients
      above a numeric threshold (1e-8) are retained. Iteration
      continues until the feature set stabilizes.

    Parameters
    ----------
    solver : str
        Solver identifier from the `LS` registry.
    setup : dict
        Solver-specific hyperparameters.
    verbose : bool, default=False
        Enable progress output.
    """

    implemented_estimators = {
        LS.S: SparseSelector,
        LS.P: PLSSelector,
        LS.L: Lasso,
        LS.E: ElasticNet,
        LS.R: LogReg
    }
    
    def __init__(self, 
        solver:  LS, 
        setup:   dict,
        verbose: bool = False
    ):
        super().__init__(verbose)
        self._set_setup(solver, setup)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _fit_sparse(self, X:pDF, y:pSe) -> list:        
        e:SparseSelector = self.get_estimator_instance(verbose=self.verbose)
        m:SparseSelector = e._fit(X, y)
        self.optim_path = m.optim_path
        return m.features
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _fit_pls(self, X:pDF, y:pSe) -> list:
        e:PLSSelector = self.get_estimator_instance(verbose=self.verbose)
        m:PLSSelector = e._fit(X, y)
        self.optim_path = m.optim_path
        return m.features
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _fit_model(self, X:pDF, y:pSe) -> Lasso|ElasticNet|LogReg:
        m:Lasso|ElasticNet|LogReg = self.get_estimator_instance()
        return m.fit(X, y)
    
    def _iter_model(self, X:pDF, y:pSe, i:int) -> list:
        c = 'coef'
        m = self._fit_model(X, y)
        b = m.coef_
        b = b if len(b.shape) == 1 else b[0]
        x = pDF(
            {c: np.transpose(b)}, 
            index = X.columns
        ).sort_values(by=c, ascending=False)
        self._update_optim_path(x, i)
        return x.loc[x[c] > 1e-8].index.to_list()
    
    def _iter_update(self, x:list, x_:list) -> list:
        if len(x_) and len(x) - len(x_):
            return x_
        return x
        
    def _fit_scikit(self, X:pDF, y:pSe, fes:list, max_iter:int=10) -> list:
        for i in range( max(1, max_iter) ):
            fes_ = self._iter_model(X[fes], y, i)
            fes  = self._iter_update(fes, fes_)
            if fes != fes_: # no change → no convergence
                break
        return fes
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def _fit(self, X:pDF, y:pSe, subset:list=None, max_iter:int=10) -> list:
        self._set_init()

        X_, y_ = self._subsample(X, y)
        X_, fe = DataUtils.align_data_with_fes(X_, subset)

        if self.solver == LS.S:
            return self._fit_sparse(X_, y_)
        
        if self.solver == LS.P:
            return self._fit_pls(X_, y_)
        
        return self._fit_scikit(X_, y_, fe, max_iter)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class LinearSelectorCV(LinearSelectorBase, PredictLinearBase):
    """
    GLM Cross-Validated Linear Feature Selection.

    Selects statistically significant WOE features using
    regularization-based coefficient shrinkage, then fits the
    final Logistic Regression on the surviving features.
    
    Why This Works for WOE + GLM
    ----------------------------
    WOE encoding produces features that are approximately
    linear in log-odds space. This means:

    - **L1/L2 regularization is a valid selection mechanism** —
      shrinking a coefficient to zero genuinely removes
      information (not just decorrelation).
    - **Convergence is fast** — the linear structure means
      the optimization landscape is convex and well-behaved.
    - **Interpretability is preserved** — the final model
      coefficients directly map to business risk.

    Solver Behaviors
    ----------------
    - **sparse**: Stepwise forward/backward selection. Most
      parsimonious, most interpretable.
    - **lasso**: L1 penalty — drives the weakest coefficients
      to exactly zero. Picks one feature from a correlated
      group.
    - **elastic**: L1+L2 penalty — combines Lasso selection
      with Ridge stability. Keeps all relevant features from
      a correlated group.
    - **logreg**: No penalty — all features retained. Useful
      as a baseline or when the feature set is already curated.
    - **pls**: Maximizes covariance with target. Best for
      high-dimensional, correlated feature spaces.

    Parameters
    ----------
    alpha : float, default=0.05
        Only used (applicable) when solver is 'sparse' or 'pls'.
        Significance level (alpha) for fe assessment.
        Statistical hypotesis significance for model's residual / performance
        change by adding/removing a feature under the assessment.

    eps : float, default=1e-3
        Length of the path. `eps=1e-3` means that
        `alpha_min / alpha_max = 1e-3`.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are smaller or equal to
        `tol`, the optimization code checks the dual gap for optimality and continues
        until it is smaller or equal to `tol`.
        
    alphas : array-like or int, default=100
        Values of alphas to test along the regularization path, used for each l1_ratio.
        If int, `alphas` values are generated automatically.
        If array-like, list of alpha values to use.

    l1_ratio : float or list of float, default=0.5
        Only used (applicable) when `solver='elastic'`.
        Float between 0 and 1 passed to ElasticNet (scaling between
        l1 and l2 penalties). For `l1_ratio = 0`
        the penalty is an L2 penalty. For `l1_ratio = 1` it is an L1 penalty.
        For `0 < l1_ratio < 1`, the penalty is a combination of L1 and L2
        This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of
        values for l1_ratio is often to put more values close to 1
        (i.e. Lasso) and less close to 0 (i.e. Ridge), as in `[.1, .5, .7,
        .9, .95, .99, 1]`.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    max_iter : int, default=1000
        The maximum number of iterations.
    
    max_components : int, optional
        Only used (applicable) when `solver='pls'`.
        The maximum number of PLS components to evaluate during PLS search. 
        Default is 10.

    cv : int, default=3
        Cross-validation generator or iterable.
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - int, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

            verbose : bool or int, default=0
        Amount of verbosity.

    n_jobs : int, default=-1
        Number of CPUs to use during the cross validation.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when `selection` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    solver : {'sparse, 'pls, 'logreg', 'elastic', 'lasso'}, default='pls'.
        Solver choses backend estimator to solve the features selection problem.
        Solver choice provides ML model: model = estimator(**params).fit(X, y).
    """

    def __init__(self, 
        alpha:        float = 0.05,
        eps:          float = 1e-3,
        tol:          float = 1e-4,
        l1_ratio:     float = 0.5, 
        alphas:    list|int = 100,
        max_iter:       int = 1000, 
        max_components: int = 10,
        cv:             int = 3,
        random_state:   int = 42,
        n_jobs:         int = -1, 
        verbose:       bool = False,
        solver: Literal['sparse', 'pls', 'logreg', 'elastic', 'lasso'] = 'sparse', 
        **kwargs
    ):
        setup = dict(
            eps          = eps,
            tol          = tol,
            alphas       = alphas,
            max_iter     = max_iter,
            cv           = cv,
            n_jobs       = n_jobs,            
            random_state = random_state,            
        ) | kwargs

        if solver in [LS.S, LS.P]: 
            setup['alpha'] = alpha
        
        if solver == LS.P:
            setup['max_components'] = max_components

        if solver == LS.E: 
            setup['l1_ratio'] = l1_ratio

        super().__init__(solver, setup, verbose)

    def _set_result_model(self, X:pDF|NDArray, y:pSe, result:list) -> None:
        X_, self.result = DataUtils.align_data_with_result(X, result)
        self._set_predict_model(X_, y)

    def fit(self, X:pDF, y:pSe, subset:list=None, max_iter:int=10):
        """
        Execute the linear feature selection pipeline.

        Selects significant features via the configured solver,
        then fits the final Logistic Regression on the selected
        subset.

        Parameters
        ----------
        X : pDF
            Feature matrix (WOE-encoded).
        y : pSe
            Target series (binary: 0/1).
        subset : list | None, default=None
            Restrict selection to a specific feature subset.
            If None, all columns in X are candidates.
        max_iter : int, default=10
            Maximum selection iterations (for iterative solvers
            like Lasso/ElasticNet).

        Returns
        -------
        self
            Fitted instance with `.result` (selected features)
            and `.model` (final Logit) set.
        """
        fes = self._fit(X, y, subset, max_iter)
        self._set_result_model(X, y, fes)
        return self         
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# GB TREE FEATURES SELECTION → SYNC XGB/LGB
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class PredictTreeBase(PredictSelectionMixin):
    """Tree prediction mixin — retains the fitted GBM on selected features."""

    model: XGB | LGB = None

    def _set_predict_model(self, model:XGB|LGB) -> None:
        if not len(self.result):
            return
        self.model = model
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
       
class TreeSelectorBase(SelectionBase):
    """
    Shadow-Importance Tree Feature Selection.

    Identifies genuinely informative features for GBM models by
    comparing their importance against a shadow baseline
    (importance of randomly permuted features).

    Mechanism
    ---------
    In each iteration (with a different random seed):
    1. Shadow features are created by permuting the target.
    2. The model is fit on [original features + shadow features].
    3. Feature importances are extracted for both sets.
    4. The shadow importance distribution defines the threshold.

    After all iterations, the mean importance of each original
    feature is compared against the mean shadow threshold.
    Features exceeding the threshold are considered genuinely
    informative (their signal is stronger than random noise).

    Why This Eliminates Overfitting
    -------------------------------
    A feature that appears important only due to noise will have
    an importance comparable to the shadow features (random
    permutations). By requiring the importance to exceed the
    shadow baseline, we filter out features whose signal is
    indistinguishable from noise.

    Parameters
    ----------
    solver : str
        Tree backend: `'xgb'` or `'lgb'`.
    setup : dict
        Tree hyperparameters (max_leaves, learning_rate, etc.).
    verbose : bool, default=False
        Enable progress output.
    """

    implemented_estimators = {
        TS.X: XGB,
        TS.L: LGB,
    }
    
    def __init__(self, 
        solver:  TS, 
        setup:   dict,
        verbose: bool = False
    ):
        super().__init__(verbose)
        self._set_setup(solver, setup)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
            
    def _get_cat_fe(self) -> list:
        s = self.get_setup()
        return s['categorical_feature']
    
    def _get_setup(self, X:pDF, shadow_seed:int=None) -> dict:
        setup = self.get_setup()
        cat = self._get_cat_fe()
        if not cat:
            return setup
        cat = [x for x in X.columns if x in cat]
        if shadow_seed is not None:
            cat = DataUtils.get_cols_with_shadow(cat)
        return setup | dict(categorical_feature=cat)
    
    def _get_estimator_instance(self, X:pDF, shadow_seed:int=None) -> XGB|LGB:
        e = self.get_estimator()
        s = self._get_setup(X, shadow_seed)
        return e(**s)
    
    def _get_X(self, X:pDF, shadow_seed:int=None) -> pDF:
        if shadow_seed is None:
            return X
        return DataUtils.extend_by_shadow_df(X, shadow_seed)

    def _align_eval_df(self, eval_X:pDF|NDArray, train_fes:list, shadow_seed:int=None) -> pDF:
        X_, _ = DataUtils.align_data_with_fes(eval_X, train_fes)
        return self._get_X(X_, shadow_seed)

    def _align_eval_set(self, X:pDF|NDArray, kwargs:dict, shadow_seed:int=None) -> dict:
        es = 'eval_set'
        eval_set = kwargs.get(es, None)
        if eval_set is None:
            return kwargs
        cols = DataUtils.remove_shadow_cols(X.columns)
        al_X = lambda eval_X: self._align_eval_df(eval_X, cols, shadow_seed)
        kwargs[es] = [(al_X( Xy[0] ), Xy[1]) for Xy in eval_set]
        return kwargs
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _fit_model(self, X:pDF, y:pSe, kwargs:dict, shadow_seed:int=None) -> XGB|LGB: 
        m = self._get_estimator_instance(X, shadow_seed)
        k = self._align_eval_set(X, kwargs, shadow_seed)
        return m.fit(X, y, **k)
    
    def _fit_model_importance(self, X:pDF, y:pSe, kwargs:dict, shadow_seed:int=None
    ) -> ArrayLike:
        model = self._fit_model(X, y, kwargs, shadow_seed)
        return model.feature_importances_

    def _iter_shadow_importance(self, X:pDF, y:pSe, kwargs:dict, quantile:float, i:int
    ) -> tuple[ArrayLike, float]:
        
        X_ = self._get_X(X, i)        
        fe_im = self._fit_model_importance(X_, y, kwargs, i)

        lc = len(X.columns)
        real_im   = fe_im[:lc]
        shadow_im = fe_im[lc:]
        threshold = np.quantile(shadow_im, np.clip(quantile, 0, 1)) 
            
        return real_im, threshold

    def _iter_update_path(self, importance:ArrayLike, shadow_th:float, fes:list, i:int|str):
        x = pDF(
            {TRES.I: np.transpose(importance).astype(float)},
            index = fes
        ).sort_values(by=TRES.I, ascending=False)

        self._iter_update(TreeSelectorStep(
            importance  = x,
            shadow_th   = shadow_th,
            org_min     = x[TRES.I].min(),
            iter_i      = i
        ))
    
    def _iter_model(self, X:pDF, y:pSe, kwargs:dict, quantile:float, i:int) -> float:
        im, th = self._iter_shadow_importance(X, y, kwargs, quantile, i)                
        self._iter_update_path(im, th, X.columns, i)
        return th
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _iter_info(self, step:TreeSelectorStep) -> str:
        i  = step.iter_i
        th = step.shadow_th
        s  = f'shadow cutoff: {th:.5f}'
        
        if not ( i == TRES.F ):
            return f'Iteration {i} ► Obtained {s}'
                
        im  = step.importance
        fes = f'► Selection {sum(im[TRES.M]>th)} from {len(im)} features'
        return f'{i} ►  Average {s} {fes}'

    def _iter_update(self, step:TreeSelectorStep) -> None:
        info = self._iter_info(step)
        self._update_optim_path(step, step.iter_i, info)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _iter_mean(self, fes:list, th:list) -> tuple[pDF, float]:
        optim_path:dict[int, TreeSelectorStep] = self.optim_path
        
        d = pDF(index = fes)
        for i, step in optim_path.items():
            d[f'iter_{i}'] = step.importance[TRES.I]
        
        th_mean  = np.average(th)
        d[TRES.M] = d.mean(axis=1)
        d[TRES.C] = np.where(d[TRES.M] > th_mean, TRES.Y, TRES.N)

        return d, th_mean

    def _final_model(self, fes:list, th:list) -> list:
        im_agg, th_mean = self._iter_mean(fes, th)
        im_mean = im_agg[TRES.M]
        self._iter_update(TreeSelectorStep(
            importance  = im_agg,
            shadow_th   = th_mean,
            org_min     = im_mean.min(),
            iter_i      = TRES.F
        ))
        return im_mean[im_mean > th_mean].index.to_list()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _fit(self, 
        X: pDF, 
        y: pSe, 
        subset: list = None, 
        quantile: float = 0.9, 
        n_iter: int = 5, 
        **kwargs
    ) -> list:        
        self._set_init()
        
        X_, y_ = self._subsample(X, y, is_gbm= True)
        X_, _  = DataUtils.align_data_with_fes(X_, subset)

        th = []
        for i in range( max(1, n_iter) ):
            th.append( 
                self._iter_model(X_, y_, kwargs, quantile, i) 
            )
        return self._final_model(X_.columns, th)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class TreeSelectorGB(TreeSelectorBase, PredictTreeBase):
    """
    Shadow-Importance Tree Feature Selection (LightGBM / XGBoost).

    Selects genuinely informative features for GBM models using
    the shadow-importance mechanism. After selection, the final
    model is refit on the surviving features.

    Selection Protocol
    ------------------
    1. For each of `n_permutations` iterations (with different
       random seeds):
       - Create shadow features (permuted target).
       - Fit the tree model on [original + shadow] features.
       - Extract feature importances.
       - Compute the shadow threshold (quantile of shadow
         importances).
    2. Aggregate: mean importance across iterations per feature.
    3. Select: features whose mean importance exceeds the mean
       shadow threshold.
    4. Refit: the final model on the selected features only.

    Parameters
    ----------
    solver : str, default='lgb'
        Tree backend: `'lgb'` (LightGBM) or `'xgb'`
        (XGBoost).
    max_leaves : int, default=31
        Maximum leaves per tree.
    learning_rate : float, default=0.1
        Boosting learning rate.
    n_estimators : int, default=100
        Number of boosting rounds.
    min_child_weight : float, default=0.1
        Minimum sum of instance weight in a leaf.
    subsample : float, default=0.7
        Fraction of samples per tree (row sampling).
    colsample_bytree : float, default=0.7
        Fraction of features per tree (column sampling).
    reg_alpha : float, default=0.0
        L1 regularization on leaf weights.
    reg_lambda : float, default=0.0
        L2 regularization on leaf weights.
    early_stopping_round : int | None, default=None
        Early stopping patience (requires eval_set).
    categorical_feature : str | list | None, default=None
        Categorical feature names (for native tree handling).
    random_state : int, default=42
        Seed for reproducibility.
    n_jobs : int, default=-1
        Parallel workers. -1 = all cores.
    verbose : bool, default=False
        Enable progress output.
    """

    def __init__(self, 
        max_leaves:               int = 31,
        learning_rate:          float = 0.1,
        n_estimators:             int = 100, 
        min_child_weight:       float = 0.1,
        subsample:              float = 0.7,
        colsample_bytree:       float = 0.7, 
        reg_alpha:              float = 0.0,
        reg_lambda:             float = 0.0,
        early_stopping_round:     int = None,
        categorical_feature: str|list = None,
        random_state:             int = 42,
        n_jobs:                   int = -1,
        solver: Literal['xgb', 'lgb'] = 'lgb', 
        verbose:                 bool = False,
        **kwargs
    ):
        setup = dict(            
            max_leaves              = max_leaves,
            learning_rate           = learning_rate,
            n_estimators            = n_estimators, 
            min_child_weight        = min_child_weight,
            subsample               = subsample,
            colsample_bytree        = colsample_bytree,
            reg_alpha               = reg_alpha,
            reg_lambda              = reg_lambda,
            early_stopping_rounds   = early_stopping_round,
            categorical_feature     = categorical_feature,
            random_state            = random_state,
            n_jobs                  = n_jobs,
        ) | kwargs    

        super().__init__(solver, setup, verbose)

    def _set_result_model(self, X:pDF|NDArray, y:pSe, kwargs:dict, result:list) -> None:
        X_, self.result = DataUtils.align_data_with_result(X, result)
        model = self._fit_model(X_, y, kwargs)
        self._set_predict_model(model)

    def fit(self, 
        X: pDF|NDArray, 
        y: pSe, 
        subset: list = None, 
        shadow_quantile: float = 0.9, 
        n_permutations:  int = 5, 
        **kwargs
    ):
        """
        Execute the shadow-importance feature selection.

        Runs multiple iterations of shadow feature generation,
        model fitting, and importance extraction. Selects
        features that exceed the shadow baseline, then refits
        the final model on the selected subset.

        Parameters
        ----------
        X : pDF | NDArray
            Feature matrix.
        y : pSe
            Target series (binary: 0/1).
        subset : list | None, default=None
            Restrict selection to a specific feature subset.
            If None, all columns in X are candidates.
        shadow_quantile : float, default=0.9
            Quantile of the shadow importance distribution
            used as the selection threshold. 0.9 means a
            feature must exceed 90% of shadow importances to
            be retained. Higher = stricter.
        n_permutations : int, default=5
            Number of shadow iterations (different random seeds).
            Higher = more stable threshold estimate.
        **kwargs : dict
            Additional fit parameters (e.g., `eval_set` for
            early stopping).

        Returns
        -------
        self
            Fitted instance with `.result` (selected features)
            and `.model` (final GBM) set.
        """
        fes = self._fit(X, y, subset, shadow_quantile, n_permutations, **kwargs)
        self._set_result_model(X, y, kwargs, fes)
        return self
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬