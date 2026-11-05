"""
Statistically driven feature selection engine for optimized model parsimony.

This module implements the core logic for identifying the most predictive 
feature subsets while strictly controlling for noise and overfitting. By 
integrating significance testing (alpha levels) with cross-validation and 
residual analysis, it transforms a high-dimensional candidate space into 
a lean, stable set of predictors that adhere to industrial standards for 
model interpretability and robustness.
"""

from ...deps.common     import  warnings
from ...deps.data       import  np, pDF, pSe
from ...deps.ml.glm     import  (
                                LogisticRegression,
                                ElasticNetCV,                                
                                )
from ...deps.ml.stats   import  chi2, t
from ...deps.exceptions import  SKCovWarning
from ...deps.display    import  display
from ...deps.types      import  (
                                Literal,
                                NDArray,
                                ArrayLike
                                )
from ..metrics          import  Gini
from ..sampling         import  (
                                DatasetCV, 
                                DataUtils
                                )
from .models            import  Logit
from .prediction        import  PredictSelectionMixin


type LogReg = Logit|LogisticRegression

FIT_INTERCEPT = False   # False → adds const to data prior | True → model internal logic


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOG-REG WOE FEATURE SPARSE SELECTION BY MODEL'S RESIDUAL CHANGE ON CV SAMPLES
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class PredictBase(PredictSelectionMixin):
    
    model:Logit = None

    def _set_predict_model(self, X:pDF, y:pSe) -> None:
        if not self.result:
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

class WorkerBase:
    """
    Statistical Rationale
    =====================

    Computational Design
    --------------------
    To bound runtime, the selector operates on a constructed sample of
    at most 10,000 rows regardless of input size. The sample is built
    as a balanced (50:50) subset drawn directly from the full dataset:

    - If total positives ≥ 5,000:
        sample = 5,000 positives + 5,000 negatives = 10,000 rows
    - If total positives < 5,000:
        sample = all positives + fill to 10K by negatives
        (total = n_pos + n_neg → 10,000 rows)

    If the input has fewer than 10,000 rows, all observations are used
    (no subsampling).

    This balanced construction guarantees a maximum 50:50 ratio under
    all circumstances, maximising the number of positive events in the
    sample while maintaining class balance for numerical stability.

    Statistical Validity
    --------------------
    In logistic regression, altering the class sampling ratio shifts only
    the intercept; slope coefficients and their standard errors (and thus
    the 1-d.f. χ² / Wald tests used at each pruning step) are invariant
    to the sampling ratio. Feature selection is therefore unbiased.

    WOE values computed on the balanced sample differ from those on the
    natural distribution by a uniform constant ln(sampling_ratio), which
    does not affect bin ordering or discriminative ranking.

    Minimum Event Requirement
    -------------------------
    The step-wise 1-d.f. LRT at each iteration is valid as long as the
    base model (k retained coefs) is stably estimated. Practical rule:

        n_events ≥ 10 × k        (at step k)

    The absolute floor is 100 events (supports ~10 retained features).
    At the design target of 30–40 features (~120–160 free parameters
    with 5-bin WOE), 5,000 events provides 31–41 EPV — ample margin.

    If the dataset cannot meet this, the selector will still run but
    later-step p-values become anti-conservative (over-selection risk).
    The user should be warned and advised to either increase sample size
    or treat the output as a screening list to be re-validated on a
    holdout with sufficient events.

    WOE Bin Stability
    -----------------
    When the positive count per WOE bin is < 5, category-level estimates
    are unstable. Bin merging was applied to WOE encoding by the previous 
    flow step to prevent noisy signals from propagating into decisions.

    Design Intent: Signal Sparsity
    ------------------------------
    Features whose individual marginal contribution to the logit is below
    the significance threshold are eliminated by design. This is the
    objective of the sparse selector — to retain only features with a
    detectable standalone signal.

    Sparsity Bound
    --------------
    The sparse selector is designed to retain 30–40 features at most.
    This is not an arbitrary cap but a structural constraint: with an
    average of 5 WOE bins per feature (4 free parameters after
    dropping the reference), 40 features imply ~160 coefficients,
    which at 5,000 events yields 31 events-per-parameter — well above
    the 10 EPV stability threshold.

    Selecting beyond ~50 features is self-contradictory: the output
    is no longer sparse, and the model enters a regime where L2
    regularization (dense selector) is the statistically appropriate
    tool. If the step-wise procedure has not converged by 50 retained
    features, the user should be warned that the data exhibit
    distributed signal and directed to the dense selector.

    Known Limitation: Distributed Signal
    ------------------------------------
    In datasets where predictive signal is fragmented across many features
    — each individually below threshold — the step-wise procedure will
    terminate early, before the combined effect becomes detectable.

    This is not a failure but a diagnostic signal: the data exhibit a
    dense, distributed structure rather than a sparse one. In this case:

    1. Switch to the dense selector (L2-regularized WOE-logit), which
        retains most features with shrunken coefficients.
    2. Interpret the early-stop as confirmation that no single feature
        dominates — the signal is collective, not concentrated.

    Both selectors are valid; the sparse selector has a different
    inductive bias and is not the appropriate tool for dense-signal
    regimes.
    """
    def __init__(self, 
        model: None|LogReg = None, 
        cv:            int = 3, 
        tol:         float = 1e-4,
        random_state:  int = 42,
        verbose:      bool = False
    ):
        self._set_init()
        self._set_model(model)     
        self.cv = cv
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        
        self.dataset = DatasetCV(
            cv, 
            subsample = 10000, 
            random_state = random_state, 
            add_constant_to_X = not FIT_INTERCEPT
        )
    # ───────────────────────────────────────────────────────────────────────────────────────────── 

    def _set_init(self) -> None:
        self.result     = []
        self.removed    = []
        self.optim_path = {}  

    @staticmethod
    def _copy_model(model:LogReg) -> LogReg:
        setup = model.get_params()
        if 'fit_intercept' in setup:
            setup = setup | dict(fit_intercept=FIT_INTERCEPT)
            model = type(model)(**setup)
        return model
    
    def _set_model(self, model:None|LogReg) -> None:
        if model is None:
            model = Logit(fit_intercept=FIT_INTERCEPT)
        else: 
            model = self._copy_model(model)
        if not isinstance(model, (Logit, LogisticRegression)):
            raise TypeError('AIG SAFETY ► Logit or LogisticRegression estimator only.')
        self._model = model
    # ───────────────────────────────────────────────────────────────────────────────────────────── 
    
    def _reduce_fe_space(self, X:pDF, y:pSe, fes:list, return_model:bool=False
    ) -> list | tuple[list, ElasticNetCV]:
        """
        Pre/post-screening - reduces features by ElasticNetCV screening of correlated fes.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SKCovWarning)
            m = ElasticNetCV(
                tol = self.tol, random_state = self.random_state, n_jobs = -1
            ).fit(X[fes], y)

        f = list( np.array(fes)[m.coef_ > 1e-6] )   
             
        if not return_model:
            return f
        return f, m
        
    def _reduce_fes_dim(self, X:pDF, y:pSe, fes:list) -> list:
        """
        Optionally reduces input features by prior ElasticNetCV.
        - fes < 50 and data signal is not prone to signular matrix → return fes
        - otherwise run ElasticNetCV screening of correlated fes
        """
        if len(fes) < 50 and len(fes) / len(X) < 0.05:
            return fes
        return self._reduce_fe_space(X, y, fes)

    def _set_fes(self, X:pDF, y:pSe, subset:list=None) -> None:
        fes = DataUtils.get_fes(X, subset)
        self._fes = self._reduce_fes_dim(X, y, fes)
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    def _info(self, info:str) -> None:
        if not self.verbose:
            return
        print(info)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _has_model_llf(self, clf:LogReg) -> bool:
        return isinstance(clf, Logit)

    @staticmethod
    def _fit_backup(X_train:pDF, y_train:pSe) -> LogisticRegression:
        return LogisticRegression(
            C = 10, max_iter = 1000, random_state = 42, fit_intercept=FIT_INTERCEPT,
        ).fit(X_train, y_train)
    
    def _fit(self, X_train:pDF, y_train:pSe) -> LogReg:
        try:    return self._model.fit(X_train, y_train)    # standard wo penalty
        except: return self._fit_backup(X_train, y_train)   # backup with penalty for perfect separation

    def _predict(self, clf:LogReg, X:pDF) -> ArrayLike:
        return clf.predict_proba(X)[:,1]

    @staticmethod
    def _log_loss(y_true:pSe, yh:pSe) -> float:
        # llf = -log_loss(y_test, yh)*len(y_test) → note 20x slower than manual comp.
        yh = np.clip(yh, 1e-15, 1 - 1e-15)
        return np.sum( y_true*np.log(yh) + (1-y_true)*np.log(1-yh) ) # PhD p27 → Log_Likelihood comp
    
    def _train_ll(self, X_train:pDF, y_train:pSe) -> float:
        clf = self._fit(X_train, y_train)
        if self._has_model_llf(clf):
            return clf.llf_
        yh = self._predict(clf, X_train)
        return self._log_loss(y_train, yh)
                  
    def _cv_sample_ll(self, X_train:pDF, y_train:pSe, X_valid:pDF, y_valid:pSe) -> float:
        clf = self._fit(X_train, y_train)
        yh  = self._predict(clf, X_valid)
        return self._log_loss(y_valid, yh)
        
    def _cv_ll(self, fes:list):
        llf = []
        for i in self.dataset.get_cv_range():
            samples = self.dataset.get_cv_subset(i, fes)
            llf.append( self._cv_sample_ll(*samples) )
        return np.sum(llf) 

    def _ll_base(self, fes:list):
        try:    return self._train_ll(self.dataset.get_X_subset(fes), self.dataset.y)
        except: return -1e10    # remove fe if no convergence     

    def _log_likelihood(self, fes:list):                
        if self.dataset.is_cv: 
            try:    return self._cv_ll(fes)   # sum(llf) → clf.llf on cross-val samples  
            except: return self._ll_base(fes) # when cv split fails → return full sample   
        return self._ll_base(fes)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class FitWorker(WorkerBase):
    """
    Logistic Regression WOE Features Basis choice based on AIG Selection Methodology.

    Parameters
    ----------
    alpha : Significance level (alpha) for fe assessment, default=0.05.
        Statistical hypotesis significance for model's residual change
        by adding/removing a feature under the assessment.
    
    cv : Number of cross-validation folds, default=3.
        Model's residual change is assessed; 
        0 means off → assess on train sample.
    
    random_state : int, RandomState instance, default=42
        The seed of the pseudo random number generator to create CV folds.        

    verbose : Verbosity of the selector features assessment, default=False.
    
    model : solver model instance to solve the objective, default=None.
        Solver choses backend estimator to solve the features selection problem.
        Solver choice provides ML model: model = estimator(**params).fit(X, y).    
    """
    def __init__(self, 
        alpha:      float = 0.05, 
        tol:        float = 1e-4,
        cv:           int = 3, 
        random_state: int = 42,
        verbose:     bool = False,
        model:     LogReg = None
    ):
        super().__init__(model, cv, tol, random_state, verbose)
        self.alpha = np.clip(alpha, 0, 0.999)   # forward-backward action safety
        self.tol   = tol

    def _init_model_stats(self, y:pSe) -> None:
        self._llf_base  = -np.inf
        self._iter_step = 0
        self.model_gini = 0

        p = np.clip(y.mean(), 1e-15, 1 - 1e-15) # logit with only const. est: p=mean(y)
        L = self._log_loss(y, p)  
        self._update_model_ll(L)
        self._update_model_tol(L)
    
    def _update_model_ll(self, L:float) -> None:
        self.model_llf  = L  

    def _update_model_tol(self, L:float) -> None:
        tol = L - self._llf_base
        self._llf_base = L
        if tol < 0: # only backward pruning can cause negative llf change
            return  # we should allow model to prune without stopping the search
        self.model_tol = tol 

    def _update_model_gini(self) -> None:
        X_  = self.dataset.get_X_subset(self.result)
        clf = self._fit(X_, self.dataset.y)        
        self.model_gini = Gini.fast_calc(self.dataset.y, self._predict(clf, X_))

    def _update_model_stats(self, L:float) -> None:
        self._update_model_ll(L)
        self._update_model_tol(L)
        self._update_model_gini()

    def _get_model_ll(self) -> float:
        return self.model_llf   
    
    def _get_model_tol(self) -> float:
        return self.model_tol
    
    def _get_model_gini(self) -> float:
        return self.model_gini    

    def _get_step(self) -> int:
        i = self._iter_step
        self._iter_step +=1
        return i
    
    def _pv_test(self, Lj:float, backward:bool=False) -> float:
        L = self._get_model_ll()        
        if not backward:
              ld = L-Lj
        else: ld = Lj-L
        Gj = -2*(ld)
        return 1 - chi2(1).cdf(Gj)  # degrees of freedom = 1 as 1 new variable is in question 

    def _ll_test(self, fes:list, backward:bool=False) -> tuple[float, float]:
        Lj = self._log_likelihood(fes)        
        return Lj, self._pv_test(Lj, backward)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_search_vectors(self) -> None:
        self._ll_values = {}
        self._p_values  = {}

    def _update_search_vectors(self, fe:str, Lj:float, pv:float) -> None:
        self._ll_values[fe] = Lj 
        self._p_values[fe]  = pv   

    def _update_optim_path(self, fes:str|list, action:str):
        if isinstance(fes,str) : 
            fes = [fes]
        gini = self._get_model_gini()
        for fe in fes:
            self.optim_path[self._get_step()] = {
                'feature':    fe,
                'action':     action, 
                'p_value':    self._p_values[fe],
                'tol':        self._get_model_tol(),
                'model_ll':   self._get_model_ll(),                
                'model_gini': gini,
            }
        a = ' '*(8 - len(str(action)))
        g = ' '*(6 - len(str(gini)))
        self._info(f'{action}{a} →  Model Gini: {gini}{g}  →  Feature: {', '.join(fes)}')

    def _update_removed(self) -> None:
        """FE pruning with respect to impact to model's residuals."""
        out = [
            k for k,v in self._p_values.items() 
            if v > self.alpha and k not in self.removed
        ]
        self.removed.extend(out)

    def _add_chosen(self) -> None:
        """
        FW Update Vector chosen predictors.
        Chosen fe as the best predictor → minimizing residuals: 
         - max() return the last value in sorted key
         - key=d.get means the list will be sorted by values of the dictionary
        """
        if not self._ll_values:
            return
        fe = max(self._ll_values, key=self._ll_values.get)
        if self._p_values[fe] > self.alpha:
            return
        self.result.append(fe)
        self._update_model_stats(self._ll_values[fe])
        self._update_optim_path(fe, 'FORWARD')

    def _forward_adding(self) -> None:
        """Forward test → min(p-value), max(llf)."""
        self._set_search_vectors()
        for fe in self._fes:
            if ( fe in self.result ) or ( fe in self.removed ):
                continue
            fw = self.result + [fe]
            Lj, pv = self._ll_test(fw)
            self._update_search_vectors(fe, Lj, pv)
        self._add_chosen()

    def _remove_chosen(self, bw_out:list) -> None:
        """Backward pruning with respect to significance of model's residual change."""
        if not bw_out:
            return
        for x in bw_out: 
            self.result.remove(x)
        L = self._ll_test(self.result)[0]
        self._update_model_stats(L)
        self._update_optim_path(bw_out, 'BACKWARD')

    def _backward_prune(self) -> None:                    
        """Backward pruning with respect to significance of model's residual change."""
        if (
            not self._ll_values or
            not len(self.result) > 1
        ):  return
        bw_out = []
        IN = self.result.copy()
        for fe in self.result[:-1]: # -1 → last added by forward out of back-check
            bw = [x for x in IN if x != fe]
            Lj, pv = self._ll_test(bw, True)
            if ( pv > self.alpha ):
                IN.remove(fe)
                bw_out.append(fe)
                self._update_search_vectors(fe, Lj, pv)
        self._remove_chosen(bw_out)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _get_elasticnet_pvalues(model: ElasticNetCV, X: NDArray, y: NDArray) -> NDArray:
        """Wald inference for penalized linear regression."""
        N, p = X.shape

        # Include intercept column if model fitted with fit_intercept=True
        if model.fit_intercept:
            X_design = np.hstack([np.ones((N, 1)), X])
            coefs = np.insert(model.coef_, 0, model.intercept_)
        else:
            X_design = X.copy()
            coefs = model.coef_.copy()

        # Residuals & degrees of freedom
        y_pred = model.predict(X)
        residuals = y - y_pred
        active_features = np.count_nonzero(coefs)
        dof = max(1, N - active_features)
        sigma_sq = np.sum(residuals**2) / dof

        # Scikit-learn scales objective by 1/N, so L2 multiplier is N * alpha * (1 - l1_ratio)
        lambda_2 = N * model.alpha_ * (1.0 - model.l1_ratio)

        # Penalized Hessian: X^T X + lambda_2 * I
        penalty_matrix = lambda_2 * np.eye(X_design.shape[1])
        if model.fit_intercept:
            penalty_matrix[0, 0] = 0.0  # Intercept is unpenalized

        XTX = X_design.T @ X_design
        H = XTX + penalty_matrix
        H_inv = np.linalg.pinv(H)

        # Sandwich Covariance Estimator
        cov_matrix = sigma_sq * (H_inv @ XTX @ H_inv)
        se = np.sqrt(np.diagonal(cov_matrix))

        # Compute t-statistics and two-tailed p-values
        t_scores = np.divide(coefs, se, out=np.zeros_like(coefs), where=se != 0)
        return 2 * (1 - t.cdf(np.abs(t_scores), df=dof))
            
    def _update_finish_stats(self, model: ElasticNetCV) -> float:
        X = self.dataset.get_X_subset(self.result)
        y = self.dataset.y
        p = lambda x: dict(zip(self.result, x))
        try:
            m = Logit().fit(X, y)
            self._p_values = p(m.pvalues[1:])
            return m.llf
        except:
            pvalues = self._get_elasticnet_pvalues(model, X, y)
            self._p_values = p(pvalues)
            return self._ll_test(self.result)[0]        

    def _finish_convergence(self) -> None:
        """Finish selection of non-sparse feature set by ElasticNetCV."""
        rem = [x for x in self._fes if x not in self.result + self.removed]
        fes = self.result + rem

        X = self.dataset.get_X_subset(fes, to_numpy=False)
        self.result, m = self._reduce_fe_space(X, self.dataset.y, fes, return_model=True)

        L = self._update_finish_stats(m)
        self._update_model_stats(L)
        self._update_optim_path(self.result, 'ELASTIC')
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _is_continuation(self, i:int) -> bool:
        m = f'Selection stopped in step {i}'

        if not len(self._p_values):
            self._info(f'{m} → No FE to add/remove further.')
            return False

        pv = min(self._p_values.values())
        if pv > self.alpha:
            self._info(f'{m} → No FE to add/remove by p_value → remaining fes min p-value: {pv}.')
            return False
        
        tol = self._get_model_tol()
        if tol < self.tol:
            self._info(f'{m} → Model Tolerance reached {tol} → parameter th {self.tol}.')
            return False

        return True

    def _is_sparse(self, i:int) -> bool:        
        if i < 40:
            return True
        
        m = f'Selection stopped in step {i} → Features exhibit non-sparse behavior'
        self._info(f'{m} → Features selected by ElasticNetCV.')

        return False
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, X:pDF, y:pSe, subset:list=None):
        """
        Logistic Regression Sparse Features Basis choice based on AIG Methodology.

        Parameters
        ----------
        X : Dataframe of exogenous features.
        y : Series of target variable.
        subset : Optional subset of features in dataset X.
        """
        self._set_init()
        self._set_fes(X, y, subset)
        self.dataset.set_data(X, y)
        self._init_model_stats(y)

        for i in range(len(self._fes)):
            self._forward_adding()            
            self._update_removed()
            self._backward_prune()
            self._update_removed()

            if not self._is_continuation(i):
                return self
            
            if not self._is_sparse(i):
                self._finish_convergence()
                return self
        
        return self
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SelectorBase:

    implemented_estimators = {
        'logit':  Logit,
        'logreg': LogisticRegression,
    }
    
    result: list = None
    optim_path: dict = None

    def __init__(self, 
        solver:  Literal['logit', 'logreg'], 
        setup:   dict,
    ):
        self._set_setup(solver, setup)

    def _get_solver(self, solver:str) -> LogReg:
        return self.implemented_estimators[solver]        
    
    def _set_setup(self, solver:str, setup:dict) -> None:
        self._solver = solver
        self.setup = {'estimator': self._get_solver(solver), 'setup': setup}

    def get_estimator(self) -> LogReg:
        return self.setup['estimator']
    
    def get_setup(self, key:str=None) -> dict:
        s = self.setup['setup']
        if key is None:
            return s
        return s[key]
    
    def _is_logit(self) -> bool:
        return self.get_estimator() is Logit
    
    def _get_setup(self) -> dict:
        s = self.get_setup()
        setup = dict(
            solver   = s['solver'], 
            tol      = s['tol'],
            max_iter = s['max_iter'],
        )
        if self._is_logit():    
            return setup
        return setup | dict(
            C            = np.inf,  # no restriction for selection use-case needed
            random_state = s['random_state']
        )

    def get_estimator_instance(self) -> LogReg:
        e = self.get_estimator()
        s = self._get_setup()
        return e(**s, fit_intercept=FIT_INTERCEPT)

    def _get_worker_params(self):
        s = self.get_setup()
        return dict(
            alpha        = s['alpha'],
            cv           = s['cv'],
            tol          = s['tol'],
            random_state = s['random_state'],
            verbose      = s['verbose'],
            model  = self.get_estimator_instance()
        )

    def _fit(self, X:pDF|NDArray, y:pSe, subset:list=None):
        X_, _  = DataUtils.align_data_with_fes(X, subset)

        params = self._get_worker_params()
        worker = FitWorker(**params).fit(X_, y)

        self.result     = worker.result
        self.optim_path = worker.optim_path

        return self
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SparseSelector(SelectorBase, PredictBase):
    """
    Logistic Regression WOE Features Base Subset search based on AIG Methodology.

    Perform sparse feature selection using an iterative forward-backward 
    stepwise methodology based on Log-Likelihood significance tests.

    The algorithm iteratively adds the most significant predictor (Forward) 
    and prunes redundant predictors (Backward). A feature is retained only if 
    the change in the model's log-likelihood is statistically significant 
    given the specified `alpha` threshold, evaluated across cross-validation folds.

    Parameters
    ----------
    alpha : Significance level (alpha) for fe assessment, default=0.05.
        Statistical hypotesis significance for model's residual change
        by adding/removing a feature under the assessment.
    
    cv : Number of cross-validation folds, default=3.
        Model's residual change is assessed on cv folds; 
        0 means off → assess on train sample.
    
    max_iter : int, default=1000
        The maximum number of iterations.
    
    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are smaller or equal to
        ``tol``, the optimization code checks the dual gap for optimality and continues
        until it is smaller or equal to ``tol``.

    random_state : int, RandomState instance, default=42
        The seed of the pseudo random number generator to create CV folds.        
        When solver is logreg, rs is also use as >
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    method : {'lbfgs', 'newton-cg'}, default='newton-cg'.
        Algorithm to use in the optimization problem. Default is 'newton-cg'.
        The `method` determines which solver from scipy.optimize | sklearn
        is used, and it can be chosen from among the following strings:
        - 'newton-cg':
            - for Newton-Conjugate-Gradient when scikit as backend
            - for Newton-Raphson, when statsmodels as backend
        - 'lbfgs' for limited-memory BFGS with optional box constraints

    solver : {'logit', 'logreg'}, default='logit'.
        Solver choses backend estimator to solve the features selection problem.
        Solver choice provides ML model: model = estimator(**params).fit(X, y).

    verbose : Verbosity of the selector features assessment, default=False.

    
    Statistical Rationale
    =====================

    Computational Design
    --------------------
    To bound runtime, the selector operates on a constructed sample of
    at most 10,000 rows regardless of input size. The sample is built
    as a balanced (50:50) subset drawn directly from the full dataset:

    - If total positives ≥ 5,000:
        sample = 5,000 positives + 5,000 negatives = 10,000 rows
    - If total positives < 5,000:
        sample = all positives + fill to 10K by negatives
        (total = n_pos + n_neg → 10,000 rows)

    If the input has fewer than 10,000 rows, all observations are used
    (no subsampling).

    This balanced construction guarantees a maximum 50:50 ratio under
    all circumstances, maximising the number of positive events in the
    sample while maintaining class balance for numerical stability.

    Statistical Validity
    --------------------
    In logistic regression, altering the class sampling ratio shifts only
    the intercept; slope coefficients and their standard errors (and thus
    the 1-d.f. χ² / Wald tests used at each pruning step) are invariant
    to the sampling ratio. Feature selection is therefore unbiased.

    WOE values computed on the balanced sample differ from those on the
    natural distribution by a uniform constant ln(sampling_ratio), which
    does not affect bin ordering or discriminative ranking.

    Minimum Event Requirement
    -------------------------
    The step-wise 1-d.f. LRT at each iteration is valid as long as the
    base model (k retained coefs) is stably estimated. Practical rule:

        n_events ≥ 10 × k        (at step k)

    The absolute floor is 100 events (supports ~10 retained features).
    At the design target of 30–40 features (~120–160 free parameters
    with 5-bin WOE), 5,000 events provides 31–41 EPV — ample margin.

    If the dataset cannot meet this, the selector will still run but
    later-step p-values become anti-conservative (over-selection risk).
    The user should be warned and advised to either increase sample size
    or treat the output as a screening list to be re-validated on a
    holdout with sufficient events.

    WOE Bin Stability
    -----------------
    When the positive count per WOE bin is < 5, category-level estimates
    are unstable. Bin merging was applied to WOE encoding by the previous 
    flow step to prevent noisy signals from propagating into decisions.

    Design Intent: Signal Sparsity
    ------------------------------
    Features whose individual marginal contribution to the logit is below
    the significance threshold are eliminated by design. This is the
    objective of the sparse selector — to retain only features with a
    detectable standalone signal.

    Sparsity Bound
    --------------
    The sparse selector is designed to retain 30–40 features at most.
    This is not an arbitrary cap but a structural constraint: with an
    average of 5 WOE bins per feature (4 free parameters after
    dropping the reference), 40 features imply ~160 coefficients,
    which at 5,000 events yields 31 events-per-parameter — well above
    the 10 EPV stability threshold.

    Selecting beyond ~50 features is self-contradictory: the output
    is no longer sparse, and the model enters a regime where L2
    regularization (dense selector) is the statistically appropriate
    tool. If the step-wise procedure has not converged by 50 retained
    features, the user should be warned that the data exhibit
    distributed signal and directed to the dense selector.

    Known Limitation: Distributed Signal
    ------------------------------------
    In datasets where predictive signal is fragmented across many features
    — each individually below threshold — the step-wise procedure will
    terminate early, before the combined effect becomes detectable.

    This is not a failure but a diagnostic signal: the data exhibit a
    dense, distributed structure rather than a sparse one. In this case:

    1. Switch to the dense selector (L2-regularized WOE-logit), which
        retains most features with shrunken coefficients.
    2. Interpret the early-stop as confirmation that no single feature
        dominates — the signal is collective, not concentrated.

    Both selectors are valid; the sparse selector has a different
    inductive bias and is not the appropriate tool for dense-signal
    regimes.    
    """

    def __init__(self, 
        alpha:      float = 0.05, 
        cv:           int = 3, 
        max_iter:     int = 1000,
        tol:        float = 1e-4,
        random_state: int = 42,
        method:   Literal['lbfgs', 'newton-cg'] = 'newton-cg',
        solver:   Literal['logit', 'logreg']    = 'logit',
        verbose:     bool = False,
        **kwargs
    ):  
        setup = dict(
            alpha        = alpha,
            cv           = cv,
            max_iter     = max_iter,
            random_state = random_state,
            tol          = tol,
            solver       = method,
            verbose      = verbose
        ) | kwargs

        super().__init__(solver, setup)    

    def _set_result_model(self, X:pDF|NDArray, y:pSe) -> None:
        X_, self.result = DataUtils.align_data_with_result(X, self.result)
        self._set_predict_model(X_, y)

    def fit(self, X:pDF|NDArray, y:pSe, subset:list=None):
        """
        Logistic Regression WOE Features Basis choice based on AIG Methodology.

        Parameters
        ----------
        X : Dataframe of exogenous features.
        y : Series of target variable.
        subset : Optional subset of features in dataset X.
        """
        self._fit(X, y, subset)
        self._set_result_model(X, y)
        return self
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬