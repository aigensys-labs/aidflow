"""
Feature Ablation - Model Performance Degradation Assessment.

This module implements the feature ablation methodology for assessing
the individual contribution of each feature to the model's predictive
power. It measures the degradation in Gini (discrimination) when a
single feature is removed from the model, providing a rigorous,
model-agnostic importance ranking.

**Methodology:**

Two solver strategies are provided depending on the model type:

1. **GLM (Logistic Regression) — Analytic Solver**
    Uses the sub-Hessian inversion (Schur complement of the Fisher 
    Information matrix) combined with a 1-step Newton-Raphson update 
    to approximate the  exact refit of the logistic model when feature j 
    is ablated.

    This eliminates full model refitting loops, reducing execution time 
    from O(N * p * iterations) to O(p^2) per feature. One or two Newton-
    Raphson steps produce results nearly identical to a full MLE refit.

    Statistical basis:
    The 1-step approximation for the reduced feature parameter vector is:

    β_reduced ≈ β_full, -j + (X_r^T W X_r)^(-1) X_r^T (y - p_full) + O(1/N)

    where W = diag(p_full * (1 - p_full)), X_r is the design matrix without 
    feature j, and β_full, -j is the full model coefficient vector excluding β_j.

    Reference:
    Harrel, D. (2015). Regression Modeling Strategies, 2nd ed. Chapter 5
    (model updating, coefficient stability).

2. **GBM (LightGBM / XGBoost) — Resampling Solver**
   Since GBM has no closed-form solution for coefficient updates,
   the ablation is performed by **ablated refits**: for each
   feature, the model is retrained on (p-1) features, and the Gini 
   drop is measured. This is computationally expensive (p refits)
   but produces valid, unbiased importance estimates.

The significance of the Gini difference is tested using the
**DeLong test** [1], which correctly accounts for the correlation
between predictions from the full and ablated models (they share
the same test observations).
"""

from ...deps.common     import  (
                                ABC, 
                                abstractmethod
                                )
from ...deps.data       import  np, pDF, pSe
from ...deps.types      import  NDArray

from ...core.factory    import  DictClass

from ..metrics          import  Gini
from ..tests            import  TestModelDiff
from .models            import  (
                                Logit, 
                                LGB, XGB, 
                                ModelUtils
                                )

GLM = Logit
GBM = LGB | XGB


class LBL(DictClass):
    """Column labels for the Feature Ablation result table."""
    F  = 'Feature'
    A  = 'Ablation_ΔGini'
    R  = 'Orig_ΔGini(%)'
    IS = 'SignificantΔ'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# MODEL ASSESSMENT → FEATURE ABLATION
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class _AblationSolver(ABC):
    """
    Abstract base class for feature ablation solvers.

    Defines the common algorithm structure:
    1. Verify model compatibility.
    2. Initialize the computation (set up design matrices, covariances, etc.).
    3. Iterate over each feature, evaluating the Gini drop.
    4. Package results into a standardized DataFrame.

    **GLM Approach** (analytic):
        - Pre-computes the full model's covariance matrix.
        - Uses the deletion formula (Schur complement) + Newton steps
          to approximate the refit with one feature dropped.
        - O(1) per feature (no refit) → fast for large p.

    **GBM Approach** (resampling):
        - For each feature, refits the model on (p-1) features.
        - O(p) refits → expensive but statistically valid.

    Parameters
    ----------
    model : Model
        Fitted model instance of supported Model type.

    X : pDF
        Training matrix of shape (N, p).

    y : pSe
        Training binary target vector (N,).

    eval_set : tuple[list[pDF, pSe], ...]
        Evaluation set pairs - matrix of shape (Z, p) with binary target vector (Z,).
        Validation / Test samples.

    alpha : float = 0.01
        Significance threshold for model's residual change 
        by adding/removing a feature under the assessment.
    """
    KIND: GLM | GBM
    
    def __init__(self, 
        model: GLM | GBM, 
        X: pDF | NDArray, 
        y: pSe | NDArray, 
        eval_set: tuple[list[pDF | NDArray, pSe | NDArray], ...],
        alpha: float = 0.05
    ):
        self._verify(model, eval_set)

        self.model = model
        self.X = X
        self.y = y
        self.alpha = alpha

        self.store = {}
        self.update_eval_set(eval_set)
    
    def update_eval_set(self, 
        eval_set: tuple[list[pDF | NDArray, pSe | NDArray], ...]
    ) -> None:
        """
        Replace the evaluation set for subsequent ablation assessments.

        This is the internal "hot swap" mechanism that allows re-evaluating
        the ablation on a different validation/test set without rebuilding
        the solver's computational state (reduced betas, cached estimators,
        covariance matrices, etc.).

        **Performance Note:**
        - **GLM solver**: The reduced beta vectors  are computed from the 
          training data and are independent of the eval set. Updating the 
          eval set is essentially free — the next `compute()` call will 
          simply predict on the new data using the already-computed betas. 
          O(p × Z) matrix multiplications, where Z is the eval set size.

        - **GBM solver**: The reduced estimators are computed from the training 
          data and are independent of the eval set. The estimators remain cached 
          and `update()` is significantly faster (prediction only, no refit).

        Parameters:
            eval_set (tuple[list[pDF | NDArray, pSe | NDArray], ...]):
                The new evaluation set pairs. Each pair is [X_eval, y_eval].
                Must contain at least one pair.

        Example:
            >>> # Initial assessment on validation set:
            >>> result_val = ablation.evaluate(X_train, y_train, eval_set=[(X_val, y_val)])
            >>>
            >>> # Re-assess on a different OOT test set (no rebuild):
            >>> result_oot = ablation.update(eval_set=[(X_oot, y_oot)])
        """        
        self.eval_set = eval_set

    def compute(self, subset:list=None) -> pDF | list[pDF]:
        """
        Execute the feature ablation assessment.

        Iterates over all (or a subset of) features, computing the
        Gini drop and DeLong significance test for each. Returns a
        ranked table sorted by ablation magnitude (largest drop first).

        Parameters:
            subset (list | None): Optional list of feature names to
                evaluate. If None, all features in X are assessed.

        Returns:
            pDF | list[pDF]: A DataFrame with columns:
                - Feature: The feature name.
                - Ablation_ΔGini: Absolute Gini change (negative = drop).
                - Orig_ΔGini(%): Ratio of ablated Gini to full Gini.
                - SignificantΔ: Boolean flag (True if p < alpha AND Gini dropped).

            Sorted by Ablation_ΔGini (ascending — most important first).

        Example:
            >>> solver = _GLMAnalyticSolver(model, X, y)
            >>> result = solver.compute()
            >>> result.head()
              Feature  Ablation_ΔGini  Orig_ΔGini(%)  SignificantΔ
            0  income      -0.0342        0.9654        True
            1  age         -0.0128        0.9872        True
            2  region      -0.0021        0.9979        False
        """
        self._set_init()
        self._set_fes(subset)
        self._set_pipeline_attrs()
        
        if not self._is_eval():
            return self._get_nan_result()

        res = []
        for X,y in self.eval_set:
            self._set_init()
            self._set_eval_set(X, y)

            for x in self._fes:
                g, z = self._fe_eval(x)
                self._add_result(x, g, z)

            res.append( self._get_result() )

        return res if len(self.eval_set)  > 1 else res[0]

    @abstractmethod
    def _fe_eval(self, x:str) -> tuple[float, float]:
        ...

    @abstractmethod
    def _set_pipeline_attrs(self) -> None:
        ...
        
    @staticmethod
    def _align_X(X:pDF) -> pDF:
        return X        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _set_eval_set(self, X:pDF, y:pSe) -> None:
        self.X_eval = self._align_X(X)
        self.y_eval = y        
        self.y_eval_proba = self.model.predict_proba(X)[:,1]
        self.model_eval_gini = self._calc_g(self.y_eval_proba) 
    
    def _set_fes(self, subset:list) -> None:
        if isinstance(self.X, pDF):
            col = self.X.columns.to_list()
        else:
            col = list(range(len(self.X)))
        if not subset:
            fes = col
        else:
            fes = [x for x in subset if x in col]

        self._fes = fes

    def _is_eval(self) -> bool:
        if len(self._fes) < 2:
            return False
        return True         
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _calc_z(self, p_idx:pSe) -> float:
        """
        Compute the DeLong z-score for comparing two correlated Gini estimates.

        The DeLong test correctly accounts for the correlation between
        the full-model predictions and the ablated-model predictions
        (they share the same test observations), which the naive
        independent z-test would ignore.

        Parameters:
            y (pSe): True binary target.
            p_full (pSe): Predictions from the full model.
            p_idx (pSe): Predictions from the ablated model.

        Returns:
            float: The DeLong z-score. |z| > 1.96 implies p < 0.05.
        """        
        return TestModelDiff(self.y_eval, self.y_eval_proba, p_idx).delong(z_score=True)

    def _calc_g(self, p_idx:pSe) -> float:
        """
        Compute the Gini coefficient for a set of predictions.

        Parameters:
            y (pSe): True binary target.
            p_idx (pSe): Predicted probabilities.

        Returns:
            float: The Gini coefficient (2 × AUC - 1).
        """        
        return Gini.fast_calc(self.y_eval, p_idx)
        
    def _get_stats(self, y_idx_proba:pSe) -> tuple[float, float]:
        return (
            self._calc_g(y_idx_proba), 
            self._calc_z(y_idx_proba)
        )    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _set_init(self) -> None:
        self.result = []

    def _is_significant(self, z:float, a:float=None) -> bool:
        """
        Determine if a feature's ablation is statistically significant.

        A feature is flagged as significant if:
        1. The DeLong test p-value < alpha (the Gini change is not due to chance).
        2. (Optional) The Gini actually decreased (a < 0), confirming
           the feature contributed positively to the model.

        Parameters:
            z (float): The DeLong z-score.
            a (float | None): The absolute Gini change (negative = drop).
                If provided, must also be negative for significance.

        Returns:
            bool: True if the ablation is statistically significant.
        """        
        if z is None:
            return False
        t = TestModelDiff.p_value(z) < self.alpha 
        if a is not None:
            return t and ( a < 0 )
        return t

    def _add_result(self, x:str, g:float, z:float) -> None:
        m = self.model_eval_gini
        a = g - m
        self.result.append({
            LBL.F:  x,
            LBL.A:  a, 
            LBL.R:  round(g / m, 4),
            LBL.IS: self._is_significant(z, a)
        })

    def _get_result(self) -> pDF:
        """
        Assemble the final ablation results DataFrame.

        If no results were computed (e.g., only 1 feature), fills
        with zeros and no significance flags.

        Returns:
            pDF: The ablation table sorted by Ablation_ΔGini (ascending).
        """        
        if not self.result:
            for x in self._fes:
                self._add_result(x, 0, None)
        return pDF(self.result).sort_values(LBL.A).reset_index(drop=True) 

    def _get_nan_result(self) -> pDF | list[pDF]:
        self.model_eval_gini = np.nan
        r = self._get_result()
        l = len(self.eval_set)
        if l == 1:
            return r
        return [r]*l
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def _verify(cls, model, eval_set):
        if not isinstance(model, cls.KIND):            
            raise TypeError('AIG SAFETY'
                f'\n► Provided {type(model).__name__} is not supported.'
                f'\n► Supported models defined by {cls.KIND}'
            )
        if not eval_set:
            raise ValueError('eval_set must contain at least one (X, y) pair.')        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
        
class _GLMAnalyticSolver(_AblationSolver):
    """
    Analytic feature ablation solver for GLM (Logistic Regression) models.

    **Method:**
        Instead of refitting the model p times (once per feature), this
        solver uses the **deletion formula** (Schur complement) to
        compute the reduced model's coefficient vector in closed form,
        then applies 2 Newton-Raphson steps to correct for the
        non-linearity of the logistic link function.

    **Mathematical Basis:**
        For logistic regression, the Fisher information matrix is:
            I = X^T W X, where W = diag(p_i(1-p_i))

        When dropping feature j, the Schur complement gives:
            V_reduced = V_full[-j,-j] - V_full[-j,j] V_full[j,j]^(-1) V_full[j,-j]

        This is the exact inverse of the reduced Fisher information.
        Starting from this as an initial approximation, 2 Newton steps
        on the reduced model recover the exact MLE to machine precision
        (for well-conditioned problems).

    **Advantages over refit:**
        - 10-20x faster than refit (sample size dependent).

    **Limitations:**
        - Requires the full model to be well-conditioned
          (not perfectly separated).
        - The Newton steps may need more iterations for strongly
          collinear features (increase `n_steps` to 3+).
        - Assumes the full model's coefficients are the global MLE.
    """
    KIND = GLM

    def _set_pipeline_attrs(self):
        self._set_y_proba()
        self._set_X()
        self._set_covariance()
        self._set_coef()

    @staticmethod
    def _align_X(X:pDF) -> NDArray:
        return np.concatenate(
            [np.array([[1]]*len(X)), np.array(X)], axis=1
        )      
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_y_proba(self) -> None:
        """Compute and store the full model's predicted probabilities."""        
        self.y_proba = self.model.predict_proba(self.X)[:,1]
    
    def _set_X(self) -> None:
        """
        Sets self.X_design :
        NDArray, (N, p+1) array `including intercept` column (+1) at the first pos. 
        """        
        self.X_ = self._align_X(self.X)

    def _set_coef(self) -> None:
        """Assemble the full coefficient vector (intercept + features)."""        
        self.coef = np.append(self.model.intercept_, self.model.coef_)

    def _set_covariance(self) -> None:
        """
        Compute the full model's parameter covariance matrix.

        The covariance matrix of the MLE in logistic regression is:
            V = (X^T W X)^(-1)
        where W = diag(p_i(1-p_i)) is the diagonal weight matrix.

        Uses the pseudo-inverse for numerical stability (handles
        near-collinearity gracefully without raising errors).
        """
        p = self.y_proba
        X = self.X_
        w = p * (1.0 - p)   # W = p*(1-p)        
        H = (X.T * w) @ X   # Fast X^T * W * X
        self.cov_matrix = np.linalg.pinv(H)        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _align_idx(self, x: str) -> int:
        idx = self._fes.index(x)
        return idx + 1  # +1 for intercept
        
    def _exclude_fe(self, X:pDF, x:str) -> NDArray:
        idx = self._align_idx(x)
        return np.delete(X, idx, axis=1)
    
    def _set_reduced_betas(self,        
        x: str,
        n_steps: int = 2,
    ) -> tuple[NDArray, NDArray]:
        """
        Approximate the exact refit of the logistic model when dropping
        a single feature, using the deletion formula + Newton steps.

        **Algorithm:**
        1. Apply the Schur complement to get the exact inverse of the
           reduced Fisher information: V_reduced.
        2. Initialize β_reduced from the full model's coefficients
           (excluding the dropped feature).
        3. Perform `n_steps` Newton-Raphson iterations on the
           reduced model to converge to the exact MLE.

        **Convergence:**
        - 1 step: ~20% faster, adequate for non-collinear features.
        - 2 steps (default): Machine precision for most practical cases.
        - 3+ steps: Recommended for strongly collinear feature sets.

        Parameters:
            drop_idx (int): The index of the feature to drop
                (0-based, excluding intercept).
            n_steps (int): Number of Newton-Raphson correction steps.
                Default is 2 (sufficient for most models).

        Returns:
            tuple[NDArray, NDArray]:
                - X_reduced: The design matrix without the dropped column.
                - beta_reduced: The converged coefficient vector
                  (intercept + remaining features).
        """
        if self.store.get(x) is not None:
            return        
        
        idx = self._align_idx(x)

        p = len(self.coef)
        mask = np.ones(p, dtype=bool)
        mask[idx] = False

        # Deletion formula: exact H_reduced^{-1}
        V_rr = self.cov_matrix[mask][:, mask]
        V_rj = self.cov_matrix[mask, idx]
        V_jj = self.cov_matrix[idx, idx]

        H_inv = V_rr - np.outer(V_rj, V_rj) / V_jj

        # Newton steps on reduced model
        X_r  = self._exclude_fe(self.X_, x)
        beta = self.coef[mask].copy()

        for _ in range(n_steps):
            eta   = X_r @ beta
            p_hat = 1.0 / (1.0 + np.exp(-eta))
            score = X_r.T @ (self.y - p_hat)
            beta  = beta + H_inv @ score

        self.store[x] = beta
    
    def _predict_proba(self, x: str) -> NDArray:
        """
        Compute predicted probabilities from a reduced model on eval set.

        Parameters:
            X_r (NDArray): Reduced design matrix (N, p).
            params (NDArray): Reduced coefficient vector (p,).

        Returns:
            NDArray: Predicted probabilities (N,).
        """
        b = self.store[x]
        X = self._exclude_fe(self.X_eval, x)
        lin = np.dot(X, b)
        lin = np.asarray(lin)
        return 1 / (1 + np.exp(-lin))    
    
    def _fe_eval(self, x:str|int) -> tuple[float, float]:
        """
        Evaluate the ablation impact of a single feature (GLM analytic).

        Uses the deletion formula + Newton steps to compute the
        reduced model's predictions, then measures the Gini drop
        and DeLong significance.

        Parameters:
            x (str|int): Feature name to exclude.

        Returns:
            tuple[float, float]: (gini_ablated, delong_z_score).
        """        
        self._set_reduced_betas(x)
        p = self._predict_proba(x)
        return self._get_stats(p)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class _GBDTResamplingSolver(_AblationSolver):
    """
    Resampling solver for GBM (LightGBM / XGBoost) ablation.

    **Method:**
        Since GBM has no closed-form solution for coefficient updates,
        the ablation is performed by **refitting the model** on
        (p-1) features. For each feature:

        a. Train the ablated model on training set 
            (p-1 features, excluding feature x).
        c. Evaluate both on eval set.
        d. Compute Gini and DeLong z-score.

    **Computational Cost:**
        - p features refits.
        - This is the primary reason the GLM analytic solver is
          preferred for iterative development.

    **Early Stopping:**
        - Disabled for ablated refits to ensure consistent tree depth.
        - The tree count is locked to the full model's optimal iteration
          (from early stopping) to maintain comparability.

    Parameters:
        model (GBM): A fitted LightGBM or XGBoost model.
        X (pDF): Feature matrix.
        y (pSe): Binary target.
        alpha (float): Significance level (default 0.05).
    """
    KIND = GBM
          
    def _set_pipeline_attrs(self) -> None:
        pass
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _store_annot(self) -> None:
        self.store: dict[str, GBM] = {}

    def _get_setup(self, X:pDF) -> dict:
        """
        Prepare the estimator configuration for ablated refits.

        Key adjustments vs. the original model:
        1. **Early stopping disabled**: Ensures the ablated model
           trains for the same number of iterations as the full model,
           making the comparison fair.
        2. **Tree count locked**: Set to the full model's optimal
           iteration (from early stopping) to maintain comparability.
        3. **Categorical features preserved**: Ensures the same
           encoding is used in the ablated refit.

        Parameters:
            X (pDF): The feature matrix (used to filter categorical
                features to those present in the reduced set).

        Returns:
            dict: The estimator configuration dictionary.
        """        
        s = self.model.get_setup()
        c = self.model.get_categorical_feature()

        # 1 Disable early stopping for ablated refits
        s["early_stopping_round"] = None

        # 2 Lock tree count to optimal iteration from full fit
        m = self.model._model
        
        if isinstance(self.model, LGB) and m.best_iteration_:
            s["n_estimators"] = m.best_iteration_
        elif getattr(self.model.booster_, "best_iteration", 0) > 0:
            s["n_estimators"] = self.model.booster_.best_iteration        

        if not c:
            return s
        
        c  = [x for x in c if x in X.columns]        
        return s | dict(categorical_feature = c)

    def _get_estimator(self, X:pDF) -> GBM:
        """
        Instantiate a new estimator with the ablation configuration.

        Parameters:
            X (pDF): The (possibly reduced) feature matrix.

        Returns:
            GBM: A new, unfitted estimator instance.
        """        
        e = type(self.model)
        p = self._get_setup(X)
        return e(**p)

    def _exclude_fe(self, X:pDF, x:str) -> pDF:
        fes = [f for f in self._fes if f!=x]
        return X[fes]
    
    def _set_estimator(self, x) -> GBM:
        """
        Fit (or retrieve cached) ablated estimator for feature x.

        Fits the model on the full training set with feature x excluded.
        Results are cached in `self.estimator` to avoid redundant refits
        when multiple eval_sets are provided.

        Parameters:
            x (str): The feature name to exclude from the training set.
        """        
        if self.store.get(x) is not None:
            return
        
        X_r = self._exclude_fe(self.X, x)
        est = self._get_estimator(X_r)        
        self.store[x] = est.fit(X_r, self.y)

    def _predict_proba(self, x:str) -> NDArray: 
        """
        Predict on the eval set using the cached ablated estimator.

        Parameters:
            x (str): The feature name (identifies the cached estimator).

        Returns:
            NDArray: Predicted probabilities on the current eval_set.
        """      
        m = self.store[x]
        X = self._exclude_fe(self.X_eval, x)
        return m.predict_proba(X)[:,1]
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _fe_eval(self, x:str) -> tuple[float, float]:
        """
        Evaluate the ablation impact of a single feature (GBM resampling).

        Refits the model on (p-1) features and measures the Gini drop with 
        DeLong significance.

        Parameters:
            x (str): The feature name to exclude.

        Returns:
            tuple[float, float]: (gini_ablated, z_score).
        """        
        self._set_estimator(x)
        p = self._predict_proba(x)
        return self._get_stats(p)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeatureAblation:
    """
    Feature Importance Assessment.

    Provides a unified interface to measure the individual contribution
    of each feature to a model's predictive power by quantifying the
    performance degradation (Gini drop) when the feature is removed.

    **Supported Models:**
    - **GLM (Logistic Regression)**: Uses the analytic deletion formula
      (fast, no refit). Recommended for iterative development.
    - **GBM (LightGBM, XGBoost)**: Uses p - refits (accurate but
      slow). Use for final validation only.

    **Output Interpretation:**
    - `Ablation_ΔGini`: The absolute change in Gini when the feature
      is removed. More negative = more important feature.
    - `Orig_ΔGini(%)`: The ratio of ablated Gini to full Gini.
      0.95 means the model retains 95% of its discrimination power
      without this feature.
    - `SignificantΔ`: Whether the Gini drop is statistically
      significant (DeLong test, p < alpha). Features with
      `SignificantΔ = False` may be candidates for removal without
      meaningful impact on model performance.

    Parameters:
        model (GLM | GBM): A fitted model instance (Logit, LGB, or XGB).
        alpha (float): Significance level for the DeLong test.
            Default is 0.05. Features with p < alpha are flagged
            as "significant."

    Example:
        >>> from aidflow.ml.models.ablation import FeatureAblation
        >>>
        >>> ablation = FeatureAblation(
        ...     model=fitted_logit,
        ...     alpha=0.01,
        ... )
        >>>
        >>> result = ablation.evaluate(X=train_X, y=train_y)
        >>> result
        >>> #    Feature  Ablation_ΔGini  Orig_ΔGini(%)  SignificantΔ
        >>> # 0  income       -0.0452       0.9548         True
        >>> # 1  age          -0.0218       0.9782         True
        >>> # 2  region       -0.0034       0.9966         False
        >>>
        >>> # Assess a subset of features:
        >>> result = ablation.evaluate(X=train_X, y=train_y, subset=['income', 'age'])

    **Regulatory Use Cases:**
    - **Model Documentation**: Provides objective evidence of which
      features drive the model's decisions (SR 11-7, EU AI Act Art. 13).
    - **Feature Justification**: Supports the business case for
      collecting specific data points (GDPR data minimization principle).
    - **Pruning Evidence**: Quantifies the cost of removing a feature,
      enabling data-driven decisions about data collection scope.
    - **Monitoring Baseline**: Establishes the expected contribution
      of each feature, against which PSI and performance drift can
      be monitored.        
    """

    def __init__(self, 
        model: GLM | GBM, 
        alpha: float = 0.05, 
    ):
        self.model = model
        self.alpha = alpha

    def evaluate(self, 
        X: pDF | NDArray, 
        y: pSe | NDArray, 
        eval_set: tuple[list[pDF | NDArray, pSe | NDArray],...], 
        subset: list = None
    ) -> pDF | list[pDF]:
        """
        Execute the feature ablation assessment.

        Automatically selects the appropriate solver based on the
        model type:
        - GLM → `_GLMAnalyticSolver` (fast, analytic).
        - GBM → `_GBDTResamplingSolver` (p-refits, slower).

        Parameters:
            X (pDF | NDArray): The feature matrix used in the original model fit.
                Must contain all features of the model.
            y (pSe | NDArray): The binary target used in the original model fit.
            eval_set : tuple[list[pDF | NDArray, pSe | NDArray], ...]
                Evaluation set pairs - matrix of shape (Z, p)         
                with binary target vector (Z,). Validation / Test samples.
            subset (list | None): Optional list of feature names to
                evaluate. If None, all features are assessed.
                Useful for focused analysis on a shortlist.
                
        Returns:
            pDF | list[pDF]: The ablation results table with columns:
                - `Feature`: Feature name.
                - `Ablation_ΔGini`: Absolute Gini change (negative = drop).
                - `Orig_ΔGini(%)`: Ratio of ablated to full Gini.
                - `SignificantΔ`: True if p < alpha AND Gini decreased.

            Sorted by `Ablation_ΔGini` ascending (most important first).
            The `list` is returned when `multiple eval_set pairs` is provided.

        Example:
            >>> ablation = FeatureAblation(model, alpha=0.01)
            >>> result = ablation.evaluate(X, y)
            >>>
            >>> # Only assess top 10 candidates:
            >>> candidates = ['income', 'age', 'region', 'dti', 'util']
            >>> result = ablation.evaluate(X, y, subset=candidates)
        """
        if ModelUtils.is_lr(self.model):
            Solver = _GLMAnalyticSolver
        else:
            Solver = _GBDTResamplingSolver

        self.solver = Solver(
            model = self.model, 
            X = X, 
            y = y, 
            eval_set = eval_set,
            alpha = self.alpha, 
        )
        return self.solver.compute(subset)

    def update(self,
        eval_set: tuple[list[pDF | NDArray, pSe | NDArray],...], 
        subset: list = None
    ) -> pDF | list[pDF]:
        """
        Re-evaluate the feature ablation on a new evaluation set.

        This is the **fast path** for assessing ablation impact on a
        different validation or test sample. The solver's computational
        state (reduced betas for GLM, fitted estimators for GBM) is
        preserved from the initial `evaluate()` call, making this
        operation significantly cheaper than a full re-assessment.

        Parameters:
            eval_set (tuple[list[pDF | NDArray, pSe | NDArray], ...]):
                The new evaluation set pairs. Each pair is [X_eval, y_eval].
                Typically a different OOT or holdout sample than the one
                used in the initial `evaluate()` call.
            subset (list | None): Optional list of feature names to
                evaluate. If None, all features are assessed.
                Useful for focused re-assessment on a shortlist.

        Returns:
            pDF | list[pDF]: The ablation results table (same format as
                `evaluate()`). A `list` is returned when multiple
                eval_set pairs are provided.

        Example:
            >>> # Step 1: Initial assessment on validation set:
            >>> result_val = ablation.evaluate(
            ...     X=X_train, y=y_train,
            ...     eval_set=[(X_val, y_val)]
            ... )
            >>>
            >>> # Step 2: Confirm on OOT test set (fast — no rebuild):
            >>> result_oot = ablation.update(
            ...     eval_set=[(X_oot, y_oot)]
            ... )
            >>>
            >>> # Step 3: Focused re-assessment on top 5 features only:
            >>> top5 = result_val.nlargest(5, 'Ablation_ΔGini')['Feature'].tolist()
            >>> result_focused = ablation.update(
            ...     eval_set=[(X_oot, y_oot)],
            ...     subset=top5
            ... )
            >>>
            >>> # Step 4: Multi-set assessment (validation + OOT):
            >>> results = ablation.update(
            ...     eval_set=[(X_val, y_val), (X_oot, y_oot)]
            ... )
            >>> # results[0] → validation ablation table
            >>> # results[1] → OOT ablation table
        """        
        self.solver.update_eval_set(eval_set)
        return self.solver.compute(subset)        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬