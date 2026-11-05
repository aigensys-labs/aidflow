"""
Model Adaptation Layer providing a standardized API across diverse ML libraries.

This module implements a unified wrapper system that homogenizes various external 
estimators—including Scikit-Learn, StatsModels, XGBoost, and LightGBM—into a 
consistent AIDflow interface. By inheriting from `ModelBase`, disparate library 
implementations are brought under a single API contract, ensuring that fitting, 
prediction, and attribute access remain identical regardless of the underlying 
computational engine.

The primary objective is to decouple the framework's orchestration logic from 
the specific idiosyncrasies of external ML packages. This abstraction allows for 
seamless switching between solvers, simplifies hyperparameter management through a 
unified setup mechanism, and ensures full compatibility with Scikit-Learn's 
ecosystem while providing deep access to original estimator instances when required.
"""

from ...deps.common     import  inspect, io
from ...deps.data       import  np, pd, pDF, pSe
from ...deps.plots      import  plt
from ...deps.display    import  (
                                display, 
                                SVG, 
                                HTML
                                )
from ...deps.ml.base    import  (
                                BaseEstimator, 
                                ClassifierMixin,                                
                                )
from ...deps.ml.glm     import  (
                                LogitSM,
                                LassoCV,
                                ElasticNetCV,
                                LogisticRegressionCV,
                                LogisticRegression,
                                )
from ...deps.ml.gbm     import  (
                                LGBMClassifier,
                                XGBClassifier
                                )
from ...deps.ml.diag    import  (
                                summary_plot,
                                TreeExplainer,
                                LinearExplainer,
                                )
from ...deps.ml.stats   import  (
                                log_loss, 
                                roc_auc_score,
                                mean_squared_error, 
                                r2_score
                                )
from ...deps.ml.dim     import  OrdinalEncoder

from ...deps.types      import  (
                                Any,
                                Literal, 
                                NDArray, 
                                ArrayLike,                                
                                )
from ...core.interop    import  Pd, Arrow
from ...utils.builtins  import  Misc

from .callbacks         import  ReduceLROnPlateau


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# CLASSIFIERS BASE SYNC
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ModelBase(ClassifierMixin, BaseEstimator):
    """Models Base Class to provide full API compatibility with scikit API."""
        
    implemented_estimators: dict
    _model: BaseEstimator = None
    _features: list = None

    def __getattr__(self, name:str):
        return getattr(self._model, name)
    
    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, '_model') is not None
    
    def _set_req_params(self, local_vars:dict) -> None:
        for k, v in local_vars.items():
            if k != "self":
                setattr(self, k, v)

    def _set_req_props(self, X:pDF|NDArray, y:pSe) -> None:        
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]        
        self.feature_names_in_ = self._features

    def _set_features(self, X:pDF|NDArray) -> None:
        if isinstance(X, np.ndarray):
            fes = list(range(X.shape[1]))
        else: 
            fes = X.columns.to_list()
        self._features = fes

    def _get_solver(self, solver:str) -> Any:
        return self.implemented_estimators[solver]

    def _set_setup(self, solver:str, setup:dict) -> None:
        self._solver = solver
        self.setup = {'estimator': self._get_solver(solver), 'setup': setup}

    def get_estimator(self) -> Any:
        return self.setup['estimator']
    
    def get_setup(self) -> dict:
        return self.setup['setup']

    def get_estimator_instance(self, **kwargs) -> Any:
        e = self.get_estimator()
        s = self.get_setup() | kwargs
        return e(**s)
    
    def get_model(self) -> Any:
        return self._model

    @property
    def features(self) -> list|None:
        """Model Exogenous Features."""
        return self._features    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# CLASSIFIERS VIEW SYNC
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ModelView:
    
    @staticmethod
    def _plot_format(fig: plt.Figure, ax: plt.Axes) -> SVG:
        """Summary SHAP beeswarm plot graphic enhancement."""

        # plt.title("SHAP Summary (Impact on Log-Odds)")
        ax.tick_params(labelsize=9.5)   # Feature names font size
        ax.set_xlabel("SHAP value: impact on model output (Log-Odds)", fontsize=10)
                
        cb_ax = fig.axes[1]             # Colorbar> right-side 
        cb_ax.tick_params(labelsize=10) # High-Low
        cb_ax.set_ylabel("Feature value", fontsize=10) 

        # Dots size
        for collection in ax.collections: 
            collection.set_sizes([10])  

        # Highlight fes grid
        for line in ax.get_lines():
            if line.get_ydata()[0] == line.get_ydata()[1]:             
                line.set_linewidth(1.0) 

        # Make it readable
        buf = io.BytesIO()                  
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close(fig)

        return SVG(buf.getvalue())

    @staticmethod
    def _n_samples(X: pDF, n_max_samples: int) -> NDArray:
        n = X.shape[0]    
        s = min(n, n_max_samples)
        return np.random.default_rng(42).choice(n, size=s, replace=False)
    
    @classmethod
    def shap_plot(cls,
        X:              pDF, 
        shap_vals:      NDArray, 
        feature_names:  list = None, 
        max_display:    int  = 20, 
        max_samples:    int  = 5000,
        sort:           bool = True,
        show:           bool = True,
        **kwargs
    ) -> None | SVG:
        """Summary SHAP beeswarm plot, colored by feature values."""
        if feature_names is None and isinstance(X, pDF):
            feature_names = X.columns
        
        fig, ax = plt.subplots(dpi=100)
        dynamic_height = max(2.5, min(shap_vals.shape[1], max_display) * 0.33)

        idx = cls._n_samples(X, max_samples)

        summary_plot(
            shap_vals[idx], X.iloc[idx], 
            feature_names = feature_names,
            max_display   = max_display,             
            plot_size     = (12.5, dynamic_height),
            sort          = sort,
            show          = False,
            **kwargs
        )        

        fig = cls._plot_format(fig, ax)

        if not show:
            return fig
        display(fig)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOG-REG BASE MODELS API SYNC
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class PredictBaseLR:

    _model: LogitSM | LassoCV | ElasticNetCV | LogisticRegressionCV = None
    
    def _predict_base(self, X:pDF|NDArray, **kwargs) -> ArrayLike:
        return self._model.predict(X, **kwargs)
                
    def predict_proba(self, X:pDF|NDArray, **kwargs) -> NDArray:
        """
        Predict response variable of a model given exogenous variables.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        array
            NDArray([1-p, p]).
        """
        p = np.clip(self._predict_base(X, **kwargs), 0, 1)
        return np.column_stack([1-p, p])
    
    def predict(self, X:pDF|NDArray, **kwargs) -> ArrayLike:
        """
        Predict response variable of a model given exogenous variables.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        array
            Fitted values at exog.
        """    
        return (self._predict_base(X, **kwargs) >= 0.5).astype(int)

    def shap_values(self, X:pDF|NDArray, **kwargs) -> NDArray:
        """Model shap values by LinearExplainer."""
        explainer = LinearExplainer(self, X, **kwargs)
        return explainer.shap_values(X)
    
    def shap_plot(self, X:pDF|NDArray, **kwargs) -> None:
        """Model visualization by SHAP beeswarm plot, colored by feature values."""
        ModelView.shap_plot(X, self.shap_values(X), **kwargs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ModelBaseSM(PredictBaseLR, ModelBase):

    implemented_estimators = {
        'logit': LogitSM
    }    
    CONST = 'intercept_const_'

    def __init__(self, solver:Literal['logit'], setup:dict):
        self._set_setup(solver, setup)
                
    def _is_intercept(self) -> bool:
        return self.get_setup().get('fit_intercept', False)

    def _set_props_api(self, prm:pSe|ArrayLike) -> None:
        prm = np.asarray(prm)                
        # (b) constant was part of train data → data has prior col 'const'
        if not self._is_intercept():
            self.intercept_ = 0
            self.coef_ = prm
            return
        # (a) const was added during _fit → data does not have 'const'
        self.coef_      = prm[1:]
        self.intercept_ = prm[0]           

    def _set_props(self, X:pDF|NDArray, y:pSe) -> None:
        self._set_req_props(X, y)
        self.params_    = self._model.params
        self.llf_       = self._model.llf        
        self._set_props_api(self.params_)

    def _set_summary(self) -> None:
        self._summary = SummaryGLM(self._model).summary()   

    @classmethod
    def _add_constant(cls, X:pDF|NDArray) -> pDF|NDArray:
        c = np.full(len(X), 1.0)
        if isinstance(X, np.ndarray):
            return np.column_stack([c , X])
        return Pd.concat([pSe(c, name=cls.CONST, index=X.index), X], axis=1)
    
    def _add_intercept(self, X:pDF|NDArray) -> pDF|NDArray:
        if self._is_intercept():
            return self._add_constant(X)
        return X     

    def _get_aligned_setup(self) -> dict:
        s = {**self.get_setup()}
        s['method'] = s.pop('solver')        
        return s
    
    def _get_setup(self) -> dict:
        e = self.get_estimator()
        s = self._get_aligned_setup()
        p = list(inspect.signature(e.fit).parameters)
        return {k:v for k,v in s.items() if k in p}
    
    def _fit_model(self, X:pDF|NDArray, y:pSe) -> LogitSM:
        e = self.get_estimator()
        s = self._get_setup()
        X = self._add_intercept(X)
        f = lambda s: e(y, X).fit(**s)
        h = lambda s: f(s | dict(skip_hessian = True))
        try: # mimicking sklearn ridge-fallback ops
            return f(s)
        except np.linalg.LinAlgError:
            if s['method'] != 'newton':
                return h(s)
            s = s | dict(method = 'lbfgs')
            try:
                return f(s)
            except np.linalg.LinAlgError:
                return h(s)
            
    def _fit(self, X:pDF|NDArray, y:pSe) -> None:
        self._model = self._fit_model(X, y)
        self._set_features(X)
        self._set_props(X, y)
        self._set_summary()        
        self._model.remove_data()

    def _predict_base(self, X:pDF|NDArray, **kwargs) -> ArrayLike:
        if self._is_intercept():    # (a) const was added during _fit → data does not have 'const'
            lin = np.dot(X, self.coef_) + self.intercept_
        else:                       # (b) constant was part of train data → data has 'const'
            lin = np.dot(X, self.params_)
        lin = np.asarray(lin)
        return 1 / (1 + np.exp(-lin))
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class Logit(ModelBaseSM):
    """
    Wrap of statsmodels Logit to follow standard ML API.
    Fit the model using maximum likelihood.

    Parameters
    ----------
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.

    solver : str, optional
        The `solver` (default 'lbfgs') determines which solver from `scipy.optimize`         
        is used, and it can be chosen from among the following strings:

        - 'newton' or 'newton-cg' for Newton-Raphson, 'nm' for Nelder-Mead
        - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        - 'lbfgs' for limited-memory BFGS with optional box constraints
        - 'powell' for modified Powell's method
        - 'cg' for conjugate gradient
        - 'ncg' for Newton-conjugate gradient
        - 'basinhopping' for global basin-hopping solver
        - 'minimize' for generic wrapper of scipy minimize (BFGS by default)

    max_iter : int, optional
        The maximum number of iterations to perform.

    full_output : bool, optional
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.

    verbose : bool, optional
        Set to True to print convergence messages.

    fargs : tuple, optional
        Extra arguments passed to the likelihood function, i.e.,
        loglike(x,*args)

    callback : callable callback(xk), optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.

    retall : bool, optional
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.

    skip_hessian : bool, optional
        If False (default), then the negative inverse hessian is calculated
        after the optimization. If True, then the hessian will not be
        calculated. However, it will be available in methods that use the
        hessian in the optimization (currently only with `"newton"`).

    kwargs : keywords
        All kwargs are passed to the chosen solver with one exception. The
        following keyword controls what happens after the fit::

            warn_convergence : bool, optional
                If True, checks the model for the converged flag. If the
                converged flag is False, a ConvergenceWarning is issued.
    """      

    def __init__(self, 
        solver:         str = 'newton-cg', 
        fit_intercept: bool = True,
        max_iter:       int = 1000, 
        verbose:       bool = False, 
        **kwargs
    ):
        setup = dict(            
            solver          = self._align_method(solver),
            maxiter         = max_iter,
            fit_intercept   = fit_intercept,
            disp            = verbose
        ) | kwargs
        
        self._set_req_params(locals())
        super().__init__('logit', setup)

    @staticmethod
    def _align_method(solver:str):
        """SM method for fast convergence when reasonable n-p."""
        if solver == 'newton-cg':
            return 'newton'
        return solver
    
    def fit(self, X:pDF|NDArray, y:pSe, **kwargs):
        """
        Fit the model using maximum likelihood.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of fitted model.        
        """
        self._fit(X, y)
        return self
        
    def summary(self, show:bool = True) -> pDF|None:
        """Econometrical summary of fitted model."""
        if not show:
            return self._summary
        display(self._summary)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ModelBaseLR(PredictBaseLR, ModelBase):

    implemented_estimators = {
        'lasso':   LassoCV,
        'elastic': ElasticNetCV,
        'logreg':  LogisticRegressionCV
    }    

    def __init__(self, solver:Literal['lasso','elastic','logreg'], setup:dict):
        self._set_setup(solver, setup)                

    def _fit_model(self, X:pDF|NDArray, y:pSe, **kwargs) -> Any:
        model = self.get_estimator_instance()
        return model.fit(X, y, **kwargs)
    
    def _fit(self, X:pDF|NDArray, y:pSe, **kwargs):
        self._model = self._fit_model(X, y, **kwargs)
        self._set_features(X)
        self._set_req_props(X, y)
        return self._model
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Lasso(ModelBaseLR):
    """
    Wrap of scikit Lasso applicable to Logit with WOE.
    Fast and efficient convergance given coef > 0 given WOE+logit math.

    Lasso linear model with iterative fitting along a regularization path.
    The optimization objective for Lasso is::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Parameters
    ----------
    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    alphas : array-like or int, default=100
        Values of alphas to test along the regularization path.
        If int, `alphas` values are generated automatically.
        If array-like, list of alpha values to use.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    precompute : 'auto', bool or array-like of shape (n_features, n_features), default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are smaller or equal to
        ``tol``, the optimization code checks the dual gap for optimality and continues
        until it is smaller or equal to ``tol``.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    cv : int, default=5
        Cross-validation generator or iterable.
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - int, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    verbose : bool or int, default=False
        Amount of verbosity.

    n_jobs : int, default=-1
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.
    """

    def __init__(self, 
        eps:         float = 1e-3,
        tol:         float = 1e-4,
        alphas:   list|int = 100,
        fit_intercept:bool = True,
        max_iter:      int = 1000, 
        cv:            int = 5,
        random_state:  int = 42,
        n_jobs:        int = -1,
        verbose:       int = 0,
        **kwargs
    ):
        setup = dict(            
            eps           = eps,
            tol           = tol,
            alphas        = alphas,
            max_iter      = max_iter,
            cv            = cv,
            fit_intercept = fit_intercept,
            random_state  = random_state,
            n_jobs        = n_jobs,
            verbose       = verbose        
        ) | kwargs

        self._set_req_params(locals())
        super().__init__('lasso', setup)

    def fit(self, X:pDF|NDArray, y:pSe, **kwargs):
        """
        Lasso linear model with iterative fitting along a regularization path.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse. Note that large sparse matrices and arrays
            requiring `int64` indices are not accepted.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of fitted model.
        """
        self._model:LassoCV = self._fit(X, y, **kwargs)
        return self
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ElasticNet(ModelBaseLR):
    """
    Wrap of scikit Elastic net applicable to Logit with WOE.
    Fast and efficient convergance given coef > 0 given WOE+logit math.

    Elastic Net model with iterative fitting along a regularization path

    Parameters
    ----------
    l1_ratio : float or list of float, default=0.5
        Float between 0 and 1 passed to ElasticNet (scaling between
        l1 and l2 penalties). For ``l1_ratio = 0``
        the penalty is an L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2
        This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of
        values for l1_ratio is often to put more values close to 1
        (i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,
        .9, .95, .99, 1]``.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    alphas : array-like or int, default=100
        Values of alphas to test along the regularization path, used for each l1_ratio.
        If int, `alphas` values are generated automatically.
        If array-like, list of alpha values to use.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    precompute : 'auto', bool or array-like of shape (n_features, n_features), default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are smaller or equal to
        ``tol``, the optimization code checks the dual gap for optimality and continues
        until it is smaller or equal to ``tol``.

    cv : int, default=5
        Cross-validation generator or iterable.
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - int, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : bool or int, default=0
        Amount of verbosity.

    n_jobs : int, default=-1
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.            
    """

    def __init__(self, 
        l1_ratio:    float = 0.5, 
        eps:         float = 1e-3,
        tol:         float = 1e-4,
        alphas:   list|int = 100,
        fit_intercept:bool = True,
        max_iter:      int = 1000, 
        cv:            int = 5,
        random_state:  int = 42,
        n_jobs:        int = -1, 
        verbose:       int = 0,
        **kwargs
    ):
        setup = dict(            
            l1_ratio      = l1_ratio, 
            eps           = eps,
            tol           = tol,
            alphas        = alphas,
            fit_intercept = fit_intercept,
            max_iter      = max_iter,
            cv            = cv,
            random_state  = random_state,
            n_jobs        = n_jobs,
            verbose       = verbose  
        ) | kwargs        

        self._set_req_params(locals())
        super().__init__('elastic', setup)

    def fit(self, X:pDF|NDArray, y:pSe, **kwargs):
        """
        Elastic Net model with iterative fitting along a regularization path.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse. Note that large sparse matrices and arrays
            requiring `int64` indices are not accepted.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of fitted model.
        """
        self._model:ElasticNetCV = self._fit(X, y, **kwargs)
        return self
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class LogReg(ModelBaseLR):
    """
    Wrap of scikit LogisticRegression net applicable to Logit with WOE.
    Fast and efficient convergance given coef > 0 given WOE+logit math.

    Parameters
    ----------
    Cs : int or list of floats, default=10
        Each of the values in Cs describes the inverse of regularization
        strength. If Cs is as an int, then a grid of Cs values are chosen
        in a logarithmic scale between 1e-4 and 1e4.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'},
            default='newton-cg'

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    max_iter : int, default=100
        Maximum number of iterations of the optimization algorithm.
        
    cv : int or cross-validation generator, default=5
        The default cross-validation generator used is Stratified K-Folds.
        If an integer is provided, it specifies the number of folds, `n_folds`, used.
        See the module :mod:`sklearn.model_selection` module for the
        list of possible cross-validation objects.

    n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        For the 'liblinear', 'sag' and 'lbfgs' solvers set verbose to any
        positive number for verbosity.

    random_state : int, RandomState instance, default=None
        Used when `solver='sag'`, 'saga' or 'liblinear' to shuffle the data.
        Note that this only applies to the solver and not the cross-validation
        generator. See :term:`Glossary <random_state>` for details.
    """

    def __init__(self, 
        Cs:            int = 10,
        tol:         float = 1e-4,
        fit_intercept:bool = True,
        solver:        str = 'newton-cg',
        max_iter:      int = 100, 
        cv:            int = 5,
        random_state:  int = 42,
        n_jobs:        int = -1,
        verbose:       int = 0,
        **kwargs
    ):
        setup = dict(
            Cs            = Cs,
            tol           = tol,
            fit_intercept = fit_intercept,
            solver        = solver,
            max_iter      = max_iter,
            cv            = cv,
            random_state  = random_state,
            n_jobs        = n_jobs,
            verbose       = verbose,
            l1_ratios     = (0,), # l1=0 → L2 penalty → Ridge is better for high dim. data
            scoring       = 'neg_log_loss',
            use_legacy_attributes = False,
        ) | kwargs

        self._set_req_params(locals())
        [setup.pop(x, None) for x in ('eps', 'alphas')] # Lasso-Elastic only
        super().__init__('logreg', setup)

    def fit(self, X:pDF|NDArray, y:pSe, **kwargs):
        """
        Logistic Regression CV (aka logit, MaxEnt) classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse. Note that large sparse matrices and arrays
            requiring `int64` indices are not accepted.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of fitted model.
        """
        self._model:LogisticRegressionCV = self._fit(X, y, **kwargs)
        return self
    
    def predict_proba(self, X:pDF|NDArray) -> NDArray:
        """
        Predict response variable of a model given exogenous variables.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        array
            NDArray([1-p, p]).
        """       
        return self._model.predict_proba(X)
    
    def predict(self, X:pDF|NDArray) -> ArrayLike:
        """
        Predict response variable of a model given exogenous variables.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        array
            Fitted values at exog.
        """        
        return self._model.predict(X)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TREE-GB BASE MODELS API SYNC
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class PredictBaseGB:

    _model: XGBClassifier | LGBMClassifier = None
    _enc:   OrdinalEncoder = None
    _cat:   list = None
        
    @Arrow.transition(contract = Arrow.Contract(safe_conversion=True))
    def _align_X_transform_cat(self, X:pDF|NDArray) -> pDF:
        if self._enc is None:
            return X
        X = X.copy()
        if isinstance(X, np.ndarray):              
            X[:, self._cat] = self._enc.transform(X[:, self._cat])
        else: 
            X[self._cat] = self._enc.transform(X[self._cat])
        return X

    def predict_proba(self, X:pDF|NDArray, **kwargs) -> NDArray:
        """
        Predict response variable of a model given exogenous variables.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        array
            NDArray([1-p, p]).
        """       
        return self._model.predict_proba(self._align_X_transform_cat(X), **kwargs)
    
    def predict(self, X:pDF|NDArray, **kwargs) -> ArrayLike:
        """
        Predict response variable of a model given exogenous variables.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        array
            Fitted values at exog.
        """        
        return self._model.predict(self._align_X_transform_cat(X), **kwargs)

    def shap_values(self, X:pDF|NDArray, **kwargs) -> NDArray:
        """Model shap values by TreeExplainer."""
        explainer = TreeExplainer(self._model, **kwargs)
        return explainer.shap_values(self._align_X_transform_cat(X)) 

    def shap_plot(self, X:pDF|NDArray, **kwargs) -> None:
        """Model visualization by SHAP beeswarm plot, colored by feature values."""
        ModelView.shap_plot(X, self.shap_values(X), **kwargs)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ModelBaseGB(PredictBaseGB, ModelBase):

    implemented_estimators = {
        'xgb': XGBClassifier,
        'lgb': LGBMClassifier,
    }

    def __init__(self, solver:Literal['xgb','lgb'], setup:dict):
        self._set_setup(solver, setup)
        self._set_private_setup(setup)

    def _get_fit_specs(self):
        if self._solver == 'xgb': return ['verbose']
        if self._solver == 'lgb': return ['categorical_feature']

    def _from_fit_specs(self):
        if self._solver == 'xgb': return ['callbacks']
        if self._solver == 'lgb': return []

    def _get_private_specs(self):
        if self._solver == 'xgb': return ['categorical_feature']
        if self._solver == 'lgb': return []

    def _clean_setup_specs(self, setup:dict) -> dict:
        specs = self._get_fit_specs() + self._get_private_specs()
        return {k:v for k,v in setup.items() if k not in specs}

    def _set_cat_fe(self, setup:dict) -> None:
        cat = setup.get('categorical_feature', None)
        self._cat = [cat] if ( cat and isinstance(cat, str) ) else cat

    def _set_fit_api_specs(self, setup:dict):
        self._fit_api_specs = {}
        for s in self._get_fit_specs():
            v = setup.get(s, None)
            if v is not None:
                self._fit_api_specs[s] = v

    def _set_private_setup(self, setup:dict) -> None:
        self._set_cat_fe(setup)
        self._set_fit_api_specs(setup)
        self._setup = self._clean_setup_specs(setup)

    def _align_callback(self, callback):
        if self._solver == 'lgb':
            def lgb_callback(env):
                return callback(env)
            return lgb_callback
        return callback
    
    def _set_callbacks(self, kwargs:dict) -> None:
        setup = self._setup
        call  = setup.pop('default_callback')
        if kwargs.get('callbacks') or not call:
            return
        kwargs['callbacks'] = [self._align_callback(
            ReduceLROnPlateau(
                patience    = 5,
                factor      = 0.5,
                min_lr      = 1e-4,
                initial_lr  = setup['learning_rate'],
        ))]

    def _get_aligned_setup_specs(self, setup:dict) -> dict:
        if self._solver == 'xgb': 
            d = {
                'early_stopping_round': 'early_stopping_rounds'
            }
            for k,n in d.items():
                if k in setup:
                    setup[n] = setup.pop(k)
        return setup

    def _align_depended_specs(self, kwargs: dict) -> dict:
        setup = {**self._setup}
        if 'eval_set' not in kwargs:
            setup.pop('early_stopping_round')
            kwargs.pop('callbacks', None)        
        return self._get_aligned_setup_specs(setup)
    
    def _get_aligned_setup(self, kwargs:dict) -> dict:
        self._set_callbacks(kwargs)
        setup = self._align_depended_specs(kwargs)
        specs = {k:kwargs.pop(k) for k in self._from_fit_specs() if k in kwargs}
        return setup | specs           
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _fit_cat_enc(self, X:pDF|NDArray) -> None:
        if not self._cat:
            return
        if isinstance(X, np.ndarray):
            X_ = X[:, self._cat]
        else: 
            X_ = X[self._cat]
        return OrdinalEncoder(
            handle_unknown          = 'use_encoded_value', 
            unknown_value           = -1, 
            encoded_missing_value   = -1, 
            dtype                   = np.int32
        ).fit(X_)
    
    def _align_eval_set(self, kwargs:dict) -> tuple[dict, list]:
        es = 'eval_set'
        eval_set = kwargs.pop(es, None)
        if eval_set is None:
            return kwargs, [None]*2
        kwargs[es] = [(self._align_X_transform_cat(Xy[0]), Xy[1]) for Xy in eval_set]
        # if self._solver == 'xgb':
        #     kwargs[es] = [(self._transform_cat(Xy[0]), Xy[1]) for Xy in eval_set]
        # if self._solver == 'lgb':
        #     kwargs['eval_X'] = [self._transform_cat(Xy[0]) for Xy in eval_set]
        #     kwargs['eval_y'] = [Xy[1] for Xy in eval_set]
        return kwargs, eval_set[0]
    
    def _set_summary(self, Xy:list) -> None:
        s = SummaryGBM(self)
        try:    s = s.summary(*Xy)
        except: s = s.summary()
        self._summary = s

    def _fit_model(self, X:pDF|NDArray, y:pSe, kwargs:dict) -> LGBMClassifier|XGBClassifier:
        e = self.get_estimator()        
        return e(**self._get_aligned_setup(kwargs)).fit(
            self._align_X_transform_cat(X), y, 
            **self._fit_api_specs, 
            **kwargs
        )
        
    def _fit(self, X:pDF|NDArray, y:pSe, **kwargs):
        self._enc   = self._fit_cat_enc(X)
        kwargs_, Xy = self._align_eval_set(kwargs)
        self._model = self._fit_model(X, y, kwargs_)
        self._set_features(X)
        self._set_summary(Xy)
        return self._model
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def get_categorical_feature(self) -> list|None:
        return self._cat

    def get_booster(self):
        return self.booster_
    
    def dump_model(self, *args, **kwargs):
        return self.booster_.dump_model(*args, **kwargs)
    
    def trees_to_dataframe(self):
        return self.booster_.trees_to_dataframe()
    
    def summary(self, show:bool=True) -> str|None:
        """Econometrical summary of fitted model."""
        if not show:
            return self._summary
        display(self._summary)

    @property
    def booster_(self):
        return self._model._Booster

    @property
    def booster_view(self):
        return self.trees_to_dataframe()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class LGB(ModelBaseGB):
    """
    Wrap of LGBMClassifier to follow standard ML API.
    Construct a gradient boosting model.

    Parameters
    ----------
    max_leaves : int, optional (default=31)
        Maximum tree leaves for base learners.
        Decrease num_leaves to reduce training time.
    
    max_depth : int, optional (default=-1)
        This parameter is an integer that controls the maximum distance between the root node 
        of each tree and a leaf node. 
        LightGBM adds nodes to trees based on the gain from adding that node, regardless of depth. 
        Because of this growth strategy, it isn’t straightforward to use max_depth alone to limit 
        the complexity of trees. The num_leaves parameter sets the maximum number of nodes per tree. 

    learning_rate : float, optional (default=0.1)
        Boosting learning rate.
        You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
        in training using ``reset_parameter`` callback.
        Note, that this will ignore the ``learning_rate`` argument in training.

    n_estimators : int, optional (default=100)
        Number of boosted trees to fit.

    min_split_gain : float, optional (default=0.)
        Minimum loss reduction required to make a further partition on a leaf node of the tree.

    min_child_weight : float, optional (default=0.1)
        Minimum sum of instance weight (Hessian) needed in a child node.
        min_hessian_leaf = ∑pi​(1−pi​)
        Behaves as second-order regularizatio, e.g. w = 0.1, p = 0.01
         - min sample of the split: w / p(1-p) → needed min sample: 10 obs

    subsample : float, optional (default=0.7)
        Subsample ratio of the training instance.

    colsample_bytree : float, optional (default=0.7)
        Subsample ratio of columns when constructing each tree.

    reg_alpha : float, optional (default=0.)
        L1 regularization term on weights.

    reg_lambda : float, optional (default=0.)
        L2 regularization term on weights.

    early_stopping_round : int, `(default=50)`
        If early stopping is set to int value, after each boosting round the model’s training accuracy is 
        evaluated against a validation set that contains data not available to the training process. 
        That accuracy is then compared to the accuracy as of the previous boosting round. If the model’s 
        accuracy fails to improve for some number of consecutive rounds, xBM stops the training process.
    
    categorical_feature : str, list (default=None)
        Categorical features specification. 
        list / str → feature names in X. 
        All values in categorical features will be cast to int32.
    
    default_callback : bool, (default=False)
        When no callbacks are specified, default_callback = True activates `ReduceLROnPlateau`,
        with following params: patience=10, factor=0.5, min_lr=1e-4.

    random_state : int, RandomState object or None, optional (default=42)
        Random number seed.
        If int, this number is used to seed the C++ code.
        If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
        If None, default seeds in C++ code are used.

    n_jobs : int or None, optional (default=None)
        Number of parallel threads to use for training (can be changed at prediction time by
        passing it as an extra keyword argument).
    
    verbose : int, default=-1
        Level of verbosity, where -1: off, 0: warnings only, 1: warning+info...        
    """
    METRIC_MAP = {
        'logloss': 'binary_logloss'
    }

    def __init__(self,
        max_leaves:               int = 31,
        max_depth:                int = -1,
        learning_rate:          float = 0.1,
        n_estimators:             int = 100,
        min_child_weight:       float = 0.1,
        subsample:              float = 0.7,
        colsample_bytree:       float = 0.7, 
        reg_alpha:              float = 0.0,
        reg_lambda:             float = 0.0,
        early_stopping_round:     int = 50,
        categorical_feature: str|list = None,
        default_callback:        bool = False,
        random_state:             int = 42,
        n_jobs:                   int = -1,
        verbose:                  int = -1,
        metric:                   str = 'logloss',
        **kwargs
    ):
        setup = dict(            
            num_leaves              = max_leaves,
            learning_rate           = learning_rate,
            n_estimators            = n_estimators, 
            min_child_weight        = min_child_weight,
            subsample               = subsample,
            colsample_bytree        = colsample_bytree,
            reg_alpha               = reg_alpha,
            reg_lambda              = reg_lambda,
            early_stopping_round    = early_stopping_round,
            categorical_feature     = categorical_feature,
            default_callback        = default_callback,
            random_state            = random_state,
            n_jobs                  = n_jobs,
            verbose                 = verbose,
            max_depth               = max_depth,
            metric                  = self.METRIC_MAP.get(metric) or metric,
            objective               = 'binary',
            boosting_type           = 'gbdt',
            importance_type         = 'gain',
            subsample_freq          = 1,
        ) | kwargs        
        
        self._set_req_params(locals())
        super().__init__('lgb', setup)

    def fit(self, X:pDF|NDArray, y:pSe, **kwargs):
        """
        LGBMClassifier model fitting with standardized API.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse. Note that large sparse matrices and arrays
            requiring `int64` indices are not accepted.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        eval_set :
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.

        Returns
        -------
        self : object
            Returns an instance of fitted model.
        """
        self._model:LGBMClassifier = self._fit(X, y, **kwargs)
        return self
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class XGB(ModelBaseGB):
    """
    Wrap of XGBClassifier to follow standard ML API.
    Construct a gradient boosting model.

    Parameters
    ----------
    max_leaves : int, optional (default=31)
        Maximum tree leaves for base learners.

    max_depth : int, optional (default=0)
        Maximum depth of a tree. Increasing this value will make the model more complex and 
        more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively 
        consumes memory when training a deep tree. "exact" tree method requires non-zero value.

    learning_rate : float, optional (default=0.1)
        Boosting learning rate.
        You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
        in training using ``reset_parameter`` callback.
        Note, that this will ignore the ``learning_rate`` argument in training.

    n_estimators : int, optional (default=100)
        Number of boosted trees to fit.

    min_split_gain : float, optional (default=0.)
        Minimum loss reduction required to make a further partition on a leaf node of the tree.

    min_child_weight : float, optional (default=0.1)
        Minimum sum of instance weight (Hessian) needed in a child node.
        min_hessian_leaf = ∑pi​(1−pi​)
        Behaves as second-order regularizatio, e.g. w = 0.1, p = 0.01
         - min sample of the split: w / p(1-p) → needed min sample: 10 obs

    subsample : float, optional (default=0.7)
        Subsample ratio of the training instance.

    colsample_bytree : float, optional (default=0.7)
        Subsample ratio of columns when constructing each tree.

    reg_alpha : float, optional (default=0.)
        L1 regularization term on weights.

    reg_lambda : float, optional (default=0.)
        L2 regularization term on weights.
    
    early_stopping_round : int, `(default=50)`
        If early stopping is set to int value, after each boosting round the model’s training accuracy is 
        evaluated against a validation set that contains data not available to the training process. 
        That accuracy is then compared to the accuracy as of the previous boosting round. If the model’s 
        accuracy fails to improve for some number of consecutive rounds, xBM stops the training process.
    
    categorical_feature : str, list (default=None)
        Categorical features specification. 
        list / str → feature names in X. 
        All values in categorical features will be cast to int32.

    default_callback : bool, (default=False)
        When no callbacks are specified, default_callback = True activates `ReduceLROnPlateau`,
        with following params: patience=10, factor=0.5, min_lr=1e-4.

    random_state : int, RandomState object or None, optional (default=42)
        Random number seed.
        If int, this number is used to seed the C++ code.
        If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
        If None, default seeds in C++ code are used.

    n_jobs : int or None, optional (default=None)
        Number of parallel threads to use for training (can be changed at prediction time by
        passing it as an extra keyword argument).

    verbose : int, default=-1
        Level of verbosity, where -1/0: off, 1: warning+info...
    """    
    METRIC_MAP = {}

    def __init__(self,
        max_leaves:               int = 31,
        max_depth:                int = 0,
        learning_rate:          float = 0.1,
        n_estimators:             int = 100, 
        min_child_weight:       float = 0.1,
        subsample:              float = 0.7,
        colsample_bytree:       float = 0.7, 
        reg_alpha:              float = 0.0,
        reg_lambda:             float = 0.0,
        early_stopping_round:     int = 50,
        categorical_feature: str|list = None,
        default_callback:        bool = False,
        random_state:             int = 42,
        n_jobs:                   int = -1,
        verbose:                  int = -1,
        metric:                   str = 'logloss',
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
            early_stopping_round    = early_stopping_round,
            categorical_feature     = categorical_feature,
            default_callback        = default_callback,
            random_state            = random_state,
            n_jobs                  = n_jobs,
            verbose                 = max(0, verbose),
            enable_categorical      = True,
            max_depth               = max(0, max_depth),
            eval_metric             = self.METRIC_MAP.get(metric) or metric,
            tree_method             = 'hist',
            grow_policy             = 'lossguide',
            objective               = 'binary:logistic',
            booster                 = 'gbtree',
            importance_type         = 'total_gain',
        ) | kwargs        
        
        self._set_req_params(locals())
        super().__init__('xgb', setup)

    def fit(self, X:pDF|NDArray, y:pSe, **kwargs):
        """
        XGBClassifier model fitting with standardized API.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse. Note that large sparse matrices and arrays
            requiring `int64` indices are not accepted.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        eval_set :
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.

        Returns
        -------
        self : object
            Returns an instance of fitted model.
        """            
        self._model:XGBClassifier = self._fit(X, y, **kwargs)
        return self
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# HELPERS TO DETERMINE PROPER BEHAVIOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ModelUtils:

    @staticmethod
    def is_tree(model) -> bool:
        """Checks model instance of AIG GBs - LGBM - XGB."""            
        return isinstance(model, (
            PredictBaseGB, LGBMClassifier, XGBClassifier
        ))
    
    @staticmethod
    def is_lr(model) -> bool:
        """Checks model instance of AIG Logit - LogisticRegression."""            
        return isinstance(model, (
            PredictBaseLR, LogisticRegression
        ))
    
    @staticmethod
    def is_n_jobs(model) -> bool:
        """Checks model instance of active n_jobs param → True if (n_jobs != 1)."""            
        if hasattr(model, 'n_jobs'):
            return model.n_jobs != 1
        return False

    @staticmethod
    def update_n_jobs(model, n_jobs:int) -> None:
        if hasattr(model, 'n_jobs'):
            model.n_jobs = n_jobs
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# MODEL SUMMARY
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class SummaryGBM:    
    """Helper to provide GLM-like econometrical HTML summary for Tree Models."""

    def __init__(self, model:LGB|XGB, is_classification:bool = True, feature_names:list = None):
        self.model = model
        self.is_classification = is_classification
        self.feature_names = feature_names
        self.is_xgb = 'XGB' in self.model.__class__.__name__
        
    def _get_meta(self):
        """Extracts unified meta-parameters across LGBM and XGB."""
        params = self.model.get_params()
        
        if self.is_xgb:
            num_features = getattr(self.model, 'n_features_in_', None)
            actual_trees = getattr(self.model, 'best_iteration', self.model.n_estimators)
            objective    = params.get('objective', 'binary:logistic')
        else:
            booster      = self.model.booster_
            num_features = getattr(self.model, 'n_features_in_', booster.num_feature())
            actual_trees = booster.current_iteration()
            objective    = params.get('objective', 'default')
            
        leaves = params.get('num_leaves', 31)
        lr     = params.get('learning_rate', 0.1)
            
        return num_features, actual_trees, objective, leaves, lr

    def _header(self) -> str:
        """Constructs the Meta Parameters HTML table."""
        model_type = self.model.__class__.__name__
        num_features, actual_trees, objective, leaves, lr = self._get_meta()
        now = Misc.timestamp('human')

        html = f"""
        <table style="width: 100%; border-collapse: collapse; border-top: 2px solid black; border-bottom: 2px solid black; font-family: monospace; text-align: left; margin-bottom: 20px;">
            <caption style="font-size: 1.2em; font-weight: bold; margin-bottom: 10px;">Gradient Boosting Model Summary</caption>
            <tr>
                <td><strong>Dep. Variable:</strong></td> <td>y</td>
                <td><strong>No. Trees / Iterations:</strong></td> <td>{actual_trees}</td>
            </tr>
            <tr>
                <td><strong>Model Type:</strong></td> <td>{model_type}</td>
                <td><strong>Max Leaves per Tree:</strong></td> <td>{leaves}</td>
            </tr>
            <tr>
                <td><strong>Objective:</strong></td> <td>{str(objective)}</td>
                <td><strong>Learning Rate (eta):</strong></td> <td>{lr}</td>
            </tr>
            <tr>
                <td><strong>Date:</strong></td> <td>{now[:15]}</td>
                <td><strong>Total Features:</strong></td> <td>{num_features}</td>
            </tr>
            <tr>
                <td><strong>Time:</strong></td> <td>{now[17:]}</td>
                <td><strong>Fit Engine:</strong></td> <td>Scikit-API</td>
            </tr>
        </table>
        """
        return html

    def _performance(self, X, y) -> str:
        """Constructs the Fit Statistics HTML table."""
        if X is None or y is None:
            return ""
            
        n_obs = len(y)
        
        if self.is_classification: 
            probs = self.model.predict_proba(X)[:, 1]
            metric_1_name, metric_1_val = "Log Loss (Deviance):", f"{log_loss(y, probs):.4f}"
            try:    metric_2_val = f"{roc_auc_score(y, probs):.4f}"
            except: metric_2_val = "NaN"
            metric_2_name = "Area Under ROC (AUC):"
        else:
            pred = self.model.predict(X)
            metric_1_name, metric_1_val = "Mean Squared Error:", f"{mean_squared_error(y, pred):.4f}"
            metric_2_name, metric_2_val = "R-squared (Pseudo):", f"{r2_score(y, pred):.4f}"

        html = f"""
        <table style="width: 100%; border-collapse: collapse; border-top: 2px solid black; border-bottom: 2px solid black; font-family: monospace; text-align: left; margin-bottom: 20px;">
            <tr>
                <td><strong>No. Observations:</strong></td> <td>{n_obs}</td>
                <td><strong>{metric_1_name}</strong></td> <td>{metric_1_val}</td>
            </tr>
            <tr>
                <td><strong></strong></td> <td></td>
                <td><strong>{metric_2_name}</strong></td> <td>{metric_2_val}</td>
            </tr>
        </table>
        """
        return html

    def _importance(self, X, num_features) -> str:       
        """Constructs the Feature Importance HTML table."""
        if self.feature_names is not None:
            feats = self.feature_names
        elif hasattr(X, 'columns'):
            feats = list(X.columns)
        else:
            feats = [f"Feature_{i}" for i in range(num_features)]

        try:
            if self.is_xgb:
                booster     = self.model.get_booster()
                gain_dict   = booster.get_score(importance_type='total_gain')
                split_dict  = booster.get_score(importance_type='weight')
                
                # XGBoost returns dicts mapping 'f0', 'f1' (or names) to scores
                # We align them based on the feature array order
                b_feats   = booster.feature_names if booster.feature_names else [f"f{i}" for i in range(len(feats))]
                gain_imp  = np.array([gain_dict.get(f, 0.0) for f in b_feats])
                split_imp = np.array([split_dict.get(f, 0.0) for f in b_feats])
            else:
                gain_imp  = self.model.booster_.feature_importance(importance_type='gain')
                split_imp = self.model.booster_.feature_importance(importance_type='split')

            total_gain  = np.sum(gain_imp) if np.sum(gain_imp) > 0 else 1
            total_split = np.sum(split_imp) if np.sum(split_imp) > 0 else 1
            
            imp_df = pDF({
                'Feature':      feats,
                'Gain_Raw':     gain_imp,
                'Gain_Pct':     (gain_imp / total_gain) * 100,
                'Splits_Raw':   split_imp,
                'Split_Pct':    (split_imp / total_split) * 100
            }).sort_values(by='Gain_Raw', ascending=False).reset_index(drop=True)
            
            # Build statsmodels-like coefficient table
            rows = ""
            for _, row in imp_df.iterrows():
                rows += f"""
                <tr>
                    <td style="text-align: left;">{row['Feature']}</td>
                    <td>{row['Gain_Raw']:.2f}</td>
                    <td>{row['Gain_Pct']:.1f}%</td>
                    <td>{int(row['Splits_Raw'])}</td>
                    <td>{row['Split_Pct']:.1f}%</td>
                </tr>
                """
                
            html = f"""
            <table style="width: 100%; border-collapse: collapse; border-top: 2px solid black; border-bottom: 2px solid black; font-family: monospace; text-align: right;">
                <thead style="border-bottom: 1px solid black;">
                    <tr>
                        <th style="text-align: left;">Feature Name</th>
                        <th>Gain (Info)</th>
                        <th>Gain %</th>
                        <th>Splits (Freq)</th>
                        <th>Split %</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            """
            return html
        except Exception as e:
            return f"<p style='color:red;'>Feature Importance Table failed: {str(e)}</p>"

    @Arrow.transition(contract = Arrow.Contract(safe_conversion=True))
    def summary(self, X=None, y=None):
        """Returns the fully rendered HTML string."""        
        num_features = self._get_meta()[0]
        header       = self._header()
        fit_stats    = self._performance(X, y)
        fe_table     = self._importance(X, num_features)
        
        final_html = f"<div style='max-width: 800px;'>{header}{fit_stats}{fe_table}</div>"
        return HTML(final_html)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SummaryGLM:
    """
    Helper to provide unified econometrical HTML summary for GLM / Logit 
    Models matching SummaryGB style.

    Accepts either a fitted statsmodels results object or a summary2 object.
    """

    def __init__(self, model):
        if hasattr(model, "summary2"):
            self.sm_summary = model.summary2()
        else:
            self.sm_summary = model

    def _header(self) -> str:
        """Constructs the Meta Parameters HTML table matching SummaryGB layout."""
        meta_df = self.sm_summary.tables[0]

        rows = ""
        for _, row in meta_df.iterrows():
            k1 = str(row[0]).strip() if pd.notna(row[0]) else ""
            v1 = str(row[1]).strip() if pd.notna(row[1]) else ""
            k2 = (
                str(row[2]).strip()
                if len(row) > 2 and pd.notna(row[2])
                else ""
            )
            v2 = (
                str(row[3]).strip()
                if len(row) > 3 and pd.notna(row[3])
                else ""
            )

            k1_str = f"<strong>{k1}</strong>" if k1 else ""
            k2_str = f"<strong>{k2}</strong>" if k2 else ""

            rows += f"""
            <tr>
                <td>{k1_str}</td> <td>{v1}</td>
                <td>{k2_str}</td> <td>{v2}</td>
            </tr>
            """

        html = f"""
        <table style="width: 100%; border-collapse: collapse; border-top: 2px solid black; border-bottom: 2px solid black; font-family: monospace; text-align: left; margin-bottom: 20px;">
            <caption style="font-size: 1.2em; font-weight: bold; margin-bottom: 10px;">Generalized Linear Model Summary</caption>
            {rows}
        </table>
        """
        return html

    def _coefficients(self) -> str:
        """Constructs the Coefficients HTML table with dynamic character-width sizing."""
        coef_df = self.sm_summary.tables[-1]

        # Calculate max string length across all feature names and header
        feat_names = [str(idx) for idx in coef_df.index]
        max_len = max([len(name) for name in feat_names] + [len("Feature Name")])
        
        # Convert character count directly to CSS `ch` units (exact character width in monospace)
        feat_width = f"{max_len}ch"

        # Headers
        headers_html = f'<th style="text-align: left; width: {feat_width}; min-width: {feat_width}; white-space: nowrap; padding-right: 15px;">Feature Name</th>'
        for col in coef_df.columns:
            headers_html += f'<th style="white-space: nowrap; padding-left: 10px;">{col}</th>'

        # Rows
        rows_html = ""
        for idx, row in coef_df.iterrows():
            vals_html = ""
            for val in row:
                val_str = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
                vals_html += f'<td style="white-space: nowrap; padding-left: 10px;">{val_str}</td>'

            rows_html += f"""
            <tr>
                <td style="text-align: left; width: {feat_width}; min-width: {feat_width}; white-space: nowrap; padding-right: 15px;">{idx}</td>
                {vals_html}
            </tr>
            """

        html = f"""
        <table style="width: 100%; border-collapse: collapse; border-top: 2px solid black; border-bottom: 2px solid black; font-family: monospace; text-align: right;">
            <thead style="border-bottom: 1px solid black;">
                <tr>
                    {headers_html}
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """
        return html

    def summary(self) -> HTML:
        """Returns the fully rendered HTML string inside an IPython HTML wrapper."""
        header = self._header()
        coef_table = self._coefficients()

        final_html = f"<div style='max-width: 800px;'>{header}{coef_table}</div>"
        return HTML(final_html)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬