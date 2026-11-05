"""
Estimator Adapter providing unified model access for feature learning processes.

This module implements a strategic adapter layer that homogenizes various ML 
estimators—ranging from linear solvers and PLS projectors to Gradient Boosting 
Machines—into a consistent interface used by the feature selection and 
importance engines. It ensures that any provided estimator adheres to the 
mandatory `.fit()` and `.predict_proba()` contracts, providing a fail-safe 
mechanism that guarantees API compatibility across different library backends.
"""

from ...deps.data           import  pDF, np
from ...deps.ml.glm         import  LogisticRegression
from ...deps.types          import  Literal
from ...ml.models           import  (
                                    Logit,
                                    LogReg,
                                    Lasso, 
                                    ElasticNet,
                                    XGB,
                                    LGB,
                                    SparseSelector,
                                    PLSSelector,
                                    BinaryPLSRegression,
                                    )
from ...ml.models.models    import  (
                                    ModelBaseLR,
                                    ModelBaseGB
                                    )
from ...ml.sampling         import  DataUtils
from .selection             import  (
                                    LinearSelectorCV,
                                    TreeSelectorGB,
                                    XGB, LGB
                                    )

OutputModel  = (
    Logit | LogReg | LogisticRegression | Lasso | ElasticNet | 
    BinaryPLSRegression | PLSSelector |
    XGB | LGB
)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FEATURE BASE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ValidSupport:
    
    @staticmethod
    def input(model) -> None:
        if model is None:
            return
        attr = [hasattr(model, x) for x in ('fit', 'predict_proba')]
        if all(attr): 
            return
        raise TypeError("AIG SAFETY ► provided estimator does't have .fit() and/or .predict_proba()")
    
    def output(model) -> None:
        if isinstance(model, OutputModel):
            return
        raise TypeError(
            f'Not support model type, please choose input model from {OutputModel}'
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class FeatureEstimator:
    """
    Adapter to provide right estimator with respect to setup params.
     - `model` can be any model from AIG library; if None then defaults Logit | LogisticRegression
     - `is_over_fit` ensusures that input model is not overriden → new instance is created
     - `is_shadow`    ensures that GB estimator has categorical features for shadow cols
     - `is_intercept` ensures that returned estimator respect fit_intercept design
    """

    def __init__(self,
        model:          LogisticRegression = None,
        solver:         Literal['lbfgs', 'newton-cg'] = 'newton-cg', 
        random_state:   int   = 42,
        is_over_fit:    bool  = False,
        is_shadow:      bool  = False,
        is_intercept:   bool  = True
    ):
        self.model  = model
        self.solver = solver
        self.random = random_state

        self._is_over_fit  = is_over_fit
        self._is_shadow    = is_shadow
        self._is_intercept = is_intercept
        ValidSupport.input(model)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def is_linear_selector(self) -> bool:
        """Feature assessment must avoid feature selection → selector's predict model behavior."""
        return isinstance(self.model, (SparseSelector, LinearSelectorCV))
    
    def is_non_linear_selector(self) -> bool:
        """Feature assessment must avoid feature selection → selector's predict model behavior."""
        return isinstance(self.model, TreeSelectorGB)
    
    def is_pls_estimator(self) -> bool:
        """Feature assessment must avoid feature selection → selector's predict model behavior."""
        return isinstance(self.model, (BinaryPLSRegression, PLSSelector))
    
    def is_non_linear_estimator(self) -> bool:
        """GB models needs extra care for potential categorical features."""
        return (
            self.is_non_linear_selector() or 
            isinstance(self.model, ModelBaseGB)
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def get_default_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            C               = np.inf, 
            tol             = 1e-4,
            max_iter        = 1000,
            solver          = self.solver, 
            fit_intercept   = self._is_intercept, 
            random_state    = self.random,
        )
    
    def get_logit_estimator(self) -> Logit:        
        return Logit(
            self.solver, 
            fit_intercept = self._is_intercept,
            max_iter      = 1000
        )

    def _get_default_estimator(self, is_sm:bool=False):
        if is_sm: 
            return self.get_logit_estimator()
        return self.get_default_estimator()

    def get_linear_estimator(self) -> Logit:
        if not self._is_over_fit:
            return self.model
        # New model's fit must avoid fe selection if provided model was a selector type
        return self.get_logit_estimator()
    
    def get_pls_estimator(self) -> BinaryPLSRegression:
        return self.get_logit_estimator()
        # NOTE the assessment of PLS features can be done in two views:
        #  (1) input → pls.transform(d) → correct view to pls components importance
        #  (2) input → d → importance view of underlying features
        #            → though not fully relevant as plc comp. smooths shocks in indv. fes
    
    def get_non_linear_estimator(self, X:pDF=None) -> XGB | LGB:
        if not self._is_over_fit or ( X is None ):
            return self.model
        
        # 1 New model's fit must avoid fe selection if provided model was a selector type
        m:ModelBaseGB | TreeSelectorGB  = self.model
        if self.is_non_linear_selector():
            m_:TreeSelectorGB = self.model
            m:ModelBaseGB = m_.model # final fitted estimator XGB | LGB

        # 2 New model's fit must have only active subset of cats for its fit-valid evaluation
        setup = self._get_non_linear_setup(m, X)
        est   = type(m)
        return est(**setup)

    def _get_non_linear_setup(self, model:ModelBaseGB, X:pDF) -> dict:
        cats  = model.get_categorical_feature()
        setup = model.get_setup()        
        if not cats:
            return setup        
        cats  = [x for x in cats if x in X.columns]
        if self._is_shadow:
            cats  = DataUtils.get_cols_with_shadow(cats)
        return setup | dict(categorical_feature = cats)
            
    def get_base_estimator(self, **kwargs) -> OutputModel:
        m: ModelBaseLR | ModelBaseGB = self.model
        est = type(m)
        setup = m.get_setup() | kwargs
        if 'fit_intercept' in m.get_params():
            setup = setup | dict(fit_intercept=self._is_intercept)
        return est(**setup)        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def get_model(self, X:pDF=None) -> OutputModel|None:
        """Returns GLM or GBM estimator with respect to the setup."""
        if self.model is None:
            raise ValueError('Safety ► self.model is None which must not be by design.')        
        if self.is_linear_selector():
            return self.get_linear_estimator()
        if self.is_pls_estimator():
            return self.get_pls_estimator()
        if self.is_non_linear_estimator():
            return self.get_non_linear_estimator(X)        
        if self._is_over_fit:
            return self.get_base_estimator()
        return self.model
                            
    def get_estimator_instance(self, X:pDF=None, is_sm:bool=False) -> OutputModel:
        """Returns GLM or GBM estimator with respect to the setup."""
        if self.model is not None:
            model = self.get_model(X)
            ValidSupport.output(model)
            return model
        return self._get_default_estimator(is_sm)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬