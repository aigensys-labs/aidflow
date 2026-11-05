"""
Statistically optimized Partial Least Squares (PLS) engine for feature selection 
and noise filtering.

This module implements a specialized PLS-driven framework designed specifically for 
binary targets and Weight of Evidence (WOE) features. It leverages the linear 
relationship between WOE values and target probability to project high-dimensional 
feature spaces into orthogonal latent components, effectively reducing multicollinearity 
and enhancing model stability through multivariate covariance optimization.
"""

from ...deps.data       import  np, pDF, pSe
from ...deps.ml.glm     import  PLSRegression                                
from ...deps.struct     import  dataclass
from ...deps.display    import  display
from ...deps.types      import  (
                                Literal,
                                Callable,
                                NDArray,
                                ArrayLike
                                )
from ...core.factory    import  DictClass
from ...core.foundation import  AigParam
from ...utils.builtins  import  Str

from ..metrics          import  Gini
from ..sampling         import  Sampling
from ..tests            import  TestModelDiff

from .models            import  Logit
from .prediction        import  PredictSelectionMixin


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class STAT(DictClass):
    F = 'Feature'
    V = 'VIP'
    S = 'Search'
    B = 'Base'

class GINI(DictClass):
    N = 'NPC'
    T = 'train_Gini'
    V = 'valid_Gini'
    P = 'p-value'
    B = 'p-base'
    D = 'valid_Giniδ'
    S = 'Significantδ'
    C = 'Choice'
    Y = '✅'
    X = '❌'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PARTIAL LEAST SQUARES REGRESSION | APPLIED TO LOG-REG WOE FEATURES
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

class ModelBase:

    implemented_estimators: dict

    def __init__(self, solver:str, setup:dict,):
        self._set_setup(solver, setup)

    def _get_solver(self, solver:str):
        return self.implemented_estimators[solver]        
        
    def _set_setup(self, solver:str, setup:dict) -> None:
        self._solver = solver
        self.setup = {'estimator': self._get_solver(solver), 'setup': setup}

    def get_estimator(self):
        return self.setup['estimator']
    
    def get_setup(self, key:str=None) -> dict:
        s = self.setup['setup']
        if key is None:
            return s
        return s[key]
        
    def get_estimator_instance(self, n_components:int = None, **kwargs):
        e = self.get_estimator()
        s = self.get_setup() | kwargs | dict(n_components = n_components)
        p = {k:v for k,v in s.items() if k in AigParam.get_definition(e)}
        return e(**p)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@dataclass
class SampleData:
    X_train: pDF
    y_train: pSe
    X_valid: pDF
    y_valid: pSe

class FitData:
    """Data helper during optim search."""
      
    @staticmethod
    def get_data(X:pDF|NDArray, subset:list=None) -> pDF|NDArray:
        X_ = X
        if subset:
            X_ = X[subset]
        if not len(X_.shape) == 2 or X_.shape[1] < 2:
            raise ValueError('PLS minimum set of input features is 2.')
        return X_
    
    @staticmethod
    def get_fes(X:pDF|NDArray, subset:list=None) -> list | int:
        if isinstance(X, pDF):
            return X.columns.to_list()
        if subset is None:
            return list(range(len(X)))
        return subset  
        
    @staticmethod
    def cv_folds(X: pDF|NDArray, y: pSe|ArrayLike, cv: int, random_state: int=42) -> tuple:
        return Sampling(
            sample_size  = 1,
            stratify     = y,
            random_state = random_state,
        ).cv_folds(X, y, n_splits = cv)

    @staticmethod
    def train_valid_split(
        X: pDF | NDArray, 
        y: pSe | ArrayLike, 
        val_size: float, 
        random_state: int = 42
    ) -> tuple:
        return Sampling(
            sample_size  = val_size,
            stratify     = y,
            random_state = random_state,
        ).train_valid_split(X, y)
        
    @staticmethod
    def get_sample_data(
        Xy_train: tuple[pDF|NDArray, pSe|ArrayLike], 
        Xy_valid: tuple[pDF|NDArray, pSe|ArrayLike]
    ) -> SampleData:
        return SampleData(
            X_train = Xy_train[0],
            y_train = Xy_train[1],
            X_valid = Xy_valid[0],
            y_valid = Xy_valid[1]
        )   
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FitBase(ModelBase):  

    implemented_estimators = {
        'pls':  PLSRegression,
    }

    COL = lambda i: f'PLS_{i}'
    
    @classmethod
    def _pls_columns(cls, npc:int) -> list:
        return [cls.COL(i) for i in range(npc)]
    
    def _fit_pls(self, X:pDF, y:pSe, npc:int) -> PLSRegression:
        pls: PLSRegression = self.get_estimator_instance(npc)
        return pls.fit(X, y) 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class FitWorker(FitBase):
    """
    Internal worker class responsible for the iterative search of the optimal 
    number of PLS components.    
    
    Parameters
    ----------
    solver : str
        The identifier for the estimator type (e.g., 'pls').
    setup : dict
        Configuration parameters including alpha, cv folds, random state, etc.        
    """
     
    def _pls_fit_trans(self, data:SampleData, npc:int) -> SampleData:
        pls = self._fit_pls(data.X_train, data.y_train, npc)
        data.X_train = pls.transform(data.X_train)
        data.X_valid = pls.transform(data.X_valid)
        return data

    def _log_fit_pred(self, data:SampleData) -> dict:
        m = Logit().fit(data.X_train, data.y_train)
        return {
            GINI.T: m.predict_proba(data.X_train)[:,1],
            GINI.V: m.predict_proba(data.X_valid)[:,1]
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def _gini_npc(self, npc:int, samples:tuple) -> tuple[dict, list]:
        gini, p1 = [], []
        for Xy_train, Xy_valid in samples:
            
            d = FitData.get_sample_data(Xy_train, Xy_valid)
            d = self._pls_fit_trans(d, npc)                
            p = self._log_fit_pred(d)
            
            gini.append([
                Gini.fast_calc(d.y_train, p[GINI.T]), 
                Gini.fast_calc(d.y_valid, p[GINI.V])
            ])
            p1.extend(p[GINI.V])        
        
        gini_avg = np.average( tuple(zip(*gini)), axis=1 )
        gini_avg = dict(zip([GINI.T, GINI.V], np.round(gini_avg, 4)))

        return gini_avg, p1
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def _info(self, vector: dict) -> None:
        if not self.get_setup('verbose'):
            return
        a = lambda x: 10 if x == GINI.P else 2 if x == GINI.N else 7
        i = [f'{k}: {v}{Str.align(a(k), str(v))}' for k,v in vector.items()]
        print(' |  '.join(i))

    def _vector(self, npc:int, gini:dict, p_val:float, base:int) -> dict:
        r = lambda x,r=4: float(round(x, r))
        vector = {
            GINI.N: int(npc), 
            GINI.T: r(gini[GINI.T]), 
            GINI.V: r(gini[GINI.V]), 
            GINI.P: r(p_val, 7), 
            GINI.B: int(base), 
        }
        self._info(vector)        
        return vector        
    
    def _npc_choice(self, op:pDF) -> int:        
        op[GINI.D] = op[GINI.V].diff()
        
        # we are interested only in pos space of sig. valid performace change
        m = ( op[GINI.P] < self.get_setup('alpha') ) & ( op[GINI.D] > 0 ) 
        
        # this design leads to the choice as the last significant change
        c = 1
        if m.sum():
            c = op.loc[m, GINI.N].iloc[-1]
        
        op[GINI.S] = np.where((m | (op[GINI.P]==-1)), True, False)
        op[GINI.C] = np.where(op[GINI.N] == c, GINI.Y, GINI.X)
        
        return c

    def _result(self, op:dict) -> tuple[int, pDF]:
        op = pDF(op)
        npc = self._npc_choice(op)
        return npc, op
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def process(self,  X:pDF|NDArray, y:pSe|ArrayLike) -> tuple[int, pDF]:
        """
        Executes the component optimization search.
        
        Iterates through the number of components from 1 to `max_components`. 
        For each step, it calculates the average Gini score via cross-validation 
        and performs a DeLong test to check if the improvement is statistically 
        significant. It keeps track of the best performing model (highest Gini 
        with p-value < alpha) and stops early if no significant improvement is 
        observed for `early_stopping` consecutive steps.
        
        Parameters
        ----------
        X : pDF, the feature matrix.
        y : pSe, the target series.
            
        Returns
        -------
        npc : int, the optimal number of components.
        optim_path : pDF, a DataFrame containing the optimization path metrics.
        """
        setup    = self.get_setup()
        patience = setup['early_stopping']
        max_npc  = setup['max_components']
        alpha    = setup['alpha']

        samples = FitData.cv_folds(X, y, setup['cv'], setup['random_state'])
        yv_true = np.concatenate( [x[1][1] for x in samples] ) # [1] → val, [1] → y

        to_npc = min(max_npc+1, X.shape[1])

        optim_path, p_val = [], -1
        for npc in range(1, to_npc):
            
            gini, p1 = self._gini_npc(npc, samples)
            
            if npc > 1:
                p_val = TestModelDiff(yv_true, p0, p1).delong()
            
            if p_val < alpha:   # reassigning the best model
                p0 = p1
                base = npc
                counter = 0
            else:
                counter+=1

            optim_path.append(
                self._vector(npc, gini, p_val, base)
            )

            if counter > patience:
                break

        return self._result(optim_path)         
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class BinaryPLSPredictBase(PredictBase):
    
    def __getattr__(self, name):
        fx = getattr(self.model, name)        
        return self._logit(fx)

    def _set_predict_model(self, X:pDF|NDArray, y:pSe|ArrayLike) -> None:
        X_ = self.transform(X)
        super()._set_predict_model(X_, y)
    
    def transform(self, X:pDF|NDArray, verbose:bool=False) -> pDF|NDArray:
        raise NotImplementedError('Worker must implement transform method.') 
        
    def predict_proba(self, X:pDF|NDArray, verbose=True, **kwargs) -> NDArray:
        """
        Predict response variable of a model given exogenous variables.
        At first, X is transformed to trained latent components on which 
        basis logit predicts probability.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        array
            NDArray([1-p, p]).
        """
        X_ = self.transform(X, verbose)
        return super().predict_proba(X_, verbose, **kwargs) 
                             
    def _logit(self, fx: Callable) -> Callable:
        
        if not callable(fx):
            return fx
        
        def wrap(*args, **kwargs):  
            p = AigParam.get_ar_kw_values_by_sig(fx, args, kwargs)          
            
            if p.get('X') is None:                
                return fx(*args, **kwargs)
            
            p['X'] = self.transform(p.get('X'))            
            return fx(**p)
        
        return wrap 
    
class BinaryPLSBase(FitBase, BinaryPLSPredictBase):
    """
    Base class for PLS-based feature selection, transformation and prediction.
    
    This class handles the fitting of the PLS regression model, determining the 
    final set of features, and providing methods to transform new data into the 
    latent variable space. It relies on `FitWorker` for the optimization of the 
    number of components if not explicitly specified.    
    """
    pls:  PLSRegression = None
    cols: list|int = None
    npc:  int = None

    result:     list = None
    stats:       pDF = None
    optim_path:  pDF = None

    def _fit_worker(self, X:pDF|NDArray, y:pSe|ArrayLike) -> tuple[int, pDF]:

        # (a) requested n_components → skip npc optim search
        npc = self.get_setup('n_components')
        if npc is not None:
            return npc, None
        
        # (b) optim npc search
        worker = FitWorker(self._solver, self.get_setup())        
        return worker.process(X, y)
    
    def _fit(self, X:pDF|NDArray, y:pSe|ArrayLike, subset:list=None) -> None:        
        """Full PLS fitting pipeline."""
        X_ = FitData.get_data(X, subset)

        self.npc, self.optim_path = self._fit_worker(X_, y)

        self.pls    = self._fit_pls(X_, y, self.npc)
        self.cols   = FitData.get_fes(X_, subset)
        self.result = self._pls_columns(self.npc)
        self.stats  = self._stats(y)

        return self

    def _stats(self, y:pSe|ArrayLike) -> pDF:
        l = self.pls.x_loadings_.T
        d = pDF( {STAT.F: self.cols} | dict(zip(self.result, l)) )
        d[STAT.V] = self._calculate_vip()
        return d.sort_values(by=STAT.V, ascending=False)   
           
    def _calculate_vip(self) -> ArrayLike:
        """Variable Importance in Projection."""
        
        # Extract weights and scores
        t = self.pls.x_scores_      # (n_samples,  n_components)
        w = self.pls.x_weights_     # (n_features, n_components)
        q = self.pls.y_loadings_    # (n_components, ) for 1 binary target
        
        n_features, n_components = w.shape
        
        # Calculate the 'Importance' of each component (SSY_k)
        # The variance explained by each component k is (t_k't_k) * (q_k'q_k)
        ssr = np.sum(t**2, axis=0) * np.sum(q**2, axis=0)   # (n_components,)
        total_ssr = np.sum(ssr)
        
        if total_ssr == 0:
            raise ValueError(
            'Model has to be fitted on proper data prior VIP calculation '
            '→ current model has explained variance 0'
            )

        return np.sqrt(n_features * np.matmul(w**2, ssr) / total_ssr)       

    def _align_data(self, X:pDF|NDArray, verbose:bool) -> pDF|NDArray:
        X_shape = X.shape
        
        if ( not len(X_shape) == 2 ) or ( X_shape[1] < len(self.cols) ):
            raise ValueError(
                'Inconsistent input → expected array shape: '
                f'{len(self.cols)}, input data: {X_shape}'
            )
        
        if X_shape[1] > len(self.cols):
            if verbose:
                PredictSelectionMixin._subset_help(len(self.cols), X_shape[1], hint='.cols')

            return X[self.cols]

        return X
        
    def transform(self, X:pDF|NDArray, verbose:bool=False) -> pDF|NDArray:
        """
        Transforms input data into the latent PLS component space.
        
        Parameters
        ----------
        X : pDF or NDArray
            Input data to transform.
        verbose : bool
            Controls verbose logging for column alignment.
            
        Returns
        -------
        X_pls : pDF or NDArray
            The data projected onto the latent components. If input X is a 
            DataFrame, the output is a DataFrame with component names as columns.
        """        
        d = self.pls.transform( self._align_data(X, verbose) )
        if not isinstance(X, pDF):
            return d
        return pDF(d, columns = self.result, index = X.index)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class BinaryPLSRegression(BinaryPLSBase):    
    """
    Partial Least Squares Regression (PLS) wrapper optimized for binary classification.
    Once optimal number of components is found, `inference model is logit` trained on 
    chosen components.
    
    Specifically designed for WOE features with binary target: it utilizes the fact that
    Weight of Evidence features have linear relationship towards binary target.
    
    Algorithm
    ---------
    1. Sets up the cross-validation splits.
    2. Loops through component counts from 1 up to `max_components`.
    3. For each component (i-th iteration) fits a PLS regression on 
       training folds and transforms both training and validation data into 
       the latent space.
    4. Trains a Logistic Regression model on the 
       transformed training data and predicts probabilities for both training 
       and validation sets.
    5. Computes the Gini coefficient for the validation 
       set. This is averaged across all CV folds for stability.
    6. Performs DeLong's test to compare the current validation predictions 
       against the best-so-far predictions. 
       This yields a p-value indicating if the improvement is statistically 
       significant (below `alpha`).
    7. If the improvement is significant, the current component count becomes 
       the new "best". If not, a patience counter increments. 
    8. The loop terminates if loop is exhausted or the counter exceeds 
       `early_stopping`, preventing overfitting.
    
    Parameters
    ----------
    max_components : int, optional
        The maximum number of PLS components to evaluate during the search. 
        Default is 10.

    alpha : float, optional
        The significance level for the statistical test (DeLong's test) comparing 
        Gini improvements. A lower alpha requires stronger evidence for component 
        addition. Default is 0.05.
        
    cv : int, optional
        The number of cross-validation folds used to estimate model performance 
        during the component search. Default is 3.

    max_iter : int, optional
        The maximum number of iterations for the PLS solver convergence. 
        Default is 500.

    tol : float, optional
        The tolerance for the PLS solver convergence. Default is 1e-6.

    random_state : int, optional
        The seed for the random number generator used in sampling and cross-validation. 
        Default is 42.

    early_stopping : int, optional
        The number of consecutive components without a significant Gini improvement 
        before stopping the search. Default is 3.

    n_components : int, optional
        If provided, skips the optimization search and fits the PLS model with 
        exactly this number of components. Default is None.

    verbose : bool, optional
        Controls the verbosity of logging output during fitting. Default is False.

    scale : bool, optional
        Whether to scale the input features before fitting PLS. Default is False.
        With WOE Features setting scale=True is discouraged as it causes loss in 
        loadings interpretability.

    solver : Literal['pls'], optional
        Specifies the underlying estimator. Currently only 'pls' is supported. 
        Default is 'pls'.
        
    Example
    -------
    >>> model = BinaryPLSRegression(max_components=15, alpha=0.01, cv=5)
    >>> model.fit(X_woe, y_target)
    >>> # Access optimal number of components
    >>> print(model.npc)
    >>> # Transform new data
    >>> X_pls = model.transform(X_woe_test)
    >>> # Predict probabilities
    >>> probs = model.predict_proba(X_woe_test)
    """
    def __init__(self, 
        max_components:     int = 10,
        alpha:            float = 0.05,
        cv:                 int = 3, 
        max_iter:           int = 500,
        tol:              float = 1e-6,
        random_state:       int = 42,
        early_stopping:     int = 3,
        n_components:       int = None,
        verbose:           bool = False,
        scale:             bool = False,
        solver:  Literal['pls'] = 'pls',
        **kwargs
    ):  
        setup = dict(
            max_components  = max_components,
            alpha           = np.clip(alpha, 0, 1),
            cv              = np.clip(cv, 2, 20),
            max_iter        = max_iter,
            tol             = tol,
            random_state    = random_state,
            early_stopping  = early_stopping,
            n_components    = n_components,
            solver          = solver,
            scale           = scale,
            verbose         = verbose
        ) | kwargs

        super().__init__(solver, setup)

    def _set_result_model(self, X:pDF|NDArray, y:pSe|ArrayLike) -> None:
        self._set_predict_model(X, y)

    def fit(self, X:pDF|NDArray, y:pSe|ArrayLike, subset:list=None):
        """
        Fits the PLS-Logit model by optimizing the number of components 
        and training a final logistic regression on the latent variables.

        Parameters
        ----------
        X : DataFrame or ArrayLike
            The matrix of exogenous features (typically WOE-transformed).
        y : Series or ArrayLike
            The binary target variable.
        subset : list, optional
            An optional list of column names or indices to select a subset of 
            features from X before fitting.
            
        Returns
        -------
        self : BinaryPLSRegression
            The fitted estimator instance.
        """
        self._fit(X, y, subset)
        self._set_result_model(X, y)
        return self
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PLS FEATURE SELECTOR | WOE-LOGIT
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class SelectorBase(ModelBase):

    implemented_estimators = {
        'pls':  BinaryPLSRegression,
    }

    result:     list = None
    stats:      dict = None
    optim_path: dict = None
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SelectorWorker(SelectorBase):
    """Internal worker class for Sequential Feature Selection using PLS."""

    @staticmethod
    def _features_order(stats:pDF, cutoff:float) -> list:
        """
        Orders features by Variable Importance in Projection (VIP) scores 
        and filters out those below a certain quantile cutoff.        
        """
        s = stats.sort_values(STAT.V, ascending=False)
        m = s[STAT.V] > s[STAT.V].quantile(cutoff)
        return s.loc[m, STAT.F].to_list()
    
    def _define_search(self, X:pDF|NDArray, y:pSe|ArrayLike) -> None:
        """Initializes the search space by fitting an initial PLS model."""                
        pls: BinaryPLSRegression = self.get_estimator_instance(verbose=False)                
        pls.fit(X, y)
        n = max(2, pls.npc)
        f = self._features_order(pls.stats, self.get_setup('prior_vip_cutoff'))
        self.stats = {
            GINI.N: n,
            STAT.B: f[:n],
            STAT.S: f[n:],
        }
        self._f = int( pSe(f[n:]).str.len().max() )
    
    def _fit_pred(self, data: SampleData, fes: list, npc: int) -> ArrayLike:        
        """
        Fits a PLS model on a specific subset of features and generates 
        predictions on the validation set.
        """
        fes = np.array(fes)
        pls: BinaryPLSRegression = self.get_estimator_instance(npc)                
        pls.fit(data.X_train[fes], data.y_train)
        return pls.predict_proba(data.X_valid[fes])[:,1]
    
    def _get_samples(self,  X:pDF|NDArray, y:pSe|ArrayLike) -> SampleData:
        """Creates a train/validation split for internal model evaluation."""
        s = self.get_setup()
        v = np.clip(1 / s['cv'], 0.25, 0.5)
        samples = FitData.train_valid_split(X, y, v, s['random_state'])
        return FitData.get_sample_data(*samples)
    
    def _info(self, f:str) -> None:
        if not self.get_setup('verbose'):
            return
        p = self.optim_path[f]
        a = lambda x: 10 if x==GINI.P else 7
        i = [f'{k}: {v}{Str.align(a(k), v)}' for k,v in p.items()]
        print(f, Str.align(self._f, f), '►', ' |  '.join(i))

    def _vector(self, f:str, g:float, pv:float=None, d:float=None, s:bool=None):
        if f == STAT.B:
            self.optim_path = {}

        r = lambda x, r=4: x if x is None else float(round(x,r))

        self.optim_path[f] = {
            GINI.V: r(g), 
            GINI.P: r(pv,7), 
            GINI.D: r(d), 
            GINI.S: bool(s) if s is not None else s
        }
        self._info(f)

    def _search(self, X:pDF|NDArray, y:pSe|ArrayLike) -> list:
        """Executes the full iterative feature selection pipeline."""
        
        self._define_search(X, y)

        selected: list = self.stats[STAT.B].copy()
        search:   list = self.stats[STAT.S]
        npc:      int  = self.stats[GINI.N]
        
        alpha   = self.get_setup('alpha')
        samples = self._get_samples(X, y)

        p0 = self._fit_pred(samples, selected, npc)
        g0 = Gini.fast_calc(samples.y_valid, p0)
        self._vector(STAT.B, g0)        
        
        for f in search:

            p1 = self._fit_pred(samples, selected + [f], npc)
            g1 = Gini.fast_calc(samples.y_valid, p1)
            pv = TestModelDiff(samples.y_valid, p0, p1).delong()            

            self._vector(f, g1, pv, g1-g0, pv < alpha)

            if ( pv < alpha ) and ( g1 > g0 ):
                selected.append(f)
                p0 = p1
                g0 = g1
        
        return selected

    def fit(self, X:pDF|NDArray, y:pSe|ArrayLike, subset:list=None) -> None:
        """Runs the feature selection process."""            
        X_ = FitData.get_data(X, subset)
        self.result = self._search(X_, y)
        return self
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PLSSelector(SelectorBase, PredictBase):
    """
    Statistically Optimized Feature Selection for Logistic Regression with WOE driven by
    PLS Projection and Delong Statistical Significance with a main aim as noise-filtering, 
    variance-reducing, stability-enhancing selector.

    This estimator performs forward-stepwise feature selection. Unlike standard greedy 
    selectors that optimize a monolithic loss function, this selector evaluates candidates 
    by assessing the statistical significance of their contribution to the multivariate 
    covariance structure between X and y.

    The algorithm iteratively adds features that significantly improve the predictive power 
    of a Partial Least Squares (PLS) model. It handles multicollinearity by projecting 
    features into orthogonal latent components before assessing the marginal utility of 
    each new feature via hypothesis testing (e.g., DeLong test for AUC/Gini).

    Parameters
    ----------
    max_components : int, optional
        The maximum number of PLS components to evaluate during the search. 
        Default is 10.

    alpha : float, optional
        The significance level for the statistical test (DeLong's test) comparing 
        Gini improvements. A lower alpha requires stronger evidence for component 
        addition. Default is 0.05.
        
    cv : int, optional
        The number of cross-validation folds used to estimate model performance 
        during the component search. Default is 3, max 10.

    max_iter : int, optional
        The maximum number of iterations for the PLS solver convergence. 
        Default is 500.

    tol : float, optional
        The tolerance for the PLS solver convergence. Default is 1e-6.

    prior_vip_cutoff : float, optional
        Reduce the feature search space by chosen quantile such as eliminate FE
        if feature VIP <  VIP.quantile(prior_vip_cutoff).
        Default is 0.2.

    random_state : int, optional
        The seed for the random number generator used in sampling and cross-validation. 

    early_stopping : int, optional
        The number of consecutive components without a significant Gini improvement 
        before stopping the search. Default is 3.

    n_components : int, optional
        If provided, skips the optimization search and fits the PLS model with 
        exactly this number of components. Default is None.

    verbose : bool, optional
        Controls the verbosity of logging output during fitting. Default is False.

    scale : bool, optional
        Whether to scale the input features before fitting PLS. Default is False.
        With WOE Features setting scale=True is discouraged as it causes loss in 
        loadings interpretability.

    solver : Literal['pls'], optional
        Specifies the underlying estimator. Currently only 'pls' is supported. 
        Default is 'pls'.

    Notes
    -----
    **Mathematical Formulation**

    1. PLS Projection:
       Let X be the feature matrix and y be the target. Let S_k be the set of 
       selected features at iteration k.
       
       PLS constructs orthogonal latent components T from X_{S_k U {f}} by 
       maximizing covariance between X and y:
       
           w_j = argmax_w Cov(Xw, y)  subject to ||w||=1
       
       The scores are t_j = Xw_j.

    2. Hypothesis Testing:
       For each candidate feature f not in S_k, we define two models:
       
       - Null Model (M0): Fit on S_k.
       - Alternative Model (M1): Fit on S_k U {f}.
       
       We compute performance scores Score_0 and Score_1.
       The null hypothesis H0 is that Score_1 <= Score_0.
       
       We calculate the p-value p_f using the specified test statistic 
       (DeLong's test for AUC difference on binary target):
       
           p_f = P(Delta_Score >= Score_1 - Score_0 | H0)

    3. Selection Criterion:
       Feature f* is added to S_k if:
       
           f* = argmax_f Score_1(f)  such that  p_f < alpha

       The process repeats until no remaining feature satisfies the significance threshold.

    **Algorithmic Logic**

    1. Initialize: Fit a baseline PLS model on available data to establish 
       - search space by order features given importance of init PLS VIP as S,
       - base set of selected features B,
       - set number of partial latent components (npc) given found optin by init PLS,
       - set initial performance Score_0.
    
    2. Reduce: Reduce the search space S by chosen prior_vip_cutoff such as 
       eliminate FE if `feature VIP <  VIP.quantile(prior_vip_cutoff)`.

    3. Iterate: For each candidate feature f in the remaining pool:
       
       a. Construct Candidate Set: Form B_temp = B U {f}.
       
       b. Refit PLS: Train the PLS model on X_{B_temp}. This re-optimizes the 
          latent components, ensuring the new feature's contribution is evaluated 
          in the context of the existing orthogonal structure.
       
       c. Evaluate Improvement: Calculate Score_1 for the new model.
       
       d. Statistical Validation: Compute the p-value for the improvement 
          Delta_Score = Score_1 - Score_0 using the specified test statistic.
       
       e. Decision: If p < alpha and Score_1 > Score_0.
       
    4. Update: Add the best feature to B, update Score_0 = Score_1, 
       and remove the feature from the candidate pool.

    5. Termination: Stop when no remaining feature provides a statistically 
       significant improvement.
    """
    def __init__(self, 
        max_components:     int = 10,
        alpha:            float = 0.05,
        cv:                 int = 3, 
        max_iter:           int = 500,
        tol:              float = 1e-6,
        prior_vip_cutoff: float = 0.2,
        random_state:       int = 42,
        early_stopping:     int = 3,
        n_components:       int = None,
        verbose:           bool = False,
        scale:             bool = False,
        solver:  Literal['pls'] = 'pls',
        **kwargs
    ):  
        setup = dict(
            max_components   = max_components,
            alpha            = np.clip(alpha, 0, 1),
            cv               = np.clip(cv, 2, 10),
            max_iter         = max_iter,
            tol              = tol,
            prior_vip_cutoff = prior_vip_cutoff,
            random_state     = random_state,
            early_stopping   = early_stopping,
            n_components     = n_components,
            solver           = solver,
            scale            = scale,
            verbose          = verbose
        ) | kwargs

        super().__init__(solver, setup)

    def _set_result_model(self, X:pDF|NDArray, y:pSe) -> None:
        self._set_predict_model(X[self.result], y)

    def _fit(self, X:pDF|NDArray, y:pSe|pDF, subset:list=None) -> None:
        worker = SelectorWorker(self._solver, self.get_setup())
        worker.fit(X, y, subset)        
        self.result     = worker.result
        self.stats      = worker.stats
        self.optim_path = worker.optim_path
        return self

    def fit(self, X:pDF|NDArray, y:pSe|pDF, subset:list=None) -> None:
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