"""
Feature Pruning — Shadow Importance & Null Distribution Analysis.

This module implements a noise-filtering framework that identifies
truly predictive features by comparing them against synthetic
"shadow" versions of themselves. By permuting feature values to
create a baseline of random noise, the pruner establishes a
statistical null distribution of importance. Only features whose
actual importance significantly exceeds a high quantile of this
noise floor are retained.

Methodology
-----------
1. **Shadow Creation**: For each candidate feature, create shadow
   copies by randomly permuting its values (breaking the signal
   while preserving marginal distributions).

2. **Combined Fit**: Fit the model on the original + shadow
   features simultaneously. The model "sees" both the real signal
   and the noise.

3. **Importance Computation**: Measure feature importance using a
   chosen solver:
   - **SHAP**: Marginal contribution to the prediction (model-
     specific, additive).
   - **Permutation**: Performance penalty when shuffled (model-
     agnostic, sensitivity-based).

4. **Null Distribution**: The shadow features' importance values
   form the null distribution — what importance looks like when
   there is no true signal.

5. **Threshold Selection**: Features whose importance exceeds a
   high quantile (e.g., 90th or 95th percentile) of the shadow
   distribution are retained. This is a data-driven, relative
   threshold — it adapts to the model, the features, and the
   sample.

6. **Stabilization**: The above is repeated across multiple
   shadow samples and CV folds. The final threshold is the
   average across all iterations, and the final decision is
   based on the mean importance exceeding the mean threshold.

Why Shadow Pruning?
-------------------
- **Spurious Correlation Filter**: Features that appear
  significant due to overfitting or chance correlation will have
  importance close to the shadow baseline. They are removed.
- **Data-Driven Threshold**: Unlike fixed p-value cutoffs, the
  threshold adapts to the specific model, feature set, and data.
- **Stability**: Multiple shadow samples + CV folds produce a
  robust estimate of the noise floor, reducing the sensitivity
  to any single random seed.
- **Model-Agnostic**: Works with any estimator (GLM, GBM) and
  any importance metric (SHAP, permutation).

Gini Path
---------
After pruning, the module computes the incremental Gini
trajectory: starting from the most important feature and
adding features one at a time (in importance order), it tracks
how Gini evolves on both training and validation sets. This
provides direct evidence of each feature's marginal contribution
to model performance — and reveals whether later features add
value or merely add noise.
"""

from ...deps.data           import  np, pDF, pSe
from ...deps.plots          import  go, make_subplots
from ...deps.display        import  display
from ...deps.types          import  Any

from ...core.factory        import  (
                                    DictClass, 
                                    Dataclass
                                    )
from ...ml.models           import  (
                                    Logit,
                                    LogReg,
                                    Lasso, 
                                    ElasticNet,
                                    XGB,
                                    LGB
                                    )
from ...ml.sampling         import  (
                                    Sampling,
                                    DatasetCV, 
                                    DataUtils
                                    )
from ...ml.metrics          import  Gini

from .estimator             import  FeatureEstimator
from .sampling              import  SamplingMixin
from .importance            import  (
                                    LBL, MTD,
                                    FeatureAttributionSHAP, 
                                    FeaturePermutation
                                    )


type InputModel = Logit | LogReg | Lasso | ElasticNet | XGB | LGB
type PrunerSolver = FeatureAttributionSHAP | FeaturePermutation


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class RES(DictClass):
    F  = 'final_model'
    M  = 'Average'
    C  = 'Choice'
    Y = "✅"
    N = "❌"

class GINI(DictClass):
    T = 'train'
    V = 'valid'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FEATURE PRUNING
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬   
@Dataclass.base
class PrunerStep:
    """
    Single iteration result of the shadow pruning loop.

    Attributes
    ----------
    importance : pDF
        Per-feature importance values (original features only).
    shadow_th : float
        Shadow threshold for this iteration (quantile of the
        shadow importance distribution).
    org_min : float
        Minimum importance among original features (for
        diagnostic context).
    shadow_i : int | str
        Shadow sample index (or `RES.F` for the final
        aggregated step).
    cv_i : int | None
        CV fold index (None for the final step).
    is_penalty : bool | None
        Whether the importance metric is a penalty (higher =
        worse). SHAP: False (higher = better). Permutation:
        True (higher = more penalty).
    """    
    importance: pDF
    shadow_th:  float
    org_min:    float
    shadow_i:   int|str
    cv_i:       int = None
    is_penalty: bool = None

@Dataclass.base
class GiniPath:
    """
    Incremental Gini trajectory across features.

    Shows how model Gini evolves as features are added one at
    a time (in importance order). Directly reveals each
    feature's marginal contribution to performance.

    Attributes
    ----------
    train_valid : pDF
        Gini values per feature (rows) for both train and
        validation sets (columns: `'train'`, `'valid'`).
    result_fes : list
        Features retained by the pruner (in importance order).
    removed_fes : list
        Features removed by the pruner (in importance order,
        after the retained set).
    """    
    train_valid: pDF
    result_fes:  list
    removed_fes: list    

class PrunerResult:
    """
    Iteration tracker and optimization path storage.

    Maintains the full diagnostic history of the pruning loop:
    per-iteration importance, shadow thresholds, and the final
    aggregated decision.
    """

    _solver_col = {
        MTD.S: LBL.S,
        MTD.P: LBL.P
    }

    def __init__(self, solver:MTD, verbose:bool=True):
        self.solver = solver
        self.verbose = verbose 
        self.optim_path: dict[str, PrunerStep] = {} 
    
    def get_solver_col(self) -> str:
        return self._solver_col[self.solver]
    
    def update_optim_path(self, step:PrunerStep):
        """
        Record a pruning iteration in the optimization path.

        Parameters
        ----------
        step : PrunerStep
            The iteration result to store.
        """        
        s = f'shadow_{step.shadow_i}'
        i = f'cv_{step.cv_i}'
        
        if not self.optim_path.get(s):
            self.optim_path[s] = {}

        if step.shadow_i == RES.F:
            s = step.shadow_i
            self.optim_path[s] = step
        else:
            self.optim_path[s][i] = step

        self.info(self.iter_info(step))
        
    def info(self, info:str):
        if not self.verbose:
            return
        print(info)

    @classmethod
    def iter_info(cls, step:PrunerStep) -> str:
        if step.shadow_i == RES.F:
            return cls._iter_info_f(step)
        return cls._iter_info_i(step)
    
    @staticmethod
    def _iter_info_s(step:PrunerStep) -> tuple:
        return (step.shadow_i, step.shadow_th)
    
    def _iter_info_th(th:float, m:float) -> str:
        return f'shadow cutoff: {th:.3f} | data min: {m:.3f}'
    
    @classmethod
    def _iter_info_i(cls, step:PrunerStep) -> str:
        s,th = cls._iter_info_s(step)
        i = step.cv_i
        return f'Iteration {s}-{i} ► Obtained {cls._iter_info_th(th, step.org_min)}'
    
    @classmethod
    def _iter_info_f(cls, step:PrunerStep) -> str:
        s,th = cls._iter_info_s(step)
        im   = step.importance
        fes  = f' ► Selection {sum(im[RES.M] > th)} from {len(im)} features'
        return f'► {s} ►  Average {cls._iter_info_th(th, step.org_min)}{fes}'     
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PrunerData:
    """
    Shadow data generation utility.

    Creates the shadow (permutation) copies of features and
    the corresponding cross-validation dataset structures
    needed for the pruning loop.
    """

    def __init__(self, 
        n_permutations: int = 3,
        cv:             int = 3, 
        random_state:   int = 42,
        shuffle:       bool = False,
        fe_subset:     list = None
    ):
        self.n_iter     = n_permutations
        self.cv         = cv
        self.random     = random_state    
        self.shuffle    = shuffle
        self.fe_subset  = fe_subset
    
    @staticmethod
    def get_orig_fes(X:pDF, subset:list=None) -> list:
        """Resolve the original feature names."""        
        return DataUtils.get_fes(X, subset) 

    def create_shadow(self, X:pDF) -> list:
        """
        Create shadow (permuted) copies of the feature matrix.

        Each shadow copy is an independent random permutation of
        the original feature values. The number of copies equals
        `n_permutations`.

        Parameters
        ----------
        X : pDF
            Feature matrix.

        Returns
        -------
        list[pDF]
            List of shadow DataFrames (same shape as X).
        """        
        fes = self.get_orig_fes(X, self.fe_subset)
        shadow_dfs = []
        for i in range(max(1, self.n_iter)):            
            shadow_dfs.append(
                DataUtils.extend_by_shadow_df(X[fes], self.random + i)
            )
        return shadow_dfs
    
    def create_shadow_cv(self, X:pDF, y:pSe) -> dict[int, DatasetCV]:
        """
        Create shadow DataFrames with CV dataset structure.

        Each shadow copy is wrapped in a `DatasetCV` instance
        with the configured number of folds and shuffling.

        Parameters
        ----------
        X : pDF
            Feature matrix.
        y : pSe
            Target series.

        Returns
        -------
        dict[int, DatasetCV]
            Shadow index → CV dataset mapping.
        """        
        shadow_df = self.create_shadow(X)
        shadow_cv = {}
        for i, shadow_X in enumerate(shadow_df):
            cv = DatasetCV(self.cv, random_state=self.random, shuffle=self.shuffle)
            cv.set_data(shadow_X, y)
            shadow_cv[i] = cv
        return shadow_cv          
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PrunerFitter:
    """
    Fitting engine of shadow pruning.

    Implements the core pruning loop:
    1. For each shadow sample × CV fold:
       - Fit the model on original + shadow features (train).
       - Compute importance on the evaluation set (valid).
       - Extract the shadow threshold (quantile of shadow
         importances).
    2. Aggregate: average importance across iterations vs.
       average threshold.
    3. Decision: retain features whose mean importance exceeds
       the mean shadow threshold.

    The model used for importance computation is a **proxy
    model** (fitted on the original + shadow features) — not
    the production model. This ensures the importance
    measurement is fair: the model "sees" both signal and
    noise, and the ranking reveals which features carry
    genuine information.

    Parameters
    ----------
    model : InputModel | None, default=None
        The model architecture to use as the proxy for
        importance computation. Should match the type of the
        production model (e.g., `Logit` for GLM agents,
        `LGB` for GBM agents).
    solver : str, default='shap'
        Importance metric: `'shap'` (model-specific marginal
        contribution) or `'permutation'` (model-agnostic
        sensitivity penalty).
    n_permutations : int, default=3
        Number of shadow samples. Higher = more stable null
        distribution at the cost of computation.
    cv : int, default=3
        Number of cross-validation folds. Higher = more stable
        threshold at the cost of computation.
    random_state : int, default=42
        Seed for reproducibility (shadow permutations, CV
        splits, subsampling).
    verbose : bool, default=True
        Enable progress display during the pruning loop.    
    """

    implemented_estimators = {
        MTD.S: dict(solver = FeatureAttributionSHAP, direction_up = True  ),
        MTD.P: dict(solver = FeaturePermutation,     direction_up = False )
    }

    def __init__(self,
        model:          InputModel,
        solver:         str = MTD.S,
        n_permutations: int = 3,
        cv:             int = 3, 
        random_state:   int = 42,
        verbose:       bool = True
    ):
        self._set_setup(solver, random_state)
        self.model   = model
        self.n_iter  = n_permutations
        self.cv      = cv
        self.rs      = random_state
        self.verbose = verbose
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _set_init(self) -> None:
        self.result     = []
        self._iter_res  = PrunerResult(self.solver, self.verbose)
        self.optim_path = self._iter_res.optim_path

    def _get_solver(self, solver:str) -> Any:
        return self.implemented_estimators[solver]['solver']
    
    def _get_solver_direction(self, solver:str) -> bool:
        return self.implemented_estimators[solver]['direction_up']

    def _set_setup(self, solver:str, random_state:int) -> None:
        setup = dict(
            random_state = random_state,
            verbose      = False
        )                
        self.solver = solver        
        self.setup  = {'estimator': self._get_solver(solver), 'setup': setup}
    # ─────────────────────────────────────────────────────────────────────────────────────────────        
    
    def get_estimator(self) -> Any:
        """Return the importance estimator class."""        
        return self.setup['estimator']
    
    def get_setup(self) -> dict:
        """Return the estimator constructor parameters."""        
        return self.setup['setup']
    
    def get_solver_direction(self) -> bool:
        """
        Return whether higher importance = more important.

        True for SHAP (higher = better). False for Permutation
        (higher penalty = worse; internally inverted).
        """        
        return self._get_solver_direction(self.solver)
    
    def get_estimator_instance(self, **kwargs) -> PrunerSolver:
        """
        Instantiate the importance estimator.

        Parameters
        ----------
        **kwargs
            Additional parameters forwarded to the estimator
            constructor.

        Returns
        -------
        PrunerSolver
            The estimator instance (not yet fitted).
        """        
        e = self.get_estimator()
        s = self.get_setup() | kwargs
        return e(**s) 
    
    def get_importance(self) -> pDF:
        """
        Return the final importance table ordered by descending importance.

        Returns
        -------
        pDF
            One row per feature with importance values across
            all shadow iterations, the average, and the
            retain/drop decision.
        """
        return self.optim_path[RES.F].importance.sort_values(RES.M, ascending=False)
    
    def get_ordered_importance_fes(self, subset:list=None) -> list:
        """
        Return features in descending importance order.

        Parameters
        ----------
        subset : list | None, default=None
            If provided, return only the features in this
            subset (preserving the importance ordering).

        Returns
        -------
        list
            Feature names, most important first.
        """
        fes = self.get_importance().index.to_list()
        if subset is None:
            return fes
        return [x for x in fes if x in subset]
        
    def is_fitted(self) -> bool:
        """Check whether the pruner has been fitted."""        
        if hasattr(self, 'optim_path'):
            return True
        return False
    
    def _verify_fitted(self) -> None:
        if self.is_fitted():
            return
        raise ValueError('Model must be fitted by .fit() prior calling results-plots.')    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    #             
    def _get_input_model_instance(self, X:pDF, is_shadow:bool = True) -> InputModel:
        est = FeatureEstimator(self.model, is_over_fit = True, is_shadow = is_shadow)
        return est.get_estimator_instance(X)
    
    def _fit_input_model(self, X:pDF, y:pSe, is_shadow:bool = True) -> InputModel:
        m = self._get_input_model_instance(X, is_shadow)
        return m.fit(X, y)

    def _fit_model(self, X_train:pDF, y_train:pSe, X_valid:pDF, y_valid:pSe) -> PrunerSolver:
        input_model = self._fit_input_model(X_train, y_train)
        pruner = self.get_estimator_instance(model=input_model)
        return pruner.fit(X_valid, y_valid)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _fit_iter_importance(self, cv_sample:tuple ) -> pDF:
        im = self._fit_model(*cv_sample).result
        if not self.get_solver_direction():
            c = self._iter_res.get_solver_col()
            im[c] = im[c]*-1
        return im
    
    def _get_iter_imp_by_fes(self, importance:pDF, org_fes:list) -> tuple[pDF, pSe, str]:
        c   = self._iter_res.get_solver_col()
        m   = importance[LBL.F].isin(org_fes)
        org = importance.loc[m]
        sh  = importance.loc[~m,c]
        return org, sh, c

    def _get_iter_shadow_th(self, shadow:pSe, quantile:float) -> float:
        q = np.clip(quantile, 0, 1)
        return np.quantile(shadow, q) 
        
    def _iter_shadow_importance(self, cv_sample:tuple, org_fes:list, quantile:float, s:int, i:int
    ) -> PrunerStep:
        im = self._fit_iter_importance(cv_sample)
        org, sh, c = self._get_iter_imp_by_fes(im, org_fes)
        th = self._get_iter_shadow_th(sh, quantile) 

        return PrunerStep(
            importance = org, 
            shadow_th  = th, 
            org_min    = float( org[c].min() ), 
            shadow_i   = s,
            cv_i       = i,
            is_penalty = self.get_solver_direction()
        )

    def _iter_model(self, cv_sample:tuple, org_fes:list, quantile:float, s:int, i:int) -> float:
        step = self._iter_shadow_importance(cv_sample, org_fes, quantile, s, i)
        self._iter_res.update_optim_path(step)
        return step.shadow_th
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _iter_agg(self, org_fes:list, th:list) -> tuple[pDF, float, pSe]:
        op:dict[str, dict[str, PrunerStep]] = self.optim_path
        c = self._iter_res.get_solver_col()
        
        d = pDF(index = org_fes)        
        for s, cv in op.items():
            for i, ps in cv.items():
                d[s+'_'+i] = ps.importance.set_index(LBL.F)[c]
        
        d[RES.M]  = d.mean(axis=1)
        th_mean   = np.average(th)
        choice    = d[RES.M] > th_mean 
        d[RES.C]  = np.where(choice, RES.Y, RES.N)

        return d, th_mean, choice
    
    @staticmethod
    def _final_features(im_agg_chosen:pDF) -> list:
        return im_agg_chosen.index.to_list() #.sort_values(RES.M, ascending=False)
        # more consistent for pruning is to not sort result by importance

    def _final_model(self, org_fes:list, th:list) -> list:
        im_agg, th_mean, choice_mask = self._iter_agg(org_fes, th)

        self._iter_res.update_optim_path(
            PrunerStep(
                importance = im_agg, 
                shadow_th  = th_mean, 
                org_min    = float( im_agg[RES.M].min() ),
                shadow_i   = RES.F,
                is_penalty = self.get_solver_direction()
        ))
        
        return self._final_features(im_agg[choice_mask])
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def process(self, datasets:dict[int, DatasetCV], org_fes:list, quantile:float) -> list:
        self._set_init()

        th = []
        for s, cv in datasets.items():
            for i in cv.get_cv_range():
                th.append( 
                    self._iter_model( cv.get_cv_sample(i), org_fes, quantile, s, i )
                )
                
        self.result = self._final_model(org_fes, th) 

        return self.result
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PrunerGini:
    """
    Incremental Gini Path Computation.

    After pruning, computes the Gini trajectory as features are
    added one at a time (in importance order). This provides
    direct evidence of each feature's marginal contribution to
    model performance:

    - **Steep initial rise**: The most important features add
      substantial Gini.
    - **Plateau**: Later features add little beyond the preceding
      ones (redundancy).
    - **Divergence (train vs. valid)**: If train Gini rises but
      valid Gini stalls, later features are overfitting.

    Parameters
    ----------
    fitter : PrunerFitter
        The fitted pruner fitter - source of feature ordering and model.
    val_size : float
        Validation fraction for the train/valid split.
    """

    def __init__(self, fitter:PrunerFitter, val_size:float):
        self.fitter = fitter
        self.val_size = val_size
        self.fitter._verify_fitted()
    
    def get_fes(self) -> tuple[list, list]:
        """
        Resolve the feature ordering: retained first, then removed.

        Returns
        -------
        tuple[list, list]
            (retained_features, removed_features) — both in
            importance order.
        """        
        fes = self.fitter.get_ordered_importance_fes()
        res = [x for x in fes if x in self.fitter.result] # importance order
        ext = [x for x in fes if x not in res]
        return res, ext
    
    def fit_model(self, Xy_train:tuple) -> InputModel:
        """Fit the proxy model on the training subset."""        
        return self.fitter._fit_input_model(*Xy_train, is_shadow=False)

    def train_valid_split(self,  X:pDF, y:pSe) -> tuple[ tuple[pDF,pSe], tuple[pDF,pSe] ]:
        """
        Stratified train/validation split.

        Parameters
        ----------
        X : pDF
            Feature matrix.
        y : pSe
            Target series.

        Returns
        -------
        tuple
            ((X_train, y_train), (X_valid, y_valid))
        """        
        return Sampling(
            sample_size  = self.val_size,
            stratify     = y,
            random_state = self.fitter.rs,
            shuffle      = True
        ).train_valid_split(X,y)

    @staticmethod
    def get_Xy_subset(Xy:tuple[pDF,pSe], fes_subset:list) -> tuple[pDF,pSe]:
        """Subset the feature matrix to the specified columns."""        
        X, y = Xy
        return X[fes_subset], y
    
    @classmethod
    def get_train_valid_subset(cls,  Xy_train_valid:tuple, fes_subset:list) -> tuple:
        """Subset both train and valid splits to the specified features."""        
        return (cls.get_Xy_subset(Xy, fes_subset) for Xy in Xy_train_valid)

    @staticmethod
    def gini_calc(model:InputModel, Xy:tuple[pDF,pSe]) -> float: 
        """Compute Gini for a model on a data sample."""        
        X,y = Xy
        y_h = model.predict_proba(X)[:,1]
        return Gini.calc(y, y_h)

    def gini_train_valid(self, Xy_train_valid:tuple, fes_subset:list) -> dict:
        """
        Fit the model on the feature subset and compute train/valid Gini.

        Parameters
        ----------
        Xy_train_valid : tuple
            ((X_train, y_train), (X_valid, y_valid))
        fes_subset : list
            Feature subset for this iteration.

        Returns
        -------
        dict
            `{'train': float, 'valid': float}`
        """        
        Xy_train, Xy_valid = self.get_train_valid_subset(Xy_train_valid, fes_subset)
        model = self.fit_model(Xy_train)        
        return {
            GINI.T: self.gini_calc(model, Xy_train),
            GINI.V: self.gini_calc(model, Xy_valid)
        }
    
    def process(self, X:pDF, y:pSe) -> GiniPath:
        """
        Compute the full incremental Gini trajectory.

        Iteratively adds features (in importance order), fitting
        the model at each step and recording train/valid Gini.

        Parameters
        ----------
        X : pDF
            Feature matrix.
        y : pSe
            Target series.

        Returns
        -------
        GiniPath
            The complete Gini trajectory with feature lists.
        """        
        res, ext = self.get_fes()
        fes = res + ext
        
        Xy_tv = self.train_valid_split(X, y)

        gini = {}
        for i, fe in enumerate(fes, start=1):
            gini[fe] = self.gini_train_valid(Xy_tv, fes[:i])
        
        return GiniPath(
            train_valid = pDF(gini).T,
            result_fes  = res,
            removed_fes = ext
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PrunerBase:
    """
    Base class for shadow feature pruning.

    Provides the common contract: solver resolution, fitter
    instantiation, result access, and optimization path storage.
    Subclasses implement the fitting loop.

    Supported Solvers
    -----------------
    - **SHAP** (`MTD.S`): Model-specific marginal contribution.
      Higher importance = more predictive.
    - **Permutation** (`MTD.P`): Model-agnostic sensitivity
      penalty. Lower importance = less penalty = more
      important (direction is inverted internally).
    """
    def __init__(self,
        model:   InputModel = None,
        solver:         str = MTD.S,
        n_permutations: int = 3,
        cv:             int = 3, 
        val_size:     float = 0.33,
        random_state:   int = 42,
        shuffle:       bool = True,
        verbose:       bool = True
    ):     
        self.fitter   = PrunerFitter(model, solver, n_permutations, cv, random_state, verbose)
        self.shuffle  = shuffle
        self.val_size = val_size
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @property
    def result(self):
        """Returns list of chosen Features - without pruned."""
        return self.fitter.result
    
    @property
    def features(self):
        """Returns list of chosen Features - without pruned."""
        return self.result

    @property
    def optim_path(self):
        return self.fitter.optim_path
    
    def get_figure(self) -> Any:
        """Return the pruning visualization figure."""
        return PrunerPlot(self.optim_path[RES.F]).create()
    
    def get_importance(self) -> pDF:
        """
        Return the final importance table ordered by descending importance.

        Returns
        -------
        pDF
            One row per feature with importance values across
            all shadow iterations, the average, and the
            retain/drop decision.
        """
        return self.fitter.get_importance()
    
    def get_ordered_importance_fes(self, subset:list=None) -> list:
        """
        Return features in descending importance order.

        Parameters
        ----------
        subset : list | None, default=None
            If provided, return only the features in this
            subset (preserving the importance ordering).

        Returns
        -------
        list
            Feature names, most important first.
        """
        return self.fitter.get_ordered_importance_fes(subset)

    def show_result(self) -> None:
        """Display the formatted importance table."""
        display(self.get_importance())
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeaturesShadowPruner(PrunerBase, SamplingMixin):
    """
    Shadow Feature Pruner — Noise Floor Filtering.

    Identifies truly predictive features by comparing their
    importance against a statistical null distribution (shadow
    baseline). Features that do not significantly exceed the
    noise floor are pruned.

    How It Works
    ------------
    1. Create shadow copies of features (random permutations).
    2. Fit the model on original + shadow features (proxy model).
    3. Compute importance (SHAP or Permutation) on the evaluation
       set.
    4. The shadow features' importance forms the null
       distribution.
    5. Retain features whose importance exceeds a high quantile
       of the null distribution.
    6. Repeat across multiple shadow samples × CV folds for
       stability.
    7. Final decision: mean importance > mean shadow threshold.

    Gini Path (Post-Pruning)
    ------------------------
    After pruning, the module computes the incremental Gini
    trajectory: features are added one at a time (in importance
    order), and train/valid Gini is recorded at each step. This
    provides direct evidence of each feature's marginal
    contribution and reveals whether later features add value or
    merely add overfitting.

    Parameters
    ----------
    model : InputModel | None, default=None
        The model architecture to use as the proxy for
        importance computation. Should match the type of the
        production model (e.g., `Logit` for GLM agents,
        `LGB` for GBM agents).
    solver : str, default='shap'
        Importance metric: `'shap'` (model-specific marginal
        contribution) or `'permutation'` (model-agnostic
        sensitivity penalty).
    n_permutations : int, default=3
        Number of shadow samples. Higher = more stable null
        distribution at the cost of computation.
    cv : int, default=3
        Number of cross-validation folds. Higher = more stable
        threshold at the cost of computation.
    val_size : float, default=0.33
        Validation fraction for the Gini path computation.
    random_state : int, default=42
        Seed for reproducibility (shadow permutations, CV
        splits, subsampling).
    shuffle : bool, default=True
        Enable shuffling in CV folds.
    verbose : bool, default=True
        Enable progress display during the pruning loop.
    """
    
    def create_shadow_data(self, X:pDF, y:pSe, subset:list=None) -> dict[int, DatasetCV]:
        """
        Create the shadow CV dataset structure.

        Parameters
        ----------
        X : pDF
            Feature matrix.
        y : pSe
            Target series.
        subset : list | None
            Optional subset of features to prune.

        Returns
        -------
        dict[int, DatasetCV]
            Shadow samples with CV structure.
        """        
        data = PrunerData(
            self.fitter.n_iter, self.fitter.cv, self.fitter.rs, self.shuffle, subset
        )
        return data.create_shadow_cv(X, y)

    def fit(self, X:pDF, y:pSe, shadow_quantile:float=0.9, subset:list=None):
        """
        Execute the full shadow pruning pipeline.

        Steps:
        1. Subsample (if large dataset).
        2. Create shadow CV datasets.
        3. Run the pruning loop (shadow × CV iterations).
        4. Aggregate and make the final decision.
        5. Compute the incremental Gini path.

        Parameters
        ----------
        X : pDF
            Feature matrix (candidate features).
        y : pSe
            Target series.
        shadow_quantile : float, default=0.9
            Quantile of the shadow importance distribution used
            as the pruning threshold. A feature must exceed this
            quantile of the shadow baseline to survive. Higher
            = stricter pruning (fewer features retained).
        subset : list | None, default=None
            Optional subset of features in X to prune. If None,
            all features are considered.

        Returns
        -------
        self
            Fitted instance with `.result` (retained features),
            `.optim_path` (full iteration history), and
            `.gini_path` (incremental Gini trajectory) set.
        """
        params   = self.fitter.get_setup()
        X_, y_   = self._get_subsample(X, y, params)
        datasets = self.create_shadow_data(X_, y_, subset)
        org_fes  = PrunerData.get_orig_fes(X_, subset)

        self.fitter.process(datasets, org_fes, shadow_quantile)
        self.gini_path = self.fe_performance(X_, y_)
        
        return self
        
    def fe_performance(self, X:pDF, y:pSe) -> GiniPath:
        """
        Compute the incremental Gini trajectory.

        Parameters
        ----------
        X : pDF
            Feature matrix.
        y : pSe
            Target series.

        Returns
        -------
        GiniPath
            Train/valid Gini at each feature addition step.
        """        
        return PrunerGini(self.fitter, self.val_size).process(X, y) 
        
    def plot(self, show:bool=None) -> go.Figure | None:
        """
        Generate the pruning visualization.

        Parameters
        ----------
        show : bool | None
            Whether to display the figure. Defaults to the
            instance's `verbose` setting.

        Returns
        -------
        go.Figure | None
            The combined importance + Gini path figure.
        """        
        show = show or self.fitter.verbose
        plot = PrunerPlot(self)
        return plot.create(show)  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class _PlotView:
    """Internal plot layout utility for the pruner visualization."""

    def __init__(self, fig:go.Figure, fes:list, cutoff_ix:int):
        self.fig = fig
        self.fes = fes
        self.cutoff_ix = cutoff_ix
    
    def vrect(self, add_annot:bool=True) -> None:
        """Highlight the pruned region with a red rectangle."""
        
        if not self.cutoff_ix < len(self.fes):
            return
            
        self.fig.add_vrect(
            x0 = self.cutoff_ix - 0.5,
            x1 = len(self.fes) - 0.5,
            fillcolor   = "red", 
            opacity     = 0.07,
            layer       = "below",
            line_width  = 0
        )

        if not add_annot:
            return
        
        self.fig.add_annotation(
            x = (self.cutoff_ix + len(self.fes) - 1) / 2,   # center it in the red zone
            y = 0.55,
            yref    = "paper",
            text    = "PRUNED FEATURES",
            font    = dict(size=12, color="rgba(255, 0, 0, 0.5)", family="Arial Black"),
            bgcolor = "rgba(255, 255, 255, 0.8)",
            showarrow = False
        )
    
    def update_layout(self, y_title:str, y2_title:str=None, show_legend:bool=False) -> None:
        """
        Apply the standard layout (axes, grid, legend, sizing).
        """        
        font  = font=dict(family="Arial", size=13, color="black")
        title = lambda t: dict(text=t, font=font)
        grid  = lambda w=0.5: dict(
                gridwidth = w, 
                gridcolor = f'rgba(255, 255, 255, {w})',
        )
        legend = dict(
            y=0.02, x=0.5, yanchor='bottom', xanchor='center', orientation='h'
        )
        xaxis  = dict(
                title = title('Feature'),
                tickfont = font,
                tickvals = self.fes,
                ticktext = list(range(len(self.fes))),
                tickmode = 'array',    
        )
        yaxis_ = lambda t: dict(
                title = title(t),
                tickfont = font,
                tickformat = ".2f",
                hoverformat = ".3f",                
        )        
        yaxis2 = yaxis_(y2_title) | dict(
                overlaying="y",
                side="right"
        ) | grid(0) if y2_title else None
        
        self.fig.update_layout(
            height = 600, 
            width  = 1050,
            margin = dict(t=20, b=20, l=0, r=0, pad=0),            
            legend = legend if show_legend else None,
            showlegend = show_legend,
            xaxis  = xaxis | grid(),
            yaxis  = yaxis_(y_title) | grid(),
            yaxis2 = yaxis2
        )          
    
    def show(self, show:bool) -> go.Figure | None:
        if not show:
            return self.fig
        self.fig.show()  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PrunerPlot:
    """
    Pruner visualization: importance distributions + Gini path.

    Combines two views into a single figure:
    - **Box plot**: Per-feature importance distribution across
      shadow iterations, with the shadow threshold line.
    - **Gini path**: Train/valid Gini as features are added
      incrementally.

    The pruned region (features below the threshold) is
    highlighted with a red overlay.
    """

    _y_title = {
        MTD.S: 'Mean Absolute Impact (SHAP)',
        MTD.P: 'Mean Performance Penalty (Permutation)'
    }

    def __init__(self, pruner:FeaturesShadowPruner):
        self.pruner = pruner
        self.fitter = pruner.fitter
        self.fitter._verify_fitted()

        self.th          = self._get_shadow_th()
        self.fes         = self._get_order_fes()
        self._cutoff_ix  = self._get_cutoff_ix()

        self._im_y_title = self._y_title[self.fitter.solver]
        self._gi_y_title = 'Model Gini (train | valid)'
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_shadow_th(self) -> float:
        return self.fitter.optim_path[RES.F].shadow_th
    
    def _get_order_fes(self) -> list:
        return self.fitter.get_ordered_importance_fes()

    def _get_cutoff_ix(self) -> int:
        im = self.fitter.get_importance()
        m  = im[RES.M] > self._get_shadow_th()
        return int( sum(m) )
    
    def _get_im_data(self) -> pDF:
        im = self.fitter.get_importance()
        return im.drop([RES.M, RES.C], axis=1).T
    
    def _get_gini_train_valid(self) -> pDF:
        return self.pruner.gini_path.train_valid.loc[self.fes]
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _view(self, fig:go.Figure, y_title:str, y2_title:str=None, show_legend:bool=False
    ) -> _PlotView:
            
        view = _PlotView(fig, self.fes, self._cutoff_ix)
        view.vrect()        
        view.update_layout(y_title,  y2_title=y2_title, show_legend=show_legend)
        
        return view        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _add_box(self, fig:go.Figure) -> None:
        im = self._get_im_data()
        
        color = (lambda i,o=1.0: 
            f'rgba(52, 191, 77, {o})' if i < self._cutoff_ix else 
            f'rgba(191, 52, 52, {o})'
        )
        for i,c in enumerate(self.fes):
            fig.add_trace(
                go.Box(
                    y = im[c], 
                    name = c,
                    line = dict(color = color(i), width=0.5),
                    fillcolor = color(i,0.4),
                    hovertemplate = c,
                    boxmean = True, 
                    quartilemethod = "linear", 
            ))

        sn = 'Shadow Noise TH: ' + f'{self.th:.3f}'
        fig.add_trace(
            go.Scatter(
                x = self.fes, 
                y = [self.th]*len(self.fes),
                name = sn,
                hovertemplate = sn,
                line = dict(color = 'rgb(0,0,0)', width=0.75, dash='dot'),
                mode = 'lines'    
        ))
            
    def box(self, show:bool=True) -> go.Figure | None:
        """Generate the importance box plot only."""        
        fig = go.Figure()
        self._add_box(fig)
        view = self._view(fig, self._im_y_title)
        return view.show(show)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _add_gini(self, fig:go.Figure, secondary_y:bool=False) -> None:
        tv = self._get_gini_train_valid()

        w = 1.5
        o = 0.7 if secondary_y else 1

        color = (lambda i, o: 
            f'rgba(191, 52, 126, {o})' if i else
            f'rgba( 52, 66, 191, {o})'
        )
        for i,x in enumerate(GINI.list()):
            fig.add_trace(
                go.Scatter(
                    y = tv[x], 
                    x = self.fes,                    
                    name = x,
                    line = dict(color = color(i,o), width=w),
                    # line_dash = 'dot' if secondary_y else None,
                    hovertemplate = [
                        f'({f}) Gini:{g:.3f} | +{v}' for f,(v,g) in enumerate(tv[x].items())
                    ]
            ), 
            secondary_y = secondary_y)

    def gini(self, show:bool=True) -> go.Figure | None:
        """Generate the Gini path plot only."""        
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        self._add_gini(fig)            
        view = self._view(fig, self._gi_y_title, show_legend=True)
        return view.show(show)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def create(self, show:bool=True) -> go.Figure | None:
        """
        Generate the combined figure (importance + Gini path).

        Parameters
        ----------
        show : bool, default=True
            Whether to display the figure.

        Returns
        -------
        go.Figure | None
            The combined figure.
        """        
        fig = make_subplots(specs=[[{"secondary_y": True}]])       
        self._add_box(fig)
        self._add_gini(fig, secondary_y=True)
        view = self._view(fig, self._im_y_title, self._gi_y_title)
        return view.show(show)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬