"""
Unified Hyperparameter Optimization engine powered by Optuna.

This module implements a high-level orchestration layer for tuning Gradient 
Boosting Machines (XGBoost and LightGBM). By abstracting backend-specific 
implementation details, it provide a single, consistent interface for defining 
search spaces and executing optimization studies. It integrates advanced 
Bayesian sampling (TPE) with Hyperband pruning to rapidly converge on optimal 
hyperparameters while minimizing computational waste.
"""

from ...deps.data           import  np, pDF, pSe
from ...deps.plots          import  go
from ...deps.ml.gbm         import  optuna
from ...deps.types          import  (
                                    Literal,
                                    Callable, 
                                    NDArray,
                                    ArrayLike
                                    )
from ...ml.models           import  XGB, LGB
from ...ml.sampling         import  Sampling

from ...utils.builtins      import  List


type GB = XGB | LGB


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TREE BASED HYPERPARAMS TUNING
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptunaConnection:

    CONNECTORS = {
        'mem':  optuna.storages.InMemoryStorage,
        'sql':  optuna.storages.RDBStorage,
        'red':  optuna.storages.JournalRedisStorage,
        'grpc': optuna.storages.GrpcStorageProxy,
    }

    def __init__(self, storage_config: str | dict[str, str] | None):
        self.storage_config = storage_config
    
    def establish(self) -> optuna.storages.BaseStorage:
        config = self.storage_config

        if not config:
            return self.CONNECTORS['mem']()
        
        if isinstance(config, str):
            if any([config.startswith(x) for x in ('sqlite', 'postgresql', 'mysql')]):
                return self.CONNECTORS['sql'](config)
            elif config.startswith('redis'):
                return self.CONNECTORS['red'](config)
            else:
                raise ValueError(f"Unknown storage URL scheme: {config[:10]}...")
                        
        if isinstance(config, dict):
            conn_type = config.get('type').lower()

            if conn_type not in self.CONNECTORS:
                raise ValueError(
                    f"Unknown storage type: {conn_type}. Supported: {list(self.CONNECTORS)}"
                )
            
            config_ = {k:v for k,v in config.items() if k != 'type'}            
            return self.CONNECTORS[conn_type](**config_)

        raise TypeError(f"Unsupported storage_config type: {type(config)}")        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
                  
class TunerBase:

    implemented_estimators = {
        'xgb': XGB,
        'lgb': LGB
    }
    
    study: optuna.Study|None = None
    space_builder: dict|Callable = None
    best_params: dict|None = None
    study_name: str|None = None
    storage_config: str|dict|None = None
    storage: optuna.storages.GrpcStorageProxy|None = None
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        self._remove_study_connection(state)
        return state
    
    def __setstate__(self, state:dict) -> None:
        self.__dict__.update(state)
        self._load_set_study_connection()

    def _remove_study_connection(self, state:dict) -> None:
        if self.storage_config is not None:
            state['storage'] = None
            state['study']   = None

    def _load_set_study_connection(self) -> None:
        if self.storage_config is not None:
            self._set_storage_connection()
            self._create_set_study(load=True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_solver(self, solver:str) -> GB:
        return self.implemented_estimators[solver]

    def _set_setup(self, solver:str, setup:dict) -> None:
        self.setup = {'estimator': self._get_solver(solver), 'setup': setup}

    def get_estimator(self) -> GB:
        return self.setup['estimator']
    
    def get_setup(self) -> dict:
        return self.setup['setup']

    def get_estimator_instance(self, **kwargs) -> GB:
        e = self.get_estimator()
        s = self.get_setup() | kwargs
        return e(**s)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _create_set_study(self, load:bool=False) -> None:
        raise NotImplemented
    
    def _set_study_name(self, solver:str, study_name:str|None) -> None:
        self.study_name = study_name or f'{solver}_optuna_hyperparams'
    
    def _set_storage_config(self, storage_config:dict|str|None) -> None:
        self.storage_config = storage_config
    
    def _set_storage_connection(self):
        if self.storage is None:
            config = self.get_storage_config()
            self.storage = OptunaConnection(config).establish()
    
    def _set_study_storage_params(self, 
        solver: str, study_name: str|None, storage_config: dict|str|None
    ) -> None:
        self._set_study_name(solver, study_name)
        self._set_storage_config(storage_config)
        self._set_storage_connection()
            
    def get_storage_config(self) -> str|dict|None:
        return self.storage_config

    def get_study_storage(self) -> optuna.storages.InMemoryStorage:
        return self.storage

    def get_study_name(self):
        return self.study_name
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def get_result(self) -> dict|None:
        return self.best_params

    def get_study(self) -> optuna.Study|None:
        return self.study
    
    def get_study_dataframe(self) -> pDF|None:
        if not self.is_fitted():
            return
        return self.get_study().trials_dataframe()

    def is_fitted(self) -> bool:
        return self.study is not None
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class HyperTunerGB(TunerBase):
    """
    Unified Hyperparameter Tuner for XGBoost and LightGBM using Optuna.

    This class orchestrates the hyperparameter optimization loop by abstracting
    backend-specific differences between XGBoost and LightGBM. It handles metric
    mapping, pruning callback injection, cross-validation strategies, and result
    extraction.

    Parameters
    ----------
    solver : Literal['xgb', 'lgb']
        Specifies the backend library to use ('xgb' for XGBoost, 'lgb' for LightGBM).
    
    n_trials : int
        Maximum number of hyperparameter evaluations to perform. Default is 100.

    random_state : int
        Seed for reproducibility of data splits and random search.
    
    categorical_feature : list, optional
        List of column names or indices to treat as categorical features.
        Passed directly to the underlying estimator.
    
    metric : dict, optional
        Configuration for the optimization metric.
        - 'name' (str): The metric name (e.g., 'logloss', 'rmse').
        - 'direction' (str): 'minimize' or 'maximize'.
    
    cv_setup : dict
        Settings for data splitting. Keys include:
        - 'valid_part': Float ratio for validation set (default 0.33).
        - 'n_splits': Integer number of CV folds (default 1).
        - 'shuffle': Boolean to shuffle data before splitting.
    
    gb_setup : dict, optional
        Base parameters passed directly to the Gradient Boosting estimator
        (e.g., 'early_stopping_rounds', 'n_jobs', 'verbose').
    
    study_name : str, optional
        Unique name for the Optuna study. Required if using persistent storage.
    
    storage_config : dict or str, optional
        Configuration for Optuna storage (e.g., database URL string or config dict)
        to enable persistent study results.

    Examples
    --------
    >>> tuner = HyperTunerGB(
    ...     solver   = 'xgb',
    ...     n_trials = 50,
    ...     metric   = {'name': 'logloss', 'direction': 'minimize'},
    ...     cv_setup = {'n_splits': 3, 'valid_part': 0.2}
    ... )
    """
    BASELINE = dict(
        max_leaves          = 31,
        max_depth           = -1,
        learning_rate       = 0.1,
        n_estimators        = 100,
        min_child_weight    = 0.1,
        subsample           = 0.7,
        colsample_bytree    = 0.7, 
        reg_alpha           = 0.0,
        reg_lambda          = 0.0,
    )

    def __init__(self, 
        solver: Literal['xgb', 'lgb'], 
        n_trials: int = 100,
        random_state: int = 42,
        categorical_feature: list = None, 

        metric: dict = dict(
            name = 'logloss', 
            direction = 'minimize'
        ),
        cv_setup: dict = dict(
            valid_part = 0.33, 
            n_splits   = 1, 
            shuffle    = True
        ),
        gb_setup: dict = dict(
            early_stopping_round = 50, 
            n_jobs  = -1, 
            verbose = -1
        ),
        study_name: str = None,
        storage_config: dict|str = None
    ):
        setup = (gb_setup or {}) | dict(
            metric = metric['name'],
            categorical_feature = categorical_feature, 
            random_state = random_state
        )
        self._set_setup(solver, setup)
        self._set_study_storage_params(solver, study_name, storage_config)

        self.solver     = solver
        self.n_trials   = n_trials
        self.metric     = metric
        
        self.cv_setup   = cv_setup or {}
        self.random     = random_state
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_metric_direction(self) -> str:
        return self.metric['direction']
    
    def _get_metric_name(self) -> str:
        m = self.metric['name']
        e = self.get_estimator()
        return e.METRIC_MAP.get(m) or m

    def _get_callback(self, trial: optuna.Trial) -> list:
        """Injects the correct Optuna Pruning and Early Stopping per backend."""
        m = self._get_metric_name()
        o = optuna.integration
        if self.solver == 'lgb':
            return [ o.LightGBMPruningCallback(trial, m) ]
        return [ o.XGBoostPruningCallback(trial, f"validation_0-{m}") ]

    def _fit(self, model:GB, trial:optuna.Trial, Xy_tv:tuple[tuple, tuple], step:int) -> None:
        """Model fit with defined callback."""
        callbacks = None
        if step == 0:
            callbacks = self._get_callback(trial)

        Xy_train, Xy_valid = Xy_tv
        model.fit(
            *Xy_train,
            eval_set  = [Xy_valid],
            callbacks = callbacks
        )

    def _extract_score(self, model:GB) -> float:
        """Standardizes metric extraction across messy backend result objects."""
        m = self._get_metric_name()
        if self.solver == 'lgb':
            return model.best_score_["valid_0"][m]
        results = model.evals_result()
        return results["validation_0"][m][-1]
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_data_splits(self, X:pDF|NDArray, y:pSe|ArrayLike) -> tuple[tuple[pDF,pSe],...]:
        """Generates data splits based on the `cv_setup` configuration."""
        n_splits = self.cv_setup.get('n_splits') or 1
        
        smp = Sampling(
            sample_size  = (self.cv_setup.get('valid_part') or 0.3) if n_splits < 2 else 1, 
            stratify     = y,
            shuffle      = self.cv_setup.get('shuffle'),
            random_state = self.random
        )

        if n_splits < 2:
            return (smp.train_valid_split(X, y),)
        
        return smp.cv_splits(X, y, n_splits=n_splits)

    def _create_objective(self, X:pDF|NDArray, y:pSe|ArrayLike, space_builder:Callable) -> Callable:
        """Creates the Optuna objective function injected with the user's space."""

        def objective(trial: optuna.Trial):
            params = space_builder(trial)
            
            scores = []
            for i, Xy in enumerate( self._get_data_splits(X, y) ):
                model = self.get_estimator_instance(**params)
                self._fit(model, trial, Xy, i)
                scores.append(self._extract_score(model))

            return np.mean(scores)
        return objective
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_trial_suggest(trial:optuna.Trial, k:str, v:tuple|str|float|int) -> float|int:
        suggest = {
            float:  trial.suggest_float,
            int:    trial.suggest_int
        }
        v = List.ensure(v)
        t = type(v[0])
        return suggest[t](k, *v)
    
    @classmethod
    def _create_space_builder(cls, space_builder:dict|Callable) -> Callable:
        """Generates Space Builder from dictionary."""
        if callable(space_builder):
            return space_builder
        
        def space(trial:optuna.Trial) -> dict:
            return {
                k: cls._get_trial_suggest(trial, k, v) for k,v in space_builder.items()
            }
        return space
    
    def _set_space_builder_confg(self, space_builder:dict|Callable) -> None:
        self.space_builder = space_builder

    def _create_set_study(self, load:bool=False) -> None:
        """Creates Optuna Tree Based Study."""       
        sampler = optuna.samplers.TPESampler(seed=self.random)
        pruner  = optuna.pruners.HyperbandPruner()

        study = optuna.create_study(
            storage     = self.get_study_storage(),
            sampler     = sampler,
            pruner      = pruner,
            study_name  = self.get_study_name(),
            direction   = self._get_metric_direction(),
            load_if_exists = load
        )
        if isinstance(self.space_builder, dict):
            study.enqueue_trial(
                {k:v for k,v in self.BASELINE.items() if k in self.space_builder}
            )
        self.study:optuna.Study = study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def optimize(self, X:pDF|NDArray, y:pSe|ArrayLike, space_builder:dict|Callable):
        """
        Executes the full hyperparameter optimization study.
        
        Sets up the Optuna study with a TPE sampler and Hyperband pruner, 
        runs the trials, and stores the best parameters found.
        
        Parameters
        ----------
        X : DataFrame or NDArray
            Feature matrix.
        y : Series or ArrayLike
            Target vector.
        space_builder : dict or Callable
            Definition of the hyperparameter search space.
            
        Returns
        ------
        dict: The best hyperparameters found (`best_params_`).
        """
        self._set_space_builder_confg(space_builder)
        self._create_set_study()
        study = self.get_study()

        param_space = self._create_space_builder(space_builder)
        objective   = self._create_objective(X, y, param_space)        
        
        study.optimize(objective, n_trials=self.n_trials, n_jobs=1)

        self.best_params = study.best_params
        self.best_score  = study.best_value
        
        return self.best_params

    def plot(self, 
        graphs: list|Literal['optim_path','coordinate', 'importance']=None,
        show:   bool=True
    ) -> None | list[go.Figure]:
        """
        Generates and displays standard Optuna visualization plots.
        
        1. Optimization History (Objective value vs Trial).
        2. Parallel Coordinates (Hyperparameter relationships).
        3. Parameter Importances (Impact of params on objective).
        """
        if not self.is_fitted():
            return
        
        vis = optuna.visualization
        lib = {
            'optim_path': vis.plot_optimization_history,
            'coordinate': vis.plot_parallel_coordinate,
            'importance': vis.plot_param_importances
        }
        study = self.get_study()
        p:list[go.Figure] = [lib[g](study) for g in List.ensure(graphs) or lib]

        if not show:
            return p
        
        [g.show() for g in p]            
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬