"""
Champion Agent Fit — Production Model Finalization.

This module implements the final fitting stage of the AIDflow pipeline.
It sits below the Agent Selector (which chooses *which* agent to deploy)
and above the raw predictor (which serves inference). Its job is to
**re-fit** the chosen champion agent on a user-curated feature subset,
producing a production-ready model with validation diagnostics.

Why a Separate Fit Stage?
-------------------------
The training phase (GLM/GBM selectors) already performed feature
selection. However, the Agent Selector may allow the user to **further
curate** the feature set (e.g., drop features with high SHAP drift or
low incremental Gini). This module handles the re-fitting on that
final subset:

- **GLM agents**: The model is re-fitted on the full training data.
  No train/validation split is needed because feature selection was
  already completed via regularization + shadow pruning. The
  validation split is used only for diagnostic metrics.

- **GBM agents**: The model is re-fitted with a stratified
  train/validation split. The validation set serves as the
  `eval_set` for early stopping, preventing overfitting on the
  reduced feature set. This mirrors the protocol used during
  original training, ensuring fair diagnostic comparison.

Champion vs. Training Agent (recap)
-----------------------------------
- **Training Agent**: The model as produced by the training stage
  (all selected features, original hyperparameters).
- **Champion Agent**: The model after this stage — same
  hyperparameters, but potentially a **subset** of features. This
  is the model that goes to production.

Parameter Inheritance
---------------------
The champion inherits all hyperparameters (regularization strength,
tree depth, learning rate, etc.) directly from the training agent.
No new hyperparameter search is performed. The only degree of
freedom at this stage is the **feature subset** chosen by the user.
This ensures the champion is a strict subset of the training agent's
knowledge — stable, reproducible, and fully auditable.

Supported Model Backends
------------------------
- **Logistic Regression** (GLM): Full-data fit, unpenalized or
  regularized depending on the original agent's solver.
- **LightGBM** (GBM): Train/valid split with early stopping.
  Categorical features are preserved from the original model.
- **XGBoost** (GBM): Same protocol as LightGBM.
"""

from ....deps.data                              import  pd, pDF, pSe
from ....deps.types                             import  NDArray

from ....core.base                              import  (
                                                        FitterTrainBase, 
                                                        FitterModelBase
                                                        )
from ....core.factory                           import  Dataclass
from ....core.foundation                        import  AigParam

from ....ml.models                              import  (
                                                        Logit,
                                                        LGB, 
                                                        XGB
                                                        )
from ....ml.sampling                            import  Sampling
from ....ml.metrics                             import  (
                                                        DiagnosticsResult,
                                                        SeparationDiagnostics
                                                        )
from ....learner.agents.fitters.glm_selector    import  FitParams as FitParams_GLM
from ....learner.agents.fitters.gbm_selector    import  FitParams as FitParams_GBM

from ...predict.adapter                         import  PredictorAdapter
                                        

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → PRODUCTION MODEL
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class FitParamsBase:
    """
    Champion fit — analysis and output settings.

    Attributes
    ----------
    agent_choice : int | str, default=1
        Identifier of the champion agent to fit.
        - **int**: Agent key (1–4, corresponding to the
          AGENTS registry order).
        - **str**: Agent name from the AGENTS registry
          (e.g., `'GLM_S'`, `'GBM_L'`).

        This agent's hyperparameters and feature list are
        inherited. The user's feature subset (if any) is
        applied via the `features` argument to `fit`.
    """        
    agent_choice: int|str = 1

@Dataclass.base(update_specs=dict(slots=False))
class FitParamsCfg:
    """
    Champion fit — pipeline component references.

    Attributes
    ----------
    predictors : PredictorAdapter
        Adapter holding all trained agents. Used to extract
        the champion's hyperparameters, feature list, and
        model type.
    target : str
        Target column name (binary: 0/1).
    val_size : float | None, default=None
        Validation fraction for GBM early stopping. If None,
        inherits the value from the original training agent's
        parameters. GLM agents ignore this (they use full data
        for fitting; validation is diagnostics-only).
    """
    predictors: PredictorAdapter
    target:     str
    val_size:   float = None

@Dataclass.base
class FitParams(FitParamsBase, FitParamsCfg):
    """Fitter params input design."""

@Dataclass.base
class FitResult:
    """
    Champion fit result — feature selection record.

    Attributes
    ----------
    features_selected : list
        Features used in the champion model (the final
        production feature set).
    features_dropped : list
        Features that were in the training agent but removed
        by the user's curation decision.
    """
    features_selected: list
    features_dropped:  list

@Dataclass.base
class FitStats:
    """
    Champion fit diagnostic output.

    Attributes
    ----------
    diagnostics : DiagnosticsResult, default=None
        Validation metrics (Gini, AUC, KS, lift table) on the
        held-out validation set. For GBM agents, this is the
        early-stopping validation split. For GLM agents, this
        is a stratified sample of the full training data.
    """
    diagnostics: DiagnosticsResult = None

@Dataclass.base
class FitDashLink:
    """
    Reference bundle for the display layer.

    Provides the display component with the validation set
    row indices, enabling direct access to per-row predictions
    for lift charts and calibration plots.

    Attributes
    ----------
    train_val_idx : pd.Index
        Validation set row indices (for metric display and
        per-row diagnostics).
    """
    train_val_idx: pd.Index
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Fitter(FitterTrainBase):
    """
    Champion Agent Fitter — Production Model Training Backend.

    Re-fits the chosen champion agent on a user-curated feature
    subset, inheriting all hyperparameters from the original
    training stage.

    Fitting Protocol
    ----------------
    **GLM agents** (Logistic Regression, Lasso, ElasticNet, PLS):
    The model is fitted on the **full training data** with the
    user-selected features. No train/validation split is used for
    fitting because feature selection was already completed during
    training (regularization + shadow pruning). A stratified
    validation sample is drawn solely for diagnostic metrics
    (Gini, AUC, KS) — ensuring fair comparison with GBM agents
    that use the same split protocol.

    **GBM agents** (LightGBM, XGBoost):
    A stratified train/validation split is created. The
    validation set serves as the `eval_set` for early stopping,
    preventing overfitting on the reduced feature set.
    Categorical features are preserved from the original model
    configuration.

    Parameter Inheritance
    ---------------------
    All model hyperparameters (regularization strength, tree
    depth, learning rate, number of estimators, etc.) are
    extracted from the original training agent and passed
    unchanged to the new model instance. The only difference
    between the training agent and the champion is the
    **feature subset** — making the champion a strict, auditable
    projection of the training agent's knowledge.

    Parameters
    ----------
    params : FitParams
        Champion agent identifier and pipeline references.
    """       
    _model: Logit | LGB | XGB = None

    def __init__(self, params: FitParams):
        self.prm = params        
        a = params.agent_choice

        self.is_gbm  = params.predictors.is_gbm(a)
        agent_model  = params.predictors.get_agent(a).model
        model_fitter = agent_model.fitter.model

        self.orig_params: FitParams_GLM | FitParams_GBM = model_fitter.params

        self.orig_features    = model_fitter.features
        self.orig_ordered_fes = model_fitter.get_ordered_fes()
        
        self._set_config(model_fitter.model)
        self._est = type(model_fitter.model)
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def _set_config(self, orig_model: Logit | LGB | XGB) -> dict:
        """
        Extract the model configuration from the original training agent.

        For GLM: captures the solver and regularization parameters.
        For GBM: captures tree hyperparameters and the list of
        categorical features (needed to re-specify them on the
        reduced feature set).

        Parameters
        ----------
        orig_model : Logit | LGB | XGB
            The original fitted model instance from which
            configuration is extracted.
        """        
        self._cat = {}
        if self.is_gbm:
            self._cat = orig_model.get_categorical_feature()
        self.config = orig_model.get_setup()

    def _get_config(self, features: list) -> dict:
        """
        Assemble the model constructor arguments for the final fit.

        For GBM models, filters the categorical feature list to
        only include features present in the champion's feature
        subset (avoiding references to dropped features).

        Parameters
        ----------
        features : list
            The champion's selected feature names.

        Returns
        -------
        dict
            Model constructor keyword arguments.
        """        
        if not self.is_gbm:
            return self.config
        
        cat = [x for x in features if x in self._cat]
        return self.config | dict(categorical_feature = cat)
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def _train_valid_split(self, X:pDF, y:pSe) -> tuple[ tuple[pDF,pSe], tuple[pDF,pSe] ]:
        """
        Stratified train/validation split for GBM early stopping.

        The validation set is used exclusively as the `eval_set`
        for the tree boosting process (early stopping criterion).
        It is **not** used for feature selection (that was done
        during training). This ensures the early stopping decision
        is based on truly held-out data.

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
        vs = self.prm.val_size or self.orig_params.val_size
        rs = self.orig_params.random_state
        return Sampling(vs, stratify=y, random_state=rs, shuffle=True).train_valid_split(X,y)
            
    def _fit(self, X:pDF, y:pSe) -> pd.Index:
        """
        Fit the champion model on the selected features.

        **GLM**: Full-data fit (no split needed for fitting).
        Returns the full data index (all rows are "validation"
        for diagnostic purposes).

        **GBM**: Train/valid split with early stopping.
        Returns the validation set index (used for diagnostics).

        Parameters
        ----------
        X : pDF
            Feature matrix (champion feature subset).
        y : pSe
            Target series.

        Returns
        -------
        pd.Index
            Validation set row indices (for diagnostic metrics).
        """
        cfg = self._get_config(X.columns)
        model: Logit | LGB | XGB = self._est(**cfg)        
        
        if self.is_gbm:
            Xy_train, Xy_valid = self._train_valid_split(X,y)
            self._model = model.fit(*Xy_train, eval_set = [Xy_valid])
            return Xy_valid[0].index
        
        self._model = model.fit(X, y) 
        return X.index        

    def _diagnostics(self, d_val:pDF) -> DiagnosticsResult:
        """
        Compute validation metrics on the held-out set.

        Produces Gini, AUC, KS, and a 10-bin lift table.
        These metrics are directly comparable with the training
        agent's validation metrics (same split protocol),
        enabling a fair assessment of the feature subset's
        impact on predictive power.

        Parameters
        ----------
        d_val : pDF
            Validation data (rows identified by the fit stage).

        Returns
        -------
        DiagnosticsResult
            Separation metrics and lift table on the
            validation set.
        """            
        t = d_val[self.prm.target]
        p = self.predict_proba(d_val)[:, 1]
        return SeparationDiagnostics(lift_bins=10).evaluate(t, p)
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def fit(self, data:pDF, features:list):
        """
        Execute the champion agent fit.

        Steps:
        1. Fit the model (GLM: full data; GBM: train/valid split
           with early stopping).
        2. Record the feature selection result (selected vs.
           dropped).
        3. Compute validation diagnostics.
        4. Set up the display link (validation indices).

        Parameters
        ----------
        data : pDF
            Training data containing target and features.
        features : list
            Champion feature subset (user-curated). Must be a
            subset of the training agent's selected features.

        Returns
        -------
        self
            Fitted instance with `features`, `stats`,
            `ordered_features`, and `dash` set.
        """

        idx = self._fit(data[features], data[self.prm.target])

        self.result = FitResult(
            features_selected = features,
            features_dropped  = [x for x in self.orig_features if x not in features]
        )

        self.features = features
        self.ordered_features = [x for x in self.orig_ordered_fes if x in features]

        self.stats = FitStats(
            diagnostics = self._diagnostics(data.loc[idx])
        )

        self.dash = FitDashLink(train_val_idx = idx)

        return self

    def predict_proba(self, data:pDF) -> NDArray:
        """
        Predict event probabilities using the champion model.

        Parameters
        ----------
        data : pDF
            Input data containing the champion's selected features.

        Returns
        -------
        NDArray
            2-D array of shape (n_samples, 2) with P(class=0)
            and P(class=1).
        """
        return self.model.predict_proba(data[self.features])    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PredictorFitter(FitterModelBase[Fitter]):
    """
    Champion Agent Fit — Public API for Production Model Finalization.

    The final fitting stage of the AIDflow pipeline. Takes a
    user-curated feature subset from the chosen agent and re-fits
    the model, producing a production-ready predictor with
    validation diagnostics.

    Fitting Protocol
    ----------------
    - **GLM agents**: Full-data fit (selection already completed
      during training). Validation split used for diagnostics only.
    - **GBM agents**: Stratified train/validation split with early
      stopping on the reduced feature set.

    The champion inherits all hyperparameters from the training
    agent. The only difference is the feature subset — making the
    champion a strict, auditable projection of the training
    agent's knowledge.

    Parameters
    ----------
    agent_choice : int | str, default=1
        Champion agent identifier (key or name from AGENTS registry).
    predictors : PredictorAdapter
        Adapter holding all trained agents (source of hyperparameters
        and feature list).
    target : str
        Target column name (binary: 0/1).
    val_size : float | None, default=None
        Validation fraction for GBM early stopping. If None,
        inherits from the original training agent.

    Notes
    -----
    **Parameter Inheritance**: No hyperparameter search is
    performed. All model parameters (regularization, tree depth,
    learning rate, etc.) are inherited from the training agent.
    This ensures reproducibility and auditability.

    **Feature Subset**: The `features` argument to `fit`
    should be a subset of the training agent's selected features.
    Features outside this set are ignored. The ordering is
    preserved from the training agent's importance ranking.

    **GBM Early Stopping**: For GBM agents, the validation split
    is used as the `eval_set` for early stopping. This prevents
    overfitting on the reduced feature set and mirrors the
    original training protocol.

    Examples
    --------
    >>> from aidflow.fitters import PredictorFitter
    >>> fitter = PredictorFitter(
    ...     predictors=predictors,
    ...     target='is_churn',
    ...     agent_choice=1
    ... )
    >>> fitter.fit(train_data, features=['tenure_WOE', 'revenue_WOE', 'segment_WOE'])
    >>> probs = fitter.predict_proba(new_data)
    >>> fitter.get_ordered_fes()  # ['tenure_WOE', 'revenue_WOE', 'segment_WOE']
    """

    def __init__(self, *args, params: dict|FitParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(FitParams, params, args, kwargs)

    def fit(self, data:pDF, features:list):        
        """
        Fit the champion model on the selected feature subset.

        Parameters
        ----------
        data : pDF
            Training data containing target and features.
        features : list
            Champion feature subset (user-curated). Must be a
            subset of the training agent's selected features.

        Returns
        -------
        self
            Fitted instance with `predict_proba` and
            `get_ordered_fes` available.
        """        
        self._fitter = Fitter(self._params)
        self._fitter.fit(data, features)
        return self
    
    def predict_proba(self, data:pDF) -> NDArray:
        """
        Predict event probabilities using the fitted champion model.

        Parameters
        ----------
        data : pDF
            Input data containing the champion's selected features.

        Returns
        -------
        NDArray
            2-D array of shape (n_samples, 2): [P(class=0), P(class=1)].
        """
        return self.fitter.predict_proba(data)

    def get_ordered_fes(self) -> list:
        """
        Return the champion features in descending importance order.

        The importance ordering is inherited from the training
        agent (determined by regularization path or tree
        importance during the original training stage).

        Returns
        -------
        list
            Feature names, most important first (restricted to
            the champion's feature subset).
        """
        return self.fitter.ordered_features    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬