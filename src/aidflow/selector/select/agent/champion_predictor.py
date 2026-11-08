"""
Champion Agent — Production Deployment Artifact.

This module implements the final, deployable unit of the AIDflow
pipeline. It takes the user's champion agent choice (from the
Selection stage) and the user-curated feature subset, re-fits the
model, and packages everything — encoders, model weights, feature
mappings — into a single self-contained inference object.

What This Module Does
---------------------
1. **Freezes** the champion: locks the chosen agent's
   hyperparameters, feature subset, and encoder states.
2. **Re-fits** the model on the final feature subset (GLM: full
   data; GBM: train/valid split with early stopping).
3. **Binds** the preprocessing chain (feature cleaner + WOE
   encoder) so that `preprocess` + `transform` together
   replicate the exact training-time data pipeline.
4. **Exports** a deployment-ready object with full feature
   lineage (selected, dropped, raw) for audit and governance.

The Result: A Self-Contained Predictor
--------------------------------------
Once fitted, `AgentPredictor` requires no external references:

- **`preprocess(data)`**: Raw → ML-ready (cleaning + encoding).
- **`transform(data)`**: ML-ready → probabilities.
- **`transform_dual(inputs)`**: Paired train/test inference
  (for pre-flight validation before go-live).

The object holds all WOE bin maps, model coefficients/tree
structures, and feature names internally. It can be serialized
(`_save`) and deployed independently of the pipeline.

Feature Lineage & Auditability
------------------------------
Every prediction carries its full lineage:

- **`features`**: Output column name (e.g., `['score']`).
- **`selected`**: WOE features actually used by the model.
- **`dropped`**: Features that were in the training agent but
  removed by the user's curation decision.
- **`raw_features`**: Original (pre-WOE) feature names — the
  business-facing identifiers.

This ensures that in production, any stakeholder can trace a
prediction back to its exact input features and model version.

GLM vs. GBM Handling
--------------------
- **GLM**: Full-data fit (selection was already completed during
  training). Validation split used for diagnostics only.
- **GBM**: Stratified train/valid split with early stopping on
  the reduced feature set. Categorical features preserved.

Dependencies
------------
Requires the following pipeline stages to be completed:
- Agent Training (4 agents: GLM variants + GBM variants)
- Feature Cleaner (raw data standardization)
- WOE Encoder (feature encoding for GLM agents)
"""

from ....deps.common                import  wraps
from ....deps.data                  import  pDF
from ....deps.types                 import  NDArray

from ....core.parameters            import  AIGFLOW
from ....core.registry              import  (
                                            FitRegister, 
                                            FitRegOpt,   
                                            DashRegister,  
                                            RegGet
                                            )
from ....core.base                  import  (
                                            FitterBase,
                                            TransformerBase,
                                            ModelProcessorBase,
                                            ModelInput,
                                            ModelResult,
                                            ModelBase,
                                            ModelBaseDual,
                                            ModelUIBase
                                            )
from ....core.display               import  DashBackground
from ....core.factory               import  (
                                            Dataclass,
                                            DictClass
                                            )
from ....core.foundation            import  AigParam
from ....core.io                    import  ModelIO

from ....learner.agents.utils_fes   import  RawFeatures

from ...predict.adapter             import  PredictorAdapter

from .fitter                        import  (
                                            PredictorFitter,
                                            FitParamsBase,
                                            FitParamsCfg,
                                            FitResult,
                                            FitStats
                                            )

MODEL = AIGFLOW.Selection.champion_agent
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG SELECTION STAGE ► PRODUCTION AGENT PREDICTOR ► FINAL CHOSEN AGENT FOR PRODUCTION USE'
DEP1  = (
        AIGFLOW.Training.agent1,
        AIGFLOW.Training.agent2,
        AIGFLOW.Training.agent3,
        AIGFLOW.Training.agent4,
        )
DEP2  = AIGFLOW.Data.fe_cleaner
DEP3  = AIGFLOW.Encoding.woe_encoder
DEPS  = DEP1 + (DEP2, DEP3)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG = 'ProductionAgentPredictor'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → PRODUCTION AGENT
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class PredictorParamslDefault:
    """Direct assignment → no global flow params."""
    params = Dataclass.schema(FitParamsBase)
    setup  = dict(params)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class FitOptions(FitRegOpt):
    """
    Model-to-Fitter Registration Layer.
    
    This class acts as the bridge between the high-level Model and the low-level Fitter.
    It utilizes a registration pattern to ensure that adding new prediction 
    algorithms only requires a new entry here, keeping the top-layer API stable.
    """       
    
    @FREG.register(REG.AIG)
    class AgentPredictorFitter(FitParamsBase):
        
        config: dict|FitParamsCfg = FitParamsCfg(
            predictors = None,
            target     = None,
        )

        def _build(self, params:FitParamsBase, config:dict) -> PredictorFitter:            
            setup  = dict(params) | config
            return PredictorFitter(**setup) 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class FitParams(FitParamsBase):
    """Fitter params input design."""
    features: list

class Fitter(FitterBase[FitParams]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """       
    
    FREG = FREG
    DREG = DREG

    def _init_dash(self, train_set: pDF, test_set: pDF) -> None:
        sync = DashBackground.set_background_sync(self)
        super()._init_dash(
            train_set   = train_set, 
            test_set    = test_set,
            verbose     = False, 
            build_dash_after_init = True,
            build_at_background   = True,
            with_background_sync  = sync
        )

    def fit(self, train_set:pDF, test_set:pDF): 
        """
        Execute the champion agent fit.

        Fits the model on the curated feature subset, collects
        results and stats, and initializes the display layer.

        Parameters
        ----------
        train_set : pDF
            Training data (WOE-encoded, containing target + features).
        test_set : pDF
            Test data (WOE-encoded) — used for the final report.

        Returns
        -------
        self
        """
        self.model.fit(train_set, features = self.params.features)
        self.result: FitResult = self.model.get_result()
        self.stats:  FitStats  = self.model.get_stats()
        self._init_dash(train_set, test_set)
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
            2-D array of shape (n_samples, 2): [P(class=0), P(class=1)].
        """           
        return self.model.predict_proba(data)   
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬        



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → PRODUCTION AGENT → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    features:   list
    selected:   list
    dropped:    list
    raw:        list

@Dataclass.base
class TransResData:
    """Transfomer data output design."""
    data: pDF

class Transformer(TransformerBase[Fitter, FitStats, TransResCols]):
    """
    Transformer meta-processing between Fitter-Model.
    
    Handles the post-processing of Fitter results. Its primary role is to standardize 
    the raw output from various fitters into a unified TransResCols, TransResData
    format, enabling seamless transition regardless of the underlying fitter choice.
    """
    
    ModelFitter = Fitter

    def _fit_data(self, inputs:"AgentPredictorInput"): 
        return inputs.train_set, inputs.test_set

    def _set_cols(self):
        res = self.fitter.result        
        return TransResCols(
            features = [self.setup.target],
            selected = res.features_selected,
            dropped  = res.features_dropped,
            raw      = RawFeatures(self.setup).get(res.features_selected)
        )

    def transform(self, data:pDF) -> TransResData: 
        """Model meta-transform."""
        d = self.fitter.predict_proba(data[self.cols.selected])[:, 1]
        return TransResData(
            data = pDF(d, columns = self.cols.features, index = data.index)
        )  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → PRODUCTION AGENT → PROCESSOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Processor(ModelProcessorBase[Transformer, TransResCols, TransResData]):
    """
    Logic Processor.
    
    Coordinates the lifecycle between FitChoice selection and Parameter resolution. 
    It ensures that params and dependencies are correctly injected into the 
    Transformer - Fitter before execution begins.
    """
    
    MODEL  = MODEL
    DESC   = DESC

    Config  = FitParamsCfg
    Mapping = PredictorParamslDefault

    ModelTransformer  = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_choice(self, fx_choice:str, inputs:"AgentPredictorInput"):
        agents = self.get_model_dependecy(DEP1, is_req=False)

        defaults = self.Config(
            predictors  = PredictorAdapter(agents),
            target      = AigParam.di(inputs.target, self.setup.target)
        )  
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)   
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def get_params(self, fx_choice: str, inputs: "AgentPredictorInput", features: list):
        schema = Dataclass.schema(FitParamsBase)
        choice = {schema.agent_choice: inputs.agent_choice}
        params: FitParamsBase = self.get_fit_params(fx_choice, inputs.fx_params, choice)
        params_ = FitParams(features = features, **params)
        return params_, self.setup
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_encoders(self) -> tuple[ModelBase, ModelBase]:
        cleaner = self.get_model_dependecy(DEP2)
        encoder = self.get_model_dependecy(DEP3, is_req=False)
        return cleaner, encoder
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → PRODUCTION AGENT
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class AgentPredictorInput(ModelInput):
    """
    Champion agent fit input specification.

    Defines the data, features, and agent selection required
    to freeze the production model.

    Attributes
    ----------
    train_set : pDF
        Training data (WOE-encoded for GLM / raw for GBM).
        Must contain the target column and all selected features.
    test_set : pDF
        Test/OOT data (same encoding as train_set).
        Used for pre-flight diagnostics and the dual transform
        validation check.
    features : list | None, default=None
        Champion feature subset (user-curated). If None, all
        selected features from the training agent are used.
    agent_choice : int | str, default=1
        Champion agent identifier (key or name from the
        AGENTS registry). Determines which training agent's
        hyperparameters and feature set are inherited.

    Notes
    -----
    **Feature Resolution**: If `features` is None, the full
    selected feature list of the chosen training agent is used.
    If provided, it must be a subset of that agent's features.
    This allows the user to drop features with high SHAP drift
    or low incremental Gini (as identified in the Selection stage).

    **Data Requirements**: The data must be WOE-encoded for
    GLM agents (use `AgentPredictor.preprocess` first if
    starting from raw data). GBM agents accept raw features
    directly.
    """
    train_set: pDF
    test_set:  pDF
    features:  list = None
    agent_choice: int|str = 1

@Dataclass.base
class AgentPredictorResult(ModelResult):
    """
    Production inference output — standardized prediction contract.

    The final output of `transform`. Contains the probability
    predictions along with full feature lineage metadata,
    ensuring every production score is traceable and auditable.

    Attributes
    ----------
    data : pDF
        Prediction data — a single-column DataFrame of
        P(class=1) probabilities, indexed identically to the
        input data.
    features : list
        Output column name(s) (e.g., `['score']`).
    selected : list
        WOE features actually used by the model.
    dropped : list
        Features that were in the training agent but removed
        by the user's curation decision.
    raw_features : list
        Original (pre-WOE) feature names — the business-facing
        identifiers. Use this for reporting and documentation.

    Notes
    -----
    **Auditability**: The `selected` + `dropped` +
    `raw_features` triple provides a complete lineage:

    - `selected`: What the model sees (WOE names).
    - `dropped`: What was intentionally removed.
    - `raw_features`: What the business calls them.

    Together, they answer: "Why does this model use these
    features, and what did we decide to exclude?"
    """
    data:         pDF
    features:     list
    selected:     list
    dropped:      list
    raw_features: list

@Dataclass.base
class AgentPredictorResultDual(AgentPredictorResult):
    """
    Synchronized dual-inference output contract.

    Extends the standard result with a second prediction set,
    enabling paired train/test comparison. Used during the
    pre-flight validation check before go-live deployment.

    Attributes
    ----------
    data : pDF
        Prediction data — a single-column DataFrame of
        P(class=1) probabilities, indexed identically to the    
        input data.
    test_data : pDF
        Prediction data for the test/OOT set (same structure
        as `data`, which holds the train-set predictions).
    features : list
        Output column name(s) (e.g., `['score']`).
    selected : list
        WOE features actually used by the model.
    dropped : list
        Features that were in the training agent but removed
        by the user's curation decision.
    raw_features : list
        Original (pre-WOE) feature names — the business-facing
        identifiers. Use this for reporting and documentation.
        
    Use Case
    --------
    Before deploying to production, run `transform_dual`
    to verify that:
    1. Both train and test sets produce valid predictions.
    2. The prediction distributions are stable (no train-test
       skew in the scoring pipeline itself).
    3. The feature mappings and encoder states produce
       consistent results across both data sources.
    """
    test_data: pDF
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class AgentPredictor(Processor, ModelBaseDual):   
    """
    Champion Agent — The Production Deployment Artifact.

    The final, self-contained unit for production inference.
    Encapsulates the champion model, its preprocessing chain,
    and full feature lineage into a single deployable object.

    Lifecycle
    ---------
    1. **`fit(inputs)`**: Freezes the champion — resolves the
       agent, binds encoders, re-fits the model on the curated
       feature subset, and saves the deployment artifact.
    2. **`preprocess(data)`**: Raw → ML-ready (cleaning +
       WOE encoding) using the locked encoder states.
    3. **`transform(data)`**: ML-ready → P(class=1)
       probabilities, with full feature lineage.
    4. **`transform_dual(inputs)`**: Paired train/test
       inference for pre-flight validation.

    Self-Containment
    ----------------
    Once fitted, the object requires no external references:

    - Model weights (Logit coefficients / LGB or XGB trees).
    - Encoder states (WOE bin maps, cleaner parameters).
    - Feature lineage (selected, dropped, raw names).
    - Agent type flag (GLM vs. GBM) for preprocessing routing.

    This enables:
    - *Serialization*: Save/load for deployment (`_save`).
    - *Portability*: Move to a scoring server without the
      full pipeline.
    - *Reproducibility*: The exact same predictions are
      produced regardless of environment, given the same input.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **GLM vs. GBM Preprocessing**:
    - GLM: `preprocess` applies cleaning → WOE encoding.
    - GBM: `preprocess` applies cleaning only (raw features
      go directly to the model).

    The `is_gbm` flag (resolved at fit time) controls this
    routing automatically.

    **Feature Lineage**: Every prediction carries its full
    lineage (selected, dropped, raw features) for audit and
    governance compliance.

    Examples
    --------
    >>> from aidflow.models import AgentPredictor
    >>> agent = AgentPredictor(ctx.setup)
    >>>
    >>> # Fit: freeze the champion
    >>> agent.fit(AgentPredictorInput(
    ...     train_set = train_woe,
    ...     test_set  = test_woe,
    ...     agent_choice = 1,
    ...     features   = ['tenure_WOE', 'revenue_WOE']
    ... ))
    >>> agents.stats    # Viet to model stats
    >>>
    >>> # Production scoring (raw data)
    >>> raw = ctx.data_io.load('new_customers.csv')
    >>> ml_ready = agent.preprocess(raw)
    >>> result = agent.transform(ml_ready)
    >>> result.data          # P(class=1) for each row
    >>> result.selected      # ['tenure_WOE', 'revenue_WOE']
    >>> result.raw_features  # ['tenure', 'revenue']
    """
    _encoders = None

    def _save(self, save_path) -> None:
        features = {
            'features': self.cols.features,
            'selected': self.cols.selected
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_agents_adapter(self):
        agents = self.get_model_dependecy(DEP1, is_req=False)
        return PredictorAdapter(agents)

    def _set_agent_type(self, agent_choice: int|str, encoder: ModelBase|None) -> None:
        agents = self._get_agents_adapter()
        self.is_gbm = agents.is_gbm(agent_choice)

        if self.is_gbm or encoder is not None:
            return
        raise ValueError(
            'AIG SAFETY ► GLM Agents are WOE based - WoeEncoder must be fitted prior.'
        )   
    
    def _get_features(self, inputs: AgentPredictorInput) -> list:
        if inputs.features:
            return inputs.features
        agents = self._get_agents_adapter()
        agent  = agents.get_agent(inputs.agent_choice)
        return agent.features.selected       
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def fit(self, inputs: AgentPredictorInput, fx_choice: str = REG.AIG):
        """
        Freeze and package the champion agent for production.

        Resolves the agent, binds encoders, re-fits the model
        on the curated feature subset, and saves the deployment
        artifact. After this call, the object is immutable and
        ready for `transform` / `preprocess`.

        Parameters
        ----------
        inputs : AgentPredictorInput
            Training/test data, feature subset, and agent choice.
        fx_choice : str, default=REG.AIG
            Fitter choice key.

        Returns
        -------
        AgentPredictor
            Fitted, deployment-ready champion agent.

        Raises
        ------
        ValueError
            If `train_set` is empty or `features` is empty.
        """
        features = self._get_features(inputs)
        verify = (
            len(inputs.train_set) and len(features)
        )
        
        self._encoders = self.get_encoders()
        self._set_agent_type(inputs.agent_choice, self._encoders[1])
        
        self.choice = self.get_choice(fx_choice, inputs)
        self.params = self.get_params(fx_choice, inputs, features)

        self.processor.fit(
            inputs,
            fx_choice = self.choice,
            params    = self.params, 
            verify    = verify
        )
        self._save(inputs.save_path)
        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────
                
    def transform(self, data:pDF) -> AgentPredictorResult:
        """
        Execute production inference: ML-ready data → probabilities.

        Applies the champion model to generate P(class=1) scores.
        Verifies that all required features are present in the
        input to prevent silent failures in production.

        Parameters
        ----------
        data : pDF
            ML-ready data (WOE-encoded for GLM, raw for GBM).
            Must contain all columns in `self.cols.selected`.

        Returns
        -------
        AgentPredictorResult
            Predictions + full feature lineage metadata.

        Raises
        ------
        ValueError
            If any required feature is missing from the input.
        """
        vars_miss = [x for x in self.cols.selected if (x not in data.columns)]   
        df = self.processor.transform(data, verify = vars_miss)
        return AgentPredictorResult(
            data         = df.data, 
            features     = self.cols.features,
            selected     = self.cols.selected,
            dropped      = self.cols.dropped,
            raw_features = self.cols.raw
        )  
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def transform_dual(self, inputs: AgentPredictorInput) -> AgentPredictorResultDual:
        """
        Synchronized dual inference: train + test in one call.

        Executes `transform` on both the training and test
        sets, returning paired predictions. Used for pre-flight
        validation before go-live deployment — ensures that
        the scoring pipeline produces consistent results across
        both data sources.

        Parameters
        ----------
        inputs : AgentPredictorInput
            Must contain both `train_set` and `test_set`
            (ML-ready data).

        Returns
        -------
        AgentPredictorResultDual
            Paired predictions: `data` (train) and
            `test_data` (test), with full feature lineage.
        """
        df = self.processor.transform(inputs.train_set)
        dt = self.processor.transform(inputs.test_set)
        return AgentPredictorResultDual(
            data         = df.data, 
            test_data    = dt.data,
            features     = self.cols.features,
            selected     = self.cols.selected,
            dropped      = self.cols.dropped,
            raw_features = self.cols.raw            
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def score(self, data: pDF) -> pDF:
        """
        One-call production inference: raw data → P(class=1) scores.
        
        Equals to:
        >>> d = self.preprocess(data)
        ... return self.transform(d).data

        Parameters
        ----------
        data : pDF
            Raw input data (pre-WOE, original feature names).

        Returns
        -------
        pDF
            Single-column DataFrame of P(class=1) probabilities.
            Column name = target name. Index matches input.
        """
        d = self.preprocess(data)
        return self.transform(d).data

    def preprocess(self, data: pDF) -> pDF:
        """
        Production data pipeline: raw → ML-ready.

        Orchestrates the full transformation chain using the
        encoder states locked during `fit`. This eliminates
        training-serving skew by guaranteeing that production
        data is processed identically to training data.

        Processing Chain
        ----------------
        - **GLM agents**: cleaning → WOE encoding.
        - **GBM agents**: cleaning only (raw features go
          directly to the model).

        Parameters
        ----------
        data : pDF
            Raw input DataFrame containing the original
            (pre-WOE) feature columns.

        Returns
        -------
        pDF
            ML-ready data, formatted exactly as the champion
            model expects. Ready for `transform`.
        """
        cleaner, woe_encoder = self.encoders
        data_ = cleaner.transform(data, subset = self.cols.raw).data
        if self.is_gbm:
            return data_
        fes = RawFeatures.get_orig_fes(self.cols.selected)
        return woe_encoder.transform(data_, subset = fes).data
    
    @property
    def encoders(self):
        """
        Frozen preprocessing encoder chain.

        Provides read-only access to the (cleaner, encoder)
        tuple. Raises if accessed before `fit`.

        Returns
        -------
        tuple
            (feature_cleaner, woe_encoder)

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._encoders is None:
            raise ValueError(
                'AIG SAFETY ► Model must be fitted prior preprocessing / prediction.'
            )
        return self._encoders
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → PRODUCTION AGENT → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class AgentPredictorUI(ModelUIBase):
    """
    AIDflow Production Agent: The Deployment Artifact.
    Model Wrapper to provide UI API for worker auto execution.        
    
    This is the final, promoted agent destined for production inference. 
    Unlike the ensemble agents used during training, this class encapsulates 
    the entire 'winning' logic—integrating its specific fitting strategy, 
    associated preprocessing encoders, and feature mappings into a single, 
    deployable unit.

    Strategic Role:
    - Inference Encapsulation: Acts as the sole interface for production scoring, 
      removing all ensemble complexity to provide a lean, high-performance API.
    - End-to-End Lineage: Bridges the gap between raw input data and final 
      probability scores by internalizing the preprocessing pipeline.
    - Governance & Audit: Maintains an immutable record of selected features 
      and encoder states, ensuring that production predictions are perfectly 
      reproducible and compliant with model governance standards. 
    """

    Model  = AgentPredictor
    Option = False

    @wraps(AgentPredictor.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(), **kwargs)  
    
    @wraps(AgentPredictor.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> AgentPredictorResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬