"""
Agent 1 — Sparse GLM Training & Feature Selection.

This module implements the first training agent of the AIDflow
pipeline. It applies a sparse (stepwise) regularization strategy
to identify the most statistically significant WOE features and
fit a parsimonious Logistic Regression model.

What It Does
------------
1. **Feature Selection**: Two-stage pipeline:
   - Stage 1: Cross-validated regularization search (stepwise)
     to identify statistically significant features.
   - Stage 2: Shadow pruning (permutation testing) to remove
     features whose importance is not significantly above random.
2. **Model Fitting**: Logistic Regression on the surviving
   features (train split only).
3. **Diagnostics**: Validation metrics (Gini, AUC, KS, lift)
   on the held-out set for fair comparison with tree-based agents.
4. **Importance**: Feature ordering by contribution to the model.

Strategy: Parsimony
-------------------
Agent 1 is the "Parsimony Expert" of the learning ensemble.
It uses stepwise (sequential) selection to produce the smallest
possible model that maintains predictive power. Each feature
must individually justify its inclusion by statistical
significance — no feature survives solely because it is
correlated with others.

This makes Agent 1 ideal for:
- Regulatory environments requiring maximum interpretability.
- Scorecards with strict feature count limits.
- Scenarios where each feature must be individually justified.

Trade-off: May miss features that are only significant in
combination with others (a limitation of sequential selection).

Role in the Ensemble
--------------------
Agent 1 provides the **interpretable baseline** against which
the ensemble comparison (Comparator) and selection (Selector)
stages evaluate the other agents. If Agent 1 performs
competitively on OOT data, it is often the preferred champion
for production due to its auditability and stability.

Dependencies
------------
Requires the WOE Encoder stage to be completed (Agent 1
consumes WOE-encoded features).
"""


from ...deps.common             import  wraps
from ...deps.data               import  pDF
from ...deps.types              import  NDArray

from ...core.parameters         import  (
                                        PRM,
                                        AIGFLOW
                                        )
from ...core.registry           import  (
                                        FitRegister, 
                                        FitRegOpt,   
                                        DashRegister,  
                                        RegGet
                                        )
from ...core.base               import  (
                                        FitterBase,
                                        TransformerBase,
                                        ModelProcessorBase,
                                        ModelInput,
                                        ModelResult,
                                        ModelBase,  
                                        OptionUIBase,
                                        ModelUIBase
                                        )
from ...core.factory            import  (
                                        Dataclass,
                                        DictClass
                                        )
from ...core.foundation         import  AigParam                                        
from ...core.io                 import  ModelIO

from .fitters.glm_selector      import  (
                                        SparseBinaryGLM, 
                                        FitParamsBase,
                                        FitParamsCfg,
                                        FitResult,
                                        FitStats,
                                        FitDashLink
                                        )
from .utils_fes                 import  RawFeatures


MODEL = AIGFLOW.Training.agent1
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG TRAINING STAGE ► AGENT 1 TRAINING ► GLM Sparse Optimization'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG: str = 'SparseGLM'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → FIT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class AgentParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(FitParamsBase)
    setup  = {
        PRM.TRAIN.glm_alpha.param       : (params.alpha,            'Feature Selection alpha'   ),
        PRM.TRAIN.glm_tol.param         : (params.tol,              'Optim Tolerance (tol)'     ),
        PRM.TRAIN.shadow_quantile.param : (params.shadow_quantile,  'Shadow Pruning Quantile'   ),
        PRM.TRAIN.cv.param              : (params.cv,               'Cross-Valid Shadow Folds'  ),
    }  
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class FitOptions(FitRegOpt):
    """
    Model-to-Fitter Registration Layer.
    
    This class acts as the bridge between the high-level Model and the low-level Fitter.
    It utilizes a registration pattern to ensure that adding new prediction 
    algorithms only requires a new entry here, keeping the top-layer API stable.
    """         

    @FREG.register(REG.AIG)
    class SparseBinaryGLMFitter(FitParamsBase):

        config: dict|FitParamsCfg = FitParamsCfg(
            target   = None,
            verbose  = True
        )

        @property
        def _model(self) -> SparseBinaryGLM:
            return self.model
        
        def _build(self, params:FitParamsBase, config:dict) -> SparseBinaryGLM:
            setup  = dict(params) | config
            return SparseBinaryGLM(**setup)

        def get_result(self) -> FitResult:
            return self._model.result

        def get_stats(self) -> FitStats:
            return self._model.stats
        
        def get_extra(self) -> FitDashLink:
            return self._model.dash
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class FitParams(FitParamsBase):
    """Fitter params input design."""
    features: list

@Dataclass.base
class DashLink(FitDashLink):
    """Fitter dash link."""
    data: pDF

class Fitter(FitterBase[FitParams]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """           
    
    FREG = FREG
    DREG = DREG
        
    def fit(self, data:pDF): 
        """
        ML Agent training.
        
        Parameters
        ----------
        data: pDF
            Input Data for Agent training.
        verbose: bool
            True shows dashboard and stats.
        """   
        self.model.fit(data, self.params.features)
        self.result: FitResult = self.model.get_result()
        self.stats:  FitStats  = self.model.get_stats()
        self._init_dash(
            fit_link = DashLink(data = data, **self.model.get_extra())
        )
        return self
    
    def predict_proba(self, data:pDF) -> NDArray:
        """
        Predict response variable of a model given chosen significant exogenous features.
        
        Parameters
        ----------
        data : pDF
            Input DataFrame containing the features required by the model.
            
        Returns
        ------
        NDArray
            2-D Numpy array of probabilities corresponding to the target class 0 / 1.
        """        
        return self.model.predict_proba(data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → TRANSFORMER DESIGN
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
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Transformer(TransformerBase[Fitter, FitStats, TransResCols]):
    """
    Transformer meta-processing between Fitter-Model.
    
    Handles the post-processing of Fitter results. Its primary role is to standardize 
    the raw output from various fitters into a unified TransResCols, TransResData
    format, enabling seamless transition regardless of the underlying fitter choice.
    """
    
    ModelFitter = Fitter               
    
    def _fit_data(self, inputs:"Agent1Input"): 
        return inputs.train_set 

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
# TRAINING → AGENT → PROCESSOR
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
    Mapping = AgentParamsMapping

    ModelTransformer = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_choice(self, fx_choice:str, inputs:"Agent1Input"):
        defaults = self.Config(
            target   = AigParam.di(inputs.target, self.setup.target),
            val_size = PRM.LOAD.valid.value,
            verbose  = AigParam.di(inputs.verbose, self.setup.verbose)
        )        
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_params(self, fx_choice:str, inputs:"Agent1Input", params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params)
        fit = FitParams(features = inputs.woe_features, **params)
        return fit, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → API
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class Agent1Input(ModelInput):
    """
    Agent 1 fit input specification.

    Defines the training data and feature set required to
    execute the sparse GLM training pipeline.

    Attributes
    ----------
    train_set : pDF
        Training data. Must contain the target column and
        WOE-encoded features. This is the primary dataset on
        which feature selection and model fitting are performed.
    woe_features : list | None, default=None
        WOE features to consider for selection. If None, all
        non-target columns in `train_set` are used as
        candidates.

    Notes
    -----
    **WOE-Encoded Required**: Agent 1 operates on WOE-encoded
    features. Ensure the WOE Encoder stage is completed before
    invoking this agent.

    **Feature Resolution**: If `woe_features` is None, the
    full column set (excluding target) is used.
    """
    train_set:      pDF
    woe_features:   list|None = None

@Dataclass.base
class Agent1Result(ModelResult):
    """
    Agent 1 output — the trained model prediction contract.

    The final result of the sparse GLM training. Contains the
    probability predictions along with full feature lineage
    metadata for audit and downstream consumption.

    Attributes
    ----------
    data : pDF
        Prediction data — a single-column DataFrame of
        P(class=1) probabilities, indexed identically to the
        input data.
    features : list
        Output column name(s) for the prediction
        (e.g., `['score']`).
    selected : list
        WOE features that survived the two-stage selection
        (regularization + shadow pruning).
    dropped : list
        Features that were candidates but removed during
        selection. The implicit reason is the stage at which
        they were dropped (Stage 1: not significant; Stage 2:
        not above shadow baseline).
    raw_features : list
        Original (raw input) feature names for the selected
        features — the business-facing identifiers.
    """
    data:           pDF
    features:       list
    selected:       list
    dropped:        list
    raw_features:   list
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Agent1Predictor(Processor, ModelBase):     
    """
    Agent 1 — Sparse GLM Training & Feature Selection.

    The "Parsimony Expert" of the learning ensemble. Applies a
    stepwise (sequential) regularization strategy to identify
    the most statistically significant WOE features and fit a
    minimal, interpretable Logistic Regression model.

    What It Does
    ------------
    1. **Selects** features via two-stage pipeline:
       - Stage 1: Cross-validated stepwise regularization.
       - Stage 2: Shadow pruning (permutation stability test).
    2. **Fits** Logistic Regression on the surviving features.
    3. **Diagnoses** on a held-out validation set (Gini, AUC,
       KS, lift table).
    4. **Orders** features by importance for downstream use.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full training pipeline — selection +
      fitting + diagnostics. Produces the trained model.
    - **transform(data)**: Inference — applies the trained
      model to generate P(class=1) probabilities.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **Interpretability**: The final model is a standard logistic
    regression. Coefficients directly represent the log-odds
    contribution of each WOE feature — ideal for regulatory
    environments requiring explainability.

    **Parsimony**: The stepwise strategy produces the smallest
    model that maintains predictive power. Each feature must
    individually justify its inclusion — no feature survives
    solely due to correlation with others.

    **Trade-off**: May miss features that are only significant
    in combination. For those scenarios, consider Agent 2
    (Lasso) or Agent 3 (ElasticNet).

    **WOE Alignment**: WOE features are approximately linear in
    log-odds space, so the linear model assumption is
    well-satisfied.

    Examples
    --------
    >>> from aidflow.training import Agent1Predictor
    >>> agent = Agent1Predictor(ctx.setup)
    >>>
    >>> # Train on WOE-encoded data
    >>> agent.fit(Agent1Input(train_set = train_woe))
    >>> agent.stats          # View to Model stats
    >>>
    >>> # Inference
    >>> result = agent.transform(test_woe)
    >>> result.data          # P(class=1) for each row
    >>> result.selected      # ['tenure_WOE', 'revenue_WOE', ...]
    >>> result.raw_features  # ['tenure', 'revenue', ...]
    """
    
    def _save(self, save_path) -> None:
        features = {
            'features': self.cols.features,
            'selected': self.cols.selected
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    #     
    def fit(self, inputs: Agent1Input, fx_choice: str = REG.AIG, **params):
        """
        Train the sparse GLM agent on WOE-encoded features.

        Executes the full two-stage feature selection pipeline
        (stepwise regularization + shadow pruning), fits the
        final Logistic Regression, and computes validation
        diagnostics.

        Parameters
        ----------
        inputs : Agent1Input
            Training configuration including train_set and
            woe_features.
        fx_choice : str, optional
            The fitting strategy identifier. Defaults to REG.AIG.
        **params : dict
            Runtime parameters passed to the internal fitter
            (glm_alpha, glm_tol, shadow_quantile, cross_val_folds).

        Returns
        -------
        Agent1Predictor
            The fitted agent instance.
        """
        verify = ( 
            len(inputs.train_set) and
            ( inputs.woe_features is None or len(inputs.woe_features) )
        )
        
        self.choice = self.get_choice(fx_choice, inputs)
        self.params = self.get_params(fx_choice, inputs, params)

        self.processor.fit(
            inputs,
            fx_choice = self.choice,
            params    = self.params, 
            verify    = verify
        )
        self._save(inputs.save_path)
        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def transform(self, data:pDF) -> Agent1Result:
        """
        Apply the trained agent to generate P(class=1) probabilities.

        Parameters
        ----------
        data : pDF
            Input data containing the selected WOE features.

        Returns
        -------
        Agent1Result
            Predictions + full feature lineage metadata.
        """
        vars_miss = [x for x in self.cols.selected if (x not in data.columns)]                
        df = self.processor.transform(data, verify = vars_miss)
        return Agent1Result(
            data         = df.data, 
            features     = self.cols.features,
            selected     = self.cols.selected,
            dropped      = self.cols.dropped,
            raw_features = self.cols.raw
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI(OptionUIBase):    
    title   = 'GLM Sparse Optimization Panel'.upper()
    split   = 2
    width   = 150
    Mapping = AgentParamsMapping
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Agent1PredictorUI(ModelUIBase):
    """
    Agent 1: Sparse GLM Optimization & Feature Selection.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    Agent 1 serves as the 'Parsimony Expert' of the learning ensemble. Utilizing a 
    Sparse fitting strategy, it focuses on identifying the most potent predictors 
    while aggressively pruning redundant or low-impact variables from the WOE feature set.    
    """

    Model  = Agent1Predictor
    Option = OptionUI
        
    @wraps(Agent1Predictor.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(), **kwargs)  
    
    @wraps(Agent1Predictor.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> Agent1Result: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬