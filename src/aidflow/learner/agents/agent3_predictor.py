"""
Agent 3 — PLS-Optimized GLM Training for High-Dimensional Spaces.

This module implements the third training agent of the AIDflow
pipeline. It applies Partial Least Squares (PLS) to identify
latent components that maximize covariance with the target,
making it suited for high-dimensional, correlated feature spaces
where the signal is distributed across many features rather than
concentrated in a few.

What It Does
------------
1. **Feature Selection**: Two-stage pipeline (same as Agents 1–2):
   - Stage 1: PLS-based component analysis. Projects features
     into a lower-dimensional space maximizing covariance with
     the target. Features contributing most to predictive
     components are retained.
   - Stage 2: Shadow pruning (permutation testing) to remove
     features whose importance is not significantly above random.
2. **Model Fitting**: Logistic Regression on the surviving features.
3. **Diagnostics**: Validation metrics (Gini, AUC, KS, lift)
   on the held-out set.
4. **Importance**: Feature ordering by contribution.

Strategy: Dimensionality Reduction
----------------------------------
Agent 3 is the "Dimensionality Expert" of the learning ensemble.
PLS works by finding the **directions** in feature space that are 
most predictive of the target — rather than selecting individual
features by penalty.

This makes Agent 3 ideal when:
- Features are highly collinear or redundant.
- The signal is spread across many correlated features
  (e.g., a dozen income-related WOE features).
- Features form natural "groups" or "axes" of variation.
- Classical linear methods suffer from multicollinearity.

Role in the Ensemble
--------------------
Agent 3 provides the **high-dimensional benchmark**. If the
features are highly collinear and Agents 1–2 produce unstable
or overly sparse selections, Agent 3's component-based approach
often produces the most robust feature set. The Comparator
stage will surface this trade-off.

Dependencies
------------
Requires the WOE Encoder stage to be completed.
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
                                        PLSBinaryGLM, 
                                        FitParamsBase,
                                        FitParamsCfg,
                                        FitResult,
                                        FitStats,
                                        FitDashLink
                                        )
from .utils_fes                 import  RawFeatures


MODEL = AIGFLOW.Training.agent3
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG TRAINING STAGE ► AGENT 3 TRAINING ► GLM Partial Least Square Optimization'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG: str = 'PLSGLM'
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
    class PLSBinaryGLMFitter(FitParamsBase):

        config: dict|FitParamsCfg = FitParamsCfg(
            target   = None,
            verbose  = True
        )

        @property
        def _model(self) -> PLSBinaryGLM:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> PLSBinaryGLM:            
            setup  = dict(params) | config
            return PLSBinaryGLM(**setup)

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

    def _fit_data(self, inputs:"Agent3Input"):
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

    def get_choice(self, fx_choice:str, inputs:"Agent3Input"):
        defaults = self.Config(
            target   = AigParam.di(inputs.target, self.setup.target),
            val_size = PRM.LOAD.valid.value,
            verbose  = AigParam.di(inputs.verbose, self.setup.verbose)
        )        
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_params(self, fx_choice:str, inputs:"Agent3Input", params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params)
        fit = FitParams(features = inputs.woe_features, **params)
        return fit, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → API
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class Agent3Input(ModelInput):
    """
    Agent 3 fit input specification.

    Defines the training data and feature set required to
    execute the PLS-optimized GLM training pipeline.

    Attributes
    ----------
    train_set : pDF
        Training data. Must contain the target column and
        WOE-encoded features. This is the primary dataset on
        which PLS component analysis and model fitting are
        performed.
    woe_features : list | None, default=None
        WOE features to consider for selection. If None, all
        non-target columns in `train_set` are used as
        candidates.

    Notes
    -----
    **WOE-Encoded Required**: Agent 3 operates on WOE-encoded
    features. Ensure the WOE Encoder stage is completed before
    invoking this agent.

    **Feature Resolution**: If `woe_features` is None, the
    full column set (excluding target) is used.    
    """
    train_set:      pDF
    woe_features:   list|None = None

@Dataclass.base
class Agent3Result(ModelResult):
    """
    Agent 3 output — the trained model prediction contract.

    The final result of the PLS-optimized GLM training. Contains
    the probability predictions along with full feature lineage
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
        WOE features that survived PLS component analysis and
        shadow pruning.
    dropped : list
        Features that were candidates but removed (low
        contribution to predictive components or below the
        shadow baseline).
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

class Agent3Predictor(Processor, ModelBase):     
    """
    Agent 3 — PLS-Optimized GLM Training for High-Dimensional Spaces.

    The "Dimensionality Expert" of the learning ensemble. Applies
    Partial Least Squares to project features into a
    lower-dimensional space maximizing covariance with the target,
    then fits an interpretable Logistic Regression on the
    surviving features.

    What It Does
    ------------
    1. **Selects** features via two-stage pipeline:
       - Stage 1: PLS component analysis — identifies latent
         directions of variation most predictive of the target.
         Features contributing to those components are retained.
       - Stage 2: Shadow pruning (permutation stability test).
    2. **Fits** Logistic Regression on the surviving features.
    3. **Diagnoses** on a held-out validation set.
    4. **Orders** features by importance.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full training pipeline — PLS selection +
      shadow pruning + logistic regression fit + diagnostics.
    - **transform(data)**: Inference — applies the trained
      model to generate P(class=1) probabilities.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **High-Dimensional**: PLS is specifically designed for
    scenarios where p ≥ n or features are highly collinear.
    It finds the "directions" of variation that are most
    predictive — rather than selecting individual features
    by penalty.

    **Interpretability**: Despite the component-based selection,
    the final model is a standard logistic regression on
    individual features. Coefficients directly represent
    log-odds contribution — suitable for regulatory
    environments.

    **When to Use**: Choose Agent 3 when the feature set is
    large, highly correlated, or when the signal is distributed
    across many features rather than concentrated in a few.

    Examples
    --------
    >>> from aidflow.training import Agent3Predictor
    >>> agent = Agent3Predictor(ctx.setup)
    >>> agent.stats          # View to Model stats
    >>>
    >>> # Train on high-dimensional WOE data
    >>> agent.fit(Agent3Input(train_set = train_woe))
    >>>
    >>> # Inference
    >>> result = agent.transform(test_woe)
    >>> result.selected      # ['rev1_WOE', 'rev2_WOE', 'ten_WOE']
    >>> result.raw_features  # ['revenue_1', 'revenue_2', 'tenure']
    """
    
    def _save(self, save_path) -> None:
        features = {
            'features': self.cols.features,
            'selected': self.cols.selected
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs: Agent3Input, fx_choice: str = REG.AIG, **params):
        """
        Train the PLS-optimized GLM agent on WOE-encoded features.

        Executes the full two-stage feature selection pipeline
        (PLS component analysis + shadow pruning), fits the
        final Logistic Regression, and computes validation
        diagnostics.

        Parameters
        ----------
        inputs : Agent3Input
            Training configuration including train_set and
            woe_features.
        fx_choice : str, optional
            The fitting strategy identifier. Defaults to REG.AIG.
        **params : dict
            Runtime parameters passed to the internal fitter
            (glm_alpha, glm_tol, shadow_quantile, cross_val_folds).

        Returns
        -------
        Agent3Predictor
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

    def transform(self, data:pDF) -> Agent3Result:
        """
        Apply the trained agent to generate P(class=1) probabilities.

        Parameters
        ----------
        data : pDF
            Input data containing the selected WOE features.

        Returns
        -------
        Agent3Result
            Predictions + full feature lineage metadata.
        """
        vars_miss = [x for x in self.cols.selected if (x not in data.columns)]                
        df = self.processor.transform(data, verify = vars_miss)
        return Agent3Result(
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
    title   = 'GLM PLS Optimization Panel'.upper()
    split   = 2
    width   = 150
    Mapping = AgentParamsMapping
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Agent3PredictorUI(ModelUIBase):
    """
    Agent 3: PLS-Optimized Predictor.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    This agent implements the third stage of the learning ensemble, focusing on 
    Partial Least Squares (PLS) optimization. It is specifically designed to 
    handle high-dimensional WOE features by projecting them into a lower-dimensional 
    space that maximizes the covariance between the predictors and the target.    
    """

    Model  = Agent3Predictor
    Option = OptionUI
        
    @wraps(Agent3Predictor.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(), **kwargs)  
    
    @wraps(Agent3Predictor.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> Agent3Result: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬