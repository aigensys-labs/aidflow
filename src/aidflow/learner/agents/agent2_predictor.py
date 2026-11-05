"""
Agent 2 — Regularized GLM Training & Feature Selection.

This module implements the second training agent of the AIDflow
pipeline. It provides three regularized GLM strategies (Lasso,
ElasticNet, LogReg) that balance feature selection with
coefficient stabilization, producing a robust predictive surface
that does not over-rely on any single feature.

What It Does
------------
1. **Feature Selection**: Two-stage pipeline (same protocol as
   Agent 1):
   - Stage 1: Cross-validated regularization search.
   - Stage 2: Shadow pruning (permutation stability test).
2. **Model Fitting**: Logistic Regression on the surviving
   features (train split only).
3. **Diagnostics**: Validation metrics (Gini, AUC, KS, lift)
   on the held-out set.
4. **Importance**: Feature ordering by contribution to the model.

Solver Strategies
-----------------
Agent 2 supports three regularization strategies, selectable via
`fx_choice`:

- **Lasso (L1)** — `REG.LASSO`: Simultaneous coefficient
  shrinkage and selection. Picks ONE feature from a correlated
  group. Best for high-dimensional, multicollinear inputs.

- **ElasticNet (L1+L2)** — `REG.ELASTIC` (default): Combines
  Lasso selection with Ridge stability. Keeps ALL relevant
  features from a correlated group (not just one). The most
  robust default for production scorecards.

- **LogReg (unpenalized)** — `REG.LOGREG`: No regularization in
  the selection stage. All features retained; shadow pruning is
  the sole elimination mechanism. Best for small, curated
  feature sets.

Strategy: Stability
-------------------
Agent 2 is the "Stability Bridge" of the learning ensemble.
It uses simultaneous regularization to handle correlated
feature groups gracefully. The L2 component prevents the
arbitrary single-feature selection behavior of pure Lasso,
producing a more stable coefficient surface.

This makes Agent 2 ideal for:
- Feature groups with high correlation (e.g., multiple
  revenue-based WOE features).
- Scenarios where you want ALL relevant features from a
  correlated group.
- Production scorecards where feature availability may
  fluctuate (stable coefficients matter).

Trade-off: More features retained than Agent 1.
The shadow pruning stage compensates by removing truly
redundant ones.

Role in the Ensemble
--------------------
Agent 2 provides the **stability-oriented baseline**. In the
ensemble comparison, if Agent 2 outperforms Agent 1 on OOT
data, it indicates that correlated feature groups carry
meaningful signal that stepwise selection would have discarded.
If Agent 1 outperforms Agent 2, the simpler model is preferred
(Occam's razor).

Dependencies
------------
Requires the WOE Encoder stage to be completed (Agent 2
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
from ...core.base               import  FitterModelBase
from ...core.factory            import  (
                                        Dataclass,
                                        DictClass
                                        )
from ...core.foundation         import  AigParam
from ...core.io                 import  ModelIO

from .fitters.glm_selector      import  (
                                        LassoBinaryGLM,
                                        ElasticBinaryGLM,
                                        LogRegBinaryGLM,
                                        FitParamsBase,
                                        FitParamsCfg,
                                        FitResult,
                                        FitStats,
                                        FitDashLink
                                        )
from .utils_fes                 import  RawFeatures


MODEL = AIGFLOW.Training.agent2
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG TRAINING STAGE ► AGENT 2 TRAINING ► GLM Dense Optimization'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    LASSO:   str = 'LassoGLM'
    ELASTIC: str = 'ElasticGLM'
    LOGREG:  str = 'LogRegGLM'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → FIT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class AgentParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(FitParamsBase)
    setup  = {
        PRM.TRAIN.glm_eps.param         : (params.eps,              'Optim Epsilon (eps)'       ),
        PRM.TRAIN.glm_tol.param         : (params.tol,              'Optim Tolerance (tol)'     ),
        PRM.TRAIN.shadow_quantile.param : (params.shadow_quantile,  'Shadow Pruning Quantile'   ),
        PRM.TRAIN.cv.param              : (params.cv,               'Cross-Valid Shadow Folds'  ),
    }  
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class InnerFitterBase(FitParamsBase):

    config: dict|FitParamsCfg = FitParamsCfg(
        target   = None,
        verbose  = True
    )

    @property
    def _model(self) -> FitterModelBase:
        return self.model  
    
    def get_result(self) -> FitResult:
        return self._model.result

    def get_stats(self) -> FitStats:
        return self._model.stats
    
    def get_extra(self) -> FitDashLink:
        return self._model.dash

class FitOptions(FitRegOpt):
    """
    Model-to-Fitter Registration Layer.
    
    This class acts as the bridge between the high-level Model and the low-level Fitter.
    It utilizes a registration pattern to ensure that adding new prediction 
    algorithms only requires a new entry here, keeping the top-layer API stable.
    """        

    @FREG.register(REG.LASSO)
    class LassoBinaryGLMFitter(InnerFitterBase):

        @property
        def _model(self) -> LassoBinaryGLM:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> LassoBinaryGLM:            
            setup  = dict(params) | config
            return LassoBinaryGLM(**setup)

    @FREG.register(REG.ELASTIC)
    class ElasticBinaryGLMFitter(InnerFitterBase):

        @property
        def _model(self) -> ElasticBinaryGLM:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> ElasticBinaryGLM:            
            setup  = dict(params) | config
            return ElasticBinaryGLM(**setup)

    @FREG.register(REG.LOGREG)
    class LogRegBinaryGLMFitter(InnerFitterBase):

        @property
        def _model(self) -> LogRegBinaryGLM:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> LogRegBinaryGLM:            
            setup  = dict(params) | config
            return LogRegBinaryGLM(**setup)                
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

    def _fit_data(self, inputs:"Agent2Input"): 
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

    def get_choice(self, fx_choice:str, inputs:"Agent2Input"):
        defaults = self.Config(
            target   = AigParam.di(inputs.target, self.setup.target),
            val_size = PRM.LOAD.valid.value,
            verbose  = AigParam.di(inputs.verbose, self.setup.verbose)
        )        
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_params(self, fx_choice:str, inputs:"Agent2Input", params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params)
        fit = FitParams(features = inputs.woe_features, **params)
        return fit, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → API
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class Agent2Input(ModelInput):
    """
    Agent 2 fit input specification.

    Defines the training data and feature set required to
    execute the dense GLM training pipeline.

    Attributes
    ----------
    train_set : pDF
        Training data. Must contain the target column and
        WOE-encoded features. This is the primary dataset on
        which regularization-based selection and model fitting
        are performed.
    woe_features : list | None, default=None
        WOE features to consider for selection. If None, all
        non-target columns in `train_set` are used as
        candidates.

    Notes
    -----
    **WOE-Encoded Required**: Agent 2 operates on WOE-encoded
    features. Ensure the WOE Encoder stage is completed before
    invoking this agent.

    **Feature Resolution**: If `woe_features` is None, the
    full column set (excluding target) is used.
    """
    train_set:      pDF
    woe_features:   list|None = None

@Dataclass.base
class Agent2Result(ModelResult):
    """
    Agent 2 output — the trained model prediction contract.

    The final result of the dense GLM training. Contains the
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
        selection.
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

class Agent2Predictor(Processor, ModelBase):     
    """
    Agent 2 — Dense GLM Training with Regularized Feature Selection.

    The "Stability Bridge" of the learning ensemble. Applies
    simultaneous regularization-based feature selection (L1,
    L1+L2, or unpenalized) to handle correlated feature groups
    robustly, then fits an interpretable Logistic Regression.

    What It Does
    ------------
    1. **Selects** features via two-stage pipeline:
       - Stage 1: Cross-validated regularization search
         (Lasso / ElasticNet / LogReg — user's choice).
       - Stage 2: Shadow pruning (permutation stability test).
    2. **Fits** Logistic Regression on the surviving features.
    3. **Diagnoses** on a held-out validation set.
    4. **Orders** features by importance.

    Two Entry Points
    ----------------
    - **fit(inputs, fx_choice)**: Full training pipeline.
      `fx_choice` selects the regularization strategy:
      - `REG.ELASTIC` (default): L1+L2 — most robust.
      - `REG.LASSO`: L1 only — most compact.
      - `REG.LOGREG`: Unpenalized — maximum control.
    - **transform(data)**: Inference — applies the trained
      model to generate P(class=1) probabilities.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **Correlated Groups**: The regularization strategies 
    handle correlated feature groups simultaneously:
    - Lasso picks ONE from a group.
    - ElasticNet keeps ALL with stable coefficients.
    - LogReg retains all (pruned by shadow quantile only).

    **Interpretability**: The final model is a standard logistic
    regression — coefficients directly represent log-odds
    contribution. Suitable for regulatory environments.

    **Stability**: Simultaneous regularization produces a
    solution that is invariant to feature ordering (unlike
    stepwise selection). Critical for reproducible scorecards.

    **Default Strategy**: ElasticNet (L1+L2) is the recommended
    default. It combines Lasso's sparsity with Ridge's
    stability — the most robust choice for production scorecards
    with correlated features.

    Examples
    --------
    >>> from aidflow.training import Agent2Predictor
    >>> agent = Agent2Predictor(ctx.setup)
    >>>
    >>> # Default: ElasticNet
    >>> agent.fit(Agent2Input(train_set = train_woe))
    >>> agent.stats          # View to Model stats
    >>>
    >>> # Or specify Lasso for maximum sparsity
    >>> agent.fit(Agent2Input(train_set = train_woe), fx_choice='LassoGLM')
    >>>
    >>> # Inference
    >>> result = agent.transform(test_woe)
    >>> result.selected      # ['rev_WOE', 'ten_WOE', 'seg_WOE']
    >>> result.raw_features  # ['revenue', 'tenure', 'segment']
    """
    
    def _save(self, save_path) -> None:
        features = {
            'features': self.cols.features,
            'selected': self.cols.selected
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs: Agent2Input, fx_choice: str = REG.ELASTIC, **params):
        """
        Train the dense GLM agent on WOE-encoded features.

        Executes the full two-stage feature selection pipeline
        (regularization search + shadow pruning), fits the
        final Logistic Regression, and computes validation
        diagnostics.

        Parameters
        ----------
        inputs : Agent2Input
            Training configuration including train_set and
            woe_features.
        fx_choice : str, optional
            The regularization strategy: REG.ELASTIC (default),
            REG.LASSO, or REG.LOGREG.
        **params : dict
            Runtime parameters passed to the internal fitter
            (glm_eps, glm_tol, shadow_quantile, cross_val_folds).

        Returns
        -------
        Agent2Predictor
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

    def transform(self, data:pDF) -> Agent2Result:
        """
        Apply the trained agent to generate P(class=1) probabilities.

        Parameters
        ----------
        data : pDF
            Input data containing the selected WOE features.

        Returns
        -------
        Agent2Result
            Predictions + full feature lineage metadata.
        """
        vars_miss = [x for x in self.cols.selected if (x not in data.columns)]
        df = self.processor.transform(data, verify = vars_miss)
        return Agent2Result(
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
    title   = 'GLM Dense Optimization Panel'.upper()
    split   = 2
    width   = 150
    Mapping = AgentParamsMapping
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Agent2PredictorUI(ModelUIBase):
    """
    Agent 2: ElasticNet & Balanced Optimization.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    Agent 2 serves as the 'Stability Bridge' of the learning ensemble. By utilizing 
    ElasticNet regularization—a hybrid of L1 (Lasso) and L2 (Ridge)—it balances 
    feature selection with coefficient shrinkage to ensure a robust predictive 
    surface that doesn't over-rely on any single feature.        
    """

    Model  = Agent2Predictor
    Option = OptionUI
        
    @wraps(Agent2Predictor.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(), **kwargs)  
    
    @wraps(Agent2Predictor.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> Agent2Result: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬