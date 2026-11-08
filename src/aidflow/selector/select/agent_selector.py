"""
Agent Selector — Champion Selection & Deployment Decision Layer.

This module implements the decision-making stage of the AIDflow
pipeline. It sits above all trained and compared agents and provides
the user with a comprehensive, evidence-based interface to:

1. **Evaluate** all trained agents on OOT/test data with full
   diagnostics (performance, SHAP drift, Gini path, permutation
   importance).
2. **Choose** the champion agent — either by full-feature usage
   or by manually curating a feature subset.
3. **Audit** the selection decision: side-by-side comparison of
   the Training Agent vs. the Champion Agent, with static HTML
   overviews for archival and stakeholder communication.
4. **Deploy**: Produce the final ``AgentPredictor`` — a
   self-contained, production-ready inference object.

Role in the Pipeline
--------------------
This is the **final authority** stage. All prior work (training,
encoding, comparison) produces evidence. This module is where
the evidence is consolidated into a single deployment decision.

The selector does **not** re-train any model. It:
- Evaluates existing agents on new data.
- Computes diagnostics supporting the choice.
- Delegates the champion fit to the ``AgentPredictor`` model.
- Packages the full audit trail (performance, SHAP, WOE plots).

Two Usage Modes
---------------
- **``fit(inputs)``**: Execute the full selection pipeline —
  evaluate agents, fit the champion, produce the audit package.
  This is the primary entry point during the AIDflow workflow.

- **``transform(data)``**: Generate a fresh audit report on any
  dataset. Use this to re-evaluate the selection decision on a
  new sample (e.g., a subsequent OOT window) without re-fitting.

UI Integration
--------------
The ``AgentSelectorUI`` variant adds a dynamic parameter panel
where the user can:
- Toggle the agent choice (int → full features; dict → subset).
- Adjust the significance level (alpha) and bin count.
- Confirm the selection with a single button click.
- View the static overview (SHAP drift, Gini path, WOE plots)
  and the dynamic champion-vs-training comparison.

Dependencies
------------
Requires the following pipeline stages to be completed:
- Agent Training (4 agents: GLM variants + GBM variants)
- Feature Cleaner (raw data standardization)
- WOE Encoder (feature encoding for GLM agents)
- Comparator (performance ranking across agents)
- Data Encoder (full preprocessing chain for inference)
"""

from ...deps.common             import  wraps
from ...deps.data               import  pDF

from ...core.parameters         import  (
                                        PRM, 
                                        AIGFLOW,
                                        Parameters
                                        )
from ...core.registry           import  (
                                        ProjectModelRegistry,
                                        FitRegister, 
                                        FitRegOpt,   
                                        DashRegister, 
                                        RegGet
                                        )
from ...core.base               import  (
                                        FitterBase,
                                        TransformerBase,
                                        ModelProcessorBase,
                                        ModelEnvelope,
                                        ModelInput,
                                        ModelResult,
                                        ModelBase,
                                        ModelUIBase
                                        )
from ...core.base.worker        import  WorkerBase
from ...core.base.adapter       import  ModelAdapterBase
from ...core.factory            import  (
                                        Dataclass,
                                        DictClass
                                        )
from ...core.foundation         import  AigParam
from ...core.io                 import  ModelIO

from ...display.verbosity       import  Verbosity

from ..predict.adapter          import  PredictorAdapter
from ..compare.processor        import  ComparatorStats

from .processor                 import  (
                                        SelectorProcessor,
                                        SelectorParamsBase,
                                        SelectorParamsCfg,
                                        SelectorResult,
                                        SelectorStats
                                        )
from .agent                     import  AgentPredictor
from .utils_ui                  import  PanelUtilsUI


MODEL = AIGFLOW.Selection.selector
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG SELECTION STAGE ► AGENT CHOICE ► SELECTION OF THE FINAL AGENT AS PRODUCTION CANDIDATE'
DEP1  = (
        AIGFLOW.Training.agent1,
        AIGFLOW.Training.agent2,
        AIGFLOW.Training.agent3,
        AIGFLOW.Training.agent4,
        )
DEP2  = AIGFLOW.Data.fe_cleaner
DEP3  = AIGFLOW.Encoding.woe_encoder
DEP4  = AIGFLOW.Selection.comparator
DEP5  = AIGFLOW.Encoding.data_encoder
DEPS  = DEP1 + (DEP2, DEP3, DEP4, DEP5)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG = 'AgentSelector'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENT CHOICE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class SelectorParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(SelectorParamsBase)
    setup  = {
        PRM.SELECT.select_alpha.param   : (params.alpha,        'Test alpha'                 ),
        PRM.SELECT.select_bins.param    : (params.proba_bins,   'Probability Discretization' ),
        PRM.SELECT.select_choice.param  : (params.agent_choice, 'Agent Selection'            )
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
    class AgentSelectorFitter(SelectorParamsBase):
        
        config: dict|SelectorParamsCfg = SelectorParamsCfg(
            predictors  = None,
            fe_cleaner  = None,
            woe_encoder = None,
            comparator  = None,
            predictor   = None,
            target      = None,
            verbose     = True
        )

        def _build(self, params:SelectorParamsBase, config:dict) -> SelectorProcessor:            
            setup  = dict(params) | config
            return SelectorProcessor(**setup)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Fitter(FitterBase[SelectorParamsBase]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """       
    
    FREG = FREG
    DREG = DREG
    
    def fit(self, test_set:pDF, train_set:pDF=None): 
        """
        Execute the selection pipeline: evaluate agents and fit champion.

        Parameters
        ----------
        test_set : pDF
            OOT/test data for evaluation and diagnostics.
            Must contain the target column.
        train_set : pDF | None
            Training data (for SHAP drift comparison and
            champion fitting). If None, only test diagnostics
            are produced.

        Returns
        -------
        self
        """    
        self.result: SelectorResult = self.model.process(test_set, train_set = train_set)
        self.stats:  SelectorStats  = self.model.get_stats()
        
        is_info = Verbosity.normalize(self.verbose) == Verbosity.INFO
        
        self._init_dash(
            verbose = self.verbose if is_info else False,
            build_dash_after_init = self.verbose            
        )
        return self
    
    def transfrom(self, data:pDF) -> SelectorResult:
        """
        Generate a fresh selection audit report on new data.

        Parameters
        ----------
        data : pDF
            Input data for the performance and SHAP evaluation.

        Returns
        -------
        SelectorResult
            Updated performance and SHAP diagnostics.
        """
        return self.model.process(data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬        



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENT CHOICE → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    pass

@Dataclass.base
class TransResData(SelectorResult):
    """Transfomer data output design."""

class Transformer(TransformerBase[Fitter, SelectorStats, TransResCols]):
    """
    Transformer meta-processing between Fitter-Model.
    
    Handles the post-processing of Fitter results. Its primary role is to standardize 
    the raw output from various fitters into a unified TransResCols, TransResData
    format, enabling seamless transition regardless of the underlying fitter choice.
    """
    
    ModelFitter = Fitter

    def _fit_data(self, inputs: "AgentSelectorInput", train_data: pDF):
        return inputs.data, train_data
    
    def transform(self, data:pDF) -> TransResData: 
        """Model meta-transform."""
        return TransResData(
            **self.fitter.transfrom(data)
        )   
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENT CHOICE → PROCESSOR
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

    Config  = SelectorParamsCfg
    Mapping = SelectorParamsMapping

    ModelTransformer  = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_choice(self, fx_choice:str, inputs:"AgentSelectorInput"):
        agents   = self.get_model_dependecy(DEP1, is_req=False)
        cleaner  = self.get_model_dependecy(DEP2)
        encoder  = self.get_model_dependecy(DEP3)
        compare  = self.get_model_dependecy(DEP4)

        defaults = self.Config(
            predictors  = PredictorAdapter(agents),
            fe_cleaner  = ModelAdapterBase(cleaner),
            woe_encoder = ModelAdapterBase(encoder),
            comparator  = ModelAdapterBase(compare),
            predictor   = ModelAdapterBase(inputs.predictor),
            target      = AigParam.di(inputs.target, self.setup.target),
            verbose     = AigParam.di(inputs.verbose, self.setup.verbose),
        )  
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)   
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def get_params(self, fx_choice:str, inputs:"AgentSelectorInput", params:dict):
        params: SelectorParamsBase = self.get_fit_params(fx_choice, inputs.fx_params, params) 
        
        # Switch of Static Performance Overview Creation
        schema  = Dataclass.schema(SelectorParamsBase)
        p_html  = schema.create_html_overview
        verbose = AigParam.di(inputs.verbose, self.setup.verbose) == Verbosity.GUI
        is_init = True if isinstance(params.agent_choice, int) and verbose else False
        params_ = SelectorParamsBase(**( dict(params) | {p_html: is_init} )) 
        
        return params_, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENT CHOICE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class AgentSelectorInput(ModelInput):
    """
    Agent selector fit input specification.

    Defines the data and champion predictor required to execute
    the selection pipeline.

    Attributes
    ----------
    data : pDF
        OOT/test data for evaluation and diagnostics.
        Must contain the target column and all agent features.
        This is the primary evidence set — the selector evaluates
        all agents on this data to support the deployment decision.
    predictor : WorkerBase | AgentPredictor
        The champion agent predictor (or its worker wrapper).
        This is the agent that will be fitted as the production
        model. In the standard AIDflow flow, this is resolved
        from the UI panel selection.

    Notes
    -----
    **Data vs. Train**: The `data` field is the OOT/test set
    used for diagnostics. The training data (for SHAP drift and
    champion fitting) is resolved separately from the pipeline
    dependencies (`DEP5`: data encoder).

    **Predictor Resolution**: In the UI flow, the user selects
    the champion agent via the parameter panel. The `predictor`
    field carries that selection. In a programmatic flow, the
    user can pass a pre-fitted `AgentPredictor` directly.
    """
    data: pDF
    predictor: WorkerBase | AgentPredictor

@Dataclass.base(dynamic_params=True)
class AgentSelectorEnvelopeCfg:
    """
    Selector envelope configuration — additional data references.

    Attributes
    ----------
    train_ml : pDF
        Training data (ML-ready) for SHAP drift comparison and
        champion fitting. Resolved from the data encoder if not
        provided directly.
    """    
    train_ml:  pDF

@Dataclass.base(dynamic_params=True)
class AgentSelectorEnvelope(ModelEnvelope):
    """
    Selector envelope — full input + config bundle.

    Wraps the primary input with additional configuration
    (training data reference) for the `fit` method.

    Attributes
    ----------
    inputs : AgentSelectorInput
        Primary input (data, predictor).
    config : AgentSelectorEnvelopeCfg
        Additional configuration (train_ml).
    """
    inputs: AgentSelectorInput 
    config: AgentSelectorEnvelopeCfg

@Dataclass.base
class AgentSelectorResult(ModelResult):
    """
    Agent selector output — the complete deployment decision package.

    The final result of the selection pipeline. Contains the
    deployable champion predictor along with the full audit
    trail supporting the selection decision.

    Attributes
    ----------
    agent_predictor : AgentPredictor
        The fitted champion agent predictor — a self-contained,
        deployable prediction interface. Use `.transform(data)`
        for production scoring, `.preprocess(data)` for raw
        data scoring.

    performance : dict[int, pDF]
        Per-agent performance comparison tables. Each table
        shows Gini/IV/KS for the Training Agent vs. the
        Champion Agent (impact of feature selection).

    shap : dict[int, pDF]
        Per-agent SHAP drift analysis tables. Each table shows
        train vs. test mean SHAP per feature, with significance
        flags (Welch's t-test). Highlights features whose
        behavior changed between train and test.

    Notes
    -----
    **Decision Support**: The `performance` and `shap`
    dictionaries provide the evidence for the deployment
    decision:

    - `Performance drift` (Test - Valid): Is the champion
      performing consistently? Large negative drift suggests
      overfitting or distribution shift.

    - `SHAP significance` Which features are behaving
      differently on test data? Features with significant
      drift are candidates for removal in the next curation
      round.

    **Champion vs. Training**: The `performance` dict
    compares the original training agent (all features) against
    the champion (possibly a subset). The difference quantifies
    the cost/benefit of the user's feature curation decision.
    """
    agent_predictor: AgentPredictor
    performance:     dict
    shap:            dict
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class AgentSelector(Processor, ModelBase):   
    """
    Agent Selector — Champion Selection & Deployment Decision.

    The final authority stage of the AIDflow pipeline. Aggregates
    all evidence from prior stages (training, comparison) and
    provides the user with a comprehensive interface to evaluate,
    select, and deploy the champion agent.

    What It Does
    ------------
    1. **Evaluates** all trained agents on OOT/test data:
       - Performance (Gini, IV, KS) — training vs. test drift.
       - SHAP value drift — feature stability between train/test.
       - Gini path — incremental value of each feature.
       - Permutation importance — feature robustness.
       - WOE plots — visual feature behavior on test.

    2. **Selects** the champion agent:
       - `agent_choice=int`: Use the agent with all features.
       - `agent_choice=dict`: Use the agent with a curated
         feature subset (e.g., `{1: ['rev_WOE', 'ten_WOE']}`).

    3. **Fits** the champion via `AgentPredictor`:
       - Re-fits the model on the final feature subset.
       - Binds the preprocessing encoders.
       - Produces a self-contained deployment artifact.

    4. **Audits** the decision:
       - Side-by-side diagnostics (Training vs. Champion).
       - Static HTML overview (for archival).
       - Dynamic comparison plot (champion vs. training Gini/Lift).

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full pipeline — evaluate, select, fit,
      audit. The primary entry point.
    - **transform(data)**: Fresh audit on any dataset.
      Re-evaluates the champion on new data without re-fitting.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **No Retraining**: The selector does not train new models.
    It evaluates existing agents and delegates the champion fit
    to `AgentPredictor` (which re-fits on the feature subset).

    **Self-Contained Output**: The `agent_predictor` in the
    result is a standalone object — it holds all encoding maps,
    model parameters, and feature names for production inference.

    Examples
    --------
    >>> from aidflow.models import AgentSelector
    >>> selector = AgentSelector(ctx.setup)
    >>>
    >>> # Full selection pipeline
    >>> selector.fit(AgentSelectorInput(
    ...     data = oot_data,
    ...     predictor = champion_worker
    ... ))    
    >>> selector.fitted     # View to model result
    >>> selector.stats      # View to model stats
    >>>
    >>> # Re-evaluate on a new sample
    >>> fresh = selector.transform(new_oot_data)
    >>>
    >>> # Inspect the audit trail
    >>> result.agent_predictor      # Deployable predictor
    >>> result.performance          # Per-agent Gini/IV/KS
    >>> result.shap                 # Per-agent SHAP drift
    """

    def _save(self, save_path) -> None:
        ModelIO(self.setup).save(MODEL, self, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_inputs(inputs: AgentSelectorEnvelope | AgentSelectorInput) -> AgentSelectorInput:
        if isinstance(inputs, AgentSelectorEnvelope):
            return inputs.inputs
        return inputs

    def _get_train_ml(self, inputs: AgentSelectorEnvelope | AgentSelectorInput) -> pDF:
        if isinstance(inputs, AgentSelectorEnvelope):
            return inputs.config.train_ml
        enc = self.get_model_dependecy(DEP5)
        return enc.transform(inputs.data).data
        
    def _get_result(self, res: TransResData):
        return AgentSelectorResult(
            agent_predictor = res.agent_predictor,
            performance = res.performance,
            shap = res.shap,
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
            
    def fit(self, inputs: AgentSelectorInput, fx_choice: str = REG.AIG, **params):
        """
        Execute the full champion selection pipeline.

        Evaluates all agents on the provided data, computes the
        diagnostic bundle (performance, SHAP drift, Gini path,
        permutation importance), fits the champion agent, and
        produces the complete audit package.

        Parameters
        ----------
        inputs : AgentSelectorInput
            Input data (OOT/test set) and the champion predictor
            reference.
        fx_choice : str, default=REG.AIG
            Fitter choice key.
        **params : dict
            Runtime parameter overrides, passed to the internal fitter
            (select_alpha, select_bins, select_choice). 

        Returns
        -------
        AgentSelector
            The fitted selector instance holding the promoted champion agent.

        Notes
        -----
        **Champion Fit**: The champion is fitted via
        `AgentPredictor` — it re-fits the model on the selected
        feature subset and binds the preprocessing encoders.
        The result is a self-contained deployment artifact.
        """
        inputs_ = self._get_inputs(inputs)
        verify  = len(inputs_.data)
        
        self.choice = self.get_choice(fx_choice, inputs_)
        self.params = self.get_params(fx_choice, inputs_, params)
        
        self.processor.fit(
            inputs_, self._get_train_ml(inputs),
            fx_choice = self.choice,
            params    = self.params, 
            verify    = verify
        )
        self._save(inputs_.save_path)
        return self    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def transform(self, data: pDF) -> AgentSelectorResult:
        """
        Generate a fresh selection audit report on any dataset.

        Re-evaluates the champion agent's diagnostics (performance,
        SHAP drift) on a new data sample without re-fitting.
        Use this to:
        - Monitor the champion on a subsequent OOT window.
        - Detect distribution shift before it impacts production.
        - Generate a fresh audit report for compliance.

        Parameters
        ----------
        data : pDF
            The dataset to evaluate (must contain target + features).
            Typically a new OOT sample.

        Returns
        -------
        AgentSelectorResult
            Updated performance and SHAP diagnostics on the
            new data, with the same champion predictor.

        Notes
        -----
        **No Re-fitting**: This method does not re-train the
        champion. It only re-computes the diagnostics on the
        new data. The `agent_predictor` remains the same
        fitted instance from `fit`.

        **Use Case**: If your OOT window shifts (e.g., next
        month's data), call `transform` on the new sample to
        verify that the champion's feature behavior remains
        stable. If SHAP drift increases, consider re-selecting.
        """ 
        res = self.processor.transform(data)    
        return self._get_result(res)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENT CHOICE → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI:    
    """UI Dynamic Parameters Panel."""

    def __init__(self, setup: Parameters, inputs: AgentSelectorInput):
        prm     = SelectorParamsMapping(setup, default=True)
        sig     = setup.project_signature
        order   = self._get_order(sig)
        verbose = AigParam.di(inputs.verbose, setup.verbose)
        self.wg = PanelUtilsUI(prm, inputs.fx_params, order, sig, verbose) 
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_order(project_sig: str) -> list:
        compare: ModelBase = ProjectModelRegistry.get_model(project_sig, DEP4)
        stats: ComparatorStats = compare.stats
        return stats.ordered
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def panel(self, stats:SelectorStats):
        """Shows Main UI Options for ML fx fit."""        
        self.wg.panel(stats)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def get_ui_setting(self) -> list:
        return self.wg.get_ui_setting()               
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENT CHOICE → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class AgentSelectorUI(ModelUIBase[AgentSelector, OptionUI]):
    """
    Champion Agent Selector.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    The 'Final Authority' of the AIDflow pipeline. This model aggregates 
    all evidence from previous selection steps (Prediction, Comparison, and Transition) 
    to officially designate a Champion Agent for production deployment.     
    """

    Model  = AgentSelector
    Option = OptionUI
    
    def custom_confirm(self, fx):
        """Confirm Button of Selected Agent."""
        return self.option_ui.wg.confirm_button.on_click(fx)    

    def static_panel(self, *args, **kwargs):
        """Update Panel with Agents Features Tuning."""
        self.option_ui.panel(self.model.stats)
    
    def dynamic_overview(self, *args, **kwargs):
        """Selected Agent Overivew by Champion vs. Training version."""
        self.model.fx.dashboard._show_without_sig()
    
    @wraps(AgentSelector.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(
            *args, 
            ui_params = self.model_ui.UIParams(
                fit_transform_pipeline      = False,
                dynamic_to_args_kwargs_FX_4 = self.inject_to_params,    
                gui_static_w_fitted_5       = self.static_panel,
                gui_dynamic_w_fitted_6      = self.dynamic_overview,
                custom_control_widget       = self.custom_confirm,
                is_button_last              = True, 
                output_wait                 = True, 
            ),
            **kwargs
        ) 
    
    @wraps(AgentSelector.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> AgentSelectorResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬