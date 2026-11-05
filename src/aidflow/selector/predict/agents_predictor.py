"""
Agent Predictor — Unified Multi-Agent Prediction Interface.

This module implements the prediction adapter stage of the AIDflow
pipeline. It provides a convenient, standardized interface to obtain
probability predictions from all trained agents on the same dataset.

What It Does
------------
Applies all trained agents (GLM and GBM variants) to a target
dataset (typically OOT/Test) in a single call, producing
synchronized predictions. Each agent's output is a DataFrame of
P(class=1) probabilities, indexed identically to the input.

Role in the Pipeline
--------------------
This is the **prediction layer** that feeds into the analytical
stages (Comparator, Transition). It ensures that:

1. All agents predict on the **exact same data** — enabling
   fair side-by-side comparison.
2. Optional **preprocessing** is applied uniformly (raw →
   model-ready) when the input data is not yet encoded.
3. The predictions are **synchronized** — same index, same
   order — so downstream stages (calibration, transition,
   scoring) can directly compare them.

Why a Separate Model?
---------------------
Individual agent models already have a `transform` method.
This model adds value by:

- **Batching**: One call produces predictions from all agents.
- **Consistency**: Guarantees the same data path for all agents
  (same preprocessing, same index alignment).
- **Convenience**: Downstream stages (Comparator, Transition,
  Selector) consume these synchronized predictions directly
  without re-predicting.

Preprocessing
-------------
The `preprocess` parameter controls whether the full encoding
pipeline (feature cleaning → WOE encoding) is applied to the
input data before prediction. Set to `True` when working with
raw data. Set to `False` or `None` when the data is already
in model-ready format (WOE-encoded for GLM agents, raw for GBM
agents).

Dependencies
------------
Requires the following pipeline stages to be completed:
- Agent Training (≥ 1 agent)
"""

from ...deps.data               import  pDF

from ...core.parameters         import  AIGFLOW
from ...core.registry           import  (
                                        FitRegister, 
                                        FitRegOpt,   
                                        RegGet
                                        )
from ...core.base               import  (
                                        FitterBase,
                                        TransformerBase,
                                        ModelProcessorBase,
                                        ModelInput,
                                        ModelResult,
                                        ModelBase,
                                        )
from ...core.factory            import  (
                                        Dataclass,
                                        DictClass
                                        )
from ...core.foundation         import  AigParam
from ...core.io                 import  ModelIO

from .adapter                   import  PredictorAdapter
from .processor                 import  (
                                        AGENTS,
                                        PredictorProcessor,
                                        PredictorParamsBase,
                                        PredictorParamsCfg,
                                        PredictorResult,
                                        PredictorStats
                                        )

MODEL = AIGFLOW.Selection.predictor
FREG  = FitRegister(MODEL)
DREG  = None
DESC  = 'AIG SELECTION STAGE ► AGENTS PREDICTOR ► APPLY TRAINED AGENTS TO PREDICT ON OOT SAMPLE'
DEPS  = (
    AIGFLOW.Training.agent1,
    AIGFLOW.Training.agent2,
    AIGFLOW.Training.agent3,
    AIGFLOW.Training.agent4,
)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG = 'AgentsPredictor'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENTS PREDICTOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class PredictorParamslDefault:
    """Direct assignment → no global flow params."""
    params = Dataclass.schema(PredictorParamsBase)
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
    class AgentsPredictorFitter(PredictorParamsBase):
        
        config: dict|PredictorParamsCfg = PredictorParamsCfg(
            target     = None,
            preprocess = None
        )

        def _build(self, params:PredictorParamsBase, config:dict) -> PredictorProcessor:            
            setup  = dict(params) | config
            return PredictorProcessor(**setup) 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Fitter(FitterBase[PredictorParamsBase]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """     
    
    FREG = FREG
    DREG = DREG
    
    def fit(self, data:pDF): 
        """
        Execute predictions with all agents on the provided data.

        Parameters
        ----------
        data : pDF
            Input data for prediction (raw or model-ready,
            depending on the `preprocess` setting).

        Returns
        -------
        self
        """   
        self.result: PredictorResult = self.model.transform(data)
        self.stats:  PredictorStats  = self.model.get_stats()
        return self
            
    def transfrom(self, data:pDF, agents:tuple = tuple(AGENTS)) -> PredictorResult:
        """
        Predict with selected agents on new data.

        Parameters
        ----------
        data : pDF
            New data for prediction.
        agents : tuple, default=all
            Subset of agents to predict with, e.g. agents=(1,2).

        Returns
        -------
        PredictorResult
            One prediction DataFrame per requested agent.
        """
        return self.model.transform(data, agents = agents)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬        



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENTS PREDICTOR → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    agent1: list|None
    agent2: list|None
    agent3: list|None
    agent4: list|None

@Dataclass.base
class TransResData(PredictorResult):
    """Transfomer data output design."""

class Transformer(TransformerBase[Fitter, PredictorStats, TransResCols]):
    """
    Transformer meta-processing between Fitter-Model.
    
    Handles the post-processing of Fitter results. Its primary role is to standardize 
    the raw output from various fitters into a unified TransResCols, TransResData
    format, enabling seamless transition regardless of the underlying fitter choice.
    """
    
    ModelFitter = Fitter

    def _set_cols(self):
        res = self.fitter.result
        f = lambda x: list(x.columns) if x is not None else None
        return TransResCols(
            agent1 = f(res.agent1),
            agent2 = f(res.agent2),
            agent3 = f(res.agent3),
            agent4 = f(res.agent4),
        )

    def transform(self, data:pDF, agents:tuple = tuple(AGENTS)) -> TransResData: 
        """Model meta-transform."""
        return TransResData(
            **self.fitter.transfrom(data, agents)
        )     
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENTS PREDICTOR → PROCESSOR
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

    Config  = PredictorParamsCfg
    Mapping = PredictorParamslDefault

    ModelTransformer  = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_choice(self, fx_choice:str, inputs:"AgentsPredictorInput"):
        defaults = self.Config(
            target     = AigParam.di(inputs.target, self.setup.target),
            preprocess = None
        )  
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)   
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def get_params(self, fx_choice:str, inputs:"AgentsPredictorInput"):
        agents  = self.get_model_dependecy(DEPS)
        params_ = PredictorParamsBase(
            predictors = PredictorAdapter(agents)
        )        
        params: PredictorParamsBase = self.get_fit_params(fx_choice, inputs.fx_params, params_)
        return params, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENTS PREDICTOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class AgentsPredictorInput(ModelInput):
    """
    Predictor input specification.

    Defines the data to predict on with all trained agents.

    Attributes
    ----------
    data : pDF
        Data to predict on. Must contain the target column and
        all features required by the trained agents.

    Notes
    -----
    **Preprocessing**: If the data is raw (not yet WOE-encoded),
    set the `preprocess` parameter at fit time to apply the
    full encoding pipeline before prediction. If the data is
    already model-ready, no preprocessing is needed.

    **Agent Count**: By default, all registered agents predict.
    Use `transform(data, agents=(1, 2))` to restrict to a
    subset.
    """
    data: pDF 

@Dataclass.base(dynamic_params=True)
class AgentsPredictorResult(ModelResult):
    """
    Predictor output — synchronized predictions per agent.

    The result contains one probability DataFrame per agent,
    all indexed identically to the input data. This
    synchronization enables direct side-by-side comparison in
    downstream stages (Comparator, Transition).

    Attributes
    ----------
    agent1 : pDF
        Agent 1 probability predictions (P(class=1)).
    agent2 : pDF
        Agent 2 probability predictions (P(class=1)).
    agent3 : pDF
        Agent 3 probability predictions (P(class=1)).
    agent4 : pDF
        Agent 4 probability predictions (P(class=1)).

    Notes
    -----
    **Synchronized Index**: All agent DataFrames share the same
    index as the input data. Row `i` in every agent's output
    corresponds to the same observation.

    **None Handling**: If an agent was not trained or not
    requested, its field will be `None`.

    **Downstream Consumption**: These predictions feed directly
    into the Comparator (scoring), Transition (cross-agent
    analysis), and Selector (champion evaluation) stages.
    """
    agent1: pDF
    agent2: pDF
    agent3: pDF
    agent4: pDF
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class AgentsPredictor(Processor, ModelBase):   
    """
    Agent Predictor — Unified Multi-Agent Prediction Interface.

    Provides a standardized, batch interface to obtain
    probability predictions from all trained agents on the same
    dataset. The synchronized output feeds directly into the
    downstream analytical stages (Comparator, Transition,
    Selector).

    What It Does
    ------------
    1. **Predicts** with all (or selected) agents on the
       provided data.
    2. **Synchronizes** all outputs to the same index, enabling
       direct comparison.
    3. **Stores** the predictions for downstream stages to
       consume without re-predicting.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Predict on the OOT sample and store
      the results. The primary entry point in the pipeline flow.
    - **transform(data, agents)**: Predict on any new data
      sample. Optionally restrict to a subset of agents.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **Not Analytical**: This model does not score, rank, or
    compare agents. It simply produces the synchronized
    predictions that the analytical stages (Comparator,
    Transition) consume. 

    **Agent Selection**: By default, all registered agents
    predict. Use `transform(data, agents=(1, 2))` to restrict
    to a subset.

    Examples
    --------
    >>> from aidflow.models import AgentsPredictor
    >>> pred = AgentsPredictor(ctx.setup)
    >>>
    >>> # Predict on OOT data
    >>> pred.fit(AgentsPredictorInput(data = oot_data))
    >>> pred.fitted     # Model result view
    >>> pred.stats      # Model stats  view
    >>>
    >>> # Predict on new data (subset)
    >>> fresh = pred.transform(new_data, agents=(1, 2))
    """

    def _save(self, save_path) -> None:
        features = dict(self.cols)
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_result(self, res: TransResData):
        return AgentsPredictorResult(
            agent1 = lambda: res.agent1, 
            agent2 = lambda: res.agent2, 
            agent3 = lambda: res.agent3, 
            agent4 = lambda: res.agent4, 
        )        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def fit(self, inputs: AgentsPredictorInput, fx_choice: str = REG.AIG, **params):
        """
        Execute predictions with all agents on the provided data.

        Applies all trained agents to the input dataset and
        stores the synchronized predictions. These predictions
        are consumed by downstream stages (Comparator,
        Transition, Selector) for scoring and analysis.

        Parameters
        ----------
        inputs : AgentsPredictorInput
            Container for target data and execution configuration.
        fx_choice : str, optional
            The prediction strategy identifier. Defaults to REG.AIG.

        Returns
        -------
        AgentsPredictor
            The fitted model instance. Predictions access by `.fitted`.
        """
        verify  = len(inputs.data)

        self.choice = self.get_choice(fx_choice, inputs)
        self.params = self.get_params(fx_choice, inputs)
        
        self.processor.fit(
            inputs,
            fx_choice = self.choice,
            params    = self.params, 
            verify    = verify
        )
        self._save(inputs.save_path)
        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────
                
    def transform(self, 
        data:   pDF, 
        agents: tuple = tuple(AGENTS)
    ) -> AgentsPredictorResult:
        """
        Predict with one or more agents on a new dataset.

        Produces synchronized probability predictions for the
        requested agents on any data sample. Use this to:
        - Score a new OOT window.
        - Predict on a subset of agents (e.g., the top 2).
        - Generate predictions for a different segment.

        Parameters
        ----------
        data : pDF
            The dataset to be scored (must contain the features
            required by the agents).
        agents : tuple, optional
            Subset of agents to engage. Defaults to all trained
            agents.

        Returns
        -------
        AgentsPredictorResult
            A unified result object containing synchronized
            predictions for all requested agents.

        Notes
        -----
        **No State Change**: This method does not modify the
        fitted state. It produces a fresh prediction on the new
        data. The original fit results remain accessible via
        `.fitted`.
        """     
        res = self.processor.transform(data, agents)
        return self._get_result(res)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬