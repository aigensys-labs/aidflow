"""
Agent Transition — Cross-Agent Agreement & Disagreement Analysis.

This module implements the cross-agent transition analysis stage
of the AIDflow pipeline. It provides a detailed, evidence-based
view of how trained agents' predictions relate to each other on
OOT/test data.

What It Answers
---------------
- **Where do agents agree?** Diagonal cells of the transition
  matrix show where both agents assign similar risk levels.
- **Where do they disagree?** Off-diagonal cells reveal
  disagreement — one agent says "low risk" while the other says
  "high risk."
- **Is the disagreement meaningful?** Significance testing
  determines whether the target rate in a cross-bin cell is
  statistically different from the marginal rate. If yes, one
  agent is providing **additional information** the other missed.

Core Concept: Transition Matrix
-------------------------------
For each agent pair (A_i, A_j), the transition matrix is a
cross-tabulation:

- **Rows** (Audited): Agent A_i's probability quantile bins.
- **Columns** (Auditor): Agent A_j's probability quantile bins.
- **Cell value**: E(T | A_i_bin, A_j_bin) — the observed target
  rate within that cross-bin cell.

The significance test asks: within A_i's bin, does A_j's
assignment to a specific sub-bin significantly change the target
rate? If yes, A_j is "auditing" A_i — finding structure that
A_i's binning alone cannot resolve.

Audited vs. Auditor Scores
--------------------------
- **Audited score** (per agent): The mean significant divergence
  found *by other agents* within this agent's bins. Higher =
  more "blind spots" — other agents can decompose this agent's
  bins into meaningfully different sub-segments.

- **Auditor score** (per agent): The mean information this agent
  reveals *about other agents'* bins. Higher = more
  discriminative breadth — this agent adds signal across many
  of the other agents' bins.

Interpretation
--------------
- **High off-diagonal significance**: Agents capture different
  information (complementary). This is desirable — the ensemble
  is diverse.
- **Low off-diagonal significance**: Agents capture similar
  information (redundant). One could potentially be dropped.
- **Agent A audits B but not vice versa**: A provides
  information B cannot. A is "stronger" in that dimension.
- **High audited score**: The agent has blind spots that other
  agents can resolve.
- **High auditor score**: The agent is broadly informative
  across the ensemble.

Role in the Pipeline
--------------------
This is a **diagnostic/analytical** model — it does not train
or select anything. It provides the evidence needed to:
- Understand ensemble diversity (are agents redundant?).
- Identify which agent's "blind spots" matter most.
- Support the champion selection decision (the Selector stage).
- Monitor cross-agent stability on new OOT windows.

Dependencies
------------
Requires the following pipeline stages to be completed:
- Agent Training (≥ 2 agents — for at least one pair)
"""

from ...deps.common             import  wraps
from ...deps.data               import  pDF

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

from ..predict.adapter          import  PredictorAdapter

from .processor                 import  (
                                        AGENTS,
                                        TransitionProcessor,
                                        TransitionParamsBase,
                                        TransitionParamsCfg,
                                        TransitionResult,
                                        TransitionStats
                                        )

MODEL = AIGFLOW.Selection.transition
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG SELECTION STAGE ► AGENTS TRANSITION ► TRANSITION OF TRAINED AGENTS ON OOT TEST SAMPLE'
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
    AIG = 'AngentsTransition'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → TRANSITION
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class TransitionParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(TransitionParamsBase)
    setup  = {
        PRM.SELECT.select_bins.param     : (params.proba_bins, 'Probability Discretization' ),
        PRM.SELECT.select_alpha.param    : (params.alpha,      'Cross-Transition alpha'    ),
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
    class AgentsTransitionFitter(TransitionParamsBase):
        
        config: dict|TransitionParamsCfg = TransitionParamsCfg(
            predictors = None,
            target     = None
        )

        def _build(self, params:TransitionParamsBase, config:dict) -> TransitionProcessor:            
            setup  = dict(params) | config
            return TransitionProcessor(**setup) 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Fitter(FitterBase[TransitionParamsBase]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """        
    
    FREG = FREG
    DREG = DREG
    
    def fit(self, data:pDF): 
        """
        Execute the transition analysis on the provided data.

        Parameters
        ----------
        data : pDF
            OOT evaluation data (must contain the target column
            and all agent features).

        Returns
        -------
        self
            Fitted fitter with `result` and `stats` set.
        """     
        self.result: TransitionResult = self.model.process(data)
        self.stats:  TransitionStats  = self.model.get_stats()
        self._init_dash()
        return self
    
    def transfrom(self, data:pDF, agents:tuple=tuple(AGENTS)) -> TransitionResult:
        """
        Execute the transition analysis on new data (re-evaluation).

        Parameters
        ----------
        data : pDF
            Evaluation data for the transition analysis.
        agents : tuple, default=all
            Subset of agents to include (e.g., `(1, 2)`).

        Returns
        -------
        TransitionResult
            Transition vectors and summary scores.
        """
        return self.model.process(data, agents = agents)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬        



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → TRANSITION → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    pass

@Dataclass.base
class TransResData(TransitionResult):
    """Transfomer data output design."""

class Transformer(TransformerBase[Fitter, TransitionStats, TransResCols]):
    """
    Transformer meta-processing between Fitter-Model.
    
    Handles the post-processing of Fitter results. Its primary role is to standardize 
    the raw output from various fitters into a unified TransResCols, TransResData
    format, enabling seamless transition regardless of the underlying fitter choice.
    """
    
    ModelFitter = Fitter
           
    def transform(self, data:pDF, agents:tuple = tuple(AGENTS)) -> TransResData: 
        """Model meta-transform."""
        return TransResData(
            **self.fitter.transfrom(data, agents)
        )   
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENTS TRANSITION → PROCESSOR
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

    Config  = TransitionParamsCfg
    Mapping = TransitionParamsMapping

    ModelTransformer  = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_choice(self, fx_choice:str, inputs:"AgentsTransitionInput"):
        agents   = self.get_model_dependecy(DEPS, is_req=False)
        defaults = self.Config(
            predictors = PredictorAdapter(agents),
            target     = AigParam.di(inputs.target, self.setup.target)
        )  
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)   
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def get_params(self, fx_choice:str, inputs:"AgentsTransitionInput", params:dict):
        params: TransitionParamsBase = self.get_fit_params(fx_choice, inputs.fx_params, params)  
        return params, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENTS TRANSITION
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class AgentsTransitionInput(ModelInput):
    """
    Transition analysis input specification.

    Defines the evaluation data required to execute the cross-agent
    transition analysis.

    Attributes
    ----------
    data : pDF
        OOT evaluation data. Must contain the target column and
        all features required by the trained agents. This is the
        primary evidence set — the transition analysis evaluates
        all agents on this data to quantify agreement and
        disagreement.

    Notes
    -----
    **Agent Count**: Requires ≥ 2 agents (for at least one pair).
    The default is all registered agents.

    **Binning**: Agents are binned into equal-population quantile
    bins (default: 10). Both agents in a pair use the same bin
    count for comparability.
    """
    data: pDF 

@Dataclass.base
class AgentsTransitionResult(ModelResult):
    """
    Transition analysis output — cross-agent relationship evidence.

    The final result of the transition analysis. Contains the
    per-pair transition vectors ready for plotting and inspection.

    Attributes
    ----------
    vectors : dict[str, TransitionVectors]
        Per-agent-pair transition data. Keys are pair labels
        (e.g., `'1 vs. 2'`). Each value contains:

        - *mean*: Cross-bin target rates (E(T | bin_i, bin_j))
          with row/column totals.
        - *sig_mean*: Significance flags per cell (NaN = not
          significant, value = significant difference from the
          marginal rate).
        - *count*: Observation counts per cross-bin cell.

    Notes
    -----
    **Decision Support**: The vectors provide the raw evidence
    for understanding ensemble diversity:

    - *Diagonal significance*: Where agents agree AND the cell
      is significant — strong consensus on those samples.
    - *Off-diagonal significance*: Where agents disagree
      meaningfully — the "contested" samples. The agent with
      more significant off-diagonal cells is the stronger
      auditor.
    - *Cell counts*: Low-count cells are unreliable — the
      significance test may flag them but the evidence is weak.
    """
    vectors: dict
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class AgentsTransition(Processor, ModelBase):   
    """
    Agent Transition — Cross-Agent Agreement & Disagreement Analysis.

    The consistency analyst of the AIDflow pipeline. Evaluates
    how trained agents' predictions relate to each other on OOT
    data: where they agree, where they disagree, and whether
    that disagreement is statistically meaningful.

    What It Does
    ------------
    1. **Predicts** with all agents on the provided data.
    2. **Bins** each agent's probabilities into quantile bins
       (equal-population, default 10 bins).
    3. **Cross-tabulates** each agent pair: the target rate
       within each (bin_i, bin_j) cell.
    4. **Tests significance**: Is the cross-bin rate
       significantly different from the marginal rate? This
       identifies where one agent adds information the other
       cannot resolve.
    5. **Summarizes**: Aggregated audited/auditor scores per
       agent — quantifying each agent's "blind spots" and
       discriminative breadth.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full analysis on the OOT sample.
      Produces the complete transition package.
    - **transform(data, agents)**: Re-evaluate on any new
      data sample (e.g., a subsequent OOT window). Optionally
      restrict to a subset of agents.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for cross-bin difference testing.
    proba_bins : int, default=10
        Number of quantile bins for probability discretization.

    Notes
    -----
    **Complementary to Comparison**: The Comparator ranks agents
    by performance. The Transition reveals *how* they differ —
    which agents are redundant, which are complementary, and
    where each has blind spots.

    Examples
    --------
    >>> from aidflow.models import AgentsTransition
    >>> tr = AgentsTransition(ctx.setup)
    >>>
    >>> # Full analysis on OOT data
    >>> tr.fit(AgentsTransitionInput(data = oot_data))
    >>> tr.fitted       # View to model result
    >>> tr.stats        # View to model stats
    >>>
    >>> # Re-evaluate on a new sample
    >>> fresh = tr.transform(new_oot_data, agents=(1, 3))
    >>> pair.vectors    # Significance
    """


    def _save(self, save_path) -> None:
        ModelIO(self.setup).save(MODEL, self, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_result(self, res: TransResData):
        return AgentsTransitionResult(
            vectors = res.vectors
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
            
    def fit(self, inputs: AgentsTransitionInput, fx_choice: str = REG.AIG, **params):
        """
        Execute the full cross-agent transition analysis.

        Predicts with all agents, computes pairwise cross-
        tabulations, tests significance, and assembles the
        complete transition stats.

        Parameters
        ----------
        inputs : AgentsTransitionInput
            Container for evaluation data and execution configuration.
        fx_choice : str, optional
            The transition strategy identifier. Defaults to REG.AIG.
        **params : dict
            Runtime parameter overrides, passed to the internal fitter
            (select_alpha, select_bins).

        Returns
        -------
        AgentsTransition
            The fitted model instance containing calculated transition vectors.

        Notes
        -----
        **Agent Pairs**: By default, all registered agents are
        included. Use `transform(data, agents=(1, 3))` to
        restrict to specific pairs.
        """
        verify  = len(inputs.data)
        
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
    
    def transform(self, 
        data:    pDF, 
        agents:  tuple = tuple(AGENTS) 
    ) -> AgentsTransitionResult:
        """
        Re-evaluate the transition analysis on a new dataset.

        Runs the full cross-agent transition analysis on any
        data sample. Use this to:
        - Monitor cross-agent stability on subsequent OOT windows.
        - Analyze a subset of agents (e.g., the top 2 from the
          comparison).
        - Generate a fresh transition report for a different
          segment or time period.

        Parameters
        ----------
        data : pDF
            The dataset for transition analysis (must contain
            target + features).
        agents : tuple, default=all
            Subset of agents to include (e.g., `(1, 2)` for
            a single pair, or `(1, 3, 4)` for multiple pairs).

        Returns
        -------
        AgentsTransitionResult
            Transition vectors for the requested agent pairs.

        Notes
        -----
        **No State Change**: This method does not modify the
        fitted state. It produces a fresh analysis on the new
        data. The original fit results remain accessible via
        `.result` and `.stats`.
        """ 
        res = self.processor.transform(data, agents)    
        return self._get_result(res)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → TRANSITION → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI(OptionUIBase):    
    title   = 'Transition Panel'.upper()
    split   = 1
    width   = 150
    Mapping = TransitionParamsMapping
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → TRANSITION → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class AgentsTransitionUI(ModelUIBase):
    """
    Agents Transition & Divergence Model.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    The 'Consistency Analyst' of the Selection Phase. This model evaluates 
    the agreement between different agents by calculating transition matrices. 
    It identifies where agents converge on a decision and, more importantly, 
    where they diverge—highlighting samples that are "contentious" across 
    different mathematical strategies.       
    """

    Model  = AgentsTransition
    Option = OptionUI
        
    @wraps(AgentsTransition.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(
            fit_transform_pipeline = False
        ), **kwargs)  
    
    @wraps(AgentsTransition.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> AgentsTransitionResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬