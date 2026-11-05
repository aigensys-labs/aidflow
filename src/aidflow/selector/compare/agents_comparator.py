"""
Agent Comparator — Multi-Dimensional Model Selection & Ranking.

This module implements the model comparison stage of the AIDflow
pipeline. It provides an objective, transparent, and reproducible
scoring system that ranks all trained agents across multiple
orthogonal dimensions to identify the optimal candidate for
production deployment.

What It Answers
---------------
- **Which agent is best?** The agent with the highest weighted
  composite score across all dimensions.
- **Why is it best?** The per-metric breakdown shows which
  dimensions it leads (performance, calibration, simplicity,
  cross-audit).
- **What are the trade-offs?** The comparison table shows raw
  values — an agent may lead on Gini but lag on calibration or
  simplicity.

Scoring Dimensions
------------------
Agents are scored across four orthogonal dimensions:

1. **Performance (PE)**: Predictive power + operational simplicity.
   - Gini, IV, KS: Separation quality (higher = better).
   - Feature Exposure: Effective degrees of freedom (lower =
     simpler, more stable).
   - Max Feature Impact: Single-point-of-failure risk (lower =
     more robust).

2. **Calibration (CA)**: Probability reliability.
   - Fraction of bins where predicted rate significantly matches
     observed rate (higher = more reliable probabilities).

3. **Cross-Audit (AE)**: Inter-agent information analysis.
   - Audited (underestimated risk): How much risk this agent
     missed, as detected by other agents.
   - Overestimated risk: How much this agent overstates risk
     relative to other agents.

Scoring Methodology
-------------------
- **Standardization**: Leader-Anchored Proportional Scaling —
  the best agent gets 1.0 on each metric; others are scaled
  proportionally. Direction-aware (higher/lower is better).
- **Weighting**: Each metric has a weight (default: equal).
  The weighted mean produces the final composite score.
- **Ranking**: Agents ordered by descending composite score.

Role in the Pipeline
--------------------
This is a **diagnostic/analytical** model — it does not train
or select anything. It provides the **ranking evidence** that
informs the champion selection decision (the Selector stage).

The comparison produces:
- A ranked list (best → worst) with the recommended agent first.
- Per-metric breakdown for transparency and auditability.
- Lift tables and calibration curves for visual verification.

Complementary to Transition
---------------------------
- **Comparator**: Ranks agents by quality (performance, stability,
  calibration). Answers "Which is best?"
- **Transition**: Analyzes how agents relate to each other
  (agreement, disagreement, blind spots). Answers "How do they
  differ?"

Together, they provide a complete picture: the Comparator tells
you *who* to pick; the Transition tells you *why* they differ
and what you gain/lose by picking one over another.

Dependencies
------------
Requires the following pipeline stages to be completed:
- Agent Training (≥ 2 agents — for cross-audit dimensions)
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
                                        ComparatorProcessor,
                                        ComparatorParamsBase,
                                        ComparatorParamsCfg,
                                        ComparatorResult,
                                        ComparatorStats
                                        )

MODEL = AIGFLOW.Selection.comparator
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG SELECTION STAGE ► AGENTS COMPARISON ► COMPARE TRAINED AGENTS ON OOT TEST SAMPLE'
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
    AIG = 'AngentsComparator'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → COMPARATOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class ComparatorParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(ComparatorParamsBase)
    setup  = {
        PRM.SELECT.select_bins.param     : (params.proba_bins, 'Probability Discretization' ),
        PRM.SELECT.select_alpha.param    : (params.alpha,      'Cross-Calibration alpha'    ),
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
    class AgentsComparatorFitter(ComparatorParamsBase):
        
        config: dict|ComparatorParamsCfg = ComparatorParamsCfg(
            predictors = None,
            target     = None
        )

        def _build(self, params:ComparatorParamsBase, config:dict) -> ComparatorProcessor:            
            setup  = dict(params) | config
            return ComparatorProcessor(**setup) 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Fitter(FitterBase[ComparatorParamsBase]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """         
    
    FREG = FREG
    DREG = DREG
    
    def fit(self, data:pDF): 
        """
        Execute the full multi-agent comparison and scoring.

        Parameters
        ----------
        data : pDF
            OOT evaluation data (must contain the target column
            and all agent features).

        Returns
        -------
        self
        """    
        self.result: ComparatorResult = self.model.process(data)
        self.stats:  ComparatorStats  = self.model.get_stats()
        self._init_dash()
        return self
    
    def transfrom(self, data:pDF, agents:tuple=tuple(AGENTS)) -> ComparatorResult:
        """
        Execute the comparison on new data (re-evaluation).

        Parameters
        ----------
        data : pDF
            Evaluation data for the comparison.
        agents : tuple, default=all
            Subset of agents to include (e.g., `(1, 2)`).

        Returns
        -------
        ComparatorResult
            Scoring, comparison, lift, and calibration.
        """
        return self.model.process(data, agents = agents)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬        



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → COMPARATOR → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    pass

@Dataclass.base
class TransResData(ComparatorResult):
    """Transfomer data output design."""

class Transformer(TransformerBase[Fitter, ComparatorStats, TransResCols]):
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
# SELECTION → AGENTS COMPARATOR → PROCESSOR
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

    Config  = ComparatorParamsCfg
    Mapping = ComparatorParamsMapping

    ModelTransformer  = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_choice(self, fx_choice:str, inputs:"AgentsComparatorInput"):
        agents   = self.get_model_dependecy(DEPS)
        defaults = self.Config(
            predictors = PredictorAdapter(agents),
            target     = AigParam.di(inputs.target, self.setup.target)
        )  
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)   
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def get_params(self, fx_choice:str, inputs:"AgentsComparatorInput", params:dict):
        params: ComparatorParamsBase = self.get_fit_params(fx_choice, inputs.fx_params, params)  
        return params, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → AGENTS COMPARATOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class AgentsComparatorInput(ModelInput):
    """
    Comparator input specification.

    Defines the evaluation data required to execute the
    multi-agent comparison and scoring.

    Attributes
    ----------
    data : pDF
        OOT evaluation data. Must contain the target column and
        all features required by the trained agents. This is the
        primary evidence set — the comparator scores all agents
        on this data.

    Notes
    -----
    **No Training Data Needed**: The comparison is purely a
    test-time diagnostic. It only requires one data sample
    where all agents can predict.

    **Agent Count**: Requires ≥ 2 agents for the cross-audit
    dimensions (audited/auditor). With a single agent, only
    performance and calibration are scored.

    **Binning**: Agents are binned into equal-population
    quantile bins (default: 10) for calibration and
    cross-transition analysis.
    """
    data: pDF 

@Dataclass.base
class AgentsComparatorResult(ModelResult):
    """
    Comparator output — the model selection decision package.

    The final result of the multi-agent comparison. Contains the
    standardized scoring tables, raw metric values, lift tables,
    and calibration assessments needed to make an informed
    deployment decision.

    Attributes
    ----------
    overview : pDF
        Raw (unstandardized) metric values per agent with
        direction indicators (Max/Min). This is the "why"
        view — it shows the actual numbers behind each score.

    scoring : pDF
        Standardized scores (best agent = 1.0 per metric)
        with weights and the total weighted composite score.
        The row with the highest total is the recommended agent.
        This is the "decision" view.

    lift : dict
        Per-agent lift tables. Each table shows the probability
        bins with observed rates, counts, weights, and lift
        values. Useful for visualizing separation quality across
        the full probability range.

    calibration : dict
        Per-agent calibration assessment. Each table shows
        predicted vs. observed rates per bin with significance
        flags. Identifies agents that systematically
        under- or over-estimate probability.

    Notes
    -----
    **Reading the Scoring Table**:
    - Each column is an agent.
    - Each row is a metric (Gini, IV, KS, Calibration, Feature
      Exposure, Max Impact, Audited, Overestimated).
    - Values are standardized: 1.0 = best agent on that metric,
      0.0 = worst.
    - The bottom row (`TOTAL_WEIGHTED_SCORE`) is the final
      ranking criterion.

    **Reading the Overview Table**:
    - Same structure as scoring, but with raw values.
    - The `Direction` column shows whether higher (Max) or
      lower (Min) is better for that metric.
    - Use this to understand *why* an agent scored where it did.

    **Ranking Access**: The ranked agent list (best first) is
    available via `.stats.ordered` (full `ComparatorStats`).
    """
    overview:    pDF
    scoring:     pDF
    lift:        dict
    calibration: dict
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class AgentsComparator(Processor, ModelBase):   
    """
    Agent Comparator — Multi-Dimensional Model Selection & Ranking.

    The decision engine of the AIDflow pipeline. Consumes
    predictions from all trained agents and computes a
    multi-dimensional performance matrix to rank them and
    identify the optimal champion candidate.

    What It Does
    ------------
    1. **Predicts** with all agents on the provided data.
    2. **Scores** each agent across four dimensions:
       - Performance (Gini, IV, KS, feature exposure, max impact).
       - Calibration (fraction of calibrated bins).
       - Cross-Audit (audited/overestimated risk).
    3. **Standardizes** all metrics to a 0–1 scale (best = 1.0).
    4. **Weights** and aggregates into a composite score.
    5. **Ranks** agents from best to worst.
    6. **Produces** lift tables and calibration curves for
       visual verification.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full comparison + scoring on the OOT
      sample. Produces the complete decision package.
    - **transform(data, agents)**: Re-evaluate on any new
      data sample. Optionally restrict to a subset of agents.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **No Training**: This model does not train or select anything.
    It is purely analytical — it provides the ranking evidence
    that informs the champion selection decision.

    **Objective & Reproducible**: The same data + agents +
    parameters will always produce the same ranking. No
    randomness in the scoring.

    **Complementary to Transition**: The Comparator ranks agents
    by quality ("which is best?"). The Transition analyzes how
    they relate to each other ("how do they differ?"). Together,
    they provide the complete picture for the selection decision.

    **Weighted Score**: By default, all metrics are weighted
    equally. Override `metric_weight` to prioritize specific
    dimensions (e.g., double the weight of calibration if
    probability reliability is critical).

    Examples
    --------
    >>> from aidflow.models import AgentsComparator
    >>> cmp = AgentsComparator(ctx.setup)
    >>>
    >>> # Full comparison on OOT data
    >>> cmp.fit(AgentsComparatorInput(data = oot_data))
    >>> cmp.fitted  # Model result view
    >>> cmp.stats   # Model stats  view
    >>>
    >>> # Re-evaluate on a new sample
    >>> fresh = cmp.transform(next_month_oot)
    >>>
    >>> # Compare a subset
    >>> pair = cmp.transform(oot_data, agents=(1, 3))
    """

    def _save(self, save_path) -> None:
        ModelIO(self.setup).save(MODEL, self, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_result(self, res: TransResData):
        return AgentsComparatorResult(
            overview    = res.comparison,
            scoring     = res.scoring,
            lift        = res.lift,
            calibration = res.calibration,
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs: AgentsComparatorInput, fx_choice: str = REG.AIG, **params):
        """
        Execute the full multi-agent comparison and scoring.

        Predicts with all agents, computes per-metric scores
        (performance, calibration, cross-audit), standardizes
        to a 0–1 scale, applies weights, and ranks agents by
        composite score.

        Parameters
        ----------
        inputs : AgentsComparatorInput
            Container for target evaluation data and configuration.
        fx_choice : str, optional
            The comparison strategy identifier. Defaults to REG.AIG.
        **params : dict
            Runtime parameter overrides, passed to the internal fitter
            (select_alpha, select_bins).

        Returns
        -------
        AgentsComparator
            The fitted model instance containing the calculated performance matrices.

        Notes
        -----
        **Agent Ranking**: The ranked list is available via
        `.stats.ordered` — the first element is the
        recommended agent.
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
    ) -> AgentsComparatorResult:
        """
        Re-evaluate the full comparison on a new dataset.

        Runs the complete multi-agent scoring pipeline on any
        data sample. Use this to:
        - Monitor agent rankings on subsequent OOT windows.
        - Compare a subset of agents (e.g., the top 2 from
          the initial ranking).
        - Generate a fresh scoring report for a different
          segment or time period.
        - Perform "What-If" analysis: does the ranking hold
          on a different sample?

        Parameters
        ----------
        data : pDF
            The dataset for performance evaluation (must contain
            target + features).
        agents : tuple, default=all
            Subset of agents to include (e.g., `(1, 2)` for
            a head-to-head comparison, or `(1, 3, 4)` for
            multiple agents).

        Returns
        -------
        AgentsComparatorResult
            Full scoring package (overview, scoring, lift,
            calibration) for the requested agents.

        Notes
        -----
        **No State Change**: This method does not modify the
        fitted state. It produces a fresh analysis on the new
        data. The original fit results remain accessible via
        `.result` and `.stats`.

        **Head-to-Head**: Use `agents=(i, j)` for a focused
        comparison between two specific agents.
        """ 
        res = self.processor.transform(data, agents)    
        return self._get_result(res)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → COMPARATOR → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI(OptionUIBase):    
    title   = 'Comparison Panel'.upper()
    split   = 1
    width   = 150
    Mapping = ComparatorParamsMapping
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → COMPARATOR → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class AgentsComparatorUI(ModelUIBase):
    """
    Agents Comparator Model.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    The 'Decision Engine' of the Selection Phase. This model consumes 
    synchronized predictions from all trained agents and computes a multi-dimensional 
    performance matrix (Scoring, Lift, Calibration). It transforms raw probability 
    outputs into actionable business metrics to identify the champion candidate.      
    """

    Model  = AgentsComparator
    Option = OptionUI
        
    @wraps(AgentsComparator.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(
            fit_transform_pipeline = False
        ), **kwargs)  
    
    @wraps(AgentsComparator.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> AgentsComparatorResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬