"""
Multi-Agent Prediction Adapter.

This module implements the unified prediction interface for the AIDflow
pipeline, providing a standardized way to obtain probability predictions
from all trained agents (GLM and GBM variants) on the same dataset.

Purpose
-------
The prediction adapter solves three problems:

1. **Unified interface**: Regardless of how many agents were trained
   (GLM, GBM, or both), predictions are accessed through a single
   ``transform`` call that returns a consistent ``PredictorResult``.

2. **Comparative evaluation**: All agents predict on the same data,
   enabling side-by-side comparison of their probability distributions,
   separation metrics, and calibration.

3. **Optional preprocessing**: If agents were trained on WOE-encoded
   features, the adapter can optionally apply the full encoding pipeline
   (cleaning → discretization → WOE) to new data before prediction.
   This ensures new data is in the same representation as training data.

Agent Naming
------------
Agents are referenced by their registry key (e.g., '1', '2', '3', '4')
and mapped to result fields (``agent1``, ``agent2``, etc.). The mapping
is safety-checked to prevent naming mismatches between the input
specification and the result structure.

Usage Flow
----------
1. Train agents (GLM/GBM) via their respective ``fit`` methods.
2. Instantiate ``PredictorProcessor`` with the trained predictors.
3. Call ``transform(data)`` to get all predictions in one call.
4. Inspect ``stats`` for comparative metrics.
"""

from ...deps.data                   import  pDF                                           
from ...core.base                   import  (
                                            FitterDataBase,
                                            FitterModelBase
                                            )
from ...core.foundation             import  AigParam
from ...core.factory                import  Dataclass
from ..base.processor               import  (
                                            AGENTS, 
                                            SelectorInputBase
                                            )
from .adapter                       import  PredictorAdapter                                            


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAININING → TUNING → AGENTS PREDICTION ON OOT
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class PredictorParamsBase:
    """
    Prediction adapter configuration — trained agents.

    Attributes
    ----------
    predictors : PredictorAdapter
        The adapter holding all trained agent instances. Must be
        populated before calling ``transform``.
    """    
    predictors: PredictorAdapter = None

@Dataclass.base(update_specs=dict(slots=False))
class PredictorParamsCfg:
    """
    Prediction adapter configuration — pipeline settings.

    Attributes
    ----------
    target : str
        Target column name in the prediction data. Used for
        computing comparative statistics (e.g., AUC, mean
        probability by class).
    preprocess : bool | None, default=None
        If True, apply the full trained encoding pipeline
        (feature cleaning → discretization → WOE) to the input
        data before prediction. If None or False, the input
        data is assumed to already be in the model-ready format.
    """    
    target:     str
    preprocess: bool|None = None

@Dataclass.base
class PredictorParams(PredictorParamsBase, PredictorParamsCfg):
    """Woker params input design."""

@Dataclass.base
class PredictorInput:
    """
    Prediction input specification.

    Attributes
    ----------
    data : pDF
        Data to predict on (raw or model-ready, depending on
        ``preprocess`` setting).
    agents : tuple | list | None
        Subset of agents to predict with. If None, all registered
        agents are used.
    """
    data:   pDF 
    agents: tuple|list|None = tuple(AGENTS)

@Dataclass.base
class PredictorResult:
    """
    Prediction output — one probability DataFrame per agent.

    Each field contains a DataFrame with the predicted probability
    (column ``Prediction``) indexed identically to the input data.
    Agents not requested (or not trained) will be ``None``.

    Attributes
    ----------
    agent1 : pDF | None
        Agent 1 predictions.
    agent2 : pDF | None
        Agent 2 predictions.
    agent3 : pDF | None
        Agent 3 predictions.
    agent4 : pDF | None
        Agent 4 predictions.
    """    
    agent1: pDF|None = None
    agent2: pDF|None = None
    agent3: pDF|None = None
    agent4: pDF|None = None

@Dataclass.base
class PredictorStats:
    """
    Comparative statistics per agent.

    Each field contains a ``describe()`` summary of the target
    distribution alongside that agent's predictions — useful for
    quick calibration and separation checks.

    Attributes
    ----------
    agent1 : pDF
        Agent 1 stats (target + prediction describe).
    agent2 : pDF
        Agent 2 stats.
    agent3 : pDF
        Agent 3 stats.
    agent4 : pDF
        Agent 4 stats.
    """    
    agent1: pDF
    agent2: pDF
    agent3: pDF
    agent4: pDF
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Fitter(FitterDataBase, SelectorInputBase[PredictorInput]):
    """
    Prediction Worker — unified multi-agent prediction execution.

    Resolves the requested agents from the predictor adapter,
    executes predictions, and assembles the standardized result
    with comparative statistics.

    Safety
    ------
    Validates that agent names in the input map correctly to
    result field names, preventing silent mismatches.

    Parameters
    ----------
    params : PredictorParams
        Adapter configuration (predictors, target, preprocess).
    """
    
    SelectionInput : type = PredictorInput
    agents_condition: int = 1

    def __init__(self, params: PredictorParams):
        self.params = params
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def predict(self, agent:str, data:pDF) -> pDF:
        """
        Obtain a probability prediction from a single trained agent.

        Parameters
        ----------
        agent : str
            Agent registry key (e.g., '1', '2').
        data : pDF
            Data to predict on. If ``preprocess`` is enabled,
            the full encoding pipeline is applied first.

        Returns
        -------
        pDF
            Predicted probability (single column), indexed
            identically to the input.
        """
        pa = self.params.predictors.get_agent(agent)
        return pa.predict(data, preprocess = self.params.preprocess)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_agents(self, inputs: PredictorInput) -> dict:
        """
        Resolve requested agents to result field names.

        Validates the mapping between agent registry keys and
        ``PredictorResult`` field names to prevent silent
        mismatches (e.g., agent '1' → field 'agent1').

        Parameters
        ----------
        inputs : PredictorInput
            Input specification with the agent list.

        Returns
        -------
        dict
            Mapping of {agent_key: result_field_name}.

        Raises
        ------
        ValueError
            If an agent key does not map to exactly one result
            field (naming mismatch).
        """
        super()._set_agents(inputs)
                
        r = Dataclass.get_names(PredictorResult)
        
        def n(a:str) -> str:
            v = [x for x in r if x[-1] == str(a)]
            if len(v) == 1:
                return v[0]
            raise ValueError('AIG SAFETY ► Unexpected agent name mismatch!')

        return {a: n(a) for a in self.agents}
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def transform(self, inputs: PredictorInput | pDF, **kwargs) -> PredictorResult:
        """
        Predict with all requested agents and assemble the result.

        For each agent:
        1. Obtain the probability prediction.
        2. Compute descriptive statistics (target + prediction).

        The result contains one prediction DataFrame per agent,
        and the stats are stored in ``self.stats``.

        Parameters
        ----------
        inputs : PredictorInput | pDF
            Either a full input specification or a raw DataFrame
            (in which case all agents are used).
        kwargs : dict
            Additional keyword arguments forwarded to input
            resolution.

        Returns
        -------
        PredictorResult
            One prediction DataFrame per requested agent.
        """
        inputs_ = self.get_inputs(inputs, kwargs)
        agents  = self._get_agents(inputs_)

        pred = {}
        for a, r in agents.items():
            pred[r] = self.predict(AGENTS[a], inputs_.data)

        t:pDF = inputs_.data[[self.params.target]]

        self.stats = PredictorStats(**{
            r: t.assign(**{'Prediction': pred[r]}).describe() 
            for r in agents.values()
        })

        return PredictorResult(**pred)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PredictorProcessor(FitterModelBase[Fitter]):
    """
    Multi-Agent Prediction Adapter.

    Provides a standardized interface to obtain probability predictions
    from all trained agents on the same dataset, enabling comparative
    model evaluation.

    Parameters
    ----------
    predictors : PredictorAdapter
        Adapter holding trained agent instances.
    target : str
        Target column name (for stats computation).
    preprocess : bool | None, default=None
        If True, apply the full encoding pipeline to input data
        before prediction (useful when input is raw, unencoded data).

    Notes
    -----
    **Optional Preprocessing**: If agents were trained on WOE-encoded
    features and you want to predict on raw data, set ``preprocess=True``.
    This triggers the full chain: feature cleaning → numerical
    discretization → WOE encoding → prediction.

    **Comparative Stats**: After ``transform``, access
    ``fitter.stats`` to get a ``describe()`` summary of each
    agent's predictions alongside the target — useful for quick
    calibration and separation checks.

    **Agent Selection**: By default, all registered agents predict.
    Specify a subset via the ``agents`` parameter in ``transform``.

    Examples
    --------
    >>> from aidflow.fitters import PredictorProcessor
    >>> pred = PredictorProcessor(
    ...     predictors=my_adapter,
    ...     target='is_churn',
    ...     preprocess=True,  # apply full encoding to raw data
    ... )
    >>> result = pred.transform(raw_oot_data)
    >>> result.agent1  # GLM predictions  # doctest: +SKIP
    >>> result.agent2  # GBM predictions  # doctest: +SKIP
    >>> pred.fitter.stats  # comparative describe() per agent  # doctest: +SKIP
    """
    
    def __init__(self, *args, params: dict|PredictorParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(PredictorParams, params, args, kwargs)
        self._fitter = Fitter(self._params)

    def predict(self, agent:str, data:pDF) -> pDF:
        """
        Predict with a single agent.

        Parameters
        ----------
        agent : str
            Agent registry key (e.g., '1' for GLM, '2' for GBM).
        data : pDF
            Data to predict on.

        Returns
        -------
        pDF
            Predicted probability, indexed identically to input.
        """
        return self._fitter.predict(agent, data)
    
    def transform(self, inputs: PredictorInput | pDF, **kwargs) -> PredictorResult:
        """
        Predict with all (or selected) agents.

        Parameters
        ----------
        inputs : PredictorInput | pDF
            Input data or full input specification.
        kwargs : dict
            Additional arguments (e.g., ``agents=['1', '2']``).

        Returns
        -------
        PredictorResult
            One prediction DataFrame per requested agent.
        """
        return self._fitter.transform(inputs, **kwargs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬