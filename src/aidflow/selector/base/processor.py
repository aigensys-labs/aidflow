"""
Selector Base Processor — Agent Selection Phase Input Handling.

This module provides the base class infrastructure for the agent
selection phase of the Champion workflow. It defines the contract
that all selector Fitters must fulfill when processing their inputs,
specifically around agent validation and minimum-count enforcement.
"""

from ...deps.data                           import  pDF
from ...deps.types                          import  (
                                                    Generic, 
                                                    TypeVar
                                                    )
from ...core.structure                      import  AIGFLOW
from ...core.factory                        import  Dataclass

from ...learner.agents.agent1_predictor     import  (
                                                    Agent1Predictor,
                                                    Agent1PredictorUI
                                                    )
from ...learner.agents.agent2_predictor     import  (
                                                    Agent2Predictor,
                                                    Agent2PredictorUI
                                                    )
from ...learner.agents.agent3_predictor     import  (
                                                    Agent3Predictor,
                                                    Agent3PredictorUI
                                                    )
from ...learner.agents.agent4_predictor     import  (
                                                    Agent4Predictor,
                                                    Agent4PredictorUI
                                                    )
DEPS  = [
    AIGFLOW.Training.agent1, 
    AIGFLOW.Training.agent2, 
    AIGFLOW.Training.agent3, 
    AIGFLOW.Training.agent4
]

agent_type = (
    Agent1Predictor | Agent1PredictorUI | 
    Agent2Predictor | Agent2PredictorUI | 
    Agent3Predictor | Agent3PredictorUI | 
    Agent4Predictor | Agent4PredictorUI
)

AGENTS = {a: m for a,m in enumerate(DEPS, 1)}


class SelectorGenericInput:
    agents: list|tuple

SelectorInputT = TypeVar('SelectorInputT', bound=SelectorGenericInput)

class SelectorInputBase(Generic[SelectorInputT]):    
    """
    Selection phase Input Base to provide flexible input functionality.
    
    Subclass Contract:
    - SelectionInput : type
    - agents_condition: int

    Where `agents_condition` verifies whether optional agents parameter is met on count.
    
    NOTE that defined SelectionInput must contain `agents` parameter.
    """

    SelectionInput : type = None
    agents_condition: int = 1

    def _set_agents(self, inputs: SelectorInputT) -> None:
        """
        Validate and filter the agents list, enforcing minimum count.

        Filters out unknown agent IDs (with a warning) and raises
        a safety error if the remaining valid agents fall below
        the strategy-specific `agents_condition`.   
        """        
        agents = [a for a in inputs.agents if a in AGENTS]
        
        c = [a for a in inputs.agents if a not in AGENTS]
        if c: 
            print(f'Unkown agents requested: {c} ► using subset of known agents: {agents}')
        
        if len(agents) < self.agents_condition:
            name = self.SelectionInput.__name__.split('Input')[0]
            raise ValueError(
                f'AIG SAFETY ► {name.capitalize()} is possible for {self.agents_condition} '
                f'and more agents, please extend the subset of current agent {agents}'
            )
        self.agents = agents
    
    def _align_input_agents(self, inputs: SelectorInputT) -> SelectorInputT:    
        """
        Validate and reconstruct the input with aligned agent list.

        Ensures the input is of the correct type, filters unknown
        agents, enforces the minimum count, and returns a fresh
        `SelectionInput` instance with the corrected `agents` field.
        """    
        self._set_agents(inputs)
        params = Dataclass.get_names(self.SelectionInput)

        if 'agents' not in params:
            n = self.SelectionInput.__name__
            raise ValueError(
                f'AIG SAFETY ► FitterSelectionBase expects "agents" among paramters in {n}.'
            )
        
        return self.SelectionInput(
            **(dict(inputs) | dict(agents = self.agents))
        )

    def get_inputs(self, inputs: SelectorInputT | pDF, kwargs:dict) -> SelectorInputT:
        """
        Resolve and validate the selector input from flexible sources.

        This is the primary entry point for input processing. It
        accepts either a pre-constructed typed input or a raw
        dataframe + kwargs, and returns a validated `SelectionInput`
        with the agent list filtered and count-checked.

        Accepts two input forms:
        1. **Typed**: A `SelectionInput` instance — all parameters
           must already be set.
        2. **Raw**: A `pDF` (the selection dataset) + kwargs for
           remaining parameters (e.g., `agents`, `threshold`).
           The dataframe is mapped to the first field of
           `SelectionInput` automatically.

        Parameters:
            inputs (SelectorInputT | pDF): Either a typed input
                instance or the raw data to be wrapped.
            kwargs (dict): Additional parameters to merge into the
                typed input (used in the raw form).

        Returns:
            SelectorInputT: A validated `SelectionInput` instance
                with aligned agents and all required fields populated.

        Raises:
            TypeError: If the input is neither a `SelectionInput`
                instance nor a `pDF`. The error message guides the
                user to the two accepted forms.

        Example:
            >>> # Form 1: Pre-constructed typed input
            >>> inp = BestSelectorInput(data=df, agents=[1, 2, 3])
            >>> validated = selector.get_inputs(inp, {})
            >>>
            >>> # Form 2: Raw data + kwargs
            >>> validated = selector.get_inputs(df, agents=[1, 2, 3], threshold=0.5)
        """        
        if isinstance(inputs, self.SelectionInput):
            return self._align_input_agents(inputs)
        
        if isinstance(inputs, pDF):
            n = Dataclass.get_names(self.SelectionInput)[0]
            d = {n :inputs} | kwargs
            p = self.SelectionInput(**d)
            return self._align_input_agents(p)
        
        n = self.SelectionInput.__name__
        raise TypeError(
            'AIG SAFETY ► Unkown inputs:'
            f'\n - {n} provides direct paramaterization of both data and agents params'
            '\n - or inputs can be provided as .process(data, agents=[...])'
        )