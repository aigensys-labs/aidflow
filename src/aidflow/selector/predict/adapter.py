"""
Handles independent agent adapter for inference based on core/adapter.
Stored AgentAdapter with lineage hash verification to eliminate 
any external dependency change.
"""

from ...core.structure      import  AIGFLOW
from ...core.base.adapter   import  ModelAdapterBase
from ..base.processor       import  AGENTS, agent_type


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAININING → TUNING → AGENTS PREDICTION ON OOT
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class AgentAdapter(ModelAdapterBase):
    """
    Creates independent agent adapter for inference.
    Stored AgentAdapter with lineage hash verification to eliminate any external dependency change.
    """
    PERSIST_FROM_REPORT = ('feature_importance',)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PredictorAdapter:
    """
    Standardized interface to obtain agents prediction.
    - use `get_agent(a:int)` to obtain an agent. Adapter has Agent's  standard API
    """

    AGENTS = AGENTS

    def __init__(self, predictors: tuple | list | dict):
        predictors: tuple[agent_type] = (
            predictors if not isinstance(predictors, dict) else tuple(predictors.values())
        )
    
        self._verify(predictors)

        self.predictors: dict[str, AgentAdapter | None] = {
            m: AgentAdapter(p) for m in AGENTS.values() 
                for p in predictors if self._is_model(p, m)
        }
    
    def get_agent(self, a:str|int, none_raises: bool = True) -> AgentAdapter | None:
        """
        Agent Adapter with standard Agent's API.
        - API extension by `predict` → transform(data).data
        - Agents mapping accessible via self.AGENTS
        """
        if isinstance(a, int):
            a = AGENTS.get(a)
        self._verify_agent(a, none_raises)
        return self.predictors.get(a)

    def is_gbm(self, a:str|int) -> bool:
        m = self.get_agent(a)
        return m.MODEL == AIGFLOW.Training.agent4
    
    @staticmethod
    def _is_model(p: agent_type, m: str) -> bool:
        if p is None:
            return False
        return p.MODEL == m
        
    def _verify_agent(self, a:str, none_raises: bool = True) -> None:
        if a not in AGENTS.values():
            raise ValueError(f'Unkown agents requested: {a} ► available agents: {AGENTS}')
        if not (self.predictors.get(a) is None and none_raises):
            return
        raise ValueError(
            f'AIG SAFETY ► Agent Choice {a} is possible only when {a} was fitted prior.'            
        )
        
    @staticmethod
    def _verify(predictors:tuple) -> None:
        e = 'AIG SAFETY ► "predictors"'
        
        if predictors is not None:
            if all([(isinstance(a, agent_type) or a is None) for a in predictors]):
                return            
            raise ValueError(
                f'{e} provided by tuple|list must be an instance of registred agent predictor:\n► {agent_type}'
            )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬