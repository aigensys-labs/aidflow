"""
Standard Report Interface registered with the DashRegistry for a specified MODEL.

This interface serves as the bridge between the Model's raw output and the 
Dashboard visualization layer, ensuring consistent reporting schemas across 
different model versions.
"""

from ....core.parameters        import  AIGFLOW
from ....core.registry          import  DashRegister

from ..agent1_predictor         import  (
                                        REG, 
                                        Fitter,
                                        DashLink
                                        )
from ..agent2_predictor         import  (
                                        REG as REG2,
                                        Fitter as Fitter2,
                                        DashLink as DashLink2
                                        )
from ..agent3_predictor         import  (
                                        REG as REG3,
                                        Fitter as Fitter3,
                                        DashLink as DashLink3
                                        )

from .base                      import  AgentDashBase


MODEL1 = AIGFLOW.Training.agent1
DREG1  = DashRegister(MODEL1)

MODEL2 = AIGFLOW.Training.agent2
DREG2  = DashRegister(MODEL2)

MODEL3 = AIGFLOW.Training.agent3
DREG3  = DashRegister(MODEL3)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT 1-3 → DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class AgentGLMDash(AgentDashBase):
    """Dash Fitter UI."""       

    TYPE:  str = 'GLM'
    AGENT: int
    MODEL: str
    DESC:  str

    def __init__(self, 
        fitter:     Fitter | Fitter2 | Fitter3, 
        fit_link:   DashLink | DashLink2 | DashLink3,
        verbose:    int|bool = True
    ):
        super().__init__(fitter, fit_link, verbose)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@DREG1.register(REG.AIG)
class Agent1Dash(AgentGLMDash):
    AGENT = 1
    MODEL = MODEL1
    DESC  = 'Sparse'

@DREG2.register(REG2.list())
class Agent2Dash(AgentGLMDash):
    AGENT = 2
    MODEL = MODEL2
    DESC  = 'Dense'

@DREG3.register(REG3.AIG)
class Agent3Dash(AgentGLMDash):
    AGENT = 3
    MODEL = MODEL3
    DESC  = 'PLS'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬