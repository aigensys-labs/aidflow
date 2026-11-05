"""
Standard Report Interface registered with the DashRegistry for a specified MODEL.

This interface serves as the bridge between the Model's raw output and the 
Dashboard visualization layer, ensuring consistent reporting schemas across 
different model versions.
"""

from ...core.parameters         import  AIGFLOW
from ...core.registry           import  DashRegister
from ...core.foundation         import  AigParam
from ...core.display            import  DashInterface
from ...core.accessor           import  Rgb, Format

from .agents_transition         import  (
                                        REG,
                                        Fitter,                                        
                                        )
from .processor                 import  (
                                        LBL_T,
                                        TransitionPlot
                                        )

MODEL = AIGFLOW.Selection.transition
DREG  = DashRegister(MODEL)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TUNING → TRANSITION → DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@DREG.register(REG.AIG)
class AgentsTransitionDash(DashInterface):
    """Dash Fitter UI."""       
    
    def __init__(self, 
        fitter:  Fitter,
        verbose: int|bool = True,
    ):
        super().__init__(fitter, verbose)

        self._p  = fitter.params
        self._f  = fitter.stats
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def dash(self):
        if not self._is_gui:
            return        
        Format.title('AGENTS TRANSITION', style=Format.StyleStats(margin_b=25))        
        TransitionPlot(self._f).dash()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_results(self):
        s = self._f.summary[LBL_T.AE]
        w = s[s == s.min()].index[0]
        f = lambda a: f'{s[a]:.4f}'
        v = lambda a: Rgb.bold_green(f(a)) if a == w else f(a)
        return '  |  '.join([f'Agent {a}: {v(a)}' for a in s.index])
        
    def summary(self, _is_view=True):
        m = (
            Rgb.bold(f'► Agents Transition →') + 
            ' Significant Audited E(T) Divergence → ' + self._get_results()
        )
        self._final_summary(m=m, is_view=_is_view)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @AigParam.with_setattr_runtime(verbose=None)    
    def build(self, save=None, custom_path=None, **kwargs):
        """Construct dashboard with option to save it to default project path or custom path."""
        self.save(save, custom_path)
        self.dash()
        self.summary(kwargs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬