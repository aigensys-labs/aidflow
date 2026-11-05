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
from ...core.accessor           import  UI, Rgb, Format

from .agent_selector            import  (
                                        REG,
                                        Fitter,                                        
                                        )
from .processor                 import  (
                                        DIAG,
                                        SelectorGiniPlot
                                        )

MODEL = AIGFLOW.Selection.selector
DREG  = DashRegister(MODEL)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TUNING → SELECTION → DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@DREG.register(REG.AIG)
class AgentSelectorDash(DashInterface):
    """Dash Fitter UI."""       
    
    def __init__(self, 
        fitter:     Fitter,
        verbose:    int|bool = True,
    ):
        super().__init__(fitter, verbose)

        self._p = fitter.params
        self._d = fitter.stats.diagnostics        
        self._set_fes(fitter)
        self._set_is_gbm(fitter)

    def _set_fes(self, fitter: Fitter) -> None:
        cols = fitter.result.agent_predictor.cols
        self._ca = len(cols.selected)
        self._ta = self._ca + len(cols.dropped)

    def _set_is_gbm(self, fitter: Fitter) -> None:
        self.is_gbm = fitter.result.agent_predictor.is_gbm
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def _get_agent(self) -> int:
        if isinstance(self._p.agent_choice, int):
            return self._p.agent_choice
        return next(iter(self._p.agent_choice))
    
    def title(self):
        if not self._is_gui:
            return
        Format.title(
            f'SELECTED AGENT {self._get_agent()} PERFORMANCE', 
            style=Format.StyleTitle(margin_t=25, margin_b=25)
        )
    
    def performance(self):
        if not self._is_gui:
            return
        SelectorGiniPlot(
            self._d, update_layout = dict(margin=dict(t=40, b=20, l=0, r=0))
        ).create()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_result(self):
        c = self._d[DIAG.C]
        t = self._d[DIAG.T]
        r = lambda fe, g, iv, ks: f': Features: {fe} | Gini: {g:.3} | IV: {iv:.3f} | KS: {ks:.3f}'
        return (
            f'{Rgb.bold("Training")} {r(self._ta, t.gini, t.IV, t.KS)} → '
            f'{Rgb.bold_blue("Champion")} {r(self._ca, c.gini, c.IV, c.KS)}'
        )

    def summary(self, _is_view=True):
        a = self._get_agent()
        d = '\n' if not UI.is_html() else ''
        m = ' '.join([
            Rgb.bold(f'► Agent Selector → Agent {a} Selected'),
            f'{d}► Test Performance →', self._get_result()            
        ])
        self._final_summary(m=m, is_view=_is_view)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @AigParam.with_setattr_runtime(verbose=None)    
    def build(self, save=None, custom_path=None, **kwargs):
        """Construct dashboard with option to save it to default project path or custom path."""
        self.save(save, custom_path)
        # self.title()
        self.performance()
        self.summary(kwargs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬