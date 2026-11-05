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


from .agents_comparator         import  (
                                        REG,
                                        Fitter,                                        
                                        )
from .processor                 import  (
                                        MAP,
                                        ScoringTable,
                                        ComparatorPlot
                                        )

MODEL = AIGFLOW.Selection.comparator
DREG  = DashRegister(MODEL)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TUNING → COMPARISON → DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@DREG.register(REG.AIG)
class AgentsComparatorDash(DashInterface):
    """Dash Fitter UI."""       
    
    def __init__(self, 
        fitter:     Fitter,
        verbose:    int|bool = True,
        theme:      int = 1
    ):
        super().__init__(fitter, verbose)
        self.theme = theme

        self._p  = fitter.params
        self._f  = fitter.stats
        self._g  = ComparatorPlot(self._f, theme)
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def table(self):
        if not self._is_gui:
            return        
        Format.title('AGENTS COMPARISON METRICS & SCORING', style=Format.StyleStats())        
        ScoringTable(self._f).create()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def sankey(self):
        if not self._is_gui:
            return        
        Format.title('AGENTS METRICS VIEW BY SANKEY', style=Format.StyleStats(margin_b=24))        
        self._g.sankey()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def radar(self):
        if not self._is_gui:
            return        
        Format.title('AGENTS SCORING VIEW BY RADAR', style=Format.StyleStats(margin_t=36, margin_b=24))        
        self._g.radar()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def gini(self):
        if not self._is_gui:
            return        
        Format.title('AGENTS PERFORMANCE DETAIL', style=Format.StyleStats(margin_t=36, margin_b=24))        
        self._g.gini()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
       
    def calibration(self):
        if not self._is_gui:
            return        
        Format.title('AGENTS CALIBRATION DETAIL', style=Format.StyleStats(margin_t=36, margin_b=24))        
        self._g.calibration()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_results(self):
        s = self._f.scoring
        a = list(self._f.lift)
        s = s.loc[MAP.S, a]
        w = self._f.ordered[0]
        f = lambda a: f'{s[a]:.1%}'
        v = lambda a: Rgb.bold_green(f(a)) if a == w else f(a)
        return '  |  '.join([f'Agent {a}: {v(a)}' for a in s.index])
        
    def summary(self, _is_view=True):
        m = (
            Rgb.bold(f'► Agents Comparator →') + 
            ' Overall Scoring → ' + self._get_results()
        )
        self._final_summary(m=m, is_view=_is_view)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @AigParam.with_setattr_runtime(verbose=None)    
    def build(self, save=None, custom_path=None, **kwargs):
        """Construct dashboard with option to save it to default project path or custom path."""
        self.save(save, custom_path)
        self.table()
        self.sankey()
        self.radar()
        self.gini()
        self.calibration()
        self.summary(kwargs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬