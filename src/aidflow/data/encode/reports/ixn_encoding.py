"""
Standard Report Interface registered with the DashRegistry for a specified MODEL.

This interface serves as the bridge between the Model's raw output and the 
Dashboard visualization layer, ensuring consistent reporting schemas across 
different model versions.
"""

from ....deps.data              import  pDF
from ....deps.data_view         import  itables_show

from ....core.parameters        import  AIGFLOW
from ....core.registry          import  DashRegister
from ....core.factory           import  Dataclass
from ....core.foundation        import  AigParam
from ....core.display           import  DashInterface
from ....core.interop           import  Pd
from ....core.accessor          import  Rgb, Format

from ....display.rgb            import  RgbANSI

from ..ixn_encoder              import  REG, Fitter
from ..fitters.ixn_extractor    import  FitStats, LBL


MODEL = AIGFLOW.Encoding.ixn_encoder
DREG  = DashRegister(MODEL)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → IXN → DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class DashStats:
    """Dash-Fitter vars connection design."""
    vars_ixn:       int
    vars_dropped:   int
    overview_in:    pDF
    overview_out:   pDF
    overviews:      list[pDF]
    views_size:     list[int]
    in_exists:      int
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@DREG.register([REG.LGB, REG.XGB])
class IxnEncoderDash(DashInterface):
    """Dash Fitter UI."""       

    def __init__(self, 
        fitter:     Fitter, 
        verbose:    int|bool = True
    ):
        super().__init__(fitter, verbose)
        
        self._p = fitter.params        
        self._f = DashStats(   
            vars_ixn = len(fitter.result.vars_ixn),
            vars_dropped = len(fitter.result.vars_dropped),
            **self._ixn_info(fitter.stats)
        )

        self.format = {
            **{x: '{:.1%}' for x in [LBL.S, LBL.T]}, 
            **{x: '{:.3f}' for x in [LBL.G, LBL.IV]}
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def _ixn_info(self, fit_stats:FitStats) -> dict:        
            
        def overview(data:dict, sortby:str):
            d = pDF(data)
            if len(d):
                d = d.sort_values(by=[sortby], ascending=False).reset_index(drop=True)
                s = len(d)
            else:
                d = pDF({LBL.V:[''], LBL.S:[0], LBL.T:[0]})
                s = 0
            return d, s
        
        view_in,  shape_in  = overview(fit_stats.stats_in,  LBL.G)
        view_out, shape_out = overview(fit_stats.stats_out, LBL.G)

        overviews  = [view_in,  view_out ]
        views_size = [shape_in, shape_out]
    
        ds = Dataclass.schema(DashStats)
        return {
            ds.overview_in:  view_in, 
            ds.overview_out: view_out, 
            ds.overviews:    overviews,
            ds.views_size:   views_size,
            ds.in_exists:    views_size[0]
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def effect_summary(self):
        if not self._is_gui:
            return        
        Format.title('► AIG NON-LINEAR INTERACTIONS SEARCH', style=Format.StyleStats(center=False, margin_t=16))
        Format.point(
            f'Transformed Interaction Features for further WOE Processing: {RgbANSI.bold_green(self._f.in_exists)}',
            f'Pruned Interaction Features based on univariate <i>Gini Cutoff</i> < {self._p.gini_cutoff:.3f}: {RgbANSI.bold_red( sum(self._f.views_size[1:]) )}',
            f'Strategy to Extract non-linear Interactions among Features by a Tree-Search',
            f'Extracted Interactions are <i>encoded to binary Categorical Features 0/1</i>'
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def status_overview(self):
        if not self._is_gui:
            return
        T = [
            'TRANSFORMATION OVERVIEW',
            'DROPPED FEATURES OVERVIEW → Unpredictive Transformation',
        ]        
        mv = lambda o: int( o[LBL.V].str.len().max() * 8 + 15)    # 8 avg letter len + 15 avg padding
        mo = max([mv(o) for o in self._f.overviews])
        si = dict(columnDefs = [{"width": f"{mo}px", "targets": 0}], style = "width:100%")
        
        for i, overview in enumerate(self._f.overviews):
            if not self._f.views_size[i]: 
                continue
            
            Format.title(T[i], style=Format.StyleStats())
            overview = overview.drop(columns=LBL.R, errors='ignore')    # Reason is shown in Title
            itables_show(Pd.style(overview, 0, LBL.V, self.format), **si) 
    # ─────────────────────────────────────────────────────────────────────────────────────────────
            
    def summary(self, _is_view=True):
        ixn = self._f.vars_ixn
        out = self._f.vars_dropped
        org = ixn + out

        f = lambda x: f"{x:,}"
        m = ' '.join([
            Rgb.bold('► Interaction Encoder →'),
            Rgb.bold_green(f(ixn)), 'Transformed Interaction Features Passed →',
            Rgb.bold_red(f(out)), 'of', Rgb.bold(f(org)), 'Dropped as Unpredictive'
        ])          
        self._final_summary(cat=ixn, m_base=m, is_view=_is_view)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @AigParam.with_setattr_runtime(verbose=None)    
    def build(self, save=None, custom_path=None, **kwargs):
        """Construct dashboard with option to save it to default project path or custom path."""
        self.save(save, custom_path)
        self.effect_summary()   
        self.status_overview()
        self.summary(kwargs)        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬