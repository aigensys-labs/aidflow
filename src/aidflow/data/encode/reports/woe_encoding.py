"""
Standard Report Interface registered with the DashRegistry for a specified MODEL.

This interface serves as the bridge between the Model's raw output and the 
Dashboard visualization layer, ensuring consistent reporting schemas across 
different model versions.
"""

from ....deps.data              import  np, pDF
from ....deps.data_view         import  itables_show

from ....core.parameters        import  AIGFLOW
from ....core.registry          import  DashRegister
from ....core.factory           import  Dataclass
from ....core.foundation        import  AigParam
from ....core.display           import  DashInterface
from ....core.interop           import  Pd
from ....core.accessor          import  Rgb, Format

from ....utils.builtins         import  Str
from ....display.style          import  TableStyle

from ..woe_encoder              import  (
                                        REG, 
                                        Fitter
                                        )
from ..fitters.woe_transformer  import  (
                                        LBL as LBL_, 
                                        FitStats, 
                                        VarType
                                        )


MODEL = AIGFLOW.Encoding.woe_encoder
DREG  = DashRegister(MODEL)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(LBL_):
    GD = 'WT_diff'
    C  = 'Count'
    SX = 'Suffix'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class DashStats:
    """Dash-Fitter vars connection design."""
    vars_woe:     int
    vars_dropped: int
    overview_in:  pDF
    overview_out: pDF
    overviews:    list[pDF]
    views_size:   list[int]
    in_exists:    int
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@DREG.register([REG.GINI, REG.TEST])
class WoeEncoderDash(DashInterface):
    """Dash Fitter UI."""       

    def __init__(self, 
        fitter:     Fitter, 
        verbose:    int|bool = True
    ):
        super().__init__(fitter, verbose)
        
        self._g = fitter.params.gini_cutoff
        self._f = DashStats(   
            vars_woe = len(fitter.result.vars_woe),
            vars_dropped = len(fitter.result.vars_dropped),
            **self._woe_binning_info(fitter.stats)
        )

        self.format = {
            LBL.M:'{:.1%}', 
            **{x: '{:.3f}' for x in [LBL.GW,  LBL.GO, LBL.GD]},
            **{x: '{:.3f}' for x in [LBL.IVW, LBL.IVO]}
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def _woe_binning_info(self, fit_stats:FitStats) -> dict:        
            
        def overview(data:dict, sortby:str):
            d = pDF(data)
            if len(d):
                d = d.sort_values(by=[sortby], ascending=False).reset_index(drop=True)
                d[LBL.GD] = round(d[LBL.GW]-d[LBL.GO], 3)
                s = len(d)
            else:
                d = pDF({LBL.V:[''], LBL.B:[0], LBL.M:[0]})
                s = 0
            if not d[LBL.B].sum():
                d = d.drop(columns=LBL.B)
            return d, s
        
        view_in, shape_in = overview(fit_stats.stats_in,  LBL.GW)
        view_oa, shape_oa = overview(fit_stats.stats_out, LBL.GW)

        overviews  = [view_in,  view_oa ]
        views_size = [shape_in, shape_oa]
    
        ds = Dataclass.schema(DashStats)
        return {
            ds.overview_in:   view_in, 
            ds.overview_out:  view_oa, 
            ds.overviews:     overviews,
            ds.views_size:    views_size,
            ds.in_exists:     views_size[0]
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def legend(self):
        a = Str.fx_align(LBL.list())
        d = lambda x: f'{x}:{a(x)}'      
        print()
        f = lambda x,t: Format.display([d(x),t], style=Format.Style(formatted=60))
        Format.display(Rgb.under('LEGEND:'))
        f(LBL.SX, 'WF = WOE transformed full · TF = input transformed full')
        f(LBL.T,  'Type of Original Feature - N → Numerical | C → Categorical')
        f(LBL.B,  'Number of unique Bins (missing treated as its own bin)')
        if self._f.in_exists: 
            f(LBL.C, 'Number of WOE features in the bin-count group')
        f(LBL.M,   'Mean missing rate (%)')
        f(LBL.IVW, 'Mean IV, transformed WOE feature (incl. missing bin)')
        f(LBL.IVO, 'Mean IV, input TF feature (incl. missing bin)')
        f(LBL.GW,  'Mean Gini, transformed WOE feature (incl. missing bin)')
        f(LBL.GO,  'Mean Gini, input TF feature (incl. missing bin)')
        f(LBL.GD,  'Gini_WF − Gini_TF (transformation effect)')
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def effect_summary(self):
        if not self._is_gui:
            return        
        oi, oo = self._f.overview_in, self._f.overview_out
        si, so = self._f.views_size
        t = lambda type: Rgb.bold_green( len(oi[oi[LBL.T]==type]) )
        o = lambda type: Rgb.bold_red( len(oo[oo[LBL.T]==type]) )

        Format.title('► AIG WOE ENCODING', style=Format.StyleStats(center=False, margin_t=16))
        if si: Format.point(f'Transformed Features to WOE entering ML Training: Numerical {t(VarType.N)} | Categorical {t(VarType.C)}')
        if so: Format.point(f'Pruned Features given univariate <i>Gini Cutoff</i> < {self._g:.3f}: Numerical {o(VarType.N)} | Categorical {o(VarType.C)}')
        Format.point(
            f'Transformation Objective to Maximize Outcome for ML Optimization',
            f'Stay with Low Bias <i>(WT_diff ~ 0)</i> and try to Lower Variance <i>(BIN_d < 0)</i>'
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def notes(self):
        if not self._is_gui:
            return
        def _any(x): 
            views = [view[x] for view in self._f.overviews if x in view.columns]
            if not views: return 0
            return max( np.concatenate(views) )

        if _any(LBL.M) > self.setup.max_miss*100:
            Format.point(
                f' - Higher Missing Rate than set Maximum_Missing {self.setup.max_miss} in User Panel is caused by eliminated Outliers observations'
            )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def mean_effect(self):
        if self._is_any(
            not self._is_gui,
            not self._f.in_exists
        ): return
        cnt_  = self._f.overview_in[LBL.B].value_counts().astype(int)
        Wmean = self._f.overview_in.drop([LBL.V, LBL.R, LBL.T], axis=1).groupby([LBL.B]).mean().round(3)
        Wmean[LBL.C] = cnt_
        Pd.move_column_inplace(Wmean, LBL.C, 0)
        Wmean = Wmean.reset_index()
        Wmean = Wmean.sort_values(by=[LBL.B])

        Format.title('TRANSFORMATION MEAN EFFECT', style=Format.StyleStats())
        cols = [c for c in Wmean.columns if c!=LBL.B]
        TableStyle.combine_pdf_tables([ 
            Pd.style( Wmean[[LBL.B]], 0, LBL.B ), 
            Pd.style( Wmean[cols], 0, format_cols = self.format )
        ])
        self.legend()        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def status_overview(self):
        if not self._is_gui:
            return
        T = [
            'TRANSFORMATION OVERVIEW',
            'DROPPED FEATURES OVERVIEW',
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

        if not self._f.in_exists: # special case for transformed vars mean report
            self.legend()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
            
    def summary(self, _is_view=True):
        woe = self._f.vars_woe
        out = self._f.vars_dropped
        org = woe + out

        f = lambda x: f"{x:,}"
        m = ' '.join([
            Rgb.bold('► Weight of Evidence Encoder →'),
            Rgb.bold_green(f(woe)), 'Transformed WoE Features Passed →',
            Rgb.bold_red(f(out)), 'of', Rgb.bold(f(org)), 'Dropped as Unpredictive'
        ])         
        self._final_summary(woe, m_base=m, is_view=_is_view)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @AigParam.with_setattr_runtime(verbose=None)    
    def build(self, save=None, custom_path=None, **kwargs):
        """Construct dashboard with option to save it to default project path or custom path."""
        self.save(save, custom_path)
        self.effect_summary()   
        self.notes()
        self.mean_effect()
        self.status_overview()
        self.summary(kwargs)        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬