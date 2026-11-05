"""
Standard Report Interface registered with the DashRegistry for a specified MODEL.

This interface serves as the bridge between the Model's raw output and the 
Dashboard visualization layer, ensuring consistent reporting schemas across 
different model versions.
"""

from ....deps.data                  import  np, pDF
from ....deps.data_view             import  itables_show

from ....core.parameters            import  AIGFLOW
from ....core.registry              import  DashRegister
from ....core.factory               import  Dataclass
from ....core.foundation            import  AigParam
from ....core.display               import  DashInterface
from ....core.interop               import  Pd
from ....core.accessor              import  Rgb, Format

from ....utils.builtins             import  Str
from ....display.style              import  TableStyle

from ..cat_encoder                  import  REG, Fitter
from ..fitters.cat_standardizer     import  (
                                            LBL as LBL_,
                                            FitStats
                                            )


MODEL = AIGFLOW.Encoding.cat_encoder
DREG  = DashRegister(MODEL)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(LBL_):
    GD = 'TO_diff'
    C  = 'Count'
    SX = 'Suffix'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → CAT → DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class DashStats:
    """Dash-Fitter vars connection design."""
    vars_cat:       int
    vars_dropped:   int
    overview_in:    pDF
    overview_out_p: pDF
    overview_out_a: pDF
    overviews:      list[pDF]
    views_size:     list[int]
    in_exists:      int
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@DREG.register(REG.AIG)
class CatEncoderDash(DashInterface):
    """Dash Fitter UI."""       

    def __init__(self, 
        fitter:     Fitter, 
        verbose:    int|bool = True
    ):
        super().__init__(fitter, verbose)
        
        self._p = fitter.params        
        self._f = DashStats(   
            vars_cat = len(fitter.result.vars_cat),
            vars_dropped = len(fitter.result.vars_dropped),
            **self._cat_binning_info(fitter.stats)
        )

        self.format = {
            LBL.M:'{:.1%}', 
            **{x: '{:.3f}' for x in [LBL.GT, LBL.GF, LBL.GO, LBL.GD]},
            **{x: '{:.3f}' for x in [LBL.IVF, LBL.IVT]},
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def _cat_binning_info(self, fit_stats:FitStats) -> dict:        
            
        def overview(data:dict, sortby:str):
            d = pDF(data)
            if len(d):
                d = d.sort_values(by=[sortby], ascending=False).reset_index(drop=True)
                d[LBL.GD] = round(d[LBL.GT]-d[LBL.GO], 3)
                s = len(d)
            else:
                d = pDF({LBL.V:[''], LBL.B:[0], LBL.M:[0]})
                s = 0
            if not d[LBL.B].sum():
                d = d.drop(columns=LBL.B)
            return d, s
        
        view_in, shape_in = overview(fit_stats.stats_in,    LBL.GT)
        view_op, shape_op = overview(fit_stats.stats_out_p, LBL.GF)
        view_oa, shape_oa = overview(fit_stats.stats_out_a, LBL.GF)

        overviews  = [view_in,  view_op,  view_oa ]
        views_size = [shape_in, shape_op, shape_oa]

        ds = Dataclass.schema(DashStats)
        return {
            ds.overview_in:      view_in, 
            ds.overview_out_p:   view_op, 
            ds.overview_out_a:   view_oa, 
            ds.overviews:        overviews,
            ds.views_size:       views_size,
            ds.in_exists:        views_size[0]
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def legend(self):
        a = Str.fx_align(LBL.list())
        d = lambda x: f'{x}:{a(x)}'      
        f = lambda x,t: Format.display([d(x),t], style=Format.Style(formatted=60))
        print()
        Format.display(Rgb.under('LEGEND:'))
        f(LBL.SX,  'F = full (missing as separate cat) · T = transformed · O = original, non-missing only')
        f(LBL.B,   'Number of unique Categories including Missing Category')
        f(LBL.M,   'Mean missing rate (%)')
        f(LBL.IVF, 'Mean IV, transformed (incl. missing cat)')
        f(LBL.IVT, 'Mean IV, transformed (excl. missing)')
        f(LBL.GF,  'Mean Gini, transformed (incl. missing cat)')
        f(LBL.GT,  'Mean Gini, transformed (excl. missing)')
        f(LBL.GO,  'Mean Gini, original (excl. missing)')
        f(LBL.GD,  'Gini_T − Gini_O (transformation effect)')
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def effect_summary(self):
        if not self._is_gui:
            return
        Format.title('► AIG CATEGORICAL ENCODING', style=Format.StyleStats(center=False, margin_t=16))
        Format.point(
            f'Transformed Categorical Features for further WOE Processing: {Rgb.bold_green(self._f.in_exists)}',
            f'Pruned Categorical Features based on univariate <i>Gini Cutoff</i> < {self._p.gini_cutoff:.3f}: {Rgb.bold_red( sum(self._f.views_size[1:]) )}',
            f'Strategy to merge categories by similar Target rate given <i>Minimum Bin Size</i>: {self._p.min_bin_share}'
        )
        if self._f.in_exists and ( self._f.overview_in[LBL.GD].max() > 0 ):
            Format.point(
                'The higher <i>Transformed Gini_T</i> vs. <i>Original Gini_O</i> is mostly caused by Target Noise'
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
            Format.point(f'Higher Missing Rate than set Maximum_Missing {self.setup.max_miss} in User Panel is caused by eliminated Outliers observations')
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def mean_effect(self):
        if self._is_any(
            not self._is_gui,
            not self._f.in_exists
        ): return
        cnt_  = self._f.overview_in[LBL.B].value_counts().astype(int)
        Wmean = self._f.overview_in.drop([LBL.V, LBL.R], axis=1).groupby([LBL.B]).mean().round(3)
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
            'DROPPED FEATURES OVERVIEW → Unpredictive Prior Transformation',
            'DROPPED FEATURES OVERVIEW → Unpredictive After Transformation'
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
        cat = self._f.vars_cat
        out = self._f.vars_dropped
        org = cat + out

        f = lambda x: f"{x:,}"
        m = ' '.join([
            Rgb.bold('► Categorical Encoder →'),
            Rgb.bold_green(f(cat)), 'Transformed Categorical Features Passed →',
            Rgb.bold_red(f(out)), 'of', Rgb.bold(f(org)), 'Dropped as Unpredictive'
        ])               
        self._final_summary(cat=cat, m_base=m, is_view=_is_view)
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