"""
Standard Report Interface registered with the DashRegistry for a specified MODEL.

This interface serves as the bridge between the Model's raw output and the 
Dashboard visualization layer, ensuring consistent reporting schemas across 
different model versions.
"""

from ....deps.data              import  pDF
from ....deps.data_view         import  itables_show
from ....deps.display           import  display

from ....core.parameters        import  AIGFLOW
from ....core.registry          import  DashRegister
from ....core.foundation        import  AigParam
from ....core.display           import  (
                                        DashInterface, 
                                        DashPlotlyLink
                                        )
from ....core.io                import  DashIO
from ....core.accessor          import  Rgb, Format

from ....ml.features            import  Correlation
from ....display.style          import  TableStyle

from ..fe_cleaner               import  (
                                        REG, 
                                        Fitter
                                        )

MODEL = AIGFLOW.Data.fe_cleaner
DREG  = DashRegister(MODEL)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → CLEANING → DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@DREG.register(REG.AIG)
class FeatureCleanerDash(DashInterface):
    """Dash Fitter UI."""       

    def __init__(self, 
        fitter:     Fitter, 
        verbose:    int|bool = True
    ):
        super().__init__(fitter, verbose)
        
        self._p = fitter.params
        self._f = fitter.result
        self._s = fitter.stats

        self._set_elements({
            'corr': 'correlation_matrix',
        })
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _highlight(self, m, *args, **kwargs):
        Format.title(m, *args, style = Format.StyleStats(center=False, **kwargs))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def overview(self):
        """Statistics of Removing Reasons."""
        if self._is_any(
            not self._is_gui,
            not len(self._f.vars_dropped)
        ):  return            

        m = lambda s: (
            f'► Feature Quality Screening → {Rgb.bold(len(self._s.vars_org))} Candidates → ' 
            f'{Rgb.bold_red(len(self._f.vars_dropped))} Dropped'
        )
        self._highlight(m, self._f.vars_dropped, margin_t=16)

        vars_out = pDF(self._f.vars_dropped)
        vars_out_ = vars_out.set_index('Feature')
        stats = (vars_out_.value_counts('Reason')).reset_index()
        stats.columns = ['Reason', 'Count']
        stats['Distribution'] = (stats['Count'] / stats['Count'].sum()).round(3)
                    
        stats = (
            stats.style
            .set_properties( subset=['Reason'], **{'width': '400px', 'text-align': 'left'})
            .set_table_styles( [ dict(selector='th', props=[('text-align', 'left')])] )
            .format({'Distribution': '{:,.1%}'})
            .hide(axis=0)
        )

        display(stats)
        
        m1 = int( vars_out['Feature'].str.len().max() *8 + 15)  # 8 avg letter len + 15 extra space
        m3 = int( vars_out['Reason'].str.len().max()   *7)      # 7 avg letter len of reasons
        si = dict(
                columnDefs = [
                    {"width": "40px",    "targets": 0},
                    {"width": f"{m1}px", "targets": 1},
                    {"width": "100px",   "targets": 2},
                    {"width": f"{m3}px", "targets": 3},
                ],
                style = "width:auto !important;",
                # style = "width:auto; float: left; paging-align: center;",
                # layout = {
                #     "topStart": "search", "topEnd": None, # "top2Start": "pageLength", 
                #     "bottomStart": "paging", "bottom2Start": "info", 'bottomEnd': None
                # },
                # buttons=["pageLength", "copyHtml5", "csvHtml5"]
        )
        itables_show(pDF(self._f.vars_dropped), **si, pageLength=5, lengthMenu=[5, 10, 25, 50, 100])
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def corr(self, paths):
        """Correlation Reasons & Heatmap."""
        if self._is_any(
            not self._is_gui,
            self._s.corr_upper is None, 
        ):  return

        lc = len(self._s.corr_upper.columns)
        
        if not ( lc > 2 ):
            return
               
        x = ( lc > 100 )
        
        m = lambda s: (
                f'► Correlation Screening of {Rgb.bold(lc)} Numeric Features passing prior Quality Checks'+
                f' → {Rgb.bold_red(self._s.corr_removed)} Feature{s} dropped given |ρ| > {self._p.max_corr}'
        )
        self._highlight(m, self._s.corr_removed, margin_b=0)

        def fig_fx():
            layout = dict(height = 600*(1+x), margin = {'t':20, 'b':20, 'l':10, 'r':10})
            return Correlation.heatmap(upper = self._s.corr_upper, update_layout = layout)
        
        fig, report_path = DashIO.resolve_fig_path(fig_fx, paths, 'corr')
        
        link = DashPlotlyLink(
            figure_path      = fig,      
            report_path      = report_path,
            link_name        = 'Correlation Heatmap',
            hide_columns     = True,
            show             = False,
        ).get_output()        

        TableStyle.combine_pdf_tables(link)
        print()
        Correlation.show(self._s.corr_upper, max_corr = self._p.max_corr)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def overview_type(self):
        """Statistics of Type Fixes."""
        if not self._is_gui:
            return
        
        lt = len(self._s.d_typefix)
        if not lt:
            return
                
        fixed = pDF(self._s.d_typefix).style.set_properties( subset=['Reason'], **{'width': '500px'} )

        m = lambda s: f'► Fixed {Rgb.bold(lt)} Feature Type{s}'
        self._highlight(m, lt)
        
        itables_show(fixed)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def summary(self, _is_view=True):
        num = self._f.numeric
        cat = self._f.object
        
        f = lambda x: f"{len(x):,}"
        m = ' '.join([
            Rgb.bold('► Feature Cleaner →'),
            Rgb.bold_green(f(num)), 'Numerical |', 
            Rgb.bold_green(f(cat)), 'Categorical Features Passed →',
            Rgb.bold_red(f(self._f.vars_dropped)), 'of', 
            Rgb.bold(f(self._s.vars_org)), 'Dropped'
        ])
        self._final_summary(num, cat, m = m, is_view = _is_view)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @AigParam.with_setattr_runtime(verbose=None)    
    def build(self, save=None, custom_path=None, **kwargs):
        """Construct dashboard with option to save it to default project path or custom path."""
        paths = self.save(save, custom_path, save_signature = True)
        self.overview()
        self.corr(paths)
        self.overview_type()
        self.summary(kwargs)        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬