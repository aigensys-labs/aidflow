"""
Standard Report Interface registered with the DashRegistry for a specified MODEL.

This interface serves as the bridge between the Model's raw output and the 
Dashboard visualization layer, ensuring consistent reporting schemas across 
different model versions.
"""

from ....deps.display           import  display

from ....core.factory           import  (
                                        DictClass,
                                        Dataclass 
                                        )
from ....core.foundation        import  AigParam
from ....core.display           import  DashInterface
from ....core.accessor          import  Rgb, Format

from ....ml.metrics             import  (
                                        LBL as LBL_D,
                                        DiagnosticsResult,
                                        GiniLiftPlot
                                        )

from ...features.importance     import  (
                                        FeatureImportance,
                                        FeatureImportanceResult
                                        )
from ...features.pruning        import  GINI

from ..fitters.glm_selector     import  FitStats as StatsGLM
from ..fitters.gbm_selector     import  FitStats as StatsGBM

from ..agent1_predictor         import  (
                                        Fitter as Fitter1,
                                        DashLink as DashLink1
                                        )
from ..agent2_predictor         import  (
                                        Fitter as Fitter2,
                                        DashLink as DashLink2
                                        )
from ..agent3_predictor         import  (
                                        Fitter as Fitter3,
                                        DashLink as DashLink3
                                        )
from ..agent4_predictor         import  (
                                        Fitter as Fitter4,
                                        DashLink as DashLink4
                                        )


type Fitter   = Fitter1   | Fitter2   | Fitter3   | Fitter4
type DashLink = DashLink1 | DashLink2 | DashLink3 | DashLink4
type FitStats = StatsGLM  | StatsGBM


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    GT = 'Train_ΣGini'
    GV = 'Valid_ΣGini'

@Dataclass.base
class DashStats:
    """Dash-Fitter vars connection design."""
    diag:      DiagnosticsResult
    perf:      dict
    gini:      dict
    chosen:    int
    pruned:    int
    selected:  int
    input_fes: int
    smp_obs:   int
    smp_ratio: float
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class AgentDashBase(DashInterface):
    """Dash Fitter UI."""       

    TYPE:  str  
    AGENT: int
    MODEL: str
    DESC:  str = None

    def __init__(self, 
        fitter:     Fitter, 
        fit_link:   DashLink,
        verbose:    int|bool = True
    ):
        super().__init__(fitter, verbose)

        self._p  = fitter.params
        self._f  = self._selection_info(fit_link, fitter.stats)
        self._pr = self._pruner_fig(fit_link)
        self._im = self._importance(fit_link)

        self.elements = {}
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    @property
    def feature_importance(self) -> FeatureImportanceResult:
        return self._im.result
    
    def _selection_info(self, fit_link: DashLink, fit_stats: FitStats) -> DashStats:
        diag = fit_stats.diagnostics
        perf = {
            'Gini': diag.gini,
            'IV'  : diag.IV,
            'KS'  : diag.KS
        }
        gini_path = fit_stats.gini_path

        fes  = gini_path.result_fes
        gini = gini_path.train_valid.loc[fes]
        gini = {
            GINI.T: gini[GINI.T].iloc[-1] if fes else 0, 
            GINI.V: gini[GINI.V].iloc[-1] if fes else 0
        }

        selected = len(fes)
        pruned   = len(gini_path.removed_fes)

        return DashStats(
            diag      = diag,
            perf      = perf,
            gini      = gini,
            chosen    = int(selected + pruned),
            pruned    = pruned,
            selected  = selected,
            input_fes = fit_link.total_fes,
            smp_obs   = fit_link.pruner.get_sample_obs(),
            smp_ratio = fit_link.pruner.get_sample_ratio()
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def _diag_perf(self):
        return ' | '.join([f"{n}: {Rgb.bold(f'{x:.3f}')}" for n,x in self._f.perf.items()])
    
    def _fe_woe(self):
        return '' if self.TYPE != 'GLM' else 'WOE '

    def _pruner_fig(self, link: DashLink):
        return link.pruner.plot(show=False)
    
    def _importance(self, link: DashLink):
        d = link.data
        t = d[~d.index.isin(link.val_idx)]
        v = d.loc[link.val_idx]

        o = link.pruner.get_ordered_importance_fes(link.features)
        g = link.pruner.gini_path.train_valid.rename(columns={GINI.V: LBL.GV})[LBL.GV]
        
        n = ' | Top 20 Features' if len(o) > 20 else ''

        return FeatureImportance(
            model      = link.model,
            dash_title = dict(
                table = 'Feature Effect by Method on Validation Sample',
                main  = 'Feature Drivers & Model Sensitivity on Validation Performance | Ordered by CV Signal-to-Noise Ratio',
                shap  = 'Feature Shap Effect on Validation Performance | Ordered by SHAP' + n
            ),
            verbose = False
        ).fit(
            X = v[link.features], 
            y = v[link.target], 
            ablation_train = (t[link.features], t[link.target]),
            features_order = o, 
            dash_extras = g
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def selection_summary(self):
        if not self._is_gui:
            return
        s = '' if not self.DESC else self.DESC.upper() + ' '
        w = self._fe_woe()

        Format.title(f'► AIG {s}{self.TYPE} SELECTOR', style=Format.StyleStats(center=False, margin_t=16))
        Format.point(
            f'Final Selected Features by Agent {self.AGENT}: {Rgb.bold_green(self._f.selected)}',
            f'Chosen Features during {self.TYPE} optimization: {Rgb.bold(self._f.chosen)} '
            f'from {Rgb.bold(self._f.input_fes)}',
            f'Pruned {w}Features by consequent pruning given <i>Shadow Quantile '
            f'{self._p.shadow_quantile:.2f}</i>: {Rgb.bold_red( self._f.pruned)}',
            f'Agent {self.AGENT} Selected Model <i>Validation Performance</i> → {self._diag_perf()}'
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def diagnostics(self):
        if not self._is_gui:
            return
        Format.title(
            'SELECTED MODEL VALIDATION PERFORMANCE | FULL VALID SAMPLE',
            style=Format.StyleStats()
        )
        d = {f'Agent {self.AGENT}': self._f.diag}    
        t = GiniLiftPlot.title_helper(d, lambda x: x)
        p = 'Valid Performance'
        titles = [
            f'Gini {p} → {t[LBL_D.G]}',
            f'Lift {p} → {t[LBL_D.L]}'
        ]
        GiniLiftPlot(d, plot_titles = titles, update_layout = dict(showlegend = False)).create()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    #     
    def pruner(self):
        if not self._is_gui:
            return
        s = f' | SUBSAMPLED TO {self._f.smp_obs:,}' if self._f.smp_ratio < 1 else ''

        Format.title(
            'CV FEATURE IMPORTANCE & SHADOW PRUNING | MODEL GINI PATH' + s,
            style=Format.StyleStats()
        )
        display(self._pr)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def importance(self):
        if not self._is_gui:
            return
        self._im.show_dash()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def summary(self, _is_view=True):
        w = self._fe_woe()
        
        f = lambda x: f"{x:,}"
        def m(s1): 
            return ' '.join([
                Rgb.bold(f'► Agent {self.AGENT} →'), 
                Rgb.bold_green(f(selected)), 'of', Rgb.bold(f(self._f.input_fes)), 
                f'{w}Feature{s1} Selected', '► Validation Performance →', self._diag_perf()
            ])
        selected = self._f.selected
        self._final_summary(m=m, s1=selected, is_view=_is_view)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @AigParam.with_setattr_runtime(verbose=None)    
    def build(self, save=None, custom_path=None, **kwargs):
        """Construct dashboard with option to save it to default project path or custom path."""
        self.save(save, custom_path)
        self.selection_summary()
        self.diagnostics()
        self.pruner()
        self.importance()
        self.summary(kwargs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬