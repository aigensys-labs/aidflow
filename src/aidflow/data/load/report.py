"""
Standard Report Interface registered with the DashRegistry for a specified MODEL.

This interface serves as the bridge between the Model's raw output and the 
Dashboard visualization layer, ensuring consistent reporting schemas across 
different model versions.
"""

from ...deps.data_view      import  itables_show

from ...core.parameters     import  AIGFLOW
from ...core.registry       import  DashRegister
from ...core.foundation     import  AigParam
from ...core.display        import  DashInterface
from ...core.accessor       import  Format, Rgb

from ...utils.builtins      import  Str

from .data_loader           import  (
                                    REG,
                                    Fitter
                                    )
from .processor             import  (
                                    LBL, FX, 
                                    ProcUtils
                                    )

MODEL = AIGFLOW.Data.loader
DREG  = DashRegister(MODEL)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → LOAD → DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@DREG.register(REG.AIG)
class DataLoaderDash(DashInterface):
    """Dash UI of Transformer."""       

    def __init__(self, 
        fitter:  Fitter, 
        verbose: int|bool = True
    ):
        super().__init__(fitter, verbose)
        
        self._i = ProcUtils._info
        self._p = fitter.params
        self._d = fitter.stats
        self._s = self._d.data_stats
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    def _get_adjustment(self) -> tuple[bool, float, float, float]:
        o   = self._s[LBL.OBS_T]
        r   = self._s[LBL.D_SUBSET] / o
        s,e = self._s[LBL.T_SUBSET]
        so  = s/o
        eo  = (e+1)/o
        is_adjusted = ( so != 0 ) or ( eo != 1 )
        return is_adjusted, r, so, eo

    def _import(self) -> list[list[str]]:
        subset_note = ''
        obs_label = 'Total'
        is_adjusted, subset_ratio, time_start_ratio, time_end_ratio = self._get_adjustment()
        
        if is_adjusted:
            obs_label = 'Chosen'
            subset_note  = (
                f" → {subset_ratio:.0%} of Total OBS → "
                f"Subset starts at {time_start_ratio:.0%} to {time_end_ratio:.0%} timewise"
            )                

        return [[
            f"► {self._i('data')} {Rgb.bold("Imported")} ",
            f"→ {self._s[LBL.VARS_CNT]:,} Total Explanatory Variables "
            f"and {self._s[LBL.D_SUBSET]:,} {obs_label} Observations{subset_note}"            
        ]]
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _split(self) -> list[list[str]]:
        align = {
            tx: Str.align(
                f'{self._s[FX.obs(tx)]:,}', 
                f'{self._s[FX.target(tx)]:,}'
            ) for tx in LBL.TT
        }
        
        splt = self._p.train_test
        s = lambda x: [f"► {self._i('data')} {Rgb.bold(x)}", f' → {1-splt:.0%} / {splt:.0%}']
        o = lambda tx,x,a='': f'{self._i(tx)}: {a}{x:,} OBS'
        d = lambda tx,x,y:    f'{self._i(tx)}: {str(x)[:16]} - {str(y)[:16]}'
        t = lambda tx,x:      f'{self._i(tx)}: {x:.1%}'

        msg = []
        def p(a,b,c,d,e):
            m = a + [f' → {b} | {c} → {d} | {e}']
            msg.append(m)

        p(
            s('ML Split'),
            *( o(tx, self._s[FX.obs(tx)]) for tx in LBL.TT ),
            *( d(tx, self._s[FX.time_n(tx)], self._s[FX.time_x(tx)]) for tx in LBL.TT )
        )
        p(
            s('Target=1'),
            *( o(tx, self._s[FX.target(tx)], align[tx]) for tx in LBL.TT ),
            *( t(tx, self._s[FX.target(tx)] / self._s[FX.obs(tx)]) for tx in LBL.TT )
        )
        return msg
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _sampling(self) -> list[list[str]]:
        info = [x for x in self._d.sampling_info if x]        
        if not info:
            return info
        info = [x.split('→') for x in info]
        info = [[x[0],  '→ '+ x[1].strip()] for x in info]
        return info
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def data_sample(self):
        if not self._is_gui:
            return
        Format.splitter(style=Format.StyleSplitter(margin_b=20))
        itables_show(self._d.train_sample)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _log(self, msg:list[str]|str):
        self._final_summary(m=''.join(msg), log_only=True)    
    
    def _summary(self, msg: list[list|str], is_log:bool = False, is_sm:bool = True):
        for m in msg:
            if not m:
                continue
            if is_log:
                self._log(m)
            if self.verbose:
                px = 150 if is_sm else 140
                Format.display(m, style = Format.Style(formatted=px))

    def summary(self, _is_view=True):
        im = self._import()
        sp = self._split()
        sm = self._sampling()
        
        for msg in (im, sp, sm):
            self._summary(msg, not _is_view, len(sm))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @AigParam.with_setattr_runtime(verbose=None)   
    def build(self, save=None, custom_path=None, **kwargs):
        """Construct dashboard with option to save it to default project path or custom path."""
        is_view = self._is_view(kwargs)
        self.save(save, custom_path)
        self.summary(is_view) 
        self.data_sample()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬