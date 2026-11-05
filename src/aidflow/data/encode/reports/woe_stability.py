"""
Standard Report Interface registered with the DashRegistry for a specified MODEL.

This interface serves as the bridge between the Model's raw output and the 
Dashboard visualization layer, ensuring consistent reporting schemas across 
different model versions.
"""

from ....deps.data              import  np, pDF
from ....deps.plots             import  go, pio
from ....deps.common            import  (
                                        Path,
                                        lru_cache
                                        )
from ....deps.display           import  (
                                        wg, 
                                        HTML, 
                                        display, 
                                        clear_output
                                        )
from ....deps.types             import  Callable
from ....core.parameters        import  AIGFLOW
from ....core.registry          import  DashRegister
from ....core.factory           import  (
                                        Dataclass, 
                                        DictClass
                                        )
from ....core.foundation        import  AigParam
from ....core.execution         import  BackgroundProcess
from ....core.display           import  (
                                        DashInterface, 
                                        DashReportLog,
                                        DashBackground, 
                                        DashPlotlyLink,
                                        Display
                                        )
from ....core.io                import  DashIO
from ....core.interop           import  Pd
from ....core.standard          import  VarStd
from ....core.accessor          import  (
                                        Rgb, 
                                        Format, 
                                        Spinner
                                        )
from ....ml.features.stability  import  (
                                        LBL  as S_LBL, 
                                        DASH as S_DASH,
                                        STAB,
                                        FEConsistencyPlot
                                        )
from ....ml.features.woe        import  (
                                        LBL  as W_LBL, 
                                        DASH as W_DASH,
                                        WoEPlot
                                        )

from ....display.plots          import  Plot
from ....display.style          import  TableStyle
from ....utils.builtins         import  Str

from ..woe_stabilizer           import  (
                                        REG, 
                                        Fitter, 
                                        FitParams, 
                                        FitResult, 
                                        FitStats, 
                                        DashData
                                        )
from ..fitters.woe_analyzer     import  TT


MODEL = AIGFLOW.Encoding.woe_stabilizer
DREG  = DashRegister(MODEL)
CACHE = 20


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    """Mapping to `keys` of S_LBL and W_LBL."""
    VALUE   = 'value'
    IS      = 'is_stable'
    GINI    = 'gini'
    DATA    = 'data'
    VERDICT = 'verdict'
    GMX     = 'gmx'
    GRAPH   = 'graph'

class DASH(DictClass):
    """Mapping to `keys` of S_DASH and W_DASH."""
    DATA    = 'data'
    FIG     = 'fig'
    ORDER   = 'order'

class UNS(DictClass):
    """var-test-stab-type-graph labels"""
    FE      = 'Feature'
    TEST    = 'Stability'
    STAB    = 'Stable'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE STABILITY → DASH DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class DashResult(FitResult):
    """Dash-Fitter vars connection design."""
    woe_stability:  dict
    woe_predict:    dict
    th:             float

SCHEMA = Dataclass.schema(DashResult)

class RES(DictClass):
    """test-woe → Link to DashStats → Main Dash Result by test type."""
    TEST = SCHEMA.woe_stability
    WOE  = SCHEMA.woe_predict
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class WoeStabilizer(DashInterface):
    """Dash Fitter UI."""       

    def __init__(self, 
        fitter:     Fitter, 
        dash_data:  DashData,
        verbose:    int|bool   = True,
        create:     bool       = True
    ):
        super().__init__(fitter, verbose)
        self.dashio = DashIO(self.setup)
                
        self._set_elements({
            'graphs':  self._create_elements()
        })
            
        self._process = DashBackground()
        self._process.set_sync_wait()

        self._summary = DashSummary(fitter.result, verbose)
        if self._is_info_only: # (1) Fast path to summary under low verbosity
            self.summary()

        if not create:
            return
                
        args = (fitter.params, fitter.result, dash_data, verbose)
        self._init_set_created( self._init_create(*args) )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _create_elements() -> dict:
        n = lambda s,t: f'dash_grid_{s}_{t}'.lower()
        return {
            tx: {sw: n(sw, tx) for sw in [RES.TEST, RES.WOE]} 
            for tx in TT.list()
        }
      
    @staticmethod
    def _init_create(
        fit_prm:     FitParams,
        fit_result:  FitResult, 
        dash_data:   DashData,        
        verbose:     int|bool = True,
    ) -> tuple[FitParams, FitResult, dict]:                    
        return (
            fit_prm, 
            fit_result,             
            DashCreator(verbose).create(dash_data),
        )
    
    def _set_init_create(self, res:tuple[FitParams, FitResult, dict]) -> None:
        fit_prm, fit_result, dash = res
        self._p = fit_prm
        self._f = dict(DashResult(
            **fit_result,
            **dash
        ))
    
    def _align_dash_result_type(self):
        if isinstance(self._f, dict):
            self._f = DashResult(**self._f)

    def _set_init_post(self) -> None:
        self._is_serialized = False
        self._snap_init_state( _process = self._process.get_sync_to_snap() )
        self._align_dash_result_type()

    def _init_set_created(self, res:tuple[FitParams, FitResult, dict]) -> None:        
        self._set_init_create(res)
        self._set_init_post()
        self._process.set_sync_ready(1)
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    @property
    def _vars_out_user(self):
        return self._interactive.get_user_out_choices()        
        
    @property
    def _vars_out_user_path(self):
        return self.dashio.get_custom_selection_path(MODEL)

    def _load_vars_out_user(self):
        return self.dashio.load(custom_path=self._vars_out_user_path)

    def _save_vars_out_user(self):
        vars_out = self._vars_out_user
        if not self._interactive.diff_user_from_default_status():
            vars_out = None
        self.dashio.save_file(vars_out, self._vars_out_user_path)
    
    def _get_vars_out_user(self) -> dict|None:
        """Memory of Updated vars_user during Training UI."""
        if not self._is_interactive():
            return self._load_vars_out_user()  # load memory
        return self._vars_out_user             # active gui

    def _update_vars_out_user(self) -> None:
        """Update dash state by Training dynamic User UI confirmed changes and save it."""              
        if not self._is_interactive(): # (2) Fast path to summary under full verbosity UI
            self.summary()
            return
        
        self._interactive.reset_button()
        self.summary()
        self._save_vars_out_user()    

    def _is_interactive(self) -> bool:
        return hasattr(self, '_interactive')

    def _is_user_vars_note(self) -> bool:
        d = self._load_vars_out_user()
        if self._is_interactive():
            d = self._interactive.diff_user_from_default_status()        
        return bool(d)    
# ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    def title(self):
        if not self._is_gui:
            return          
        t = f' WOE Features Dashboard | TH {self._f.th}'.upper()
        Format.title(t.upper(), style=Format.StyleStats(margin_t=16))
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def update(self, _is_view=None):
        if self._is_any(
            not self._is_gui,
            _is_view, # report view
            not self._is_user_vars_note(),
        ): return          
        self._info_highlight(
            'CUSTOM SELECTION ACTIVE ► Feature status reflects a prior custom selection '
            f'► Report provides view by current threshold {self._f.th} '
            f'► Use Reset to restore TH-based selection',
            is_view=_is_view
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def graphs(self, paths: dict[str, dict[str, Path]]):
        if not self._is_gui:
            return
        d = DashGrid(self._f, paths)
        d.create(self._is_gui)
        
        for tx in TT.list():
            d.notes(tx)
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def unstable(self):
        if not self._is_gui:
            return                    
        DashUnstable(self._f).create(limit=10, verbose=self._is_gui)
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def interact(self, _is_view=False):
        if not self._is_gui:
            return
        self._interactive = DashInteract(self._f, self._get_vars_out_user(), _is_view)
        self._interactive.create()
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def summary(self, _is_view=True):
        self._summary(self._get_vars_out_user(), is_view = _is_view)
    # ───────────────────────────────────────────────────────────────────────────────────────────── 

    def _get_init_clone(self, save=None):
        if not self._is_save(save):
            return self
        init_clone = self._clone_init_state(_is_serialized=True)        
        if not self._is_serialized:
            self._is_serialized = DashMixin(init_clone._f)._serialize(0)
        return init_clone
    
    def save(self, save=None, custom_path=None):
        """Saves dashboard - serves reporting purpose of dashboard - for trained, stored models."""
        return self.dashio.save(
            self._get_init_clone(save), save, custom_path, save_signature = True
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @AigParam.with_setattr_runtime(verbose=None)
    def build(self, save=None, custom_path=None, **kwargs):
        """Construct dashboard with option to save it to default project path or custom path."""
        paths = self.save(save, custom_path)
        is_vw = self._is_view(kwargs)
        self.title()
        self.update(is_vw) 
        self.graphs(paths)
        self.unstable()
        self.interact(is_vw)
        if is_vw: # (3) offline report only → during training shown by 1 / 2
            self.summary(is_vw)
        self._process.set_sync_ready(2)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def show(self):
        """Show dashboard - serves reporting purpose of dashboard - for trained, stored models."""
        self._align_dash_result_type()
        if self._is_serialized: 
            DashMixin(self._f)._serialize(1)        
        self._show()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@DREG.register(REG.AIG)
class WoeStabilizerDash(WoeStabilizer):

    def __init__(self, 
        fitter:     Fitter, 
        dash_data:  DashData = None,
        verbose:    int|bool = True,
    ):
        create_at_background = True  # DEBUG → set False 

        args = (fitter, dash_data, verbose)
        super().__init__(*args, not create_at_background)        

        if create_at_background:
            self._constructor(*args)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _constructor(self,
        fitter:     Fitter, 
        dash_data:  DashData = None,
        verbose:    int|bool = True,
    ):  
        args = (fitter.params, fitter.result, dash_data, verbose)    
        if dash_data.dash_instance is None:

            # (a) init state - Fitter.fit - run at the background
            l = len(fitter.result.vars_woe)*4
            s = l*0.045 # s = (train,test) → 2x stability + 2x woe | 20ms graph construct + overhead
            m = f' → Compute {l} Graphs at the Background → Expected time to finish: {int(s)}s'
            
            BackgroundProcess(
                callback    = self._init_set_created, 
                new_process = True,
                daemon      = False,
                verbose     = self._is_gui, 
                message     = DashCreator.M + m, 
                show_timer  = True, 
            ).run( self._init_create, *args )
        
        else:
            # (b) update state - Fitter.update - run directly on created dash_instance
            self._init_set_created( self._init_create(*args) )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _super_fx(self, kind: str) -> Callable:
        parent_fx = getattr(super(), kind)
        def fx(res:tuple):
            args, kwargs = res
            return parent_fx(*args, **kwargs)
        return fx
    
    def _with_sync(self, fx:str, args:tuple = None, kwargs:dict = None):
        func = self._super_fx(fx)
        self._process.run_with_sync(func, 1, *(args or ()), **(kwargs or {}))

    def save(self, *args, **kwargs):
        if self.verbose:
            func = self._super_fx('save')
            return self._process.run_with_result(func, 1, *args, **kwargs)        
        self._with_sync('save', args, kwargs)            

    def build(self, *args, **kwargs):
        self._with_sync('build', args, kwargs)

    def show(self, *args, **kwargs):
        self._with_sync('show', args, kwargs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashCreator:
    """
    Creates dashboard figures and stats for:
    1. woe predictive consistency
    2. woe predictiveness
    """
    
    M = 'Generating WOE Data and Graphs'

    def __init__(self, verbose:bool=True):
        self.verbose = verbose
    
    @staticmethod
    def _get_th_info(data:DashData) -> dict:
        return {
            SCHEMA.th: data.th
        }        
    
    @staticmethod
    def _get_previous_dash_result(data:DashData) -> None|DashResult:
        """Fitter Update provides full dash data instance from previous fit."""
        dash_instance: WoeStabilizerDash|None = data.dash_instance
        if dash_instance is None:
            return None
        dash_result:DashResult = dash_instance._f
        return dash_result
    
    def _is_th_update_only(self, data:DashData) -> bool:
        if self._get_previous_dash_result(data) is None:
            return False
        return data.is_th_only

    @staticmethod
    def title(tx, encoding=False) -> str:
        t = 'Predictiveness' if encoding else 'Predictive Stability'
        return f'{tx} Sample → WOE {t}'.upper()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
            
    def gen_c(self, tx: str, dash_data: DashData, features: list ) -> dict:          
        """
        Generates Consistency Test Graphs for each feature x in X_sorted and overall Dash Figure.
        """
        data:FitStats = dash_data.consistency
                
        if self._is_th_update_only(dash_data):  # (2) Fitter Update
            dash_result = self._get_previous_dash_result(dash_data)
            return FEConsistencyPlot.update_dash(
                data.consistency[tx],           # updated data from fitter
                dash_result.woe_stability[tx]        # previous data from dash instance → update fig titles
            )                       

        return FEConsistencyPlot.dash(         # (1) Fitter Fit → full compute
            data.consistency[tx],
            grid_title  = self.title(tx), 
            features    = features,
            verbose     = False
        )    
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def gen_w(self, tx: str, dash_data: DashData, features: list) -> dict:        
        """Generates WOE Graphs overall Dash Figure by X_sorted."""
        
        dash_result = self._get_previous_dash_result(dash_data)
        if dash_result is not None:              # (2) Fitter Update
            return dash_result.woe_predict[tx]   # → no change in update for woe → direct use of dash instance
                
        is_train = (tx == TT.TRAIN)             # (1) Fitter Fit → first ever compute
        if (
            not is_train and dash_data.test_data_woe is None
        ): 
            return {k:None for k in W_DASH.list()}

        return WoEPlot(
            encoder = dash_data.woe_encoder,    
            apply_data_quality = False, # during training pipeline Train/Test data after quality
            apply_woe_encoding = False  # incoming data_preprocess encoded to WOE already
        ).dash(
            get_trained = is_train,
            data        = dash_data.test_data_woe,   # active only for Test  → Train encoding is loaded
            target      = dash_data.target,          # active only for Test  → Train encoding is loaded
            features    = features,
            grid_title  = self.title(tx, True),
            verbose     = False    
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _create(self, dash_data:DashData) -> dict:
        """Creates dashboard figures and stats for 1. woe consistency and 2. woe predictiveness."""
        test, woe = {}, {}
        o = (None, None)

        for tx in TT.list():
            test[tx] = self.gen_c(tx, dash_data, o[0])
            woe[tx]  = self.gen_w(tx, dash_data, o[1])
            o = (
                test[tx][S_DASH.ORDER], 
                woe[tx][W_DASH.ORDER]
            )
        return {RES.TEST: test, RES.WOE: woe} | self._get_th_info(dash_data)

    def create(self, dash_data:DashData) -> dict:
        """Creates dashboard figures and stats for 1. woe consistency and 2. woe predictiveness."""
        if not self._is_th_update_only(dash_data):
            return self._create(dash_data)
        
        with Spinner('Updating to the new TH').progress(self.verbose):
            return self._create(dash_data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashSummary:
    
    def __init__(self, result:FitResult, verbose:bool):
        self.vars_all = result.vars_woe + result.vars_dropped
        self.vars_ko  = result.vars_dropped
        self.verbose  = verbose
    
    def _get_status(self, out:list=None) -> list:
        if out is None: 
            out = self.vars_ko
        woe = [x for x in self.vars_all if x not in out]
        return len(woe), len(out)       
                 
    def __call__(self, vars_out_user:dict=None, is_view:bool=False):        
        lw, lo = self._get_status(vars_out_user)
        org = lw + lo

        f = lambda x: f"{x:,}"
        m = ' '.join([
            Rgb.bold('► Weight of Evidence Stabilizer →'),
            Rgb.bold_green(f(lw)), 'WoE Features Passed →',
            Rgb.bold_red(f(lo)), 'of', Rgb.bold(f(org)), 'Dropped as Unstable'
        ])
        DashReportLog(
            self.verbose, is_view=is_view, margin_t=10
        ).final_summary(lw, m_base=m)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashMixin:
    
    _WL = {
        LBL.VALUE:   W_LBL.GINI,
        LBL.DATA:    W_LBL.DATA,
        LBL.GRAPH:   W_LBL.GRAPH,
        LBL.GMX:     W_LBL.GMX,
        LBL.VERDICT: W_LBL.VERDICT,
    }
    _WD = { 
        DASH.DATA:  W_DASH.DATA,
        DASH.FIG:   W_DASH.FIG,
        DASH.ORDER: W_DASH.ORDER

    }
    _SL = {
        LBL.VALUE:   S_LBL.VALUE,
        LBL.DATA:    S_LBL.DATA,
        LBL.GRAPH:   S_LBL.GRAPH,
        LBL.GMX:     S_LBL.GMX,
        LBL.VERDICT: S_LBL.VERDICT,
        LBL.IS:      S_LBL.IS,
    }
    _SD = { 
        DASH.DATA:  S_DASH.DATA,
        DASH.FIG:   S_DASH.FIG,
        DASH.ORDER: S_DASH.ORDER

    }

    def __init__(self, result:DashResult):
        self._f = result

    @staticmethod
    def _is_woe(w:bool|str) -> bool:
        match w:
            case str() if w==RES.WOE:
                return True
            case bool() if w:
                return True
        return False      
              
    @staticmethod
    def _map_var_keys(to_get:str, is_woe:bool) -> str:
        if is_woe:
            if to_get == LBL.VALUE:
                return LBL.GINI
        return to_get
    
    @classmethod
    def _get_rx(cls, is_woe:bool=False) -> str:
        return RES.WOE if cls._is_woe(is_woe) else RES.TEST 
    
    @staticmethod
    def _get_tx(tx:str=None) -> str:
        return tx if tx is not None else TT.TRAIN
    
    @classmethod
    def _get_dash_key(cls, to_get:str, is_woe:bool|str=False) -> None|str:
        """Retrieve to_get key from dash labels given is_woe."""        
        if cls._is_woe(is_woe):
            return cls._WD[to_get]        
        return cls._SD[to_get]

    def _get_tx_result(self, to_get:str, tx:str=None, is_woe:bool|str=False) -> dict:
        """Main Result: data-fig-order"""
        dx = self._get_dash_key(to_get)
        tx = self._get_tx(tx)
        rx = self._get_rx(is_woe)
        return self._f[rx][tx][dx]
                
    def _is_empty(self, tx:str=None, is_woe:bool|str=False) -> bool:
        r = self._get_tx_result(DASH.DATA, tx, is_woe)
        return ( r is None ) or ( not r )
        
    @classmethod
    def _get_var(cls, x:str, is_woe:bool|str=False) -> str:
        if cls._is_woe(is_woe): 
            return VarStd.fe_to_var(x) # training + WOE.create_trained_based_encoding design
        return x
        
    @classmethod
    def _get_var_key(cls, to_get:str=None, is_woe:bool|str=False) -> None|str:
        """Retrieve to_get key from labels given is_woe."""
        if to_get is None:
            return None
        is_woe = cls._is_woe(is_woe)
        if is_woe:
            return cls._WL[to_get]        
        return cls._SL[to_get]

    def get_var_data(self, x:str, tx:str=None, to_get:str=None, is_woe:bool|str=False) -> dict|str|float:
        x   = self._get_var(x, is_woe)
        var = self._get_tx_result(DASH.DATA, tx, is_woe)[x]    # x's data of all stats -- s_stat / w_stat
        key = self._get_var_key(to_get, is_woe)
        if key is None: 
            return var
        return var.get(key)
    
    def get_vars(self) -> list:
        return self._get_tx_result(DASH.ORDER)

    def get_vars_out(self) -> list:
        return self._f.vars_dropped
    
    def _serialize(self, to_io_or_from_io:int=0):        
        fx = lambda x: DashInterface._plotly_serialize(x, to_io_or_from_io)        
        for sw in RES.list():
            for tx in TT.list():
                r = self._f[sw][tx]
                k = self._get_dash_key(DASH.FIG, sw)
                r[k] = fx(r[k])                             # dash plotly fig to/from json
                k = self._get_dash_key(DASH.DATA, sw)
                d = r[k]
                for x in d:
                    x  = self._get_var(x, sw)
                    k_ = self._get_var_key(LBL.GRAPH, sw)
                    d[x][k_] = fx(d[x][k_])                 # each plotly graph to/from json
        return True
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashGrid(DashMixin):
    
    _T = {
            RES.TEST: 'WOE ▲ STABILITY', 
            RES.WOE:  'WOE ▼ GINI'
    }

    def __init__(self, result: DashResult, paths: dict[str, dict[str, Path]]):
        super().__init__(result)  
        self.is_load, self.paths = DashIO.resolve_path(paths, 'graphs')

    def _save(self, fig: go.Figure | str, tx:str, sw:str) -> dict:
        path = self.paths[tx][sw]
        return DashPlotlyLink(
            figure_path    = path if self.is_load else fig,
            report_path    = path,
            link_name      = tx.upper(),
            make_clickable = False,
            show           = False
        ).get_output()

    def _dash(self, sw:str):
        data = []
        for tx in TT.list():
            dash = self._get_tx_result(DASH.FIG, tx, sw)
            data.append( self._save(dash, tx, sw) )
        return TableStyle.clickable(data, show=False)    
      
    def create(self, verbose:bool=True):
        data = {}
        for sw in RES.list():
            data[sw] = self._dash(sw)

        return TableStyle.combine_pdf_to_html(
            data, self._T, 
            h_cols_per_row   = 3, 
            h_gap_horizontal = 20, 
            show = verbose
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def notes(self, tx: str) -> None:
        t, n, na = self._status(tx)
        T = tx + Str.align(*TT.list(), with_sx=tx)

        s = Format._sp(t)[0]         

        if not n and t!=na:
            Format.display([
                T, f'► All Features are Statistically Stable by WOE Stability-Test'
            ], style = Format.Style(formatted=40))

        if n:
            Format.display([
                T, f'► {n} of {t} Feature{s} classified Unstable → Recommended for elimination'
            ], style = Format.Style(formatted=40))
            
        if not na:
            return
        
        Format.display([
            T, f'► {na} of {t} Feature{s} classified Unassessable due to lack of available data'
        ], style = Format.Style(formatted=40))
        
        
    def _status(self, tx: str) -> list:
        v = self.get_vars()
        r = [self.get_var_data(x, tx, LBL.IS) for x in v]
        f = lambda l: sum([x==l for x in r])
        s = [len(v)]
        for l in (STAB.N, STAB.NA):
            s.append(f(l))
        return s
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class DashInterBase(DashMixin):

    _H = 310
    _W = 520

    def __init__(self, result:DashResult, vars_out_user:dict|None=None, _is_view:bool=False):
        super().__init__(result)        
        self.vars_out_user = vars_out_user
        self._reset_output = wg.Output()
        self._graph_output = wg.Output()
        self._is_view = _is_view
        self.set_vars_status_placeholder()
        self._prior_cache = {}
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _is_stable(self, x:str, tx:str=None) -> bool:
        return self.get_var_data(x, tx, LBL.IS) == STAB.Y
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _is_not_in_user_out(self, x:str) -> bool:
        return x not in self.vars_out_user
    
    def vars_init_status(self, x:str, reset:bool=False) -> bool:
        if self.vars_out_user is None or reset:
            return self._is_stable(x)       # default stat result
        return self._is_not_in_user_out(x)  # user choices memory

    def set_vars_status_placeholder(self, reset:bool=False) -> None:
        self.variables_status = {}  # wg check buttons placeholder → link to dynamic eliminate
        for x in self.get_vars():
            self.variables_status[x] = self.vars_init_status(x, reset)    
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def get_vars_selection(self) -> dict:
        return self.variables_status
    
    def get_user_out_choices(self) -> dict:
        var = self.get_vars_selection()
        return [x for x,v in var.items() if not v]
    
    def get_var_status(self, x:str) -> bool:
        return self.variables_status[x]
    
    def set_var_status(self, x:str, v:bool) -> None:
        self.variables_status[x] = v
        
    def diff_user_from_default_status(self) -> int:
        s  = self.get_vars_out()
        u  = self.get_user_out_choices()
        d1 = [x for x in s if x not in u]
        d2 = [x for x in u if x not in s]
        return len(d1+d2)               
    # ─────────────────────────────────────────────────────────────────────────────────────

    def _g_title(self, x:str, tx:str, is_woe=False) -> str:
        ver = self.get_var_data(x, tx, LBL.VERDICT, is_woe)
        val = self.get_var_data(x, tx, LBL.VALUE,   is_woe)
        val =  f'{val:.2f}' if ver != STAB.NA else ''
        desc= 'PREDICTION' if self._is_woe(is_woe) else 'STABILITY'
        return tx + ' → ' + desc + ' → ' + f'{ver} {val}'
    
    @classmethod
    def _create_graphs(cls, fes:list, graph_fx, title_fx, extra_layout={}, apply_css:bool=True) -> go.Figure:
        m = {'margin': dict(l=60, r=0, t=20, b=0)}
        update_layout = {**extra_layout, **m}
        update_xaxes  = {'tickfont': dict(family="Arial", size=11, color="black")}        
        fig = Plot.create_subplots(
            fes, 
            graph_fx, 
            title_fx, 
            secondary_y = True,
            grid_cols = 2, 
            height = cls._H, 
            width = cls._W, 
            vertical_spacing = 0.15, 
            horizontal_spacing = 0.15,
            update_layout = update_layout,
            update_xaxes = update_xaxes
        )
        if apply_css:
            return Plot.adjust_to_css(fig)
        return fig
    
    def _g_shared_y(self, x_, is_woe=False) -> list:
        share_yaxis = [0,1]
        for tx in TT.list():
            g = self.get_var_data(x_, tx, LBL.GMX, is_woe)
            if not g:
                continue
            m, x = g

            # min | max set
            if tx == TT.TRAIN:
                share_yaxis = [m, x]
            else:
                if m < share_yaxis[0]: share_yaxis[0] = m
                if x > share_yaxis[1]: share_yaxis[1] = x  
        
        # min | max adj
        m = np.mean(np.abs(share_yaxis))
        for i,v in enumerate(share_yaxis):
            o = -1 if i==0 else 1 
            share_yaxis[i] = v+(o*m*0.1)

        return share_yaxis

    def _get_g_shared_y(self, x, tx, is_woe=False) -> list:
        if tx == TT.TRAIN:
            self._current_shared_y = self._g_shared_y(x, is_woe)
        return self._current_shared_y
    
    def _g_yaxis(self, x, tx, is_woe=False) -> dict:
        yaxis  = {'title': {'text': None}}
        yaxis2 = {'range': self._get_g_shared_y(x, tx, is_woe)}
        if tx == TT.TRAIN:
            return dict(yaxis2={**yaxis, **yaxis2})
        return dict(yaxis=yaxis, yaxis2=yaxis2)  

    def _g_layout(self, x, tx, is_woe=False) -> dict:
        is_woe = self._is_woe(is_woe)
        angle  = 15 if is_woe else 40
        layout = dict(xaxis = {'tickangle': angle})
        return {**layout, **self._g_yaxis(x, tx, is_woe)}

    @lru_cache(maxsize=CACHE)
    def create_graph(self, x:str, apply_css:bool=True) -> go.Figure:
        gf, gt = {}, {}                        
        X = [(tx, sw, x+tx+sw) for sw in RES.list()[::-1] for tx in TT.list()]

        for tx, sw, x_ in X:    
            args   = ( x, tx, sw )
            gt[x_] = self._g_title(*args)
            update = self._g_layout(*args)
            gf[x_] = self.get_var_data(x, tx, LBL.GRAPH, sw)
            gf[x_].update_layout(update)

        graph_fx = lambda x: gf[x[-1]]
        title_fx = lambda x: gt[x[-1]]
        b_layout = dict(barmode = 'stack', bargap = 0.0)
        # ► NOTE ◄
        # plotly's barmode and bargap is a global figure feature
        # i.e. under same fig (make_subplots) we cannot combine different barmode/bargap layout        
        return self._create_graphs(X, graph_fx, title_fx, b_layout, apply_css = apply_css)

    def update_graph(self, x:str) -> None:
        g = self._prior_cache.get(x)
        if g is None:
            g = self.create_graph(x)
        with self._graph_output:                
            clear_output(wait=True)                
            display(g)
    # ─────────────────────────────────────────────────────────────────────────────────────
        
    @classmethod
    def _create_cache(cls, instance:"DashInterBase", dash_result:dict, fes:list[str]) -> dict:
        instance._f = DashResult(**dash_result) # Critical: recreate inside the new process
        css = Plot.get_css_classes()
        
        ha  = f'height:{int(cls._H*2 + 32)}px'  # pio doesn't respect specified h-w
        wa  = f'width:{int(cls._W*2  + 30)}px'
        
        cache = {}
        for x in fes[:CACHE]:
            g = instance.create_graph(x, apply_css = False)
            g = Plot.adjust_to_css(g, add_class = False)
            h = pio.to_html(
                g, 
                include_plotlyjs = False, 
                full_html        = False,
                validate         = False,
            ).replace(
                f'<div style="height:{int(cls._H*2)}px; width:{int(cls._W*2)}px', 
                f'<div style="height:{ha}; width:auto', 1
            ) 
            s = (f'style="{ha} !important; {wa} !important;"')
            h = h.replace(
                'class="plotly-graph-div"', 
                f'class="plotly-graph-div {css[0]} {css[1]}" {s}'
            )                
            cache[x] = HTML(h)
        return cache

    def _set_cache_graphs(self, cache:dict) -> None:
        self._prior_cache = cache

    def cache_graphs(self, fes:list) -> None:
        instance = object.__new__(type(self))
        d_result = dict(self._f) # Critical: send plain dict → new process would destroy data 
        m = f'Caching interactive grid for the first {CACHE} graphs → Expected time to finish: {int(CACHE*0.4)}s'
        
        BackgroundProcess(
            callback    = self._set_cache_graphs, 
            new_process = True,
            verbose     = True,
            show_timer  = True,
            message     = m
        ).run(self._create_cache, instance, d_result, fes)
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def create_feature(self, x:str, vars_all:list):
        layout   = wg.Layout(width='auto', height='30px')
        self.ui_feature:wg.Dropdown = wg.Dropdown(            
            value   = x,
            options = vars_all,
            layout  = layout,
            description = 'WOE:   '
        )        

    def get_feature(self) -> str:
        return self.ui_feature.value     
    # ─────────────────────────────────────────────────────────────────────────────────────       
        
    def _check_desc(self, x:str):
        th  = self._f.th
        ST  = [self.get_var_data(x, tx, LBL.VALUE, False) for tx in TT.list()]
        GT  = [self.get_var_data(x, tx, LBL.VALUE,  True) for tx in TT.list()]
        aig = 'AIG RECOMMENDATION →'
        dft = lambda x: f' {x} '
        sgx = f'→ Gini: {GT[0]:.2f} | {GT[1]:.2f} → Stability: {ST[0]:.2f} | {ST[1]:.2f}'
        if   all([t > th for t in ST]): desc = f'{aig} {dft('KEEP')} {sgx}'
        elif all([t<= th for t in ST]): desc = f'{aig} {dft('DROP')} {sgx}'
        elif all([t < 0  for t in ST]): desc = f'{aig} {dft('DROP')} Not Computable'
        else:  desc = f'NA → Samples have different Verdict {sgx}'
        return desc  
    
    def create_check(self, x:str) -> None:
        layout = wg.Layout(width='auto', height='30px')
        self.ui_check:wg.Checkbox = wg.Checkbox(
                value       = self.get_var_status(x),
                description = self._check_desc(x),
                disabled    = self._is_view,
                indent      = False, 
                layout      = layout
        ).add_class("left-spacing-class")
        display(HTML("<style>.left-spacing-class {margin-left: 30px;}</style>"))
    
    def get_check(self) -> bool:
        return self.ui_check.value    

    def update_check(self, x) -> None:
        self.ui_check.value = self.get_var_status(x)
        self.ui_check.description = self._check_desc(x)  
    # ─────────────────────────────────────────────────────────────────────────────────────

    def _reset_text(self, diff:int=None):
        if diff is None:
              t = '► Propagate reset of Custom changes to stored WOE Stability function and train data by Custom Update'
        else: t = f'► Custom Selection Propagated ► Difference compared to set TH {self._f.th} ► ({Rgb.bold_red(diff)})'
        layout = wg.Layout(width='auto', height='30px', margin='0px 7px 2px 7px')
        return wg.HTML(t, layout=layout)
    
    def _reset_status(self, change) -> None:        
        self.set_vars_status_placeholder(True)
        self.update_check(self.get_feature())
        with self._reset_output:
            clear_output(wait=True)
            display(self._reset_text())
    
    def create_reset_button(self):
        layout = wg.Layout(width='35px', height='30px', margin='0px 7px 2px 7px')
        self.ui_reset:wg.Button = wg.Button(
            layout = layout,
            icon = 'trash',
            tooltip = 'RESET CUSTOM CHANGES',
            button_style='danger'
        )
        self.ui_reset.on_click(self._reset_status) 
    
    def reset_button(self) -> None:
        d = self.diff_user_from_default_status()
        if not d or self._is_view:
            return
        display(self._reset_output)
        b = wg.HBox([self._reset_text(d), self.ui_reset])
        with self._reset_output:
            clear_output(wait=True)
            display(b)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class DashInteract(DashInterBase):
    
    def __init__(self, result:DashResult, vars_out_user:list = None, _is_view:bool = False):
        super().__init__(result, vars_out_user, _is_view)
        self.create_reset_button()
        
    def create(self):                    
                
        var_order = self.get_vars()  # we should show first the least stable variables
        var_init  = var_order[0]           

        # Exo Control Panel
        self.create_feature(var_init, var_order)
        self.create_check(var_init)          

        # Dynamics
        def redraw_ui(change):            
            x = self.get_feature()
            self.update_check(x)                        
            self.update_graph(x)

        def change_var_status(change):
            x = self.get_feature()
            v = self.get_check()
            self.set_var_status(x, v)

        self.ui_feature.observe(redraw_ui)
        self.ui_check.observe(change_var_status)

        # UI
        f_box = wg.HBox([self.ui_feature, self.ui_check], layout=wg.Layout(width='auto',height='60px'))
        g_box = lambda: self.update_graph(var_init)

        t = 'WOE FEATURES CUSTOM SELECTION'
        t = Format.title(t, style=Format.StyleStats(margin_t=30, margin_b=20), show=False) 
        d = [t, f_box, self._graph_output, g_box]
        Display.construct(d)

        # Extra Cache
        with self._graph_output:
            self.cache_graphs(var_order)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashUnstable(DashMixin):

    def __init__(self, result:DashResult):
        super().__init__(result)  

    def _data(self, x:str, tx:str) -> dict:
        return {
            UNS.FE:  x, 
            UNS.TEST: self.get_var_data(x, tx, LBL.VALUE),
            UNS.STAB: self.get_var_data(x, tx, LBL.IS)
        }
        
    def _un(self, tx:str) -> pDF:
        o = self.get_vars()
        r = [self._data(x, tx) for x in o]
        r = pDF(r)
        return r[r[UNS.STAB] == STAB.N]
    
    def create(self, limit=20, verbose:bool=False) -> None:  
        """WOE Unstable Predictive Behavior."""                
        if self._is_empty():
            return 
        
        dun = {tx: self._un(tx) for tx in TT.list()}
        lun = {tx: len(u) for tx,u in dun.items() if len(u)}
        if not any(lun.values()):
            return
                
        note = lambda f: f'{f}WOE UNSTABLE FEATURES'
        
        def _note(ltt, f = ''):            
            if ltt > limit: f = f + f'TOP{limit} '         
            Format.title(note(f), style=Format.StyleStats(margin_t=34))
        
        def _style(d:pDF):
            return (Pd.style(
                d.head(limit).reset_index(),
                hide_axis=0, bold_cols=UNS.FE, format_cols={UNS.TEST:'{:.3f}'}
            ))

        d = [_style(dun[tx]) for tx in lun]
        
        if not verbose:
            return
        
        _note(max(lun.values()))
        TableStyle.combine_pdf_to_html(d, list(lun), tables_per_block = 1, h_cols_per_row = 2)     
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬