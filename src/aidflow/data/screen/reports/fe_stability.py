"""
Standard Report Interface registered with the DashRegistry for a specified MODEL.

This interface serves as the bridge between the Model's raw output and the 
Dashboard visualization layer, ensuring consistent reporting schemas across 
different model versions.
"""

from ....deps.common            import  (
                                        Path,
                                        contextmanager
                                        )
from ....deps.data              import  np, pd, pDF, pSe
from ....deps.plots             import  go
from ....deps.display           import  (
                                        wg, 
                                        HTML, 
                                        display, 
                                        clear_output
                                        )
from ....core.parameters        import  AIGFLOW
from ....core.registry          import  DashRegister
from ....core.factory           import  DictClass
from ....core.foundation        import  AigParam
from ....core.execution         import  BackgroundProcess
from ....core.display           import  (
                                        DashInterface, 
                                        DashReportLog,
                                        DashPlotlyLink,
                                        Display
                                        )
from ....core.io                import  DashIO
from ....core.interop           import  Pd, GetValue
from ....core.accessor          import  UI, Rgb, Format

from ....utils.builtins         import  Str
from ....display.plots          import  Plot
from ....display.style          import  TableStyle

from ..fitters.fe_analyzer      import  TimeStationarity
from ..fe_stabilizer            import  (
                                        REG, 
                                        Fitter, 
                                        FitResult, 
                                        TAResult
                                        )

MODEL = AIGFLOW.Data.fe_stabilizer
DREG  = DashRegister(MODEL)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    """var-ur-stab-type-graph-tt labels"""
    VAR   = 'Feature'
    UR    = 'DF-Test'
    STAB  = 'Stable'
    GRAPH = 'Graph_Name'
    TT    = ('TRAIN', 'TEST')

COL = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → TEMPORAL STABILITY → GRAPHS & DASHBOARD
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@DREG.register(REG.AIG)
class FeatureStabilizerDash(DashInterface):
    """Dash Fitter UI."""       

    def __init__(self, 
        fitter:     Fitter, 
        verbose:    int|bool = True, 
        ipy_header: dict = None
    ):
        super().__init__(fitter, verbose)

        self._p  = fitter.params
        self._f  = fitter.result

        self._is_target = self._p.is_target

        self._set_dash_creator(ipy_header)
        self._summary = DashSummary(verbose, self._f)

        self._set_elements({
            'graphs': {
                tx: f'dash_grid_{tx}'.lower() for tx in LBL.TT
            }
        })
        self.dashio = DashIO(self.setup)
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

    def _update_vars_out_user(self) -> None:
        """Update dash state by Training dynamic User GUI confirmed changes and save it."""              
        self._interactive.reset_button()
        self._summary(self._vars_out_user)
        self._save_vars_out_user()
        
    def _get_vars_out_user(self) -> dict|None:
        """Memory of Updated vars_user during Training GUI."""
        if not self._is_interactive():
            return self._load_vars_out_user()     # load memory
        return self._vars_out_user                # active gui

    def _is_user_vars_note(self) -> bool:
        d = self._load_vars_out_user()
        if self._is_interactive():
            d = self._interactive.diff_user_from_default_status()        
        return bool(d)
    
    def _is_interactive(self) -> bool:
        return hasattr(self, '_interactive')    
    # ─────────────────────────────────────────────────────────────────────────────────────────────        
    
    def _get_ipy_header(self):
        """Provides stored notebook's cell id origin to allow dash update by widget."""
        return self._ipy_header
    
    def _set_exo_args(self, ipy_header:dict=None):
        """Stores notebook's cell id origin to allow dash update by widget."""
        self._ipy_header = None
        if self._is_target:
            return
        if ipy_header is None:
            self._ipy_header = BackgroundProcess()._get_ipython_header()
            self._is_update  = False
        else: 
            self._ipy_header = ipy_header
            self._is_update  = True
    
    def _set_dash_creator(self, ipy_header:dict=None):
        self._set_exo_args(ipy_header)
        if self._is_target:
            self._gn = DashTarget(self.verbose, self._f)
        else: 
            self._gn = DashExoGrid(self._f, self._ipy_header, self._is_update)
    
    def _set_interactive(self, _is_view:bool):
        self._interactive = DashInteract(
            self._f, self._get_vars_out_user(), _is_view
        ) # → memory of user ui chages

    @property
    def _is_not_exo_active(self):
        return not (self.verbose and not self._is_target)        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    @contextmanager
    def _with_temporal_target_view(self):
        gn = self._gn
        it = self._is_target        
        try:
            self._gn = self.dashio.load(MODEL, file_name='target')._gn
            self._is_target = True
            yield
        finally:
            self._gn = gn
            self._is_target = it
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def title(self):
        if self._is_any(
            self._is_not_exo_active,
            not self._is_gui
        ):  return  
        t = f' Exogenous Features Dashboard | TH {self._f.th}'.upper()
        Format.title(t.upper(), style=Format.StyleStats())
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def update(self, _is_view=None):
        if self._is_any(
            self._is_not_exo_active,
            _is_view, # report view
            not self._is_user_vars_note()
        ): return
        self._info_highlight(
            'CUSTOM SELECTION ACTIVE ► Feature status reflects a prior custom selection '
            f'► Report provides view by current threshold {self._f.th} '
            f'► Use Reset to restore TH-based selection',
            is_view = _is_view
        )        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def graphs(self, paths: dict[str, dict[str| Path]]):                
        if not self._is_gui:
            return
        self._gn.graphs(paths)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def notes(self):
        if not self._is_gui:
            return               
        for tx in LBL.TT:
            self._gn.notes(tx)
            if self._is_target:
                return
        if isinstance(self._gn, (DashExo, DashExoGrid)):
            self._gn.notes2()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def unstable(self): 
        if self._is_any(
            not self._is_gui,
            self._is_not_exo_active,
        ):  return        
        if isinstance(self._gn, (DashExo, DashExoGrid)):
            self._gn.unstable(limit=10)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def interact(self, _is_view=None):
        if self._is_any(
            not self._is_gui,
            self._is_not_exo_active,
        ):  return        
        self._set_interactive(_is_view)
        self._interactive.create()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def summary(self, _is_view=True):
        if self._is_any(
            self._is_not_exo_active,
            self._is_gui and (not _is_view), # not report  → interactive _save_vars_out_user
        ):  return
        self._summary( self._get_vars_out_user(), _is_view )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def save(self, save=None, custom_path=None):        
        """Saves dashboard - serves reporting purpose of dashboard - for trained, stored models."""
        name = 'target' if self._is_target else None
        return self.dashio.save(
            self, save, custom_path, report_name = name, save_signature = not self._is_target
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
        self.notes()
        self.unstable()
        self.interact(is_vw)        
        self.summary(is_vw)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def show(self, is_target:bool = False):
        """Show dashboard - serves reporting purpose of dashboard - for trained, stored models."""
        if not is_target:
            self._show()
            return
        with self._with_temporal_target_view():
            self._show()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashMixin:

    def __init__(self, result:FitResult):
        self._f = result
    
    def _get_tx(self, tx) -> str:
        if tx is None: 
            return LBL.TT[0]
        return tx
        
    def _get_tx_result(self, tx=None) -> TAResult:
        tx = self._get_tx(tx)
        return self._f[tx]    
    
    def _ts(self, tx=None) -> pDF:
        """Temporal Stats of all Vars."""
        return self._get_tx_result(tx).ts_stat
    
    def _td(self, tx=None) -> pDF:
        """Temporal Data of all Vars."""
        return self._get_tx_result(tx).ts_data

    def _un(self, tx=None) -> pDF:
        """Temporal Unstable Vars."""
        return self._get_tx_result(tx).unstable

    def _na(self, tx=None) -> pDF:        
        """Temporal Unassessable (NaN) Vars."""
        un = self._un(tx)
        return TimeStationarity.get_status_nan(un)
    
    def _is_empty(self, tx=None) -> bool:
        return self._ts(tx).empty
        
    def get_vars(self) -> pSe:
        return self._ts()[LBL.VAR]
    
    def _get_train_ts(self, ts:pDF|None = None) -> pDF:
        if ts is None:
            ts = self._ts()
        return ts        
    
    def get_vars_out(self, ts:pDF|None=None) -> list:
        ts = self._get_train_ts(ts)
        return TimeStationarity.get_status_unstable(ts)

    def get_vars_num(self, ts:pDF|None=None) -> list:
        ts = self._get_train_ts(ts)
        return TimeStationarity.get_num_vars(ts)
    
    def get_vars_cat(self, ts:pDF|None=None) -> list:
        ts = self._get_train_ts(ts)
        return TimeStationarity.get_cat_vars(ts)    
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def create_scatter_graph(
        y, x, 
        line_color:str = None, 
        xaxis:dict = dict(type='date', tickformat='%Y-%m'), 
        yaxis:dict = dict(hoverformat='.4f'),
        **kwargs
    ) -> go.Figure: 
        """Scatter TS default setting."""
        line = {'width': 1}
        if line_color: 
            line = line | {'color': line_color}
        data = dict(y = y, x = x, line = line)
        fig  = go.Figure(
            go.Scatter(**data, **kwargs), 
            layout=go.Layout(xaxis=xaxis, yaxis=yaxis)
        ) 
        fig.update_traces(
           hovertemplate='%{x|%Y-%m-%d} → %{y}'
        )   
        return fig
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashSummary(DashMixin):
    
    def __init__(self, verbose:bool, result:FitResult):
        super().__init__(result)
        self.verbose = verbose

    def _get_org_status(self):
        ts  = self._ts()
        out_org = self.get_vars_out(ts)
        num_org = self.get_vars_num(ts)
        cat_org = self.get_vars_cat(ts)
        return num_org, cat_org, out_org
    
    def _get_in_num_cat(self, out:list=None) -> tuple[list, list]:
        num_org, cat_org, out_ = self._get_org_status()
        if out is None: 
            out = out_
        num_in  = [x for x in num_org if x not in out]
        cat_in  = [x for x in cat_org if x not in out]
        return num_in, cat_in
    
    def _summary_stats(self, vars_out_user:dict|None) -> dict:
        num_org, cat_org, out = self._get_org_status()
        if vars_out_user is not None:
            out = vars_out_user
        num_in, cat_in = self._get_in_num_cat(out)        
        return (
            len(num_org), len(cat_org), 
            len(num_in),  len(cat_in), len(out)
        )
                 
    def __call__(self, vars_out_user:dict = None, is_view:bool = False):        
        ln_org, lc_org, ln, lc, lo = self._summary_stats(vars_out_user)

        f   = lambda x: f"{x:,}"
        np  = Rgb.bold_green(f(ln)) + ' Numerical'
        cp  = Rgb.bold_green(f(lc)) + ' Categorical'
        div = (' | ' if ln and lc else '')
        nd  = Rgb.bold_red(f(ln_org-ln)) + f' of {Rgb.bold(f(ln_org))} Numerical'
        cd  = Rgb.bold_red(f(lc_org-lc)) + f' of {Rgb.bold(f(lc_org))} Categorical'

        l = lambda x,l: x if l else ''

        m = ' '.join([
            Rgb.bold('► Feature Stabilizer →'),
            l(np, ln) + div + l(cp, lc), 'Features Passed →',
            l(nd, ln) + div + l(cd, lc), 'Dropped as Non-Stationary'
        ])
        th_ = f'given chosen TH {self._f.th}'
        m_all_out = f'due to Temporal Instability {th_}'

        DashReportLog(self.verbose, is_view = is_view, margin_t = 10).final_summary(
            num = ln, 
            cat = lc, 
            m_base = m, 
            m_all_out = m_all_out,
        )    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashExo(DashMixin):
    """Exogenous Graphs-Notes."""

    def __init__(self, result: FitResult):
        super().__init__(result)
    
    def graphs(self, tx:str, report_path=None, cols:int=4, verbose:bool=False) -> pDF:
        """Overview of Time & Dashboard for All Vars as Time Series."""                        
        if self._is_empty(tx):
            return 

        fes = self._ts(tx)[LBL.VAR].values          # vars sorted by UR
        ix  = self._td(tx).index                    # timeline same for all vars        
        titles = self._ts(tx).set_index(LBL.VAR)    # graph title in ts stats

        grid_title = f'{tx} Sample → Temporal Stability'.upper()

        dtick = self._dtick(ix)

        def graph(x):
            return self.create_scatter_graph(
                self._td(tx)[x], ix, 
                xaxis = dict(type='date', tickformat='%Y-%m-%d', dtick=dtick), 
                text  = 'TS', 
                name  = 'TS'
            )
                
        def title(x):
            return titles.at[x, LBL.GRAPH]

        fig = Plot.create_subplots(
                fes,
                graph_fx    = graph,
                title_fx    = title,
                height      = 300,
                width       = 450,
                grid_cols   = cols,
                grid_title  = grid_title,
        )
        return self._dash_link(fig, report_path, verbose)
    
    @staticmethod
    def _dtick(ix: pSe, ticks: int = 15) -> int:
        try:
            span_ms = (
                pd.to_datetime(ix.max()) - pd.to_datetime(ix.min())
            ).total_seconds() * 1000
            raw = int(span_ms / ticks)
        except:
            raw = 604_800_000

        ladder = [
            3_600_000,      # 1h
            86_400_000,     # 1d
            604_800_000,    # 1w
        ]
        for t in ladder:
            if t >= raw:
                return t
        return raw
    
    def _dash_link(self, figure_path:go.Figure|str, report_path:str = None, verbose:bool = True):
        return DashPlotlyLink(
            figure_path    = figure_path,      
            report_path    = report_path,
            make_clickable = False,
            show           = verbose
        ).get_output()        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def notes(self, tx: str) -> None:        
        if self._is_empty(tx):
            return
        
        td_l  = len(self._td(tx).columns)
        na_l  = len(self._na(tx))
        un_l  = len(self._un(tx)) - na_l # Unstable contains both N and NaN
        na_ko = (na_l == td_l)
        T = tx + Str.align(*LBL.TT, with_sx=tx)
        
        if not (un_l or na_ko) and not na_l:
            Format.display([
                T, '► All Features are Statistically Stable by Stationarity ADF-Test'
            ], style = Format.Style(formatted=40))
            return

        if na_ko:
            bold = Rgb.bold_red if T == LBL.TT[0] else (lambda x: x)
            Format.display([bold(T), bold(
                '► Warning All Features have insufficient length of populated Time Units '
                '→ Recommended to Stop and Revise the Input Data'
            )], style = Format.Style(formatted=40))
            return

        e = '→ Recommended for elimination'
        s = Format._sp(td_l)[0]         

        if un_l:
            Format.display([
                T, f'► {un_l} of {td_l} Feature{s} classified Unstable {e}'
            ], style = Format.Style(formatted=40))
            
        if not na_l:
            return
        
        Format.display([
            T, f' ► {na_l} of {td_l} Feature{s} Unassessable due to sparsity → Total series '
            f'length {len(self._td(tx))} is sufficient but populated part is less than '
            f'{TimeStationarity.MIN_LENGTH} {e}'
        ], style = Format.Style(formatted=40))
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def notes2(self) -> None:
        if self._is_empty():
            return
        if any([len(self._na(tx)) for tx in LBL.TT]):
            Format.display([
                'NOTE', f'► {TimeStationarity.NA} = Unassessable → Features with populated '
                f'Time Units < {TimeStationarity.MIN_LENGTH} cannot be reliably tested '
                '→ Data-sufficiency not met → Recommended for elimination'                
            ], style = Format.Style(formatted=40))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def unstable(self, limit=20, verbose:bool=True) -> None:
        if self._is_empty():
            return 
        
        lun = {T: len(self._un(T)) for T in LBL.TT}
        if not any(lun.values()):
            return
        
        un = [t for t,l in lun.items() if l]
        na = [len(self._na(tx)) for tx in un]
        
        def title():
            ltt = max(lun.values())
            f,u = ['']*2

            if ltt > limit: 
                f = f + f'TOP{limit} '

            if sum(na):
                u = ' / UNASSESSABLE'

            note = f'{f}UNSTABLE{u} EXOGENOUS FEATURES'
            Format.title(note, style=Format.StyleStats())
        
        def _style(d:pDF):
            return (Pd.style(
                d.head(limit).reset_index(),
                hide_axis=0, bold_cols=LBL.VAR, format_cols={LBL.UR:'{:.3f}'}
            ))

        d  = [_style(self._un(tx)) for tx in un]
        
        if not verbose:
            return
        
        title()
        TableStyle.combine_pdf_to_html(d, un, tables_per_block = 1, h_cols_per_row = 2)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class DashExoGrid:

    def __init__(self, result:FitResult, ipy_header:dict, is_update:bool):
        self.dash_exo   = DashExo(result)
        self.ipy_header = ipy_header
        self.is_update  = is_update
    
    def __getattr__(self, name):
        if name.startswith('__'): # copy.copy recursion fix
            raise AttributeError(name)        
        return getattr(self.dash_exo, name)
   
    @classmethod
    def _construct(cls, worker:DashExo, paths:dict, is_load:bool=False) -> list[pDF]:
        links = []
        for tx in LBL.TT:
            path = paths[tx]
            if is_load:
                link = worker._dash_link(path, verbose=False)
            else:
                link = worker.graphs(tx, path)
            if link is None: 
                continue
            links.append(link)
        return links

    def _grid(self, links: list[pDF]):        
        links_ = [TableStyle.clickable(x, show=False) for x in links]
        self.grid = TableStyle.combine_pdf_to_html(
            links_, LBL.TT, tables_per_block=1, h_cols_per_row=2, h_gap_horizontal=10,
        )
        
    def _create_graphs(self, paths: dict[str, dict[str| Path]]):
        m = 'Dashboard Grid Graphs'
        m = f'Rebuilding {m} to the New TH' if self.is_update else f'Constructing {m}'
        BackgroundProcess(
            callback    = self._grid,
            new_process = True,
            verbose     = True,
            message     = m,
            show_timer  = True, 
            ipy_header  = self.ipy_header
        ).run(self._construct, self.dash_exo, paths)

    def graphs(self, paths: dict[str, dict[str| Path]]):
        is_load, paths_ = DashIO.resolve_path(paths, 'graphs')

        if not is_load:
            self._create_graphs(paths_)
            return
        
        links = self._construct(self.dash_exo, paths_, is_load=True)
        self._grid(links)        

    def get_result(self):
        return self.grid
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashTarget(DashMixin):
    """Target Graphs-Notes."""
    
    def __init__(self, verbose:bool, result:FitResult):
        super().__init__(result)
        self.verbose = verbose
    
    def graphs(self, paths: dict = None) -> go.Figure:                        
        if self._is_empty(): 
            return 
        
        gt = {tx: self._ts(tx)[LBL.GRAPH].iloc[0] for tx in LBL.TT}
        dt = lambda t: f'{t} → TARGET STABILITY'        
        cl = lambda t: COL[1] if 'UNSTABLE' in gt[t] else COL[0]        
        tt = lambda t: f'{t} Target Temporal Stability'        

        def graph(T):
            return DashExo.create_scatter_graph(
                self._f[T].ts_data.squeeze(), 
                self._f[T].ts_data.index, 
                text = tt(T), 
                name = tt(T),
                line_color = cl(T)
            )
                
        def title(T):
            return dt(T) +  f' → {gt[T].split(' → ')[-1]}'

        grid_update = dict(margin= {'t': 40,'r':10,'l':10,'b':60})
        grid_axes = dict(gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.5)')

        fig = Plot.create_subplots(
            LBL.TT,
            graph_fx      = graph,
            title_fx      = title,     
            shared_yaxes  = True, 
            height        = 370,           
            width         = 530,           
            grid_cols     = 2,                
            update_layout = grid_update,
            update_xaxes  = grid_axes,
            update_yaxes  = grid_axes,
            horizontal_spacing = 0.02,
            annotations_font = dict(family="Arial", size=12, color="black")
        )    
        
        fig = Plot.adjust_to_css(fig)                
        display(fig)

        return fig
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def notes(self, _) -> None:
        if self._is_empty(): 
            return 
        
        un = [tx for tx in LBL.TT if len(self._un(tx))]   
        na = [tx for tx in LBL.TT if len(self._na(tx))]   
        ln = len(na)
        lu = len(un) - ln # Unstable contains both N and NaN
        
        di = lambda m,b: DashReportLog(self.verbose).final_summary(m=m, b=b)

        if not (lu or ln):
            m = f'► TRAIN and TEST Target Variable is Stationary by ADF-Test on TH {self._f.th}'
            di(m=m, b="#9AEECFD7")
            return
        
        s = 'Non-Stationary'
        u = 'Unassessable'

        # Both events on different samples
        if lu == 1 and ln == 1:                    
            mn = f'{un[0]} Target Variable is {s}'.upper()
            ma = f'{na[0]} Target Variable is {u}'.upper()
            f  = lambda x,y: f'► {x} | {y}'
            if LBL.TT[0] in un:
                m = f(mn, ma)
            else:
                m = f(ma, mn)

        # Either event on one sample
        elif lu == 1 or ln == 1: 
            n, x = un, s
            if ln == 1:
                n, x = na, u                    
            m = f'► {n[0]} Target Variable is {x}'.upper()

        # One event on both samples
        else:
            x = s if lu == 2 else u
            m = f'► TRAIN AND TEST TARGET VARIABLE IS {x}'.upper()
            e = True
        
        di(m=m, b="#E688A9BF")

        Format.display(
        Rgb.bold('\nGood Practice:')+
        '\n1. Revise the input data and/or the modeling time window to ensure sufficient coverage.'
        '\n2. Check whether the target is observation-lagged - outcome visible only after a delay from the reference event.'
        '\n3. Run the full AIG flow to verify whether the temporal instability is shared with explanatory features.'
        '\n4. Compare feature stability patterns against the target break to identify the common root cause.'
        '\n5. Once the driving factor is clear, revise data quality, aggregation level, or window length.')
        Format.splitter()        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashInterBase(DashMixin):
    
    def __init__(self, result:FitResult, vars_out_user:dict|None=None, _is_view:bool=False):
        super().__init__(result)        
        self.vars_out_user = vars_out_user
        self._reset_output = wg.Output()
        self._is_view = _is_view
        self.set_vars_status_placeholder()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _is_stationary(self, x:str, tx:str=None) -> bool:
        return TimeStationarity.is_stationary(self.get_ur(x, tx), self._f.th)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _is_not_in_user_out(self, x:str) -> bool:
        return x not in self.vars_out_user
    
    def vars_init_status(self, x:str, reset:bool=False) -> bool:
        if self.vars_out_user is None or reset:
            return self._is_stationary(x)   # default stat result
        return self._is_not_in_user_out(x)  # user choices memory

    def set_vars_status_placeholder(self, reset:bool=False) -> None:
        self.variables_status = {}  # wg check buttons placeholder → link to dynamic eliminate
        for x in self.get_vars():
            self.variables_status[x] = self.vars_init_status(x, reset)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
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

    @GetValue.at_x_of_col(LBL.VAR)
    def get_var_data(self, stat:pDF, x:str, to_get:str=None):
        ...

    def get_ur(self, x:str, tx=None) -> float: 
        tx = self._get_tx(tx)
        return self.get_var_data(self._ts(tx), x, LBL.UR)   
    # ─────────────────────────────────────────────────────────────────────────────────────

    def _g_shared_y(self, x_):        
        for tx in LBL.TT:
            if self._is_empty(tx):
                continue
            v = self._td(tx)[x_]
            n = v.notnull().sum()>0
            m = v.min() if n else 0
            x = v.max() if n else 0

            # min | max set
            if tx == LBL.TT[0]: 
                share_yaxis = [m, x]
            else: 
                if m < share_yaxis[0]: share_yaxis[0] = m
                if x > share_yaxis[1]: share_yaxis[1] = x  
        
        # min | max adj
        m = np.mean(np.abs(share_yaxis))
        for i,v in enumerate(share_yaxis):
            o = -1 if i==0 else 1 
            share_yaxis[i] = v+(o*m*0.05)

        return share_yaxis

    def _get_g_shared_y(self, tx, x):
        if tx == LBL.TT[0]:
            self._current_shared_y = self._g_shared_y(x)
        return self._current_shared_y
    
    def _g_title(self, tx, x):
        desc  = self.get_var_data(self._ts(tx), x, LBL.GRAPH)
        title = {
            'text': tx + ' → ' + desc,
            'y': 0.99, 'x': 0.1, 'xanchor': 'left', 'yanchor': 'top',
            'font': dict(family="Arial", size=12, color="black")
        }
        return title
    
    def _g_layout(self, tx, x):
        width  = 520
        height = 320
        margin = {'t': 40,'l':0,'r':30,'b':40}
        grid_axes = dict(gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.5)')
        return dict(
            title = self._g_title(tx, x), 
            margin=margin, width=width, height=height, autosize=False,
            yaxis={'range': self._get_g_shared_y(tx, x), 'hoverformat':'.4f', **grid_axes}, 
            xaxis={'type':'date', 'tickformat':'%Y-%m', 'tickangle':0, **grid_axes}           
        )
    
    def _g_color(self, tx:str, x:str) -> str:
        if self._is_stationary(x, tx):
            return COL[0]
        return COL[1]

    def create_graph(self, x:str) -> dict[str, go.Figure]:
        self.ui_graphs = {}
        for tx in LBL.TT:
            if self._is_empty(tx):
                continue
            self.ui_graphs[tx] = go.FigureWidget(
                data = DashExo.create_scatter_graph(
                            y = self._td(tx)[x], 
                            x = self._td(tx).index, 
                            line_color = self._g_color(tx, x),
                            name='', text=''
                        ).data,
                layout = go.Layout(self._g_layout(tx, x))
            )        

    def update_graph(self, x:str) -> None:
        u = {}
        for tx in LBL.TT:
            if self._is_empty(tx):
                continue
            l = self._g_layout(tx, x)
            u[tx] = dict(
                y = self._td(tx)[x], 
                c = self._g_color(tx, x), 
                t = l['title']['text'],
                r = l['yaxis']['range']
            )
        for tx in u:
            with self.ui_graphs[tx].batch_update():
                self.ui_graphs[tx].data[0].y          = u[tx]['y']
                self.ui_graphs[tx].data[0].line.color = u[tx]['c']
                self.ui_graphs[tx].layout.title.text  = u[tx]['t']
                self.ui_graphs[tx].layout.yaxis.range = u[tx]['r']

    def get_graphs(self) -> list:
        return [Plot.adjust_to_css(g, add_class=False) for g in self.ui_graphs.values()]
    # ─────────────────────────────────────────────────────────────────────────────────────

    def create_feature(self, x:str, vars_all:list) -> None:
        layout = wg.Layout(width='auto', height='30px' )
        self.ui_feature:wg.Dropdown = wg.Dropdown(            
            value   = x,
            options = list(vars_all),
            layout  = layout,
            description = 'FEATURE:   ',
        )

    def get_feature(self) -> str:
        return self.ui_feature.value
    # ─────────────────────────────────────────────────────────────────────────────────────
    
    def _check_desc(self, x:str) -> str:
        th  = self._f.th
        URx = [self.get_ur(x, tx) if not self._is_empty(tx) else self.get_ur(x) for tx in LBL.TT]
        aig = 'AIG RECOMMENDATION →'
        dft = lambda x: f'{x} → DF-Tests →'        
        urx = f'{round(URx[0],2)} | {round(URx[1],2)}'
        if   all([0 <= u < th for u in URx]): desc = f'{aig} {dft('KEEP')} {urx}'
        elif all([     u > th for u in URx]): desc = f'{aig} {dft('DROP')} {urx}'
        elif all([     u < 0  for u in URx]): desc = f'{aig} {dft('DROP')} Not Computable'
        else:  desc = f'NA → Samples have different Verdict → DF-Tests → {urx}'
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
              t = '► Propagate reset of Custom changes to stored Temporal function and train data by Custom Update'
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

    def __init__(self, result:FitResult, vars_out_user:list = None, _is_view:bool = None):
        super().__init__(result, vars_out_user, _is_view)
        self.create_reset_button()
            
    def create(self):

        vars_all = self.get_vars()
        var_init = vars_all.iloc[0]

        # Exo Control Panel
        self.create_feature(var_init, vars_all)
        self.create_check(var_init)                                        
        self.create_graph(var_init)

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
        g_box:wg.HBox = wg.HBox(self.get_graphs(), layout=wg.Layout(width='1050px',height='330px'))
        
        g_box.add_class( UI.get_css_id('panel') )
        g_box.add_class( UI.get_css_id('graph') )
        
        t = 'EXOGENOUS FEATURES CUSTOM SELECTION'
        t = Format.title(t.upper(), style=Format.StyleStats(margin_t=30, margin_b=20), show=False)
        d = [t, f_box, g_box]
        Display.construct(d)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬