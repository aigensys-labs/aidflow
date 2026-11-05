from ...deps.display        import  (
                                    wg, 
                                    HTML,
                                    display, 
                                    )
from ...core.parameters     import  PRM
from ...core.factory        import  (
                                    DictDataclass,
                                    Dataclass,
                                    DictClass
                                    )
from ...core.io             import  TempRegistry
from ...core.accessor       import  UI
from ...core.config         import  ConfigCss
from ...display.css         import  TUNING
from ...display.verbosity   import  Verbosity

from .processor             import  (
                                    ID as ID_P,
                                    LBL as LBL_P,
                                    STAT as STAT_P,
                                    SelectorStats,
                                    SelectorParamsBase
                                    )

class LBL(DictClass):
    N = lambda x: f'Agent {x}'    


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION → UI → PANEL
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class AgentsOverviewPanel:
    """Tuninng Choice Overview of chosen significant features by Agent."""

    def __init__(self, 
        project_id: str,
        title: str = 'Agent Selection ►', 
        span:  str = 'Performance Overview'
    ):
        self.p_id  = project_id
        self.title = title.upper()
        self.span  = span.upper()

        self.panel_id  = UI.get_css_id('panel')
        self.tuning_id = UI.get_css_id('tuning')

    def _get_path(self, a:int) -> str:
        return TempRegistry.pop(self.p_id, ID_P.P(a))
    
    @staticmethod
    def _button(label:str, path:str) -> str:
        return (
            f'<a href="{path}" target="_blank" rel="noopener noreferrer"'
            f'class="{TUNING.B}">{label}</a>'
        )

    def _span(self) -> str:        
        if not self.span:
            return ''        
        return f'<span> {self.span} </span>'

    def create(self, agents: dict|list) -> None:
        """Create agent buttons as a list of HTML strings."""
        buttons = "".join([self._button(LBL.N(a), self._get_path(a)) for a in agents])

        panel = f'''
        <div class="{self.panel_id}">
            <div class="header">
                <div class="panel-title"> {self.title} {self._span()} </div>
            </div>
            <div class="{TUNING.C}">
                {buttons}
            </div>
        </div>
        '''

        display(HTML(panel))
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class AgentsFeaturesPanel:
    """Tuninng Choice Features Selection by Agents."""
    _confirm_button = None
    fe_tabs = None

    def __init__(self, 
        agent_choice:   int, 
        agents_ordered: list, 
        verbose:        int,
        panel_id:       str = 'agent-final-selection'
    ):
        self.prior_choice = agent_choice
        self.panel_id = panel_id
        
        self._set_agent_choice(agents_ordered)
        
        if Verbosity.normalize(verbose) != Verbosity.GUI:
            return
        
        ConfigCss(self.panel_id, UI.theme)
    
        self.fe_tabs: dict[int, dict] = {}
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_agent_choice(self, order: list) -> None:
        if self.prior_choice > 0:
            choice = self.prior_choice
        else:            
            choice = order[0]

        self.agent_choice = choice
        self._selected    = choice
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @property
    def confirm_button(self) -> wg.Button:
        if self._confirm_button is not None:
            return self._confirm_button        
        self._set_confirm_button()
        return self._confirm_button

    def _set_confirm_button(self) -> None:
        self._confirm_button:wg.Button = wg.Button(description="CONFIRM FINAL AGENT")
        self._confirm_button.add_class('gh-btn-primary')        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _create_row(self, agent:int, fe:str, definition: dict[str, float|dict]) -> wg.HBox:
        
        if not self.fe_tabs.get(agent):
            self.fe_tabs[agent] = {}

        check = wg.Checkbox(value = True, layout = wg.Layout(width='30px', margin='5px 10px 0 10px'))
        self.fe_tabs[agent][fe] = check
        
        fe_ = lambda x: fe[:x] + ('' if len(fe) <= x else '...')
        lbl = wg.HTML(f"""
            <div class='gh-label' style='width: 350px;'>{fe_(45)}</div>
        """)
        tool = wg.HTML(f"""
            <div class="gh-tooltip" style="margin: 7px 25px;">?
                <span class="tooltip-text">
                <i>{fe_(32)}</i><br>
                √ Checked features enter the final agent fit<br>
                <br>
                <i>Hints Metrics Samples:</i><br>
                • Gini Value  → Validation<br>
                • Permutation → Validation<br>
                • Shap Values → Validation x Test
                </span>
            </div>
        """)
        sh = definition[STAT_P.SH]
        desc = wg.HTML(f"""
            <div class='gh-desc' style="margin: 0 0 0 0;">
            Performance → Gini: {definition[STAT_P.GP]} • 
            Permutation: {definition[STAT_P.PM]} • 
            Shap → Valid: {sh[LBL_P.V]} | Test: {sh[LBL_P.T]}
            </div>
        """)

        row:wg.HBox = wg.HBox(
            [check, lbl, tool, desc],
            layout=wg.Layout(width='100%', align_items='center')
        )
        row.add_class('gh-row')
        return row
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _header(self) -> wg.HTML:
        return wg.HTML(f'''
            <div class="gh-header">
                <h1>AGENT <span> CONFIGURATOR </span> </h1>
            </div>
        ''')
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _build_ui(self, stats: dict[int, dict]) -> None:
        
        # ROWS BY TABS
        tab_fes: dict[int, dict] = {}
        for a, stat in stats.items():
            fes: dict  = stat[STAT_P.FD]
            tab_fes[a] = wg.VBox(
                [self._create_row(a, fe, definition) for fe, definition in fes.items()]
            )

        # INIT TAB
        tab_container:wg.Box = wg.Box(
            [tab_fes[self.agent_choice]], layout=wg.Layout(width='100%')
        )

        # TABS
        def tab_desc(a:int):
            return LBL.N(a)
        
        def switch(tab:wg.Button):
            for i, a in enumerate(stats):
                if tab.description == tab_desc(a):
                    tab_container.children = [tab_fes[a]]
                    tab.add_class('active')
                    self._selected = a
                else:
                    tabs[i].remove_class('active')

        tabs:list[wg.Button] = []
        for a in stats:
            tab:wg.Button = wg.Button(description = tab_desc(a))
            tab.on_click(switch)
            tab.add_class('gh-tab-btn')
            if a == self.agent_choice: 
                tab.add_class('active')
            tabs.append(tab)

        tabs_menu:wg.HBox = wg.HBox(tabs)
        tabs_menu.add_class('gh-tabs')

        # LAYOUT
        header = self._header()
        
        ui:wg.VBox = wg.VBox([header, tabs_menu, tab_container, self.confirm_button])
        ui.add_class(self.panel_id) 

        display(ui)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def create(self, stats: SelectorStats) -> None:
        self._build_ui(stats.summary)

    def get_final_choice(self) -> int:        
        return self._selected
    
    def get_selected_fes(self) -> dict[int, list]:
        choice = self.get_final_choice()
        return {choice: [fe for fe, c in self.fe_tabs[choice].items() if c.value]}
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PanelUtilsUI:

    def __init__(self, 
        params: DictDataclass, 
        fx_params: dict|None, 
        agents_order: list, 
        project_sig: str,
        verbose: int
    ):
        self.prm = params        

        choice  = self._get_prior_choice(params, fx_params)
        self.wg = AgentsFeaturesPanel(choice, agents_order, verbose)

        self.project_sig = project_sig
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @property
    def confirm_button(self) -> None | wg.Button:
        """Confirmation button of champion agent choice."""
        return self.wg.confirm_button
    
    @staticmethod
    def _get_prior_choice(params: DictDataclass, fx_params: dict|None) -> int:
        s_prm = Dataclass.schema(SelectorParamsBase)
        fx_params     = fx_params or {}
        direct_choice = fx_params.get(s_prm.agent_choice)
        global_choice = getattr(params, PRM.SELECT.select_choice.param)        
        return direct_choice if direct_choice is not None else global_choice
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def get_ui_setting(self) -> dict:
        """Returns actual UI selected features of chosen agent."""
        prm = lambda x: dict(self.prm) | {PRM.SELECT.select_choice.param: x}
        if not self.wg.fe_tabs:
            return prm(self.wg.agent_choice)
        return prm(self.wg.get_selected_fes())
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def overview(self, stats: SelectorStats) -> None:
        """Shows Overview Buttons by Agents."""
        AgentsOverviewPanel(self.project_sig).create(stats.summary)

    def options(self, stats: SelectorStats) -> None:
        """Shows Main Configuration Panel."""    
        self.wg.create(stats)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def panel(self, stats: SelectorStats) -> None:        
        """Shows Agents Tuning Selection Panel."""
        self.overview(stats)
        self.options(stats)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬