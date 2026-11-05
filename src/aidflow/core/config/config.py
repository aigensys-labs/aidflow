"""
Interactive Configuration Gateway for AIDflow Pipeline Orchestration.

This module implements the primary user interface layer, transforming 
abstract parameter schemas defined in `core.parameters` into high-fidelity, 
interactive control panels within Jupyter environments. It serves as the 
bridge between human intuition and machine execution, allowing users to 
configure complex ML pipelines through a curated visual experience.

Key operational pillars:
1. Dynamic UI Synthesis: Automatically generates adaptive input widgets 
   (sliders, dropdowns, comboboxes) based on the parameter's `kind` and 
   metadata, ensuring that the GUI is always perfectly synced with the 
   backend schema.

2. State Persistence & Recovery: Implements a robust state management system 
   via `UIConfig`, enabling the automatic loading and saving of user-defined 
   setups to eliminate redundant configuration across notebook sessions.

3. Hierarchical Configuration Flow: Orchestrates the setup process through 
   two specialized interfaces:
     - ProjectGenesis: The "Gateway" panel for root project initialization 
       (paths, targets, and global identifiers).
     - PipelineConfigurator: The "Tuning" panel for iterative hyperparameter 
       optimization using a tabbed, category-based navigation system.

4. Reactive Propagation: Utilizes a callback mechanism to immediately propagate 
   confirmed configurations into the pipeline's execution memory, ensuring 
   that subsequent model fits are applied with the exact user-specified settings.
"""

from ...deps.common     import  ast
from ...deps.data       import  np
from ...deps.display    import  (
                                display, 
                                HTML, 
                                wg
                                )
from ...deps.types      import  (
                                Any, 
                                Literal,
                                Callable
                                )
from ..parameters       import  (
                                GEN, 
                                PRM, 
                                ParamBase,
                                Parameters, 
                                )
from ..factory          import  Dataclass
from ..interop          import  Pd
from ..io               import  ProjectPaths

from .state             import  ConfigState
from .css               import  ConfigCss


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# AIG PIPELINE PARAMETERS CONFIGURATION PANEL
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ProjectGenesis:
    """
    The primary entry-point interface for AIDflow Project Initialization.
    
    ProjectGenesis orchestrates the 'Genesis Phase' of the pipeline, capturing 
    root-level configurations such as project naming, data sourcing, and target 
    variable definition. It ensures that the foundational workspace is correctly 
    mapped before any ML execution occurs.

    State Resolution
    ----------------
    Initialization values are resolved in the following
    precedence order (highest first):

    1. **setup** (dict): Explicit dynamic values passed at
       construction. These override everything else.
    2. **Stored state**: Persisted values from a previous
       session (loaded via `ConfigState`).
    3. **Defaults**: Parameter defaults defined in the
       `Parameters` schema.

    Parameters
    ----------
    theme : str = 'dark
        Color theme from 'dark', 'light', 'pro'.
    kind : str ='h
        Orientation of the panel → horizontal: 'h' | vertical: 'v'.
    panel_css_id : str = 'gateway-v1'
        Panel element css id.
    setup : dict, default=None
        Optional pre-set values for genesis parameters.
        Takes precedence over stored state.
    """

    def __init__(self, 
        theme: str = 'dark', 
        kind: Literal['h','v'] = 'h', 
        panel_css_id: str = 'gateway-v1',
        setup: dict = None
    ):
        self._verify_params()
        self.config  = {}
        self.widgets = {}

        self.kind = kind
        self.panel_id = panel_css_id
        ConfigCss(self.panel_id, theme)

        self.ui_state  = self._get_state(setup)
        self.container = self._build()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _verify_params() -> None:
        params = Dataclass.get_names(Parameters)
        for g in GEN.values():
            g:ParamBase = g
            if g.param in params:
                continue
            raise ValueError(
                f"AIG SAFETY ► Gen Panel Parameter {g.param} mismatch with Parameters definiton"
            )

    @staticmethod
    def _validate_setup(setup:dict) -> dict:
        gen: tuple[ParamBase] = GEN.values()
        
        t = lambda x: x if isinstance(x, tuple) else (x,)
        prm = {g._prm: t(g._kind) for g in gen}

        for p,v in setup.items():
            if p not in prm:
                raise ValueError(
                    f"AIG SAFETY ► Gen Panel Parameter {p} mismatch with Parameters definiton"
                )
            
            if v is np.nan: # normalize load
                setup[p] = None
                continue

            t = tuple(x for x in prm[p] if x is not None)
            if not isinstance(v, t):
                raise ValueError(
                    f"AIG SAFETY ► Gen Parameter {p} value mismatch with expected type {t}"
                )
            
        return setup
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _save_state(config:dict) -> None:
        ConfigState().save(config)
    
    @classmethod
    def _get_state(cls, setup: dict = None) -> dict:
        """
        Resolve initialization values from dynamic setup + stored state.

        Precedence: `setup` (explicit) > stored state (persisted)
        > schema defaults. Type-mismatch between setup and
        stored state raises an error (prior safety).
        """
        state = ConfigState().load() or {}
        if not setup:
            return cls._validate_setup(state)
        
        cfg = {**setup}

        for k,v in state.items():
            if k not in cfg:
                cfg[k] = v

        return cls._validate_setup(cfg)   
        
    def _get_state_value(self, prm:str, value:list|str) -> list|str:
        if not self.ui_state:
            return value
        return self.ui_state.get(prm, value)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _create_widget(self, definition: ParamBase) -> wg.Text | wg.Dropdown | wg.Combobox:
        is_v = self.kind == 'v'

        prm  = definition.param
        val  = self._get_state_value(prm, definition.value)
        opt  = definition.get_v_options()

        if opt is None: 
            return wg.Text(
                value = val,
                placeholder = definition.get_v_holder(),
                layout = wg.Layout(width='400px' if is_v else '230px')
            )
        if prm == GEN.time_f.param: 
            return wg.Combobox(
                value   = val, 
                options = opt, 
                layout  = wg.Layout(width='400px' if is_v else '140px')
            )        
        return wg.Dropdown(
            value   = val,
            options = opt,
            layout  = wg.Layout(width='400px' if is_v else '140px')
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _create_row_v(self, definition: ParamBase) -> wg.HBox:
        widget = self._create_widget(definition)
        self.widgets[definition.param] = widget

        a = 20
        # if isinstance(widget, (wg.Dropdown, wg.Combobox)):
        #     a = 170

        lbl = wg.HTML(f"""
            <div class='gh-label' style='width: 130px;'>{definition.name}</div>
        """)
        tool = wg.HTML(f"""
            <div class="gh-tooltip" style="margin: 10px 15px;">?
                <span class="tooltip-text">{definition.tip}</span>
            </div>
        """)
        desc = wg.HTML(f"""
            <div class='gh-desc' style="margin: 0 0 0 {a}px;">{definition.desc}</div>
        """)

        row: wg.HBox = wg.HBox(
            [lbl, tool, widget, desc],
            layout=wg.Layout(width='100%', align_items='center')
        )        
        row.add_class('gh-row')

        return row
        
    def _create_row_h(self, definition: ParamBase) -> wg.HBox:
        widget = self._create_widget(definition)
        self.widgets[definition.param] = widget

        lbl = wg.HTML(f"""
            <div class='gh-label'>{definition.name}</div>
        """)
        tool = wg.HTML(f"""
            <div class="gh-tooltip" style="margin: 0px 0px 0px 15px;">?
                <span class="tooltip-text">{definition.tip}</span>
            </div>
        """)
        h = definition.desc_h
        desc = h if h else definition.desc
        desc = wg.HTML(f"""
            <div class='gh-desc' style="margin: 0 0 0 0px;">{desc}</div>
        """)

        row: wg.HBox = wg.VBox(
            [wg.HBox([lbl, tool]), widget, desc],
            layout=wg.Layout(width='100%', align_items='flex-start')
        )        
        row.add_class('gh-row')

        return row
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    def _create_rows(self):
        # (A) Vertical design by rows
        order = [
            GEN.project,  
            GEN.data, GEN.source, 
            GEN.target,
            GEN.time, GEN.time_f,
            GEN.cols, GEN.flow
        ]        
        if self.kind == 'v':
            rows = []
            for gen in order:
                rows.append( self._create_row_v(gen) )
            return wg.VBox(rows)
        
        # (B) Horizontal design by cols
        order = [
            [GEN.project, GEN.target], 
            [ 
              [GEN.data, GEN.source], 
              [GEN.time, GEN.time_f] 
            ],
            [GEN.cols, GEN.flow]
        ]
        columns = []
        for defs in order:
            if not isinstance(defs[0], list):
                columns.append( wg.VBox([self._create_row_h(definition) for definition in defs]) )
                continue
            j = []
            for joined in defs:
                j.append( wg.HBox([self._create_row_h(definition) for definition in joined]) )
            columns.append(wg.VBox(j))

        return wg.HBox(columns)        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _header(self) -> wg.HTML:
        return wg.HTML(f'''
            <div class="gh-header">
                <h1>PROJECT <span> GENESIS </span> </h1>
            </div>
        ''')

    def _button(self) -> wg.Button:
        btn: wg.Button = wg.Button(description='INITIALIZE PROJECT')
        btn.add_class('gh-btn-primary')
        btn.on_click(self._on_submit)   
        return btn
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _build(self) -> wg.VBox:
        rows   = self._create_rows()
        header = self._header()
        button = self._button()
        self.output:wg.Output = wg.Output()

        ui:wg.VBox = wg.VBox([header, rows, button, self.output])
        ui.add_class(self.panel_id) 
        
        self._set_config()

        return ui
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _verify_submit(cfg:dict) -> bool:        

        def ver(group:list[ParamBase], is_conditioned:bool = False):
            if is_conditioned and not cfg[group[0].param]:
                return True
            
            is_ok = True
            for m in [m.param for m in group]:                
                if not cfg[m]: 
                    is_ok = False
                    display(HTML(f"<div class='gh-label'>❌ {m} is required</div>"))
            return is_ok
        
        mandatory = [GEN.project, GEN.data, GEN.target]
        condition = [GEN.time, GEN.time_f]

        return ver(mandatory) & ver(condition, True)
    
    @staticmethod
    def _confirm(cfg):
        display(HTML(f"""
        <div class='gh-toast'>
            ✓ Initialization Complete: Workspace locked to <b>{cfg[GEN.project.param]}</b>. 
            Targeting <b>{cfg[GEN.target.param]}</b> via {cfg[GEN.source.param].upper()}.
        </div>
        """))    

    def _on_submit(self, _) -> None:
        self._set_config()

        with self.output:
            self.output.clear_output()
            
            cfg = self.get_config()
            if not self._verify_submit(cfg):
                return
            self._confirm(cfg)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_config(prm:str, val:str) -> str|list:        
        if prm not in [GEN.cols.param, GEN.flow.param]:
            return val.strip()
        if not val:
            return None
        if val[0] in ('{', '[', '('):
            return ast.literal_eval(val)
        return [ast.literal_eval(x.strip()) for x in val.split(',') if x.strip()]

    def _set_config(self) -> None:             
        self.config = {p: self._format_config(p, w.value) for p, w in self.widgets.items()}
        self._save_state(self.config)

    def get_config(self) -> dict:
        """Returns genesis config state."""
        return self.config
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def show(self) -> None:
        """
        Renders the Project Genesis panel within the Jupyter Notebook output cell.
        
        This method triggers the display of the header, parameter rows, 
        and initialization button using the currently active VDS theme.
        """        
        display(self.container)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class PipelineConfigurator:
    """
    Interactive Hyperparameter Tuning Interface for Pipeline Optimization.
    
    The PipelineConfigurator provides a tabbed navigation system that allows users 
    to iteratively refine model parameters without modifying source code. It 
    bridges the gap between high-level pipeline settings and low-level fitter 
    configurations via reactive callbacks.
    
    State Resolution
    ----------------
    Initialization values are resolved in the following
    precedence order (highest first):

    1. **setup** (dict): Explicit dynamic values passed at
       construction. These override everything else.
    2. **Stored state**: Persisted values from a previous
       session (loaded via `ProjectPaths`).
    3. **Defaults**: Parameter defaults defined in the
       `Parameters` schema.

    Parameters
    ----------
    theme : str
        Color theme from 'dark', 'light', 'pro'.
    panel_css_id : str
        Panel element css id.
    callback : Callable
        External function triggered upon configuration confirmation.
    setup : dict, default=None
        Optional pre-set values for pipeline parameters.
        Takes precedence over stored state.    
    """
    def __init__(self, 
        theme: Literal['dark', 'light', 'pro'] = 'dark', 
        panel_css_id: str = 'panel-v1',
        callback: Callable = None,
        setup : dict = None
    ):
        self._verify_params()
        self.config  = {}
        self.sliders = {}

        self.panel_id = panel_css_id
        ConfigCss(self.panel_id, theme)
        
        self.callback  = callback
        self.ui_state  = self._get_user_setup(setup)
        self.container = self._build_ui()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _verify_params(self) -> None:
        params = Dataclass.get_names(Parameters)
        for prm in PRM.values():
            prm: dict[str, ParamBase] = prm
            for p in prm.values():
                if p.param in params:
                    continue
                raise ValueError(
                    f"AIG SAFETY ► Panel Parameter {p.param} mismatch with Parameters definiton"
                )

    def _callback(self, config:dict):
        if self.callback is not None:
            self.callback(config)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_user_setup(setup:dict = None) -> dict | None:
        """
        Resolve initialization values from dynamic setup + stored state.

        Precedence: `setup` (explicit) > stored state (persisted)
        > schema defaults. Type-mismatch between setup and
        stored state raises an error (prior safety).
        """
        gen = ProjectGenesis._get_state()
        if not (gen or setup):
            return # no project definition → raises by Parameters verification

        project = gen.get(GEN.project.param)
        if project in (None, np.nan):
            return # no project definition → raises by Parameters verification
        
        state = ProjectPaths(project, is_raise=False).get_setup()
        state = state.__dict__ if state else {}
        
        if not setup:
            return state
        
        cfg = {**setup}

        for p, v in state.items():
            if p not in cfg:
                cfg[p] = v
                continue
            
            t = type(v)
            if not isinstance(cfg[p], t):
                raise ValueError(
                    f"AIG SAFETY ► Parameter {p} value mismatch with expected type {t}"
                )
        return cfg  
            
    def _get_user_val(self, prm:str, value:float|int) -> float|int:
        if not self.ui_state:
            return value
        return self.ui_state.get(prm, value)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _create_slider(self, definition:ParamBase) -> wg.FloatSlider | wg.IntSlider:
        f = ''        
        val = self._get_user_val(
            definition.param, 
            definition.value
        )

        if isinstance(val, float): 
            s = wg.FloatSlider
            s_= definition.get_v_step()
            f = f'{s_:.10f}'.rstrip('0').split('.')[1]
            f = f'.{len(f)}f'
        elif isinstance(val, int): 
            s = wg.IntSlider
    
        return s(
            **definition.get_values(val),
            continuous_update = True, 
            readout_format = f, 
            layout = wg.Layout(width='350px')
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _create_row(self, definition:ParamBase) -> wg.HBox:
        
        slider = self._create_slider(definition)
        self.sliders[definition.param] = slider

        lbl = wg.HTML(f"""
            <div class='gh-label' style='width: 130px;'>{definition.name}</div>
        """)
        tool = wg.HTML(f"""
            <div class="gh-tooltip" style="margin: 12px 15px;">?
                <span class="tooltip-text">{definition.tip}</span>
            </div>
        """)
        desc = wg.HTML(f"""
            <div class='gh-desc' style="margin: 0 0 0 20px;">{definition.desc}</div>
        """)

        row:wg.HBox = wg.HBox(
            [lbl, tool, slider, desc],
            layout=wg.Layout(width='100%', align_items='center')
        )
        row.add_class('gh-row')
        return row
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _header(self) -> wg.HTML:
        return wg.HTML(f'''
            <div class="gh-header">
                <h1>PIPELINE <span> CONFIGURATOR </span> </h1>
            </div>
        ''')

    def _button(self) -> wg.Button:
        btn:wg.Button = wg.Button(description="CONFIRM & PROPAGATE CONFIG")
        btn.add_class('gh-btn-primary')
        btn.on_click(self._on_submit)        
        return btn
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> wg.VBox:
        
        # ROWS BY TABS
        rows = []
        for prm in PRM.values():
            rows.append(
                wg.VBox([self._create_row(definitions) for definitions in prm.values()])
            )

        # INIT TAB
        tab_container:wg.Box = wg.Box([rows[0]], layout=wg.Layout(width='100%'))         

        # TABS
        def tab_desc(i:int, prm:str):
            return f"{i}. {prm}"
        
        def switch(tab:wg.Button):
            for i, prm in enumerate(PRM.keys()):
                if tab.description == tab_desc(i,prm):
                    tab_container.children = [rows[i]]
                    tab.add_class('active')
                else:
                    tabs[i].remove_class('active')

        tabs:list[wg.Button] = []
        for i, prm in enumerate(PRM.keys()):
            tab:wg.Button = wg.Button(description = tab_desc(i,prm))
            tab.on_click(switch)
            tab.add_class('gh-tab-btn')
            if i == 0: 
                tab.add_class('active')
            tabs.append(tab)

        tabs_menu:wg.HBox = wg.HBox(tabs)
        tabs_menu.add_class('gh-tabs')

        # LAYOUT
        header = self._header()
        button = self._button()
        self.output:wg.Output = wg.Output()
        
        ui:wg.VBox = wg.VBox([header, tabs_menu, tab_container, button, self.output])
        ui.add_class(self.panel_id) 

        # INIT CONFIG
        self._set_config()

        return ui
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_config(x: wg.FloatLogSlider | wg.IntSlider) -> float | int:
        return round(x.value, Pd.get_num_precision(x.step))
            
    def _set_config(self) -> None:               
        self.config = {p: self._format_config(w) for p, w in self.sliders.items()}
        self._callback(self.config)
    
    def get_config(self) -> dict:
        """Returns params config state."""
        return self.config        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _on_submit(self, b) -> None:
        self._set_config()

        with self.output:
            self.output.clear_output()        
            display(wg.HTML("""
                <div class='gh-toast'>
                    ✓ Configuration locked into memory. Pipeline ready for execution.
                </div>
            """))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def show(self) -> None:
        """
        Renders the tabbed Configuration Panel within the Jupyter Notebook.
        
        Initializes the visual tabs based on `PRM` categories and displays 
        the corresponding tuning sliders and readouts.
        """        
        display(self.container)        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬