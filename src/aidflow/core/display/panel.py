"""
Dynamic User Interface Synthesis Engine for Step-Level Parameterization.

This module implements the core logic for constructing interactive control 
panels that accompany every step of the AIDflow pipeline. It transforms 
abstract parameter configurations into a curated visual experience, providing 
users with an intuitive way to tune hyperparameters and constraints in real-time.

Key operational capabilities:
1. Adaptive Widget Synthesis: Automatically maps Python type definitions to 
   the optimal UI component (e.g., Logarithmic FloatSliders for continuous 
   ranges or BoundedIntText for discrete constraints), ensuring a high-fidelity 
   interaction model.

2. Constraint-Driven Rendering: Enforces strict layout and validation rules 
   via `PanelUtils`, preventing the creation of invalid UI states (such as 
   Max < Min) and guaranteeing that all widgets adhere to the AIGENSYS design system.

3. Reactive Parameter Linking: Implements a "Shared Parameter" mechanism, 
   allowing multiple widgets to be linked bi-directionally. This enables 
   complex behaviors, such as automatically disabling a slider when a manual 
   override value is provided.

4. Intelligent Layout Orchestration: Dynamically organizes widgets into 
   optimized grids (HBox/VBox) based on configurable slices, ensuring that 
   even the most complex ML models remain legible and manageable within a 
   Jupyter Notebook.
"""

from ...deps.display    import wg, display
from ...deps.common     import math, re
from ...deps.types      import FuncType

from ..parameters       import PRM_links
from ..foundation       import AigParam
from ..factory          import DictDataclass
from ..accessor         import UI


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# OPTIONS PANEL PROCESSOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class PanelConstructor:    
    """
    Orchestrator for the synthesis and layout of interactive parameter panels.
    
    This class reads a set of parameter definitions and constructs a structured 
    visual interface. It manages the lifecycle of widgets—from initial creation 
    and CSS styling to state extraction and reactive linking.

    **Default layout**
    slider_layout = {'width': '390px'}
    value_layout  = {'width': '210px'}
    """

    sliders: dict[wg.FloatSlider | wg.IntSlider] = {}
    values:  dict[wg.BoundedIntText] = {}
    shared:  list[str] = []
    
    slider_types: float|int = [float, int]

    def __init__(self, 
        params: DictDataclass, 
        sliders_slice:  int  = 2,
        values_slice:   int  = 4,
        desc_width:     str  = '130px', 
        handle_color:   str  = '#4E70E2', 
        slider_layout:  dict = None,
        value_layout:   dict = None,
        title_e: str = ' ► New Application to Train Data and Model by Confirm'.upper()
    ):
        self.prm      = params
        self.s_slice  = sliders_slice
        self.v_slice  = values_slice
        self.desc_w   = desc_width
        self.color    = handle_color
        self.title_e  = title_e     
        self.s_layout = slider_layout or {'width': '390px'} # 'margin':'10px 0px 5px 15px'}
        self.v_layout = value_layout  or {'width': '210px'} # 'margin':'10px 0px 5px 15px'}
        self.style    = {'description_width': self.desc_w, 'handle_color': self.color}
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_ui_setting(self) -> dict:
        """
        Extracts and resolves the current state of all active UI widgets.
        
        This method performs a 'snapshot' of the user's current selections, 
        resolving linked shared parameters to ensure that only the most 
        authoritative value (slider vs. manual input) is returned to the 
        execution engine.

        Returns:
            dict: A mapping of parameter IDs to their currently selected values.
        """
        sliders = {x: s.value for x,s in self.sliders.items()}
        values  = {x: v.value for x,v in self.values.items() }
                
        chosen = [x for x,s in self.sliders.items() if ( not s.disabled ) and ( x in self.shared )]
        values = {x:v for x,v in values.items() if x not in chosen}

        return sliders | values
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _header(title:str, desc:str=None) -> wg.HTML:
        e = ''
        if desc:
            e = f'<span> {desc} </span>'
        return wg.HTML(f'''
            <div class="header">
                <div class="panel-title"> {title} {e} </div>
            </div>
        ''')

    def panel(self, title:str, show:bool=True) -> None | wg.VBox:
        """
        Constructs and renders the interactive options panel.
        
        This is the primary entry point for displaying step-level options. It 
        synthesizes a hierarchical layout consisting of a stylized header, 
        multi-column widget grids (sliders first, then values), and applies 
        the global `panel` CSS class for professional branding.

        Args:
            title (str): The display name of the panel section.
            show (bool): If True, immediately renders the panel in the notebook output.

        Returns:
            wg.VBox | None: The rendered widget hierarchy if show=False.
        """
        main_layout = wg.Layout(
            justify_content = 'center', 
            align_items = 'flex-start',
            margin = '0 0 10px 0',            
        )
        stack_layout = wg.Layout(
            justify_content = 'center', 
            align_items = 'flex-start',
        )        
        inner_layout = wg.Layout(
            overflow='visible',
        )

        sliders = list(self.sliders.values())
        values  = list(self.values.values())
        
        VBox1 = []
        for i in range(0, len(sliders), self.s_slice):
            VBox1.append( 
                wg.VBox(sliders[i:i+self.s_slice], layout = inner_layout) 
            )
        
        VBox2 = []
        for i in range(0, len(values), self.v_slice):
            VBox2.append(
                wg.VBox(values[i:i+self.v_slice], layout = inner_layout) 
            )        
        
        HBox1 = wg.HBox(VBox1, layout = stack_layout)
        HBox2 = wg.HBox(VBox2, layout = stack_layout)
        
        header = self._header(title, self.title_e)
        panel:wg.VBox = wg.VBox([header, HBox1, HBox2], layout = main_layout)
        panel.add_class(UI.get_css_id('panel'))

        if not show:
            return panel
        display(panel)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def wg_slider_f(self, value, description, mini=0, maxi=1, tip='') -> wg.FloatSlider: 
        o = int(math.log10(maxi-mini)) # we want to raise if range is out of sanity (<=0)
        s = abs(o-2) # order-2 → 10^2 = 100 → we want ~100 slider values
        step = 1/10**s
        form = f'.{s}f'
        return wg.FloatSlider(
            value = value, min = mini, max = maxi, step = step, 
            description = description, 
            style   = self.style,
            layout  = wg.Layout(**self.s_layout),
            tooltip = tip,
            readout_format = form,
            continuous_update = False
        ) 
    
    def wg_slider_i(self, value, description, mini=0, maxi=1, tip='') -> wg.IntSlider:         
        return wg.IntSlider(
            value = value, min = mini, max = maxi, step = 1, 
            description = description, 
            style  = self.style,
            layout = wg.Layout(**self.s_layout),
            tooltip = tip
        ) 
    
    def wg_slider(self, fi:int|float=float):    
        if fi is float:
              return self.wg_slider_f
        else: return self.wg_slider_i
    
    def wg_value(self, value, description, mini=0, maxi=int(1e10), tip='') -> wg.BoundedIntText:
        return wg.BoundedIntText(
            value = value, min = mini, max = maxi, step = 1, 
            description = description, 
            style   = {'description_width': self.desc_w}, 
            layout  = wg.Layout(**self.v_layout),
            tooltip = tip
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _get_tip(p:str) -> str:
        if p in PRM_links:
            return re.sub(r'<[^>]+>', '', PRM_links[p].tip)
        return ''
    
    @staticmethod
    def _is_widget(val_desc:list[float|int|None, str|wg.Widget]) -> bool:
        """
        Created widget can be provided directly instead of [min, max]
        desc[p] = [str.description | wg.Widget, opt: [min,max] , opt: type i|f]
        """
        val, *cfg = val_desc
        if isinstance(cfg[0], wg.Widget):
            return True
        return False
    
    @staticmethod
    def _get_widget(val_desc:list[float|int|None, wg.Widget]) -> wg.Widget:
        val, *cfg = val_desc
        return cfg[0]
    
    def _to_param_val_desc(self, desc:dict) -> dict[str, list[float|int, str|wg.Widget]]:
        """
        Combine params with value and description.
        Return dict in a form:
        >>> dict[p] = [value, desc[p]]
        >>> where desc[p] = [str.description | wg.Widget, opt: [min,max] , opt: type i|f]
        """
        desc_verif = {p:v for p,v in desc.items() if not self._is_widget([None,v])}
        AigParam.verify_ui_params(self.prm, desc_verif)
        vars_ = {}
        tips  = {}
        order = {p: self.prm[p] if p in self.prm.keys() else desc[p] for p in desc}        
        for p, v in order.items():
            v,c = PanelUtils.val_cfg_constrain(p, v, desc)
            cfg = PanelUtils.desc_min_max(p, c)
            cfg = [cfg] if not isinstance(cfg, (list, tuple)) else cfg
            vars_[p] = [v, *cfg]
            tips[p]  = self._get_tip(p)
        return vars_, tips
    
    def params_to_sliders(self, desc:dict) -> None:
        """
        Automatically generates a set of sliders based on a provided description schema.

        Parameters Design
        -----------------
        desc = {
            'param_x': config_x,
            'param_y': config_y,
            ...
        }
        where 
        config_x = ['Desc of Param', [min, max], int|float, constrain_fx]

        - Mandatory :
          - Slider Description (str) : Description of slider in first position in config - 'Desc of Param'

        - Optional :
          - min-max (str|list) : Slider boundaries - either as a value or list of values [min, max]
          - slider type (type, default float): int for integer | float for float type
          - constrain to value (fx) : constrain_fx which can adjust/modify default value of slider
        """
        params_val_desc, tips = self._to_param_val_desc(desc)
        self.sliders = {}
        for p, val_desc_mm in params_val_desc.items():
            if self._is_widget(val_desc_mm):
                self.sliders[p] = self._get_widget(val_desc_mm)
                continue
            fi, val_desc_mm = PanelUtils.slider_type_cfg(p, val_desc_mm)            
            self.sliders[p] = self.wg_slider(fi)(*val_desc_mm, tip=tips[p])

    def params_to_values(self, desc:dict) -> None:
        """
        Autolink: Provide set of wg.value based on param-description.
        
        Parameters Design
        -----------------
        desc = {
            'param_x': config_x
        }
        where 
        config_x = ['Desc of Param', [min, max], constrain_fx]

        - Mandatory :
          - Value Description (str) : Description of value in first position in config - 'Desc of Param'

        - Optional :
          - min-max (str|list) : Value boundaries - either as a value or list of values [min, max]
          - constrain to value (fx) : constrain_fx which can adjust/modify default value of slider
        """
        params_val_desc, tips = self._to_param_val_desc(desc)
        self.values = {}
        for p, val_desc_mm in params_val_desc.items():
            if self._is_widget(val_desc_mm):
                self.sliders[p] = self._get_widget(val_desc_mm)
                continue            
            self.values[p] = self.wg_value(*val_desc_mm, tip=tips[p])
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def shared_param_link(self, linked_params:dict) -> None:
        """Creates link between connected slider-value widget given provided link-fx."""
        for p,fx in linked_params.items():
            for x in self.values:
                if x == p: break
            for y in self.sliders:
                if y == p: break
            self.values[x].observe( fx(self.sliders[y] ), names='value')
        self.shared = [x for x in self.sliders if x in self.values]
        
    def shared_relat_abs(self, slide:wg.FloatSlider):
        """Base use of link: Disable slider when value widget is activated."""        
        def link(change):
            if change['new'] > 0: 
                  slide.disabled = True
            else: slide.disabled = False        
        return link
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PanelUtils:
    """
    Validation and transformation utilities for UI widget configuration.
    
    This class provides a set of static guardrails to ensure that parameters 
    passed to the `PanelConstructor` are structurally sound and visually 
    consistent, preventing runtime errors during GUI rendering.
    """

    slider_types = PanelConstructor.slider_types
    _W = lambda p: f'AIG SAFETY ► GUI PANEL ► {p} ►'
    
    @classmethod
    def val_cfg_constrain(cls, p:str, val, desc:dict) -> tuple[float|int, list]:
        """
        Validates and applies constraint functions to parameter values.
        
        Ensures that only one constraint function is applied per parameter 
        and resolves the initial value based on the provided logic before 
        the widget is instantiated.
        """        
        cfg = desc[p]
        if not isinstance(cfg, list):
            return val, cfg
        fx = [i for i,x in enumerate(cfg) if isinstance(x, FuncType)]
        if not fx:
            return val, cfg
        if len(fx) > 1:
            raise ValueError(f'{cls._W(p)} ► Parameter constrain can be applied by one FX only')
        fx = cfg.pop(fx[0])
        return fx(val), cfg

    @classmethod
    def desc_min_max(cls, p:str, cfg:list|str) -> list:
        if not isinstance(cfg, list):
            return cfg
        mm = []
        desc, *cfg = cfg
        if not isinstance(desc, str):
            raise ValueError(f'{cls._W(p)} WG description must be defined as the first position')
        if cfg:             
            mm, *cfg = cfg        
        if not isinstance(mm, list):
            mm = [mm]
        if any([not isinstance(x, (float,int)) for x in mm]):
            raise ValueError(f'{cls._W(p)} MIN, MAX ► Provided config for min-max is incorrect:\n {mm}')
        if len(mm)==2 and mm[0] > mm[1]:
            raise ValueError(f'{cls._W(p)} MAX < MIN ? ► Provided config for min-max is incorrect')
        return [desc, *mm, *cfg]
    # ─────────────────────────────────────────────────────────────────────────────────────────────        

    @classmethod
    def slider_type_cfg(cls, p:str, val_desc:list) -> tuple[type, list]:
        """
        Resolves the appropriate numeric type (int vs float) for a slider.
        
        Parses the configuration list to determine if the parameter requires 
        integer precision or floating-point granularity, defaulting to 
        float if not explicitly specified.
        """
        fi = float  # user doesn't need to provide type if type is float
        if len(val_desc) == 2:  # [value, desc] → mandatory min design
            return fi, val_desc
        fi_ = [i for i,x in enumerate(val_desc) if x in cls.slider_types]
        if not fi_:
            return fi, val_desc
        if len(fi_) > 1: 
            raise ValueError(f'{cls._W(p)} SLIDER CHOICE ► int for integer or float for float type') 
        fi = val_desc.pop(fi_[0])
        return fi, val_desc
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬