"""
Visual Identity System and Dynamic Theme Engine for the AIDflow User Interface.

This module implements the comprehensive styling engine used to render 
AIDflow's interactive UI components. It utilizes a decoupled design pattern 
where structural layout, component geometry, and thematic aesthetics are 
managed as independent layers, allowing for real-time theme switching without 
compromising interface stability.

Key visual architecture features:
1. Dynamic Theme Engine: Supports multiple professional themes (Dark, Light, 
   Metal, Pro) via a CSS variable system that maps semantic roles (e.g., `--accent`, 
   `--bg-panel`) to specific aesthetic palettes.

2. High-Fidelity Components: Implements premium UI elements including 
   interactive data tables with zebra-striping, holographic "shine" animations 
   on buttons, and neon-glow readouts for AI state monitoring.

3. Responsive Layout Geometry: Enforces a standardized coordinate system 
   for panels, sliders, and widgets to ensure consistent rendering across different 
   Notebook environments (JupyterLab, VS Code, Colab).

4. Semantic Styling: Uses the `STYLE` and `TUNING` registries as a centralized 
   ID map, ensuring that UI changes can be propagated globally without 
   modifying individual component implementations.
"""

from ..deps.display     import  HTML, display
from ..deps.data_view   import  CSS_IT

from ..core.factory     import  DictClass
from ..core.config      import  THEME


class CSS(DictClass):
    """AIG Flow Global CSS ID Access."""
    STEP:   str = 'AIG-markdown-step'
    LINE:   str = 'AIG-invisible-line'
    PANEL:  str = 'AIG-panel-step'
    TUNING: str = 'AIG-panel-tuning-choice'
    BUTTON: str = 'AIG-confirm-button'
    GRAPH:  str = 'AIG-graph-card'
    TABLE:  str = CSS_IT

class TUNING(DictClass):
    """AIG Tuning Choice - Global CSS ID Access."""
    C = f'{CSS.TUNING}-container'
    B = f'{CSS.TUNING}-button'


class CssStyle:
    """Styling engine used to render AIDflow's interactive UI components."""
    ALL = CSS.list()

    def __init__(self, theme:str):
        self.theme = theme
        self._css_fx = dict(zip(
            self.ALL, 
            [self.step, self.line, self.panel, self.tuning, self.button, self.graph, self.table]
        ))

    def ui(self) -> list[str]:
        fx  = lambda x: self._css_fx[x](x)
        [fx(x) for x in self.ALL]
        return self.ALL

    def step(self,
        css_id: str = CSS.STEP,
        padding:str = '15px',
        radius: str = '12px'
    ) -> None:
        display(HTML(f'''
        <style>
        .{css_id}-logic {{
            background: rgba(186, 195, 204, 0.15); 
            border-left: 5px solid #BAC3CC; 
            padding: 10px 15px; 
            margin: 20px 0; 
            border-radius: {radius};
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            color: #1E293B;
            line-height: 1.7;
        }}
        .{css_id}-grey {{
            background: rgba(69, 75, 84, 0.1); 
            border-left: 5px solid #B3B1B1; 
            padding: {padding};
            color: #000000;
            border-radius: {radius};
        }}             
        .{css_id}-red {{
            background: rgba(246, 59, 115, 0.1); 
            border-left: 5px solid #F63B73; 
            padding: {padding};
            color: #F63B73;
            border-radius: {radius};
        }}        
        .{css_id}-blue {{
            background: rgba(59, 130, 246, 0.1); 
            border-left: 5px solid #3B82F6; 
            padding: {padding};
            color: #3B82F6;
            border-radius: {radius};
        }}
        .{css_id}-green {{
            background: rgba(9, 200, 127, 0.1); 
            border-left: 5px solid #09C87F; 
            padding: {padding};
            color: #09C87F;
            border-radius: {radius};
        }}
        .{css_id}-purple {{
            background: rgba(155, 11, 244, 0.1); 
            border-left: 5px solid #9B0BF4; 
            padding: {padding};
            color: #9B0BF4;
            border-radius: {radius};
        }}
        </style>
        '''))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def line(self,
        css_id: str = CSS.LINE,
    ) -> None:
        display(HTML(f'''
        <style>
        .{css_id} {{
            display: none !important;
        }}        
        </style>
        '''))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def panel(self, css_id = CSS.PANEL) -> None:
        panel = CssPanel(css_id, self.theme)
        self.panel_id = panel.get_css_id()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def tuning(self, css_id = CSS.TUNING) -> None:
        self._verify_panel()
        CssTuningPanel(self.panel_id, css_id) 
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def button(self,
        css_id: str = CSS.BUTTON,
        cl_bg:  str = '#E1E1E1', 
        cl_hv:  str = '#D0D0D0', 
        cl_ac:  str = '#A8A8A8',
    ) -> None:
        display(HTML(f'''                
        <style>
        /* 1. Base State: Smooth, slightly rounded, ready for animation */
        .{css_id} {{
            position: relative !important;
            overflow: hidden !important; /* Contains the sweeping shine */
            background-color: {cl_bg} !important; 
            border: 1px solid rgba(255, 255, 255, 0.1) !important; /* Subtle glass border */
            border-radius: 8px !important; /* 8px is the modern standard, better than 3px */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1) !important; 
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important; 
            cursor: pointer !important;
            z-index: 1 !important;
        }}        

        /* 2. The Shine: Fast sweep across the face */
        .{css_id}::before {{
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 50% !important;
            height: 100% !important;
            background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
            transform: skewX(-20deg) !important;
            transition: all 0.5s ease-in-out !important;
            z-index: -1 !important;
        }}

        /* 3. Hover State: Glows, lifts slightly, and triggers the shine */
        .{css_id}:hover {{
            background-color: {cl_hv} !important; 
            border-color: rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 5px 8px rgba(0, 0, 0, 0.5) !important;
            transform: translateY(-1px) !important; 
        }}

        /* Triggers the light beam sweep on hover */
        .{css_id}:hover::before {{
            left: 150% !important; 
        }}

        /* 4. Active State (The Click): Satisfying tactile squish */
        .{css_id}:active {{
            background-color: {cl_ac} !important;  
            /* Deep inner shadow makes it look pressed into the screen */
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2), inset 0 4px 8px rgba(0, 0, 0, 0.4) !important; 
            transform: translateY(1px) scale(0.96) !important;
            transition: all 0.1s !important; /* Snaps fast for a responsive feel */
        }}

        /* 5. Focus State: Accessibility ring without breaking the design */
        .{css_id}:focus {{
            outline: none !important; 
        }}                  
        </style>
        '''))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def graph(self, css_id = CSS.GRAPH) -> None:
        """Graphs aligned with panel css design → prior panel css application dependency."""
        self._verify_panel()
        display(HTML(f'''
        <style>
        .{self.panel_id}.{css_id} {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 8px !important;
            box-shadow: var(--shadow-base);
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease;
        }}
        </style>
        '''))        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _verify_panel(self):
        if not hasattr(self, 'panel_id'):
            raise TypeError(
                'AIG SAFETY ► Graph design is dependent on general panel design, '
                'please define .panel() prior to graphs.'
            )        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def table(self, css_id = CSS.TABLE) -> None:
        CssData(css_id)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

class CssData:
    """INTERACTIVE TABLES (itables / DataTables.net) STYLING."""

    def __init__(self, css_id = CSS.TABLE):
        self.css_id = css_id

        self.base_style()

    def base_style(self):
        display(HTML(f'''
        <style>
        /* 1. Outer Container Wrap */
        .dt-container {{
            padding: 0px;
            background-color: transparent;
            width: fit-content;
            max-width: 100%;
            margin-left: auto;
            margin-right: auto;
        }}

        /* 2. The Interactive Table Card Base */
        table.dataTable.{self.css_id} {{
            width: 100%;
            border-collapse: separate !important;
            border-spacing: 0 !important;
            
            /* The Premium Floating AI Gradient */
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            
            /* Ambient Drop Shadow */
            box-shadow: 0 5px 10px rgba(15, 23, 42, 0.08), 0 4px 8px rgba(15, 23, 42, 0.03) !important;
            border: 1px solid #cbd5e1 !important;
            margin: 16px 0 !important;
        }}

        /* 3. The Headers */
        table.dataTable.{self.css_id} thead th {{
            background-color: rgba(241, 245, 249, 0.8) !important;
            color: #475569 !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
            font-size: 11.5px !important;
            letter-spacing: 0.06em !important;
            padding: 14px 16px !important;
            border-bottom: 2px solid #cbd5e1 !important;
        }}

        /* 4. Cell Geometry & Fonts */
        table.dataTable.{self.css_id} tbody th,
        table.dataTable.{self.css_id} tbody td {{
            padding: 10px 16px !important; 
            border-bottom: 1px solid #e2e8f0 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
            font-size: 12.5px !important;
            color: #334155 !important;
            background-color: transparent !important; 
        }}

        /* 5. High-Visibility Zebra Striping */
        table.dataTable.{self.css_id} tbody tr:nth-child(even) {{
            background-color: rgba(15, 23, 42, 0.04) !important;
        }}
        table.dataTable.{self.css_id} tbody tr:nth-child(odd) {{
            background-color: transparent !important;
        }}

        /* Clean corner rendering on the last row */
        table.dataTable.{self.css_id} tbody tr:last-child td {{
            border-bottom: none !important;
        }}

        /* 6. Glowing Interactive Row Hover */
        table.dataTable.{self.css_id} tbody tr:hover td {{
            background-color: rgba(99, 102, 241, 0.08) !important; 
            transition: background-color 0.1s ease-in-out;
        }}
        </style>
        '''))
    
    def ui_controls(self):
        """UI CONTROLS (Search Bar, Page Selectors, Pagination Buttons)"""
        display(HTML(f'''
        <style>                                
        /* Search boxes and Entries selectors dropdowns */
        .dt-search input, 
        .dt-length select {{
            border: 1px solid #cbd5e1 !important;
            border-radius: 6px !important;
            padding: 6px 12px !important;
            background-color: #ffffff !important;
            color: #334155 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 13px !important;
            outline: none !important;
        }}
        
        .dt-search input:focus, 
        .dt-length select:focus {{
            border-color: #6366f1 !important; 
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        }}

        /* Info text ("Showing 1 to 10 of 50 entries") */
        .dt-info {{
            font-family: 'Inter', sans-serif !important;
            font-size: 13px !important;
            color: #64748b !important;
            padding-top: 12px !important;
        }}

        /* Pagination controls */
        .dt-paging .dt-paging-button {{
            font-family: 'Inter', sans-serif !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
            border: 1px solid #e2e8f0 !important;
            background: #ffffff !important;
            color: #475569 !important;
            margin: 0 2px !important;
            padding: 5px 10px !important;
        }}

        .dt-paging .dt-paging-button.current {{
            background: #6366f1 !important; 
            color: #ffffff !important;
            border-color: #6366f1 !important;
        }}

        .dt-paging .dt-paging-button:hover:not(.current) {{
            background: #f1f5f9 !important;
            color: #0f172a !important;
            border-color: #cbd5e1 !important;
        }}
        </style>
        '''))

class CssTuningPanel:
    """
    Panel extension for Tuning Selection Step.
    NOTE When consistency of panels requred, panel_id has to be same as CssPanel ID.
    """

    def __init__(self, 
        panel_id:   str = CSS.PANEL, 
        css_fes_id: str = CSS.TUNING
    ):
        self.panel_id   = panel_id
        self.css_fes_id = css_fes_id

        self.choice_style()

    def get_css_id(self) -> str:
        return self.panel_id

    def choice_style(self) -> None:
        """Final Tuning Choice Panel Style."""
        c = TUNING.C
        b = TUNING.B
        display(HTML(f'''
        <style>    
        /* The Container: Aligned with panel width and spacing */
        .{self.panel_id} .{c} {{
            display: flex;
            width: 100%;
            gap: 12px;
            justify-content: space-between;
            align-items: center;
            margin: 15px 0;
        }}

        /* The Button: Styled using the Panel's Theme Variables */
        .{self.panel_id} .{b} {{
            flex: 1;
            text-align: center;
            padding: 8px 0;
            background-color: var(--accent) !important;
            color: rgb(244, 244, 244) !important;
            border-radius: 6px;
            text-decoration: none !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 13px;
            font-weight: 600;
            transition: var(--transition); 
            border: 1px solid transparent;
            cursor: pointer;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        /* Hover state */
        .{self.panel_id} .{b}:hover {{
            filter: brightness(1.1);
            box-shadow: 0 0 0 5px var(--accent-glow) !important;
            transform: translateY(-1px);
        }}

        .{self.panel_id} .{b}:active {{
            transform: scale(0.98);
        }}
        </style>        
        '''))        

class CssPanel:

    def __init__(self, 
        panel_id: str = CSS.PANEL, 
        theme:    str = 'dark'
    ):
        self.panel_id = panel_id
        
        # 1. Inject the structural CSS first (Layout, Z-Index, Jupyter Fixes)
        self.base_style()
        self.choice_style()        

        # 2. Inject only the colors and shadows for the selected theme
        if theme == THEME.DARK    : self.dark_theme()
        if theme == THEME.DARK2   : self.dark_theme()
        if theme == THEME.LIGHT   : self.light_theme()
        if theme == THEME.LIGHT2  : self.light_theme()
        if theme == THEME.LIGHT3  : self.light_theme()
        if theme == THEME.METAL   : self.metal_theme()
        if theme == THEME.METAL2  : self.metal_theme()
        if theme == THEME.PRO     : self.pro_theme()
        if theme == THEME.PRO2    : self.pro_theme()

    def get_css_id(self):
        return self.panel_id
    
    def base_style(self) -> None:
        """Contains all structural CSS, Z-index logic, and Jupyter overrides common to ALL themes."""
        display(HTML(f'''            
        <style>
        /* Main Panel Container */
        .{self.panel_id} {{
            box-sizing: border-box !important;            
            width: 100% !important;
            max-width: 1070px !important;
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 12px; 
            padding: 24px;
            box-shadow: var(--shadow-base);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            transition: var(--transition);
        }}
        
        /* Forces vertical spacing between stacked widgets in VBoxes */
        .{self.panel_id} .widget-vbox > .jupyter-widgets {{
            margin-top: 12px !important;
            margin-right: 50px !important;
            margin-bottom: 12px !important;
            margin-left: 0px !important;
        }}

        /* Header & distinct background */
        .{self.panel_id} .widget-html,
        .{self.panel_id} .widget-html-content {{
            width: 100% !important;
            max-width: 100% !important;
            display: block !important;
            margin: 0 !important;
            padding: 0 !important;
        }}        
        .{self.panel_id} .header {{
            margin: -24px -24px 20px -24px !important;
            padding: 14px 24px !important;               
            background: var(--header-bg) !important;
            border-bottom: 1px solid var(--border) !important;
            border-radius: 12px 12px 0 0 !important;
            display: block !important;
            width: auto !important; 
        }}

        .{self.panel_id} .header .panel-title {{
            font-size: 0.8rem !important; 
            margin: 0 !important; 
            font-weight: 600 !important;
            color: var(--text-main); 
        }}
        .{self.panel_id} .header .panel-title span {{ 
            color: var(--accent); 
        }}

        /* The Labels */
        .{self.panel_id} .widget-label {{
            color: var(--text-main) !important;
            font-weight: 450;
            font-size: 13px;
            text-align: right;
            transition: var(--transition);
        }}

        /* Hover Highlight */
        .{self.panel_id} .widget-hbox:hover > .widget-label,
        .{self.panel_id} .widget-inline-hbox:hover > .widget-label {{
            color: var(--text-main) !important;
            text-shadow: 0 0 0.5px var(--text-main), 0 0 0.5px var(--text-main) !important;                
            transition: all 0.2s ease !important;
        }}

        /* Readouts */
        .{self.panel_id} .widget-readout {{
            background: var(--readout-bg) !important;
            color: var(--readout-text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            font-family: 'Courier New', monospace !important;
            font-size: 12px !important;
            font-weight: 600;
            padding: 4px 8px !important;
            text-align: center !important;
            box-shadow: none !important;
            margin-left: 15px !important; 
            transition: var(--transition);
            outline: none !important;
        }}

        .{self.panel_id} .widget-readout:hover, 
        .{self.panel_id} .widget-readout:focus, 
        .{self.panel_id} .widget-readout:active {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px var(--accent-glow) !important;
            background: var(--bg-panel) !important; 
        }}

        /* Sliders */
        .{self.panel_id} {{
            --jp-widgets-slider-handle-background: var(--accent) !important;
            --jp-widgets-slider-handle-border-color: transparent !important;
            --jp-widgets-slider-active-background: var(--accent) !important;
            --jp-widgets-slider-track-background: var(--border) !important;
        }}

        .{self.panel_id} .ui-slider, 
        .{self.panel_id} .noUi-target,
        .{self.panel_id} .noUi-base {{
            background: var(--border) !important;
            border: none !important;
            height: 2px !important;
            margin-top: 2px !important; 
        }}
        .{self.panel_id} .noUi-connect {{
            background: var(--accent) !important;
        }}

        .{self.panel_id} .ui-slider-handle,
        .{self.panel_id} .noUi-handle {{
            background: var(--accent) !important; 
            border: none !important;
            outline: none !important;
            width: 14px !important;
            height: 14px !important;
            border-radius: 50% !important;
            margin-left: -7px !important;   /* Centers 16px handle on the point */
            margin-top: 0px !important;    /* Vertically centers handle on track */
        }}

        .{self.panel_id} .ui-slider-handle:hover,
        .{self.panel_id} .noUi-handle:hover {{
            transform: scale(1.2);
            box-shadow: 0 0 0 4px var(--accent-glow) !important;
        }}        

        /* Text & Number Inputs */
        .{self.panel_id} input[type="number"],
        .{self.panel_id} input[type="text"] {{
            background: var(--bg-panel) !important;
            color: var(--text-main) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
            font-size: 12px !important;
            padding: 4px 8px !important;
            outline: none !important;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.02) !important;
            transition: var(--transition) !important;
        }}

        .{self.panel_id} input[type="number"]:hover,
        .{self.panel_id} input[type="text"]:hover,
        .{self.panel_id} input[type="number"]:focus,
        .{self.panel_id} input[type="text"]:focus {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px var(--accent-glow) !important;
        }}

        /* SelectionRangeSlider Specific */
        .{self.panel_id} .ui-slider-range {{
            background: var(--accent) !important;
            border-radius: 10px !important;
        }}
        .{self.panel_id} .widget-type-selection-range-slider .widget-readout {{
            min-width: 140px !important; 
        }}
        </style>
        '''))  

    def choice_style(self) -> None:
        """Checboxes and Radio buttons."""
        display(HTML(f'''
        <style>                   
        /* 1. Base Reset for both */
        .{self.panel_id} input[type="checkbox"],
        .{self.panel_id} input[type="radio"] {{
            -webkit-appearance: none !important;
            appearance: none !important;
            background: var(--bg-panel) !important;
            border: 1px solid var(--border) !important;
            width: 16px !important;
            height: 16px !important;
            cursor: pointer !important;
            outline: none !important;
            position: relative !important;
            margin: 0 0px 0 15px !important;
            transition: all 0.2s ease !important;
            flex-shrink: 0 !important;
        }}

        /* Hover Effects */
        .{self.panel_id} input[type="checkbox"]:hover,
        .{self.panel_id} input[type="radio"]:hover {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px var(--accent-glow) !important;
        }}

        /* CHECKBOX SPECIFIC */
        .{self.panel_id} input[type="checkbox"] {{
            border-radius: 4px !important;
        }}
        
        /* Checked State Background */
        .{self.panel_id} input[type="checkbox"]:checked {{
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }}
        
        /* The Custom Checkmark Icon */
        .{self.panel_id} input[type="checkbox"]:checked::after {{
            content: '' !important;
            position: absolute !important;
            left: 4px !important;
            top: 1px !important;
            width: 4px !important;
            height: 8px !important;
            border: solid var(--bg-panel) !important;
            border-width: 0 2px 2px 0 !important;
            transform: rotate(45deg) !important;
        }}

        /* RADIO SPECIFIC */
        .{self.panel_id} input[type="radio"] {{
            border-radius: 50% !important;
        }}
        
        /* Checked State (Creates the classic "Dot" effect) */
        .{self.panel_id} input[type="radio"]:checked {{
            border: 5px solid var(--accent) !important;
            background: var(--bg-panel) !important;
        }}

        /* Jupyter Structural Fixes for Radio Groups */
        .{self.panel_id} .widget-radio .widget-radio-box {{
            display: inline-flex !important;
            flex-direction: column !important;
            gap: 6px !important;
            margin-top: 4px !important;
        }}
        
        .{self.panel_id} .widget-radio label {{
            display: flex !important;
            align-items: center !important;
            gap: 8px !important; 
            color: var(--text-main) !important;
            font-size: 13px !important;
            cursor: pointer !important;
        }}

        /* button first*/
        .{self.panel_id} .widget-radio label input {{
            order: -1 !important; 
        }}

        /* Hover Highlight */
        .{self.panel_id} .widget-radio label:hover {{
            color: var(--text-main) !important;
            text-shadow: 0 0 0.5px var(--text-main), 0 0 0.5px var(--text-main) !important;                
            transition: all 0.2s ease !important;
        }}

        /* Hides empty labels and the dreaded 'null' tooltips */
        .{self.panel_id} .widget-radio label:empty {{
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            pointer-events: none !important;
        }}        
        </style>        
        '''))        

    def light_theme(self) -> None:
        """Elevated Light Theme (Contrast + Depth + Premium Header)"""
        display(HTML(f'''            
        <style>
        .{self.panel_id} {{
            --bg-panel: linear-gradient(135deg, #f8f9fc 0%, #ffffff 100%);
            --header-bg: #f6f8fa;
            --border: #e1e4e8;
            --text-main: #24292f;
            --accent: #0969da;
            --accent-glow: rgba(9, 105, 218, 0.15);
            --track-bg: #ffffff; 
            --readout-bg: #f6f8fa;
            --readout-text: #0969da;
            --shadow-base: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        </style>
        '''))  
    
    def dark_theme(self) -> None:
        """Standard GitHub/Vercel Dark Mode Overrides"""
        display(HTML(f'''            
        <style>
        /* Base Variables mapped from Configurator Dark Theme */
        .{self.panel_id} {{
            --bg-panel: #161b22;
            --header-bg: rgba(22, 27, 34, 0.95); 
            --border: #30363d;
            --text-main: #c9d1d9;
            --accent: #58a6ff;
            --accent-glow: rgba(88, 166, 255, 0.15);
            --track-bg: #0d1117; 
            --readout-bg: #0d1117;
            --readout-text: #58a6ff;
            --shadow-base: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        /* Subtle green hint on the readouts to match the source's --success */
        .{self.panel_id} .widget-readout,
        .{self.panel_id} input[type="number"],
        .{self.panel_id} input[type="text"] {{
            border: 1px solid rgba(63, 185, 80, 0.2) !important;
        }}
        </style>
        '''))

    def metal_theme(self) -> None:
        """Precision Titanium / Brushed Metal Panel"""
        display(HTML(f'''            
        <style>
        /* Base Variables mapped from Configurator Metal Theme */
        .{self.panel_id} {{
            --bg-panel: linear-gradient(145deg, #f4f6f9, #dde2e8);
            --header-bg: transparent; /* Bleeds panel gradient through */
            --border: #c9d1dc;
            --text-main: #1e293b;
            --accent: #5C6C8A; /* Colder titanium */
            --accent-glow: rgba(167, 175, 184, 0.4);
            --track-bg: #cbd5e1;
            --readout-bg: #eef1f5;
            --readout-text: #059669; /* Success green */
            
            /* Complex outer shadows for brushed metal depth */
            --shadow-base: 6px 6px 14px rgba(163, 177, 198, 0.45), 
                           -6px -6px 14px rgba(255, 255, 255, 0.9), 
                           inset 0 1px 0 rgba(255,255,255,0.9), 
                           inset 0 -1px 0 rgba(163,177,198,0.2);
            --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        /* Etched Header Separation */
        .{self.panel_id} .header {{
            border-bottom: none !important;
            box-shadow: inset 0 -1px 0 rgba(163, 177, 198, 0.25), 
                        inset 0 1px 0 rgba(255,255,255,0.8) !important;
        }}

        /* Inset LCD Screen depth for Readouts & Inputs */
        .{self.panel_id} .widget-readout,
        .{self.panel_id} input[type="number"],
        .{self.panel_id} input[type="text"] {{
            border: none !important;
            box-shadow: inset 2px 2px 4px rgba(163, 177, 198, 0.5), 
                        inset -2px -2px 4px rgba(255, 255, 255, 0.95), 
                        0 0 0 1px rgba(255,255,255,0.6) !important;
        }}

        /* Engraved Slider Track */
        .{self.panel_id} .ui-slider {{
            box-shadow: inset 1px 1px 2px rgba(163, 177, 198, 0.5), 
                        inset -1px -1px 2px rgba(255,255,255,0.7) !important;
        }}

        /* Metallic Slider Handle */
        .{self.panel_id} .ui-slider-handle {{
            background: linear-gradient(to bottom, #f9fafb, #cbd5e1) !important;
            border: 1px solid #94a3b8 !important;
            box-shadow: 0 2px 5px rgba(163, 177, 198, 0.5), 
                        inset 0 1px 0 rgba(255,255,255,0.9) !important;
        }}
        </style>
        '''))
            
    def pro_theme(self) -> None:
        """Neon/Cyber AI 2030 Overrides"""
        display(HTML(f'''            
        <style>
        /* Base Variables mapped from Configurator Pro Theme */
        .{self.panel_id} {{
            --bg-panel: #070911;
            --header-bg: rgba(7, 9, 17, 0.95);
            --border: #132536;
            --text-main: #e2f1f8;
            --accent: #00e5ff;
            --accent-glow: rgba(0, 229, 255, 0.3);
            --track-bg: #030409;
            --readout-bg: rgba(0, 229, 255, 0.05);
            --readout-text: #00ffaa;
            --shadow-base: 0 0 30px rgba(0, 229, 255, 0.03);
            --transition: all 0.2s ease;
        }}

        /* Glowing Header Text */
        .{self.panel_id} .header h1 {{
            letter-spacing: 0.5px !important;
        }}
        .{self.panel_id} .header h1 span {{
            text-shadow: 0 0 10px rgba(0, 229, 255, 0.5) !important;
        }}

        /* Neon Glow for Readouts & Inputs */
        .{self.panel_id} .widget-readout,
        .{self.panel_id} input[type="number"],
        .{self.panel_id} input[type="text"] {{
            border: 1px solid rgba(0, 255, 170, 0.3) !important;
            box-shadow: 0 0 8px rgba(0, 255, 170, 0.1) !important;
        }}
        
        /* Intensify glow on hover/focus */
        .{self.panel_id} .widget-readout:hover,
        .{self.panel_id} input[type="number"]:hover,
        .{self.panel_id} input[type="text"]:hover {{
            box-shadow: 0 0 12px rgba(0, 255, 170, 0.3) !important;
            border-color: #00ffaa !important;
        }}

        /* Glowing Slider Handle */
        .{self.panel_id} .ui-slider-handle {{
            box-shadow: 0 0 8px var(--accent) !important;
        }}
        </style>
        '''))