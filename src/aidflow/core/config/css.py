"""
Visual Identity System and Dynamic Theme Engine for the AIDflow User Interface.

This module implements a centralized CSS orchestration layer that manages the 
aesthetic and structural presentation of all AIDflow GUI components within 
Jupyter environments. By separating layout logic from stylistic tokens, it 
enables a highly flexible "skinning" system where the entire visual interface 
can be swapped via theme presets (e.g., DARK, METAL, PRO) without affecting 
functional behavior.

Key Design Pillars:
1. Structural Decoupling: Isolates core layout rules (Z-index, flexbox grids, 
   and spacing) in `base_style` while delegating color palettes and effects 
   to specialized theme methods.

2. Immersive Themes: Provides a range of professional visual profiles—from 
   high-contrast 'Metal' industrial interfaces to neon-accented 'Pro' AI 
   aesthetics—using CSS variables for consistent branding.

3. Jupyter Ecosystem Integration: Includes critical CSS overrides to resolve 
   common JupyterLab/Notebook rendering issues, such as widget clipping, 
   input box styling, and interactive element visibility.

4. Dynamic Component Styling: Implements a sophisticated custom UI kit for 
   checkboxes, sliders, and tooltips that replaces native OS elements with 
   a consistent, modern design language across all themes.
"""

from ...deps.display   import  display, HTML
from ..factory         import  DictClass


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# AIG PIPELINE CONFIGURATION CSS THEMES
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class THEME(DictClass):
    """Theme of UI HTML elements DARK | DARK2 | LIGHT | LIGHT2 | METAL | METAL2 | PRO | PRO2."""
    DARK    = 'dark'
    DARK2   = 'dark2'
    LIGHT   = 'light'
    LIGHT2  = 'light2'
    LIGHT3  = 'light3'
    METAL   = 'metal'
    METAL2  = 'metal2'
    PRO     = 'pro'
    PRO2    = 'pro2'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ConfigCss:
    """
    Centralized CSS orchestration layer that manages the aesthetic and structural 
    presentation of all AIDflow GUI components within Jupyter environments.
    """
    
    def __init__(self, 
        panel_id: str = 'panel-v1', 
        theme:    str = 'dark'
    ):
        self.panel_id = panel_id
        
        # 1. Inject the structural CSS first (Layout, Z-Index, Jupyter Fixes)
        self.base_style()
        self.choice_style()
        self.genesis_extras()

        # 2. Inject only the colors and shadows for the selected theme
        if theme == THEME.DARK    : self.dark_theme()
        if theme == THEME.DARK2   : self.dark_theme2()
        if theme == THEME.LIGHT   : self.light_theme()
        if theme == THEME.LIGHT2  : self.light_theme2()
        if theme == THEME.LIGHT3  : self.light_theme3()
        if theme == THEME.METAL   : self.metal_theme()
        if theme == THEME.METAL2  : self.metal_theme2()
        if theme == THEME.PRO     : self.pro_theme()
        if theme == THEME.PRO2    : self.pro_theme2()

    def base_style(self):
        """Contains all structural CSS, Z-index logic, and Jupyter overrides common to ALL themes."""
        display(HTML(f"""
        <style>
            /* Main Container Layout */
            .{self.panel_id} {{
                box-sizing: border-box !important;            
                width: 100% !important;
                max-width: 1070px !important;
                margin-bottom: 12px !important;
                background: var(--card-bg); 
                border: 1px solid var(--border); 
                border-radius: 12px;                
                width: 100%; 
                font-family: 'Segoe UI', sans-serif; 
                padding: 20px;
                overflow: visible !important;
                position: relative !important;
            }}
                        
            .{self.panel_id} .gh-header {{ 
                padding: 20px 25px; 
                border-bottom: 1px solid var(--border);
            }}
            .{self.panel_id} .gh-header h1 {{ 
                font-size: 1.2rem; margin: 0; 
                color: var(--text-main); 
                font-weight: 600; 
            }}
            .{self.panel_id} .gh-header h1 span {{ 
                color: var(--accent); 
            }}
            
            /* Tabs */
            .{self.panel_id} .gh-tabs {{ 
                display: flex; 
                background: var(--tab-bg); 
                border-bottom: 1px solid var(--border); 
                padding: 0 10px; 
                gap: 5px;
                position: relative !important;
                overflow: visible !important;
                z-index: 1 !important;                
            }}
            .{self.panel_id} .gh-tab-btn {{ 
                background: none; border: none; 
                color: var(--text-muted); 
                padding: 12px 20px !important; 
                cursor: pointer; 
                border-bottom: 2px solid transparent !important; 
                text-transform: uppercase; 
                font-size: 0.75rem; 
                font-weight: 600; 
                height: auto !important;
                min-height: 45px !important;    
                position: relative !important;
                overflow: visible !important;
                z-index: 5 !important;
                transition: all 0.3s;
            }}
            .{self.panel_id} .gh-tab-btn.active {{ 
                border-bottom: 2px solid var(--accent) !important; 
                background: var(--card-bg);            
            }}
            .{self.panel_id} .gh-tab-btn.active:hover {{
                border-bottom: 2px solid var(--accent) !important;
                background: var(--card-bg) !important;
            }}
            
            .{self.panel_id} .gh-tab-btn:focus {{ 
                outline: none !important;
            }}

            /* Rows */
            .{self.panel_id} .gh-row {{ 
                display: flex; 
                align-items: center; 
                padding: 15px 25px; 
                width: 100%;                
            }}
            .{self.panel_id} .gh-row:hover {{ 
                position: relative !important;
                overflow: visible !important;
                z-index: 100 !important;
            }}
            .{self.panel_id} .gh-label-group {{ 
                flex: 1; 
                min-width: 200px; 
            }}
            .{self.panel_id} .gh-label {{ 
                color: var(--text-main); 
                font-size: 0.85rem; 
                font-weight: 400; 
                margin-bottom: 2px;
            }}
            .{self.panel_id} .gh-desc {{ 
                color: var(--text-muted); 
                font-size: 0.75rem; 
            }}

            .{self.panel_id} .gh-value {{ 
                font-family: monospace; 
                color: var(--success); 
                padding: 2px 8px;
                border-radius: 4px; 
                margin: 0 15px; 
                min-width: 45px; 
                text-align: center; 
            }}
            
            /* Tooltip */
            .{self.panel_id} .gh-tooltip {{
                position: relative; 
                display: inline-flex; 
                align-items: center; 
                justify-content: center;
                width: 18px; 
                height: 18px;
                color: var(--accent);
                border-radius: 50%; 
                font-size: 0.7rem; 
                font-weight: bold; 
                cursor: help; 
                margin-left: 8px;
            }}
            .{self.panel_id} .gh-tooltip .tooltip-text {{
                visibility: hidden; 
                width: 250px;
                color: var(--text-main); 
                text-align: left;
                border-radius: 6px; 
                padding: 10px 12px;
                position: absolute; 
                bottom: 150%; 
                left: 50%; 
                margin-left: -125px;
                opacity: 0; 
                transition: opacity 0.2s, transform 0.2s; 
                transform: translateY(5px);
                font-weight: 400; 
                font-size: 0.8rem; 
                line-height: 1.4; 
                z-index: 9999 !important;
            }}
            .{self.panel_id} .gh-tooltip:hover .tooltip-text {{
                visibility: visible; 
                opacity: 1; 
                transform: translateY(0);
            }}

            /* --- Buttons --- */
            .{self.panel_id} .gh-btn-primary {{ 
                background: var(--accent) !important; 
                border-radius: 6px !important; 
                border: none !important; 
                margin: 25px 5px 5px 0px !important; 
                padding: 0px !important; 
                cursor: pointer !important;
                height: auto !important; 
                min-height: 45px !important; 
                width: calc(100%) !important; 
                transition: all 0.2s !important;
            }}

            .{self.panel_id} .gh-btn-primary:hover {{ 
                transform: translateY(-1px) !important; 
                box-shadow: 0 5px 8px var(--accent-glow) !important;
            }}   

            .{self.panel_id} .gh-btn-primary:active {{ 
                transform: translateY(1px) scale(0.96) !important;
                transition: all 0.1s !important;
            }}

            .{self.panel_id} .gh-btn-primary:focus {{ 
                outline: none !important;
                box-shadow: None !important;
                border: none !important;
            }}

            /* Animations */
            @keyframes slideFadeOut {{
                0% {{ opacity: 1; max-height: 50px; padding: 12px; margin-top: 15px; }}
                80% {{ opacity: 1; max-height: 50px; padding: 12px; margin-top: 15px; }}
                100% {{ opacity: 0; max-height: 0; padding: 0; margin-top: 0; overflow: hidden; border: none; }}
            }}
            .{self.panel_id} .gh-toast {{
                color: var(--success); 
                font-weight: 500; 
                border-radius: 4px;
                animation: slideFadeOut 2.5s forwards; 
            }}

            /* Jupyter Clipping Fixes */
            .{self.panel_id} .widget-hbox {{ 
                padding: 0 !important; 
                margin: 0 !important; 
                overflow: visible !important;
            }}
            .{self.panel_id} .widget-vbox {{ 
                width: 100% !important; 
                overflow: visible !important;
            }}
            .{self.panel_id} .lm-Widget,
            .{self.panel_id} .p-Widget {{
                overflow: visible !important;
            }}
            .{self.panel_id} .widget-readout {{ 
                font-family: 'Courier New', monospace !important; 
                color: var(--success) !important; 
                padding: 4px 10px !important; 
                border-radius: 4px !important; 
                min-width: 60px !important; 
                text-align: right !important; 
                box-shadow: none !important; 
                margin-left: 20px !important; 
                font-weight: 600 !important;
            }}
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
                height: 2px !important; 
                border: none !important; 
            }}
            .{self.panel_id} .ui-slider-handle,
            .{self.panel_id} .noUi-handle{{ 
                background: var(--accent) !important; 
                border-radius: 50% !important; 
                width: 14px !important;
                height: 14px !important; 
                border: none !important; 
                outline: none !important;
            }}
            .{self.panel_id} .ui-slider-handle:hover,
            .{self.panel_id} .noUi-handle:hover {{
                transform: scale(1.2);
                box-shadow: 0 0 0 4px var(--accent-glow) !important;
            }}               
        </style>
        """))

    def choice_style(self) -> None:
        """Checboxes and Radio buttons."""
        display(HTML(f'''
        <style>                   
        /* 1. Base Reset for both */
        .{self.panel_id} input[type="checkbox"] {{
            -webkit-appearance: none !important;
            appearance: none !important;
            background: var(--bg-panel) !important;
            border: 1px solid var(--border) !important;
            width: 16px !important;
            height: 16px !important;
            cursor: pointer !important;
            outline: none !important;
            position: relative !important;
            transition: all 0.2s ease !important;
            flex-shrink: 0 !important;
            border-radius: 4px !important;
            margin-left: 5px !important;
            margin-right: 5px !important;
        }}

        /* Hover Effects */
        .{self.panel_id} input[type="checkbox"]:hover {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px var(--accent-glow) !important;
        }}

        /* The Custom Checkmark Background & Icon */
        .{self.panel_id} input[type="checkbox"]:checked {{
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='20 6 9 17 4 12'%3E%3C/polyline%3E%3C/svg%3E") !important;
            background-size: 60% !important; 
            background-position: center center !important;
            background-repeat: no-repeat !important;            
        }}
        
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

        /* Hides empty labels and the dreaded 'null' tooltips */
        .{self.panel_id} .widget-checkbox .widget-label {{
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            pointer-events: none !important;        
        }}
        </style>        
        '''))        

    def genesis_extras(self):
        """Contains extra CSS for ProjectGenesis."""
        display(HTML(f"""
        <style>        
                     
        /* The Grid Container */
        .{self.panel_id} .genesis-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr; /* Two equal columns */
            gap: 20px 40px; /* 20px vertical gap, 40px horizontal gap */
            padding: 10px;
        }}
        /* Individual Input Sections */
        .{self.panel_id} .input-group {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }}
        /* The Label & Tooltip Row */
        .{self.panel_id} .input-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }}
        .{self.panel_id} .input-label {{
            color: var(--text-main);
            font-size: 0.85rem;
            font-weight: 600;
        }}
        /* The Native Text Input (Overriding standard white boxes) */
        .{self.panel_id} .pro-input {{
            width: 100%;
            background: rgba(0, 0, 0, 0.2); /* Dark immersive background */
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 10px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            transition: all 0.2s ease;
            outline: none;
        }}
        .{self.panel_id} .pro-input:focus {{
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2); /* Sleek blue glow */
            background: var(--input-bg);
        }}
        /* The Under-Input Description */
        .{self.panel_id} .input-desc {{
            color: var(--text-muted);
            font-size: 0.7rem;
            margin-top: 6px;
            font-style: italic;
        }}
        /* 1. Jupyter widget containers */
        .{self.panel_id} .widget-text input[type="text"],
        .{self.panel_id} .widget-combobox input[type="text"],
        .{self.panel_id} .widget-dropdown select {{
            background-color: var(--input-bg) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-main) !important;
            border-radius: 6px !important;
            padding: 6px 12px !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.2s ease-in-out !important;
            height: 32px !important; /* Keep them compact */
        }}

        /* 2. Focus and Hover States (The "Glow") */
        .{self.panel_id} .widget-text input[type="text"]:focus,
        .{self.panel_id} .widget-combobox input[type="text"]:focus,
        .{self.panel_id} .widget-dropdown select:focus,
        .{self.panel_id} .widget-dropdown select:hover {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2), inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            outline: none !important;
        }}

        /* 3. Style the Placeholders so they don't clash */
        .{self.panel_id} input::placeholder {{
            color: var(--text-muted) !important;
            opacity: 0.6 !important;
            font-style: italic !important;
        }}
        
        /* 4. Dropdown Specifics*/
        .{self.panel_id} .widget-dropdown select {{
            appearance: none !important; /* Hides default OS arrow */
            background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' \
            viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' \
            stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e") !important;
            background-repeat: no-repeat !important;
            background-position: right 8px center !important;
            background-size: 14px !important;
            padding-right: 28px !important;
            cursor: pointer;
        }}               
        </style>
        """))

    def dark_theme(self):
        """Standard GitHub/Vercel Dark Mode Overrides"""
        display(HTML(f"""
        <style>
            .{self.panel_id} {{
                --bg-color: #0f1115; 
                --card-bg: #161b22; 
                --tab-bg: #21262d;
                
                --accent: #58a6ff; 
                --accent-glow: rgba(88, 166, 255, 0.15);
                --text-main: #c9d1d9; 
                --text-muted: #8b949e;
                
                --border: #30363d; 
                --success: #3fb950; 
                --input-bg: #0d1117;
            }}
            .{self.panel_id} .gh-header {{ 
                background: rgba(22, 27, 34, 0.95); .
            }}
            .{self.panel_id} .gh-tab-btn.active {{ 
                color: var(--accent); 
            }}
            .{self.panel_id} .gh-tab-btn:hover {{
                color: var(--text-main) !important; 
                background: rgba(255, 255, 255, 0.03) !important;
                border-bottom: 2px solid var(--border) !important;
            }}
            .{self.panel_id} .gh-row {{ 
                border-bottom: 1px solid rgba(48, 54, 61, 0.5); 
            }}
            .{self.panel_id} .gh-row:hover {{ 
                background: rgba(255,255,255,0.01); 
            }}
            
            /* Values & Badges */
            .{self.panel_id} .gh-value {{ 
                background: var(--input-bg); 
                border: 1px solid rgba(63, 185, 80, 0.2); 
            }}
            .{self.panel_id} .widget-readout {{ 
                background: var(--input-bg) !important; 
                border: 1px solid rgba(63, 185, 80, 0.2) !important; 
            }}
            .{self.panel_id} .gh-toast {{ 
                background: rgba(16, 185, 129, 0.1); 
                border-left: 4px solid #10b981; 
            }}
            
            /* Tooltips & Buttons */
            .{self.panel_id} .gh-tooltip {{ 
                background: rgba(56, 189, 248, 0.1); 
            }}
            .{self.panel_id} .gh-tooltip .tooltip-text {{
                background-color: #1e293b; 
                color: #f8fafc; border: 1px solid rgba(56, 189, 248, 0.2);
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            }}
            .{self.panel_id} .gh-btn-primary {{ 
                color: #0d1117 !important; 
                font-weight: 700 !important; 
            }}
            .{self.panel_id} .gh-btn-primary:hover {{ 
                box-shadow: 0 0 25px rgba(56, 189, 248, 0.7) !important; 
            }}
        </style>
        """))

    def dark_theme2(self):
        """Obsidian Glass / High-End AI Dark Mode (Fixed Tooltips + Interaction)"""

        display(HTML(f"""
        <style>
        .{self.panel_id} {{
        --bg-color: #0b0e11; 
        --card-bg: #111418; 
        --tab-bg: #1a1f26;

        --accent: linear-gradient(135deg, #38bdf8, #818cf8);
        --accent-solid: #38bdf8;
        --accent-glow: rgba(88, 166, 255, 0.15);

        --text-main: #f1f5f9; 
        --text-muted: #94a3b8;

        --border: rgba(255, 255, 255, 0.08); 
        --success: #10b981; 
        --input-bg: #07090b;
        
        background: var(--bg-color) !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px);
    }}

    /* HEADER */
    .{self.panel_id} .gh-header {{ 
        background: rgba(17, 20, 24, 0.85) !important; 
        border-bottom: 1px solid var(--border) !important;
        backdrop-filter: blur(8px);
    }}

    /* TABS (NOW ALIVE) */
    .{self.panel_id} .gh-tab-btn {{
        transition: all 0.2s ease;
        border-bottom: 2px solid transparent !important;
    }}

    .{self.panel_id} .gh-tab-btn.active {{ 
        color: #f1f5f9 !important;
        background: rgba(56, 189, 248, 0.06) !important;
        border-bottom: 2px solid var(--accent-solid) !important;

        box-shadow: 0 0 10px rgba(56,189,248,0.25);
    }}

    .{self.panel_id} .gh-tab-btn:hover {{
        color: #ffffff !important;
        background: rgba(56, 189, 248, 0.08) !important;

        box-shadow: inset 0 -2px 0 rgba(56,189,248,0.6);
    }}

    /* INPUTS */
    .{self.panel_id} .gh-value, 
    .{self.panel_id} .widget-readout,
    .{self.panel_id} input,
    .{self.panel_id} select {{ 
        background: var(--input-bg) !important; 
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: #38bdf8 !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.5) !important;
        border-radius: 8px !important;
    }}

    /* BUTTON */
    .{self.panel_id} .gh-btn-primary {{ 
        background: var(--accent) !important; 
        color: #ffffff !important; 
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;

        box-shadow:
            0 4px 12px rgba(56, 189, 248, 0.25),
            0 0 20px rgba(56, 189, 248, 0.15);

        position: relative !important;
        overflow: hidden !important;

        transition: all 0.25s ease !important;
    }}

    .{self.panel_id} .gh-btn-primary::after {{
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
        transition: 0.6s;
    }}

    .{self.panel_id} .gh-btn-primary:hover {{ 
        transform: translateY(-2px);

        box-shadow:
            0 10px 25px rgba(56, 189, 248, 0.45),
            0 0 40px rgba(56, 189, 248, 0.25) !important;
    }}

    .{self.panel_id} .gh-btn-primary:hover::after {{
        left: 100%;
    }}

    /* ROWS */
    .{self.panel_id} .gh-row {{ 
        border-bottom: 1px solid rgba(255, 255, 255, 0.03) !important; 
    }}

    .{self.panel_id} .gh-row:hover {{ 
        background: linear-gradient(
            90deg,
            rgba(255,255,255,0.02),
            rgba(56,189,248,0.05),
            rgba(255,255,255,0.02)
        ) !important;
    }}

    /* TOOLTIP */
    .{self.panel_id} .gh-tooltip {{
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;

        box-shadow: 0 0 10px rgba(56,189,248,0.3);
    }}

    .{self.panel_id} .gh-tooltip .tooltip-text {{
        background: rgba(15, 23, 42, 0.95);   /* FIX: real background */
        color: #f8fafc;

        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 8px;

        box-shadow:
            0 15px 35px rgba(0, 0, 0, 0.8),
            0 0 20px rgba(56,189,248,0.2);

        backdrop-filter: blur(10px);
    }}

    /* TOAST */
    .{self.panel_id} .gh-toast {{ 
        background: #064e3b !important; 
        border: 1px solid #10b981 !important;
        color: #ecfdf5 !important;

        box-shadow:
            0 0 15px rgba(16, 185, 129, 0.25),
            inset 0 0 10px rgba(16,185,129,0.1);
    }}

    </style>
    """))

    def light_theme(self):
        """Elevated Light Theme (Contrast + Depth + Premium Header)"""

        display(HTML(f"""
        <style>
        .{self.panel_id} {{
            --bg-color: #eef2f6;
            --card-bg: linear-gradient(145deg, #ffffff, #f1f5f9);
            --tab-bg: #e6ebf2;

            --accent: #2563eb;
            --accent-glow: rgba(9, 105, 218, 0.15);
            --text-main: #0f172a;
            --text-muted: #64748b;

            --border: rgba(15, 23, 42, 0.08);
            --success: #16a34a;
            --input-bg: #f1f5f9;

            box-shadow:
                0 10px 30px rgba(15, 23, 42, 0.08),
                0 2px 6px rgba(15, 23, 42, 0.04);

            border: 1px solid rgba(15, 23, 42, 0.06);
        }}

        /* HEADER → FLOATING CONTROL BAR */
        .{self.panel_id} .gh-header {{ 
            background: linear-gradient(
                180deg,
                rgba(255,255,255,0.95),
                rgba(241,245,249,0.85)
            );

            backdrop-filter: blur(10px);

            border-bottom: 1px solid rgba(15, 23, 42, 0.08);

            box-shadow:
                0 4px 12px rgba(15, 23, 42, 0.08),
                inset 0 1px 0 rgba(255,255,255,0.9);

            position: relative;
            z-index: 2;
        }}

        /* HEADER ACCENT LINE */
        .{self.panel_id} .gh-header::after {{
            content: "";
            position: absolute;
            left: 20px;
            right: 20px;
            bottom: 0;

            height: 2px;

            background: linear-gradient(
                90deg,
                transparent,
                rgba(37, 99, 235, 0.6),
                transparent
            );

            opacity: 0.7;
        }}

        /* HEADER TEXT */
        .{self.panel_id} .gh-header h1 {{
            letter-spacing: 0.3px;
            color: var(--text-main);

            text-shadow:
                0 1px 0 rgba(255,255,255,0.6);
        }}

        .{self.panel_id} .gh-header h1 span {{
            color: var(--accent);
        }}

        /* TABS */
        .{self.panel_id} .gh-tab-btn.active {{ 
            color: var(--text-main);

            background: #ffffff;

            box-shadow:
                0 2px 6px rgba(15, 23, 42, 0.08),
                inset 0 -2px 0 var(--accent);
        }}

        .{self.panel_id} .gh-tab-btn:hover {{
            color: var(--text-main) !important;

            background: rgba(37, 99, 235, 0.06) !important;

            border-bottom: 2px solid rgba(15, 23, 42, 0.1) !important;
        }}

        /* ROWS */
        .{self.panel_id} .gh-row {{ 
            border-bottom: 1px solid rgba(15, 23, 42, 0.05);
        }}

        .{self.panel_id} .gh-row:hover {{ 
            background: linear-gradient(
                90deg,
                rgba(0,0,0,0.015),
                rgba(37, 99, 235, 0.04),
                rgba(0,0,0,0.015)
            );
        }}

        /* VALUES */
        .{self.panel_id} .gh-value {{ 
            background: rgba(22, 163, 74, 0.08); 
            border: 1px solid rgba(22, 163, 74, 0.2);

            box-shadow:
                0 1px 3px rgba(0,0,0,0.05),
                inset 0 1px 0 rgba(255,255,255,0.6);
        }}

        .{self.panel_id} .widget-readout {{ 
            background: rgba(22, 163, 74, 0.08) !important; 
            border: 1px solid rgba(22, 163, 74, 0.2) !important;

            box-shadow:
                0 1px 3px rgba(0,0,0,0.05),
                inset 0 1px 0 rgba(255,255,255,0.6) !important;
        }}

        /* TOAST */
        .{self.panel_id} .gh-toast {{ 
            background: rgba(22, 163, 74, 0.08); 
            border-left: 4px solid var(--success);

            box-shadow: 0 4px 12px rgba(22,163,74,0.15);
        }}

        /* TOOLTIP */
        .{self.panel_id} .gh-tooltip {{ 
            background: rgba(37, 99, 235, 0.12);
        }}

        .{self.panel_id} .gh-tooltip .tooltip-text {{
            background-color: #ffffff; 
            border: 1px solid rgba(15, 23, 42, 0.08);

            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
        }}

        /* BUTTON */
        .{self.panel_id} .gh-btn-primary {{ 
            background: linear-gradient(135deg, #2563eb, #4f46e5) !important;

            color: #ffffff !important; 
            font-weight: 600 !important;

            box-shadow:
                0 6px 18px rgba(37, 99, 235, 0.35),
                inset 0 1px 0 rgba(255,255,255,0.3);
        }}

        .{self.panel_id} .gh-btn-primary:hover {{ 
            box-shadow:
                0 10px 30px rgba(37, 99, 235, 0.5) !important;

            transform: translateY(-1px);
        }}

    </style>
    """))

    def light_theme2(self):
        """Crisp, Clean Light Mode Overrides"""
        display(HTML(f"""
        <style>
            .{self.panel_id} {{
                --bg-color: #f6f8fa; --card-bg: #ffffff; --tab-bg: #f3f4f6;
                --accent: #0969da; --text-main: #24292f; --text-muted: #57606a;
                --accent-glow: rgba(9, 105, 218, 0.15);
                --border: #d0d7de; --success: #1a7f37; --input-bg: #f6f8fa;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }}
            .{self.panel_id} .gh-header {{ 
                background: rgba(255, 255, 255, 0.95); 
            }}
            .{self.panel_id} .gh-tab-btn.active {{ 
                color: var(--text-main); 
            }}
            .{self.panel_id} .gh-tab-btn:hover {{
                color: var(--text-main) !important; background: rgba(0, 0, 0, 0.03) !important;
                border-bottom: 2px solid var(--border) !important;
            }}
            .{self.panel_id} .gh-row {{ 
                border-bottom: 1px solid #e1e4e8; 
            }}
            .{self.panel_id} .gh-row:hover {{ 
                background: rgba(0,0,0,0.02); 
            }}
            
            /* Values & Badges */
            .{self.panel_id} .gh-value {{ 
                background: rgba(26, 127, 55, 0.1); 
                border: 1px solid rgba(26, 127, 55, 0.2); 
            }}
            .{self.panel_id} .widget-readout {{ 
                background: rgba(26, 127, 55, 0.1) !important; 
                border: 1px solid rgba(26, 127, 55, 0.2) !important; 
            }}
            .{self.panel_id} .gh-toast {{ 
                background: rgba(26, 127, 55, 0.1); 
                border-left: 4px solid var(--success); 
            }}
            
            /* Tooltips & Buttons */
            .{self.panel_id} .gh-tooltip {{ 
                background: rgba(9, 105, 218, 0.1); 
            }}
            .{self.panel_id} .gh-tooltip .tooltip-text {{
                background-color: #ffffff; 
                border: 1px solid var(--border); 
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }}
            .{self.panel_id} .gh-btn-primary {{ 
                color: #ffffff !important; 
                font-weight: 600 !important; 
            }}
            .{self.panel_id} .gh-btn-primary:hover {{ 
                background: #0a58ca !important; 
                box-shadow: 0 4px 12px rgba(9, 105, 218, 0.3) !important; 
            }}
        </style>
        """))

    def light_theme3(self):
        """High-end JupyterLab / Premium IDE Overrides"""
        display(HTML(f"""
        <style>
            .{self.panel_id} {{
                --bg-color: #f5f5f5; 
                --card-bg: #ffffff; 
                --tab-bg: #eeeeee;
                --accent: #1976D2; /* Classic JupyterLab Active Blue */
                --accent-glow: rgba(9, 105, 218, 0.15);
                --text-main: #313131; 
                --text-muted: #757575;
                --border: #e0e0e0; 
                --success: #4CAF50; 
                --input-bg: #fcfcfc;
                
                /* Crisp, "floating paper" shadow typical of modern notebook cells */
                box-shadow: 0 2px 6px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
            }}
            .{self.panel_id} .gh-header {{ 
                background: #ffffff; 
                border-bottom: 2px solid var(--border);
            }}
            .{self.panel_id} .gh-header h1 span {{ color: #F37626; }} /* Jupyter Orange highlight */
            
            /* Tabs that look like active Notebook cells */
            .{self.panel_id} .gh-tab-btn.active {{ 
                color: var(--text-main);
                background: var(--card-bg);
                /* Signature Jupyter active cell left-border shifted to the top */
                border-top: 3px solid var(--accent) !important; 
                border-bottom: 2px solid transparent !important;
                border-left: 1px solid var(--border);
                border-right: 1px solid var(--border);
                transform: translateY(1px); /* Connects to the body seamlessly */
            }}
            .{self.panel_id} .gh-tab-btn:hover:not(.active) {{
                background: #e4e4e4 !important;
                border-bottom: 2px solid var(--border) !important;
            }}
            .{self.panel_id} .gh-row {{ border-bottom: 1px solid #f0f0f0; }}
            .{self.panel_id} .gh-row:hover {{ background: #fafbfc; }}
            
            /* Clean, IDE-like Values & Badges */
            .{self.panel_id} .gh-value, .{self.panel_id} .widget-readout {{ 
                background: var(--input-bg) !important; 
                border: 1px solid #e0e0e0 !important; 
                color: #2e7d32 !important; /* Darker green for readability */
                box-shadow: inset 0 1px 2px rgba(0,0,0,0.02) !important;
            }}
            
            /* Standard Material Design Tooltips & Buttons */
            .{self.panel_id} .gh-tooltip {{ background: rgba(25, 118, 210, 0.1); }}
            .{self.panel_id} .gh-tooltip .tooltip-text {{
                background-color: #313131; color: #ffffff; border: none;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            }}
            .{self.panel_id} .gh-btn-primary {{ 
                background: var(--accent) !important; color: #ffffff !important; 
                font-weight: 500 !important; letter-spacing: 0.5px !important;
                box-shadow: 0 2px 4px rgba(25, 118, 210, 0.3) !important; 
            }}
            .{self.panel_id} .gh-btn-primary:hover {{ 
                background: #1565C0 !important; /* Darker blue */
                box-shadow: 0 4px 8px rgba(25, 118, 210, 0.4) !important; 
            }}
            .{self.panel_id} .ui-slider {{ background: #e0e0e0 !important; height: 3px !important; }}
        </style>
        """))    

    def metal_theme(self):
        """Precision Titanium / Brushed Metal Panel"""
        display(HTML(f"""
        <style>
        .{self.panel_id} {{
            --bg-color: #e6e9ef; 
            --card-bg: linear-gradient(145deg, #f4f6f9, #dde2e8); 
            --tab-bg: #d6dbe3;

            --accent: #5C6C8A; /* colder titanium */
            --accent-glow: rgba(163, 177, 198, 0.4);
            --text-main: #1e293b; 
            --text-muted: #64748b;

            --border: #c9d1dc; 
            --success: #059669; 
            --input-bg: #eef1f5;

            /* Sharper, more metallic depth */
            box-shadow:
                6px 6px 14px rgba(163, 177, 198, 0.45),
                -6px -6px 14px rgba(255, 255, 255, 0.9),
                inset 0 1px 0 rgba(255,255,255,0.9),
                inset 0 -1px 0 rgba(163,177,198,0.2);

            border: none;
        }}

        /* Header */
        .{self.panel_id} .gh-header {{ 
            background: transparent; 
            border-bottom: none;

            /* etched separation */
            box-shadow:
                inset 0 -1px 0 rgba(163, 177, 198, 0.25),
                inset 0 1px 0 rgba(255,255,255,0.8);
        }}

        /* Tabs */
        .{self.panel_id} .gh-tabs {{ border-bottom: none; }}

        .{self.panel_id} .gh-tab-btn {{ 
            color: var(--text-muted);
        }}

        .{self.panel_id} .gh-tab-btn.active {{ 
            color: #0f172a; 
            background: var(--card-bg);

            box-shadow:
                inset 0 2px 0 var(--accent),
                inset 0 1px 0 rgba(255,255,255,0.9),
                inset 0 -1px 0 rgba(163,177,198,0.25);
        }}

        .{self.panel_id} .gh-tab-btn:hover:not(.active) {{
            color: var(--text-main) !important;
            background: rgba(255,255,255,0.35) !important;
        }}

        /* Rows */
        .{self.panel_id} .gh-row {{ 
            border-bottom: none;

            box-shadow:
                inset 0 -1px 0 rgba(163, 177, 198, 0.25),
                inset 0 1px 0 rgba(255,255,255,0.7);
        }}

        .{self.panel_id} .gh-row:hover {{ 
            background: rgba(255,255,255,0.25);
        }}

        /* Value / LCD */
        .{self.panel_id} .gh-value, 
        .{self.panel_id} .widget-readout {{ 
            background: var(--input-bg) !important; 
            border: none !important;
            color: var(--success) !important; 

            box-shadow:
                inset 2px 2px 4px rgba(163, 177, 198, 0.5), 
                inset -2px -2px 4px rgba(255, 255, 255, 0.95),
                0 0 0 1px rgba(255,255,255,0.6) !important;
        }}

        /* Buttons */
        .{self.panel_id} .gh-btn-primary {{ 
            background: linear-gradient(to bottom, #f9fafb, #cbd5e1) !important; 
            color: #0f172a !important; 
            border: 1px solid #94a3b8 !important;

            text-shadow: 0 1px 0 #ffffff;

            box-shadow:
                0 2px 4px rgba(163, 177, 198, 0.4),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
        }}

        .{self.panel_id} .gh-btn-primary:hover {{ 
            box-shadow:
                0 6px 14px rgba(163, 177, 198, 0.55),
                inset 0 1px 0 rgba(255,255,255,1) !important;
        }}

        /* Tooltip */
        .{self.panel_id} .gh-tooltip {{ 
            background: linear-gradient(145deg, #e5eaf0, #cbd5e1); 
            color: var(--accent);

            box-shadow:
                1px 1px 3px rgba(163, 177, 198, 0.5),
                -1px -1px 3px rgba(255,255,255,0.9);
        }}

        .{self.panel_id} .gh-tooltip .tooltip-text {{
            background: linear-gradient(145deg, #f8fafc, #e2e8f0);
            color: #0f172a;
            border: 1px solid #ffffff;

            box-shadow:
                6px 6px 18px rgba(163, 177, 198, 0.55);
        }}

        /* Slider */
        .{self.panel_id} .ui-slider {{ 
            background: #cbd5e1 !important;

            box-shadow:
                inset 1px 1px 2px rgba(163, 177, 198, 0.5),
                inset -1px -1px 2px rgba(255,255,255,0.7);
        }}

        .{self.panel_id} .ui-slider-handle {{ 
            background: linear-gradient(to bottom, #f9fafb, #cbd5e1) !important;
            border: 1px solid #94a3b8 !important;

            box-shadow:
                0 2px 5px rgba(163, 177, 198, 0.5),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
        }}

    </style>
    """))  

    def metal_theme2(self):
        """Industrial Precision Metal Panel (High Contrast, Machined Feel)"""

        display(HTML(f"""
        <style>
        .{self.panel_id} {{
            --bg-color: #d9dee6;

            /* STRONGER METAL SURFACE */
            --card-bg: linear-gradient(160deg, #f8fafc 0%, #d6dbe3 100%);
            --tab-bg: #cfd6df;

            --accent: #56638C;
            --accent-glow: rgba(163, 177, 198, 0.4);
            --text-main: #0f172a;
            --text-muted: #475569;

            --border: #b8c1cc;
            --success: #059669;
            --input-bg: #eef2f6;

            /* HARD SHADOW (less soft, more physical) */
            box-shadow:
                4px 4px 10px rgba(0,0,0,0.15),
                inset 0 1px 0 rgba(255,255,255,0.9),
                inset 0 -2px 3px rgba(0,0,0,0.08);

            border: 1px solid #c2c9d3;
        }}

        /* HEADER → HARD EDGE CUT */
        .{self.panel_id} .gh-header {{
            background: linear-gradient(to bottom, #f1f5f9, #dce2ea);
            border-bottom: 1px solid #aeb6c2;

            box-shadow:
                inset 0 1px 0 rgba(255,255,255,1),
                inset 0 -1px 0 rgba(0,0,0,0.15);
        }}

        /* TABS → MACHINED BLOCKS */
        .{self.panel_id} .gh-tab-btn {{
            color: var(--text-muted);
        }}

        .{self.panel_id} .gh-tab-btn.active {{
            color: #020617;

            background: linear-gradient(to bottom, #ffffff, #d1d7e0);

            border: 1px solid #aeb6c2 !important;

            box-shadow:
                inset 0 2px 0 rgba(255,255,255,1),
                inset 0 -2px 3px rgba(0,0,0,0.15),
                0 2px 4px rgba(0,0,0,0.15);
        }}

        .{self.panel_id} .gh-tab-btn:hover:not(.active) {{
            background: linear-gradient(to bottom, #eef2f6, #d6dbe3) !important;
            color: var(--text-main) !important;
        }}

        /* ROWS → ENGRAVED */
        .{self.panel_id} .gh-row {{
            border-bottom: none;

            box-shadow:
                inset 0 -1px 0 rgba(0,0,0,0.15),
                inset 0 1px 0 rgba(255,255,255,0.8);
        }}

        .{self.panel_id} .gh-row:hover {{
            background: linear-gradient(to right, rgba(255,255,255,0.35), rgba(0,0,0,0.02));
        }}

        /* VALUE → DIGITAL MODULE (VERY VISIBLE CHANGE) */
        .{self.panel_id} .gh-value,
        .{self.panel_id} .widget-readout {{
            background: linear-gradient(to bottom, #e2e8f0, #cbd5e1) !important;

            color: #065f46 !important;

            border: 1px solid #94a3b8 !important;

            box-shadow:
                inset 0 2px 4px rgba(0,0,0,0.2),
                inset 0 -1px 1px rgba(255,255,255,0.7) !important;
        }}

        /* BUTTON → PHYSICAL BUTTON (BIG VISUAL SHIFT) */
        .{self.panel_id} .gh-btn-primary {{
            background: linear-gradient(to bottom, #f8fafc, #cbd5e1) !important;

            color: #020617 !important;

            border: 1px solid #64748b !important;

            box-shadow:
                0 3px 6px rgba(0,0,0,0.25),
                inset 0 2px 0 rgba(255,255,255,0.9),
                inset 0 -2px 3px rgba(0,0,0,0.2) !important;
        }}

        .{self.panel_id} .gh-btn-primary:hover {{
            transform: translateY(1px);

            box-shadow:
                0 1px 3px rgba(0,0,0,0.3),
                inset 0 2px 3px rgba(0,0,0,0.25),
                inset 0 -1px 0 rgba(255,255,255,0.6) !important;
        }}

        /* TOOLTIP → HARD PANEL */
        .{self.panel_id} .gh-tooltip {{
            background: linear-gradient(to bottom, #e2e8f0, #cbd5e1);
            color: var(--accent);

            border: 1px solid #94a3b8;

            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        }}

        .{self.panel_id} .gh-tooltip .tooltip-text {{
            background: #f8fafc;
            border: 1px solid #cbd5e1;

            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }}

        /* SLIDER → HARD TRACK */
        .{self.panel_id} .ui-slider {{
            background: linear-gradient(to bottom, #cbd5e1, #94a3b8) !important;

            box-shadow:
                inset 0 2px 4px rgba(0,0,0,0.25),
                inset 0 -1px 0 rgba(255,255,255,0.7);
        }}

        .{self.panel_id} .ui-slider-handle {{
            background: linear-gradient(to bottom, #f8fafc, #cbd5e1) !important;

            border: 1px solid #64748b !important;

            box-shadow:
                0 2px 5px rgba(0,0,0,0.3),
                inset 0 1px 0 rgba(255,255,255,0.9);
        }}

    </style>
    """))

    def pro_theme(self):
        """Neon/Cyber AI 2030 Overrides"""
        display(HTML(f"""
        <style>
            .{self.panel_id} {{
                --bg-color: #030409; --card-bg: #070911; --tab-bg: #05060b;
                --accent: #00e5ff; --text-main: #e2f1f8; --text-muted: #5e7b94;
                --accent-glow: rgba(0, 229, 255, 0.3);
                --border: #132536; --success: #00ffaa; --input-bg: rgba(0, 229, 255, 0.05);
                box-shadow: 0 0 30px rgba(0, 229, 255, 0.03);
            }}
            .{self.panel_id} .gh-header {{ 
                background: rgba(7, 9, 17, 0.95); 
            }}
            .{self.panel_id} .gh-header h1 {{ 
                letter-spacing: 0.5px; 
            }}
            .{self.panel_id} .gh-header h1 span {{ 
                text-shadow: 0 0 10px rgba(0, 229, 255, 0.5); 
            }}
            
            .{self.panel_id} .gh-tab-btn {{ 
                letter-spacing: 1px; 
            }}
            .{self.panel_id} .gh-tab-btn.active {{ 
                color: var(--accent); 
                box-shadow: inset 0 -4px 15px -10px rgba(0, 229, 255, 0.4); 
            }}
            .{self.panel_id} .gh-tab-btn:hover {{
                color: var(--text-main) !important; 
                background: rgba(0, 229, 255, 0.03) !important;
                border-bottom: 2px solid rgba(0, 229, 255, 0.3) !important;
            }}
            .{self.panel_id} .gh-row {{ 
                border-bottom: 1px solid rgba(19, 37, 54, 0.6); 
            }}
            .{self.panel_id} .gh-row:hover {{ 
                background: linear-gradient(90deg, rgba(0, 229, 255, 0.04) 0%, transparent 100%); 
            }}
            
            /* Values & Badges */
            .{self.panel_id} .gh-value {{ 
                background: rgba(0, 255, 170, 0.08); 
                border: 1px solid rgba(0, 255, 170, 0.3); 
                box-shadow: 0 0 8px rgba(0, 255, 170, 0.1); 
            }}
            .{self.panel_id} .widget-readout {{ 
                background: var(--input-bg) !important; 
                border: 1px solid rgba(0, 255, 170, 0.3) !important; 
                box-shadow: 0 0 8px rgba(0, 255, 170, 0.1) !important;
            }}
            .{self.panel_id} .gh-toast {{ 
                background: rgba(0, 255, 170, 0.1); 
                border-left: 4px solid var(--success); 
                box-shadow: 0 0 10px rgba(0, 255, 170, 0.1); 
            }}
            
            /* Tooltips & Buttons */
            .{self.panel_id} .gh-tooltip {{ 
                background: rgba(0, 229, 255, 0.15); 
                border: 1px solid rgba(0, 229, 255, 0.3); 
                box-shadow: 0 0 5px rgba(0, 229, 255, 0.2); 
            }}
            .{self.panel_id} .gh-tooltip .tooltip-text {{
                background-color: #030409; 
                border: 1px solid var(--accent); 
                box-shadow: 0 10px 25px rgba(0, 229, 255, 0.15);
            }}
            .{self.panel_id} .gh-btn-primary {{ 
                color: #000000 !important; 
                font-weight: 800 !important; 
                letter-spacing: 1px !important; 
                box-shadow: 0 0 15px rgba(0, 229, 255, 0.3) !important; 
            }}
            .{self.panel_id} .gh-btn-primary:hover {{ 
                box-shadow: 0 0 25px rgba(0, 229, 255, 0.6), inset 0 0 10px rgba(255,255,255,0.5) !important; 
                }}
            .{self.panel_id} .ui-slider-handle {{ 
                box-shadow: 0 0 8px var(--accent) !important; 
            }}
        </style>
        """))

    def pro_theme2(self):

        display(HTML(f"""
        <style>

        .{self.panel_id} {{
            --bg-color: #05070a;
            --card-bg: rgba(15, 23, 42, 0.75);
            --tab-bg: rgba(30, 41, 59, 0.7);

            --accent: #38bdf8;              /* neon blue */
            --accent-glow: rgba(0, 229, 255, 0.3);
            --text-main: #e6edf3;
            --text-muted: #94a3b8;

            --border: rgba(148, 163, 184, 0.2);
            --success: #22c55e;
            --input-bg: rgba(2, 6, 23, 0.8);

            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;

            /* GLASS + DEPTH */
            backdrop-filter: blur(14px);
            box-shadow:
                0 0 0 1px rgba(56, 189, 248, 0.05),
                0 20px 60px rgba(0, 0, 0, 0.8);

            width: 100%;
            font-family: 'Segoe UI', sans-serif;
            padding: 20px;

            overflow: visible !important;
            position: relative !important;
        }}

        /* Header */
        .{self.panel_id} .gh-header {{
            background: rgba(15, 23, 42, 0.6);
            border-bottom: 1px solid var(--border);
        }}

        /* Tabs */
        .{self.panel_id} .gh-tab-btn:hover {{
            background: rgba(56, 189, 248, 0.08) !important;
        }}

        .{self.panel_id} .gh-tab-btn.active {{
            background: rgba(15, 23, 42, 0.9);
            box-shadow: inset 0 -2px 0 var(--accent);
        }}

        /* Rows */
        .{self.panel_id} .gh-row:hover {{
            background: rgba(56, 189, 248, 0.03);
        }}

        /* Value badge (slightly more "alive") */
        .{self.panel_id} .gh-value {{
            border: 1px solid rgba(34, 197, 94, 0.3);
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.15);
        }}

        /* Tooltip */
        .{self.panel_id} .gh-tooltip {{
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
        }}

        .{self.panel_id} .gh-tooltip .tooltip-text {{
            background-color: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(56, 189, 248, 0.25);
            box-shadow: 0 15px 40px rgba(0,0,0,0.9);
        }}

        /* Button */
        .{self.panel_id} .gh-btn-primary {{
            background: linear-gradient(135deg, #38bdf8, #6366f1) !important;
            color: #020617 !important;
        }}

        .{self.panel_id} .gh-btn-primary:hover {{
            box-shadow:
                0 0 25px rgba(56, 189, 248, 0.6),
                0 0 50px rgba(99, 102, 241, 0.4) !important;
        }}

        /* Toast */
        .{self.panel_id} .gh-toast {{
            background: rgba(34, 197, 94, 0.12);
            border-left: 4px solid #22c55e;
            color: #22c55e;
        }}

        /* Slider */
        .{self.panel_id} .ui-slider {{
            background: rgba(148, 163, 184, 0.3) !important;
        }}

        .{self.panel_id} .ui-slider-handle {{
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.8);
        }}

    </style>
    """))
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬