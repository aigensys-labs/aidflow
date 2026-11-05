from ...deps.display        import  wg, display
from ...core.parameters     import  PRM
from ...core.factory        import  DictDataclass
from ...core.display        import  PanelConstructor


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SCREENING → GUI → PANEL
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class PanelUtilsUI:

    def __init__(self, params:DictDataclass, title:str, is_ta:bool=False, desc:bool=True):
        self.prm    = params
        self.is_ta  = is_ta
        self.title  = title
        
        self.wg = PanelConstructor(
            params,
            title_e = (
                ' ► New Application to Train Data and Model by adjusting TH'.upper()
                if desc else ''
            )
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _options(self):
        """Shows Main GUI Options for ML fx fit."""
        if self.is_ta:
              definition, desc = PRM.SCREEN.th_st, 'Stationarity'
        else: definition, desc = PRM.SCREEN.th_ol, 'Outliers'
        
        desc = definition.get_panel_defaults(f'{desc} Threshold')        
        self.wg.params_to_sliders(desc)

    def _panel(self) -> None:
        """
        Shows the main panel = slider.
        
        This must be construct within slider scope to keep track of 
        created widgets by the ModelRunner.
        """
        self._options()
        self.wg.panel(self.title)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def slider(self) -> wg.FloatSlider:         
        """Provides wg.FloatSlider."""
        self._panel()
        return self.wg.sliders[self.prm.key()]

    def get_ui_setting(self):
        return self.wg.get_ui_setting()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬