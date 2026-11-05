from ...deps.data           import  np, pd
from ...deps.display        import  wg, display, HTML

from ...core.parameters     import  PRM
from ...core.factory        import  DictDataclass
from ...core.foundation     import  AigParam
from ...core.interop        import  Pd
from ...core.display        import  PanelConstructor


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SCREENING → GUI → PANEL
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class PanelUtilsUI:

    _CUSTOM_WG = '_time_start_end'

    def __init__(self, params:DictDataclass):
        self.prm = params
        self.wg  = PanelConstructor(params)
        self._set_prm_sig()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_prm_sig(self) -> None:
        from .data_loader import DataProcParamsBase as prm
        self.prm_sig = AigParam.get_definition(prm)

    def _get_time_index(self, value:str) -> float:
        p = self._time_u.searchsorted(value)
        return p / (len(self._time_u) - 1)
    
    def get_ui_setting(self) -> dict:
        """Returns actual UI values of all active panel options."""
        ui_prm  = self.wg.get_ui_setting()
        custom  = ui_prm.pop(self._CUSTOM_WG)
        sl_prm_ = [p for p in self.prm_sig if p not in ui_prm]
        sl_prm  = {x: self._get_time_index(custom[i]) for i,x in enumerate(sl_prm_)}        
        return sl_prm | ui_prm
    
    @staticmethod
    def slider(time:np.ndarray, start:int, end:int, extra_px:int) -> wg.SelectionRangeSlider:
        slider = wg.SelectionRangeSlider(
            options     =  time,
            index       = (start, end),                           
            description = 'Temporal Subset',
            layout      = {'width': f'{410+extra_px}px', 'margin':f'15px 0px 5px 33px'}, 
            style       = {'description_width': f'97px'},
            tooltip     = ('Data import temporal range. Temporal subset might be beneficial when '
                           'input data exhibits inconsistent behavior in tails.'),
            disabled    = False, 
            continuous_update = False
        )   
        slider.add_class("wide-readout-selectionrangeslider")
        return slider

    @staticmethod
    def _style(px:int=125, extra_px:int=0):
        display(HTML(f"""
            <style>
            .wide-readout-selectionrangeslider .widget-readout {{
                min-width: {px+extra_px}px !important;
                width: {px+extra_px}px !important;
                text-align: left !important;
            }}
            </style>
            """
        ))      
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def options(self, slider: wg.SelectionRangeSlider):
        """Shows Main GUI Options for ML fx fit."""    

        desc = (
              {self._CUSTOM_WG : slider}
            | PRM.LOAD.test.get_panel_defaults() 
            | PRM.LOAD.sample.get_panel_defaults()
            | PRM.LOAD.features.get_panel_defaults()
        )
        self.wg.params_to_sliders(desc)          
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _date_to_slider(d: np.ndarray) -> tuple[np.ndarray, int]:
        e = 0
        try:
            t = pd.to_datetime(d)
            r = Pd.get_time_resolution(t)
            f = '%y/%m/%d'
            if r not in ('D','h'):
                f += '|%H'
                e = 45
            return t.strftime(f).to_numpy(), e
        except: 
            return d, e
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def panel(self, stats:dict) -> bool:
        from .processor import LBL

        self._time_u, extra_px = self._date_to_slider(stats[LBL.F_TIME_U])
        self._style(extra_px=extra_px)

        slider = self.slider(
            self._time_u, 
            stats[LBL.F_TIME_S], 
            stats[LBL.F_TIME_E],
            extra_px
        )
        self.options(slider)

        print()
        self.wg.panel('DATA IMPORT PANEL')
        return True
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬