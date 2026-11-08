"""
Visual Representation Engine for Model State and Metadata Reporting.

This module implements the final aesthetic layer of AIDflow, transforming 
internal configuration states and model metadata into high-fidelity HTML 
representations. It serves as the primary mechanism for communicating a 
model's "frozen state"—displaying exactly which hyperparameters, settings, 
and versions were used to produce a specific result.
"""

from ...deps.types import (
    get_args,
    get_origin,
    Union,
    UnionType
)
from ...display.rgb import (
    RgbANSI
)
from ..factory import (
    DictClass, 
    Dataclass
)
from ..accessor import (
    Rgb
)

class REPR(DictClass):
    T = 'title'
    H = 'highlight'

class ReprUtils:
    """
    Geometric and normalization utilities for UI layout calculations.
    
    This class provides the underlying logic for 'Content-Aware Scaling', ensuring 
    that HTML containers expand to fit their contents without breaking the 
    overall design system's alignment.
    """

    def __init__(self):
        self.extra_lenght = 0

    def _adj_len(self, length:int):        
        x = int(12*(length-20)) if length > 20 else 0
        self.extra_lenght = min(250, max(self.extra_lenght, x))

    def adj_max_len(self, val):
        """
        Analyzes value length and updates the global layout width requirement.
        
        This method recursively inspects values (including nested dictionaries) 
        to determine the maximum required character width, then calculates an 
        additional pixel offset to prevent text wrapping in the rendered HTML.

        Args:
            val (Any): The value or dictionary of values to analyze.

        Returns:
            Any: A normalized version of the input for display purposes.
        """
        if not isinstance(val, dict):
            l = len(str(val))
            self._adj_len(l)
            return val
                
        v_f = '{<br>'
        for k,v in val.items():
            v_f += f'{k}: {v}<br>'
            l = len(str(k)) + len(str(v))
            self._adj_len(l)
        return v_f + ' }'
    
    @staticmethod
    def _norm_val(val):
        r = repr(val)
        if (
            sum([x in r for x in ['<br>' , '\n']]) or 
            r.count('.') > 1
        ):
            return type(val).__name__
        return val
    
    @classmethod
    def normalize_val(cls, val):
        """
        Standardizes complex Python objects into a clean string representation.
        
        To prevent UI breakage, this method converts large collections or 
        complex objects (like DataFrames or custom classes) into their type names 
        if they contain line breaks or multiple decimals, ensuring the summary 
        tables remain concise and legible.
        """         
        if not isinstance(val, (list, tuple, dict)):
            return cls._norm_val(val)
        
        if isinstance(val, dict):
            d = {}
            for k,v in val.items():
                d[k] = cls._norm_val(v)
            return d
        
        t = type(val)
        c = []
        for x in val:
            c.append(cls._norm_val(x))
        return t(c)

class ReprCreator:
    """
    Synthesizer for professional HTML representations of model configurations.
    
    This class takes a model's configuration dictionary and metadata, then 
    constructs a stylized 'Model Card'. These cards serve as the visual record 
    of a model's state, including its identity, hyperparameters, and 
    associated help documentation.

    Attributes:
        title (dict): Contains the primary name (`title`) and optional highlight (`highlight`).
        prm_cfg (dict): The resolved parameter configuration to be tabled.
        help (str): Documentation string associated with the model state.
    """

    def __init__(self, title:dict, prm_cfg:dict[str, dict], dynamic_help:str):
        self.title   = title
        self.prm_cfg = prm_cfg
        self.help    = dynamic_help
        self._hid    = f'helpDiv_{id(self)}'
        self._utl    = ReprUtils()

    def _repr_font(self, size:int=13, color:str='#333') -> str:
        return f"font-family: 'Segoe UI', Arial, sans-serif; font-size:{size}px; line-height:1.5; color:{color};"

    def _repr_name(self) -> str:
        h = self.title.get(REPR.H)
        h = f'<i>{h}</i>' if h else '<div></div>'
        return f"""
            <div style="font-size:16px; font-weight:bold; color:#3366cc; 
            margin:10px 0; font-family:'Segoe UI', Arial, sans-serif;">
                {self.title[REPR.T]} 
            </div>
            {h}
        """    
    
    def _repr_table(self, p_c:str) -> str:
        prm_cfg = self.prm_cfg
        param_table = ""
        for key, value in prm_cfg[p_c].items():
            value = self._utl.normalize_val(value)
            value = self._utl.adj_max_len(value)

            param_table += f"""
            <tr><th style='text-align: left; padding: 5px;'>
            {key}</th><td style='padding: 5px;'>{value}</td></tr>
            """
        return f"""
            <h5 style="color: #3366cc;">{p_c}</h5>
            <table style="width:100%; border-collapse: collapse; margin-bottom:15px; {self._repr_font()}">
                {param_table}
            </table>
        """          

    def _repr_help(self) -> str:
        return f"""
            <br>
            <button 
                onclick="document.getElementById('{self._hid}').style.display = 
                         document.getElementById('{self._hid}').style.display === 'none' ? 'block' : 'none';"
                style="background:#3366cc; color:white; padding:5px 10px; border:none; 
                       border-radius:4px; cursor:pointer;">
                Show Help
            </button>
            
            <div style="width: 100%; box-sizing: border-box; margin-top: 15px; margin-left: 0; 
             {self._repr_font()}">
                <div id="{self._hid}" style="display:none; margin-top:10px; 
                 padding:10px; border-radius:4px; border:1px solid #ddd; width: 100%; 
                 box-sizing: border-box; background:#f0f0f0;">
                    {Rgb.convert_ansi_to_html(''.join(self.help))}
                </div>
            </div>
        """
    
    def html(self):
        """
        Generates a complete, standalone HTML snippet of the model's summary card.
        
        This method orchestrates the construction of the name header, 
        the parameter tables, and the interactive help toggle, wrapping them 
        in a responsive container with dynamically calculated widths.

        Returns:
            str: A fully styled HTML string ready for rendering in a Jupyter cell.
        """        
        name = self._repr_name()
        
        prm_cfg = ''
        for p in self.prm_cfg:
            prm_cfg += self._repr_table(p)
        
        button = self._repr_help()
        return f"""
            <div style="border:1px solid #ccc; border-radius:5px; padding:15px; 
            max-width: {int(300 + self._utl.extra_lenght)}px; margin-top:10px; margin-left: 0; 
            background:#f9f9f9;">
                {name + prm_cfg}
            </div>
            <div> {button} </div>
        """

class ModelRepr:

    def __init__(self, model: type):
        from ..base import ModelBase, FIT
        from ..base.model import ModelState

        self.model: ModelBase = model
        self.FIT = FIT

        self._path  = f'{type(model).__module__}.{type(model).__name__}'
        self._state = ', '.join(Dataclass.get_names(ModelState))
                                                   
    def help(self, is_worker:bool=False):
        """Worker-Model repr."""
        worker = "worker's " if is_worker else ''   
        io = (
            f"\n  • {self.FIT.SPEC}   → model's input / output / meta"
            f"\n  • result → model's output access"
        ) if self.FIT.SPEC in self.model.__dict__ else ''
        
        return (
           f"► {self.model.MODEL} {worker}model at {self._path}:"
            "\n  • transform(data) → application on new data"
           f"{io}"
            "\n  • fitter → model's internal fitter"
            "\n  • state  → model's internal state"
           f"\n  • state's attrs: {self._state}"
            "\n    → state.fitter   → internal fitter view"
            "\n    → state.stats    → fitter metadata view"
            "\n    → state.report.show() → report view"
        )
    
class InputsRepr:

    @staticmethod
    def _get_name(anno) -> str:
        a = lambda a: a.__name__ if hasattr(a, '__name__') else str(a)
        o = get_origin(anno)
        if o in (UnionType, Union):
            args = get_args(anno)
            return ' | '.join([a(arg) for arg in args])
        return a(anno)

    @classmethod
    def get_params(cls, inputs: type) -> dict:         
        from ..base.model import ModelConfigOverrides
        prm = {
            'data inputs - types':   inputs,
            'custom config - types': ModelConfigOverrides
        }
        return {
                p: {i: cls._get_name(a) for i,a in inputs.__annotations__.items()}
            for p, inputs in prm.items()
        }

    @staticmethod
    def help() -> str:
        """Model Inputs repr."""
        return (
           f"\n► {RgbANSI.bold('Available custom overrides in the inputs_class:')}"
            "\n• target    → when prototyping during ML dev, testing behavior of specific worker on various targets might be beneficial"
            "\n• time_var  → similar usecase as the target - checking temporal behavior under different time variable"
            "\n• fx_choice → inner fitter (strategy) choice with default by flow parametrization - can be changed via UI or by fx_choice"
            f"\n• {RgbANSI.blue('fx_params')} → custom configuration of worker's internal fitter critical params"
            "\n  -- parameters chosen as critical setup for ML optimization in the overall flow context"
            "\n  -- this params set can be seen as a proposition for tuning: we consider it as the worker's `moving force`"
            "\n  -- similarly to fx_config, fx_params targets the internal fitter instance state"
            f"\n• {RgbANSI.blue('fx_config')} → custom configuration of worker's internal fitter"
            "\n  -- for instance, in outliers screening the worker's internal fitter can be IsolationForest from scikit"
            "\n  -- fx_config provides direct link to the final fitter responsible for the ML training"
            "\n  -- in our example of IsolationForest we can provide various params, e.g."
            "\n  -- fx_config = dict("
            "\n  n_estimators = 500,"
            "\n  max_features = 0.8,"
            "\n  n_jobs = -1,"
            "\n  ..."
            "\n )"
            "\n• save_path → custom path to preserve worker's state when testing / tuning is in place "
            "\n• verbose   → worker's custom level of verbosity - when different than set to the pipeline config is desirable"
        )    