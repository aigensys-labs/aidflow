"""
The semantic validation engine for AIDflow pipeline parametrization.

This module provides the critical guardrails that ensure all system parameters 
adhere to their architectural design constraints. It acts as a secondary, strict 
validation layer that sits between the raw configuration input and the ML execution 
logic, preventing runtime failures by catching type mismisatches and boundary 
violations at the earliest possible stage.
"""

from ...deps.types  import FuncType, Callable, Any
from ...display.rgb import RgbANSI
from ..foundation   import AigStop


Value = int | float | str | dict | list | None


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# AIG PARAMETERS VERIFICATION
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class _Check:
    """
    Utility class for validating on AIG pipeline parameters.
    
    This class implements a flexible validation that checks parameter values
    against strict design constraints (types, min/max bounds, allowed values).

    It separates the validation logic from the user-facing reporting, allowing for
    distinct 'Hard Checks' (raising errors) and 'Help Checks' (printing guidance).

    The uses a dictionary-based configuration system where each parameter
    definition contains callable lambdas that dynamically generate validation rules
    and help messages at runtime.    
    """

    _A   = ' ► '

    _D   = 'design'
    _E   = 'extra'
    _I   = 'info'
    _IE  = 'info_e' 

    _V   = 'val'
    _VK  = 'v_kind'
    _VN  = 'v_min'
    _VX  = 'v_max'
    _VV  = 'v_values'
    _VNI = 'v_min_incl'


    @classmethod
    def _info(cls, *args):            
        """Concatenates info segments with the separator."""
        i = ''
        for a in args: 
            i += a + cls._A
        return i

    @classmethod
    def _multi_info(cls, *args):
        """Generates a header and a footer line for formatted output."""
        i = cls._info(*args)
        f = '\n' + ' '*( len(i) - len(cls._A) ) + cls._A
        return i, f
    
    @staticmethod
    def _div(l=150): 
        print('─'*l)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_none(val:Value, v_kind:list[Value]) -> tuple[list, bool]:
        if None not in v_kind:
            return v_kind, False        
        ix = v_kind.index(None)
        v_kind = v_kind[:ix] + v_kind[ix+1:]
        return v_kind, (val is None)
    
    @classmethod
    def _is_allowed_kind(cls, val:Value, v_kind:Value | list[Value]) -> bool:
        if isinstance(v_kind, type):
            v_kind = [v_kind]

        v_kind, is_none = cls._is_none(val, v_kind)

        is_kind = [isinstance(val, ins) for ins in v_kind]
        is_fnum = isinstance(val, int) if float in v_kind else False 
        # allow 0 instead of 0.0 etc. → min, max will capture bounds
        return any(is_kind + [is_fnum, is_none])
    
    @classmethod
    def _design_check(cls,
        *,
        val:        Value, 
        v_kind:     Value|tuple = float, 
        v_min:      int|float = None, 
        v_max:      int|float = None, 
        v_values:   list = None, 
        v_min_incl: bool = False,
        extra:      list = None, 
        info:       str
    ) -> None:
        """
        Core validation logic.
        
        Checks:
        1. Type consistency (or None handling).
        2. Membership in allowed value list (v_values).
        3. Numerical range constraints (min/max).
        
        Raises ValueError with the provided `info` string if any check fails.
        """            
        mn, mx, te = [False]*3
        ik = not cls._is_allowed_kind(val, v_kind)

        if not ik:          
            if v_values:
                te = (val not in v_values)
            else:                
                if v_min is not None: 
                    z = v_min_incl
                    if z: mn = (val <  v_min)            
                    else: mn = (val <= v_min)            
                if v_max is not None: 
                    mx = (val > v_max)
        
        is_raise = ik or mn or mx or te or any((extra or []))
        if is_raise: 
            AigStop.custom(RgbANSI.bold_red('INVALID SETUP') +' ► '+ info, bold=False, red=False)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def _design_value(cls, di:dict[str, Callable]) -> Value:
        """Extracts the validated value from a design dictionary."""
        return di[cls._D](None)[cls._V]

    @classmethod
    def _flexi_args(cls, x:str, di:dict[str, Callable], args:tuple):
        """
        Transforms input arguments dynamically.
        If the first argument is a Function, it executes it with the current parameter
        value, allowing for dynamic adjustment of arguments before validation.
        """        
        if not args:
            return args
        if isinstance(args[0], FuncType):
            val  = cls._design_value(di)
            FX   = args[0]
            args = args[1:]                        
            args = FX(x, val, *args)                    
            if not isinstance(args, (tuple, list)):
                args = [args]
        return args
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def base_check(cls, prm_def:dict[str, dict], step_name:str, *args):    
        """
        Executes strict validation on a param definiton (dict).
        
        Parameters
        ----------
        prm_def : dict
            Parameter mapped to its definition.
        step_name : str
            Contextual name for the validation step (used in error messages).
        *args : tuple
            Additional dynamic arguments that may modify validation context.
        
        Raises
        ------
        ValueError
            If any parameter fails validation.
        """   
        for x, di in prm_def.items():
            info, f = cls._multi_info(step_name, x)
            info = info + di[cls._I](f)
            
            design = di[cls._D](info)     
                                                            
            args_f = cls._flexi_args(x, di, args)
            design[cls._E] = args_f
            
            cls._design_check(**design)        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def help_check(cls, prm_def:dict[str, dict], step_name:str, *args):
        """
        Generates user-friendly help messages for parameters.
        
        This method does not raise errors. Instead, it prints guidance
        messages (derived from `info_extra`) if the parameter definition
        includes specific help text and the context is valid.
        
        Parameters
        ----------
        prm_def : dict
            Parameter mapped to its definition.
        step_name : str
            Contextual name for the step.
        *args : tuple
            Dynamic arguments.
        """
        for x, di in prm_def.items():  
            if not di.get(cls._IE):
                continue
                                                                
            args_f = cls._flexi_args(x, di, args)                    
            
            if any(args_f):                    
                info, f = cls._multi_info(step_name, x) 
                print(info + di[cls._IE](f))
                cls._div()    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ParamsValidator(_Check):
    """
    Parameter validation engine for the core AIG configuration system.
    
    This class serves as a factory and validator for parameter definitions stored in `Parameters`.
    It constructs validation rules based on parameter metadata (type, min/max, allowed options)
    and executes checks to ensure provided values adhere to constraints. It supports both
    single-parameter validation and batch processing of multiple parameters.
    
    Attributes
    ----------
    params : Parameters
        The instance of the core Parameters object containing the actual values to validate.
    """

    def __init__(self, param_instance: object):
        from ..parameters import Parameters
        self.params: Parameters = param_instance

    @staticmethod
    def _param(definition):
        from ..parameters import ParamBase
        df:ParamBase = definition
        return df

    @classmethod
    def _kind(cls, definition:dict) -> Value:
        return cls._param(definition).kind

    @classmethod
    def _min(cls, definition:dict) -> dict:
        df = cls._param(definition)
        return {cls._VN: df.get_v_min()}

    @classmethod
    def _max(cls, definition:dict) -> dict:
        df = cls._param(definition)
        return {cls._VX: df.get_v_max()}
                
    @classmethod
    def _min_max(cls, definition:dict) -> dict:
        return cls._min(definition) | cls._max(definition)
        
    @classmethod    
    def _is_num(cls, definition:dict) -> bool:
        kind = cls._kind(definition)
        num  = (float, int)
        if isinstance(kind, type):
            return kind in num
        is_num = False
        for k in kind:
            is_num = is_num or (k in num)
        return is_num
    
    @classmethod    
    def _nx(cls, definition:dict, min_max:bool, mini:bool) -> dict:
        if not cls._is_num(definition) :
            return {}
        if min_max: return cls._min_max(definition)
        if mini:    return cls._min(definition)

    @classmethod
    def _design(cls,
        definition: dict, 
        prm_value:  Any, 
        *,
        mini:     bool = False,
        min_max:  bool = True,
        min_incl: bool = True,
        values:   list = None,
        kwargs:   dict = None
    ) -> Callable:
        """
        Constructs the validation rule generator.
        Returns a lambda that accepts an `info` string and returns a dictionary
        ready for `_Check._design_check`.
        """    
        nx = cls._nx(definition, min_max, mini)
        return lambda info: nx | (kwargs or {}) | {
            cls._V:     prm_value, 
            cls._VK:    cls._kind(definition), 
            cls._VNI:   min_incl, 
            cls._VV:    values,
            cls._I:     info
        }
    
    def define(self, 
        definition: dict, 
        *,
        info:       str  = None,
        info_extra: str  = None, 
        mini:       bool = False,
        min_max:    bool = True,
        min_incl:   bool = True,
        values:     list = None,
        **kwargs
    ) -> dict | Callable:
        """
        Factory method to create definition of parameter verification.
        
        Parameters
        ----------
        definition : dict
            The raw parameter definition from the core config.
        prm_value : Any
            The actual value of the parameter to be validated.
        info : str, optional
            If provided, returns a ready-to-use dict for a single parameter check.
        info_extra : str, optional
            If provided, enables the "Help Check" mode to print guidance messages.
        mini : bool
            If True, only validates the minimum value.
        min_max : bool
            If True, validates both min and max values.
        min_incl : bool
            If True, the minimum bound is inclusive.
        values : list, optional
            Explicit list of allowed values. Overrides definition options.
        
        Returns
        -------
        dict | Callable
            If `info` is provided, returns a dictionary formatted for `base_check`.
            If `info` is None, returns a callable function to be used with `extend()`.
        """
        df= self._param(definition)
        p = df.param
        k = dict(
            mini     = mini, 
            min_max  = min_max, 
            min_incl = min_incl, 
            values   = values if values is not None else df.get_v_options() # direct > indirect 
        )
        d = lambda kw: self._design(
            definition, 
            getattr(self.params, df.param),
            **(k | kw), # kw direct group shared > idiv. kwargs
            kwargs = kwargs 
        )
        v = lambda info, kw = {}: {
            p : {
                self._D:  d(kw), 
                self._I:  info, 
                self._IE: info_extra
            }}
        if info is not None:
            return v(info)  # individual use of .param()
        return v            # group use of .param() via .extend()

    @staticmethod
    def extend(info:str, *args, **kwargs_shared) -> dict:
        """
        Aggregates multiple parameter definitions into a single batch dictionary.
        
        This allows processing a group of parameters together, sharing a common
        context (`info`) and dynamic arguments (`kwargs_shared`).
        
        Parameters
        ----------
        info : str
            The context string used to generate error/help messages.
        *args : callables
            Variable arguments returned by `Verify.define` (without `info` set).
        **kwargs_shared : dict
            Shared keyword arguments passed to all parameter validations.
            `Set shared kw overrides individual params (same indv. params are ignored).`
        
        Returns
        -------
        dict
            A dictionary mapping parameter names to their full validation structures.
        """        
        X = {}
        for define in args:
            X.update( define(info, kw = kwargs_shared) )
        return X

    @staticmethod
    def _apply_ak_fx(fx:Callable, definition:dict):
        ak = fx(definition) 
        return ak if isinstance(ak, dict) else {}

    @staticmethod
    def _parse_ar(args:dict) -> tuple[list, str]:
        ar = args.get('args', False)
        if not isinstance(ar, (tuple, list)):
            ar = [ar]
        i = args.get('info', '')
        i = '\n'*2 + i if i else i
        return ar, i
                
    def auto_check(self,
        definitions: dict, 
        name: str, 
        ar_fx: Callable = lambda definition: {},
        kw_fx: Callable = lambda definition: {}
    ) -> None:
        """
        Automatically validates all parameters in a given definitions dictionary.
        
        Iterates through the definitions, constructs a detailed info message for each
        (including name, values, description, and tips), applies any custom keyword
        arguments via `kw_fx`, and executes the base check.
        
        Parameters
        ----------
        definitions : dict
            A dictionary where keys are parameter names and values are raw definitions.
        
        name : str
            The name of the configuration block or section being validated.
        
        ar_fx : Callable, optional
            A function that accepts a `ParamBase` definition and returns a dictionary
            of additional validation arguments. 
            If dict is returned it must provide args and optionally extra info, e.g.
            `dict(args = check | (check1, check2), info = 'Extra to print')`
            Defaults to returning an empty dict.            
        
        kw_fx : Callable, optional
            A function that accepts a `ParamBase` definition and returns a dictionary
            of additional validation keyword arguments → e.g. `dict(min_max=False, mini=True)`. 
            Defaults to returning an empty dict.
            Available options with defaults:
            - mini:       bool = False → if True, only validates the minimum value
            - min_max:    bool = True  → if True, validates both min and max values
            - min_incl:   bool = True  → If True, the minimum bound is inclusive
            - values:     list = None  → Explicit list of allowed values; Overrides definition options

        Raises
        ------
        Exception
            If any parameter fails the validation checks defined in `_Check.base_check`.
        """           
        for df in definitions.values():
            df = self._param(df)        

            ar = self._apply_ak_fx(ar_fx, df) 
            ar, info = self._parse_ar(ar)
            i = lambda f: (
                f'{df.name}:'
                f'\n► supported types → {df.kind}'
                f'\n► values defaults → {df.get_values()}'
                f'\n► {df.desc or df.desc_h}'
                f'\n► {df.tip.replace('<br>', '\n')}' + info
            )
            kw = self._apply_ak_fx(kw_fx, df) 
            
            vd = self.define(df, info = i, **kw)
            self.base_check(vd, name, *ar)     
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬