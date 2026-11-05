"""
Data context verification during AIG flow.
"""

from ...deps.data           import  pDF
from ...display.rgb         import  RgbANSI

from ..base                 import  ModelInput
from ..factory              import  Dataclass
from ..foundation           import  Dynamic, StopExecution


class DataValidation:
    """Data context verification during AIG flow."""

    @classmethod
    def gui_dynamic_args(cls, args:tuple, kwargs:dict, pos_param:dict|None):
        if pos_param is None: 
            return
        P = Dynamic.get_parent_args_kwargs(args, kwargs, pos_param)
        if P: 
            cls.data_continuation(*P)    

    @classmethod
    def data_continuation(cls, data: pDF|ModelInput, *features) -> None:
        """Verify data/vars continuation over AIG flow steps."""
                
        if isinstance(data, ModelInput):
            mandatory = Dataclass.get_mandatory_fields(data)
            for p, d in data.items():
                if ( p not in mandatory ) or isinstance(d, (bool,int,float,str)):
                    continue
                if d is None or not len(d):
                    cls._error(f'DATA ► WorkerInput "{p}"')

        if data is None or not len(data):
            cls._error('DATA')
        
        if features:
            fes = sum([len(v) for v in features]) 
            if not fes:        
                cls._error('FEATURES/DATA')

    @classmethod
    def target_continuation(cls, data:pDF, target:str, desc:str=None) -> None:
        """Verify data/target continuation over AIG flow steps."""

        data_, target_ = Dynamic.params((data, target))[0]
        
        if ( data_ is None ) or ( len(data_[target_].value_counts()) == 1 ):    
            cls._error((desc or 'data'), True)

    @staticmethod
    def _error(x:str, is_target:bool=False):
        x = x.upper()
        if not is_target:
            print(
            f'\n{RgbANSI.bold_red('AIG SAFETY')}'
            f'\n► FLOW CONTINUATION NOT POSSIBLE GIVEN NO INCOMING {x}'
            f'\n► PLEASE ADJUST PRECEDING STEPS SETTINGS AS {RgbANSI.bold(x)} ENTERING CURRENT STEP ARE EMPTY'
            )
        else:
            print(
            f'\n{RgbANSI.bold_red('AIG SAFETY')}'
            f'\n► FLOW CONTINUATION NOT POSSIBLE GIVEN NO TARGET INFORMATION ON {RgbANSI.bold(x)} SAMPLE'
            f'\n► TARGET VARIABLE HAS NO VARIABILITY ON {RgbANSI.bold(x)} SAMPLE → TARGET IS A CONSTAT VALUE'
            f'\n► PLEASE ADJUST PRECEDING STEPS SETTINGS → NO FURTHER ML TASK IS POSSIBLE'
            )
        raise StopExecution