from ...core.parameters             import  AIGFLOW, Parameters
from ...core.registry               import  ProjectModelRegistry
from ...core.standard               import  VarStd
from ...data.encode.ixn_encoder     import  IxnEncoder

DEPS  = AIGFLOW.Encoding.ixn_encoder


class RawFeatures:
    """
    Helper to extract min set of mandatory raw features based on selected fes by agent.
    """
    def __init__(self, setup: Parameters):
        self.encoder: IxnEncoder = (
            ProjectModelRegistry.get_model(setup.project_signature, DEPS)
        )

    def _get_raw(self, fes:list) -> list:
        ixn  = self.encoder.cols.encoding
        fes_ = []

        for f in fes:
            if f in ixn:                
                fes_.extend(ixn[f].vars)
            else:
                fes_.append(f)
                       
        return list(dict.fromkeys(fes_))
            
    def get(self, selected:list) -> list:
        """
        Returns min set of mandatory raw features based on selected fes by agent.
        """
        fes = VarStd.fe_to_var(selected)

        if (
            self.encoder is None or 
            not self.encoder.cols.encoding
        ):
            return fes
        
        return self._get_raw(fes)
    
    @staticmethod
    def get_orig_fes(selected: list) -> list:
        """Returns feature names without any AIG feature suffix."""
        return [VarStd.fe_to_var(x) for x in selected]