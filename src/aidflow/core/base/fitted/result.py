"""
Model-Level Data Result - Secondary Offload for Auxiliary Model Outputs.

This module provides a variant of the FittedResult descriptor that sits
directly on the `ModelBase` class. It is used for **secondary** result
data that a Model may produce beyond the mandatory fitted result.

Key Distinction from `fitted.py`:
    - `fitted.py` (FittedResult): The **primary** result that flows from
      Fitter → Model → Worker. Routing is complex (Fitter vs. ModelRunner).
    - `result.py` (ModelDataResult): A **secondary** result that lives
      directly on the Model instance. Routing is simple (always the Model).

Use Case:
    When a Model produces auxiliary outputs (e.g., a correlation matrix,
    a feature importance ranking, or intermediate statistics) that are not
    part of the main fitted result but still need memory management.

Example:
    >>> # On ModelBase:
    >>> class MyModel(ModelBase):
    ...     _result = ModelDataResult()
    ...
    >>> # Offload the auxiliary result to free RAM:
    >>> ModelDataResult.offload('AIGFLOW.Data.fe_stabilizer', project_sig)
"""

from ...parameters import (
    Parameters
)
from .base import (
    FittedIDBase, 
    FittedResultBase,
    FittedIORegistry
)

        
class FittedID(FittedIDBase):
    """
    Routing implementation for ModelBase-level data results.

    Unlike the primary FittedID (which routes between Fitter and ModelRunner),
    this implementation always resolves to the ModelBase instance itself.
    The data descriptor lives directly on the Model class.
    """
    
    @property
    def setup(self) -> Parameters:
        from ..model import ModelBase
        
        if isinstance(self.instance, ModelBase):
            return self.instance.setup
        
        raise TypeError(self._error())   
    
    @property
    def model(self) -> str:
        from ..model import ModelBase

        if isinstance(self.instance, ModelBase):
            return self.instance.MODEL
        
        raise TypeError(self._error())

    def get_target_instance(self):
        """
        Return the ModelBase instance itself as the data target.

        Since ModelDataResult always sits on the Model, the target is
        always the instance itself. No routing required.

        Returns:
            ModelBase: The instance holding the result data.
        """
        from ..model import ModelBase, ModelUIBase
        
        if isinstance(self.instance, ModelBase):
            return self.instance
        
        if isinstance(self.instance, ModelUIBase):
            return self.instance.model
        
        raise TypeError(self._error())

    def _get_fitted(self, descriptor:bool=False):
        """
        Retrieve the result data from the ModelBase's `_result` descriptor.

        Parameters:
            descriptor (bool): If True, return the descriptor object.
                If False, return the resolved value.

        Returns:
            ModelDataResult | Any: The descriptor or the result data.
        """        

        instance = self.get_target_instance()
        cls_ = type(instance)
        
        desc: "ModelDataResult" = cls_._result

        if descriptor:        
            return desc

        return desc.get_result(instance)

    def _error(self):        
        return ('AIG SAFETY ► '
            f'ModelDataResult does not support {type(self.instance)}: '
            'ModelDataResult supports ModelBase instance only.'
        )


class ModelDataResult(FittedResultBase):
    """
    Descriptor for ModelBase-level auxiliary result data.

    Provides the same offload/re-hydration semantics as `FittedResult`
    but scoped to the Model instance directly. Use this for secondary
    outputs that a Model produces (e.g., statistics, rankings, matrices)
    that should participate in the pipeline's memory management.

    The key difference from `FittedResult` is that routing is always
    to the Model itself — there is no Fitter/ModelRunner ambiguity.
    """    
    _id = FittedID
    _io = FittedIORegistry()