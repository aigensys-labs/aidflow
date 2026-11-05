"""
Relay Data Routing - Concrete Implementation for Fitter Data.

This module provides the concrete `FittedID` implementation that resolves
the routing logic for the fitter relay data in the AIDflow pipeline.

Routing Logic:
    - If the instance IS a Fitter → it IS the target.
    - Otherwise route via its Model.
"""

from .base import (
    FittedResultBase,
    FittedIORegistry
)
from .fitted import (
    FittedID as FittedIDBase, 
    _FittedRoute
)


class FittedID(FittedIDBase):
    """
    Concrete routing implementation for the FitterDataRelay descriptor.

    Resolves the Fitter → Model hierarchy to
    identify:
    - The correct `Parameters` (setup).
    - The correct model identifier string.
    - The target instance holding the result data.
    - The result value itself.

    This is the FittedID used by the optional relay that flows around Fitter.
    """
    
    def get_target_instance(self):
        """
        Resolve the instance that holds the relay data.

        Routing logic:
        1. If the instance IS a Fitter → return it directly.
        2. Otherwise route via ModelBase.

        Returns:
            FitterBase : The target data-holding instance.

        Raises:
            TypeError: If the routing cannot be resolved.
        """
        Fitter, Model, ModelUI, Runner = _FittedRoute.get_allowed()

        route = _FittedRoute(self.instance)

        if isinstance(self.instance, Fitter):
            return self.instance

        model = route.get_model_instance()

        if isinstance(model, Model):
            return model.fx.fitter
        
        raise TypeError(self._error())

    def _get_fitted(self, descriptor:bool=False):
        """
        Retrieve the relay data (or its descriptor) from the target instance.

        Resolves the target instance via `get_target_instance()`, then
        accesses the appropriate descriptor attribute to get the value.

        Parameters:
            descriptor (bool): If True, return the descriptor object.
                If False, return the resolved value.

        Returns:
            FitterDataRelay | Any: The descriptor or the result data.
        """        
        from ..model import FitterBase

        instance: FitterBase = self.get_target_instance()
        cls_ = type(instance)
        
        desc: FitterDataRelay = cls_._relay

        if descriptor:        
            return desc

        return desc.get_result(instance)

    def _error(self):        
        return ('AIG SAFETY ► '
            f'FitterDataRelay does not support {type(self.instance)}: '
            'FitterDataRelay supports FitterBase instance only.'
        )

class FitterDataRelay(FittedResultBase):
    """
    The FitterDataRelay descriptor for pipeline Fitter.

    This is the descriptor that gets attached to Fitter classes to provide 
    transparent offload/re-hydration of the relay data.

    When accessed (`fitter._relay`), it transparently:
    1. Routes to the correct target instance (FitterBase).
    2. Checks if data was offloaded and needs re-hydration from disk.
    3. Returns the result Dataclass with all fields populated.

    When set (e.g., during `fit()`), it caches the result and
    registers the IO handler for future offload operations.
    """    
    _id = FittedID
    _io = FittedIORegistry()