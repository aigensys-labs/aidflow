"""
Fitted Result Routing - Concrete Implementation for Fitter/ModelRunner Data.

This module provides the concrete `FittedID` implementation that resolves
the routing logic for the primary fitted result data in the AIDflow pipeline.

The critical design decision it encodes:
    The final result data can reside on either the **Fitter** or the
    **_ModelRunner**, depending on whether the Model implements a
    `_get_result()` method. This module determines which layer holds
    the data and ensures the descriptor reads/writes to the correct target.

Routing Logic:
    - If the instance IS a Fitter or ModelRunner → it IS the target.
    - If the Model defines `_get_result` → route to `model.fx.fitter`.
    - If the Model does NOT define `_get_result` → route to the ModelRunner.

This file is the "primary" fitted result implementation, used by the
mandatory result that every pipeline step must produce.

For secondary model-level results (e.g., auxiliary outputs), see `result.py`.
"""

from ...parameters import  (
    Parameters
)
from .base import  (
    FittedIDBase, 
    FittedResultBase,
    FittedIORegistry
)


class _FittedRoute:
    """
    Internal routing utility for FittedID resolution.

    Encapsulates the type-checking and navigation logic needed to
    traverse the Fitter → Model → ModelUI → ModelRunner hierarchy
    and identify the correct data-holding instance.

    Parameters:
        instance (Any): The object from which to resolve the route.    
    """

    def __init__(self, instance):
        self.instance = instance

    @staticmethod
    def get_allowed():
        """
        Import and return the set of instance types supported by FittedResult routing.

        Returns:
            tuple: (FitterBase, ModelBase, ModelUIBase, _ModelRunner)
        """        
        from ..model import FitterBase, ModelBase, ModelUIBase
        from ..model_ui import _ModelRunner
        return FitterBase, ModelBase, ModelUIBase, _ModelRunner

    @staticmethod
    def get_fit():
        """
        Retrieve the FIT constants (e.g., RESULT method name).

        Returns:
            FIT: The FIT namespace containing routing constants.
        """        
        from ..model import FIT
        return FIT

    def get_model_instance(self):
        """
        Resolve the Model instance from the current object.

        If the current instance is a ModelUI, returns its underlying Model.
        Otherwise, returns the instance itself (assuming it IS a Model).

        Returns:
            ModelBase | Any: The resolved Model instance.
        """        
        FitterBase, ModelBase, ModelUIBase, _ModelRunner = self.get_allowed()

        if isinstance(self.instance, ModelUIBase):
            return self.instance.model

        return self.instance

    def is_target_instance(self) -> bool:
        """
        Check if the current instance is already the target data holder.

        Returns:
            bool: True if the instance is a FitterBase or _ModelRunner.
        """        
        FitterBase, ModelBase, ModelUIBase, _ModelRunner = self.get_allowed()
        return isinstance(self.instance, (FitterBase, _ModelRunner))

    def has_get_result(self):
        """
        Determine whether the Model defines a `_get_result` method.

        This is the routing factor: if the Model implements `_get_result`,
        the Fitter is the final data placeholder. Otherwise, the ModelRunner
        is responsible for assembling the result.

        Returns:
            bool: True if routing should go to the Fitter.
        """
        if self.is_target_instance():
            return False
        
        model = self.get_model_instance()            
        return model._has_get_result()
    

class FittedID(FittedIDBase):
    """
    Concrete routing implementation for the primary FittedResult descriptor.

    Resolves the Fitter → Model → ModelUI → ModelRunner hierarchy to
    identify:
    - The correct `Parameters` (setup).
    - The correct model identifier string.
    - The target instance holding the result data.
    - The result value itself.

    This is the "primary" FittedID used by the mandatory pipeline result
    that flows from Fitter through Model to the Worker interface.
    """
    
    @property
    def setup(self) -> Parameters:
        Fitter, Model, ModelUI, Runner = _FittedRoute.get_allowed()

        if isinstance(self.instance, (Model, Fitter)):
            return self.instance.setup
        
        if isinstance(self.instance, (ModelUI, Runner)):
            return self.instance.model.setup
                
        raise TypeError(self._error())   
    
    @property
    def model(self) -> str:
        Fitter, Model, ModelUI, Runner = _FittedRoute.get_allowed()

        if isinstance(self.instance, (Model, Fitter)):
            return self.instance.MODEL
        
        if isinstance(self.instance, (ModelUI, Runner)):
            return self.instance.model.MODEL
        
        raise TypeError(self._error())

    def get_target_instance(self):
        """
        Resolve the instance that holds the fitted result data.

        Routing logic:
        1. If the instance IS a Fitter or ModelRunner → return it directly.
        2. If the Model defines `_get_result` → the Fitter holds the data.
           Return `model.fx.fitter`.
        3. Otherwise → the ModelRunner holds the data.
           Return `instance.model_ui._runner`.

        Returns:
            FitterBase | _ModelRunner: The target data-holding instance.

        Raises:
            TypeError: If the routing cannot be resolved.
        """
        Fitter, Model, ModelUI, Runner = _FittedRoute.get_allowed()

        route = _FittedRoute(self.instance)

        if route.is_target_instance():
            return self.instance
        
        # (A) _get_result defined → Fitter is the final data placeholder
        if route.has_get_result():
            model = route.get_model_instance()
            return model.fx.fitter

        # (B) _get_result not defined → Runner is the final data placeholder
        if isinstance(self.instance, ModelUI):
            return self.instance.model_ui._runner

        raise TypeError(self._error())

    def _get_fitted(self, descriptor:bool=False):
        """
        Retrieve the result data (or its descriptor) from the target instance.

        Resolves the target instance via `get_target_instance()`, then
        accesses the appropriate descriptor attribute (`.result` on Fitter
        or `._fitted` on ModelRunner) to get the value.

        Parameters:
            descriptor (bool): If True, return the descriptor object.
                If False, return the resolved value.

        Returns:
            FittedResult | Any: The descriptor or the result data.
        """        
        from ..model import FitterBase
        from ..model_ui import _ModelRunner

        instance = self.get_target_instance()
        cls_ = type(instance)
        
        if isinstance(instance, FitterBase):
            desc: FittedResult = cls_.result
        else: 
            desc: FittedResult = cls_.fitted

        if descriptor:        
            return desc

        return desc.get_result(instance)

    def _error(self):
        FIT = _FittedRoute.get_fit()
        return ('AIG SAFETY ► '
            f'FittedResult does not support {type(self.instance)}:'
            f'\n► When result is routed to the Fitter, verify that {FIT.GRESULT} is implemented.'
            '\n► FittedResult supports '
            'FitterBase | ModelBase | ModelUIBase | _ModelRunner '
            'instance only.\n'
        )


class FittedResult(FittedResultBase):
    """
    The primary FittedResult descriptor for pipeline step results.

    This is the descriptor that gets attached to Fitter and ModelRunner
    classes to provide transparent offload/re-hydration of the mandatory
    result data that flows through every AIDflow pipeline step.

    When accessed (e.g., `worker.result`), it transparently:
    1. Routes to the correct target instance (Fitter or ModelRunner).
    2. Checks if data was offloaded and needs re-hydration from disk.
    3. Returns the result Dataclass with all fields populated.

    When set (e.g., during `fit()`), it caches the result and
    registers the IO handler for future offload operations.
    """    
    _id = FittedID
    _io = FittedIORegistry()