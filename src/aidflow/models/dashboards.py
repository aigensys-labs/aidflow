"""
Centralized manifest of all available Fitter, Model and Dashboard implementations.

This module serves as a high-level convenience layer that aggregates every 
available model implementation within the framework. Upon import, it triggers 
the automatic discovery and registration process, populating the `ModelRegistry` 
and `DashRegistry` with all defined components. This ensures that any Fitter or 
Dashboard added to the library is immediately available for selection and use 
within the Model pipeline without requiring manual configuration.
"""

from ..data.load.report import (
    DataLoaderDash
)

from ..data.screen.reports import (
    FeatureCleanerDash,
    FeatureStabilizerDash,
    ObservationStabilizerDash
) 

from ..data.encode.reports import (
    NumEncoderDash,
    CatEncoderDash,
    IxnEncoderDash,
    WoeEncoderDash,
    WoeStabilizerDash,
) 

from ..learner.agents.reports import (
    Agent1Dash,
    Agent2Dash,
    Agent3Dash,
    Agent4Dash,
)

from ..selector.compare.report import (
    AgentsComparatorDash
)

from ..selector.transition.report import (
    AgentsTransitionDash
)

from ..selector.select.report import (
    AgentSelectorDash
)

from ..selector.select.agent.report import (
    AgentPredictorDash
)