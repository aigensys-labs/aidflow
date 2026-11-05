"""
Centralized manifest of all available Fitter, Model and Dashboard implementations.

This module serves as a high-level convenience layer that aggregates every 
available model implementation within the framework. Upon import, it triggers 
the automatic discovery and registration process, populating the `ModelRegistry` 
and `DashRegistry` with all defined components. This ensures that any Fitter or 
Dashboard added to the library is immediately available for selection and use 
within the Model pipeline without requiring manual configuration.
"""

from ..data.load import (
    DataLoaderInput,
    DataLoaderResult,
    DataLoader
)

from ..data.screen import (
    FeCleanerInput,
    FeCleanerResult,
    FeatureCleaner,
    FeStabilizerInput,
    FeStabilizerResult,
    FeatureStabilizer,
    ObsStabilizerInput,
    ObsStabilizerResult,
    ObservationStabilizer
)

from ..data.encode import (
    NumEncoderInput,
    NumEncoderResult,
    NumEncoder,
    CatEncoderInput,
    CatEncoderResult,
    CatEncoder,
    IxnEncoderInput,
    IxnEncoderResult,
    IxnEncoder,
    WoeEncoderInput,
    WoeEncoderResult,
    WoeEncoder,
    WoeStabilizerInput,
    WoeStabilizerResult,
    WoeStabilizer,
)

from ..data.transform import (
    DataEncoderInput,
    DataEncoderResult,
    DataEncoder
)

from ..learner.agents import (
    Agent1Input,
    Agent1Result,
    Agent1Predictor,
    Agent2Input,
    Agent2Result,
    Agent2Predictor,
    Agent3Input,
    Agent3Result,
    Agent3Predictor,
    Agent4Input,
    Agent4Result,
    Agent4Predictor
)

from ..selector.predict import (
    AgentsPredictorInput,
    AgentsPredictorResult,
    AgentsPredictor
)

from ..selector.compare import (
    AgentsComparatorInput,
    AgentsComparatorResult,
    AgentsComparator
)

from ..selector.transition import (
    AgentsTransitionInput,
    AgentsTransitionResult,
    AgentsTransition
)

from ..selector.select import (
    AgentSelectorInput,
    AgentSelectorResult,
    AgentSelector
)

from ..selector.select.agent import (
    AgentPredictorInput,
    AgentPredictorResult,
    AgentPredictor
)