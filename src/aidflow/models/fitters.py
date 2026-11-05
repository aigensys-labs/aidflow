"""
Centralized manifest of all available Fitter, Model and Dashboard implementations.

This module serves as a high-level convenience layer that aggregates every 
available model implementation within the framework. Upon import, it triggers 
the automatic discovery and registration process, populating the `ModelRegistry` 
and `DashRegistry` with all defined components. This ensures that any Fitter or 
Dashboard added to the library is immediately available for selection and use 
within the Model pipeline without requiring manual configuration.
"""

from ..data.load.processor import (
    DataProcessor
)

from ..data.screen.fitters import (
    FeatureStandardizer,
    FeatureTemporalAnalyzer,
    ObservationAnalyzer
)

from ..data.encode.fitters import (
    NumericalDiscretizor,
    CategoricalEncoder,
    InteractionExtractorLGB,
    InteractionExtractorXGB,
    WoeTransformerTest,
    WoeTransformerGini,
    WoeStabilityAnalyzer
)

from ..data.transform.processor import (
    DataEncoderProcessor
)

from ..learner.agents.fitters import (
    SparseBinaryGLM,
    LassoBinaryGLM,
    ElasticBinaryGLM,
    LogRegBinaryGLM,
    PLSBinaryGLM,
    LightBinaryGBM,
    ExtremeBinaryGBM    
)

from ..selector.predict.processor import(
    PredictorProcessor
)

from ..selector.compare.processor import(
    ComparatorProcessor
)

from ..selector.transition.processor import(
    TransitionProcessor
)

from ..selector.select.processor import(
    SelectorProcessor
)

from ..selector.select.agent.fitter import(
    PredictorFitter
)