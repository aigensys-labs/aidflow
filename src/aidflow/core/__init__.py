"""
Core Infrastructure — AIDflow Internal Framework.

This package provides the foundational building blocks that
underpin every stage of the AIDflow pipeline. It is NOT
user-facing — it is the internal scaffolding that connects
models, fitters, processors, and the display layer into a
coherent execution framework.

Submodules Highlights
---------------------
base
    Abstract base classes and execution contracts:
    - `ModelBase`: Low-level connection layer.
    - `FitterBase`: Low-level compute layer.
    - `TransformerBase`: Post-processing between Fitter and Model.
    - `ModelProcessorBase`: Parameter resolution and fit orchestration.
    - `ModelInput` / `ModelResult`: I/O data contracts.
    - `OptionUIBase` / `ModelUIBase`: Display layer contracts.
    - `WorkerBase`: Public API base covering Model-Fitter task execution.
    - `ModelAdapterBase`: Stateful model adapter (for predictor
      and champion layers).

factory
    Declarative data structures:
    - `Dataclass`: Typed configuration/data containers with
      schema, mapping, and validation capabilities.
    - `DictClass`: Static label/constant namespaces.

foundation
    Foundational utilities:
    - `AigParam`: Parameter resolution (positional/keyword
      mapping, runtime setattr injection).

parameters
    Flow-level parameter definitions:
    - `AIGFLOW`: Pipeline stage registry (Schema → Data →
      Encoding → Training → Selection).
    - `PRM`: Parameter namespace (stage → param → default).
    - `Parameters`: Runtime parameter container.

registry
    Model-to-Fitter registration system:
    - `FitRegister`: Register fitter implementations per model.
    - `FitRegOpt`: Registration option base class.
    - `DashRegister`: Display-layer registration.
    - `RegGet`: Resolve registered components at runtime.

io
    Persistence and path management:
    - `ModelIO`: Save/load models to project storage.
    - `ProjectPaths`: Resolve project folder structure.
    - `TempRegistry` / `Temp`: Temporary artifact management.

standard
    Standardized conventions:
    - `VarStd`: Variable naming/typing standards.

accessor
    Data format accessors:
    - `Format`: Standardized output formatting.

interop
    External library integration:
    - `Pd`: Pandas interoperability helpers.
"""