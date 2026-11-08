AIDflow is built on five architectural principles:
- **Context-Driven Orchestration**
- **Three-Layer Abstraction**
- **Strict Data Lineage**
- **Memory-Aware Relay**
- **Distributed Compute**

---

## 1. Context-Driven Orchestration (`ProjectContext`)

Most ML projects pass data between functions as disconnected variables (`df_clean`, `df_stable`, `X_train`). Hidden state. No traceability. Reproducibility is manual luck.


**AIDflow replaces this with a single `ProjectContext`.**

The context is the source of truth. When a worker executes, it registers its output into the context — not into a variable you have to remember.

- **Traceability.** Every dataset has a semantic name and a parent. You can trace any feature back to its raw origin.
- **Reproducibility.** Load the context, call the worker, get the same result. Every time.
- **Sequence Safety.** A step cannot execute until its prerequisite has registered. The pipeline enforces itself.

---

## 2. The Three-Layer Abstraction (The "Super-Gear")

AIDflow decouples the user's intent from the mathematical execution using a three-layer stack. This allows us to change the underlying ML algorithm without ever breaking the User Interface.

**Practical effect:** switch from Lasso to ElasticNet by passing one parameter. The worker API, the state schema, and the reporting pipeline remain unchanged.

### Layer 1: The Worker (Interface)
*   **Role:** The Orchestrator.
*   **Responsibility:** Handles UI, input validation, and state registration.
*   **User Experience:** *"I want to clean my data"* → `ctx.workers.fe_cleaner()`.

### Layer 2: The Model (The Gear)
*   **Role:** The Adapter.
*   **Responsibility:** Decouples the Worker from the Fitter. It manages "Fit Options," allowing a single Worker to support multiple different algorithms (e.g., switching between a LogReg and an ElasticNet).

### Layer 3: The Fitter (The Engine)
*   **Role:** The Compute Engine.
*   **Responsibility:** Pure mathematical execution. This is where the actual matrix multiplication, optimization, and fitting occur. It has no knowledge of the UI or the Project Context—it only knows data and parameters.

---

## 3. Strict Data Lineage

AIDflow enforces a rigid transformation chain:

`Raw` → `Clean` → `Stable` → `Encoded` → `ML-Ready` → `Scored`

Each transition is logged. Each state is immutable once registered. This guarantees:

- **No leakage.** Future information from the test set cannot leak into training-phase transformations.
- **No drift.** The exact encoders fitted on training data are locked and applied identically at inference time.
- **Full audit trail.** Every transition is recorded — an immutable chain for regulatory review.

---

## 4. Memory-Aware Relay (Offload)

High-dimensional scorecard projects — hundreds of features, millions of rows — push working memory beyond what a single machine can hold. AIDflow handles this natively.

**Relay Offload** serializes intermediate data structures to disk (via the Arrow/Parquet engine) when a step's output has no downstream in-memory consumer. When the next step needs it, it streams back.

```python
ctx.params(offload=1)
```

---

## 5. Distributed Compute (Ray)

By default, AIDflow uses local multiprocessing — zero configuration, immediate start. When your workload outgrows a single machine, the same pipeline scales to a cluster without a code rewrite.

Enable Ray as the compute backend:

```bash
pip install aidflow[ray]
```


## Traditional ML vs AIDflow

| Dimension | Traditional | AIDflow |
|-----------|-------------|---------|
| **Data flow** | Variable passing (`df1` → `dfN`) | Context registration (semantic lineage) |
| **Architecture** | Monolithic script | Worker → Model → Fitter (swappable layers) |
| **Extension** | Rewrite the function | Define a new Fitter, register it. Done. |
| **Stability** | Manual checks, date slicing | Built-in temporal phases (pre-training) |
| **Memory** | RAM-bound. Sample or add hardware. | Arrow offload. Runs on any machine. |
| **Compute** | `multiprocessing` or manual `joblib` | Local multiprocessing (default) → Ray cluster |
| **Reproducibility** | "Works on my machine" | Context-locked. Run-to-run identical. |
| **AI** | Manual step-by-step orchestration | Single-call processes pipeline → production DAG. Stateless, idempotent, AI-friendly. |

[Explore the API Reference](/api/execution-interface/Parameters)