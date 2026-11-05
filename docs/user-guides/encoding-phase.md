The Encoding Phase transforms stable features into ML-ready signals. This is where raw business variables become the mathematical inputs your models actually learn from.

AIDflow's encoding suite is **target-aware** and **stability-verified** — every transformation is validated against the binary target and screened for temporal drift before it reaches an agent.

---

## The Objective: Stable → Encoded → ML-Ready

The Encoding Phase follows a strict linear progression. WOE-ML steps depend on the previous ones.

### 1. Numerical Discretization (`num_encoder`)
**The Binning Engine.** Continuous features are transformed into discrete bins using the selected fitting strategy. This creates the foundation for WOE encoding and enables interpretable bin-level analysis.

*   Bin boundaries are learned from the training distribution.
*   Each bin is validated for minimum population share and target separation.
*   The resulting encoding is locked — the same boundaries apply at inference time.

### 2. Categorical Encoding (`cat_encoder`)
**Category Normalization.** Object and categorical features are mapped into a consistent encoded representation.

*   Low-frequency categories are handled (merged).
*   The category → code mapping is stored as an immutable artifact.
*   Ensures the model sees a stable representation regardless of original category order.

### 3. Interaction Discovery (`ixn_encoder`)
**Pattern Amplifier.** Generates new interaction variables by combining stable numerical and categorical features. Captures relationships that individual features cannot express alone.

*   Uses a tree-based search to identify high-signal feature pairs.
*   Interactions are screened for target relevance before inclusion.
*   Optional — skip if your features already capture the relevant relationships.

### 4. WOE Encoding (`woe_encoder`)
**The Core Transformation.** Converts all encoded features (numerical bins, categorical codes, interactions) into Weight of Evidence values relative to the binary target.

*   Each bin's WOE = ln(Bad Rate / Good Rate) for that segment.
*   The result is a feature that is approximately linear in log-odds space — ideal for GLM agents.
*   WOE values are locked at training time. New observations map to the same WOE values at inference.

### 5. WOE Stability Filter (`woe_stabilizer`)
**The Second Guardrail.** Screens WOE-encoded features for temporal stability. A feature whose WOE distribution shifts significantly across time folds is flagged as unstable and pruned.

*   Complements the Data Phase stability check (which operates on raw features distribution).
*   This operates on the *encoded* representation — verifies time stability of target variable across feature bins.
*   **Result:** The final feature set is stable both in its distribution AND in its target behavior.

### 6. ML Sample Synchronization (`data_encoder`)
**The Handoff Point.** Applies all previously trained encodings to produce the final `train.ml` and `test.ml` datasets. This is the exact input that agents will train on.

*   Ensures the same encoding pipeline is applied consistently to both train and test.
*   Prevents the "train on one version, test on another" class of errors.
*   The `train.ml` / `test.ml` datasets are registered into the context — agents consume them directly.

---

## Practical Implementation

Execute the full Encoding Phase:

```python
# Step-by-step execution
num_encoder    = ctx.flow.encoding.num_encoder()      # Numerical binning
cat_encoder    = ctx.flow.encoding.cat_encoder()      # Categorical encoding
ixn_encoder    = ctx.flow.encoding.ixn_encoder()      # Interaction discovery
woe_encoder    = ctx.flow.encoding.woe_encoder()      # WOE transformation
woe_stabilizer = ctx.flow.encoding.woe_stabilizer()   # WOE stability filter
data_encoder   = ctx.flow.encoding.data_encoder()     # ML sample synchronization
```

Or run the entire phase at once:

```python
ctx.flow.process('encoding')
```

Once complete, the ML-ready datasets are available:

```python
train = ctx.datasets.train.ml
test  = ctx.datasets.test.ml
```

## Deep Dive: Technical Reference
For detailed information on binning strategies, WOE computation, and stability thresholds:

[View NumEncoder API](/api/encoding-interface/NumEncoderWorker) | [View WoeEncoder API](/api/encoding-interface/WoeEncoderWorker) | [View WoeStabilizer API](/api/encoding-interface/WoeStabilizerWorker)