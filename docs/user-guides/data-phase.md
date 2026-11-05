The Data Phase is where model quality is won or lost. AIDflow's approach: **stability before sophistication.** No model — no matter how powerful — survives contact with unstable, noisy, or leaky data.

This phase enforces a rigorous quality gate. Every feature that survives here is one that has proven its temporal stationarity and distributional validity. Only then does it earn a place in the encoding suite.

---

## The Objective: Raw → Clean → Stable

The Data Phase follows a strict linear progression. Each step filters the feature set further. Skipping any step introduces risk that compounds downstream.

### 1. Ingestion (`data_loader`)
**The Gateway.** Brings raw data into the `ProjectContext`. Establishes the immutable starting point — every subsequent transformation is traceable back to this state.

*   Splits data into `train` and `test` (time-based or random, per config).
*   Registers `dtypes.raw` — the semantic type of every column.
*   Creates the `train.raw` / `test.raw` lineage that all later steps consume.

### 2. Quality Screening (`fe_cleaner`)
**Noise Reduction.** Enforces enterprise data standards. Features that violate basic validity rules are identified and removed before they can distort downstream statistics.

*   Missing value ratio exceeds threshold → drop.
*   Single-value dominance (>90% one category) → drop.
*   Collinearity above threshold (|ρ| > cutoff) → keep the more populated.
*   ID/time-like features (near-unique values) → flag or drop.
*   Result: a valid, typed feature set split into `features.num.clean` and `features.cat.clean`.

### 3. Temporal Stability (`fe_stabilizer`)
**The Anti-Leakage Guardrail.** A signature AIDflow step. Analyzes whether each feature's distribution is stationary across time folds. Features that fluctuate wildly over time are pruned — even if they look predictive in aggregate.

*   Splits the training period into sequential folds.
*   Compares per-fold distributions (KS, PSI, or stationarity test).
*   Features exceeding the instability threshold are removed from the active set.
*   **Why it matters:** A feature that "works" in 2023 but shifts in 2024 will overfit to the past. This step catches it before the model ever sees it.

### 4. Observation Stability (`obs_stabilizer`)
**The Outlier Filter.** Stable features can still contain distorted observations. This step identifies and flags individual rows that are statistically anomalous — the data points that would pull a decision boundary in the wrong direction.

*   Uses isolation-based anomaly detection across the stable feature space.
*   Flags (or removes) observations beyond the training distribution envelope.
*   Produces `train.stable` — the final, clean training set that the Encoding Phase will transform.

---

## Practical Implementation

To execute the full Data Phase, follow this sequence in your notebook:

```python
# Step-by-step execution
data_loader    = ctx.flow.data.data_loader()     # Raw data ingestion
fe_cleaner     = ctx.flow.data.fe_cleaner()      # Quality screening
fe_stabilizer  = ctx.flow.data.fe_stabilizer()   # Temporal stability check
obs_stabilizer = ctx.flow.data.obs_stabilizer()  # Outlier removal
```

Or run the entire phase at once:

```python
ctx.flow.process('data')
```

## The Datasets

Once complete, the datasets context holds:

```python
# The clean training set (ready for encoding)
train = ctx.datasets.train.stable

# The surviving feature sets
num = ctx.datasets.features.num.stable.dist
cat = ctx.datasets.features.cat.stable.dist

# Features that were dropped (with reason)
dropped_num = ctx.datasets.features.num.dropped.stable.dist
dropped_cat = ctx.datasets.features.cat.dropped.stable.dist
```
This is the exact input the Encoding Phase will consume.


## Deep Dive: Technical Reference
For detailed information on the underlying algorithms, parameters, and internal logic, refer to the API specifications:

[View DataLoader API](/api/data-interface/DataLoaderWorker) | [View FeatureCleaner API](/api/data-interface/FeatureCleanerWorker) | [View FeatureStabilizer API](/api/data-interface/FeatureStabilizerWorker)