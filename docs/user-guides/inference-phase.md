The Inference Phase is where the model meets production. Two components complete the pipeline:

1. **Predictor** — offline scoring engine. Applies the full trained pipeline (cleaning → encoding → prediction) to new data using the exact parameters locked during training.
2. **ProductionExporter** — deployment artifact. Serializes the champion into a self-contained scoring module with an orchestration layer.

Together, they guarantee that the model that passed the Selection Phase is the exact model that scores in production. Zero drift. Zero re-derivation. Zero ambiguity.

---

## The Objective: Trained Model → Scored Data → Deployed Artifact

The Inference Phase has no training. Everything is locked. The only question is: **what data are you scoring, and where does it deploy?**

---

## Component 1: Predictor (Inference Engine)

The `Predictor` loads all trained models from a stored project and provides a symmetrical API to the training pipeline. Every stage is accessible — from raw data cleaning to champion prediction.

**Requirement:** The project must have been trained with `save=1`.

### Usage Modes

- **`preprocess=True`** (default for agent predictions): Pass raw input columns (same schema as training). The full pipeline — cleaning → stability → encoding → prediction — is applied automatically.
- **`preprocess=False`**: Pass pre-encoded data (e.g., WOE features). Skips preprocessing. Useful when chaining stages or when encoding is applied externally.

### Accessible Stages

| Stage | Method | Returns |
|-------|--------|---------|
| Data Quality | `pred.fe_clean(data)` | Cleaned data |
| Temporal Stability | `pred.fe_stabilize(data)` | Stable features |
| Outlier Detection | `pred.obs_stabilize(data)` | Flagged observations |
| Numerical Binning | `pred.num_encode(data)` | Binned features |
| Categorical Encoding | `pred.cat_encode(data)` | Encoded categories |
| Interactions | `pred.ixn_encode(data)` | Interaction features |
| WOE Encoding | `pred.woe_encode(data)` | WOE values |
| WOE Stability | `pred.woe_stabilize(data)` | Stable WOE features |
| ML Sample Sync | `pred.data_encode(data)` | ML-ready data |
| Agent 1–4 | `pred.agent1_predict(data)` … `pred.agent4_predict(data)` | P(class=1) |
| **Champion** | **`pred.champion_predict(data)`** | **P(class=1)** |
| Schema | `pred.get_schema()` | Full feature lineage |

### Chainable Access

Individual stages can be called independently to inspect intermediate results:

```python
pred = aid.Predictor(project='my_scorecard')

# Inspect what cleaning does to new data
cleaned = pred.fe_clean(new_data)

# Apply only WOE encoding (for custom downstream use)
woe = pred.woe_encode(cleaned.data)

# Full pipeline — raw in, probability out
score = pred.champion_predict(new_data, preprocess=True)
```

---

## Component 2: ProductionExporter (Deployment Artifact)

The ProductionExporter serializes the champion agent into production-ready artifacts. The output is a closed, versioned unit — deployable to any inference infrastructure.


### What's Exported

| ARTIFACT | FORMAT | PURPOSE |
|----------|--------|---------|
| Champion model | `.pkl` (pickle) | Self-contained scoring engine. Contains: model + encoding maps + feature lineage. No AIDflow import required at load time. |
| Airflow DAG | `.py` (Python) | Orchestration layer. Schedules: load raw data → preprocess → score → push to warehouse. |


### The Exported Agent Contract
The exported pickle exposes exactly four methods:

| METHOD | PURPOSE | 
|--------|---------|
| `preprocess(data)` | Raw → ML-ready (cleaning + encoding) |
| `transform(data)` | ML-ready → P(class=1) |
| `score(data)` | Raw → ML-ready → P(class=1) (all at once) |
| `is_fitted` | Boolean: confirms scorable state |


This is the complete runtime interface. No other attributes are accessed at inference time.

### Export Configuration
The DAG is fully configurable via `ExportConfig`:

| PARAMETER | DEFAULT | PURPOSE |
|-----------|---------|---------|
| `dag_id` | aidflow_scoring | Airflow DAG identifier |
| `schedule` | 0 2 * * * | Nightly at 02:00 (cron) |
| `model_path` | /models/champion.pkl | Where the pickle lives in production |
| `raw_data_loader` | scoring.load_raw_data | Dotted path to data loader |
| `result_sink` | scoring.push_scores | Dotted path to warehouse writer |
| `retries` | 3 | Retries per task on failure |
| `retry_delay_min` | 5 | Delay between retries (exponential backoff) |


## Practical Implementation

```python
import aidflow as aid

# Inference
pred  = aid.Predictor(project='my_scorecard')
score = pred.champion_predict(new_data, preprocess=True)

# Individual agents for monitoring/comparison
a1 = pred.agent1_predict(new_data, preprocess=True)
a4 = pred.agent4_predict(new_data, preprocess=True)

# Deployment - active ctx state after training
champion = ctx.workers.champion_predictor  # the fitted champion

exporter = ProductionExporter(champion)
paths = exporter.export_all('./production/')

# paths['pickle']      → production/champion.pkl
# paths['airflow_dag'] → production/dag.py
```

### The Output

```
./production/
├── champion.pkl    # Self-contained scoring engine (~size depends on model)
└── dag.py          # Airflow DAG (load → score → push)
```

champion.pkl — load and score with no AIDflow dependency:

```python
import pickle
with open('./production/champion.pkl', 'rb') as f:
    agent = pickle.load(f)

scores = agent.score(new_raw_data)
```

dag.py — drop into your Airflow dags/ folder. Configure raw_data_loader and result_sink to point to your data source and warehouse.


## Deep Dive: Technical Reference
[View Predictor API](/api/inference-interface/Predictor) |
[View Exporter API](/api/inference-interface/ProductionExporter)