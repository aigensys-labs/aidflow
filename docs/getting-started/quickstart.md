Welcome to AIDflow. This guide will take you through the "Golden Path" — the standardized, 
high-performance workflow for moving from raw data ingestion to a production-ready champion model.

---

# Raw Data → Champion Model

Everything is managed via a single `ProjectContext`. No juggling variables. 100% reproducible.


## Run on Demo Data (60 seconds)

```python
import aidflow as aid

# generate demo data
data_path = aid.assets.DataGetter.create_data('small')

# define context
ctx = aid.ProjectContext()

ctx.genesis(
    project      = 'demo',
    input_data   = data_path, 
    target       = 'target',
    yy_mm        = 'time',
    yy_mm_format = '%Y-%m-%d',
)
ctx.params(save=1, logging=1, verbose=1)

# run
ctx.flow.process('all')

# predict
data  = ctx.data_io.load(data_path)
pred  = aid.Predictor(project='demo')
score = pred.champion_predict(data, preprocess=True)
```

> Runs entirely on built-in synthetic data. No external file required.

---

## Use Your Own Data

```python
import aidflow as aid

ctx = aid.ProjectContext()

ctx.genesis(
    project        = 'my_scorecard',
    input_data     = 'path/to/your.data', # csv/gzip/parquet | spark | db
    target         = 'is_churn',
    yy_mm          = 'decision_date',
    yy_mm_format   = '%Y-%m-%d',
)
ctx.params(save=1, logging=1, verbose=1)
```

> Adapt the *target, yy_mm, and yy_mm_format* to your dataset, finetune flow params. Everything else is the same.


## Run the Full Pipeline
Execute all five phases in one call. Intended for automated re-training in CLI env.

```python
# Full pipeline training
ctx.flow.process('all')

# Predict on new data
pred  = aid.Predictor(project='my_scorecard')
score = pred.champion_predict(new_data, preprocess=True)

# Export production artifact → pickle & Airflow DAG
prod = aid.ProductionExporter(ctx.selection.champion_predictor)
prod.export_all('./production')
```

---

## The 5-Phase Workflow
Execute phase-by-phase for inspection, fine-tuning, or stakeholder presentation in UI env.

> Phases are sequence-dependent. Each step registers its output before the next can execute.


### Phase 1: Data Preparation
The goal of this phase is to move from **Raw → Clean → Stable**.

```python
# Data Ingestion - Load your data into the global context
data_loader = ctx.flow.data.data_loader()

# Quality Screening - Remove noise and identify valid feature types
fe_cleaner = ctx.flow.data.fe_cleaner()

# Temporal Stability - Remove non-stationary features
fe_stabilizer = ctx.flow.data.fe_stabilizer()

# Observation Stability - Remove outliers
obs_stabilizer = ctx.flow.data.obs_stabilizer()
```

### Phase 2: Encoding
Now we translate your stable features into ML-ready signals using the AIDflow encoding suite.

```python
# Numerical Discretization
num_encoder = ctx.flow.encoding.num_encoder()

# Categorical Encoding
cat_encoder = ctx.flow.encoding.cat_encoder()

# Interaction Gen & Encoding
ixn_encoder = ctx.flow.encoding.ixn_encoder()

# The Final Guardrail: Weight of Evidence (WOE)
woe_encoder = ctx.flow.encoding.woe_encoder()

# Feature-Target Predictive Temporal Stability
woe_stabilizer = ctx.flow.encoding.woe_stabilizer()

# Create ML-ready train-test samples
data_encoder = ctx.flow.encoding.data_encoder()
```

### Phase 3: Learning Ensemble
Instead of training one model, AIDflow trains an ensemble of four specialized agents to capture different data patterns.

```python
# Sparse Specialist
agent1_predictor = ctx.flow.training.agent1_predictor() 

# Balanced/ElasticNet
agent2_predictor = ctx.flow.training.agent2_predictor()

# Dimensionality/PLS
agent3_predictor = ctx.flow.training.agent3_predictor()

# Non-Linear Powerhouse (GBM)
agent4_predictor = ctx.flow.training.agent4_predictor()
```

### Phase 4: Champion Selection
Evaluate the ensemble on Out-of-Time (OOT) samples and promote the winner.

```python
# Generate predictions for all candidates
agents_predictor = ctx.flow.selection.agents_predictor()

# Compare performance via Lift & Calibration matrices
agents_comparator = ctx.flow.selection.agents_comparator()

# Analyze model disagreement/divergence
agents_transition = ctx.flow.selection.agents_transition()

# Promote the objective champion and generate SHAP audit report
agent_selector = ctx.flow.selection.agent_selector()
```

### Phase 5: Inference
The final stage: test behavior on scoring samples. Once verified, ship it to prod.

```python
# Predict on new data
pred  = aid.Predictor(project='my_scorecard')
score = pred.champion_predict(new_data, preprocess=True)

# Export production artifact → pickle & Airflow DAG
prod = aid.ProductionExporter(ctx.workers.champion_predictor)
prod.export_all('./production')
```

---

## Power-User Tips: 

### Navigate Datasets

AIDflow maintains a user-friendly registry of every dataset created during the pipeline. Navigate by **data type and stage**.

```python
# Shows Full IO structure (flow view)
ctx.datasets
```

```python
# Example: see how numerical features evolved step-by-step
ctx.datasets.features.num
# Entity: features.num ► DataNode(states=['clean', 'stable', 'dropped', 'encoded', 'selected', 'mapping'])
```
Each level shows the pipeline stages in execution order — how the data was created, modified, and enhanced at each step. No lost DataFrames. No "where did this column come from?"

Alternative view — worker-centric:
```python
# What a specific worker consumed and produced
ctx.workers.fe_cleaner.spec
# .spec.inputs
# .spec.outputs
```

| VIEW | QUESTION IT ANSWERS |
|------|---------------------|
| ctx.datasets.features.num | "How did my numerical features evolve?" |
| ctx.workers.fe_cleaner.spec | "What does the cleaner consume and produce?" |


### Inspect Any Step
AIDflow provides absolute transparency. If you need to inspect exactly how a step was calculated or access internal model coefficients, you can "hop" from the high-level interface directly into the compute engine:

```python
# See what the cleaner did
ctx.workers.fe_cleaner.result

# See exact parameters used
ctx.workers.fe_cleaner.params

# Jump to the compute engine (fitted weights, coefficients)
ctx.workers.fe_cleaner.fitter

# View the full report
ctx.workers.fe_cleaner.report.show()
```

- Inspect Parameters: Access ctx.workers.worker_name.params to see exact configurations.
- Deep Dive into Math: Use ctx.workers.worker_name.fitter to access the underlying computation engine and inspect fitted weights or internal states.

[Explore the Full API Reference](/api/execution-interface/Parameters)