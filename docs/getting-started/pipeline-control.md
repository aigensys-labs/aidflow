AIDflow is not a rigid pipeline. You control what runs, in what order, and with what data.

The Golden Path (`ctx.flow.process('all')`) is the default. But you can skip steps, train 
a subset of agents, sideload in-memory data — all without breaking sequence safety.

---

## Skip Steps

Not every agent needs every step. Agent 4 (GBM) works directly on stable numerical features — no WOE encoding required.

**Minimal GBM-only path:**

```python
import aidflow as aid

# generate demo data
data_path = aid.assets.DataGetter.create_data('small')

# define context
ctx = aid.ProjectContext()

ctx.genesis(
    project      = 'test',
    input_data   = data_path,
    target       = 'target',
    yy_mm        = 'time',
    yy_mm_format = '%Y-%m-%d'
)
ctx.params(save=1, logging=1, verbose=1)

# Data phase (clean + stabilize)
ctx.flow.process('data')

# Skip encoding entirely — GBM doesn't need WOE
# Just create train-test samples
ctx.workers.data_encoder()

# Train GBM only
ctx.workers.agent4_predictor()

# Set as champion directly
ctx.workers.champion_predictor(agent_choice=4)
```

> When to skip encoding: Agent 4 (GBM) operates directly on stable numerical and categorical features. 
> If your only goal is a non-linear model with maximum predictive power, WOE is unnecessary overhead.

---

## Train a Subset

You don't have to train all four agents. Train the ones you need — as long as at least 2 are fitted, 
the Selection Phase runs its full comparison and promotes the objective champion.

**Two-agent competitive path:**

```python
# Data + Encoding (standard)
ctx.flow.process('data')
ctx.flow.process('encoding')

# Train only Agent 2 (ElasticNet) and Agent 4 (GBM)
ctx.workers.agent2_predictor()
ctx.workers.agent4_predictor()

# Full Selection Phase — comparison, transition, champion
ctx.flow.process('selection')
```

> Constraint: Selection requires ≥ 2 fitted agents. With only 1, there's nothing to compare — set it as 
> champion directly via champion_predictor(agent_choice=N).

---

## Sideload In-Memory Data

Already have your data in a DataFrame? Skip file I/O entirely. Pass it directly 
to process — the import step is bypassed.

```python
import pandas as pd

df = pd.read_csv('train.csv')  # or Spark, DB, API — any source

ctx.flow.process('all', data=df, verbose=1)
```

> Applies to: phase='all' or phase='data' only. The data is used as-is for downstream 
> processing — input_data in genesis is ignored.

---

## Execution Modes at a Glance

| GOAL | COMMAND | STEPS SKIPPED |
|------|---------|---------------|
| Full pipeline (default) | ctx.flow.process('all') | None |
| GBM-only, no WOE | process('data') → data_encoder() → agent4_predictor() | Encoding + Selection |
| 2-agent competition | process('data') + process('encoding') → 2 agents → process('selection') | 2 of 4 agents |
| In-memory data | process('all', data=df) | File import |
| Single step | ctx.workers.fe_cleaner() | Everything else |


## When to Use What

| SCENARIO | MODE | 
|----------|------|
| Automated re-training (CLI, Airflow) | process('all') — full pipeline, no interaction |
| GBM-only model (max predictive power) | Skip encoding → train Agent 4 → set champion |
| Head-to-head comparison | Train 2 agents → full selection |
| Data already loaded (Spark, DB) | process('all', data=df) |
| Debugging one step | ctx.workers.worker_name() |
| Stakeholder presentation (UI) | Phase-by-phase with verbose=2 |


[Explore the Full API Reference](/api/execution-interface/Parameters)