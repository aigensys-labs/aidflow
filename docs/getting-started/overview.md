Transform raw data into production-ready scorecards with absolute reproducibility and rigorous stability.

# Automated Interactive Development Framework

AIDflow is a pipeline-first framework for *binary classification scorecards* in regulated environments. Built for industries where model governance, temporal stability, and regulatory transparency are non-negotiable.

Framework where a human directs and an AI agent executes the full pipeline — stability-first, audit-ready, deployable. No notebook spaghetti. No 6-week committee cycles.

*Built over 15 years of credit scoring, fintech R&D, and regulatory practice.*

![AIDflow Preview](/assets/aidflow-preview.gif)

---

## Who This Is For

- **AI Orchestration Endpoints** — one command surface that lets LLM 
  agents drive the full pipeline (genesis → training → champion) and 
  return a deployable scoring model without human intervention.  
- **Data Scientists** building scorecards in Fintech, Banking, or Insurance
  who need full control over every transformation step.
- **Model Risk Teams** who require complete audit trails, reproducible
  execution, and standardized reporting artifacts.
- **ML Engineers** who need a production scoring artifact with zero
  framework dependency at inference time.

If your scoring model must survive a committee review, AIDflow is your development environment.

---

## Why AIDflow?

- **Human-in-the-Loop.** Every phase is a decision point. Pause, inspect,
re-configure, re-run — without losing state. All in an interactive UI.

- **Context-Driven.** A single `ProjectContext` links every stage from
ingestion to champion export. No fragile variable chains.

- **Stability-First + Ensemble.** Temporal screening runs *before* training.
  Four orthogonal agents compete on stabilized data. An objective scoring
  system promotes the champion.

- **Full Feature Lineage.** Trace any data/feature from raw origin to final score.
Every intermediate state is registered and navigable by stage.

- **Modular Extensibility.** Add a new fitter in minutes. It auto-propagates
to the UI, panels, and reporting — no manual wiring.

- **Self-Contained Scoring.** The exported artifact encapsulates all
encoding and model state. Inference requires no training data or prior
pipeline state.

- **Audit-Ready Governance.** Every production artifact is packaged with full 
lineage, SHAP explainability vectors, and performance benchmarks for seamless 
regulatory approval.

---

## Get Started in Seconds

### Install & Run AIDflow
```bash
# Pure CLI
pip install aidflow
aidflow run demo/

# Interactive UI 
pip install aidflow[notebook]
aidflow notebook ui_demo/
```

---


### The Golden Path to Production
AIDflow guides you through a standardized five-phase journey:

1. **Data Phase**: Ingest → Clean → Stabilize
2. **Encoding Phase**: Numerical → Categorical → Interaction → WOE Encoding
3. **Training Phase**: Train an ensemble of 4 specialized ML Agents
4. **Selection Phase**: Predict → Compare → Transition → Promote Champion
5. **Inference Phase**: Prediction Tests → Production Export


[Read the Developer Handbook](/getting-started/core-concepts)