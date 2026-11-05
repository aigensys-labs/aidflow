The Selection Phase is the decision point. Four agents are trained. One goes to production. AIDflow's approach: **let the evidence decide — not intuition.**

This phase evaluates all agents on the same OOT sample (`test.ml`) across multiple orthogonal dimensions, identifies where they agree and where they diverge, and promotes the champion with a full audit trail.

---

## The Objective: Trained Ensemble → Champion → Production Artifact

The Selection Phase operates as a series of gates. Each gate adds a dimension of evidence. The champion must survive all four.

### Gate 1: Ensemble Scoring (`agents_predictor`)
**The Evidence Base.** Generates probability predictions for all four agents on the OOT sample. This is the fair, "blind" test — every agent scores on the same unseen data, under the same conditions.

*   Produces `predictions.test.agent1` through `predictions.test.agent4`.
*   Creates the raw input for all subsequent evaluation steps.
*   **No decision is made here** — only evidence is generated.

### Gate 2: Performance Benchmarking (`agents_comparator`)
**The Scoring Matrix.** Evaluates all agents across multiple orthogonal dimensions and produces a standardized, weighted composite score.

*   **Performance:** Gini, IV, KS + feature exposure (effective degrees of freedom) + max single-feature impact.
*   **Calibration:** Fraction of probability bins where predicted rate matches observed rate.
*   **Cross-Audit:** How much risk this agent missed (or overstated) relative to the other agents.
*   **Result:** A ranked list. The top agent is the recommended champion.

### Gate 3: Divergence Analysis (`agents_transition`)
**The Agreement Check.** A unique AIDflow safety mechanism. Analyzes where agents agree and where they fundamentally disagree on a sample's risk level.

*   **High agreement** (Agent 1 and Agent 4 both assign high risk): strong confidence in the signal.
*   **High divergence** (Agent 1 says low, Agent 4 says high): a contention segment — the signal is ambiguous or the agents are capturing different patterns.
*   **Why it matters:** A model that is "best on average" but wildly wrong on a specific segment is a liability in production. This gate surfaces that risk before deployment.

### Gate 4: Champion Promotion (`agent_selector`)
**The Executive Decision.** Aggregates evidence from Gates 1–3 and officially promotes one agent. Locks the winning model's state and generates the audit package.

*   User can override the recommendation (choose a different agent or a feature subset).
*   Generates SHAP-based explainability report for the champion.
*   Produces the full selection diagnostics (per-agent SHAP drift, permutation importance, Gini path).
*   **Result:** The champion is locked. The selection is recorded.

### Final Step: Production Artifact (`champion_predictor`)
**The Deployment Unit.** The champion agent is re-fitted (if a feature subset was selected) and packaged as a self-contained inference module.

*   Encapsulates: encoding maps + model parameters + feature lineage.
*   Zero dependency on training data or prior pipeline state at inference time.
*   The output of this step is what you deploy, test, and audit.

---

## Implementation Workflow

The selection process is executed as a sequence of high-level orchestration calls:

```python
ctx.flow.selection.agents_predictor()    # Gate 1: Evidence
ctx.flow.selection.agents_comparator()   # Gate 2: Scoring
ctx.flow.selection.agents_transition()   # Gate 3: Agreement
ctx.flow.selection.agent_selector()      # Gate 4: Promotion
ctx.flow.selection.report()              # Governance report
```

Or run the entire phase at once:

```python
ctx.flow.process('selection')
```

## The Datasets

Once complete, the datasets context holds:

```python
# Champion predictions (train + test)
champion_train = ctx.datasets.predictions.train.champion_agent
champion_test  = ctx.datasets.predictions.test.champion_agent

# Champion feature set
selected  = ctx.datasets.features.champion_agent.selected
dropped   = ctx.datasets.features.champion_agent.dropped

# Full selection audit trail
stats = ctx.datasets.stats.agents.selection

# The production artifact itself
champion = ctx.workers.champion_predictor
```

## The Production Artifact: AgentPredictor
Once a champion is selected, AIDflow packages it into a Production Agent. This artifact is a self-contained inference engine that includes:

- The Locked Fitter: The exact mathematical weights of the winning agent.
- Integrated Preprocessing: All encoders from the Data Phase are bundled inside, ensuring no "Training-Serving Skew" during production scoring.

## What the Governance Report Contains
The report() call produces a stakeholder-ready summary:

- Performance benchmarks — Gini, KS, IV, lift table on OOT.
- Calibration — predicted vs. observed rates per bin.
- Population Stability Index (PSI) - Model and Features view.
- Model Statistical Properties - Econometrical report.
- Feature importance — SHAP values + permutation stability.
- Feature Predictive Stability and more.


## Deep Dive: Technical Reference
For detailed information on the underlying algorithms, parameters, and internal logic, refer to the API specifications:

[View Predictor API](/api/selection-interface/AgentsPredictorWorker) |
[View Comparator API](/api/selection-interface/AgentsComparatorWorker) |
[View Champion API](/api/selection-interface/AgentPredictorWorker)