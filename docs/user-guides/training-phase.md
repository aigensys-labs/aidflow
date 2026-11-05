The Training Phase is where data becomes prediction. AIDflow's approach: **don't choose — compare.** Instead of committing to one algorithm, train four mathematically distinct agents on the same data. Let the evidence decide.

Every agent consumes the identical `train.ml` dataset produced by the Encoding Phase. The only difference between them is the mathematical strategy. This makes the comparison in the Selection Phase fair and objective.

---

## The Objective: ML-Ready Data → Trained Ensemble

All four agents train on the same synchronized dataset. Each applies a different selection and fitting strategy. The result: four independent perspectives on the same signal.

### Agent 1: The Sparse Specialist (`agent1_predictor`)
**Strategy: Stepwise GLM — minimum viable model.**

Finds the smallest feature set that maintains predictive power. Each feature must individually justify its inclusion by statistical significance. No feature survives solely because it correlates with others.

*   Two-stage selection: regularization search → shadow pruning.
*   Produces the most interpretable, auditable model.
*   **Best when:** regulatory transparency is the priority.

### Agent 2: The Dense Optimizer (`agent2_predictor`)
**Strategy: Lasso | ElasticNet | LogReg (L1/L2) GLM — simultaneous shrinkage and selection.**

Uses L1/L2 regularization to drive weak features to exactly zero. From a group of correlated features, it selects one representative — producing a compact model with generalization strength.

*   Handles multicollinearity by picking one from a correlated group.
*   Approx. double the size of features retained — shadow pruning removes the truly redundant.
*   **Best when:** yfeatures form natural correlated groups (e.g., multiple revenue-based WOE features).

### Agent 3: The PLS Stabilizer (`agent3_predictor`)
**Strategy: PLS Projection + Statistical Hypothesis Testing — noise-filtering, stability-enhancing selection.**

Projects features into orthogonal latent components (PLS), then iteratively adds features that significantly improve predictive power as validated by DeLong's test (Gini difference). Handles multicollinearity by evaluating each feature's contribution within the orthogonal structure — not in isolation.

- Forward-stepwise: adds only features that pass a statistical significance threshold (p < alpha).
- Reduces variance by selecting features that contribute to the joint X→y covariance structure.
- Prior VIP cutoff prunes the search space before iteration begins.
- **Best when:** features are highly correlated and you want a stable, variance-reduced model that is robust to sample variation.

### Agent 4: The Non-Linear Powerhouse (`agent4_predictor`)
**Strategy: Gradient Boosting (GBM) — maximum predictive ceiling.**

Captures interactions and non-linear patterns that no GLM can detect. Uses early stopping on a validation split to prevent overfitting. Does not require WOE encoding — works directly on stable numerical and categorical features.

*   Highest capacity of the ensemble.
*   No interpretability guarantee (black-box interactions).
*   **Best when:** you need maximum predictive power and can accept reduced explainability.

---

## Implementation Workflow

The Training Phase begins with a critical synchronization step to ensure all agents see the exact same version of the truth.

```python
ctx.flow.training.agent1_predictor()   # Sparse GLM
ctx.flow.training.agent2_predictor()   # L1/L2 GLM
ctx.flow.training.agent3_predictor()   # PLS GLM
ctx.flow.training.agent4_predictor()   # GBM
```

Or run the entire phase at once:

```python
ctx.flow.process('training')
```

## The Datasets

Once all four agents are trained, the context holds:

```python
# Predictions from each agent (train set)
pred_1 = ctx.datasets.predictions.train.agent1
pred_2 = ctx.datasets.predictions.train.agent2
pred_3 = ctx.datasets.predictions.train.agent3
pred_4 = ctx.datasets.predictions.train.agent4

# Selected features per agent
fes_1 = ctx.datasets.features.agent1.selected
fes_2 = ctx.datasets.features.agent2.selected
fes_3 = ctx.datasets.features.agent3.selected
fes_4 = ctx.datasets.features.agent4.selected

# Training diagnostics per agent
stats_1 = ctx.datasets.stats.agents.agent1
stats_2 = ctx.datasets.stats.agents.agent2
stats_3 = ctx.datasets.stats.agents.agent3
stats_4 = ctx.datasets.stats.agents.agent4
```


## The Comparison Questions
The power of the ensemble is realized in the Selection Phase. With all four trained, you can now ask:

- "Is my signal linear and captured by a small set? (If Agent 1 matches or beats Agent 4.)"
- "Do I need non-linear capacity? (If Agent 4 significantly outperforms all GLM agents.)"
- "Am I overfitting? (If the train→OOT Gini gap is largest for Agent 4.)"


## Deep Dive: Technical Reference
For detailed information on the underlying algorithms, parameters, and internal logic, refer to the API specifications:

[View Agent1 API](/api/training-interface/Agent1PredictorWorker) | [View Agent2 API](/api/training-interface/Agent2PredictorWorker) | [View Agent3 API](/api/training-interface/Agent3PredictorWorker) | [View Agent4 API](/api/training-interface/Agent4PredictorWorker)