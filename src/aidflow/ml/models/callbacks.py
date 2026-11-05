"""
GBM Training Callbacks — Adaptive Learning Rate Scheduling.

This module provides unified training callbacks for the GBM
backends (LightGBM and XGBoost) used by the training agents.
"""

from ...deps.common     import json
from ...deps.ml.gbm     import TrainingCallback
from ...core.logging    import Logger


logger = Logger(__name__)

_HIGHER_IS_BETTER = ("auc", "map", "ndcg", "hit_rate", "f1", "auc_pr")


class ReduceLROnPlateau(TrainingCallback):
    """
    Unified LR-Shrinkage Callback for LightGBM and XGBoost.

    Monitors a validation metric during boosting and
    multiplicatively decays the learning rate when the metric
    stagnates. Compatible with both LightGBM (`__call__` protocol) 
    and XGBoost (`before_training` / `after_iteration` protocol).

    Decay Protocol
    --------------
    1. Track the best score on the monitored dataset/metric.
    2. If the score does not improve for `patience`
       consecutive rounds, decay:
       `lr = max(lr * factor, min_lr)`
    3. Reset the stagnation counter.
    4. If the LR is already at the floor, no further decay.

    Metric Direction
    ----------------
    Whether higher or lower is better is determined by:
    1. Explicit `is_higher_better` override (if provided).
    2. LightGBM's native `env.evaluation_result_list` flag.
    3. Inference from the metric name (e.g., `'auc'` →
       higher is better; `'rmse'` → lower is better).

    Parameters
    ----------
    patience : int = 5
        Rounds without improvement before decay.
    factor : float = 0.5
        Multiplicative decay factor.
    min_lr : float = 1e-6
        Floor.
    dataset_idx : int = 0
        Eval-set index (LightGBM only; XGBoost uses dataset_name_xgb).
    dataset_name_xgb : str = "validation_0"
        Key in XGBoost's evals_log.
    metric_name : str = None
        Metric to monitor. If None, uses the first reported metric.
    is_higher_better : bool = None
        Override. If None, inferred from LGBM env or metric name.
    initial_lr : float = None
        Seed LR. If None, auto-detected from env/config.
    verbose : bool = False
        Log on decay.
    """

    def __init__(
        self,
        patience: int = 5,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        dataset_idx: int = 0,
        dataset_name_xgb: str = "validation_0",
        metric_name: str | None = None,
        is_higher_better: bool | None = None,
        initial_lr: float | None = None,
        verbose: bool = True
    ):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.dataset_idx = dataset_idx
        self.dataset_name_xgb = dataset_name_xgb
        self.metric_name = metric_name
        self.is_higher_better = is_higher_better
        self.initial_lr = initial_lr
        self.verbose = verbose

        self.best_score: float | None = None
        self.no_improvement_count: int = 0
        self.current_lr: float | None = None

    def __repr__(self):
        lr = f"{self.current_lr:.6g}" if self.current_lr else "?"
        return f"{type(self).__name__}(patience={self.patience}, factor={self.factor}, lr={lr})"
    
    # LightGBM
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def __call__(self, env):
        """
        LightGBM callback protocol.

        Invoked after each boosting round. Reads the validation
        score from the environment, checks for stagnation, and
        updates the learning rate if the patience threshold is
        exceeded.
        """        
        if not env.evaluation_result_list:
            return
        if self.dataset_idx >= len(env.evaluation_result_list):
            raise IndexError(f"dataset_idx={self.dataset_idx} out of range")

        data_name, metric, score, lgb_hb = env.evaluation_result_list[self.dataset_idx]

        # Honour explicit metric_name if it differs
        if self.metric_name and metric != self.metric_name:
            for entry in env.evaluation_result_list:
                if entry[1] == self.metric_name:
                    data_name, metric, score, lgb_hb = entry
                    break
            else:
                raise ValueError(f"metric '{self.metric_name}' not in eval results")

        hb = self.is_higher_better if self.is_higher_better is not None else lgb_hb

        if self.current_lr is None:
            self.current_lr = float(env.params.get("learning_rate", 0.1))

        self._evaluate(
            score, hb, env.iteration, data_name, metric,
            lambda lr: env.model.reset_parameter({"learning_rate": lr}),
        )

    # XGBoost
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def before_training(self, model):
        """
        XGBoost callback — called once before training begins.

        Captures the initial learning rate from the model
        configuration.
        """        
        if self.current_lr is None:
            if self.initial_lr is not None:
                self.current_lr = self.initial_lr
            else:
                cfg = json.loads(model.save_config())
                self.current_lr = float(
                    cfg["learner"]["gradient_booster"]["model"]["tree_param"]["learning_rate"]
                )
        return model

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        """
        XGBoost callback — called after each boosting round.

        Reads the validation score from `evals_log`, checks
        for stagnation, and updates the learning rate via
        `model.set_param('eta', lr)`.

        Returns
        -------
        bool
            Always False (training continues; early stopping
            is handled separately).
        """        
        if not evals_log or self.dataset_name_xgb not in evals_log:
            return False

        ds_log = evals_log[self.dataset_name_xgb]
        metric = self.metric_name or next(iter(ds_log))
        if metric not in ds_log:
            return False

        score = ds_log[metric][-1]
        hb = self._infer_higher_better(metric)

        if self.current_lr is None:
            self.before_training(model)  # lazy init fallback

        def _set_lr(new_lr: float):
            model.set_param("eta", new_lr)

        self._evaluate(score, hb, epoch, self.dataset_name_xgb, metric, _set_lr)
        return False

    # Shared logic 
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def _infer_higher_better(self, metric: str) -> bool:
        if self.is_higher_better is not None:
            return self.is_higher_better
        return metric.lower() in _HIGHER_IS_BETTER

    def _evaluate(self, score, hb, iteration, ds_label, metric_label, update_fn):
        """
        Core stagnation detection and LR decay logic.

        Parameters
        ----------
        score : float
            Current metric value on the monitored dataset.
        hb : bool
            Whether higher is better.
        iteration : int
            Current boosting round (for logging).
        ds_label : str
            Dataset name (for logging).
        metric_label : str
            Metric name (for logging).
        update_fn : callable
            Function that sets the new learning rate on the
            model.
        """        
        if self.best_score is None:
            self.best_score = score
            return

        improved = score > self.best_score if hb else score < self.best_score
        if improved:
            self.best_score = score
            self.no_improvement_count = 0
            return

        self.no_improvement_count += 1
        if self.no_improvement_count < self.patience:
            return

        new_lr = max(self.current_lr * self.factor, self.min_lr)
        if new_lr >= self.current_lr:
            self.no_improvement_count = 0
            return  # already at floor

        self.current_lr = new_lr
        self.no_improvement_count = 0
        update_fn(new_lr)

        if self.verbose:
            logger.info(
                type(self).__name__ + " | [%d] %s/%s stagnant %d rounds → lr=%.6f",
                iteration, ds_label, metric_label, self.patience, new_lr,
            )

    def reset(self):
        """
        Reset all internal state for a new training run.

        Clears the best score, stagnation counter, and current
        LR — allowing the same callback instance to be reused
        across multiple training runs.
        """
        self.best_score = None
        self.no_improvement_count = 0
        self.current_lr = None