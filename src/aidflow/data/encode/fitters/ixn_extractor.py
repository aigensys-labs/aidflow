"""
Feature Interaction Extraction (Tree Mining + Binary Encoding).

This module implements the interaction discovery stage of the pipeline,
responsible for extracting significant non-linear feature interactions
and encoding them as binary (0/1) categorical variables.

Methodology
-----------
1. **Train a shallow GB model**: A LightGBM or XGBoost model with
   constrained depth (`max_leaves`) is trained on the input features
   against the target. The model serves as a *search mechanism*, not
   a predictive model — its tree structure is mined for patterns.
2. **Mine interaction rules**: The `TreeMiner` extracts rules
   (conjunctions of feature conditions) from the tree leaves that
   satisfy minimum support and target-rate thresholds. Rules that are
   highly correlated (redundant) are pruned.
3. **Gate on predictive power**: Each extracted rule is measured
   (Gini/IV). Rules below the Gini cutoff are dropped.
4. **Encode as binary**: Surviving rules become binary (0/1)
   categorical features — 1 if the observation matches the rule
   condition, 0 otherwise.

Why Shallow Trees?
------------------
Deep trees produce highly specific rules (many conditions combined)
that overfit to training noise. A shallow tree (≤31 leaves)
produces broader, more stable interaction patterns that generalise
to OOT data. The constraint acts as a regulariser on the
interaction complexity.

Why Binary Encoding?
--------------------
Each interaction is a discrete *event* (the condition is either met
or not). Binary encoding is the natural representation: it is
category-agnostic, stable under monotone transforms, and directly
consumable by the WOE encoder.

Output
------
Per interaction:
- A binary (0/1) column indicating rule membership.
- The rule condition (e.g., `revenue > 500 AND segment == 'B'`).
- The contributing features.
- Performance metrics (Gini, IV, population share, target rate).
"""

from ....deps.struct                import  dataclass
from ....deps.data                  import  pDF, pSe

from ....core.base                  import  (
                                            FitterDataBase, 
                                            FitterModelBase,
                                            )
from ....core.factory               import  (
                                            Dataclass, 
                                            DictClass
                                            )
from ....core.foundation            import  AigParam
from ....core.standard              import  VarType
from ....core.accessor              import  Spinner

from ....ml.features                import  (
                                            TreeMiner, 
                                            RuleStats, 
                                            Rule,
                                            EncodingMap
                                            )
from ....ml.models                  import  XGB, LGB
from ....ml.metrics                 import  SeparationMetrics
from ....ml.sampling                import  (
                                            Sampling,
                                            Subsampling
                                            )
from ....display.verbosity          import  Verbosity



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONFIG | CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    """Output column labels for interaction stats."""
    V  = 'VAR'
    R  = 'Reason'
    S  = 'Ixn_Pop_Share'
    T  = 'Ixn_Target_Rate'
    G  = 'Gini_TF'
    IV = 'IV_TF'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class GB(DictClass):
    """Implemented Estimators."""       
    X = XGB
    L = LGB

@Dataclass.base
class EstimatorCfg:
    """
    Estimator selection and configuration.

    Attributes
    ----------
    choice : XGB | LGB
        The gradient boosting backend to use for tree mining.
    config : dict
        Additional hyperparameters passed to the estimator
        (merged with the pipeline defaults).
    """
    choice: XGB|LGB
    config: dict
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → IXN → TRANSFORM
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class FitParamsBase:
    """
    Interaction mining parameters.

    Attributes
    ----------
    gb_max_leaves : int, default=31
        Maximum leaves per tree in the mining model. Constrains
        interaction complexity (fewer leaves = broader, more stable
        rules). 31 ≈ depth-5 trees, which capture 2-3 feature
        interactions without over-specialisation.
    gb_estimators : int, default=100
        Number of boosting rounds. More estimators provide more tree
        structure to mine from, but with diminishing returns.
    gb_learning_rate : float, default=0.1
        Boosting learning rate. Affects the magnitude of leaf splits
        (not the rule structure directly, but influences which splits
        the miner considers significant).
    gb_min_child_weight : float, default=0.1
        Minimum Hessian sum in a leaf. Prevents the model from fitting
        to extremely small, noisy subgroups (which would produce
        unstable interaction rules).
    ixn_min_target : float, default=0.15
        Minimum target rate within a rule's population. Rules where
        the target event rate is below this are considered too weak
        to be meaningful interactions.
    ixn_min_share : float, default=0.03
        Minimum population share for a rule. Rules matching fewer than
        3% of observations are too rare to be stable in production.
    ixn_max_corr : float, default=0.9
        Maximum correlation among extracted rules. Rules with
        correlation above this threshold are considered redundant
        (they capture the same pattern) and one is pruned.
    gini_cutoff : float, default=0.02
        Minimum Gini coefficient for an interaction to be kept.
        Applied after extraction as a final predictive gate.
    """ 
    gb_max_leaves:       int   = 31
    gb_estimators:       int   = 100
    gb_learning_rate:    float = 0.1
    gb_min_child_weight: float = 0.1
    ixn_min_target:      float = 0.15
    ixn_min_share:       float = 0.03
    ixn_max_corr:        float = 0.9
    gini_cutoff:         float = 0.02

@Dataclass.base(update_specs=dict(slots=False))
class FitParamsCfg:
    """
    Pipeline configuration.

    Attributes
    ----------
    target : str
        Target column name. Interactions are mined against this target.
    val_size : float, default=0.33
        Validation split fraction for early stopping during the
        mining model fit.
    random_state : int, default=42
        Seed for reproducibility (train/valid split, estimator).
    verbose : bool | Verbosity, default=True
        Enable progress display and stats output.
    """   
    target:         str
    val_size:       float = 0.33
    random_state:   int = 42
    verbose:        bool|Verbosity = True

@Dataclass.base
class FitParams(FitParamsBase, FitParamsCfg):
    """Fitter params input design."""

@Dataclass.base
class FitResult:
    """
    Fit output: interaction encodings and feature verdicts.

    Attributes
    ----------
    encoding : dict
        Per-interaction encoding record: `EncodingMap` containing
        the rule condition, contributing features, type, and stats.
    vars_ixn : list
        Interaction names that passed all gates.
    vars_org : list
        Original features that contribute to at least one kept
        interaction (for downstream awareness).
    vars_dropped : list
        Interaction names that failed a gate, with reason.
    vars_map : dict
        Mapping of interaction name → contributing feature names.
    """
    encoding:       dict
    vars_ixn:       list
    vars_org:       list
    vars_dropped:   list
    vars_map:       dict

@Dataclass.base
class FitStats:
    """
    Diagnostic statistics for user-facing display.

    Attributes
    ----------
    fit_worker : dict
        Per-interaction worker output (rule condition, stats).
    stats_in : list
        Interactions that PASSED the gate (with performance metrics).
    stats_out : list
        Interactions DROPPED (with reason).
    """
    fit_worker: dict
    stats_in:   list
    stats_out:  list

@Dataclass.base
class WorkerResult:
    """
    Per-interaction worker output (for detailed inspection).

    Attributes
    ----------
    encoding : Rule
        The interaction rule (condition + features).
    stats : pDF | None
        Full rule statistics (population share, target rate,
        supporting trees, etc.).
    """
    encoding: Rule
    stats: None|pDF = None
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class FitWorker:
    """
    Interaction Mining Worker.

    Executes the full pipeline:
    1. Train a shallow GB model (search mechanism).
    2. Mine interaction rules from the tree structure.
    3. Gate each rule on predictive power (Gini).
    4. Register surviving rules as binary (0/1) encodings.

    The GB model is NOT used for prediction — it is a search tool
    whose tree structure is decomposed into interpretable rules.
    """
    implemented_estimators = GB
    
    def __init__(self, params:FitParams, config:EstimatorCfg = None):
        self.prm     = params
        self.target  = params.target
        self.cutoff  = params.gini_cutoff
        self.verbose = params.verbose

        self.miner_est = TreeMiner
        self._set_estimator(config)

        self.result: FitResult = Dataclass.create_default_instance(FitResult)
        self.stats:  FitStats  = Dataclass.create_default_instance(FitStats)

        self._diag = SeparationMetrics()
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    @dataclass(slots=True)
    class VarData:
        """Per-interaction working state."""
        x:           str
        features:    list
        pop_share:   float
        target_rate: float
        target_sum:  int
        condition:   Rule

    @Dataclass.base
    class Performance:
        """Predictive performance metrics for an interaction."""
        gini: float|int
        iv:   float|int
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def _set_estimator(self, config:EstimatorCfg | None) -> None:
        """
        Resolve the gradient boosting estimator and its configuration.
        Defaults to LightGBM if no explicit config is provided.        
        """        
        est, cfg = GB.L, {}
        if config is not None:
            est = config.choice
            cfg = config.config        
        self.gb_est: XGB|LGB = est
        self.gb_cfg: dict = cfg
    
    def _get_setup(self, cat:list) -> dict:
        """
        Assemble the GB estimator constructor arguments.

        Merges: pipeline defaults → user config → interaction-mining
        hyperparameters (max_leaves, n_estimators, etc.).        
        """        
        default = dict(
            early_stopping_round = 100,
            random_state         = self.prm.random_state,
        )
        prm = dict(
            max_leaves           = self.prm.gb_max_leaves,
            n_estimators         = self.prm.gb_estimators,
            learning_rate        = self.prm.gb_learning_rate,
            min_child_weight     = self.prm.gb_min_child_weight,
            categorical_feature  = cat
        )
        return default | self.gb_cfg | prm

    def _train_valid_split(self, data:pDF, num:list, cat:list
    ) -> tuple[ tuple[pDF,pSe], tuple[pDF,pSe] ]:
        """Create a stratified train/validation split for early stopping."""
        return Sampling(
            sample_size  = self.prm.val_size, 
            stratify     = data[self.target],
            random_state = self.prm.random_state, 
            shuffle      = True
        ).train_valid_split(data[num+cat], data[self.target])
    
    def fit_gb(self, data:pDF, num:list, cat:list) -> None:
        """
        Train the shallow GB model (interaction search mechanism).

        The model is fitted with early stopping on a validation split.
        It is NOT used for prediction — only its tree structure is
        subsequently mined by `TreeMiner`.

        Parameters
        ----------
        data : pDF
            Full dataset.
        num : list
            Numerical feature names.
        cat : list
            Categorical feature names.
        """        
        Xy_train, Xy_valid = self._train_valid_split(data, num, cat)
        self.gb: XGB|LGB = self.gb_est(**self._get_setup(cat))
        self.gb.fit(*Xy_train, eval_set = [Xy_valid])

    def fit_miner(self, data:pDF, num:list, cat:list) -> None:
        """
        Mine interaction rules from the fitted GB model.

        Extracts rules from tree leaves that satisfy:
        - Minimum population share (`ixn_min_share`).
        - Minimum target rate (`ixn_min_target`).
        - Maximum correlation among rules (`ixn_max_corr`).

        Parameters
        ----------
        data : pDF
            Full dataset.
        num : list
            Numerical feature names.
        cat : list
            Categorical feature names.
        """        
        self.miner = self.miner_est(
            self.gb,
            min_rule_support    = self.prm.ixn_min_share,
            min_target_support  = self.prm.ixn_min_target,
            overlap_threshold   = self.prm.ixn_max_corr,                
            max_correlation     = self.prm.ixn_max_corr,
            random_state        = self.prm.random_state,
            verbose             = False
        ).fit(data[num+cat], data[self.target])
   
    def get_interactions(self, data:pDF) -> pDF:
        """
        Generate binary (0/1) interaction columns for the full data.

        Each column represents one mined rule: 1 if the observation
        matches the rule condition, 0 otherwise.

        Parameters
        ----------
        data : pDF
            Full dataset (all columns).

        Returns
        -------
        pDF
            Binary matrix (one column per interaction rule).
        """        
        return self.miner.transform(data)

    def create_var(self, x:str) -> VarData:
        """
        Create the working state for a single interaction rule.

        Parameters
        ----------
        x : str
            Interaction name (rule ID from the miner).

        Returns
        -------
        VarData
            Rule metadata (features, pop share, target rate, condition).
        """        
        ixn:RuleStats = self.miner.get_rule(x)
        return self.VarData(
            x           = x,
            features    = ixn.features,
            pop_share   = ixn.pop_share,
            target_rate = ixn.target_rate,
            target_sum  = ixn.target_rate,
            condition   = ixn.condition
        )
        
    def calc_perf(self, y:pSe, x:pSe) -> Performance:
        """
        Compute Gini and IV for a binary interaction variable.

        Parameters
        ----------
        y : pSe
            Target values.
        x : pSe
            Binary interaction values (0/1).

        Returns
        -------
        Performance
            Gini and IV of the interaction.
        """        
        return self.Performance(
            gini = abs( self._diag.gini(y, x.astype(int)) ),
            iv   = self._diag.iv(y, x)
        )    
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def is_predictive(self, var:VarData, perf:Performance) -> bool:
        """
        Gate: check if the interaction meets the Gini threshold.

        Parameters
        ----------
        var : VarData
            Interaction working state.
        perf : Performance
            Gini and IV of the interaction.

        Returns
        -------
        bool
            True if the interaction passes. If False, it is
            registered as dropped with the reason.
        """        
        if ( perf.gini > self.cutoff ):
            return True
        reason = f'Transformation < TH {self.cutoff}'
        self.set_res_vars_out(var, perf, reason)        
        return False    
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def stats_io(self, IO:list, var:VarData, perf:Performance, reason:str) -> None: 
        """
        Append an interaction's verdict record to the stats output list.

        Parameters
        ----------
        IO : list
            Target stats list to append to.
        var : VarData
            Interaction working state.
        perf : Performance
            Gini and IV.
        reason : str
            Verdict reason.
        """        
        IO.append({
            LBL.V:  var.x,
            LBL.R:  reason,
            LBL.S:  var.pop_share,
            LBL.T:  var.target_rate,
            LBL.IV: perf.iv,
            LBL.G:  perf.gini,
        })

    def set_res_vars_out(self, var:VarData, perf:Performance, reason:str) -> None: 
        """Register an interaction as DROPPED (failed the gate)."""
        self.result.vars_dropped.append(var.x)
        io = self.stats.stats_out
        reason = 'Unpredictive → ' + reason
        self.stats_io(io, var, perf, reason)

    def set_res_vars_in(self, var:VarData, perf:Performance) -> None: 
        """
        Register an interaction as PASSED (kept).

        Also tracks the contributing original features in
        `result.vars_org` for downstream awareness.
        """        
        self.result.vars_ixn.append(var.x)
        fes = [x for x in var.features if x not in self.result.vars_org]
        self.result.vars_org.extend(fes)
        io = self.stats.stats_in
        reason = f'Passed TH {self.cutoff}'
        self.stats_io(io, var, perf, reason)

    def set_res_encoding(self, var:VarData, perf:Performance) -> None:
        """
        Register the final interaction encoding.

        Stores:
        - The rule condition in `result.encoding`.
        - The contributing features in `result.vars_map`.
        - The worker diagnostics in `stats.fit_worker`.
        """        
        self.result.vars_map[var.x] = var.features
        self.result.encoding[var.x] = EncodingMap(
            vars  = var.features, 
            type  = VarType.C,
            map   = var.condition,
            stats = perf
        )
        self.stats.fit_worker[var.x] = WorkerResult(
                encoding = var.condition,
                stats    = self.miner.get_rule(var.x)
        )

    def encode(self, var:VarData, perf:Performance) -> None:
        """
        Register the interaction encoding (gate already passed).

        Parameters
        ----------
        var : VarData
            Interaction working state.
        perf : Performance
            Gini and IV.
        """
        self.set_res_vars_in(var, perf)
        self.set_res_encoding(var, perf)     
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def pipeline(self, data:pDF, num:list, cat:list) -> None:     
        """
        Execute the full interaction mining pipeline.

        Steps:
        1. Train shallow GB model.
        2. Mine interaction rules (with support/correlation filters).
        3. Generate binary interaction columns.
        4. Gate each rule on Gini.
        5. Register surviving encodings.

        Parameters
        ----------
        data : pDF
            Full dataset (features + target).
        num : list
            Numerical feature names.
        cat : list
            Categorical feature names.
        """        
        self.fit_gb(data, num, cat)
        self.fit_miner(data, num, cat)
        
        ixn = self.get_interactions(data)
        y = data[self.target]

        for x in ixn.columns:

            perf = self.calc_perf(y, ixn[x])
            var  = self.create_var(x)

            if not self.is_predictive(var, perf):
                continue

            self.encode(var, perf)
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def process(self, data:pDF, num:list, cat:list) -> tuple[TreeMiner, FitResult, FitStats]:
        """
        Entry point: execute the mining pipeline and return results.

        Parameters
        ----------
        data : pDF
            Full dataset.
        num : list
            Numerical feature names.
        cat : list
            Categorical feature names.

        Returns
        -------
        tuple[TreeMiner, FitResult, FitStats]
            Fitted miner (for transform), encoding maps, and stats.
        """       
        self.pipeline(data, num, cat)
        return self.miner, self.result, self.stats
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Fitter(FitterDataBase):
    """
    Interaction Extraction — execution backend.

    Orchestrates the fit-transform contract for interaction mining:

    - **fit**: Train the shallow GB model, mine rules, gate on
      predictive power, and store binary encodings.
    - **transform**: Apply the mined rules to new data, producing
      binary (0/1) interaction columns.

    Parameters
    ----------
    params : FitParams
        Mining thresholds and pipeline configuration.
    config : EstimatorCfg | None
        Estimator selection (LGB/XGB) and extra config.
    """
    
    def __init__(self, params:FitParams, config: EstimatorCfg = None):
        self.prm = params 
        self.cfg = config

    def fit(self, data:pDF, num:list, cat:list):
        """
        Fit the interaction mining model on the feature set.

        Trains the shallow GB model, mines rules, gates on predictive
        power, and stores the surviving binary encodings.

        Parameters
        ----------
        data : pDF
            Training data (must contain target + num/cat features).
        num : list
            Numerical feature names.
        cat : list
            Categorical feature names.

        Returns
        -------
        self
            Fitted instance with `encoding`, `features`, and
            `miner` set.
        """
        worker = FitWorker(self.prm, self.cfg)

        t = 'Interaction transformation in progress'
        with Spinner(t, show_timer=True).progress(self.prm.verbose):
            self.miner, self.result, self.stats = worker.process(data, num, cat)

        self.encoding = self.result.encoding
        self.features = self.result.vars_ixn
        
        return self
    
    def _verify_subset(self, ixn_subset:list) -> None:
        if not [x for x in ixn_subset if x not in self.encoding]:
            return
        raise(
            'Requested Subset of Interaction Vars has not been fitted.'
            f'\n► Available IXN:\n {', '.join(list(self.encoding))}'
            f'\n► Requested IXN:\n {', '.join(ixn_subset)}'
        )        
        
    def transform(self, data:pDF, ixn_subset:list=None) -> pDF:
        """
        Apply fitted interaction rules to new data.

        Evaluates each mined rule condition against the input data,
        producing binary (0/1) columns.

        Parameters
        ----------
        data : pDF
            New data containing the original features referenced by
            the mined rules.
        ixn_subset : list, optional
            Specific interaction names to evaluate. If None, all
            fitted interactions are produced.

        Returns
        -------
        pDF
            Binary (0/1) DataFrame — one column per interaction rule.
        """      
        subset = self.encoding
        if ixn_subset is not None:
            self._verify_subset(ixn_subset)
            subset = ixn_subset
        return self.miner.transform(data, subset)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → IXN → ENCODER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class IxnUtils:
    """
    Utility helper for interaction extraction data preparation.

    Resolves feature lists and applies stratified subsampling
    (GBM-optimized) to reduce computation cost for large datasets.
    """
    def __init__(self, target:str):
        self.target = target

    def get_fes(self, data:pDF, num:list, cat:list) -> tuple[list, list]:
        if num is None and cat is None:
            return [x for x in data.columns if x != self.target], []  # default setup
        return num or [], cat or []
    
    def subsample(self, data:pDF) -> pDF:   
        return Subsampling(data, data[self.target]).gbm()[0]
            
    def get_data(self,
        data:   pDF, 
        num:    list = None, 
        cat:    list = None,
    ) -> tuple[pDF, list, list]:
        """
        Prepare the data for interaction mining.

        Resolves feature lists, selects relevant columns, and applies
        subsampling.

        Parameters
        ----------
        data : pDF
            Full dataset.
        num : list, optional
            Numerical features.
        cat : list, optional
            Categorical features.

        Returns
        -------
        tuple[pDF, list, list]
            (subsampled_data, num_features, cat_features)
        """        
        num, cat = self.get_fes(data, num, cat)
        cols = [self.target] + num + cat
        
        return self.subsample(data[cols]), num, cat
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class InteractionExtractor(FitterModelBase[Fitter]):
    """
    Feature Interaction Extraction (Tree Mining + Binary Encoding).

    Discovers significant non-linear feature interactions by training
    a shallow gradient boosting model and mining its tree structure
    for interpretable rules. Each surviving rule is encoded as a
    binary (0/1) categorical variable.

    Parameters
    ----------
    target : str
        Target column name. Interactions are mined against this target.
    gb_max_leaves : int, default=31
        Maximum leaves per tree (interaction complexity constraint).
    gb_estimators : int, default=100
        Number of boosting rounds for the mining model.
    gb_learning_rate : float, default=0.1
        Boosting learning rate.
    gb_min_child_weight : float, default=0.1
        Minimum Hessian sum in a leaf (noise filter).
    ixn_min_target : float, default=0.15
        Minimum target rate within a rule (signal strength gate).
    ixn_min_share : float, default=0.03
        Minimum population share of a rule (stability gate).
    ixn_max_corr : float, default=0.9
        Maximum correlation among rules (redundancy filter).
    gini_cutoff : float, default=0.02
        Minimum Gini for an interaction to be kept.
    val_size : float, default=0.33
        Validation split fraction for early stopping.
    random_state : int, default=42
        Seed for reproducibility.
    verbose : bool | Verbosity, default=True
        Enable progress and stats display.

    Notes
    -----
    **Search, not prediction**: The GB model is a search mechanism.
    Its tree structure is decomposed into interpretable rules. The
    model itself is never used to predict.

    **Shallow by design**: Constrained tree depth (≤31 leaves)
    ensures broad, stable interaction patterns rather than
    over-specific training noise.

    **Binary output**: Each interaction is a discrete event
    (rule met / not met). Binary encoding is category-agnostic,
    stable, and directly consumable by the WOE encoder.

    **Redundancy control**: Highly correlated rules (above
    `ixn_max_corr`) are pruned to avoid capturing the same
    pattern multiple times.

    Examples
    --------
    >>> from aidflow.fitters import InteractionExtractorLGB
    >>> ixn = InteractionExtractorLGB(
    ...     target='is_churn',
    ...     gb_max_leaves=31,
    ...     ixn_min_share=0.03,
    ...     ixn_min_target=0.15,
    ... )
    >>> ixn.fit(data, num_features=['revenue', 'tenure'], cat_features=['segment'])
    >>> encoded = ixn.transform(data_new)
    >>> encoded.columns  # ['ixn_0', 'ixn_1', ...]
    """      
    _estimator: GB = None

    def __init__(self, *args, params: dict|FitParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(FitParams, params, args, kwargs)
        self._config = EstimatorCfg(
            choice = self._estimator, 
            config = self.get_estimator_config(self._params, kwargs)
        )
        
    def fit(self, data:pDF, num_features:list = None, cat_features:list = None):        
        """
        Fit the interaction mining model.

        Trains the shallow GB model, mines interaction rules, gates
        on predictive power, and stores the surviving binary encodings.

        Parameters
        ----------
        data : pDF
            Training data (must contain target + features).
        num_features : list | None, default=None
            Numerical features to search for interactions.
            If None, all non-target columns are used.
        cat_features : list | None, default=None
            Categorical features to search for interactions.

        Returns
        -------
        self
            Fitted instance ready for `transform`.
        """           
        di, num_, cat_ = IxnUtils(self._params.target).get_data(data, num_features, cat_features)
        self._fitter = Fitter(self._params, self._config)
        self._fitter.fit(di, num_, cat_)
        return self
    
    def transform(self, data:pDF, ixn_subset:list=None) -> pDF:
        """
        Apply fitted interaction rules to new data.

        Parameters
        ----------
        data : pDF
            New data containing the original features.
        ixn_subset : list, optional
            Specific interactions to produce. If None, all are produced.

        Returns
        -------
        pDF
            Binary (0/1) DataFrame — one column per interaction rule.
        """          
        return self.fitter.transform(data, ixn_subset)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@Dataclass.verify
class InteractionExtractorLGB(InteractionExtractor):
    _estimator = GB.L
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@Dataclass.verify
class InteractionExtractorXGB(InteractionExtractor):
    _estimator = GB.X
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬