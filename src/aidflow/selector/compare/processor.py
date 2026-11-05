"""
Multi-Agent Model Comparison and Selection Scoring.

This module implements the model comparison stage of the AIDflow
pipeline, providing an objective, multi-dimensional assessment of
all trained agents (GLM and GBM variants) to determine the optimal
model for production deployment.

Comparison Dimensions
---------------------
Agents are scored across four orthogonal dimensions:

1. **Performance** (PE):
   - Gini, IV, KS: How well the agent separates target classes.
   - Feature Exposure: Effective degrees of freedom (Shannon entropy
     of permutation importance). Lower = simpler, more stable.
   - Max Feature Impact: The single strongest feature's contribution.
     Lower = less single-point-of-failure risk.

2. **Calibration** (CA):
   - Fraction of probability bins where the agent's predicted rate
     significantly deviates from the observed rate. Higher = more
     reliably calibrated predictions.

3. **Cross-Audit** (AE):
   - **Audited**: How much "blind spot" other agents detect in this
     agent's predictions (weakness).
   - **Auditor**: How much new pattern information this agent reveals
     about other agents' blind spots (strength).

4. **Operational Risk** (embedded in PE):
   - Simpler models (fewer effective features) are preferred for
     production stability, interpretability, and maintenance.

Scoring Methodology
-------------------
- **Standardization**: Leader-Anchored Proportional Scaling — the
  best agent gets score 1.0, others are scaled proportionally.
  Negative-scale metrics are inverted so that higher always = better.
- **Weighting**: Each metric is assigned a weight (default: equal).
  The weighted sum produces the final composite score.
- **Ranking**: Agents are ordered by descending composite score.

The result is an objective, transparent, reproducible model selection
criterion — not a single metric, but a holistic assessment across
predictive power, stability, calibration, and operational risk.
"""

from ...deps.data                   import  np, pDF, pSe
from ...deps.plots                  import  go
from ...deps.types                  import  (
                                            Literal,
                                            Callable, 
                                            ArrayLike
                                            )
from ...core.base                   import  (
                                            FitterDataBase, 
                                            FitterModelBase
                                            )
from ...core.factory                import  (
                                            Dataclass, 
                                            DictClass
                                            )
from ...core.foundation             import  AigParam
from ...core.display                import  Display
from ...core.interop                import  Pd
from ...ml.metrics                  import  (
                                            LBL as LBL_M,
                                            DiagnosticsResult,
                                            SeparationDiagnostics,
                                            GiniLiftPlot as GLP
                                            )
from ...ml.models.calibration       import  (
                                            LBL as LBL_C,
                                            CalibrationAssessment,
                                            CalibrationPlot
                                            )
from ...ml.models.transition        import  (
                                            LBL as LBL_T,
                                            ModelsCrossTransition,
                                            ModelCrossDivergence
                                            )
from ...learner.features.importance import  (
                                            MTD as FI_M, 
                                            LBL as FI_L,
                                            FeatureImportanceResult
                                            )
from ...display.style               import  TableStyle

from ..base.processor               import  (
                                            AGENTS,
                                            SelectorInputBase
                                            )
from ..predict.adapter              import  PredictorAdapter


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TUNING → AGENTS VOTING ON OOT
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    """Scoring dimension labels and display names."""
    N  = lambda x: f'Agent {x}'
    A  = 'alpha_level'
    S  = 'Total'
    U  = 'Underestimated'
    O  = 'Overestimated'

class STAT(DictClass):
    """Per-agent performance and operational metric keys."""
    FE  = 'fe_exposure'
    FX  = 'fe_max_impact'
    FS  = 'fe_selected'
    GT  = 'gini_test'
    IV  = 'IV_test'
    KS  = 'KS_test'
    PR  = 'positive_rate'

class RANK(DictClass):
    """Scoring dimension identifiers."""
    PE = 'performance'
    CA = 'calibration'
    AE = 'audited'
    AO = 'auditor'

class MAP(DictClass):
    """
    Column mapping, weights, and scoring configuration.

    Defines:
    - The canonical column names for the scoring table.
    - The default weights for each metric in the final score.
    - The direction (higher/lower is better) for each metric.
    """
    D = 'higher_is_better'
    W = 'WEIGHT'
    S = 'TOTAL_WEIGHTED_SCORE'

    COL = {
        STAT.GT:  'eval_test_gini',
        STAT.IV:  'eval_test_grid_iv',
        STAT.KS:  'eval_test_grid_ks',
        RANK.CA:  'eval_model_calibration',
        STAT.FE:  'ops_features_exposure',     
        STAT.FX:  'ops_feature_max_impact',
        LBL.U:    'audit_underestimated_risk',
        LBL.O:    'audit_overrestimated_risk',
        LBL.S:    'audit_agent_as_cross_auditor'    
    }
    WEIGHT = dict(zip(COL.values(), [1.0]*len(COL)))
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@Dataclass.base
class ComparatorParamsBase:
    """
    Comparison scoring parameters.

    Attributes
    ----------
    alpha : float, default=0.05
        Statistical significance level for calibration and
        cross-transition assessments.
    proba_bins : int, default=10
        Number of quantile bins for probability discretization
        (used in calibration, cross-transition, and lift analysis).
    metric_weight : dict, default={equal weights}
        Per-metric weights for the composite score.
        Higher weight = more influence on the final ranking.
    """    
    alpha:        float = 0.05
    proba_bins:     int = 10
    metric_weight: dict = Dataclass.field(MAP.WEIGHT)

@Dataclass.base(update_specs=dict(slots=False))
class ComparatorParamsCfg:
    """
    Comparison input configuration.

    Attributes
    ----------
    predictors : PredictorAdapter
        Adapter holding the trained agents to compare.
    target : str
        Binary target column name in the evaluation data.
    preprocess : bool | None, default=None
        If True, apply the full encoding pipeline to input data
        before prediction.
    """    
    predictors: PredictorAdapter
    target:     str
    preprocess: bool|None = None

@Dataclass.base
class ComparatorParams(ComparatorParamsBase, ComparatorParamsCfg):
    """Woker params input design."""

@Dataclass.base
class ComparatorInput:
    """
    Comparator input specification.

    Attributes
    ----------
    data : pDF
        OOT evaluation data (must contain the target column).
    agents : tuple | list
        Which agents to compare. Default: all registered.
    """
    data:   pDF 
    agents: tuple|list = tuple(AGENTS)

@Dataclass.base
class ComparatorResult:
    """
    Comparator output — the model selection decision package.

    Attributes
    ----------
    comparison : pDF
        Standardized (normalized) scores per metric per agent,
        with direction indicators. The "raw" view.
    scoring : pDF
        Weighted composite scores with the final ranking.
        The "decision" view.
    lift : dict
        Per-agent lift table (probability bins → observed rates).
    calibration : dict
        Per-agent calibration assessment (predicted vs observed
        by bin, with significance flags).
    """    
    comparison:  pDF
    scoring:     pDF
    lift:        dict
    calibration: dict

@Dataclass.base
class ComparatorStats:
    """
    Full diagnostic output for detailed inspection.

    Attributes
    ----------
    crosstab : dict
        Cross-agent transition matrix (probability bin × agent).
    significant : dict
        Statistically significant differences between agents
        per probability bin.
    divergance : dict
        Per-pair divergence metrics.
    diverge_sum : dict
        Aggregated divergence summaries per agent.
    performance : dict
        Per-agent raw performance metrics (Gini, IV, KS, etc.).
    summary : dict
        Per-dimension scores (pre-standardization).
    fpr_tpr : dict
        Per-agent FPR/TPR at the target positive rate.
    lift : dict
        Per-agent lift tables.
    calibration : dict
        Per-agent calibration data.
    comparison : pDF
        Normalized score matrix.
    scoring : pDF
        Weighted composite score matrix.
    ordered : list
        Agent ranking (best first).
    """    
    crosstab:    dict
    significant: dict
    divergance:  dict
    diverge_sum: dict
    performance: dict
    summary:     dict
    fpr_tpr:     dict
    lift:        dict
    calibration: dict
    comparison:  pDF
    scoring:     pDF
    ordered:     list
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PredictorStats:
    """
    Prediction and stats collector for the comparator.

    Obtains probability predictions from all requested agents and
    computes the per-agent statistics needed for the multi-dimensional
    scoring (performance, feature importance, operational metrics).

    Parameters
    ----------
    predictors : PredictorAdapter
        Adapter holding trained agents.
    preprocess : bool, default=True
        Apply the full encoding pipeline before prediction.
    """
    def __init__(self, predictors: PredictorAdapter, preprocess:bool=True):
        self.predictors = predictors
        self.preprocess = preprocess
                    
    def predict(self, data:pDF, agents:tuple[int, ...]) -> dict[int, pSe]:
        """
        Obtain probability predictions from all requested agents.

        Parameters
        ----------
        data : pDF
            Evaluation data.
        agents : tuple
            Agent keys to predict with.

        Returns
        -------
        dict[int, pSe]
            {agent_key: probability_series}
        """
        pred = {}
        for a in agents:
            pa = self.predictors.get_agent(a)
            pred[a] = pa.predict(data, preprocess=self.preprocess).squeeze()
        return pred
    
    @staticmethod
    def proba_array(pred:pDF|pSe) -> pSe:
        """Ensure prediction output is a 1-D Series."""
        if isinstance(pred, pDF):
            return pred.squeeze()
        return pred

    def get_stats(self, 
        target: pSe, 
        pred:   dict[int, pSe], 
        agents: tuple[int, ...]
    ) -> dict[int, dict]:
        """
        Compute per-agent performance and operational statistics.

        For each agent:
        - Separation metrics (Gini, IV, KS).
        - Feature importance (permutation-based).
        - Positive rate (for context).

        Parameters
        ----------
        target : pSe
            True target values.
        pred : dict
            {agent_key: probability_series}.
        agents : tuple
            Agent keys to compute stats for.

        Returns
        -------
        dict[int, dict]
            {agent_key: {metric_key: value}}
        """
        stats  = {}
        for a in AGENTS:
            if a not in agents:
                continue
            
            model = self.predictors.get_agent(a)

            fes = model.cols.selected

            imp: FeatureImportanceResult = model.report.feature_importance
            imp = imp.method_results[FI_M.P].set_index(FI_L.F)[FI_L.P]
            
            diag = SeparationDiagnostics().evaluate(target, self.proba_array(pred[a]))

            stats[a] = {
                STAT.GT: diag.gini, 
                STAT.IV: diag.IV,
                STAT.KS: diag.KS,
                STAT.FE: imp, 
                STAT.FS: fes,
                STAT.PR: diag.PR
            }

        return stats  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Scoring:
    """
    Multi-Dimensional Agent Scoring and Ranking.

    Combines performance, operational risk, calibration, and
    cross-audit metrics into a standardized, weighted composite
    score that ranks agents from best to worst.

    Scoring Dimensions
    ------------------
    1. **Performance (PE)**: Predictive power + operational simplicity.
       - Gini, IV, KS (higher = better separation).
       - Feature Exposure (lower = simpler, more stable).
       - Max Feature Impact (lower = less single-point-of-failure).

    2. **Calibration (CA)**: Probability reliability.
       - Fraction of bins where predictions are significantly
         calibrated (higher = more reliable).

    3. **Cross-Audit (AE)**: Inter-agent disagreement analysis.
       - Audited weakness (lower = fewer blind spots).
       - Auditor strength (higher = more pattern discovery).

    Standardization
    ---------------
    Leader-Anchored Proportional Scaling: the best agent receives
    a score of 1.0 on each metric; all others are scaled relative
    to the leader. This produces a comparable 0-1 scale across
    all metrics regardless of their natural units.

    Weighting
    ---------
    The final composite score is the weighted mean of all standardized
    metrics. Default: equal weights. Custom weights allow the user
    to prioritize certain dimensions (e.g., stability over raw
    performance).
    """
    # active metrics with setup of higher_is_better (True/False)
    METRIC = {
        RANK.PE : {            
            STAT.GT: True,  # positive scale (how far from optim agent is)
            STAT.IV: True,  # positive scale (how far from optim agent is)
            STAT.KS: True,  # positive scale (how far from optim agent is)
            STAT.FE: False, # positive scale (number of effective fes)
            STAT.FX: True,  # negative scale (negative impact to performance)
        },
        RANK.CA: None,      # direct scale with meaning 
        RANK.AE: {
            LBL.U: False,   # positive scale (how much risk agent did not see)
            LBL.O: True     # negative scale (how much agent overestimated risk)
        },
        # Turned off as it captures different type of info than the rest
        # Agent as an auditor provides info how agent sees risk differently
        # However, it is not about agents production quality and target understanding
        # RANK.AO: {
        #     LBL.S: True,    # sum of abs. vals U+O
        # }
    }

    @staticmethod
    def _ops_exposure(gini_permutation: pSe) -> float:
        """
        Compute effective feature exposure (effective degrees of freedom).

        Uses the exponential of Shannon entropy over the normalized
        permutation importance distribution. This produces a "fair
        count" of how many features effectively contribute to the
        model's predictive power.

        Interpretation:
        - 1.0 → all importance concentrated in one feature.
        - N   → importance evenly spread across N features.

        Lower = simpler model = better for production stability.

        Parameters
        ----------
        gini_permutation : pSe
            Per-feature permutation Gini drop (absolute values).

        Returns
        -------
        float
            Effective number of contributing features.
        """
        p = np.abs(gini_permutation)
        p = p / np.sum(p)
        entropy = -np.sum(p * np.log(p + 1e-12))
        return np.exp(entropy)
    
    @staticmethod
    def _ops_max(gini_permutation: pSe) -> int:
        """
        The single strongest feature's Gini impact.

        Represents the maximum single-point-of-failure risk: if this
        feature degrades in production, the model loses this much
        predictive power.

        Interpretation:
        - Close to 0 → no single feature dominates (robust).
        - Large → one feature carries most of the signal (fragile).

        Parameters
        ----------
        gini_permutation : pSe
            Per-feature permutation Gini drop.

        Returns
        -------
        int
            The minimum (most negative) permutation Gini value.
        """
        return gini_permutation.min()
    
    @classmethod
    def performance(cls, fe_stat: dict[int, list]) -> pDF:
        """
        Compute the Performance dimension scores.

        Combines:
        - Predictive metrics (Gini, IV, KS): higher = better.
        - Operational metrics (Exposure, Max Impact): lower = better.

        Parameters
        ----------
        fe_stat : dict
            {agent_key: {metric_key: value}} from `PredictorStats.get_stats`.

        Returns
        -------
        pDF
            Performance scores per agent (rows = agents, cols = metrics).
        """       
        fe = {}
        for a, x in fe_stat.items():
            fe[a] = {
                STAT.GT: x[STAT.GT],
                STAT.IV: x[STAT.IV],
                STAT.KS: x[STAT.KS],
                STAT.FE:  cls._ops_exposure(x[STAT.FE]),
                STAT.FX:  cls._ops_max(x[STAT.FE]),
            }
        return pDF(fe).T
    
    @staticmethod
    def calibration(calibration: dict[int, pDF]) -> pSe:
        """
        Compute the Calibration dimension score.

        Measures the fraction of probability bins where the agent's
        predictions are statistically calibrated (observed rate within
        the confidence interval of the predicted rate).

        Interpretation:
        - 1.0 → perfectly calibrated (all bins significant match).
        - 0.0 → no bins are calibrated (systematic miscalibration).

        Parameters
        ----------
        calibration : dict
            {agent_key: calibration DataFrame with significance flags}.

        Returns
        -------
        pSe
            {agent_key: calibrated_fraction}
        """
        cal = {}
        for a, x in calibration.items():
            cal[a] = (~x[LBL_C.V]).mean()
        return pSe(cal)

    @staticmethod
    def audit(divergance_sum: dict[int, dict[str, pDF]]) -> dict[str, pDF]:
        """
        Compute the audit dimension scores per agent.

        - **Audited** (weakness): Mean blind-spot score — how much
          risk this agent missed, as detected by other agents finding
          significant patterns within its bins.

        - **Auditor** (strength): Mean informational contribution —
          how much new information this agent reveals about other
          agents' predictions.

        Parameters
        ----------
        divergence_sum : dict[int, dict[str, pDF]]
            Per-agent aggregated divergence (U, O, S components).

        Returns
        -------
        dict[str, pDF]
            {RANK.AE: audited_scores, RANK.AO: auditor_scores}
        """
        audited = {}
        auditor = {}
        for a, x in divergance_sum.items():
            audited[a] = x[LBL_T.AE].mean()
            auditor[a] = x[LBL_T.AO].mean()

        return {RANK.AE: pDF(audited).T, RANK.AO: pDF(auditor).T}

    @classmethod
    def process(cls,
        stats:   dict[int, dict],
        calib:   dict[int, pDF],
        div_sum: dict[int, dict[str, pDF]], 
        alpha:   float
    ) -> dict[str, pSe|pDF]:
        """
        Compute the raw (unstandardized) scores across all dimensions.

        Parameters
        ----------
        stats : dict[int, dict]
            Per-agent performance stats.
        calib : dict[int, pDF]
            Per-agent calibration tables.
        div_sum : dict[int, dict[str, pDF]]
            Per-agent divergence summaries.
        alpha : float
            Significance level used.

        Returns
        -------
        dict
            {dimension: scores_per_agent}
        """
        summary = {}
        audit = cls.audit(div_sum)
        summary[RANK.PE] = cls.performance(stats)
        summary[RANK.CA] = cls.calibration(calib)
        summary[RANK.AE] = audit[RANK.AE]
        summary[RANK.AO] = audit[RANK.AO]
        summary[LBL.A]   = alpha
        return summary

    @staticmethod
    def _standardize(scores:pSe, higher_is_better:bool|None=True) -> pSe:
        """
        Leader-Anchored Proportional Scaling.

        Normalizes scores so the best agent = 1.0 and others are
        proportional. Handles direction (higher/lower is better)
        and edge cases (all-negative, zeros).

        Parameters
        ----------
        scores : pSe
            Raw scores per agent.
        higher_is_better : bool | None
            Direction. None = natural scale (no transformation).

        Returns
        -------
        pSe
            Standardized scores (best = 1.0, others in (0, 1]).
        """
        
        # stats with natural scale and meaning 
        if higher_is_better == None:
            return scores

        # metric on negative scale
        n = (scores < 0).sum()
        if n:
            if n == len(scores):
                scores = scores.abs()
                higher_is_better = not higher_is_better
            else:
                scores = scores - scores.min()
        
        scores = scores.where(scores!=0, 1e-9)

        if higher_is_better:
             scores = 1 / scores

        return scores.min() / scores    

    @staticmethod
    def _data_map(scores: dict) -> pDF:
        """Map metric keys to canonical column names."""
        m = MAP.COL
        c = {k: m[k] for k in scores if k in m}
        return pDF(scores).rename(columns=c).T
    
    @staticmethod
    def _weights(norm:pDF, weights:dict|None) -> pDF:
        """
        Apply metric weights and compute the total weighted score.

        Parameters
        ----------
        norm : pDF
            Standardized scores.
        weights : dict | None
            Per-metric weights (default: all 1.0).

        Returns
        -------
        pDF
            Scores with weight column and total score row.
        """        
        w = np.array([[weights[k]] if k in weights or {} else 1.0 for k in norm.index])        
        norm.loc[:,MAP.W] = w
        norm.loc[MAP.S,:] = (norm*w).sum() / w.sum() 
        norm.loc[MAP.S, MAP.W] = np.nan
        return norm
    
    @staticmethod
    def _order(data:pDF) -> pDF:
        """Order rows by canonical metric order."""
        m = MAP.COL.values()
        o = [c for c in m if c in data.index]
        x = [x for x in data.index if x not in o]
        return data.loc[o+x]
            
    @classmethod
    def standardize(cls, summary: dict[str, pSe|pDF], weights:dict=None) -> tuple[pDF, pDF]:
        """
        Compute the final standardized scoring table.

        Applies leader-anchored proportional scaling to each metric,
        applies weights, and produces:
        - `score`: Standardized table (best = 1.0 per metric).
        - `overview`: Original values + direction indicators.

        Parameters
        ----------
        summary : dict[str, pSe|pDF]
            Raw scores from `Scoring.process`.
        weights : dict | None
            Per-metric weights.

        Returns
        -------
        tuple[pDF, pDF]
            (scoring_table, overview_table)
        """
        def add(metric:str, scores:pSe, direction:bool|None) -> None:
            orig[metric] = scores
            norm[metric] = cls._standardize(scores, direction)
            
            d = direction if isinstance(direction, bool) else True
            direct.append(d)

        orig, norm, direct = {}, {}, []

        for metric, direction in cls.METRIC.items():
            scores = summary[metric]
            if not isinstance(direction, dict):
                add(metric, scores, direction)
                continue
            for m, d in direction.items():
                add(m, scores[m], d)

        d = cls._data_map(norm)
        d = cls._weights(d, weights)

        o = cls._data_map(orig)
        o.loc[MAP.S,:] = d.loc[MAP.S,:]
        o.loc[:,MAP.D] = direct + [True]

        return cls._order(d), cls._order(o)

    def get_ordered_agents(score: pDF) -> list:
        """
        Extract the agent ranking from the scoring table.

        Parameters
        ----------
        score : pDF
            The standardized scoring table.

        Returns
        -------
        list
            Agent keys ranked by total score (best first).
        """            
        r = score.loc[MAP.S]
        a = [a for a in AGENTS if a in r.index]
        return r[a].sort_values(ascending=False).index.to_list()    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Fitter(FitterDataBase, SelectorInputBase[ComparatorInput]):
    """
    Comparison Worker — full multi-agent evaluation pipeline.

    Executes the complete comparison process:
    1. Predict with all agents.
    2. Binarize predictions into quantile bins.
    3. Compute per-agent performance stats.
    4. Run calibration assessment.
    5. Run cross-agent transition (crosstab + divergence).
    6. Compute audit scores (audited + auditor).
    7. Standardize and score all dimensions.
    8. Rank agents by total score.

    All results are stored in `self.stats` (`ComparatorStats`).

    Parameters
    ----------
    params : ComparatorParams
        Comparison configuration.
    """    
    SelectionInput:  type = ComparatorInput
    agents_condition: int = 2

    stats: ComparatorStats = None
    
    def __init__(self, params: ComparatorParams):
        self.prm = params
        self.predictor = PredictorStats(self.prm.predictors, self.prm.preprocess)

    def get_weights(self) -> dict:
        """Merge default weights with user-specified overrides."""
        default = MAP.WEIGHT
        return default | self.prm.metric_weight
    # ─────────────────────────────────────────────────────────────────────────────────────────────    

    def predict(self, data:pDF) -> dict[int, pSe]:
        """
        Obtain probability predictions from all requested agents.

        Parameters
        ----------
        data : pDF
            Evaluation data.

        Returns
        -------
        dict[int, pSe]
            {agent_key: probability_series}
        """
        return self.predictor.predict(data, self.agents)  
        
    def _bins(self, a:int, pred:pSe) -> tuple[pSe, dict]:
        """
        Discretize one agent's probabilities into quantile bins.

        Parameters
        ----------
        a : int
            Agent key (for naming).
        pred : pSe
            Probability series.

        Returns
        -------
        tuple
            (binned_series, bin_category_mapping)
        """        
        b = Pd.qcut(pred, self.prm.proba_bins, min_left=0, max_right=1, duplicates='drop')
        m = {q:i for i,q in enumerate(b.cat.categories)}
        b = b.cat.rename_categories(m)
        b.name = LBL.N(a)
        return b, m

    def binning(self, pred:dict[int, pSe]) -> tuple[dict[int, pSe], dict[int, dict]]:
        """
        Discretize all agents' probabilities into quantile bins.

        Parameters
        ----------
        pred : dict[int, pSe]
            Per-agent probability series.

        Returns
        -------
        tuple
            (binned_predictions, bin_mappings)
        """
        bins, mapping = {}, {}
        for a in self.agents:
            pa = self.predictor.proba_array(pred[a])
            bins[a], mapping[a] = self._bins(a, pa)
        return bins, mapping
              
    def model_stats(self, target:pSe, pred:dict) -> tuple[dict[int, dict]]:
        """
        Compute per-agent performance metrics for scoring.

        Parameters
        ----------
        target : pSe
            True target values.
        pred : dict[int, pSe]
            Per-agent predictions.

        Returns
        -------
        tuple
            (stats_per_agent, positive_rates)
        """
        stats = self.predictor.get_stats(target, pred, self.agents)
        prs = {}
        for a in self.agents:
            prs[a] = stats[a].pop(STAT.PR, None)
        return stats, prs    
    
    @staticmethod
    def _calib(calib:dict) -> dict:
        """Returns agents calibration assessment data only."""
        c = {**calib}
        c.pop(LBL_C.A, None)
        return c
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    def process(self, inputs: ComparatorInput | pDF, **kwargs) -> None:        
        """
        Execute the full multi-agent comparison pipeline.

        Steps:
        1. Predict with all agents.
        2. Binarize into quantile bins.
        3. Compute performance stats.
        4. Run calibration assessment.
        5. Run cross-agent transition (crosstab + significance).
        6. Compute divergence and audit scores.
        7. Standardize and score.
        8. Rank agents.

        All results stored in `self.stats`.

        Parameters
        ----------
        inputs : ComparatorInput | pDF
            Evaluation data or input specification.
        kwargs : dict
            Additional arguments.
        """
        inputs_ = self.get_inputs(inputs, kwargs)
        target  = inputs_.data[self.prm.target]
        
        pred = self.predict(inputs_.data)
        bins,  mapping = self.binning(pred)
        stats, fpr_tpr = self.model_stats(target, pred)

        cal   = CalibrationAssessment(self.prm.alpha, self.prm.proba_bins, kind='uniform')
        calib = cal.process(target, pred)

        cc = ModelsCrossTransition(self.prm.alpha)
        x_tab = cc.process(target, bins)
        diff  = cc.get_sig_diff(x_tab)
        total = cc.get_totals(x_tab, mapping)

        div   = ModelCrossDivergence.process(diff, x_tab)
        div_s = ModelCrossDivergence.get_summary(div)

        summary = Scoring.process(stats, self._calib(calib), div_s, self.prm.alpha)
        score, overview = Scoring.standardize(summary, self.get_weights())
        ordered = Scoring.get_ordered_agents(score)

        self.stats = ComparatorStats(
            crosstab    = x_tab,
            significant = diff,
            divergance  = div,
            diverge_sum = div_s,
            performance = stats,
            summary     = summary,
            fpr_tpr     = fpr_tpr,
            lift        = total,
            calibration = calib,        
            comparison  = overview,
            scoring     = score,
            ordered     = ordered
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class ComparatorProcessor(FitterModelBase[Fitter]):
    """
    Multi-Agent Comparator — Public API for Model Selection Decision.

    Provides an objective, multi-dimensional comparison of all trained
    agents to determine which model to deploy. Evaluates across
    performance, calibration, and cross-agent audit dimensions, then
    produces a ranked list with the recommended agent first.

    Parameters
    ----------
    alpha : float, default=0.05
        Statistical significance level for divergence testing.
    proba_bins : int, default=10
        Number of quantile bins for probability discretization.
    metric_weight : dict, default=None
        Per-metric weights for the final score. None = equal weights.
        Override to emphasize specific dimensions (e.g., weight
        calibration higher if probability reliability is critical).
    predictors : PredictorAdapter
        Adapter holding all trained agents.
    preprocess : bool | None, default=None
        If True, apply the full encoding pipeline to input data.
    target : str
        Target column name in the evaluation data.

    Output Interpretation
    ---------------------
    - **`scoring`**: The standardized score table. The row with
      the highest `TOTAL_WEIGHTED_SCORE` is the recommended agent.
    - **`comparison`**: The original (unstandardized) metric values
      — useful for understanding *why* an agent scored where it did.
    - **`ordered`** (in stats): The ranked agent list (best first).
    - **`calibration`** (in result): Per-bin predicted vs. observed
      rates — check for systematic miscalibration.
    - **`lift`** (in result): Per-bin lift — check for separation
      quality across the probability range.

    Notes
    -----
    **Requires ≥ 2 agents**: The cross-agent audit dimensions
    (audited/auditor) require at least two agents to compare.
    With a single agent, only performance and calibration are scored.

    **Reproducibility**: The same data + agents + parameters will
    always produce the same ranking. No randomness in the scoring.

    **Weighted Score**: By default, all metrics are weighted equally.
    Override `metric_weight` to prioritize specific dimensions:
    e.g., `{'eval_model_calibration': 2.0}` doubles the weight
    of calibration in the final score.

    Examples
    --------
    >>> from aidflow.fitters import ComparatorProcessor
    >>> cmp = ComparatorProcessor(
    ...     predictors=my_adapter,
    ...     target='is_churn',
    ...     alpha=0.05,
    ...     proba_bins=10,
    ... )
    >>> result = cmp.process(oot_data)
    >>> result.scoring  # standardized scores per agent
    >>> result.comparison  # original metric values
    >>> cmp.fitter.stats.ordered  # ['2', '1', '3'] (agent 2 recommended)
    """
    
    def __init__(self, *args, params: dict|ComparatorParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(ComparatorParams, params, args, kwargs)
        self._fitter = Fitter(self._params)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def process(self, inputs: ComparatorInput | pDF, **kwargs) -> ComparatorResult:
        """
        Execute the full comparison and return the decision tables.

        Parameters
        ----------
        inputs : ComparatorInput | pDF
            OOT evaluation data (must contain the target column).
        kwargs : dict
            Additional arguments (e.g., `agents=(1, 2)`).

        Returns
        -------
        ComparatorResult
            Scoring table, comparison table, lift, and calibration.
        """
        self._fitter.process(inputs, **kwargs)
        stats: ComparatorStats = self.stats
        return ComparatorResult(
            comparison  = stats.comparison,
            scoring     = stats.scoring,
            lift        = stats.lift,
            calibration = stats.calibration
        )  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
  
class ScoringStatsBase:

    def __init__(self, stats: ComparatorStats):
        if stats is None or not isinstance(stats, ComparatorStats):
            raise ValueError('Populated ComparatorStats instance required prior calling plots.')

        self.stats = stats
        self.agents = self._get_agents() 
        self._name  = lambda x: LBL.N(x)
        
        scoring:pSe  = self._get_scoring().loc[MAP.S, self.agents]
        self.rank    = scoring.rank(method='max', ascending=False)

    def _get_agents(self) -> tuple:
        return tuple(self.stats.crosstab)
    
    def _get_overview(self) -> pDF:
        return self.stats.comparison
    
    def _get_scoring(self) -> pDF:
        return self.stats.scoring
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class ScoringTheme(ScoringStatsBase):
    
    def __init__(self, stats: ComparatorStats, theme:int, is_idx:bool=False):
        super().__init__(stats)
        self.theme = theme
        self._is_idx = is_idx

    def _mapping(self, a:int) -> int:
        if not self._is_idx:
            return a 
        return list(AGENTS)[a]

    def _opacity(self, agent:int, opacity:float):        
        if opacity is not None:
            return lambda x: x
        
        sco = self._get_scoring()
        return lambda x, a=agent: round(sco.loc[x,a],2)

    def get_color(self, a:int, opacity:float=None) -> str | Callable:
        a_ = self._mapping(a)
        rank = self.rank[a_]
        op = self._opacity(a_, opacity)

        # High-Contrast Tech
        if self.theme == 0: 
            c = {
            1: lambda x: f'rgba(  0, 210, 150, {op(x)})', # Electric Mint
            2: lambda x: f'rgba(255, 105,  95, {op(x)})', # Hi-Fi Coral
            3: lambda x: f'rgba( 85,  80, 160, {op(x)})', # Cyber Indigo
            4: lambda x: f'rgba(110, 120, 130, {op(x)})'  # Charcoal Slate            
        }[rank]

        # Cyberpunk
        if self.theme == 1: 
            c = {
            1: lambda x: f'rgba(  5, 175, 120, {op(x)})', # Laser Emerald
            2: lambda x: f'rgba(165,  75, 155, {op(x)})', # Electric Orchid
            3: lambda x: f'rgba( 40, 140, 170, {op(x)})', # Baltic Cyan
            4: lambda x: f'rgba( 85,  85, 100, {op(x)})'  # Dark Obsidian                
        }[rank]
        
        # Quantum
        if self.theme == 2: 
            c = {
            1: lambda x: f'rgba( 15,  90, 185, {op(x)})', # Cobalt Blue
            2: lambda x: f'rgba(190,  55,  90, {op(x)})', # Crimson Rose
            3: lambda x: f'rgba( 95, 125,  65, {op(x)})', # Olive Forest
            4: lambda x: f'rgba(185, 190, 195, {op(x)})'  # Pale Platinum                
        }[rank]
        
        # Warm
        if self.theme == 3: 
            c = {
            1: lambda x: f'rgba( 35, 120, 100, {op(x)})', # Imperial Jade
            2: lambda x: f'rgba(215, 140,  45, {op(x)})', # Spiced Amber
            3: lambda x: f'rgba(165,  80,  70, {op(x)})', # Venetian Red
            4: lambda x: f'rgba(160, 155, 150, {op(x)})'  # Sandstone Gray                
        }[rank]
        
        if self.theme == 4: 
            c = {
            1: lambda x: f'rgba(210, 105,  55, {op(x)})', # Terracotta Copper
            2: lambda x: f'rgba( 30,  65, 110, {op(x)})', # Deep Ocean Navy
            3: lambda x: f'rgba( 90, 145, 170, {op(x)})', # Sea Glass Blue
            4: lambda x: f'rgba(170, 175, 180, {op(x)})'  # Soft Pebble                
        }[rank]
            
        # Tech
        if self.theme == 5: 
            c = {
            1: lambda x: f'rgba(33,  115, 115, {op(x)})', # Deep Teal
            2: lambda x: f'rgba(70,  120, 150, {op(x)})', # Steel Blue
            3: lambda x: f'rgba(120, 135, 145, {op(x)})', # Slate Gray
            4: lambda x: f'rgba(175, 185, 190, {op(x)})', # Light Ash
        }[rank]

        # Muted
        if self.theme == 6: 
            c = {
            1: lambda x: f'rgba(195, 125,  60, {op(x)})', # Burnished Amber
            2: lambda x: f'rgba( 70,  90, 130, {op(x)})', # Deep Indigo
            3: lambda x: f'rgba(105, 120, 150, {op(x)})', # Twilight Blue
            4: lambda x: f'rgba(140, 145, 160, {op(x)})', # Muted Ink
        }[rank]

        # Monochromatic
        if self.theme == 7: 
            c = {
            1: lambda x: f'rgba( 25,  50,  90, {op(x)})', # Midnight Navy
            2: lambda x: f'rgba( 60,  95, 140, {op(x)})', # Royal Slate
            3: lambda x: f'rgba(110, 140, 180, {op(x)})', # Cadet Slate
            4: lambda x: f'rgba(180, 195, 215, {op(x)})', # Ice Glaze
        }[rank]

        # Emerald & Earth Academic
        if self.theme == 8: 
            c = {
            1: lambda x: f'rgba( 45, 110,  85, {op(x)})', # Racing Green
            2: lambda x: f'rgba(115, 135, 105, {op(x)})', # Olive Sage
            3: lambda x: f'rgba(150, 135, 120, {op(x)})', # Warm Taupe
            4: lambda x: f'rgba(175, 175, 170, {op(x)})'  # Driftwood Gray
        }[rank]

        if opacity is not None: 
            return c(opacity)
        return c
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class GiniLiftPlot(ScoringStatsBase):
    
    def __init__(self, stats: ComparatorStats, theme:int = 1):
        super().__init__(stats)
        vis = self.rank.sort_values().index[0]
        
        self._glp = GLP(
            eval_stats    = self._eval_stats(stats), 
            visible_agent = self._name(vis), 
            theme         = ScoringTheme(stats, theme, is_idx=True)
        )
    
    def _eval_stats(self, stats: ComparatorStats) -> dict:
        es = {}
        for a in stats.performance:
            p = stats.performance[a]
            f = stats.fpr_tpr[a]

            l = stats.lift[a][[LBL_T.T, LBL_T.C, LBL_T.W, LBL_T.L]]
            l.columns = [LBL_M.T, LBL_M.C, LBL_M.W, LBL_M.L]

            es[self._name(a)] = DiagnosticsResult(                
                obs     = int(l[LBL_M.C].sum()),
                events  = int((l[LBL_M.T] * l[LBL_M.C]).sum()),
                gini    = p[STAT.GT],
                KS      = p[STAT.KS],
                IV      = p[STAT.IV],
                PR      = f,
                lift    = l,
            )
        return es

    def gini(self, is_grid:bool=False) -> go.Figure:
        return self._glp.gini(is_grid)
    
    def lift(self, is_grid:bool=False) -> go.Figure:
        return self._glp.lift(is_grid)

    def create(self, show:bool = True) -> go.Figure | None:
        return self._glp.create(show)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ScoringSankey(ScoringStatsBase):

    def __init__(self, 
        stats: ComparatorStats, 
        theme:  int = 2, 
        header_color: str = 'rgba(70, 80, 90, 1.0)'
    ):
        super().__init__(stats)
        self._theme = ScoringTheme(stats, theme)
        self._col_score = self.add_col_score()
        self._header = header_color

    def add_col_score(self) -> pDF:
        sco = self._get_scoring()
        sc_ = sco.loc[:, self.agents]
        sc_ = sc_.clip(1e-6).copy() # clip 0 to avoid sankey colapsing
        sc_[0] = 1                  # [0] = 1 → the header row
        return sc_[[0] + list(self.agents)]

    def labels(self) -> list:
        org  = self._get_overview()
        cols = self._col_score.index

        col = lambda l: '<br>'.join([x.capitalize() for x in l.split('_')])
        lbl = lambda a, l: (
            f'{f"<b>A{a} → </b>" if l == cols[0] else ''}'
            f"{'<b>' if l == cols[-1] else ''}"
            f'{org.loc[l,a]:.3f}</b>'
        )
        return [
            col(l) if a==0 else lbl(a, l) 
            for l in cols for a in self._col_score.columns
        ]
    
    def hover(self) -> list:
        d = self._get_overview().loc[:, MAP.D]
        o = self._get_overview().loc[:, self.agents]
        s = self._get_scoring().loc[:, self.agents]

        mm  = lambda x,l: x.loc[l].max() if d.loc[l] else x.loc[l].min()
        col = lambda l: [
            s.columns[s.loc[l] == s.loc[l].max()], 
            mm(o,l), s.loc[l].max(), l
        ]
        
        lbl = lambda a,l: [a, o.loc[l,a], s.loc[l,a], l]

        return [ 
            col(l) if a==0 else lbl(a, l)
            for l in self._col_score.index for a in self._col_score.columns 
        ]
        
    def values(self) -> ArrayLike:
        return self._col_score.to_numpy().flatten()
    
    def chain(self) -> tuple[ArrayLike, ArrayLike]:
        mx = int( len(self._col_score.columns) * len(self._col_score.index) )
        source = np.array(range(mx))
        target = source + len(self._col_score.columns)
        return source, target
    
    def _colors(self) -> list:
        colors = [self._theme.get_color(a) for a in self.agents]
        colors = [[self._header] + [c(x) for c in colors]  for x in self._get_scoring().index]
        return [c for x in colors for c in x]
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class ScoringPlot(ScoringStatsBase):
    
    def __init__(self, stats: ComparatorStats, theme:int = 1):
        super().__init__(stats)
        self._sankey = ScoringSankey(stats, theme)
        self._theme  = ScoringTheme(stats, theme)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def radar(self, show:bool=True) -> go.Figure | None:        
        agents = self._get_agents()
        scores = self._get_scoring()
        values = self._get_overview().loc[scores.index]

        fig = go.Figure()        
        vis = self.rank.sort_values().index

        col = lambda l: ' '.join([x.capitalize() for x in l.split('_')[1:]])
        col = [col(x) for x in scores.index]
        col[-1] = '<i>'+col[-1]+'</i>'

        for a in agents:    
            fig.add_trace(go.Scatterpolar(
                r = scores[a].round(4),
                theta   = col,
                fill    = 'toself',
                name    = self._name(a),
                hoveron = 'points',
                line    = dict(
                    color = self._theme.get_color(a, 1.0),
                    width = 2 
                ),
                fillcolor = self._theme.get_color(a, 0.3),
                customdata    = values[a],
                hovertemplate = "Score: %{r:.1%}, <br>Value: %{customdata:.4f}, <br>Metric: %{theta}",
                visible = True if a in [vis[0], vis[-1]] else 'legendonly'
        ))

        fig.update_layout(
            height = 500,
            legend = dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin = dict(t=20,r=0,b=0,l=0),
            polar  = dict(        
                angularaxis=dict(
                    categoryarray = col,
                    direction = 'clockwise'
            ))
        )
        if not show:
            return fig
        fig.show()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
       
    def sankey(self, show:bool=True) -> go.Figure | None:        
        source, target = self._sankey.chain()

        fig = go.Figure(go.Sankey(  
            arrangement = "fixed",  

            node = dict(
                pad = 10,
                thickness = 10,
                line = dict(color = "rgba(0,0,0,0.1)", width = 0.5),
                label = self._sankey.labels(),
                color = "gainsboro",
                hovertemplate = "%{label}<extra></extra>"
            ),    

            link = dict(
                source = source,
                target = target,
                value  = self._sankey.values(),
                color  = self._sankey._colors(),
                customdata = self._sankey.hover(),
                hovertemplate = (
                    "<b>Agent: %{customdata[0]}</b><br>" +            
                    "Raw Value: <b>%{customdata[1]:.3f}</b><br>" +
                    "Standardized Score: %{customdata[2]:.1%}<br>"+
                    "Metric Focus: %{customdata[3]}<extra></extra>"
                ),
                hoverlabel = dict(
                bgcolor = "rgba(0, 0, 0, 1.0)",
                font_size = 12,
                font_family = "Arial"
            ))
        ))

        fig.update_layout(
            font_size = 11.5,
            font_color = 'white',
            height = 400, 
            width  = 1050,
            margin = dict(t=0,r=0,b=0,l=0)
        )

        if not show:
            return fig
        fig.show()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class ScoringTable(ScoringStatsBase):
    
    _A = 'Agent'
    _G = 'Group'
    _D = 'Direction'
    _W = 'Weight'
    _M = 'Metric'

    _C = ['IV', 'KS']

    @classmethod
    def _capital(cls, x:str) -> str:
        if x.upper() in cls._C:
            return x.upper()
        return x.capitalize()
    
    def _metrics(self) -> pDF:
        index   = self._get_overview().index.to_list()
        index_  = [[self._capital(x) for x in m.split('_')] for m in index]
        groups  = [x[0] for x in index_]
        metrics = [' '.join(x[1:]) for x in index_]
        d = self._get_overview()[MAP.D]
        w = self._get_scoring()[MAP.W]
        return pDF({
            self._G: groups, 
            self._W: w,
            self._D: ['Max' if x else 'Min' for x in d.values],
            self._M: metrics, 
        }, index = index)
    
    @staticmethod
    def _style_total() -> dict:
        return {
            'subset': (MAP.S, slice(None)), 
            'font-weight': '600', 
            'background-color': "#d5daefdf"
        }
    
    def _style_values(self, d:pDF, is_score:bool=False) -> pDF:
        agents = self.agents
        col = lambda a: f'{self._A} {a}'

        d_ = d.loc[:, agents]
        d_.columns = [col(a) for a in agents]
        fc = '{:.1%}' if is_score else '{:.4f}'

        return Pd.style(
            d_, hide_axis = 0, 
            format_cols = {col(a): fc for a in agents},
            properties  = self._style_total()
        )
    
    def _style_metrics(self) -> pDF:
        d = self._metrics()
        return Pd.style(
            d, hide_axis = 0, 
            bold_cols   = self._M,
            format_cols = {self._W: '{:.2f}'},
            properties  = self._style_total()
        )

    def create(self, show:bool=True):
        d = [self._style_metrics()]
        d = d + [self._style_values(d,i) for i,d in enumerate(
            [self._get_overview(), self._get_scoring()]
        )]
        t = TableStyle.combine_pdf_to_html(
            d, ['Metrics', 'Values', 'Scoring'], 1, h_align='center', show=False
        )
        if not show:
            return t
        Display.construct(t)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ComparatorPlot:
    
    def __init__(self, 
        stats: ComparatorStats, 
        theme: int = 1,
        dst_kind: Literal[1,2] = 2, 
        bar_kind: Literal[1,2] = 2
    ):        
        self.stats = stats
        self.theme = theme
        self.dst_kind = dst_kind
        self.bar_kind = bar_kind
        
    def gini(self, show:bool=True) -> go.Figure | None:
        return GiniLiftPlot(self.stats, theme = self.theme).create(show)
    
    def radar(self, show:bool=True) -> go.Figure | None:
        return ScoringPlot(self.stats, theme = self.theme).radar(show)

    def sankey(self, show:bool=True) -> go.Figure | None:
        return ScoringPlot(self.stats, theme = self.theme).sankey(show)
    
    def calibration(self, show:bool=True) -> go.Figure | None:
        return CalibrationPlot(
            self.stats.calibration, 
            dst_kind = self.dst_kind, 
            bar_kind = self.bar_kind,
            graph_titles = {a: LBL.N(a) for a in self.stats.calibration} 
        ).create(show)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬