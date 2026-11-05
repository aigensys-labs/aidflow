"""
WOE Feature Stability Analysis (Time-Wise Consistency Check).

This module implements the post-WOE stability screening stage of the
pipeline, responsible for verifying that each WOE-encoded feature
maintains consistent predictive behaviour over time.

Runs after WOE transformation. It is the final feature quality gate
before model training. Features that fail the stability check are
excluded from the model, ensuring that only time-consistent signals
are used.

What is Measured?
-----------------
For each WOE feature, the relationship between WOE bins and the target
rate is observed across time units (typically weeks). Stability is
assessed at two levels:

1. **Bin-level consistency**: For each WOE bin, is the target rate
   (event rate) consistent over time? A bin whose event rate fluctuates
   wildly suggests the bin is capturing noise rather than a stable
   signal.

2. **Feature-level stability**: Is the overall predictive power (Gini/
   separation) of the feature stable over time? A feature whose
   discriminative power degrades or flips over time is unreliable.

A feature passes if its overall consistency score exceeds
`stability_cutoff` (default 0.65). This is verified on both the
training and test samples.

Why This Matters
----------------
A WOE feature can have excellent average Gini yet be driven by a few
anomalous time periods. If the feature's signal is not temporally
consistent, it will fail in production when the underlying process
shifts. This gate catches such "unstable" features and prevents them
from entering the model.

Output
------
- Per-feature: stability verdict (pass/fail) and consistency score.
- Aggregate: list of stable features (`vars_woe`) and unstable
  features (`vars_dropped`).
- The `transform` method simply selects the stable WOE columns
  from the input data (no transformation needed — the WOE values
  are already computed in the previous stage).
"""

from ....deps.data                  import  np, pDF
from ....deps.types                 import  Literal

from ....core.base                  import  (
                                            FitterDataBase, 
                                            FitterModelBase,
                                            )
from ....core.factory               import  (
                                            Dataclass,
                                            DictClass
                                            )
from ....core.foundation            import  AigParam
from ....core.accessor              import  Parallel
from ....core.standard              import  VarStd

from ....ml.features.stability      import  (
                                            LBL, 
                                            STAB,
                                            FeConsistency
                                            )
from ....ml.sampling                import  Subsampling
from ....display.verbosity          import  Verbosity


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class TT(DictClass):
    """Sample type identifiers for stability test results."""
    TRAIN = 'TRAIN'
    TEST  = 'TEST'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → WOE → TRANSFORM
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class FitParamsBase:
    """
    Stability thresholds.

    Attributes
    ----------
    stability_cutoff : float, default=0.65
        Minimum feature-level consistency score for a WOE feature
        to be considered time-stable. The score represents the
        proportion of time units where the feature maintains its
        target-separation direction and magnitude. Features below
        this threshold are dropped.

        Higher values are more conservative (fewer features pass).
        Lower values are more permissive (more features pass, but
        with higher risk of temporal instability in production).
    alpha : float, default=0.05
        Statistical significance level for the bin-level consistency
        test (e.g., t-test on target rates across time units).
    """
    stability_cutoff: float = 0.65
    alpha: float = 0.05

@Dataclass.base(update_specs=dict(slots=False))
class FitParamsCfg:
    """
    Pipeline configuration.

    Attributes
    ----------
    target : str
        Target column name.
    time_var : str
        Time column name (e.g., 'AIG_YY_MM'). Used to group
        observations into time units.
    time_unit : 'week' | 'day' | None, default=None
        Granularity of time grouping. If None, the data's native
        time resolution is used. Setting 'week' groups daily data
        into weekly aggregates for more stable per-unit estimates.
    max_time_units : int | None, default=None
        Maximum number of time units to consider (most recent).
        Useful when the history is very long and older periods
        are less representative of the current regime.
    verbose : bool | Verbosity, default=True
        Enable progress display and stats output.
    """
    target:         str
    time_var:       str
    time_unit:      Literal['week','day',None] = None
    max_time_units: int = None
    verbose:        bool|Verbosity = True

@Dataclass.base
class FitParams(FitParamsBase, FitParamsCfg):
    """Fitter params input design."""

@Dataclass.base
class FitResult:
    """
    Stability screening output.

    Attributes
    ----------
    vars_woe : list
        WOE feature names that passed the stability check
        (time-consistent predictive power).
    vars_dropped : list
        WOE feature names that failed the stability check
        (temporally unstable, excluded from model).
    """
    vars_woe:     list
    vars_dropped: list

@Dataclass.base
class FitStats:
    """
    Diagnostic statistics for stability analysis.

    Attributes
    ----------
    consistency : dict
        Per-sample (TRAIN / TEST) feature-level consistency scores
        and bin-level test results. Used for user-facing display
        and for the `update` method (re-evaluate with a new
        threshold without re-running the full test).
    """
    consistency: dict
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class FitWorker:
    """
    WOE Feature Stability Worker — per-feature consistency test.

    Executes the stability test for a set of WOE features on a
    single sample (train or test):
    1. Group observations by time unit.
    2. For each WOE feature, measure bin-level target rate
       consistency across time units.
    3. Compute the overall feature-level stability score.
    4. Compare against `stability_cutoff` to determine pass/fail.

    Parameters
    ----------
    params : FitParams
        Stability thresholds and configuration.
    """

    def __init__(self, params:FitParams):
        self.prm = params

    def run_consistency(self, fes:list=None) -> FeConsistency:
        """
        Instantiate the feature consistency tester.

        Parameters
        ----------
        fes : list | None
            Feature names to test. If None, auto-detect from data.

        Returns
        -------
        FeConsistency
            Configured consistency tester.
        """        
        return FeConsistency( 
            fes, self.prm.target, 
            time_var       = self.prm.time_var, 
            c_test_th      = self.prm.stability_cutoff, 
            t_test_alpha   = self.prm.alpha,
            time_unit      = self.prm.time_unit, 
            max_time_units = self.prm.max_time_units 
        )

    def run_test(self, data:pDF, fes:list, **kwargs) -> dict:
        """
        Run the stability test on a single sample.

        Parameters
        ----------
        data : pDF | None
            Sample data (target + time + WOE features). If None,
            returns None (sample not provided).
        fes : list
            WOE feature names to test.
        kwargs : dict
            Forwarded to the consistency tester.

        Returns
        -------
        dict | None
            Per-feature consistency results, or None.
        """        
        if data is None:
            return None
        fec = self.run_consistency(fes)
        return fec.test(data, verbose = self.prm.verbose, **kwargs)

    def get_fes_by_status(self, stat:dict) -> tuple[list, list]:
        """
        Partition features into stable (in) and unstable (out).

        Parameters
        ----------
        stat : dict
            Per-feature consistency results.

        Returns
        -------
        tuple[list, list]
            (stable_features, unstable_features)
        """        
        fes_in  = [x for x,v in stat.items() if v[LBL.IS]==STAB.Y]
        fes_out = [x for x in stat if x not in fes_in]
        return {'in':fes_in, 'out':fes_out}
    
    def get_result(self, train:dict, test:dict) -> tuple[FitResult, FitStats]:
        """
        Assemble the final stability result from train and test outcomes.

        A feature is kept only if it passes on the training sample
        (the test sample provides additional diagnostic context).

        Parameters
        ----------
        train : dict
            Training sample consistency results.
        test : dict
            Test sample consistency results.

        Returns
        -------
        tuple[FitResult, FitStats]
            Final verdicts and diagnostic stats.
        """        
        vars_io = self.get_fes_by_status(train)
        return (
            FitResult(
                vars_woe     = vars_io['in'],
                vars_dropped = vars_io['out'],
            ),
            FitStats(consistency = {TT.TRAIN: train, TT.TEST: test})
        )

    def process(self, data_woe:pDF, fes:list, data_test_woe:pDF=None, **kwargs):
        """
        Execute the full stability test on train (and optionally test).

        Parameters
        ----------
        data_woe : pDF
            Training data (target + time + WOE features).
        fes : list
            WOE feature names to test.
        data_test_woe : pDF | None
            Out-of-time test data (for OOS stability verification).
        kwargs : dict
            Forwarded to the consistency tester.

        Returns
        -------
        tuple[FitResult, FitStats]
            Stability verdicts and diagnostics.
        """        
        stat_train = self.run_test(data_woe,      fes, **kwargs)
        stat_test  = self.run_test(data_test_woe, fes, **kwargs)
        return self.get_result(stat_train, stat_test)
    
    def update(self, stats:dict|FitStats):
        """
        Re-evaluate stability verdicts with a new threshold.

        Uses the previously computed consistency scores (no re-test
        needed) to re-partition features into stable/unstable at
        the current `stability_cutoff`.

        Parameters
        ----------
        stats : dict | FitStats
            Previously stored consistency results.

        Returns
        -------
        tuple[FitResult, FitStats]
            Updated verdicts.
        """        
        if isinstance(stats, FitStats):
            stats = stats.consistency
        fec = self.run_consistency()
        res = {}
        for tx, d in stats.items():
            res[tx] = fec.update(d)            
        return self.get_result(res[TT.TRAIN], res[TT.TEST])
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Fitter(FitterDataBase):
    """
    WOE Stability Analysis — execution backend.

    Orchestrates the fit contract for WOE feature stability screening:

    - **fit**: Run the time-consistency test on all WOE features
      (train + optional test), and store the stable/unstable verdicts.
    - **update**: Re-evaluate verdicts with a new threshold (no
      re-computation needed).
    - **transform**: Select the stable WOE columns from the input
      data (pass-through — no value transformation).

    Parameters
    ----------
    params : FitParams
        Stability cutoff, alpha, time configuration.
    """
    def __init__(self, params:FitParams):
        self.prm = params 

    @staticmethod
    def _parallel_th(fes:list):
        p_th = ( len(fes) > 1e2 )       # check data complexity 
        return 0 if p_th else np.inf    # only run parallel (p_th=0) if complex data

    def _set_results(self, result: tuple[FitResult, FitStats]) -> None:
        self.result, self.stats = result
        self.features = self.result.vars_woe

    def fit(self, data:pDF, test_data:pDF, features:list):
        """
        Run stability analysis on all WOE features.

        Parameters
        ----------
        data : pDF
            Training data (target + time variable + WOE features).
        test_data : pDF
            Out-of-time test data (same columns as train).
        features : list
            WOE feature names to test.

        Returns
        -------
        self
            Fitted instance with `features` (stable WOE names) set.
        """
        p_th   = self._parallel_th(features)
        worker = FitWorker(self.prm)
        p_fit  = Parallel(
            data_and_fe = 1, 
            th          = p_th, 
            kwargs_pos  = True, 
            verbose     = self.prm.verbose,
            spin_title  = 'WoE stability analysis in progress'
        ).process(worker.process)        
        result = p_fit(data, features, test_data)
        self._set_results(result)

        return self
    
    def update(self, stats:dict|FitStats):
        """
        Re-evaluate stability verdicts with a new threshold.

        Useful for interactive threshold tuning: run `fit` once,
        then call `update` multiple times with different
        `stability_cutoff` values without re-running the full test.

        Parameters
        ----------
        stats : dict | FitStats
            Previously stored consistency results (from `fit`).

        Returns
        -------
        self
            Updated instance with new `features` list.
        """
        worker = FitWorker(self.prm)
        result = worker.update(stats)
        self._set_results(result)

        return self
    
    def transform(self, data:pDF) -> pDF:
        """
        Select stable WOE features from the input data.

        This is a pass-through selection (no value transformation) —
        the WOE values were already computed in the previous stage.
        Only the stable features (per the fit verdict) are retained.

        Parameters
        ----------
        data : pDF
            Data containing WOE features.

        Returns
        -------
        pDF
            Subset containing only the stable WOE columns.
        """
        return data[self.features]
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → WOE → STABILIZER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class WoeStabilityAnalyzer(FitterModelBase[Fitter]):
    """
    WOE Feature Stability Analyzer (Time-Wise Consistency Gate).

    Verifies that each WOE-encoded feature maintains consistent
    predictive behaviour over time. Features whose target-separation
    is temporally unstable are excluded before model training.

    Final feature quality gate. Runs after WOE transformation, before
    model training. Its output is the final feature set for modelling.

    Parameters
    ----------
    target : str
        Target column name.
    time_var : str
        Time column name (e.g., 'AIG_YY_MM').
    stability_cutoff : float, default=0.65
        Minimum consistency score for a feature to be considered
        time-stable. Higher = more conservative.
    alpha : float, default=0.05
        Significance level for bin-level consistency testing.
    time_unit : 'week' | 'day' | None, default=None
        Time grouping granularity. 'week' aggregates daily data
        into weekly units for more stable estimates.
    max_time_units : int | None, default=None
        Maximum number of time units (most recent) to consider.
    verbose : bool | Verbosity, default=True
        Enable progress and stats display.

    Notes
    -----
    **What "stability" means**: A WOE feature is stable if, across
    time units, each bin consistently separates the target (event
    rate direction and magnitude remain consistent). Unstable features
    may have excellent average Gini but are driven by a few anomalous
    periods — they will fail in production when the regime shifts.

    **Train + Test verification**: Stability is checked on both the
    training sample and the out-of-time test sample. A feature must
    pass on training to be kept; the test result provides
    additional OOS confidence.

    **Update without recompute**: After `fit`, you can call
    `update` with a new `stability_cutoff` to re-partition
    features without re-running the full consistency test. This
    enables interactive threshold tuning.

    **Transform is a selection**: The `transform` method does not
    modify values — it selects only the stable WOE columns from the
    input. The WOE values themselves were computed in the previous
    (WOE transformer) stage.

    **Subsampling**: For large datasets, a stratified subsample is
    used for the consistency test.

    Examples
    --------
    >>> from aidflow.fitters import WoeStabilityAnalyzer
    >>> stab = WoeStabilityAnalyzer(
    ...     target='is_churn',
    ...     time_var='AIG_YY_MM',
    ...     stability_cutoff=0.65,
    ...     time_unit='week',
    ... )
    >>> stab.fit(train_data, test_data, features=woe_features)
    >>> stable_features = stab.transform(data) 
    >>> stable_features.columns  # Only time-stable WOE features
    """

    def __init__(self, *args, params: dict|FitParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(FitParams, params, args, kwargs)

    def _get_fes(self, data:pDF, fes:list|None) -> list:
        if fes is not None:
            return fes
        return [x for x in data.columns if x not in [self._params.target, self._params.time_var]]

    def _get_woe_cols(self, fes: list) -> list:
        tt  = [self._params.target, self._params.time_var]
        woe = [x for x in fes if VarStd.is_woe_fe(x)]
        if not woe:
            raise ValueError(
                'AIG SAFETY ► Input data must contain WOE Features to analyze.'
            )
        return tt + woe
    
    @staticmethod
    def _verify_test(test:pDF, cols:list) -> None:
        m = [x for x in cols if x not in test.columns]
        if not m:
            return
        raise ValueError('AIG SAFETY'
            '\n► Test Data must contain all WOE features as Train data, '
            'including Target and Time Variable.'
            f'\n► Missing columns\n: {', '.join(m)}'
        )
        
    def _subsample(self, data:pDF, test:pDF, cols:list) -> list[pDF, pDF]:
        self._verify_test(test, cols)        
        res = []
        for d in (data, test):
            res.append(
                Subsampling(d[cols], d[self._params.target]).univar()[0]
            )
        return res

    def fit(self, train_data:pDF, test_data:pDF, features:list = None):        
        """
        Run stability analysis on all WOE features.

        Parameters
        ----------
        train_data : pDF
            Training data (target + time variable + WOE features).
        test_data : pDF
            Out-of-time test data (same structure as train).
        features : list | None, default=None
            Specific WOE features to test. If None, all WOE-suffixed
            columns are used.

        Returns
        -------
        self
            Fitted instance with `features` (stable WOE names) set.
        """      
        fes  = self._get_fes(train_data, features)
        cols = self._get_woe_cols(fes)
        d, t = self._subsample(train_data, test_data, cols)
        
        self._fitter = Fitter(self._params)
        self._fitter.fit(d, t, fes)

        return self
    
    def update(self, stats:dict|FitStats, woe_stability:float=None):        
        """
        Re-evaluate stability verdicts with a new threshold.

        After `fit`, this allows interactive threshold tuning
        without re-running the full consistency test.

        Parameters
        ----------
        stats : dict | FitStats
            Previously stored consistency results.
        woe_stability : float | None, default=None
            New `stability_cutoff` value. If None, the current
            parameter value is used.

        Returns
        -------
        self
            Updated instance with new `features` list.
        """
        if woe_stability is not None:
            self._params = AigParam.param_ar_kw(
                FitParams, self._params, woe_stability = woe_stability
            )
        self._fitter = Fitter(self._params)
        self._fitter.update(stats)
        return self
    
    def transform(self, data:pDF) -> pDF:
        """
        Select stable WOE features from the input data.

        Parameters
        ----------
        data : pDF
            Data containing WOE features.

        Returns
        -------
        pDF
            Subset containing only the time-stable WOE columns.
        """
        return self.fitter.transform(data) 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬