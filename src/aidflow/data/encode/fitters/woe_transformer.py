"""
WOE (Weight of Evidence) Transformation — Final Feature Encoding.

This module implements the final encoding stage of the pipeline,
responsible for transforming all encoded features (numerical,
categorical, and interaction) into their optimal WOE representation.

What is WOE?
------------
Weight of Evidence encodes each category/bin as:

    WOE = ln( Distribution of Goods / Distribution of Bads )
                                or
    WOE = ln( Distribution of Bads / Distribution of Goods )

This produces a **monotonic, interpretable, and stable** numeric
representation where:
- Each bin's value reflects its relative risk (positive = good,
  negative = bad, zero = neutral).
- The encoding is invariant to the original value scale.
- Tree-based and linear models can use WOE features interchangeably.

Why WOE?
--------
- **Interpretability**: Each WOE value has a direct business meaning
  (risk direction and magnitude per segment).
- **Stability**: WOE bins are less sensitive to small data changes
  than raw values (especially for skewed distributions).
- **Linearity**: Transforms non-linear feature-target relationships
  into (approximately) linear ones, benefiting logistic regression
  and linear scoring models.
- **Comparability**: All features are on a common risk scale.

Strategies
----------
Two search strategies are available to find the optimal bin structure
before WOE computation:

- **Gini Strategy (S1)**: Sequentially merges adjacent bins from top
  to bottom, accepting each merge only if the Gini drop is below
  the alpha threshold. Produces fewer bins (lower variance, higher
  bias). Recommended for features with strong train-overfit tendency.

- **T-Test Strategy (S2)**: Merges adjacent bins whose target means
  are not statistically different (Welch's t-test). Searches across
  multiple alpha levels to find the best balance of bins vs.
  performance. Produces more bins (higher variance, lower bias).
  Recommended when preserving granular signal is important.

Both strategies are reactions to the observed problem of
train-overfit binning in aggressive scoring flows — where features
with strong train signal but poor OOT stability confuse the
optimization algorithm (Stepwise/Lasso) at early stages.

Output
------
Per feature:
- **xW**: The encoded category (bin label) — for interpretability
  and model feature selection.
- **xWOE**: The WOE value — the final numeric representation.

Features that fail the Gini threshold after WOE transformation are
dropped.
"""

from ....deps.struct                import  dataclass, asdict
from ....deps.data                  import  np, pd, pDF, pSe

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
from ....core.interop               import  (
                                            Arrow,
                                            TypeStandard
                                            )
from ....core.standard              import  (
                                            VarType, 
                                            VarStd
                                            )
from ....core.display               import  Display

from ....ml.features                import  (
                                            Binning, 
                                            WoE,
                                            Encoding,
                                            EncodingMap
                                            )
from ....ml.metrics                 import  (
                                            Gini, 
                                            SeparationMetrics
                                            )
from ....ml.tests                   import  (
                                            TestModelDiff, 
                                            TestSampleDiff
                                            )
from ....ml.sampling                import  Subsampling
from ....display.rgb                import  RgbANSI
from ....display.verbosity          import  Verbosity



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class STR(DictClass):
    """WOE bin-pruning strategy identifiers."""
    G = 'gini'
    T = 't-test'

class LBL(DictClass):
    """Output column labels for WOE stats and search diagnostics."""
    V   = 'VAR'
    T   = 'Type'
    R   = 'Reason'
    M   = VarStd.MISSING
    W   = VarStd.WOE
    J   = 'Join'
    C   = 'Choice'
    S   = 'Significantδ'
    P   = 'p-value'
    PB  = 'p-base'
    A   = 'alpha'
    AB  = 'a-base'
    B   = 'Bins'
    BD  = 'Binsδ'   
    GO  = 'Gini_TF'
    GW  = 'Gini_WF'
    GD  = 'Giniδ_WT'
    IVO = 'IV_TF'
    IVW = 'IV_WF'

class STAT(DictClass):
    """Aggregation column labels for group-level stats."""
    M = 'mean'
    C = 'count'
    S = 'sum'

class TST(DictClass):
    """T-test result column label."""
    T = 'means_stat_diff'

class SIG(DictClass):
    """Binary sign indicators for display."""
    Y = "✅"
    N = "❌"
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → WOE → TRANSFORM
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class FitParamsBase:
    """
    WOE transformation thresholds.

    Attributes
    ----------
    gini_cutoff : float, default=0.02
        Minimum Gini coefficient for a feature's WOE encoding to be
        retained. If the WOE-encoded feature's Gini is below this
        threshold, the feature is dropped (insufficient predictive
        signal after encoding).
    alpha : float, default=0.05
        Statistical significance threshold. Used in two contexts:
        - **Gini strategy**: Accept a bin merge if the Delong p-value
          for the performance drop is ≥ alpha (i.e., not significant).
        - **T-test strategy**: Control the Welch t-test significance
          level for bin merging decisions.
    """     
    gini_cutoff: float = 0.02
    alpha:       float = 0.05

@Dataclass.base(update_specs=dict(slots=False))
class FitParamsCfg:
    """
    Pipeline configuration.

    Attributes
    ----------
    target : str
        Target column name. WOE is computed against this target.
    verbose : bool | Verbosity, default=True
        Enable progress display and stats output.
    """       
    target:  str
    verbose: bool|Verbosity = True

@Dataclass.base
class FitParams(FitParamsBase, FitParamsCfg):
    """Fitter params input design."""

@Dataclass.base
class FitResult:
    """
    Fit output: WOE encoding maps and feature verdicts.

    Attributes
    ----------
    encoding : dict
        Per-feature `EncodingMap` containing the WOE mapping
        (original → bin → WOE value), type, and performance stats.
    vars_woe : list
        Feature names (WOE-standard) that passed and were encoded.
    vars_dropped : list
        Feature names that failed the Gini gate, with reason.
    vars_map : dict
        Mapping of original feature name → [encoded_name, woe_name].
    """
    encoding:       dict
    vars_woe:       list
    vars_dropped:   list
    vars_map:       dict

@Dataclass.base
class FitStats:
    """
    Diagnostic statistics for user-facing display.

    Attributes
    ----------
    fit_worker : dict
        Per-feature worker output (encoding table, WOE data,
        original data, search stats).
    stats_in : list
        Features that PASSED the WOE gate (with Gini/IV metrics).
    stats_out : list
        Features DROPPED (with reason and performance metrics).
    """
    fit_worker: dict
    stats_in:   list
    stats_out:  list

@Dataclass.base
class WorkerResult:
    """
    Per-feature WOE worker output.

    Attributes
    ----------
    encoding : pDF
        The full encoding table (original value → WOE value),
        including the Missing category.
    encoded : pDF
        The pruned bin table with WOE values (final representation).
    original : pDF
        The original (unpruned) bin table with WOE values (for comparison).
    stats : pDF | None
        Search diagnostics (merge sequence with Gini/p-values).
    """
    encoding: pDF
    encoded:  pDF
    original: pDF
    stats:    pDF|None = None
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# WOE TRANSFORMATION UTILS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class WorkerUtils:
    """
    Shared utilities for WOE bin-pruning workers.

    Provides:
    - Data preprocessing (group-by aggregation, missing separation,
      categorical sorting).
    - Bin merging (join adjacent bins into one statistical group).
    - WOE encoding and output formatting.
    """

    @dataclass(slots=True)
    class VarData:
        """Per-feature working state during WOE transformation."""
        x:    str  # var in processing
        xW:   str  # var in processing naming standard Transformed
        df:   pDF  # Orig full length data x-target
        dt:   pDF  # Transformation Grouped Data → treatment by bins at the WOE EXECUTOR step
        do:   pDF  # Original Grouped Data → to be used for join for data by original values
        miss: pDF  # Missing Bin → excluding from the treatment of bin share merge
        type: str  # Num | Cat

    @staticmethod
    def copy_var(var:VarData) -> VarData:
        """Create a deep copy of the working state (for trial merges)."""
        var = WorkerUtils.VarData(**asdict(var))
        var.dt = var.dt.copy()
        return var

    @classmethod
    def update_var_join_bins(cls, var:VarData, bins:list) -> str:
        """
        Merge specified bins into a single bin in the working data.

        Parameters
        ----------
        var : VarData
            Feature working state.
        bins : list
            Bin values to merge.

        Returns
        -------
        str
            The merged bin label.
        """        
        merged_bin = Binning.merge(bins, vtype=var.type) 
        dt = var.dt 
        ix = dt[dt[var.x].isin(bins)].index
        dt.loc[ix, var.x] = merged_bin
        var.dt = cls.join_bins(dt, ix)
        return merged_bin

    @staticmethod
    def join_bins(data:pDF, indexes:pSe) -> pDF:
        """
        Consolidate merged bin rows into a single row.

        Aggregates: sum of counts/sums, weighted mean of target rate.
        Removes the now-redundant individual bin rows.

        Parameters
        ----------
        data : pDF
            Bin-level stats table.
        indexes : pSe
            Index positions of the bins being merged.

        Returns
        -------
        pDF
            Updated bin-level stats with merged rows removed.
        """
        d = data.loc[indexes,:]                         # bins indexes: from - to data merge
        s = d[STAT.S].sum()
        c = d[STAT.C].sum()
        m = ( d[STAT.M] * d[STAT.C] ).sum() / max(1, c)   # Target mean of merged Bins

        data.loc[indexes[-1], STAT.list()] = [m, c, s]
        return data.drop(indexes[:-1], axis=0)
    
    @staticmethod
    def join_sign(J) -> str:
        """Convert boolean merge flag to display indicator."""
        if J: return SIG.Y
        else: return SIG.N
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def data_preprocess(cls, data:pDF, target:str, x:str, vtype:str) -> VarData:
        """
        Prepare the working state for WOE transformation.

        Steps:
        1. Extract feature + target data.
        2. Aggregate bin-level stats (mean, count, sum) by feature value.
        3. Separate the missing bin (for WOE computation with adjustment).
        4. For categorical variables: sort bins by target rate (descending)
           to ensure consistent merge direction.
        5. Standardize bin labels to string for uniform processing.

        Parameters
        ----------
        data : pDF
            Full dataset (feature + target).
        target : str
            Target column name.
        x : str
            Feature column name.
        vtype : str
            Variable type ('N' or 'C').

        Returns
        -------
        VarData
            Working state with bin-level stats and naming.
        """
        df = data.loc[:, [x, target]].reset_index(drop=True)

        d_xt = (
            df.groupby(x, observed=True)
                .agg( STAT.list() )
                .droplevel(0, axis=1)
                .replace(np.nan, 0) 
                .reset_index()
        ) 
        miss = Binning.ensure_missing(d_xt, x)     
        d_xt = d_xt[d_xt[x] != VarStd.MISSING]  

        if vtype == VarType.C: 
            d_xt = d_xt.sort_values(by=[STAT.M], ascending=False)
            # Feature Type Treatment → For Cat variables sort by Target Mean
        
        d_xt = d_xt.reset_index(drop=True)
        orig = d_xt.copy()

        d_xt[x] = d_xt[x].astype(str)   # bins operation processing standard

        return cls.VarData(
            x  = x,
            xW = VarStd.fe_w(x),
            df = df,
            dt = d_xt,
            do = orig,
            miss = miss,
            type = vtype
        )    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def data_encode(var: VarData, stats:pDF|None=None) -> WorkerResult:
        """
        Finalize WOE encoding and produce the result package.

        Steps:
        1. Restore interval format for numerical bins.
        2. Reattach the missing bin.
        3. Compute WOE values for both pruned and original bin structures.
        4. Build the mapping (original value → bin → WOE) for transform.
        5. Package into `WorkerResult`.

        Parameters
        ----------
        var : VarData
            Feature working state (post-merge).
        stats : pDF | None
            Strategy-specific search diagnostics.

        Returns
        -------
        WorkerResult
            Final WOE encoding package.
        """        
        x  = var.x
        xW = var.xW
        dt = var.dt.reset_index(drop=True)
        do = var.do

        # Num Vars formatter back
        if ( var.type == VarType.N ): 
            dt[x] = dt[x].apply(Binning.to_interval_type)

        # Standard WOE Result output - bins + missing
        dt = pd.concat([dt, var.miss])
        do = pd.concat([do, var.miss])
        dw = do.copy()

        # WOE Compute and Mapping (dt as final pruned WOE)
        dt[LBL.W] = WoE.extend(dt, x, adjust_missing=True)[LBL.W] # FINAL WOE AFTER TRANSFORMATION
        do[LBL.W] = WoE.extend(do, x, adjust_missing=True)[LBL.W] # COMPARE WOE → TRANSF vs ORIG       

        # Transformed Input → Output to Join Train Data by orig var
        dw[xW] = Binning.mapping(do[x], dt[x], var.type, VarStd.MISSING)
        dw[xW] = TypeStandard.any_to_cat(dw[xW])        
        
        # dw for efficient join to measure performance
        dt = dt.rename(columns={x: xW})
        dw[LBL.W] = Binning.data_mapping(dw, dt, xW, LBL.W)
        return WorkerResult(
            encoding = dw,
            encoded  = dt,
            original = do,
            stats    = stats
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# WOE TRANSFORMATION BY WELCH t-TEST
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FitWorkerTest:        
    """
    WOE Bin Pruning via Welch's t-Test Alpha Search (Strategy S2).

    Prunes bins whose target rates are not statistically different,
    searching across multiple alpha levels to find the optimal balance
    between bin count (variance) and predictive power (bias).

    Algorithm
    ---------
    1. For each alpha in the search space (default: 0.001 → 0.2):
       - Merge bins whose target rates fail the Welch t-test at that alpha.
       - Compute the resulting WOE and Gini.
    2. Sort candidates by Gini (ascending).
    3. Compute DeLong p-values between successive candidates (model
       difference test) to identify where performance drops significantly.
    4. Select the alpha that:
       - Preserves the highest Gini among non-significantly-different models.
       - Has the minimum bin count at that performance level.
       - Fallback: the minimum alpha (most conservative) if all models
         are significantly different.

    Bias-Variance Profile
    ---------------------
    - **Lower bias** than Gini strategy: preserves more bin distinctions.
    - **Higher variance** than Gini strategy: may retain more bins.
    - Best suited for features where preserving fine-grained signal
      is more important than maximum stability.

    Parameters
    ----------
    target : str
        Target column name.
    x : str
        Feature column name.
    vtype : str
        Variable type ('N' or 'C').
    alpha : float, default=0.05
        Base significance level.
    alpha_search_space : tuple, default=(0.001, 0.01, 0.05, 0.1, 0.2)
        Alpha levels to explore. Lower alpha = stricter = more merging.

    Returns
    -------
    WorkerResult
        WOE encoding with t-test-pruned bins and search diagnostics.
    """
    def __init__(self,
        target: str,
        x:      str,
        vtype:  str,
        alpha:  float = 0.05,
        alpha_search_space: tuple = (0.001, 0.01, 0.05, 0.1, 0.2)
    ):
        self.target = target
        self.x      = x
        self.vtype  = vtype
        self.alpha  = alpha
        self.alpha_search_space = alpha_search_space
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def welch_test(data:pDF, alpha:float=0.05) -> pDF:
        """
        Run Welch's t-test on adjacent bins.

        Tests whether each bin's target rate is significantly different
        from the next bin's rate. Bins that fail the test (not
        significantly different) are candidates for merging.

        Parameters
        ----------
        data : pDF
            Bin-level stats (mean, count, sum).
        alpha : float
            Significance level.

        Returns
        -------
        pDF
            Test results with merge flags and running bin count.
        """
        test = TestSampleDiff(alpha).welch_se(data).rename(columns={TestSampleDiff.RES: TST.T})
        t = 1-test[TST.T]               # 1: can be merged | 0: cannot        
        b = 1+len(data) + -1*t.cumsum() # +1 - the first row is irrelevant as it is always False
        j = [np.nan] + [WorkerUtils.join_sign(j) for j in t[1:]]
        return test.assign(**{LBL.J:j, LBL.B:b})
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _is_significant(self, p_val:float|pSe) -> bool|pSe:
        """Check if a p-value indicates statistical significance."""
        return p_val < self.alpha
                
    @classmethod
    def _stand_search_res(cls, *p_res, space_v:dict=None) -> list:
        """Format a search result entry (with or without base comparison)."""      
        C = [LBL.A, LBL.B, LBL.GW]
        if space_v is None:
            return dict(zip( C + [LBL.W], p_res))
        s = [space_v[c] for c in C]
        return dict(zip( C + [LBL.P, LBL.PB, LBL.AB], (*s, *p_res) ))

    def _evaluate_search_space(self, var:WorkerUtils.VarData) -> tuple[list, pSe]:
        """
        Evaluate WOE performance at each alpha level.

        For each alpha in the search space:
        1. Prune bins using Welch t-test at that alpha.
        2. Compute WOE and Gini on the pruned structure.
        3. Record the result.

        Parameters
        ----------
        var : VarData
            Feature working state.

        Returns
        -------
        tuple[list, pSe]
            Search results and the target series (for DeLong testing).
        """        # Data for WOE Search → Only on non-missing → Missing separated treatment
        d  = var.df
        dw = d.loc[d[self.x] != VarStd.MISSING,:].reset_index(drop=True)
        y_ = dw[self.target]

        def prune(alpha:float) -> tuple[dict, pSe]:            
            woe  = self.prune_encode(var, alpha)
            woe_ = Binning.data_mapping(dw, woe.encoding, self.x, LBL.W).astype(float)
            gini = Gini.fast_calc(y_, woe_)
            bins = len(woe.encoded) - 1  # Number of Pruned Bins after test wo Missing (-1)
            return self._stand_search_res(alpha, bins, gini, woe_)
        
        search = []
        for alpha in self.alpha_search_space:
            search.append( prune(alpha) )
            
        return search, y_

    @staticmethod
    def _extract_search_vals(s:dict) -> tuple[pSe, int]:
        """Extract (WOE series, bin count, alpha) from a search entry."""
        return s[LBL.W], s[LBL.B], s[LBL.A]
      
    def _run_search(self, search:list, y_true:pSe) -> list:
        """
        Compute DeLong p-values between successive search candidates.

        Walks through the Gini-sorted search results, computing the
        model-difference p-value between each candidate and the
        previous "significant" baseline. Updates the baseline when
        a significant difference is found.

        Parameters
        ----------
        search : list
            Gini-sorted search results.
        y_true : pSe
            True target (for DeLong test).

        Returns
        -------
        list
            Results enriched with p-values and base comparisons.
        """        
        res = []
        for i, s in enumerate(search):
            if ( i == 0 ):
               p_val, p_base, a_base = [-1,0,0] # ► init vals to set dynamic xs0 as y_pred1 (woe1)
            else:
                xs1   = s[LBL.W]                # y_pred2 (woe2) 
                p_val = TestModelDiff(y_true, xs0, xs1).delong()            
            res.append( self._stand_search_res(p_val, p_base, a_base, space_v=s) )
            if self._is_significant(p_val):     # ► Dynamic update - the last sig. to the next
                xs0, p_base, a_base = self._extract_search_vals(s)
        return res
    
    def _format_search_res(self, result:list) -> pDF:
        """
        Format the alpha search results into a structured table.

        Adds computed columns: bin reduction, Gini change,
        and significance flag.

        Parameters
        ----------
        result : list
            Enriched search results.

        Returns
        -------
        pDF
            Formatted search table.
        """        
        res:pDF = pDF(result)
        res[LBL.BD] = res[LBL.B]  - res[LBL.PB]
        res[LBL.GD] = res[LBL.GW] - res[LBL.AB].map(res.set_index(LBL.A)[LBL.GW])
        res[LBL.S]  = self._is_significant(res[LBL.P])
        return res
            
    def _search_vector(self, var:WorkerUtils.VarData) -> pDF:
        """
        Execute the full alpha search pipeline.

        Parameters
        ----------
        var : VarData
            Feature working state.

        Returns
        -------
        pDF
            Complete alpha search results table.
        """
        search, y = self._evaluate_search_space(var)
        search.sort(key=lambda x: x[LBL.GW]) # ► Gini sort → important for search algo to work properly
        res = self._run_search(search, y)
        return self._format_search_res(res)    
    # ─────────────────────────────────────────────────────────────────────────────────────────────    

    def _alpha_choice(self, search_stats:pDF) -> float:
        """
        Select the optimal alpha from the search results.

        Selection logic:
        1. Filter to "break points" (significant model differences or
           bin reduction).
        2. If any significant difference exists: select the maximum
           Gini among significant candidates (preserve signal).
        3. From that subset: select the minimum bin count (reduce variance).
        4. Return the corresponding alpha.

        Fallback: if no significant differences found, return the
        minimum alpha (most conservative, maximum merging).

        Parameters
        ----------
        search_stats : pDF
            Formatted alpha search results.

        Returns
        -------
        float
            Selected alpha level.
        """
        def p_min(w:pDF): return w[LBL.P].min()
        def s_max(w:pDF): return w[w[LBL.S]].index[-1]
        def g_max(w:pDF): return w.loc[s_max(w): , :]
        def b_min(w:pDF): return w[LBL.B] == w[LBL.B].min()
        def a_min(w:pDF): return w.loc[b_min(w), LBL.A].iloc[0]

        def optim_space(w:pDF) -> pDF:
            return w.loc[ self._is_significant(w[LBL.P]) | ( w[LBL.BD] < 0 ), :]   
            # points of interests → break points in performance

        def optim_search(stats:pDF) -> pDF:
            w = optim_space(stats)
            if not len(w):
                return stats
            return w
        
        def alpha(stats:pDF) -> float:
            w = optim_search(stats)
            if self._is_significant( p_min(w) ): 
                w = g_max(w) # significant models diff goes to max perf.
            return a_min(w)

        return alpha(search_stats)

    @staticmethod
    def _update_search_choice(search:pDF, alpha:float) -> None:
        """Mark the selected alpha in the search table with an indicator."""
        search[LBL.C] = np.where(search[LBL.A].to_numpy() == alpha, SIG.Y, SIG.N)
    # ─────────────────────────────────────────────────────────────────────────────────────────────    

    def bins_alpha_search(self, var:WorkerUtils.VarData) -> dict[pDF, float]:
        """
        Execute the full alpha search and select the optimal alpha.

        This is the core decision function: it explores the alpha
        space, evaluates performance at each level, and selects the
        alpha that best balances bin count (variance) and predictive
        power (bias).

        Parameters
        ----------
        var : VarData
            Feature working state.

        Returns
        -------
        dict
            {'alpha_search': pDF, 'alpha_selected': float}
        """
        search = self._search_vector(var)
        alpha  = self._alpha_choice(search)
        self._update_search_choice(search, alpha)    
        return {'alpha_search':search, 'alpha_selected': alpha}
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def bins_search_var_data(var:WorkerUtils.VarData, alpha=float|None, search=dict|None
    ) -> tuple[WorkerUtils.VarData, float, dict]:
        """
        Validate and prepare inputs for the prune-encode step.

        Ensures either an explicit alpha or a search result is provided.
        Copies the variable state if no search context exists.

        Parameters
        ----------
        var : VarData
            Feature working state.
        alpha : float | None
            Explicit alpha (skips search).
        search : dict | None
            Search results (from `bins_alpha_search`).

        Returns
        -------
        tuple[VarData, float, dict]
            Validated inputs for `prune_encode`.

        Raises
        ------
        ValueError
            If neither alpha nor search result is provided.
        """
        if not alpha and not isinstance(search, dict):
            raise ValueError('AIG SAFETY ► Please provide either alpha value or result of alpha search')        
        if search is None:
            var = WorkerUtils.copy_var(var)
        alpha  = alpha  or search['alpha_selected']
        search = search or {}
        return var, alpha, search
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_bins(x:str, dt:pDF, wt:pDF) -> list:
        """
        Determine which bins to merge based on Welch t-test results.

        Adjacent bins that are not significantly different are grouped.
        If a merged group's neighbour is also part of the next merge
        group, cascading merges are applied.

        Parameters
        ----------
        x : str
            Feature column name.
        dt : pDF
            Bin-level data.
        wt : pDF
            Welch t-test results.

        Returns
        -------
        list
            List of merge groups (each group is a list of bin values).
        """
        I = wt[~wt[TST.T]].index[1:] -1 # -1 → First positions towards second → -1 goes to first as merge
        B = [[dt.loc[i,x], dt.loc[i+1,x]] for i in I]   # Merge of two stat same bins

        # Check of neigbors → if next pair is same too then merge with previous pair
        def neigbor(B):
            I = [i for i in range(len(B)-1) if any(True for b in B[i] if b in B[i+1])]
            return I, len(I)

        I,R = neigbor(B)
        while R > 0:
            i = I[0]
            B[i].extend(B[i+1])
            B.remove(B[i+1])
            I,R = neigbor(B)
        return B
    
    def _dedup_bins(B:list, var_type:str) -> list:
        """
        Deduplicate merged bin groups (categorical only).

        For categorical variables, ensures each unique value appears
        only once in the merged group.

        Parameters
        ----------
        B : list
            Merge groups.
        var_type : str
            Variable type.

        Returns
        -------
        list
            Deduplicated merge groups.
        """
        if var_type == VarType.N:
            return B
        BF = []
        for b in B:
            u = list( np.unique(b[1:]) )
            BF.append([b[0]] + u)  # b[0] → group origin → stat same with the rest
        return BF        

    @classmethod
    def prune_encode(cls, var:WorkerUtils.VarData, alpha:float=None, search:dict=None) -> WorkerResult:
        """
        Execute bin pruning via t-test and finalize WOE encoding.

        Parameters
        ----------
        var : VarData
            Feature working state.
        alpha : float | None
            Alpha for t-test merging (or taken from search).
        search : dict | None
            Alpha search results (if alpha not provided directly).

        Returns
        -------
        WorkerResult
            Final WOE encoding package with t-test diagnostics.
        """
        var, alpha, search = cls.bins_search_var_data(var, alpha, search)
        
        x  = var.x
        dt = var.dt
        wt = cls.welch_test(dt, alpha)

        B = cls._merge_bins(x, dt, wt)
        B = cls._dedup_bins(B, var.type)

        # Propagate to Data
        for b in B:
            WorkerUtils.update_var_join_bins(var, b)

        stats = {
            'welch_t_test': wt,
            **search
        }
        return WorkerUtils.data_encode(var, stats) 
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def process(self, data:pDF) -> WorkerResult:
        """
        Execute the full t-test WOE transformation pipeline.

        Parameters
        ----------
        data : pDF
            Data containing the feature and target columns.

        Returns
        -------
        WorkerResult
            WOE encoding with t-test-pruned bins.
        """
        var = WorkerUtils.data_preprocess(data, self.target, self.x, self.vtype)
        
        if not len(var.dt) > 1:
            return WorkerUtils.data_encode(var, None)
        
        search = self.bins_alpha_search(var)
        return self.prune_encode(var, search=search)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# WOE TRANSFORMATION BY GINI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FitWorkerGini:
    """
    WOE Bin Pruning via Gini-Impact Sequential Merge (Strategy S1).

    Merges adjacent bins sequentially (top-to-bottom) if the Gini
    drop from the merge is not statistically significant (DeLong test).
    Produces the minimum bin structure that preserves predictive power.

    Algorithm
    ---------
    1. Compute the original WOE and Gini (baseline).
    2. Starting from bin 1, merge it with bin 2.
    3. Recompute WOE and Gini on the merged structure.
    4. Run DeLong test: if the Gini drop is significant (p < alpha),
       reject the merge (restore bin). Otherwise, accept it.
    5. Continue to the next pair.
    6. Result: minimum bins with maximum preserved Gini.

    Bias-Variance Profile
    ---------------------
    - **Higher bias** than t-test strategy: merges more aggressively.
    - **Lower variance** than t-test strategy: fewer bins = more stable.
    - Best suited for features where maximum OOT stability is the
      priority and fine-grained signal is less critical.

    Parameters
    ----------
    target : str
        Target column name.
    x : str
        Feature column name.
    vtype : str
        Variable type ('N' or 'C').
    alpha : float, default=0.05
        Significance level for the DeLong model-difference test.

    Returns
    -------
    WorkerResult
        WOE encoding with Gini-pruned bins and merge diagnostics.
    """
    def __init__(self,
        target: str,
        x:      str,
        vtype:  str,
        alpha:  float = 0.05,
    ):
        self.target = target
        self.x      = x
        self.vtype  = vtype
        self.alpha  = alpha
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _gini(self, var:WorkerUtils.VarData, woe:pSe) -> float:
        """Compute Gini of the target separation by WOE values."""
        d = var.df
        return Gini.fast_calc(d[self.target], woe)
    
    def _woe(self, var:WorkerUtils.VarData) -> pSe:
        """Compute WOE values for the current bin structure."""
        woe  = WorkerUtils.data_encode(var)
        return Binning.data_mapping(var.df, woe.encoding, self.x, LBL.W).astype(float)    
    
    def _merge_bins(self, var:WorkerUtils.VarData, i:int) -> tuple[list[str, str], pSe, float]:        
        """
        Trial-merge bins i and i+1, then measure the resulting Gini.

        Parameters
        ----------
        var : VarData
            Feature working state.
        i : int
            Position of the first bin to merge.

        Returns
        -------
        tuple
            (bin values merged, WOE series, resulting Gini).
        """
        var  = WorkerUtils.copy_var(var)        
        ix   = [i,i+1]                      # pos of bins to check
        bins = list(var.dt.loc[ix, self.x]) # val of bins to check
        WorkerUtils.update_var_join_bins(var, bins)
        xs = self._woe(var)
        return bins, xs, self._gini(var, xs)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def prune_encode(self, var:WorkerUtils.VarData) -> WorkerResult:
        """
        Execute sequential Gini-based bin pruning and finalize WOE encoding.

        Walks through all adjacent bin pairs (top-to-bottom), merging
        each pair if the Gini impact is not significant. Records the
        full merge sequence for diagnostics.

        Parameters
        ----------
        var : VarData
            Feature working state.

        Returns
        -------
        WorkerResult
            Final WOE encoding with Gini merge diagnostics.
        """    
        stats = lambda j: io.append({
                LBL.GO:  gd, 
                LBL.GW:  gs, 
                LBL.GD:  gs-gd, 
                LBL.P :  p_v,
                LBL.J:   WorkerUtils.join_sign(j),
                LBL.B:   bd, 
                LBL.BD:  bd-bo
        })
        io = []
        bo = bd = len(var.dt)    # init number of bins
        xd = self._woe(var)      # init woe (y_pred)
        gd = self._gini(var, xd) # init orig. performance                
        
        for i in range(bo-1):
            
            bins, xs, gs = self._merge_bins(var, i)
            p_v = TestModelDiff(var.df[self.target], xd, xs).delong()

            if p_v < self.alpha:
                stats(0)
                continue
            
            bd -= 1
            WorkerUtils.update_var_join_bins(var, bins)
            stats(1) 
            xd = xs
            gd = gs

        return WorkerUtils.data_encode(var, pDF(io))
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def process(self, data:pDF) -> WorkerResult:        
        """
        Execute the full Gini WOE transformation pipeline.

        Parameters
        ----------
        data : pDF
            Data containing the feature and target columns.

        Returns
        -------
        WorkerResult
            WOE encoding with Gini-pruned bins.
        """
        var = WorkerUtils.data_preprocess(data, self.target, self.x, self.vtype)

        if not len(var.dt) > 1:
            return WorkerUtils.data_encode(var, None)
        
        return self.prune_encode(var)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# WOE TRANSFORMATION
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FitWorker:
    """
    WOE Transformation Orchestrator — per-feature pipeline.

    Coordinates the full WOE transformation for a single feature:
    1. Prepare the working state (bin-level stats).
    2. Execute the selected pruning strategy (Gini or t-test).
    3. Compute performance (Gini/IV) for original vs pruned.
    4. Gate: check if the WOE-encoded feature meets the Gini threshold.
    5. Register the encoding (or drop with reason).

    Parameters
    ----------
    params : FitParams
        WOE transformation parameters.
    strategy : 'gini' | 't-test'
        Bin pruning strategy.
    """
    def __init__(self, params:FitParams, strategy:str = STR.G):
        self.prm         = params
        self.target      = params.target
        self.alpha       = params.alpha
        self.gini_cutoff = params.gini_cutoff
        self.verbose     = params.verbose
        self.strategy    = strategy      

        self.result: FitResult = Dataclass.create_default_instance(FitResult)
        self.stats : FitStats  = Dataclass.create_default_instance(FitStats)
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    @dataclass(slots=True)
    class VarData:
        """Per-feature working state at the WOE worker level."""
        x:     str      # orig. var name
        xT:    str      # input var → prior transformed (encoded) by num/cat
        xW:    str      # _W  naming standard Transformed of WOE
        xWOE:  str      # WOE naming standard Transformed
        x_t:   list     # target-var names
        df:    pDF      # data of x_t full length
        type:  str      # var type TYPE
        miss:  float    # missing rate of d_xt

    @Dataclass.base     # Output standard
    class Performance:  # Used by ml/woe.py _get_stats_gini() to retrieve trained gini perf.
        """
        WOE transformation performance metrics.

        Attributes
        ----------
        gini_org : float
            Gini of the original (unpruned) bin structure.
        gini_woe : float
            Gini of the pruned WOE-encoded structure.
        iv_org : float
            IV of the original structure.
        iv_woe : float
            IV of the pruned WOE-encoded structure.
        """        
        gini_org: float|int
        gini_woe: float|int
        iv_org:   float|int
        iv_woe:   float|int
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def prune(self, data:pDF, x:str, type:str) -> WorkerResult:
        """
        Dispatch to the selected WOE pruning strategy.

        Parameters
        ----------
        data : pDF
            Feature + target data.
        x : str
            Feature name.
        type : str
            Variable type ('N' or 'C').

        Returns
        -------
        WorkerResult
            WOE encoding result from the selected strategy.
        """
        if self.strategy == STR.G:
            return FitWorkerGini(self.target, x, type, self.alpha).process(data)

        if self.strategy == STR.T:
            return FitWorkerTest(self.target, x, type, self.alpha).process(data)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def set_var(self, data:pDF, xT:str, type:str) -> VarData:
        """
        Initialize the working state for a feature at the WOE level.

        Parameters
        ----------
        data : pDF
            Full dataset.
        xT : str
            Transformed feature name (post-discretization).
        type : str
            Variable type.

        Returns
        -------
        VarData
            Working state with naming and data.
        """        
        x    = VarStd.fe_to_var(xT)
        xW   = VarStd.fe_w(xT)
        xWOE = VarStd.fe_woe(xT)
        x_t  = [xT, self.target]
        d_xt = data.loc[:, x_t].reset_index(drop=True)
        miss = (d_xt[xT]==VarStd.MISSING).sum() / len(d_xt)
        return self.VarData(
            x    = x,
            xT   = xT,
            xW   = xW,
            xWOE = xWOE,
            x_t  = x_t,
            df   = d_xt,
            type = type,
            miss = miss
        )

    def get_woe_encoding(self, var:VarData) -> WorkerResult:
        """Execute the WOE pruning strategy for this feature."""
        return self.prune(var.df, var.xT, var.type)
    
    def calc_perf(self, var:VarData, woe:WorkerResult) -> Performance:
        """
        Compute performance metrics for original vs WOE-encoded feature.

        Parameters
        ----------
        var : VarData
            Feature working state.
        woe : WorkerResult
            WOE encoding result.

        Returns
        -------
        Performance
            Gini and IV for both original and WOE-encoded structures.

        Raises
        ------
        ValueError
            If WOE join produces NaN (mapping failure).
        """        
        xT = var.xT
        dx = var.df
        
        w = Binning.data_mapping(dx, woe.encoding, xT, LBL.W).astype(float)  # Pruned WOE perf.
        o = Binning.data_mapping(dx, woe.original, xT, LBL.W).astype(float)  # Original WOE perf.
        
        if ( w.isna().sum() ) > 0 or ( o.isna().sum() > 0 ):
            raise ValueError(RgbANSI.bold_red(f'WARNING: WOE_TRANSFORM → {xT}: WOE JOIN FAILED!'))

        return self.Performance(
            gini_org = Gini.fast_calc(dx[self.target], o),
            gini_woe = Gini.fast_calc(dx[self.target], w),
            iv_org   = SeparationMetrics.iv_total(woe.original),
            iv_woe   = SeparationMetrics.iv_total(woe.encoding),
        )

    def is_predictive(self, var:VarData, perf:Performance, woe:WorkerResult) -> bool:
        """
        Gate: check if the WOE-encoded feature meets the Gini threshold.

        Parameters
        ----------
        var : VarData
            Feature working state.
        perf : Performance
            Computed performance metrics.
        woe : WorkerResult
            WOE encoding result.

        Returns
        -------
        bool
            True if the feature passes. If False, registered as dropped.
        """        
        if ( perf.gini_woe > self.gini_cutoff ):
            return True
        reason = f'Transformation < TH {self.gini_cutoff}'
        self.set_res_vars_out(var, perf, woe, reason)        
        return False 
        
    def encode(self, var:VarData, perf:Performance, woe:WorkerResult) -> None:
        """Register the final WOE encoding for a passed feature."""        
        map = woe.encoded   # pruned bins with woe encoding 
        self.set_res_vars_in(var, perf, woe)
        self.set_res_encoding(var, perf, map, woe)       
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def set_res_encoding(self, var:VarData, perf:Performance, map:pDF, woe:WorkerResult) -> None:
        """
        Store the WOE encoding in the result.

        Registers:
        - Feature mapping in `result.vars_map`.
        - Encoding map in `result.encoding`.
        - Worker diagnostics in `stats.fit_worker`.
        """        
        self.result.vars_map[var.x] = [var.xW, var.xWOE]
        self.result.encoding[var.x]  = EncodingMap(
            vars  = [var.x, var.xW, var.xWOE], 
            type  = var.type,
            map   = map, 
            stats = perf, 
        )
        self.stats.fit_worker[var.x] = woe

    def get_res_bins(self, var:VarData, woe:WorkerResult) -> int:
        """
        Count final bins after WOE pruning.

        Includes the 'Missing' category if present.
        """        
        miss = 0 if var.miss else -1    # non-Missing present on Train should be considered
        return len(woe.encoded) + miss  # bins after WOE pruning; by default w Missing Category
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def stats_io(self, IO:list, x:str, vtype:str, miss:float, bins:int, reason:str, perf:Performance
    ) -> None: 
        """Append a feature's WOE verdict record to the stats output list."""        
        IO.append({
            LBL.V:   x,
            LBL.R:   reason,
            LBL.T:   vtype,
            LBL.B:   bins,
            LBL.M:   miss,
            LBL.IVW: perf.iv_woe,
            LBL.IVO: perf.iv_org,
            LBL.GW:  perf.gini_woe,
            LBL.GO:  perf.gini_org
        })

    def set_res_vars_in(self, var:VarData, perf:Performance, woe:WorkerResult) -> None: 
        """Register a feature as PASSED (WOE-encoded and kept)."""
        self.result.vars_woe.append(var.xWOE)
        io      = self.stats.stats_in
        bins    = self.get_res_bins(var, woe)
        reason  = f'Passed TH {self.gini_cutoff}'
        self.stats_io(io, var.xW, var.type, var.miss, bins, reason, perf)

    def set_res_vars_out(self, var:VarData, perf:Performance, woe:WorkerResult, reason:str) -> None: 
        """Register a feature as DROPPED (failed WOE gate)."""        
        self.result.vars_dropped.append(var.xT)
        io      = self.stats.stats_out
        bins    = self.get_res_bins(var, woe)
        reason  = 'Unpredictive → ' + reason
        self.stats_io(io, var.xT, var.type, var.miss, bins, reason, perf)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def pipeline(self, data:pDF, fe_types:dict, **kwargs):
        """
        Execute the WOE transformation pipeline for all features.

        Per feature:
        1. Set working state.
        2. Execute WOE pruning (strategy-specific).
        3. Compute performance.
        4. Gate: predictive power check.
        5. Register encoding (or drop).

        Parameters
        ----------
        data : pDF
            Full dataset (features + target).
        fe_types : dict
            Mapping of feature name → type ('N' or 'C').
        kwargs : dict
            Forwarded to progress display.
        """
        Nloop, progress = Display.processing_progress(
            fe_types, kwargs, self.verbose, desc='Features WOE Transformation'
        )
        for xT in Nloop: # xT → loop over Transformed (encoded) vars
            Display.update_progress(progress)

            var  = self.set_var(data, xT, fe_types[xT])
            woe  = self.get_woe_encoding(var)
            gini = self.calc_perf(var, woe)

            if not self.is_predictive(var, gini, woe):
                continue
            
            self.encode(var, gini, woe)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def process(self, data:pDF, fe_types:dict, **kwargs) -> tuple[FitResult, FitStats]:
        """
        Entry point: execute WOE transformation and return results.

        Parameters
        ----------
        data : pDF
            Full dataset.
        fe_types : dict
            Feature type mapping.
        kwargs : dict
            Forwarded to pipeline.

        Returns
        -------
        tuple[FitResult, FitStats]
            WOE encoding maps + diagnostic stats.
        """        
        self.pipeline(data, fe_types, **kwargs)
        return self.result, self.stats
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → WOE → TRANSFORM
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Fitter(FitterDataBase):
    """
    WOE Transformation — execution backend.

    Orchestrates the fit-transform contract for WOE encoding:

    - **fit**: For each feature, execute bin pruning (Gini or t-test),
      compute WOE values, gate on predictive power, and store encoding.
    - **transform**: Apply the learned WOE mappings to new data,
      producing both the bin label and the WOE numeric value.

    Strategy Selection
    ------------------
    - **Gini (S1)**: Sequential merge, DeLong-gated. Fewer bins,
      lower variance, higher bias. Best for stability-critical features.
    - **t-Test (S2)**: Alpha search, Welch-merged, DeLong-selected.
      Preserves more signal, slightly higher variance. Best for
      signal-critical features.

    Both strategies prevent overfitting of bin structure to training
    noise, which is a common failure mode in scoring flows that use
    greedy binning.

    Parameters
    ----------
    params : FitParams
        Gini cutoff and alpha threshold.
    strategy : 'gini' | 't-test'
        Bin pruning strategy.
    """

    def __init__(self, params:FitParams, strategy:str=STR.G):
        self.prm = params 
        self.strategy = strategy

    def fit(self, data:pDF, fe_types:dict):
        """
        Fit WOE encoding on all features.

        Parameters
        ----------
        data : pDF
            Training data (target + all encoded features).
        fe_types : dict
            Mapping of feature name → type ('N' for numerical,
            'C' for categorical).

        Returns
        -------
        self
            Fitted instance with `encoding` and `features` set.
        """
        worker = FitWorker(self.prm, self.strategy)
        p_fit  = Parallel(
            data_and_fe = 1, 
            th          = 30, 
            kwargs_pos  = True, 
            verbose     = self.prm.verbose,
            spin_title  = 'WoE transformation in progress'        
        ).process(worker.process)         
        result, stats = p_fit(data, fe_types)
        
        self.result: FitResult = result
        self.stats : FitStats  = stats

        self.encoding = self.result.encoding
        self.features = self.result.vars_woe

        return self
        
    @Arrow.transition(contract = Arrow.Contract(safe_conversion=True))
    def transform(self, data:pDF) -> pDF:
        """
        Apply fitted WOE encodings to new data.

        For each encoded feature, produces:
        - The bin label (categorical, e.g., '[-1.5, 0.2)')
        - The WOE value (numeric, e.g., 0.342)

        Unknown values (not seen during fit) are mapped to the
        'Missing' category.

        Parameters
        ----------
        data : pDF
            New data containing the original (pre-WOE) features.

        Returns
        -------
        pDF
            DataFrame with two columns per encoded feature:
            `{name}_W` (bin label) and `{name}_WOE` (WOE value).
        """
        # NOTE Arrow → np.select has hard cond → if cond.dtype.type is not np.bool   
        dt = {}
        for var in data.columns:                # transforms orig. variables in input data
            if var not in self.encoding:        # removed variables during fit - not predictive
                continue

            fitted:EncodingMap = self.encoding[var]
            xT_xW   = fitted.vars
            vtype   = fitted.type
            mapping = fitted.map

            xT, xW, xWOE = xT_xW

            woe, x_w = Encoding(
                ensure_trf = True, 
                incl_miss  = True
            ).encode(data[var], mapping, xW, LBL.W, vtype)

            dt.update( {xW: x_w, xWOE: woe.astype(float)} )

        return pDF(dt)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → WOE → ENCODER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class TransUtils:
    """
    Utility helpers for the WOE transformer public API.

    Handles feature list resolution, type assignment, and subsampling
    for statistical and memory efficiency.
    """
    def __init__(self, target:str):
        self.target = target

    def get_fes(self, data:pDF, num:list, cat:list) -> tuple[list, list]:
        if num is None and cat is None:
            return [x for x in data.columns if x != self.target], []  # default setup
        return num or [], cat or []
    
    def subsample(self, data:pDF) -> pDF:
        return Subsampling(data, data[self.target]).woe()[0]
            
    def get_data(self,
        data:   pDF,         
        num:    list = None, 
        cat:    list = None,
    ) -> tuple[pDF, dict]:
        """
        Prepare the data and type mapping for WOE fitting.

        Parameters
        ----------
        data : pDF
            Full dataset.
        num : list | None
            Numerical features.
        cat : list | None
            Categorical features.

        Returns
        -------
        tuple[pDF, dict]
            (subsampled_data, feature_type_mapping)
        """        
        num, cat = self.get_fes(data, num, cat)
        cols = [self.target] + num + cat
        
        fe_types = dict(zip(num+cat, [VarType.N]*len(num) + [VarType.C]*len(cat)))

        return self.subsample(data[cols]), fe_types
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class WoeTransformer(FitterModelBase[Fitter]):
    """
    WOE (Weight of Evidence) Transformer — Base Class.

    Transforms all encoded features into their WOE representation with
    bin pruning for OOT stability. This is the model-facing
    representation: each feature is expressed as a single numeric WOE
    value per observation, derived from its bin assignment.

    Parameters
    ----------
    target : str
        Target column name.
    gini_cutoff : float, default=0.02
        Minimum Gini for a feature to receive WOE encoding.
    alpha : float, default=0.05
        Statistical significance level for bin merge decisions.
    verbose : bool | Verbosity, default=True
        Enable progress and stats display.

    Notes
    -----
    **WOE Benefits**:
    - Unified scale across all features (comparable magnitudes).
    - Monotonic relationship with risk (higher WOE = higher event rate).
    - Approximately linear in logistic models (coefficients = risk direction).
    - Robust to scale changes and outliers.

    **Bin Pruning**: Prevents overfitting of bin structure to training
    noise. The two strategies (Gini / t-test) offer different
    bias-variance trade-offs — choose based on whether stability or
    signal preservation is the priority.

    **Missing Handling**: Missing values are treated as a distinct
    bin with its own WOE value, preserving their predictive signal.

    **Output Format**: Two columns per feature:
    - `{name}_W`: the bin label (categorical, for interpretability).
    - `{name}_WOE`: the WOE value (numeric, for model input).

    Examples
    --------
    >>> from aidflow.fitters import WoeTransformerGini
    >>> woe = WoeTransformerGini(target='is_churn', gini_cutoff=0.02)
    >>> woe.fit(data, num_features=['revenue_W', 'tenure_W'],
    ...         cat_features=['segment_W'])
    >>> encoded = woe.transform(data_new)
    >>> encoded.columns  # ['revenue_W', 'revenue_WOE', ...] 
    """
    _strategy:  str = None

    def __init__(self, *args, params: dict|FitParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(FitParams, params, args, kwargs)
        self._config = self.get_estimator_config(self._params, kwargs)

    def fit(self, data:pDF, num_features:list = None, cat_features:list = None):
        """
        Fit WOE encoding on all features.

        Parameters
        ----------
        data : pDF
            Training data (target + encoded features).
        num_features : list | None, default=None
            Numerical (already binned) features to WOE-encode.
            If None and cat_features is also None, all non-target
            columns are treated as numerical.
        cat_features : list | None, default=None
            Categorical features to WOE-encode.

        Returns
        -------
        self
            Fitted instance ready for `transform`.
        """
        dw, fe_types = TransUtils(self._params.target).get_data(
            data, num_features, cat_features
        )
        self._fitter = Fitter(self._params, self._strategy)
        self._fitter.fit(dw, fe_types)
        return self
    
    def transform(self, data:pDF) -> pDF:
        """
        Apply fitted WOE encodings to new data.

        Parameters
        ----------
        data : pDF
            New data containing the original (pre-WOE) features.

        Returns
        -------
        pDF
            WOE-encoded features (bin label + WOE value per feature).
        """
        return self.fitter.transform(data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class WoeTransformerTest(WoeTransformer):
    _strategy = STR.T

class WoeTransformerGini(WoeTransformer):
    _strategy = STR.G
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬