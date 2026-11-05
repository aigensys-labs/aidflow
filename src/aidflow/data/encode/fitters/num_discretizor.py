"""
Numerical Feature Discretization (Binning + WOE Encoding).

This module implements the numerical encoding stage of the pipeline,
responsible for transforming continuous numerical features into
categorical interval features (bins) optimized for predictive
discrimination against the target.

Methodology
-----------
A **non-greedy** binning search is performed per feature:

1. **Predictive gate (original)**: The feature's raw Gini/IV against
   the target is measured. Features below `gini_cutoff` are dropped
   (insufficient signal to justify encoding).
2. **Bin search**: An optimal number of bins is found via a statistical
   alpha-based stability search (not greedy tree splitting). This
   avoids overfitting bin boundaries to training noise.
3. **Binning**: The feature is cut into the discovered number of bins
   (quantile-based for continuous, value-based for low-cardinality).
4. **Min-bin-share merge**: Bins with fewer than `min_bin_share` of
   observations are merged with their neighbours to ensure statistical
   reliability in production.
5. **Predictive gate (encoded)**: The binned feature's Gini/IV is
   re-measured. If encoding destroyed the signal, the feature is dropped.

Why Non-Greedy?
---------------
Greedy algorithms (e.g., decision tree splitting) maximise impurity
reduction at each step, which captures training noise in bin boundaries.
The non-greedy approach finds the simplest bin structure that preserves
predictive power — lower variance, better OOT stability. This aligns
with the bias-variance trade-off: fewer free parameters → less
overfitting → more robust production performance.

Output
------
Per feature:
- An encoding map (original value → bin → grouped bin).
- A performance record (original vs encoded Gini/IV).
- A binary verdict (kept / dropped) with reason.
"""

from ....deps.struct                import  dataclass
from ....deps.data                  import  np, pDF, pSe
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
from ....core.interop               import  (
                                            Pd,
                                            Arrow,
                                            TypeStandard
                                            )
from ....core.standard              import  (
                                            VarType, 
                                            VarStd
                                            )
from ....core.display               import  Display

from ....ml.features                import  (
                                            Bins, 
                                            MinBinResult,
                                            EncodingMap
                                            )
from ....ml.metrics                 import  SeparationMetrics
from ....ml.sampling                import  Subsampling
from ....display.verbosity          import  Verbosity



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    """Output column labels for binning stats and verdicts."""    
    V   = 'VAR'
    R   = 'Reason'
    M   = VarStd.MISSING
    B   = 'Bins'
    GF  = 'Gini_TF'
    GT  = 'Gini_T'
    GO  = 'Gini_O'
    IVF = 'IV_TF'
    IVT = 'IV_T'
    S   = 'Bin_Search'

class STAT(DictClass):
    """Dash-Fitter vars connection design."""
    out = lambda x: f'stats_out_{x}'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → NUM → TRANSFORM
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class FitParamsBase:
    """
    Binning search parameters.

    Attributes
    ----------
    max_num_bins : int, default=20
        Maximum number of bins in the search space. The algorithm
        explores bin counts up to this limit to find the optimal
        (stable, predictive) structure.
    gini_cutoff : float, default=0.02
        Minimum Gini coefficient for a feature to be considered
        predictive. Applied at two gates:
        - **Prior**: original feature Gini must exceed this (else dropped).
        - **After**: encoded (binned) feature Gini must exceed this
          (else encoding destroyed the signal → dropped).
    alpha : float, default=0.003
        Statistical significance level for bin stability. Two adjacent
        bins are considered "the same" if their Gini difference is
        below alpha. This controls the granularity of the search —
        lower alpha allows finer bin structures.
    min_bin_share : int | float, default=0.03
        Minimum share of observations per bin (3%). Bins below this
        threshold are merged with their neighbour to ensure statistical
        reliability in production (avoids bins with too few samples
        for stable WOE computation).
    """     
    max_num_bins:   int   = 20
    gini_cutoff:    float = 0.02
    alpha:          float = 0.003
    min_bin_share:  int|float = 0.03

@Dataclass.base(update_specs=dict(slots=False))
class FitParamsCfg:
    """
    Pipeline configuration.

    Attributes
    ----------
    target : str
        Target column name. Binning is optimized against this target.
    discrete : int, default=10
        If a feature has ≤ this number of unique values, it is treated
        as "discrete" and binned by unique values (not quantile-based).
        This avoids forcing artificial splits on inherently categorical
        numerical columns (e.g., age groups, counts).
    verbose : bool | Verbosity, default=True
        Enable progress display and stats output.
    """     
    target:   str
    discrete: int = 10
    verbose:  bool|Verbosity = True

@Dataclass.base
class FitParams(FitParamsBase, FitParamsCfg):
    """Fitter params input design."""

@Dataclass.base
class FitResult:
    """
    Fit output: encoding maps and feature verdicts.

    Attributes
    ----------
    encoding : dict
        Per-feature encoding record: `EncodingMap` containing the
        bin mapping (original → bin → grouped bin), type, and stats.
    vars_num : list
        Feature names that passed all gates and were encoded.
    vars_dropped : list
        Feature names that failed a gate, with reason.
    vars_map : dict
        Mapping of original feature name → final encoded feature name
        (after min-bin-share grouping).
    bin_search : dict
        Per-feature bin search diagnostics (or 'No Binning Search'
        for discrete variables).
    """
    encoding:       dict    # Fitted Encoding
    vars_num:       list    # Encoded num vars
    vars_dropped:   list    # Removed num vars
    vars_map:       dict    # Map org num var - num fe
    bin_search:     dict    # Binning Optim Search

@Dataclass.base
class FitStats:
    """
    Diagnostic statistics for user-facing display.

    Attributes
    ----------
    fit_worker : dict
        Per-feature worker output (encoding table, encoded data,
        original data, search stats) for detailed inspection.
    stats_in : list
        Features that PASSED all gates (with performance metrics).
    stats_out_p : list
        Features DROPPED at the prior gate (insufficient original signal).
    stats_out_a : list
        Features DROPPED at the after gate (encoding destroyed signal).
    """
    fit_worker:  dict
    stats_in:    list
    stats_out_p: list   # out prior encoding
    stats_out_a: list   # out after encoding

@Dataclass.base
class WorkerResult:
    """
    Per-feature worker output (for detailed inspection).

    Attributes
    ----------
    encoding : pDF
        The bin encoding table (original value → bin → grouped bin).
    encoded : pDF
        The data after binning and min-bin-share grouping.
    original : pDF
        The original data (pre-encoding) for comparison.
    stats : pDF | None
        Bin search diagnostics (per-candidate-bin performance).
    """
    encoding: pDF    # Encoding Join 
    encoded:  pDF    # Encoded
    original: pDF    # Original 
    stats:    pDF|None = None # Encoding Stats
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class FitWorker:
    """
    Non-Greedy Binning Search and Encoding for a Single Numerical Feature.

    Executes the full per-feature pipeline:
    1. Measure original predictive power (Gini/IV).
    2. Determine bin strategy (discrete vs. continuous search).
    3. Search for optimal bin count (alpha-based stability).
    4. Bin the feature (qcut or value-based cut).
    5. Apply min-bin-share merging.
    6. Re-measure encoded predictive power.
    7. Register encoding or drop with reason.

    Strategy
    --------
    The goal is to find a **stable** (train/OOT) bin structure that
    preserves the feature's predictive power with minimal loss.

    By the bias-variance trade-off, fewer free parameters (fewer bins)
    means lower variance and less overfitting. However, too few bins
    destroy discriminative power (high bias). The non-greedy search
    finds the optimal balance without capturing training noise in
    bin boundaries.

    In contrast, greedy algorithms (e.g., decision tree splitting)
    maximise impurity reduction at each split, which tends to
    overfit bin boundaries to training-specific noise. This produces
    excellent train Gini but poor OOT performance.
    """

    def __init__(self, params:FitParams):
        self.prm        = params
        self.target     = params.target
        self.max_bins   = params.max_num_bins
        self.cutoff     = params.gini_cutoff
        self.alpha      = params.alpha
        self.min_share  = params.min_bin_share
        self.verbose    = params.verbose
        self.discrete   = params.discrete

        self.result: FitResult = Dataclass.create_default_instance(FitResult)
        self.stats:  FitStats  = Dataclass.create_default_instance(FitStats)

        self._diag  = SeparationMetrics(incl_miss=True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    @dataclass(slots=True)
    class VarData:
        """Per-feature working state during the binning pipeline."""
        x:     str      # var in processing
        xT:    str      # var in processing naming standard Encoded
        xG:    str      # var in processing naming standard Min Share Grouped
        x_t:   list     # target-var names
        df:    pDF      # data of x_t full length
        ds:    pDF      # data of non missing d_xt space
        miss:  float    # missing rate of d_xt

    @dataclass(slots=True)
    class BinSpace:
        """Bin strategy determination for the current feature."""
        unique: int     # Number of unique values in the feature
        cross:  bool    # True if discrete (use value-based cut, not qcut)
    
    @dataclass(slots=True)
    class SearchSpace:
        """Result of the non-greedy bin count search."""
        bins:   int = 0     # Optimal bin count found
        gini: float = 0     # Gini at the optimal bin count
        stats:  pDF = None  # Per-candidate-bin diagnostics

    @Dataclass.base
    class Performance:
        """
        Predictive performance metrics at each pipeline stage.

        Attributes
        ----------
        gini_org : float
            Original feature Gini (non-missing data only).
        gini_full : float
            Original feature Gini (full data, including missing as a group).
        iv_full : float
            Original feature IV (full data).
        gini_enc_o : float
            Encoded feature Gini (non-missing data only).
        gini_enc_f : float
            Encoded feature Gini (full data, including missing as a group).
        iv_enc_o : float
            Encoded feature IV (non-missing only).
        iv_enc_f : float
            Encoded feature IV (full data).
        """        
        gini_org:   float|int
        gini_full:  float|int
        iv_full:    float|int
        gini_enc_o: float|int = 0
        gini_enc_f: float|int = 0
        iv_enc_o:   float|int = 0
        iv_enc_f:   float|int = 0
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def create_var(self, data:pDF, x:str) -> VarData:
        """
        Initialize the working state for a feature.

        Splits the data into full (with missing) and non-missing
        (bin search) subsets, and resolves the naming conventions.

        Parameters
        ----------
        data : pDF
            Full dataset containing the feature and target.
        x : str
            Feature name.

        Returns
        -------
        VarData
            Working state with data subsets and naming.
        """        
        xT    = VarStd.fe_t(x)
        xG    = VarStd.fe_g(x)
        x_t   = [x, self.target]
        d_xt  = data.loc[:, x_t].copy()             # Including Missing
        space = d_xt.loc[d_xt[x].notnull()].copy()  # Excluding Missing → bin-search space
        return self.VarData(
            x    = x,
            xT   = xT,
            xG   = xG,
            x_t  = x_t,
            df   = d_xt,
            ds   = space,
            miss = 1 - ( len(space) / len(d_xt) )
        )
    
    def update_var_set_dxT(self, var:VarData, dt:pDF) -> None:
        """Set the binned values (xT) on both the non-missing and full data subsets."""
        xT = var.xT
        ds = var.ds
        df = var.df
        ds[xT] = dt
        df[xT] = dt

    def update_var_dxT_missing(self, var:VarData) -> None:
        """Add a 'Missing' category to xT and impute NaN rows in the full data."""
        xT = var.xT
        df = var.df
        df[xT] = TypeStandard.cat_fill_missing(df[xT])
  
    def bin_space(self, var:VarData) -> BinSpace:
        """
        Determine the binning strategy for this feature.

        A feature is "discrete" (cross=True) if its unique value count
        is ≤ `max_bins` or ≤ `discrete`. Discrete features use
        value-based cutting (each unique value is a candidate bin);
        continuous features use quantile-based cutting.

        Parameters
        ----------
        var : VarData
            Feature working state.

        Returns
        -------
        BinSpace
            Strategy: unique count and whether to use discrete mode.
        """                    
        unique = len(var.ds[var.x].unique())
        cross  = ( self.max_bins >= unique ) or ( unique <= self.discrete )
        return self.BinSpace(
            unique = unique,
            cross  = cross  # Discrete num variables
        )

    def run_bin_search(self, var:VarData, bins:BinSpace) -> SearchSpace:
        """
        Execute the non-greedy bin count search.

        For discrete features (cross=True), returns an empty search
        (bin count = unique value count, no search needed).

        For continuous features, runs the alpha-based stability search
        to find the optimal bin count.

        Parameters
        ----------
        var : VarData
            Feature working state.
        bins : BinSpace
            Bin strategy.

        Returns
        -------
        SearchSpace
            Optimal bin count, Gini at optimum, and diagnostics.
        """        
        search = self.SearchSpace()
        if bins.cross:
            return search
        try:
            t, x   = var.ds[self.target], var.ds[var.x]
            b,g,s  = Bins.num_bin_search(t, x, self.max_bins, self.alpha)
            search = self.SearchSpace(bins = b, gini = g, stats = s)
        except:
            pass
        return search
    
    def get_enc_data(self, d: tuple[pDF,pSe] | pDF, x: str|None) -> tuple[pSe,pSe]:
        """
        Normalize the data input to (target, feature) pair.

        Accepts either a pre-split tuple or a DataFrame with column
        selection, returning a consistent (target, x) pair.
        """        
        if isinstance(d, tuple):
            d_, x_ = d
            return d_[self.target], x_
        return d[self.target], d[x]  
    
    def encode_x(self, target:pSe, x:pSe) -> tuple[pSe, float]:
        """
        Normalize the data input to (target, feature) pair.

        Accepts either a pre-split tuple or a DataFrame with column
        selection, returning a consistent (target, x) pair.
        """        
        x_, woe, _ = self._diag.preprocess_x(target, x, ensure_enc=True, add_iv=True)
        return x_, self._diag.iv_total(woe)

    def calc_gini(self, target:pSe, x:pSe|None) -> float:
        """Compute the Gini coefficient of target separation by feature groups."""        
        return self._diag.gini(target, x)
    
    def calc_perf(self, *dx, x:str=None) -> list[dict[str, float]]:
        """
        Compute Gini and IV for one or more data subsets.

        Parameters
        ----------
        *dx : pDF | tuple
            Data inputs (each either a DataFrame or (df, series) pair).
        x : str, optional
            Column name for DataFrame inputs.

        Returns
        -------
        list[dict]
            One {'gini': ..., 'iv': ...} per input.
        """        
        r = []
        for d in dx:
            t,  x_ = self.get_enc_data(d, x)
            xe, iv = self.encode_x(t, x_)
            gini   = self.calc_gini(t, xe)
            r.append({'gini':gini, 'iv':iv})
        return r
    
    def org_perf(self, var:VarData) -> Performance:
        """
        Measure the original (unencoded) feature performance.

        Computes Gini on non-missing data and Gini+IV on full data
        (missing treated as its own group).

        Returns
        -------
        Performance
            Original feature metrics (gini_org, gini_full, iv_full).
        """        
        po = self.calc_gini(var.ds[self.target], var.ds[var.x])
        pf = self.calc_perf(var.df, x = var.x)[0]
        # NOTE abs has to be present at this step 
        # once encoded, Gini is always positive value by Encoding woe design
        return self.Performance(
            gini_org  = abs(po),
            gini_full = abs(pf['gini']),
            iv_full   = pf['iv']
        )
    
    def update_perf(self, var:VarData, perf:Performance, xg:pSe) -> Performance:
        """
        Update performance with encoded (binned + MBS) metrics.

        Computes Gini and IV for the grouped feature on both
        non-missing and full data, appending to the original metrics.

        Parameters
        ----------
        var : VarData
            Feature working state.
        perf : Performance
            Original performance (to preserve).
        xg : pSe
            Grouped (post-MBS) bin values.

        Returns
        -------
        Performance
            Full performance record (original + encoded).
        """        
        pf, po = self.calc_perf( (var.df, xg), (var.ds, xg[xg != VarStd.MISSING]) )

        return self.Performance(
            gini_org    = perf.gini_org,
            gini_full   = perf.gini_full,
            iv_full     = perf.iv_full,
            gini_enc_o  = po['gini'],
            gini_enc_f  = pf['gini'],
            iv_enc_o    = po['iv'],
            iv_enc_f    = pf['iv']
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    
    def is_predictive(self, var:VarData, perf:Performance, step:Literal['p','a']='p') -> bool:
        """
        Gate: check if the feature (or its encoding) meets the Gini threshold.

        Parameters
        ----------
        var : VarData
            Feature working state.
        perf : Performance
            Current performance metrics.
        step : 'p' | 'a'
            'p' = prior gate (original feature Gini).
            'a' = after gate (encoded feature Gini).

        Returns
        -------
        bool
            True if the feature passes the threshold. If False, the
            feature is registered as dropped with the reason.
        """        
        if step == 'p':
            g = perf.gini_full
        else: 
            g = perf.gini_enc_f
        if ( g > self.cutoff ):
            return True
        if step=='p':
            reason = f'Gini of Original Values < TH {self.cutoff}'
        else: 
            reason = f'Transformation < TH {self.cutoff}'
        self.set_res_vars_out(var, perf, step, reason)        
        return False    

    def has_transformation(self, var:VarData, perf:Performance, bins:BinSpace, search:SearchSpace) -> bool: 
        """
        Gate: verify that a valid bin structure was found.

        Fails if the search returned 0 bins (no stable structure found)
        and the feature is not discrete.

        Returns
        -------
        bool
            True if a valid bin structure exists.
        """        
        if search.bins or bins.cross:
            return True
        self.set_res_vars_out(var, perf, 'p', 'No Transformation Found')  
        return False
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def apply_binning(self, var:VarData, bins:BinSpace, search:SearchSpace) -> None:
        """
        Apply the binning cut to the feature.

        Discrete features: value-based cut (each unique value is a bin).
        Continuous features: quantile-based cut at the searched bin count.

        The result is stored as `xT` on both data subsets.

        Parameters
        ----------
        var : VarData
            Feature working state.
        bins : BinSpace
            Bin strategy (discrete vs. continuous).
        search : SearchSpace
            Optimal bin count (for continuous features).
        """
        x  = var.x
        ds = var.ds

        if bins.cross: # for var with low unique values and transformation decrement is higher than TH → use orig var
              dt = Pd.cut( ds[x], bins.unique, duplicates='drop')   # pd.cut →  focus to have similar length between bins (orig values) → ideal for discrete num variables                
        else: dt = Pd.qcut(ds[x], search.bins, duplicates='drop')   # Use Transformation to bins | note: uniq_org +2 → ensures that for high value concentration interval is not merged to another
        self.update_var_set_dxT(var, dt)
        self.res_bin_search(var, bins, search)

    def add_missing(self, var:VarData) -> None:
        """
        Add a 'Missing' category to the full data.

        Rows with NaN in the original feature are assigned the
        'Missing' bin label, so they participate in WOE computation
        as a distinct group.
        """
        self.update_var_dxT_missing(var)

    def encode(self, var:VarData, perf:Performance) -> None:
        """
        Apply min-bin-share merging and register the final encoding.

        Bins with fewer than `min_bin_share` of observations are
        merged with their neighbour. The final mapping (original →
        bin → grouped bin) is registered in the result.

        If the encoded feature fails the after-gate (Gini below
        threshold), the feature is dropped.

        Parameters
        ----------
        var : VarData
            Feature working state.
        perf : Performance
            Original performance (to preserve in the final record).
        """
        xT = var.xT
        xG = var.xG
        df = var.df

        th  = self.min_share
        mbs = Bins.min_bin_share(df, self.target, xT, th, xG=xG)    # MBS Fx to Optimize minimal share of bins
        map = mbs.mapping[[xT, xG]]                                 # Final encoded Var with MBS Treatment
        
        xg   = Bins.data_mapping(df, map, xT, xG)
        perf = self.update_perf(var, perf, xg)

        if not self.is_predictive(var, perf, 'a'):
            return

        self.set_res_vars_in(var, perf, mbs)
        self.set_res_encoding(var, perf, map, mbs)        
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def set_res_encoding(self, var:VarData, perf:Performance, map:pDF, mbs:MinBinResult) -> None:
        """
        Register the final encoding for a passed feature.

        Stores:
        - The encoding map in `result.encoding`.
        - The final feature name in `result.vars_map`.
        - The worker diagnostics in `stats.fit_worker`.
        """                 
        self.result.vars_map[var.x] = var.xG
        self.result.encoding[var.x]  = EncodingMap(
            vars  = [var.xT, var.xG],
            type  = VarType.N,
            map   = map, 
            stats = perf
        )
        self.stats.fit_worker[var.x] = WorkerResult(
                encoding = mbs.mapping,
                encoded  = mbs.dt,
                original = mbs.do,
                stats    = self.result.bin_search[var.x]
        )

    def res_bin_search(self, var:VarData, bins:BinSpace, search:SearchSpace):
        """Record bin search diagnostics (or 'No Search' for discrete)."""                
        na = 'No Binning Search - Discreate Values' 
        self.result.bin_search[var.x] = na if bins.cross else search.stats
    
    def get_res_bins(self, var:VarData, mbs:MinBinResult) -> int:
        """
        Count final bins after MBS.

        Includes the 'Missing' category if the feature has any missing
        values (it counts as a bin for WOE purposes).
        """        
        miss = 0 if var.miss else -1    # non-Missing present on Train should be considered
        return len(mbs.dt) + miss       # number of Bins after MBS by default w Missing Category        

    def stats_io(self, IO:list, x:str, miss:float, bins:int, reason:str, perf:Performance) -> None: 
        """Append a feature's verdict record to the stats output list."""
        IO.append({
            LBL.V:   x,
            LBL.R:   reason,
            LBL.B:   bins,
            LBL.M:   miss,
            LBL.IVF: perf.iv_enc_f,
            LBL.IVT: perf.iv_enc_o,
            LBL.GF:  perf.gini_enc_f,
            LBL.GT:  perf.gini_enc_o,
            LBL.GO:  perf.gini_org,
        })

    def set_res_vars_in(self, var:VarData, perf:Performance, mbs:MinBinResult) -> None: 
        """Register a feature as PASSED (encoded and kept)."""
        self.result.vars_num.append(var.xG)
        io      = self.stats.stats_in
        bins    = self.get_res_bins(var, mbs)
        reason  = f'Passed TH {self.cutoff}'
        self.stats_io(io, var.xG, var.miss, bins, reason, perf)

    def set_res_vars_out(self, 
        var:VarData, perf:Performance, step:Literal['p','a'], reason:str
    ) -> None: 
        """Register a feature as DROPPED (failed a gate)."""
        self.result.vars_dropped.append(var.x)
        io      = self.stats[STAT.out(step)]
        reason  = 'Unpredictive → ' + reason
        self.stats_io(io, var.x, var.miss, 0, reason, perf)
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def pipeline(self, data:pDF, num:list, **kwargs) -> None:     
        """
        Execute the full per-feature binning pipeline for all features.

        Per feature (in order):
        1. Create working state.
        2. Measure original performance.
        3. Determine bin strategy.
        4. Gate: original predictive power.
        5. Run bin search.
        6. Gate: valid transformation exists.
        7. Bin the feature.
        8. Handle missing.
        9. Apply MBS + gate: encoded predictive power.
        10. Register encoding (or drop).

        Parameters
        ----------
        data : pDF
            Full dataset (feature + target).
        num : list
            Feature names to process.
        kwargs : dict
            Forwarded to progress display.
        """
        Nloop, progress = Display.processing_progress(
            num, kwargs, self.verbose, desc='Numerical Features Transformation'
        )
        for x in Nloop:
            Display.update_progress(progress)

            var  = self.create_var(data, x)
            perf = self.org_perf(var)
            bins = self.bin_space(var)

            if not self.is_predictive(var, perf):
                continue

            search = self.run_bin_search(var, bins)

            if not self.has_transformation(var, perf, bins, search): 
                continue
    
            self.apply_binning(var, bins, search)
            self.add_missing(var)
            self.encode(var, perf)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def process(self, data:pDF, num:list, **kwargs) -> tuple[FitResult, FitStats]:
        """
        Entry point: execute the binning pipeline and return results.

        Parameters
        ----------
        data : pDF
            Full dataset.
        num : list
            Feature names.
        kwargs : dict
            Forwarded to pipeline.

        Returns
        -------
        tuple[FitResult, FitStats]
            Encoding maps + diagnostic stats.
        """        
        self.pipeline(data, num, **kwargs)
        return self.result, self.stats
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Fitter(FitterDataBase):
    """
    Numerical Discretization — execution backend.

    Orchestrates the fit-transform contract for numerical binning:

    - **fit**: Run the non-greedy bin search per feature, apply MBS,
      gate on predictive power, and store encoding maps.
    - **transform**: Apply the learned bin mappings to new data,
      producing categorical interval features.

    Parameters
    ----------
    params : FitParams
        Binning thresholds and pipeline configuration.
    """

    def __init__(self, params:FitParams):
        self.prm = params 

    def _parallel_th(self, data: pDF, num:list):
        p_th = (
            ( len(data) > 5e4 and len(num) > 5 ) or 
            ( len(num)*(self.prm.max_num_bins) > 1e3 )
        )
        return 0 if p_th else np.inf

    def fit(self, data:pDF, num:list):
        """
        Fit the binning model on all numerical features.

        Executes the non-greedy bin search, MBS merging, and
        predictive gates for each feature. Results are stored as
        encoding maps ready for `transform`.

        Parameters
        ----------
        data : pDF
            Training data (must contain the target column and
            the numerical features).
        num : list
            Numerical feature names to encode.

        Returns
        -------
        self
            Fitted instance with `encoding` and `features` set.
        """
        p_th   = self._parallel_th(data, num)
        worker = FitWorker(self.prm)
        p_fit  = Parallel(
            data_and_fe = 1, 
            th          = p_th, 
            kwargs_pos  = True, 
            verbose     = self.prm.verbose,
            spin_title  = 'Numerical transformation in progress'        
        ).process(worker.process)
        result, stats = p_fit(data, num)
        
        self.result: FitResult = result
        self.stats:   FitStats = stats

        self.encoding = self.result.encoding
        self.features = self.result.vars_num

        return self
   
    @Arrow.transition(contract = Arrow.Contract(safe_conversion=True))
    def transform(self, data:pDF) -> pDF:
        """
        Apply fitted bin encodings to new data.

        Maps each numerical feature to its binned (xT) and grouped
        (xG) categorical values using the learned encoding maps.

        Parameters
        ----------
        data : pDF
            New data containing the original numerical features.

        Returns
        -------
        pDF
            DataFrame with one column per encoded feature (grouped
            bin values, including 'Missing' category where applicable).
        """            
        # NOTE Arrow → np.select has hard cond → if cond.dtype.type is not np.bool   
        dt = {}
        for num in data.columns:
            if num not in self.encoding:
                continue                            
            fitted: EncodingMap  = self.encoding[num]
            xT, xG  = fitted.vars
            mapping = fitted.map
            XT = Bins.transform_numerical_data(data, mapping, num, xT)
            XG = Bins.data_mapping(XT, mapping, xT, xG) 
            dt.update( {xG: XG} )

        return pDF(dt)        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → NUM → ENCODER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class NumericalDiscretizor(FitterModelBase[Fitter]):
    """
    Numerical Feature Discretization (Non-Greedy Binning).

    Transforms continuous numerical features into categorical interval
    features using a non-greedy, stability-optimized bin search. The
    resulting bins are designed for OOT stability and feed into the
    WOE encoding step.

    Parameters
    ----------
    target : str
        Target column name. Binning is optimized against this target.
    max_num_bins : int, default=20
        Maximum bins in the search space.
    gini_cutoff : float, default=0.02
        Minimum Gini for a feature to be kept. Applied at both the
        original and encoded gates.
    alpha : float, default=0.003
        Statistical significance for bin stability (controls search
        granularity).
    min_bin_share : int | float, default=0.03
        Minimum observation share per bin. Smaller bins are merged.
    discrete : int, default=10
        Features with ≤ this unique values use value-based binning.
    verbose : bool | Verbosity, default=True
        Enable progress and stats display.

    Notes
    -----
    **Non-Greedy Strategy**: Unlike decision-tree-based binning (which
    captures training noise in boundaries), this approach finds the
    simplest stable bin structure that preserves predictive power.
    Result: lower variance, better OOT performance.

    **Dual Gate**: Features must pass a Gini threshold both before
    (original signal) and after (encoded signal) binning. This
    prevents encoding from destroying discriminative power.

    **Missing as a Bin**: Missing values are treated as a distinct
    category in the encoding, preserving their predictive signal
    (if any) in the WOE computation.

    **Subsampling**: For large datasets, a stratified subsample is
    used for the bin search (statistical and memory efficiency).
    The encoding is then applied to the full data.

    Examples
    --------
    >>> from aidflow.fitters import NumericalDiscretizor
    >>> num_enc = NumericalDiscretizor(
    ...     target='is_churn',
    ...     max_num_bins=20,
    ...     gini_cutoff=0.02,
    ...     min_bin_share=0.03,
    ... )
    >>> num_enc.fit(data, num_features=['revenue', 'tenure', 'interactions'])
    >>> encoded = num_enc.transform(data_new)
    >>> encoded.columns  # ['revenue_G', 'tenure_G', 'interactions_G', ...]
    """

    def __init__(self, *args, params: dict|FitParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(FitParams, params, args, kwargs)

    def _get_num(self, data:pDF, num:list|None) -> list:
        if num is not None:
            return num
        return [x for x in data.columns if x!=self._params.target]
        
    def _subsample(self, data:pDF, num:list) -> pDF:
        """
        Stratified subsample for efficient bin search.

        Reduces computation cost for large datasets while preserving
        the target distribution (stratified by target value).
        """        
        t = self._params.target
        return Subsampling(data[num + [t]], data[t]).univar()[0]
            
    def fit(self, data:pDF, num_features:list|None = None):        
        """
        Fit the binning model on all numerical features.

        Parameters
        ----------
        data : pDF
            Training data (must contain target + numerical features).
        num_features : list | None, default=None
            Specific features to encode. If None, all non-target
            columns are used.

        Returns
        -------
        self
            Fitted instance ready for `transform`.
        """      
        num   = self._get_num(data, num_features)
        data_ = self._subsample(data, num)
        self._fitter = Fitter(self._params)
        self._fitter.fit(data_, num)
        return self
    
    def transform(self, data:pDF) -> pDF:
        """
        Apply fitted bin encodings to new data.

        Parameters
        ----------
        data : pDF
            New data containing the original numerical features.

        Returns
        -------
        pDF
            Categorical interval features (grouped bins), ready for
            WOE encoding.
        """
        return self.fitter.transform(data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬