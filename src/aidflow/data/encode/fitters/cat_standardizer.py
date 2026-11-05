"""
Categorical Feature Encoding (Rare Category Merging + WOE Preparation).

This module implements the categorical encoding stage of the pipeline,
responsible for transforming raw categorical features into stable,
well-supported category groups optimized for predictive discrimination
against the target.

Methodology
-----------
Unlike numerical features (which require a bin search), categorical
features already have discrete values. The encoding here is a
**stability optimization** rather than a structure search:

1. **Predictive gate (original)**: The feature's raw Gini/IV against
   the target is measured. Features below `gini_cutoff` are dropped.
2. **Missing handling**: NaN/missing values are converted to a distinct
   'Missing' category, preserving their predictive signal in WOE.
3. **Min-category-share merge**: Categories with fewer than
   `min_bin_share` of observations are merged with their neighbours
   (by target response) to ensure statistical reliability.
4. **Predictive gate (encoded)**: The merged feature's Gini/IV is
   re-measured. If merging destroyed the signal, the feature is dropped.

Why Merge Rare Categories?
--------------------------
Categories with very few observations produce unstable WOE estimates
(high variance) and are prone to overfitting. Merging them into
neighbouring categories (by response behaviour) reduces variance while
preserving the overall discriminative structure. This is equivalent
to regularisation in the categorical space.

Output
------
Per feature:
- An encoding map (original category → grouped category).
- A performance record (original vs encoded Gini/IV).
- A binary verdict (kept / dropped) with reason.
"""

from ....deps.struct                import  dataclass
from ....deps.data                  import  np, pDF, pSe
from ....deps.types                 import  Literal

from ....core.base                  import  (
                                            FitterDataBase, 
                                            FitterModelBase
                                            )
from ....core.factory               import  (
                                            Dataclass,
                                            DictClass
                                            )
from ....core.foundation            import  AigParam
from ....core.accessor              import  Parallel
from ....core.interop               import  TypeStandard
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
    """Output column labels for categorical encoding stats and verdicts."""
    V  = 'VAR'
    R  = 'Reason'
    M  = VarStd.MISSING
    B  = 'Cats'
    GF = 'Gini_TF'
    GT = 'Gini_T'
    GO = 'Gini_O'
    IVF = 'IV_TF'
    IVT = 'IV_T'

class STAT(DictClass):
    """Stats output list selectors (for user-facing display)."""
    out = lambda x: f'stats_out_{x}'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → CAT → TRANSFORM
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class FitParamsBase:
    """
    Categorical encoding thresholds.

    Attributes
    ----------
    gini_cutoff : float, default=0.02
        Minimum Gini coefficient for a feature to be considered
        predictive. Applied at two gates:
        - **Prior**: original feature Gini must exceed this (else dropped).
        - **After**: encoded (MBS-merged) feature Gini must exceed this
          (else merging destroyed the signal → dropped).
    min_bin_share : int | float, default=0.03
        Minimum share of observations per category (3%). Categories
        below this threshold are merged with their neighbour (by target
        response) to ensure stable WOE estimates in production.
    """   
    gini_cutoff:   float = 0.02
    min_bin_share: int|float = 0.03

@Dataclass.base(update_specs=dict(slots=False))
class FitParamsCfg:
    """
    Pipeline configuration.

    Attributes
    ----------
    target : str
        Target column name. Encoding is optimized against this target.
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
    Fit output: encoding maps and feature verdicts.

    Attributes
    ----------
    encoding : dict
        Per-feature encoding record: `EncodingMap` containing the
        category mapping (original → grouped), type, and stats.
    vars_cat : list
        Feature names that passed all gates and were encoded.
    vars_dropped : list
        Feature names that failed a gate, with reason.
    vars_map : dict
        Mapping of original feature name → final encoded feature name
        (after min-share grouping).
    """
    encoding:       dict
    vars_cat:       list
    vars_dropped:   list
    vars_map:       dict

@Dataclass.base
class FitStats:
    """
    Diagnostic statistics for user-facing display.

    Attributes
    ----------
    fit_worker : dict
        Per-feature worker output (encoding table, encoded data,
        original data) for detailed inspection.
    stats_in : list
        Features that PASSED all gates (with performance metrics).
    stats_out_p : list
        Features DROPPED at the prior gate (insufficient original signal).
    stats_out_a : list
        Features DROPPED at the after gate (merging destroyed signal).
    """
    fit_worker:  dict
    stats_in:    list
    stats_out_p: list
    stats_out_a: list
    
@Dataclass.base
class WorkerResult:
    """
    Per-feature worker output (for detailed inspection).

    Attributes
    ----------
    encoding : pDF
        The category encoding table (original → grouped).
    encoded : pDF
        The data after MBS merging.
    original : pDF
        The original data (pre-encoding) for comparison.
    stats : pDF | None
        Additional diagnostics (unused for categorical, reserved).
    """
    encoding: pDF
    encoded:  pDF
    original: pDF
    stats:    pDF|None = None
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class FitWorker:
    """
    Categorical Encoding Worker: MBS Merging and Predictive Gating.

    Executes the per-feature pipeline:
    1. Measure original predictive power (Gini/IV).
    2. Gate: original signal must exceed `gini_cutoff`.
    3. Add 'Missing' as a distinct category.
    4. Apply min-category-share merging (rare → neighbour by response).
    5. Gate: encoded signal must still exceed `gini_cutoff`.
    6. Register encoding map or drop with reason.

    Unlike numerical discretization, there is no bin search — the
    categories are already discrete. The optimization is purely in
    the merging of rare categories to ensure WOE stability.
    """   
    def __init__(self, params:FitParams):
        self.prm        = params
        self.target     = params.target
        self.cutoff     = params.gini_cutoff
        self.min_share  = params.min_bin_share
        self.verbose    = params.verbose
        
        self.result: FitResult = Dataclass.create_default_instance(FitResult)
        self.stats:  FitStats  = Dataclass.create_default_instance(FitStats)
        
        self._diag  = SeparationMetrics(incl_miss=True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @dataclass(slots=True)
    class VarData:
        """Per-feature working state during the encoding pipeline."""        
        x:     str      # var in processing
        xG:    str      # var in processing naming standard Min Share Grouped
        x_t:   list     # target-var names
        df:    pDF      # data of x_t full length
        ds:    pDF      # data of non missing d_xt space
        miss:  float    # missing rate of d_xt

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
        Initialize the working state for a categorical feature.

        Splits the data into full (with missing) and non-missing
        subsets, and resolves the naming convention.

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
        xG    = VarStd.fe_g(x)
        x_t   = [x, self.target]
        d_xt  = data.loc[:, x_t].copy() # Including Missing
        space = d_xt.loc[( d_xt[x] != VarStd.MISSING ) & ( d_xt[x].notnull() )] 
        return self.VarData(
            x  = x,
            xG = xG,
            x_t  = x_t,
            df   = d_xt,
            ds   = space,
            miss = 1 - ( len(space) / len(d_xt) )
        )

    def update_var_dx_missing(self, var:VarData) -> None:
        """
        Add a 'Missing' category and impute NaN rows in the full data.

        Rows with NaN are assigned the 'Missing' label so they
        participate in WOE computation as a distinct group.
        """        
        x  = var.x
        df = var.df
        df[x] = TypeStandard.cat_fill_missing(df[x])  # Add New Missing category + impute by 'Missing'

    def get_enc_data(self, d: tuple[pDF,pSe] | pDF, x: str|None) -> tuple[pSe,pSe]:
        """
        Normalize the data input to a (target, feature) pair.

        Accepts either a pre-split tuple or a DataFrame with column
        selection, returning a consistent (target, x) pair.
        """        
        if isinstance(d, tuple):
            d_, x_ = d
            return d_[self.target], x_
        return d[self.target], d[x]   

    def encode_x(self, target:pSe, x:pSe) -> tuple[pSe, float]:
        """
        Encode a feature to WOE and compute its total IV.

        Parameters
        ----------
        target : pSe
            Binary target values.
        x : pSe
            Feature values (categorical).

        Returns
        -------
        tuple[pSe, float]
            WOE-encoded values and total IV.
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

        Computes Gini and IV on both non-missing and full data
        (missing treated as its own group).

        Returns
        -------
        Performance
            Original feature metrics.
        """        
        pf, po = self.calc_perf(var.df, var.ds, x = var.x)
        # NOTE abs has to be present at this step 
        # once encoded, Gini is always positive value by Encoding woe design        
        return self.Performance(
            gini_org  = abs(po['gini']),
            gini_full = abs(pf['gini']),
            iv_full   = pf['iv']
        )
    
    def update_perf(self, var:VarData, perf:Performance, xg:pSe) -> Performance:
        """
        Update performance with encoded (post-MBS) metrics.

        Computes Gini and IV for the merged feature on both
        non-missing and full data, appending to the original metrics.

        Parameters
        ----------
        var : VarData
            Feature working state.
        perf : Performance
            Original performance (to preserve).
        xg : pSe
            Merged (post-MBS) category values.

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
        else: reason = f'Transformation < TH {self.cutoff}'
        self.set_res_vars_out(var, perf, step, reason)        
        return False    
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def add_missing(self, var:VarData) -> None:
        """
        Handle missing values by converting NaN → 'Missing' category.

        After this step, all rows in `df` have a valid category
        (either an original value or the 'Missing' label).
        """
        self.update_var_dx_missing(var)

    def encode(self, var:VarData, perf:Performance) -> None:
        """
        Apply min-bin-share merging and register the final encoding.

        Categories with fewer than `min_bin_share` of observations
        are merged with their neighbour. The final mapping
        (original → merged) is registered in the result.

        If the encoded feature fails the after-gate (Gini below
        threshold), the feature is dropped.

        Parameters
        ----------
        var : VarData
            Feature working state.
        perf : Performance
            Original performance (to preserve in the final record).
        """
        x  = var.x
        xG = var.xG
        df = var.df

        th  = self.min_share
        mbs = Bins.min_bin_share(df, self.target, x, th, vtype='C', xG=xG) # MBS Fx to Optimize minimal share of bins
        map = mbs.mapping[[x, xG]]                                         # Final Transformed Var with MBS Treatment
        
        xg = Bins.transform_categorical_data(df, map, x, xG)
        perf = self.update_perf(var, perf, xg)

        if not self.is_predictive(var, perf, 'a'):
            return

        self.set_res_vars_in(var, perf, mbs)
        self.set_res_mapping(var, perf, map, mbs)        
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def set_res_mapping(self, var:VarData, perf:Performance, map:pDF, mbs:MinBinResult) -> None:
        """
        Register the final encoding for a passed feature.

        Stores:
        - The encoding map in `result.encoding`.
        - The final feature name in `result.vars_map`.
        - The worker diagnostics in `stats.fit_worker`.
        """        
        self.result.vars_map[var.x] = var.xG
        self.result.encoding[var.x] = EncodingMap(
            vars  = [var.x, var.xG], 
            type  = VarType.C,
            map   = map, 
            stats = perf
        )
        self.stats.fit_worker[var.x] = WorkerResult(
                encoding = mbs.mapping,
                encoded  = mbs.dt,
                original = mbs.do
        )

    def get_res_cats(self, var:VarData, mbs:MinBinResult=None) -> int:
        """
        Count final categories after MBS.

        Includes the 'Missing' category if the feature has any missing
        values (it counts as a category for WOE purposes).

        Parameters
        ----------
        var : VarData
            Feature working state.
        mbs : MinBinResult, optional
            MBS result. If None, counts original unique categories.

        Returns
        -------
        int
            Number of categories (post-MBS if mbs provided, else original).
        """        
        miss = 0 if var.miss else -1    # non-Missing present on Train should be considered
        if mbs is not None:
            return len(mbs.dt) + miss   # number of Bins after MBS by default w Missing Category          
        return len(var.df[var.x].unique())

    def stats_io(self, IO:list, x:str, miss:float, cats:int, reason:str, perf:Performance) -> None: 
        """
        Append a feature's verdict record to the stats output list.

        Parameters
        ----------
        IO : list
            Target stats list to append to.
        x : str
            Feature name.
        miss : float
            Missing rate.
        cats : int
            Number of categories (post-MBS or original).
        reason : str
            Verdict reason.
        perf : Performance
            Performance metrics.
        """        
        IO.append({
            LBL.V:   x,
            LBL.R:   reason,
            LBL.B:   cats,
            LBL.M:   miss,
            LBL.IVF: perf.iv_enc_f,
            LBL.IVT: perf.iv_enc_o,            
            LBL.GF:  perf.gini_enc_f,
            LBL.GT:  perf.gini_enc_o,
            LBL.GO:  perf.gini_org
        })

    def set_res_vars_in(self, var:VarData, perf:Performance, mbs:MinBinResult) -> None: 
        """Register a feature as PASSED (encoded and kept)."""
        self.result.vars_cat.append(var.xG)
        io      = self.stats.stats_in
        cats    = self.get_res_cats(var, mbs)
        reason  = f'Passed TH {self.cutoff}'
        self.stats_io(io, var.xG, var.miss, cats, reason, perf)

    def set_res_vars_out(self, 
        var:VarData, perf:Performance, step:Literal['p','a'], reason:str
    ) -> None: 
        """Register a feature as DROPPED (failed a gate)."""
        self.result.vars_dropped.append(var.x)
        io      = self.stats[STAT.out(step)]
        cats    = self.get_res_cats(var)
        reason  = 'Unpredictive → ' + reason
        self.stats_io(io, var.x, var.miss, cats, reason, perf)
    # ───────────────────────────────────────────────────────────────────────────────────────────── 

    def pipeline(self, data:pDF, cat:list, **kwargs) -> None:     
        """
        Execute the full per-feature encoding pipeline for all features.

        Per feature (in order):
        1. Create working state.
        2. Measure original performance.
        3. Gate: original predictive power.
        4. Handle missing (NaN → 'Missing' category).
        5. Apply MBS + gate: encoded predictive power.
        6. Register encoding (or drop).

        Parameters
        ----------
        data : pDF
            Full dataset (feature + target).
        cat : list
            Feature names to process.
        kwargs : dict
            Forwarded to progress display.
        """
        Nloop, progress = Display.processing_progress(
            cat, kwargs, self.verbose, desc='Categorical Features Transformation'
        )
        for x in Nloop:
            Display.update_progress(progress)

            var  = self.create_var(data, x)
            perf = self.org_perf(var)

            if not self.is_predictive(var, perf):
                continue
    
            self.add_missing(var)
            self.encode(var, perf)
    # ───────────────────────────────────────────────────────────────────────────────────────────── 
        
    def process(self, data:pDF, num:list, **kwargs) -> tuple[FitParams, FitStats]:     
        """
        Entry point: execute the encoding pipeline and return results.

        Parameters
        ----------
        data : pDF
            Full dataset.
        cat : list
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
    Categorical Feature Encoding — execution backend.

    Orchestrates the fit-transform contract for categorical MBS encoding:

    - **fit**: Apply predictive gates, missing handling, and MBS merging
      per feature. Store encoding maps.
    - **transform**: Apply the learned category mappings to new data,
      producing merged categorical features.

    Parameters
    ----------
    params : FitParams
        Encoding thresholds and pipeline configuration.
    """

    def __init__(self, params:FitParams):
        self.prm = params 

    def _parallel_th(self, data:pDF, cat:list):
        d = data[cat]
        m = d.describe().T['unique'].mean()
        p_th = (
            ( len(data) > 5e4 and len(cat) > 20 ) or 
            ( len(cat)*(m) > 1e3 )
        )
        return 0 if p_th else np.inf

    def fit(self, data:pDF, cat:list):
        """
        Fit the categorical encoding on all features.

        Executes the predictive gates, missing handling, and MBS
        merging for each feature. Results are stored as encoding
        maps ready for `transform`.

        Parameters
        ----------
        data : pDF
            Training data (must contain the target column and
            the categorical features).
        cat : list
            Categorical feature names to encode.

        Returns
        -------
        self
            Fitted instance with `encoding` and `features` set.
        """
        p_th   = self._parallel_th(data, cat)
        worker = FitWorker(self.prm)
        p_fit  = Parallel(
            data_and_fe = 1, 
            th          = p_th, 
            kwargs_pos  = True, 
            verbose     = self.prm.verbose,
            spin_title  = 'Categorical transformation in progress'
        ).process(worker.process)
        result, stats = p_fit(data, cat)
        
        self.result: FitResult = result
        self.stats:   FitStats = stats

        self.encoding = self.result.encoding
        self.features = self.result.vars_cat
        
        return self
   
    def transform(self, data:pDF) -> pDF:
        """
        Apply fitted category mappings to new data.

        Maps each categorical feature to its merged (post-MBS) category
        values using the learned encoding maps.

        Parameters
        ----------
        data : pDF
            New data containing the original categorical features.

        Returns
        -------
        pDF
            DataFrame with one column per encoded feature (merged
            categories, including 'Missing' where applicable).
        """     
        dt = {}
        for cat in data.columns:
            if cat not in self.encoding:     # removed variables during fit - not predictive
                continue                            
            fitted:EncodingMap = self.encoding[cat]
            x, xG   = fitted.vars
            mapping = fitted.map
            XG = Bins.transform_categorical_data(data, mapping, cat, xG)
            dt.update( {xG: XG} )

        return pDF(dt)        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → CAT → ENCODER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class CategoricalEncoder(FitterModelBase[Fitter]):
    """
    Categorical Feature Encoding (Min-Bin-Share Merging).

    Transforms categorical features into stable, low-cardinality
    encoded features by merging rare categories. The resulting
    features are designed for OOT stability and feed into the WOE
    encoding step.

    Parameters
    ----------
    target : str
        Target column name. Encoding is optimized against this target.
    gini_cutoff : float, default=0.02
        Minimum Gini for a feature to be kept. Applied at both the
        original and encoded gates.
    min_bin_share : int | float, default=0.03
        Minimum observation share per category. Rarer categories
        are merged with their neighbour.
    verbose : bool | Verbosity, default=True
        Enable progress and stats display.

    Notes
    -----
    **Min-Bin-Share Merge**: Rare categories (below the share
    threshold) are merged with their neighbour to ensure each group
    has sufficient sample size for stable WOE computation. This
    reduces encoding variance without significant loss of signal.

    **Dual Gate**: Features must pass a Gini threshold both before
    (original signal) and after (merged signal) the MBS step. This
    prevents the merge from destroying discriminative power.

    **Missing as a Category**: Missing values are treated as a distinct
    category, preserving their predictive signal in the WOE computation.

    **Subsampling**: For large datasets, a stratified subsample is
    used for the encoding fit (statistical and memory efficiency).

    Examples
    --------
    >>> from aidflow.fitters import CategoricalEncoder
    >>> cat_enc = CategoricalEncoder(
    ...     target='is_churn',
    ...     gini_cutoff=0.02,
    ...     min_bin_share=0.03,
    ... )
    >>> cat_enc.fit(data, cat_features=['region', 'segment', 'channel'])
    >>> encoded = cat_enc.transform(data_new)
    >>> encoded.columns  # ['region_G', 'segment_G', 'channel_G', ...]
    """

    def __init__(self, *args, params: dict|FitParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(FitParams, params, args, kwargs)

    def _get_cat(self, data:pDF, cat:list|None) -> list:
        if cat is not None:
            return cat
        return [x for x in data.columns if x!=self._params.target]
    
    def _subsample(self, data:pDF, cat:list) -> pDF:
        """
        Stratified subsample for efficient encoding fit.

        Reduces computation cost for large datasets while preserving
        the target distribution (stratified by target value).
        """
        t = self._params.target
        return Subsampling(data[cat + [t]], data[t]).univar()[0]
        
    def fit(self, data:pDF, cat_features:list = None):        
        """
        Fit the categorical encoding on all features.

        Parameters
        ----------
        data : pDF
            Training data (must contain target + categorical features).
        cat_features : list | None, default=None
            Specific features to encode. If None, all non-target
            columns are used.

        Returns
        -------
        self
            Fitted instance ready for `transform`.
        """       
        cat   = self._get_cat(data, cat_features)
        data_ = self._subsample(data, cat)
        self._fitter = Fitter(self._params)
        self._fitter.fit(data_, cat)
        return self
    
    def transform(self, data:pDF) -> pDF:
        """
        Apply fitted category mappings to new data.

        Parameters
        ----------
        data : pDF
            New data containing the original categorical features.

        Returns
        -------
        pDF
            Merged categorical features (post-MBS), ready for
            WOE encoding.
        """
        return self.fitter.transform(data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬