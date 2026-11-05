"""
Time Stationarity Screening for Feature Pre-Selection.

This module implements the temporal stability analysis stage of the pipeline,
responsible for screening features based on time-series stationarity before
any downstream modelling. It performs the following:

1. Transforms raw data (numerical and categorical) into time-series form.
2. Applies the Augmented Dickey-Fuller (ADF) unit-root test per feature.
3. Classifies each feature as STABLE, UNSTABLE, or UNASSESSABLE (insufficient
   observations).

Design Principles
-----------------
- **Verifiability gate**: Features with fewer than 60 non-NaN observations
  are classified as UNASSESSABLE (not "unstable"). At n < 60 the ADF test
  statistic has not converged to its asymptotic distribution (validated via
  Monte Carlo simulation), so no reliable verdict can be issued.
- **P-value based**: The stationarity threshold is a significance level
  (default α = 0.10). A feature is stationary when p < α (reject unit root).
- **Adaptive maxlag**: The ADF maxlag is computed as logₙ(1000/n, n) for
  n ≤ 700, capped at 20 for longer series, following the modified
  Schwarz formula adapted for finite samples.
- **Categorical proxy**: Categorical features are reduced to the temporal
  share of their dominant category set (cumulative share ≥ 50%), producing
  a continuous proxy suitable for ADF testing.
"""

from ....deps.data                  import  np, pDF, pSe
from ....deps.ml.stats              import  adfuller

from ....core.factory               import  (
                                            Dataclass, 
                                            DictClass
                                            )
from ....core.base                  import  (
                                            FitterDataBase,
                                            FitterModelBase
                                            )
from ....core.foundation            import  AigParam
from ....core.accessor              import  Parallel
from ....core.interop               import  Pd, GetValue
from ....core.display               import  Display

from ....display.verbosity          import  Verbosity
from ....ml.sampling                import  (
                                            Sampling, 
                                            SubsamplingSpec
                                            )


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    """var-ur-stab-type-graph-tt labels"""
    VAR   = 'Feature'
    UR    = 'DF-Test'
    STAB  = 'Stable'
    TYPE  = 'Type'
    GRAPH = 'Graph_Name'
    TT    = ('TRAIN', 'TEST')
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → TIME SERIES → ANALYTICAL DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class DataToTimeSeries:
    """
    Transform tabular data into time-series representation per feature type.

    Numerical features are aggregated (mean) per time period.
    Categorical features are reduced to a continuous proxy representing the
    temporal share of the dominant category set (smallest subset of categories
    whose cumulative frequency ≥ 50%). This proxy captures the "regime" of the
    categorical variable over time, making it amenable to ADF testing.

    Parameters
    ----------
    time_var : str
        Name of the column representing the time period (e.g., 'AIG_YY_MM').
    verbose : bool
        Enable progress display during categorical transformation.
    """
    
    def __init__(self, time_var:str, verbose:bool=True):
        self.time_var = time_var
        self.verbose  = verbose

    def ts_index(self, data:pDF, ts_index:pDF=None) -> pDF:
        """Data Time Series index."""
        if ts_index is None:
            return pDF(index=data[self.time_var].unique())
        return ts_index.copy()

    def num_to_ts(self, data:pDF, num:list, ts_index:pDF=None) -> pDF:
        """Aggregate numerical features to time-series form (mean per period)."""
        if not num:
            return self.ts_index(data, ts_index)
        return data[[self.time_var] + num].groupby(self.time_var).mean()
                    
    def cat_to_ts(self, data:pDF, cat:list, ts_index:pDF=None) -> pDF:
        """
        Transform categorical features to time-series proxy form.

        Each categorical column is reduced to the temporal share of its
        dominant category set (smallest subset with cumulative frequency ≥ 50%),
        yielding a continuous series in [0, 1] suitable for ADF testing.
        """        
        desc    =' Categorical Data TS Transformation'
        loop, _ = Display.processing_progress(cat, verbose=self.verbose, desc=desc)        
        ts_cat  = self.ts_index(data, ts_index)
        for x in loop:                                                
            ts_cat[x] = self._cat_to_ts(data, x)
        return ts_cat  
        
    def _cat_to_ts(self, data:pDF, x:str) -> pSe:
        """
        Compute the dominant-category proxy for a single categorical feature.

        The proxy is defined as:
            share_t = count(dominant_cats at period t) / count(all obs at period t)

        where dominant_cats is the smallest subset of categories whose
        cumulative frequency across the full sample ≥ 50%.
        """        
        cat:pSe = data[x]
        if not cat.count():
            return np.nan        
        cat_d = data[[self.time_var, x]]            
        dist  = cat.value_counts(normalize=True)                                # Distribution of categories
        proxy = [dist[:c+1] for c in range(len(dist)) if sum(dist[:c])<0.5][-1] # New category with share >0.5 as aggregated category
        proxy = cat_d[cat.isin(proxy.index)]                                    # New proxy ts category        
        return ( 
            proxy.groupby(self.time_var)[x].count() /                           # Time evolution of aggregated variable
            cat_d.groupby(self.time_var)[x].count() 
        )           
    
    def process(self, data:pDF, num:list, cat:list) -> tuple[pDF, pDF, pDF]:
        """
        Execute full data-to-time-series transformation.

        Parameters
        ----------
        data : pDF
            Source data containing the time column and feature columns.
        num : list
            Numerical feature column names.
        cat : list
            Categorical feature column names.

        Returns
        -------
        tuple[pDF, pDF, pDF]
            ts_all : Combined time series (numerical + categorical proxies).
            ts_num : Numerical features only.
            ts_cat : Categorical proxies only.
        """          
        ts_ix  = self.ts_index(data)
        ts_num = self.num_to_ts(data, num, ts_ix)
        ts_cat = self.cat_to_ts(data, cat, ts_ix)     
        ts_all = Pd.concat([ts_num, ts_cat], axis=1)
        
        return ts_all, ts_num, ts_cat
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class TimeStationarity:
    """
    Dickey-Fuller Stationarity Screening.

    Applies the Augmented Dickey-Fuller (ADF) unit-root test to each feature's
    time series and classifies it as STABLE, UNSTABLE, or UNASSESSABLE.

    Classification Logic
    --------------------
    - **STABLE** ('Y'): ADF p-value < α (default 0.10). The null hypothesis
      of a unit root is rejected; the series is stationary.
    - **UNSTABLE** ('N'): ADF p-value ≥ α. The null is not rejected; the
      series is treated as non-stationary.
    - **UNASSESSABLE** ('NaN'): Fewer than `MIN_LENGTH` (60) non-NaN
      observations. The ADF test has insufficient power at this sample size
      (validated via Monte Carlo simulation), so no verdict is issued.

    Parameters
    ----------
    th_ts : float
        Significance level α for the ADF test (default 0.10).
        A feature is stationary when p < α.
    verbose : bool
        Enable progress display during testing.

    Notes
    -----
    - **Adaptive maxlag**: For series length n ≤ 700, maxlag = ⌊logₙ(1000/n, n)⌋.
      For n > 700, maxlag is capped at `MAX_LAG` (20). This follows the
      standard lag-selection heuristic adapted for finite samples.
    - **Two regressions**: Each feature is tested with constant-only ('c')
      and constant+trend ('ct') detrending. The maximum p-value is reported
      (conservative — requires stationarity under both specifications).
    - **Parallelism**: For large datasets (>200 features, >1000 periods),
      workers are parallelised via process pool.
    """      

    DF_DEFAULT = -1 
    MIN_LENGTH = 60
    MAX_LAG = 20
    
    NUM = 'NUM'
    CAT = 'CAT'

    Y  = 'Y'
    N  = 'N'
    NA = 'NaN'

    def __init__(self, 
        th_ts:   float = 0.1, 
        verbose: bool  = True
    ):
        self.train_order_map = None
        self.is_update = False
        self.th_ts = th_ts
        self.verbose = verbose and not self.is_update
    
    @classmethod
    def _df_maxlag(cls, ts:pSe) -> int:
        """
        Compute ADF maximum lag adaptively based on series length.

        Uses the modified Schwarz formula: maxlag = ⌊logₙ(1000/n, n)⌋ for n ≤ 700,
        capped at `MAX_LAG` (20) for longer series to avoid over-differencing.

        Parameters
        ----------
        ts : pSe
            Time series (non-NaN values).

        Returns
        -------
        int
            Maximum lag for the ADF regression.
        """        
        length = len(ts)
        if length > 700:
            return cls.MAX_LAG
        return round(np.emath.logn(1e3/length, length))     

    @classmethod
    def _df_test(cls, ts:pSe, reg:str) -> float|int:
        """Execute ADF test and return the p-value."""        
        try:    
            return adfuller(
                ts, regression = reg, maxlag = cls._df_maxlag(ts), result_object = True
            )[1]
        except: 
            return cls.DF_DEFAULT

    @classmethod
    def df_test(cls, ts_var:pSe) -> tuple[float, float]:
        """
        Run the full Dickey-Fuller stationarity assessment for one feature.

        Performs two ADF tests (constant-only and constant+trend) and returns
        both p-values. If the feature has fewer than `MIN_LENGTH` (60)
        non-NaN observations, returns (DF_DEFAULT, DF_DEFAULT) to indicate
        the feature is UNASSESSABLE.

        Parameters
        ----------
        ts_var : pSe
            Feature values in time-series form (may contain NaN).

        Returns
        -------
        tuple[float, float]
            (p_value_constant, p_value_constant_trend).
            Both are -1 (DF_DEFAULT) if the series is too short to assess.
        """          
        ts  = ts_var[ts_var.notnull()]
        
        if not len(ts) > cls.MIN_LENGTH:                 
            return (cls.DF_DEFAULT,)*2
            
        return cls._df_test(ts,'c'), cls._df_test(ts,'ct')
    
    @staticmethod
    def is_stationary(unit_root_test:float, th:float) -> bool:
        """
        Determine stationarity from ADF p-value.

        A feature is stationary when 0 ≤ p-value < α (significance level).
        This rejects the null hypothesis of a unit root.

        Parameters
        ----------
        unit_root_test : float
            ADF p-value (or DF_DEFAULT = -1 for unassessable).
        th : float
            Significance level α (e.g., 0.10).

        Returns
        -------
        bool
            True if the series is stationary at level α.
        """        
        return bool(0 <= unit_root_test < th)
    
    @classmethod
    def set_status(cls, unit_root_test:float, is_stationary:bool) -> str:
        """
        Map ADF result to a categorical status label.

        Returns
        -------
        str
            'NaN' — unassessable (insufficient observations)
            'Y'   — stationary
            'N'   — non-stationary
        """        
        if unit_root_test == cls.DF_DEFAULT: return cls.NA
        if is_stationary : return cls.Y
        return cls.N
    
    @staticmethod
    def _get_vars(ts_stats:pDF):
        return list(ts_stats[LBL.VAR])
    
    @classmethod
    @AigParam.verify_data_arg(pDF())
    def get_status_stable(cls, ts_stats:pDF, return_list_or_pDF=0) -> list|pDF:
        """
        Retrieve features classified as STABLE.

        Parameters
        ----------
        ts_stats : pDF
            Stationarity stats output.
        return_list_or_pDF : int
            0 → return list of names. 1 → return the filtered DataFrame.

        Returns
        -------
        list | pDF
            Feature names (or full rows) with status 'Y'.
        """        
        st = ts_stats[ts_stats[LBL.STAB]==cls.Y]
        if return_list_or_pDF: 
            return st
        return cls._get_vars(st)

    @classmethod
    @AigParam.verify_data_arg(pDF())
    def get_status_unstable(cls, ts_stats:pDF, return_list_or_pDF=0) -> list|pDF:
        """
        Retrieve features classified as UNSTABLE or UNASSESSABLE.

        This is the combined exclusion set used for default feature screening.

        Parameters
        ----------
        ts_stats : pDF
            Stationarity stats output.
        return_list_or_pDF : int
            0 → return list of names. 1 → return the filtered DataFrame.

        Returns
        -------
        list | pDF
            Feature names (or full rows) with status 'N' or 'NaN'.
        """        
        un = ts_stats[ts_stats[LBL.STAB].isin([cls.N, cls.NA])]
        if return_list_or_pDF: 
            return un
        return cls._get_vars(un)        
    
    @classmethod
    @AigParam.verify_data_arg(pDF())
    def get_status_nan(cls, ts_stats:pDF, return_list_or_pDF=0) -> list|pDF:
        """
        Retrieve features classified as UNASSESSABLE (insufficient data).

        Parameters
        ----------
        ts_stats : pDF
            Stationarity stats output.
        return_list_or_pDF : int
            0 → return list of names. 1 → return the filtered DataFrame.

        Returns
        -------
        list | pDF
            Feature names (or full rows) with status 'NaN'.
        """        
        na = ts_stats[ts_stats[LBL.STAB]==cls.NA]
        if return_list_or_pDF: 
            return na
        return cls._get_vars(na)

    @classmethod
    @AigParam.verify_data_arg([])
    def get_num_vars(cls, ts_stats:pDF) -> list:
        """Retrieve names of numerical features from stats."""        
        return cls._get_vars(ts_stats.loc[ts_stats[LBL.TYPE]==cls.NUM])

    @classmethod
    @AigParam.verify_data_arg([])
    def get_cat_vars(cls, ts_stats:pDF) -> list:
        """Retrieve names of categorical features from stats."""
        return cls._get_vars(ts_stats.loc[ts_stats[LBL.TYPE]==cls.CAT])
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def set_update_th(self, th:float) -> None:       
        self.th_ts = th
        self.is_update = True
    
    @GetValue.at_x_of_col(LBL.VAR, LBL.UR)
    def _get_ur_stat_fast(self, data:pDF, x:str) -> float:
        ... 

    def get_unit_root(self, data:pDF, x:str) -> float:
        """
        Retrieve or compute the ADF p-value for a feature.

        In update mode, reads from pre-computed stats. Otherwise, runs
        the full ADF test (both 'c' and 'ct' regressions) and returns
        the maximum p-value (conservative).

        Parameters
        ----------
        data : pDF
            Time-series data or pre-computed stats (in update mode).
        x : str
            Feature name.

        Returns
        -------
        float
            ADF p-value (rounded to 3 decimals), or -1 if unassessable.
        """        
        if self.is_update:    
            return self._get_ur_stat_fast(data, x)
        return round( max( self.df_test(data[x]) ), 3)
    
    def get_vars_types(self, data:pDF, types:dict) -> tuple[list, dict]:
        """
        Resolve feature list and type mapping.

        In update mode, reads from the pre-computed stats DataFrame.
        Otherwise, extracts column names from data and assigns types.

        Parameters
        ----------
        data : pDF
            Time-series data or pre-computed stats (in update mode).
        types : dict | None
            Mapping of feature name → 'NUM'/'CAT'. If None, all are NUM.

        Returns
        -------
        tuple[list, dict]
            (feature_names, type_mapping)
        """        
        if self.is_update:                                    
            return (
                data[LBL.VAR].to_list(), 
                data.set_index(LBL.VAR)[LBL.TYPE].to_dict()
            )
        X = list(data.columns)
        if types is None: 
            types = {x:self.NUM for x in X}
        return X, types
    
    def sort_by_unit_root(self, data:pDF) -> pDF:
        """Sort features by ADF p-value (ascending — most stable first)."""
        if self.is_update: 
            return data
        return data.sort_values(by=[LBL.UR], ascending=False).reset_index(drop=True) 
    
    def set_train_order(self, data:pDF) -> None:
        """
        Record the current feature order as the canonical 'train order'.

        Subsequent calls to `sort_by_train_order` will restore this
        ordering, ensuring consistency between screening and model stages.

        Parameters
        ----------
        data : pDF
            Stationarity stats in the desired canonical order.
        """          
        self.train_order_map = {x:i for i,x in enumerate(data[LBL.VAR])}

    def sort_by_train_order(self, data:pDF) -> pDF:
        """
        Restore the previously recorded train order.

        Parameters
        ----------
        data : pDF
            Stationarity stats to reorder.

        Returns
        -------
        pDF
            DataFrame in the canonical train order.
        """        
        if self.is_update:
            return data
        return data.sort_values(
            LBL.VAR, key=lambda x: x.map(self.train_order_map)
        ).reset_index(drop=True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def worker_progress(self, X, kwargs):
        return Display.processing_progress(
            X, kwargs, verbose = self.verbose, 
            desc = 'Time Series Stationarity Testing'
        )  
    
    def worker_data(self, data:pDF, x:str, type_:str) -> dict:
        """
        Compute stationarity verdict and display metadata for one feature.

        Parameters
        ----------
        data : pDF
            Time-series data.
        x : str
            Feature name.
        type_ : str
            Feature type ('NUM' or 'CAT').

        Returns
        -------
        dict
            Row for the stats DataFrame with keys from `LBL`.
        """
        ur = self.get_unit_root(data, x)                
        st = self.is_stationary(ur, self.th_ts)
        na = (ur == self.DF_DEFAULT)
        xt = x +' | '+ type_ + ' → '
        df = lambda u='': xt + f'{u}STABLE | DF:' + str(round(ur,2))
        return {
             LBL.VAR  : x
            ,LBL.UR   : ur
            ,LBL.STAB : self.set_status(ur, st)
            ,LBL.TYPE : type_
            ,LBL.GRAPH: xt + self.NA if na
                        else df() if st
                        else df('UN')
        }
              
    def worker(self, data:pDF, fes:list, types:dict, **kwargs) -> list[dict]: 
        """
        Process all features: compute ADF verdicts and build stats rows.

        Parameters
        ----------
        data : pDF
            Time-series data (one column per feature).
        fes : list
            Feature names to test.
        types : dict
            Feature name → type mapping.
        kwargs : dict
            Forwarded to progress display.

        Returns
        -------
        list[dict]
            One dict per feature, ready for DataFrame construction.
        """
        fes, progress = self.worker_progress(fes, kwargs)
        stat = []
        with GetValue.loop_scope(self):
            for x in fes:
                Display.update_progress(progress)
                stat.append(self.worker_data(data, x, types[x]))                
        return stat    

    def _p_th(self, data, fes) -> float:
        th = np.inf        
        if not self.is_update:                              # Parallel process → run only for init comp
            c  = len(fes) * len(data) > 2e5                 # 1. check data complexity
            th = 0 if c else th                             # 2. run in parallel (th=0) if complex data
        return th
    
    def run_worker(self, data:pDF, fes:list, types:dict) -> pDF:
        """
        Execute ADF testing across features with adaptive parallelism.

        Parameters
        ----------
        data : pDF
            Time-series data.
        fes : list
            Feature names.
        types : dict
            Feature type mapping.

        Returns
        -------
        pDF
            Stationarity stats DataFrame.
        """
        th = self._p_th(data, fes)
        worker = Parallel(
            data_and_fe = 1, 
            th          = th, 
            kwargs_pos  = True, 
            verbose     = self.verbose,
            spin_title  = 'Stationarity analysis in progress'        
        ).process(self.worker)          
        return pDF( worker(data, fes, types) )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @AigParam.verify_data_arg(pDF(columns=LBL.keys()))
    def process_data(self, data:pDF, types:dict=None) -> pDF:
        """
        Run full stationarity screening and return sorted stats.

        Parameters
        ----------
        data : pDF
            Time-series data or pre-computed stats (in update mode).
        types : dict, optional
            Feature type mapping.

        Returns
        -------
        pDF
            Sorted stationarity stats DataFrame.
        """        
        fes,types_ = self.get_vars_types(data, types)
        ts_stats   = self.run_worker(data, fes, types_)
        return self.sort_by_unit_root(ts_stats)
    
    @AigParam.verify_data_arg(pDF(columns=LBL.keys()))
    def set_data_order(self, ts_stats) -> pDF:        
        """
        Enforce canonical train order on stats (if previously recorded).

        Parameters
        ----------
        ts_stats : pDF
            Stationarity stats.

        Returns
        -------
        pDF
            Stats in canonical order.
        """  
        if self.train_order_map is None:
            self.set_train_order(ts_stats)
            return ts_stats
        return self.sort_by_train_order(ts_stats)        
        
    @AigParam.verify_data_arg(pDF(columns=LBL.keys()))
    def get_unstable(self, ts_stats:pDF) -> pDF:
        """
        Extract the combined exclusion set (UNSTABLE + UNASSESSABLE).

        This is the default feature drop-list returned to the pipeline.

        Parameters
        ----------
        ts_stats : pDF
            Stationarity stats.

        Returns
        -------
        pDF
            Excluded features (without the display/graph column).
        """       
        uns = self.get_status_unstable(ts_stats, 1)
        return uns.drop(LBL.GRAPH, axis=1)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def process(self, data:pDF, types:dict=None) -> tuple[pDF,pDF]:
        """
        Execute the full stationarity screening pipeline.

        Parameters
        ----------
        data : pDF
            Time-series data (or pre-computed stats in update mode).
        types : dict, optional
            Feature type mapping.

        Returns
        -------
        tuple[pDF, pDF]
            ts_stats : Full stationarity stats for all features.
            unstable : Features to exclude by default (UNSTABLE + UNASSESSABLE).
        """    
        ts_stats = self.process_data(data, types)        
        ts_stats = self.set_data_order(ts_stats)
        unstable = self.get_unstable(ts_stats)
        return ts_stats, unstable       
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class TAParamsBase:
    """TimeAnalysis params input design."""       
    th_stationarity: float = 0.1

@Dataclass.base(update_specs=dict(slots=False))
class TAParamsCfg:
    """TimeAnalysis params input design."""       
    time_var: str = 'AIG_YY_MM'
    verbose: bool|Verbosity = True

@Dataclass.base
class TAParams(TAParamsBase, TAParamsCfg):
    """TimeAnalysis params input design."""

@Dataclass.base
class TAResult:
    """TimeAnalysis output design."""
    ts_data:  pDF
    ts_stat:  pDF
    unstable: pDF
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Fitter(FitterDataBase):
    """
    Time Series Stationarity Screening — execution backend.

    Orchestrates the full screening pipeline:
    1. Transform raw data to time-series form (`DataToTimeSeries`).
    2. Apply ADF unit-root test per feature (`TimeStationarity`).
    3. Return stats + default exclusion set.

    Supports incremental threshold updates without re-running the ADF test
    (O(n) reclassification vs O(n × T) re-computation).
    """      

    def __init__(self, params: TAParams):
        self.prm = params
        self.dts = DataToTimeSeries(self.prm.time_var, self.prm.verbose)
        self.ts  = TimeStationarity(self.prm.th_stationarity, self.prm.verbose)
        
    def process(self, data:pDF, num_features:list=None, cat_features:list=None) -> TAResult:
        """
        Execute full stationarity screening on raw input data.

        Parameters
        ----------
        data : pDF
            Raw input data containing the time column and feature columns.
        num_features : list
            Column names of numerical features.
        cat_features : list
            Column names of categorical features.

        Returns
        -------
        TAResult
            Time-series data, stationarity stats, and default exclusion set.
            If data is None, returns empty structures with correct schemas.
        """
        if data is None:
            return TAResult(
                ts_data  = pDF(), 
                ts_stat  = pDF(columns=LBL.keys()), 
                unstable = pDF(columns=LBL.keys())
            )
        num = num_features or []
        cat = cat_features or []
        
        types = (
            {x: TimeStationarity.NUM for x in num} |
            {x: TimeStationarity.CAT for x in cat}
        )
        ts_data = self.dts.process(data, num, cat)[0]
        self.stats, unstable = self.ts.process(ts_data, types = types)
        return TAResult(
            ts_data  = ts_data, 
            ts_stat  = self.stats, 
            unstable = unstable
        )
    
    def update(self, ta_result:TAResult, th:float) -> TAResult:
        """
        Re-evaluate stationarity with a new significance level.

        Reuses pre-computed ADF p-values — only re-applies the threshold.
        This is significantly faster than a full re-run when the user
        adjusts the threshold interactively.

        Parameters
        ----------
        ta_result : TAResult
            Previous screening result (must contain computed stats).
        th : float
            New significance level α.

        Returns
        -------
        TAResult
            Updated stats and exclusion set under the new threshold.
        """
        self.ts.set_update_th(th)
        self.stats, unstable = self.ts.process(ta_result.ts_stat)
        return TAResult(
            ts_data  = ta_result.ts_data, 
            ts_stat  = self.stats, 
            unstable = unstable
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeatureTemporalAnalyzer(FitterModelBase[Fitter]):
    """
    Time Series Stationarity Screening for Feature Pre-Selection.

    Screens features based on temporal stability using the Augmented
    Dickey-Fuller unit-root test before any ML training begins. Features
    that are non-stationary or unassessable (insufficient observations)
    are flagged for exclusion by default.

    Parameters
    ----------
    th_stationarity : float, default=0.10
        Significance level α for the ADF test. A feature is classified as
        stationary when its ADF p-value < α. Lower values are more permissive
        (require stronger evidence of stationarity). Default 0.10 corresponds
        to a 10% significance level.

    time_var : str, default='AIG_YY_MM'
        Column name in the input data that represents the time period
        (e.g., 'AIG_YY_MM' for monthly data).

    verbose : bool | Verbosity, default=True
        Enable progress display and result output during processing.

    Methods
    -------
    process(data, num_features, cat_features)
        Run full stationarity screening.
    update(ta_result, th)
        Re-evaluate with a new threshold (fast, no re-computation).

    Notes
    -----
    **Verifiability Gate**: Features with fewer than 60 non-NaN observations
    are classified as UNASSESSABLE ('NaN') rather than UNSTABLE ('N').
    At n < 60, the ADF test statistic has not converged to its asymptotic
    distribution (validated via Monte Carlo simulation, p < 0.05), so no
    reliable stationarity verdict can be issued. This is a data-sufficiency
    gate, not a quality judgment — the feature may be perfectly stationary;
    we simply cannot verify it.

    **Categorical Features**: Categorical variables are reduced to a
    continuous proxy (temporal share of dominant category set) before
    ADF testing, enabling uniform treatment of all feature types.

    **Interactive Override**: All auto-decisions are reversible. The
    `update` method allows adjusting the significance level without
    re-running the ADF test, and the model stage can opt any feature
    back in regardless of its screening status.

    Examples
    --------
    >>> from aidflow.fitters import FeatureTemporalAnalyzer
    >>> analyzer = FeatureTemporalAnalyzer(
    ...     th_stationarity=0.10,
    ...     time_var='date_month'
    ... )
    >>> result = analyzer.process(data, num_features=['revenue', 'costs'],
    """
    
    def __init__(self, *args, params: dict|TAParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(TAParams, params, args, kwargs)
        self._fitter = Fitter(self._params)

    def _subsample(self, data:pDF, num:list, cat:list) -> pDF:
        cols = [self._params.time_var] + num + cat
        size = SubsamplingSpec(None).get_univar_spec()[0]
        return Sampling(size, random_state=42).subsample(data[cols])

    def process(self, data:pDF, num_features:list=None, cat_features:list=None) -> TAResult:
        """
        Execute full stationarity screening on input data.

        Parameters
        ----------
        data : pDF
            Input data containing the time column and feature columns.
        num_features : list, default=None
            Column names of numerical features to screen.
        cat_features : list, default=None
            Column names of categorical features to screen.

        Returns
        -------
        TAResult
            Contains:
            - `ts_data`: features in time-series form
            - `ts_stat`: per-feature ADF p-value, status, type
            - `unstable`: default exclusion set (UNSTABLE + UNASSESSABLE)
        """
        num = num_features or []
        cat = cat_features or []        
        df  = self._subsample(data, num, cat)
        return self._fitter.process(df, num, cat)
    
    def update(self, ta_result:TAResult, th:float) -> TAResult:
        """
        Re-evaluate stationarity with a new significance level.

        Reuses pre-computed ADF p-values — only re-classifies features
        under the new threshold. O(n) in features, no re-testing.

        Parameters
        ----------
        ta_result : TAResult
            Previous screening result.
        th : float
            New significance level α (e.g., 0.05 for stricter screening).

        Returns
        -------
        TAResult
            Updated stats and exclusion set.
        """
        return self._fitter.update(ta_result, th)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬