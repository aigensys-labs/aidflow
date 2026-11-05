"""
Feature Predictive Consistency engine for temporal stability assessment.

This module implements a rigorous statistical framework to evaluate whether 
a feature's predictive relationship with the target remains stable over time. 
By discretizing data into dynamic time units and analyzing the variance of 
bin-level target means, it distinguishes between truly predictive features 
and those that exhibit volatile behavior or "random walks" across different 
time horizons.
"""

from ...deps.data           import np, pd, pDF, pSe
from ...deps.plots          import go, make_subplots
from ...deps.types          import Literal, FuncType

from ...core.factory        import DictClass, Dataclass
from ...core.interop        import Pd, Dict
from ...core.display        import Display
from ...core.accessor       import Spinner
from ...core.standard       import TypeDef

from ...utils.builtins      import List
from ...display.plots       import Plot

from ..tests                import TestSampleDiff


class TU(DictClass):
    """Time Unit Resolution for stability test."""
    W = 'week'
    D = 'day'

class LBL(DictClass):
    """value - is_stable - data - time_unit - verdict - gmx - graph"""
    VALUE   = 'value'
    DATA    = 'data'
    IS      = 'is_stable'
    TU      = 'time_unit'
    VERDICT = 'verdict'
    GMX     = 'gmx'
    GRAPH   = 'graph'

class STAB(DictClass):
    """Stability state: Y - N - NA"""
    Y  = 'Y'
    N  = 'N'
    NA = 'NaN'

class DASH(DictClass):
    """data - fig - order"""
    DATA  = 'data'
    FIG   = 'fig'
    ORDER = 'order'

_TR = 'TR' # Target Ratio


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → WOE → CONSISTENCY TEST
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FeConsistency:
    """
    Engine for assessing the Temporal Predictive Stability of features.

    This class evaluates whether a feature's ability to separate the target 
    variable remains consistent across different time intervals. Unlike static 
    correlation tests, this engine analyzes the 'leverage' of each feature bin 
    over time; if a feature is highly predictive in one month but irrelevant 
    in the next, it is flagged as unstable and potentially overfitted.

    The assessment process involves dynamic time-binning, weighted target ratio 
    analysis (TR), and statistical denoising via t-tests to merge bins with 
    similar behavior. The final output is a consistency score where 1.0 represents 
    absolute temporal stability and 0.0 represents random walk behavior.

    Methodology
    -----------
    Consistency Test → How Stable is Prediction Power by bins (distribution) 
    of X over Time towards Target:
    - Test requires that tested X has separation power towards TARGET → if not 
      such bins' E(T) will behave like random walk around overall average E(T) in time

    - Test focuses on the fact that if bins' E(T) in time is too volatile (measure is 
      towards time unit's E(E(T)) → average in given time) then it is better to avoid 
      such variable due to possible overfit for certain time aspect rather than stable 
      contribution

    - Percentage shows consistency that Bins' Targets are consistently in position towards 
      Average Target across the Time

    - This methodology is in eseence WOE stability in Time → but in one overall number
    - e.g. - 0.8 means that on average in 80% cases bins' E(T) are consistently either 
             above or below average Target
           - thus prediction can be expected more accurate and stable 

    Intentionally, test doesn't assess how stable given bin's prediction is:
    - bin's E(T) can decrease over time significantly and still be above Target
    - this design choice comes from the nature of AIG flow - this is pre step of 
      data screening - and  the fact that real-world data are volitile, thus, we 
      don't want to eliminate features just by volatility of E(T) in bins
    
    NOTE:
    Further, more strict alternative of this algo would be to compute E(T) and std 
    of each bin and measure how many times each bin was out of its stability zone (p-val).
    Such approach would eliminate significantly more features - might be beneficial to
    add option for noise data patterns.

    Parameters
    ----------
    features : str|list
        Features to test stability.

    target : str
        Target variable to test predictive stability of a feature.

    time_var : str
        Time variable to test predictive stability over.

    c_test_th : float = 0.65
        Threshold below which feature is considered statistically unstable.

    t_test_alpha : float = 0.65
        t-test alpha value for merging statistically same bins.

    time_unit : Literal['week','day'] = None
        Time variable (time_var) grouping unit factor: week | day | None → auto.

    max_time_units : int = None
        Time variable (time_var) max groups - merging groups until reached; None → auto.      
    """

    def __init__(self, 
        features:   str|list, 
        target:     str, 
        time_var:   str,

        c_test_th:    float = 0.65,
        t_test_alpha: float = 0.05,

        time_unit: Literal['week','day'] = None, 
        max_time_units: int = None, 
    ):        
        self.target   = target
        self.time_var = time_var

        self.c_test_th    = c_test_th
        self.t_test_alpha = t_test_alpha

        self.time_unit      = time_unit
        self.max_time_units = max_time_units

        self.fes  = List.ensure(features)

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # Time Unit → Max Length 
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _time_bins(d:pSe, time_unit:str) -> pSe:
        """
        Returns isocalendar time variable to resolution of requested 
        time_unit for binning purpose.
        """
        c = d.dt.isocalendar()
        k = c.year.astype(str)
        f = lambda x: k + '-' + c[x].astype(str).str.zfill(2)                
        k: pSe = f('week')

        if time_unit == TU.D:  
            k = f('day')

        k.name = time_unit
        return k

    @staticmethod
    def _get_time_unit(k:pSe) -> str:
        return k.name
    
    @classmethod
    def _determine_time_unit(cls, data:pDF, yy_mm:str, time_unit:str|None, tu_max:int|None
    ) -> tuple[str,int]:
        """        
        Determines time unit and max time units based on data time resolution 
        and count, if tu / tu_max is None, otherwise return requested values.
        """ 
        d  = pd.to_datetime(data[yy_mm])
        tu = time_unit

        if tu is None:
            p = Pd.get_time_resolution(d)

            if p == 'D':
                tu = TU.W
            else: 
                tu = TU.D

        if tu_max is not None:
            return tu, tu_max

        t = cls._time_bins(d, tu)
        c = t.value_counts().min()

        if   c > 200: m = 30  # on average FE ~5bins → min 40 obs in each TU-bin
        elif c > 100: m = 20
        else:         m = 10

        return tu, m
        
    @classmethod
    def _apply_max_time_bins(cls, data:pDF, yy_mm:str, time_unit:str, tu_max:int) -> pSe:
        """Groups time units with respect to tu_max."""
        d = pd.to_datetime(data[yy_mm])
        k = cls._time_bins(d, time_unit)

        folds = []
        units = list(k.unique())
        ul = len(units)

        if ul <= tu_max:
            return k

        x  = int(ul/tu_max+1)
        xl = int(ul/x)
        r  = int(xl*x)
            
        folds = [units[i*x:(i+1)*x] for i in range(xl)]

        un = units[r:]
        ul = len(un)
        if ul:               
            if ul < x/2:
                  folds[-1].extend(un)
            else: folds.append(un)

        for F in folds:
            k.loc[k.isin(F)] = F[int(len(F)/2)]

        return k
    # ─────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # BIN → Weighted MEAN-STD
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _weighted_mean(D:pDF, groupby_var:str|list, mean_var:str):
        """ 
        Returns: weighted mean by OBS (count) for varmean over All TU by bins of X
        Pre-req: 'count' of OBS by Bins of groupby_var, 'mean' of mean_var
        """          
        D = D.copy()
        s, c = ['__sp__','count']
        D[s] = D[mean_var] * D[c]                                 # 1 Sum Product
        M = D.groupby(groupby_var, observed=True)[[s,c]].sum()    # 2 Sum over groupby_var
        M['mean'] = M[s] / M[c]                                   # 3 Weigted MEAN
        return M.drop(s, axis=1)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def _weighted_stats(cls, D:pDF, groupby_var:str, mean_var:str):    
        """ 
        Returns: weighted mean and std by OBS (count) for varmean over All TU by bins of X
        Pre-req: 'count' of OBS by Bins of groupby_var, 'mean' of mean_var
        """    
        if groupby_var not in D.index.names:
            D = D.reset_index().set_index(groupby_var)

        D = D.loc[:,[mean_var, 'count']].copy()
        D = D.loc[ (D[mean_var].notnull()) & (D['count'] != 0) ]    

        M = cls._weighted_mean(D, groupby_var, mean_var)
        
        c = 'count'
        S = ((D[mean_var] - M['mean'])**2) * D[c]       # 4 STD Sum Product of Variance Nominator     
        S = S.groupby(groupby_var, observed=True).sum() 
        M['std'] =  np.sqrt(S / M[c])                   # 5 Weigted populational STD over groupby_var
        
        return M[['mean','std','count']]
    # ─────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # T-Test → Time Consistency 
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _t_test(data:pDF, x:str, alpha:float=0.05):    
        # sort ! → crucial step to correctly join same bins for c-test evaluation
        D = data.sort_values('mean').reset_index(drop=True) 
        D['means_stat_diff'] = TestSampleDiff(alpha).welch_se(D,'std')[TestSampleDiff.RES] 
        # True - cannot be merged | False - stat. same

        # Merging Two stat same bins sitting next to each other (pair)
        # ─────────────────────────────────────────────────────────────────────────────────────────
        I  = D[~D.means_stat_diff].index[1:] -1         # -1 →First positions towards second → -1 goes to first as merge 
        M  = [[D.loc[i,x], D.loc[i+1,x]] for i in I]    # Merge of two stat same bins

        # Final → Check of neigbors → if next pair is same too then merge with previous pair
        # ─────────────────────────────────────────────────────────────────────────────────────────
        get_same = lambda M: [i for i in range(len(M)-1) if any(True for x in M[i] if x in M[i+1])]
        
        I = get_same(M)
        R = len(I) 
        while R > 0:
            m = I[0]
            M[m].extend(M[m+1])
            M.remove(M[m+1])
            I = get_same(M)
            R = len(I) 
            
        # Unique Lists
        MF = []
        for m in M:
            MF.append([m[0]]+list(set(m[1:])))  # m[0] → group origin → stat same with the rest
            
        return D, MF
    # ─────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # TEST → Helpers
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def _status(self, test_stat:float) -> str:
        """
        Feature status:
        - Y:  above stability TH
        - N:  below TH or lack of data variance
        - NA: implicit set as default - when no data point available
        """
        if test_stat is np.nan:
            return STAB.NA
        if test_stat > self.c_test_th: 
            return STAB.Y
        return STAB.N
    
    def _set_status(self, consistency:dict, test_value:float=None) -> None:
        """Sets variable status callable by `LBL.is_stable` with respect to its stat test value."""
        if test_value is None:
            test_value = consistency[LBL.VALUE]
        consistency[LBL.IS] = self._status(test_value)
    
    def _set_value(self, consistency:dict, test_value:float) -> None:
        """Sets variable status callable by `LBL.value` with respect to its stat test value."""
        consistency[LBL.VALUE] = test_value

    def _set_data(self, consistency:dict, test:pDF) -> None:
        """Sets variable status callable by `LBL.data` with respect to its stat test value."""
        consistency[LBL.DATA]  = test

    def _set_defaults(self, time_unit:str) -> dict:
        """Sets `consistency result default vector` values."""
        return {LBL.IS:STAB.NA, LBL.VALUE:np.nan, LBL.DATA:None, LBL.TU: time_unit}
    # ─────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # (A) TEST → Compute Main TS Stats
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def _calc_stats(self, 
        data: pDF,
        time_bins: pSe,
        x: str = None,         
        calc_type: Literal[1,2] = 2,        
        apply_binning: bool = False, 
        bins: int = 10,    
    ) -> pDF:
        """Calulates target lift and obs by feature bins and Time Unit."""
        T  = self.target
        K  = self._get_time_unit(time_bins)
        dt = data.loc[:,[T, x]].copy()        
        dt[K] = time_bins

        # Variable Binning
        if apply_binning and ( dt[x].dtype in TypeDef.numeric ):
            if np.unique(dt[x]) < 20:
                  fx = Pd.cut
            else: fx = Pd.qcut
            dt[x] = fx(dt[x], bins, duplicates='drop').astype(str)  # type str → includes Missing category as 'nan'

        # Data prep 
        # - TG → Stat by Bins in TU 
        # - TT → Stat by TU
        tg = dt[[x, K, T]].groupby([K, x], observed=True)[T]
        tt = dt[[x, K, T]].groupby([K],    observed=True)[T]

        # NOTE V1 / V2 provides exact same result → choice is rather performance-wise

        # (V1) Full data length
        # ───────────────────────────────────────────────────────────────────────────
        if ( calc_type == 1 ):
            
            # T-Test Test Statistic → TR → Target Ratio as Variable Time Stability Check
            TGa = tg.transform('mean')      # By Time Unit and Bins → Target Mean
            TTa = tt.transform('mean')      # By Time Unit → Target Mean
            dt[_TR] = TGa / TTa - 1         # TR → Target Stability in Time by Bins → Prediction Stability Check
                        
            # Final Stability Test Data → part (C)
            test = dt[[x, K, _TR]].groupby([K,x], observed=True).agg(['mean','count']).droplevel(0, axis=1)
            return test.rename(columns={'mean':_TR})    
            
        # (V2) Agg data length
        # ───────────────────────────────────────────────────────────────────────────
        # T-Test Test Statistic → TR
        TGa = tg.agg(['mean','count'])
        TTa = tt.agg(['mean'])

        # Force index names → Fix of *Arrow* index naming bug 1
        TGa.index.names = [K, x]
        TTa.index.name  = K

        TGa[_TR] = TGa['mean'] / TTa['mean'] - 1 
        # Bin's E(T) in given TU vs E(T) of TU 
        # → i.e. Bin's leverage over mean in given TU → for good X → has to be stable in Time             

        # Final Stability Test Data → part (C)
        return TGa.drop('mean', axis=1)
    # ─────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # (B) TEST → Denoise Stats
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def _denoise_stats(self, test:pDF, x:str, K:str) -> dict[str, pDF|list]:
        """
        DENOISE → T-TEST OF STAT DIFF 
        - 1 Bins check if they are stat different from each other
        - 2 Merged Bins check if they are stat different from sample Mean 
            - If not then no separation power → should be eliminated from final Test (C)
        - 3 Remove bins with 0 predictive power → bin mean ~ total mean
        """
        t  = 'TOTAL'
        tb = self._weighted_stats(test, x, _TR).reset_index()    # adj.TR mean-std-count over X                
        tt = self._weighted_stats(tb.assign(**{t:t}), t, 'mean') # Total  mean-std-count over whole sample
        td = pd.concat([tb, tt.assign(**{x:t})], axis=0).reset_index(drop=True)

        td,M = self._t_test(td, x, self.t_test_alpha)
        mft  = [m for m in M if t in m]                         # Bins with mean of total sample mean 
        mf   = [m for m in M if m not in mft]                   # Bins with same mean

        test = test.reset_index(names=[K,x])                    # Names to eliminate Arrow-pd index name issue 2
        for m in mft: test = test.loc[~test[x].isin(m)]         # Remove bins with average mean
        for m in mf:  test.loc[test[x].isin(m), x] = m[0]       # Merge t-test same bins   

        # Correct TR MEAN by Merged Bins → sets index to unique [x,K] after merge above
        test = self._weighted_mean(test, [x,K], _TR).rename(columns={'mean':_TR})

        return {
            'denoised_test' : test, 
            'prior_test': td, 
            'bins_w_same_mean': mf, 
            'bins_w_total_sample_mean': mft
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # (C) TEST → Focus on both Predictive and Stability aspects
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def _calc_test_stat(self, test:pDF, x:str, K:str) -> float:
        """
        Calulates final test_stat → Focus on both Predictive and Stability aspects.
        
        **W1: Volatility Aspect**
        - W1 is minimizing impact of volatility from small OBS to the Test Result
        - Trying to focus on important predictive Time behavior of given Bin in Time rather than 
        random "shocks" from small OBS
        - Idea behind: Even smaller parts of X may explain important aspects of Y
        - Construct W1 has to be by bins → otherwise it would be extra biased towards bigger bins 
        just by OBS
        - However, that would be against potential predictive contribution of X to explain Y

        **W2: Predictive Aspect**
        - The further from sample mean (TR=0), the more important to be Stable in Time as such bin 
        plays the key role in ML prediction
        - By the Test design TR is close to WOE → thus abs value is the leverage - predictive power - 
        of X towards Target by given bin        
        """

        # W1: Volatility Aspect
        weight     = test['count'].groupby(x).mean()
        test['W1'] = test['count'] / weight 
        test['W1'] = test['W1'].clip(upper=1)   # Max Weight W1 = 1, otherwise too much weight to single Time Unit
        
        # W2: Predictive Aspect
        test['W2'] = abs(test[_TR])
        weight     = self._weighted_stats(test, x, 'W2')['mean']
        test['W2'] = 1   # dummy → Easy way to create new var which has single index same with one of multi-index
        test['W2'] = test['W2'] * abs(weight) 

        # W: Final Weight → W1*W2
        test['W'] = test['W1'] * test['W2']
        test['W'] = test['W']  / test['W'].mean()   # Normalization to achive sum(W) == len(Test)
        test['W'] = test['W'].clip(upper=2.7)       # Outliers
        

        # TEST STAT 
        test['H'] = np.where(test[_TR] > 0, test['W'], 0)   # In Each Time Unit → Bin's E(Target) is over  TU's E(Target) → Weight is relative share of OBS for given TU → 1 when == avg obs
        test['L'] = np.where(test[_TR] < 0, test['W'], 0)   # In Each Time Unit → Bin's E(Target) is below TU's E(Target) → Weight is relative share of OBS for given TU → 1 when == avg obs
        
        stability  = sum(abs(test.groupby(x).H.sum() - test.groupby(x).L.sum()))  
        number_OBS = sum(test['W'])             # Potential Outliers Reduction in the last W step leads to lower value than len(Test)        
        test_stat  = stability / number_OBS     # 1 if 100% consistent behavior | 0 if absolute random - once above, once below
        
        if pd.isna(test_stat): # arrow <NA> check
            test_stat = np.nan

        t = float(test_stat)
        u = len(test.index.get_level_values(K).unique()) < 2
        
        if t is np.nan: # nan → data invariance (const. value) = 0 → sits on the mean
            return 0.0
        if u:           # no data available to assess → real NA
            return np.nan
        return t
    # ─────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # Time Consistency Test
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def _test(self, 
        data: pDF,
        time_bins: pSe,
        x: str = None, 
        *,             

        min_bin_obs = 20, 
        calc_type: Literal[1,2] = 2,
        
        apply_binning: bool = False, 
        bins: int = 10,                                                  
        
        return_tests_or_verdict: Literal[0,1] = 1, 
        
        show_graph: bool = False, 
        fig: go.Figure = None, 
        row: int=1, 
        col: int=1,
        **kwargs
    ) -> dict:
        
        if x is None: 
            x = self.fes[0]

        K = self._get_time_unit(time_bins)
        consistency = self._set_defaults(K)

        # (A) Calulate TS stats
        test = self._calc_stats(data, time_bins, x, calc_type, apply_binning, bins)                
        test = test[test['count'] > min_bin_obs] # min_bin_obs of bins → min stat relevance of mean

        # (B) Denoise TS stats
        denoised = self._denoise_stats(test, x, K)
        
        # (END?) → Requested Tests OR Non-significant FE
        if return_tests_or_verdict == 0: 
            return denoised

        test = denoised['denoised_test']

        # (END) → no data points → status NA
        if len(test.index.unique()) < 2: 
            return consistency
        
        # (C) Test stat
        test_stat = self._calc_test_stat(test, x, K)
        
        # (D) Test decision
        self._set_status(consistency, test_stat)
        self._set_value(consistency, test_stat)
        self._set_data(consistency, test)

        if show_graph: 
            FEConsistencyPlot.graph(test, x, fig, row, col, show_graph)
        
        return consistency
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def test(self, data:pDF, *, extend_by_graphs = False, verbose = True, **kwargs) -> dict:
        """
        Executes a full Predictive Consistency Test across the specified time horizon.

        The method performs a multi-stage stability analysis:
        1. Temporal Slicing: Automatically determines the optimal time resolution 
           and groups data into intervals (TUs).
        2. Leverage Calculation: For each feature bin, it calculates the Target Ratio (TR)—
           the ratio of the bin's target mean to the overall population mean for that TU.
        3. Statistical Denoising: Applies t-tests and weighted averages to merge 
           statistically identical bins and remove noise from low-observation slices.
        4. Consistency Scoring: Quantifies how consistently each bin maintains its 
           position (above or below the mean) across all time units, weighted by 
           the predictive importance of that bin.

        Parameters
        ----------
        data : pDF
            The input dataset containing features, target, and time variable.
        extend_by_graphs : bool, default=False
            If True, attaches Plotly visualization objects to the result for 
            visual inspection of temporal drifts.
        verbose : bool, default=True
            Enables progress tracking and detailed logs during feature processing.
        **kwargs:
            Allows overriding internal test parameters:
            - calc_type : Literal[1,2]
            - apply_binning : bool = False
            - bins : int = 10                                            
            - return_tests_or_verdict : Literal[0,1] = 1
            - show_graph : bool = False
            - fig : go.Figure = None
            - row : int=1
            - col : int=1

        Returns
        -------
        **return_tests_or_verdict = 1** (default)

        dict : dictionary by feature with consistency stats:
                - stats['is_stable']  = test > TH → Y | N | NA    # str
                - stats['value']      = test_stat                 # float        
                - stats['data']       = test                      # pDF | None
                - stats['time_unit']  = test's time unit param    # str
                - stats['graph']      = extend_by_graphs=True     # plotly.Figure
    
        **return_tests_or_verdict = 0** 

        dict : dictionary by feature with prior test stats:
                - stats['denoised_test']
                - stats['prior_test']
                - stats['bins_w_same_mean']
                - stats['bins_w_total_sample_mean']
        """
        dt = data.loc[:, [self.target, self.time_var] + self.fes]

        tu, mtu = self._determine_time_unit(data, self.time_var, self.time_unit, self.max_time_units)
        tb = self._apply_max_time_bins(data, self.time_var, tu, mtu)
        
        Xloop, progress = Display.processing_progress(
            self.fes, kwargs, verbose, desc='Features Test of Predictive Consistency'
        )

        data = {}
        for x in Xloop:
            Display.update_progress(progress)
            data[x] = self._test(dt, tb, x, **kwargs)

        if kwargs.get('return_tests_or_verdict') == 0:
            return data
        
        if extend_by_graphs:
            return FEConsistencyPlot.dash(data)[0]
        
        return data
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def update(self, data:dict[str,dict]) -> dict:
        """        
        Features Update for new c_test_th of Test of Predictive Consistency in across time_unit.
        NOTE:
         - .update() is applicable only when `just c_test_th` is changed; 
         - when t_test_alpha is changed then .test() must be run instead.

        Parameters
        ----------
        data (dict) : output dictionary from .test() and/or .dash()

        Returns
        -------
        dict : updated dictionary by X with dictionary consistency stats:
                - stats['is_stable']  = test > TH → Y | N | NA    # str         → UPDATED
                - stats['value']      = test_stat                 # float        
                - stats['data']       = test                      # pDF | None
                - stats['time_unit']  = test's time unit param    # str
                - stats['graph']      = extend_by_graphs=True     # plotly.Figure
                - stats[...]          ▼ if data was output of dash() 
                - stats['verdict']    = is_stable STABLE|UN...    # str         → UPDATED
        """        
        data = Dict.deep_copy(data)
        for consistency in data.values():
            self._set_status(consistency)
            if consistency.get(LBL.VERDICT):
                FEConsistencyPlot._set_dash_verdict(consistency)
        return data    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FEConsistencyPlot:
    """Feature Predictive Consistency Test Plotting Functinality."""

    @Dataclass.base
    class FigLayout:
        update_layout:       dict = None
        graph_height:         int = 300
        graph_width:          int = 470
        horizontal_spacing: float = 0.2
        vertical_spacing:   float = 0.3         

    @staticmethod
    def _get_names_x_K(test:pDF):
        i = test.index.names 
        return {'x': i[0], 'K':i[1]} # test data design

    @classmethod
    def _graph_data(cls, test:pDF):
        x,K = cls._get_names_x_K(test).values()

        # Final Data        
        df = test[[_TR, 'count']].copy()
        
        # Bins Distribution → on the same index
        tc = df['count'].groupby([K]).sum()
        gc = df['count'].groupby([x,K]).sum()
        df['DR'] = gc/tc
        
        # Time Unit Mean → TA single vs D_cooked multi
        ta = FeConsistency._weighted_stats(test, K, _TR)['mean']     
        df['TA'] = 1 # dummy → easy single-multiindex new var
        df['TA'] = df['TA']*ta
        
        # Plotly needs
        bins = df[_TR].groupby(x).mean()    
        b_mx = bins.loc[(bins==bins.max()) | (bins==bins.min())].index.astype(str)
        bins = bins.index.astype(str)
        
        df    = df.reset_index()
        df[x] = df[x].astype(str)    
        time_x = ta.reset_index()[K]  

        return df, x, K, bins, b_mx, time_x
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _graph_update(fig, row, col, show, time_x, x):
        x_, y1, y2 = Plot.get_sub_axes(fig, row, col).values()

        size = 14
        if not show: size = 11
        font = dict(family="Arial", size=size, color="black")
        
        fig.layout[x_].update(type='category', categoryarray=time_x, tickfont=font)
        fig.layout[y1].update(range=[0, 1], tickfont=font, showgrid=False)
        fig.layout[y2].update(tickfont=font)

        # Show help axes desc only for the first graph in a row (otherwise too noisy in the grid)
        if ( col == 1 ):
            standoff = {}
            if not show: standoff = dict(standoff=5)
            fig.layout[y1].update(title=dict(text="Bins Distribution (%)", font=font, **standoff))
            fig.layout[y2].update(title=dict(text="Bins E(Target) (%)",    font=font, **standoff))

        fig.update_layout(
            height = 500, width = 1100,
            title  = {'text':x, 'x':0.5, 'xanchor': 'center','yanchor': 'top', 'font':font},
            barmode= 'stack',
            bargap = 0.0,
            showlegend = False
        ) 
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def graph(cls, test:pDF, fig:go.Figure=None, row=1, col=1, show=False):

        # ───────────────────────────────────────────────────────────
        # INIT
        # ───────────────────────────────────────────────────────────
        # Allows directly create dash-like figure in loop 
        if ( fig is None ):
            fig = make_subplots(
                    rows  = 1, 
                    cols  = 1,
                    specs=[[{"secondary_y": True}]],
                ) 
                    
        if ( test is None ) or ( not len(test) ):
            NA = fig.add_trace(go.Scatter(
                    x=[0.5], y=[0.5],  
                    mode='text',text=['No data'],
                ), row=row, col=col)
            return NA 
        
        ColorsB  = ['(13, 117, 207)', '(22, 173, 198)', '(22, 198, 134)', '(211, 189, 63)', '(203, 76, 76)']*3
        ColorsL  = ['( 0,  31, 183)', '( 0, 254, 245)', '(80, 252,  74)', '(236, 254,  0)', '(183,  0,  0)']*3

        # ───────────────────────────────────────────────────────────
        # DATA
        # ───────────────────────────────────────────────────────────
        df, x, K, bins, b_mx, time_x = cls._graph_data(test)    
        
        # ───────────────────────────────────────────────────────────
        # GRAPH Construct
        # ───────────────────────────────────────────────────────────        
        for i,b in enumerate(bins):

            # 1 Distribution
            # ───────────────────────────────────────────────────────
            fig.add_trace(
                go.Bar(
                     x = df[df[x]==b][K]
                    ,y = df[df[x]==b]['DR']
                    ,hovertemplate = b+'<br>Share: %{y:.0%}'
                    ,name = 'Bin'
                    ,marker_color ='rgba'+str(ColorsB[i])[:-1]+',0.7)'
            ), row=row, col=col)

            # 2 E(T|B) → By Bins → Prediction Stability of X
            # ────────────────────────────────────────────────────────
            width = 1
            if b in b_mx: width = 3

            fig.add_trace(
                go.Scatter(
                     x = df[df[x]==b][K]
                    ,y = df[df[x]==b][_TR]
                    ,hovertemplate = b+'<br>E(T|B): %{y:.1%}'
                    ,name = 'E(Target|Bin)'
                    ,line_color = 'rgb'+str(ColorsL[i])
                    ,line_width = width
                    ,yaxis='y2'
            ), row=row, col=col, secondary_y=True)

            # 3 E(T) → AVG in Time → Target Time behavior
            # ────────────────────────────────────────────────────────
            width = 1
            fig.add_trace(
                go.Scatter(
                     x = df[df[x]==b][K]
                    ,y = df[df[x]==b]['TA']
                    ,hovertemplate = b+'<br>E(T): %{y:.1%}'
                    ,name = 'E(Target)'
                    ,line_color = 'rgb(0,0,0)'
                    ,line_width = width
                    ,line_dash = 'dash'
                    ,mode='lines'
                    ,yaxis='y2'
            ), row=row, col=col, secondary_y=True)
            
        # Axes Updates
        # ───────────────────────────────────────────────────────────                
        cls._graph_update(fig, row, col, show, time_x, x)

        if not show:
            return fig

        fig.show()  
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dash_order(test_data:dict, features:list, ascending=True):
        if features is None:
            a = 1 if ascending else -1
            I = np.argsort([a*t[LBL.VALUE] for t in test_data.values()])
            return [list(test_data)[i] for i in I]
        
        features   = List.ensure(features)
        fe_present = [x for x in features if x in test_data]            
        if not fe_present: 
            raise ValueError('AIG SAFETY ► Provided features has no overlap with data.')
        if len(fe_present) != len(features):
            not_in_data = [x for x in features if x not in fe_present]
            print(f'WARNING: Not all provided features is present in data:\n {not_in_data}')
        return fe_present
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def _dash_graph_title(cls, x:str, data:dict[str,dict]):
        v = cls._dash_verdict(data[x])
        d = f' | {round(data[x][LBL.VALUE],2)}' if v!= STAB.NA else ''
        return f'{x} → {v}{d}' 
    
    @staticmethod
    def _dash_verdict(consistency:dict):
        s = consistency[LBL.IS]
        if s == STAB.Y: 
            return 'STABLE'
        if s == STAB.N: 
            return 'UNSTABLE'
        return STAB.NA
    
    @classmethod
    def _set_dash_verdict(cls, consistency:dict):
        """Sets variable dash verdict callable by `LBL.verdict` with respect to its stat test value."""
        consistency[LBL.VERDICT] = cls._dash_verdict(consistency)     # extras → woe_stability

    @classmethod    
    def _dash(cls, 
        test_data:  dict, 
        grid_title: str  = 'Features Stability', 
        grid_cols:  int  = 4, 
        features:   list = None, 
        ascending:  bool = True, 
        fig_layout: FigLayout = FigLayout()
    ) -> dict:
        """Dashboard Gen of Feature Predictive Consistency Test."""

        # TEST DATA
        # ────────────────────────────────────────────────────────────────────        
        data = Dict.deep_copy(test_data)
        features = cls._dash_order(data, features, ascending)
        
        # DASH
        # ────────────────────────────────────────────────────────────────────
        def graph(x):
            return cls.graph( data[x][LBL.DATA] )

        def update(x, g):
            d = data[x][LBL.DATA]
            data[x][LBL.GRAPH] = g     
            data[x][LBL.GMX]   = min_max(d)     # extras → woe_stability
            cls._set_dash_verdict(data[x])

        def min_max(d:pDF):
            if d is None: return
            return d[_TR].agg(['min','max']).to_list()
                
        def title(x):
            return cls._dash_graph_title(x, data)
        
        update_layout = dict(barmode = 'stack', bargap = 0.0) | (fig_layout.update_layout or {})

        fig = Plot.create_subplots(
            features,
            graph_fx            = graph,
            title_fx            = title,
            update_fx           = update,        
            secondary_y         = True,
            grid_title          = grid_title,
            grid_cols           = grid_cols,
            height              = fig_layout.graph_height,
            width               = fig_layout.graph_width,
            update_layout       = update_layout,
            horizontal_spacing  = fig_layout.horizontal_spacing,
            vertical_spacing    = fig_layout.vertical_spacing,            
            desc_axes_by_first_only = True
        )
        
        return {
            DASH.DATA:  data, 
            DASH.FIG:   fig, 
            DASH.ORDER: features
        }

    @classmethod
    def dash(cls, 
        test_data:  dict, 
        grid_title: str  = 'Features Stability', 
        grid_cols:  int  = 4, 
        features:   list = None, 
        ascending:  bool = True, 
        fig_layout: FigLayout = FigLayout(),
        verbose:    bool = True
    ) -> dict:
        """
        Dashboard Gen of Feature Predictive Consistency Test.

        Parameters
        ----------
        data_test   : Dataframe to use for creationg of test or created test data by .test().
        grid_title  : Dashboard figure main title.
        grid_cols   : Number of figured (columns) in a row of the fig matrix.
        features    : Prior features (subset) and fes order in dashboard.
        ascending   : Sorting order of features if no features provided.
        fig_layout  : Figure extras to tune layout by FigLayout.
        verbose     : True shows waiting spinner.

        Returns
        -------
        (dict of) : ▼ data (dict), fig (plotly.Figure), order (list) ▼
        data : dictionary by X with consistency stats:
                - data['graph']      = plotly figure             # plt.Figure
                - data['gmx']        = figure's min-max          # tuple[float,float]
                - data['is_stable']  = test > TH → Y | N | NA    # str
                - data['verdict']    = is_stable STABLE|UN...    # str
                - data['value']      = test_stat                 # float        
                - data['data']       = test                      # pDF | None
        fig : figure matrix of generated plotly graphs
        order : list ordering of figure - useful for sync order with test data
        """                
        with Spinner('Generating Stability Dashboard').progress(verbose):
            return cls._dash(
                test_data   = test_data,
                grid_title  = grid_title,
                grid_cols   = grid_cols,
                features    = features,
                ascending   = ascending,
                fig_layout  = fig_layout
            )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def update_dash(cls, updated_test_data:dict, dash_data:dict) -> dict:
        """
        Updates previously generated dash fig based on provided:
         - updated_test_data → updated test data from `FeConsistency.update()` 
         - dash_data → previously created dash data by `FeConsistencyPlot.dash()`
            - updates titles of `DASH.FIG` based updated test_data
        
        NOTE:
         - .dash_update() is applicable only when `just c_test_th` was changed and applied via .update(); 
         - when t_test_alpha was changed then .dash() must be run instead.

        Returns
        -------
        (dict of) : ▼ data (dict), fig (plotly.Figure), order (list) ▼
        data : dictionary by X with consistency stats:
                - data['graph']      = plotly figure             # plt.Figure
                - data['gmx']        = figure's min-max          # tuple[float,float]
                - data['is_stable']  = test > TH → Y | N | NA    # str                  → UPDATED
                - data['verdict']    = is_stable STABLE|UN...    # str                  → UPDATED
                - data['value']      = test_stat                 # float        
                - data['data']       = test                      # pDF | None
        fig : figure matrix of generated plotly graphs                                  → UPDATED
        order : list ordering of figure - useful for sync order with test data
        """
        
        # 1. Update extended DASH.data by updated test data from .update()
        dorg = dash_data[DASH.DATA]
        data = {}
        for x in updated_test_data:
            data[x] = dorg[x] | updated_test_data[x]
            cls._set_dash_verdict(data[x])

        # 2. Update DASH.FIG → titles update needed only →         
        order = dash_data[DASH.ORDER]
        fig   = dash_data[DASH.FIG]
        anns  = fig.layout.annotations        
        for i, x in enumerate(order):
            anns[i].text = cls._dash_graph_title(x, data)

        return {
            DASH.DATA:  data, 
            DASH.FIG:   fig, 
            DASH.ORDER: order
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def subset_dash(cls,         
        dash_data:  dict, 
        features:   list, 
        graph_fx:   FuncType = None,
        title_fx:   FuncType = None,
        grid_title: str = 'Features Stability', 
        grid_cols:  int = 4,
        fig_layout: FigLayout = FigLayout()
    ) -> go.Figure:
        """
        Subsets previously generated dash fig based on provided:
         - dash_data → previously created dash data by `FeConsistencyPlot.dash()`

        Parameters
        ----------
        dash_data   : Previously created dash data by `FeConsistencyPlot.dash().
        features    : Prior ordered features for the dashboard subset creation.
        graph_fx    : Optional graph function; default: go.Figure( dash_data[LBL.data][x][LBL.graph] ).
        title_fx    : Optional title function; default: cls._dash_graph_title(x, dash_data[LBL.data] ).
        grid_title  : Dashboard figure main title.
        grid_cols   : Number of figured (columns) in a row of the fig matrix.
        fig_layout  : Figure extras to tune layout by FigLayout.

        Returns
        -------        
        fig : figure matrix of generated plotly graphs        
        """

        update_layout = dict(barmode = 'stack', bargap = 0.0) | (fig_layout.update_layout or {})

        graph = lambda x: go.Figure( dash_data[LBL.DATA][x][LBL.GRAPH] )
        title = lambda x: cls._dash_graph_title(x, dash_data[LBL.DATA] )

        return Plot.create_subplots(
            features,
            graph_fx            = graph_fx or graph,
            title_fx            = title_fx or title ,
            secondary_y         = True,
            grid_title          = grid_title,
            grid_cols           = grid_cols,
            height              = fig_layout.graph_height,
            width               = fig_layout.graph_width,
            update_layout       = update_layout,
            horizontal_spacing  = fig_layout.horizontal_spacing,
            vertical_spacing    = fig_layout.vertical_spacing,
            desc_axes_by_first_only = True
        )        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬