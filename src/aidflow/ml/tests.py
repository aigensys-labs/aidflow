"""
Statistical hypothesis testing suite for model comparison and sample differentiation.

This module implements a rigorous set of statistical tests designed to quantify 
the significance of differences between ML models and data samples. By providing 
robust implementations of DeLong's test for correlated AUC comparisons and Welch's 
t-test for unequal variance means, it enables the framework to distinguish 
genuine predictive improvements from stochastic noise during model optimization.
"""

from ..deps.data        import np, pd, pDF, pSe
from ..deps.ml.stats    import t, norm, rankdata
from ..deps.types       import ArrayLike, NDArray

from .sampling          import Bootstrap
    

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# STATISTICAL TESTS OF DIFFERENCE SIGNIFICANCE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class TestModelDiff:
    """
    Statistical test of two models under H0: δ = 0 → models have same performance.
    Class provides 2 options where both provides identical result if bootstrap 
    sample is big enough.
     - delong is preferred as it is significantly faster to compute.
     - bootstrap is for verification of edge cases or fall-back.

    Parameters
    ----------
    y_true  : True values of target variable.
    y_pred1 : Model1 prediction of the target.
    y_pred2 : Model2 prediction of the target.
    """

    def __init__(self,
        y_true:  ArrayLike, 
        y_pred1: ArrayLike, 
        y_pred2: ArrayLike
    ):    
        self.y_true  = np.asarray(y_true)
        self.y_pred1 = np.asarray(y_pred1)
        self.y_pred2 = np.asarray(y_pred2)
        self._validation()
    
    def _validation(self):
        if ( 
            self.y_true.shape != self.y_pred1.shape or 
            self.y_true.shape != self.y_pred2.shape
        ):
            raise ValueError("All input arrays must have the same shape.")
        
        u = np.unique(self.y_true)
        if not np.array_equal(u, [0, 1]):
            raise ValueError("Ground truth must be binary {0, 1}.")
                

    @staticmethod
    def p_value(z_score:float) -> float:
        """Returns p_value from z_score."""
        return 2 * norm.sf(np.abs(z_score))


    def delong(self, z_score=False) -> float:
        """
        Perform DeLong's test for comparing the AUCs of two (correlated) models.

        Parameters
        ----------
        z_score : False (default) returns p_value | True returns z_score

        Returns
        -------
        p_value | z_score : 
            min statistical significance level (alpha) where H0 can be rejected.
        """        
        # 1. Separate predictions by class
        pos_mask = (self.y_true == 1)
        neg_mask = (self.y_true == 0)
        
        m = np.sum(pos_mask)  # Number of positive samples
        n = np.sum(neg_mask)  # Number of negative samples
        
        # We treat models A and B as a matrix to vectorize components
        preds = np.vstack([self.y_pred1, self.y_pred2]) 

        # 3. Calculate Structural Components (V10 and V01)
        # DeLong's method using the Rank-based approach
        v10  = np.zeros((2, m))
        v01  = np.zeros((2, n))
        aucs = np.zeros(2)

        for i in range(2):
            # Full ranks across all samples
            # This tells us the 'position' of every score in the whole dataset
            # 'average' method handles ties correctly for Mann-Whitney
            ranks = rankdata(preds[i], method='average')
            
            # Get ranks of positive and negative samples within their own groups
            rank_pos = rankdata(preds[i][pos_mask], method='average')
            rank_neg = rankdata(preds[i][neg_mask], method='average')
            
            # Structural components
            # V10: Proportion of negatives strictly smaller than each positive
            # V01: Proportion of positives strictly larger than each negative
            v10[i] = (ranks[pos_mask] - rank_pos) / n            
            v01[i] = (m - (ranks[neg_mask] - rank_neg)) / m
            
            # The mean of V10 is exactly the AUC
            aucs[i] = np.mean(v10[i])

        # 4. Covariance Matrix Calculation
        # S10 is the 2x2 covariance matrix of the positive components
        # S01 is the 2x2 covariance matrix of the negative components
        s10 = np.cov(v10)
        s01 = np.cov(v01)
        s = s10 / m + s01 / n

        # 5. Compute Z-score and P-value
        # Var(AUC_A - AUC_B) = Var(A) + Var(B) - 2*Cov(A, B)
        auc_diff = aucs[0] - aucs[1]
        var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
        
        # Handle edge case: identical models
        if var_diff <= 0:
            return 1.0

        # Z = (AUC_A - AUC_B) / sqrt(Var(AUC_A - AUC_B))
        z_score_ = auc_diff / np.sqrt(var_diff)
        if z_score: 
            return z_score_
        
        return self.p_value(z_score_)


    def bootstrap(self, n_samples:int=200) -> float:
        """
        Boostrap approach to estimate p_value of two models 
        under H0: δ = 0 → models have same Gini performance.

        Parameters
        ----------
        n_samples : Number of bootstrap samples to construct dst.
        
        Returns
        -------
        p_value : min statistical significance level (alpha) where H0 can be rejected.        
        """
        from .metrics import Gini

        # 1. Compute observed difference
        gini1 = Gini.fast_calc(self.y_true, self.y_pred1)
        gini2 = Gini.fast_calc(self.y_true, self.y_pred2)
        observed_diff = gini1 - gini2
        
        # 2. Generate dist.
        diffs = []        
        for idx, _ in Bootstrap(n_samples).split(self.y_true):
            b_y_true  = self.y_true[idx]
            b_y_pred1 = self.y_pred1[idx]
            b_y_pred2 = self.y_pred2[idx]
            b_gini1   = Gini.fast_calc(b_y_true, b_y_pred1)
            b_gini2   = Gini.fast_calc(b_y_true, b_y_pred2)
            boot_diff = b_gini1 - b_gini2
            diffs.append(boot_diff)
    
        # 3. P-value calculation (Two-Tailed)
        # We center the distribution around 0 to simulate 
        # the Null Hypothesis H0: δ = 0
        boot_diffs = np.array(diffs)
        null_dist  = boot_diffs - np.mean(boot_diffs)

        # 4. Compute p-value
        # How often does the Null distribution produce something as extreme 
        # as our observed difference?        
        p_value = np.sum(np.abs(null_dist) >= np.abs(observed_diff)) / n_samples
        return p_value
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class TestSampleDiff:
    """
    Statistical test of 2 sample means `statistical difference`.

    Parameters
    ----------
    alpha (float) : test's alpha th signifigance
    """

    RES   = 'significant'
    MEAN  = 'mean'
    STD   = 'std' 
    DF    = 'd.f.'
    TEST  = 'test_value'
    T_VAL = 't_value'

    _NORM = 10

    def __init__(self, alpha:float=0.05):
        self.alpha = alpha

    def _welch_adj(self, m:pSe|NDArray, n:pSe|NDArray, v:pSe|NDArray, is_proba_adj:bool
    ) -> pSe | NDArray:
        """
        Welch t-test assumes normality of tested probability means, 
        we ensure to be close; otherwise nan.
        """
        if not is_proba_adj:
            return n

        s = lambda x: x if isinstance(x, (int,float)) else x.sum()
        
        target_majority = s( ( m * (n / s(n.sum())) ).sum() )
        target_majority = np.where(target_majority < 0.5, 0, 1)        
        mn = np.where(pd.isna(m * n), target_majority, m * n).astype(float)      
        
        a = np.where(
            (n < self._NORM) | 
            ( (v == 0) & (mn.astype(int) == target_majority) ),
            True, False
        )

        if not s(a.sum()):
            return n
        
        n_ = n.copy().astype(float)        
        n_[a] = np.nan
        return n_
    
    @staticmethod
    def _get_variance(m:pSe|NDArray, s:pSe|NDArray|str|None, data:pDF=None) -> pSe|NDArray:
        """Returns variance → binary target m*(1-m) | direct std input req."""
        if s is None:
            return m*(1-m)
        if data is not None:
            if not isinstance(s, str):
                raise ValueError('AIG Safety ► Wrong combination of input')
            return data[s]**2
        return s**2
    
    def _get_stats(self, data:pDF, std:str=None) -> tuple[pSe,pSe,pSe]:
        """Returns data stats: count - mean - variance."""
        n = data['count']   # Number of observations of a bin
        m = data['mean']    # Average value of a bin
        v = self._get_variance(m, std, data)
        a  = std is None                    # Safety adjustment for proba means
        n_ = self._welch_adj(m, n, v, a)    # Ensure normality of observed means
        vn = v/n_                           # Variance adjusted of a obs → mean variance
        return m, n_, vn

    def _get_stats_above(self, *args:pSe) -> tuple[pSe, ...]:
        """Returns same stat from bin above."""
        r = []
        for a in args:
            r.append(a.shift(1))
        return tuple(r)
    
    @staticmethod
    def _welch_test(
        m1:pSe|NDArray, vn1:pSe|NDArray, m2:pSe|NDArray, vn2:pSe|NDArray
    ) -> pSe | NDArray:
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.abs(m1 - m2) / np.sqrt(vn1 + vn2)
    
    @staticmethod
    def _welch_df(
        n1:pSe|NDArray, vn1:pSe|NDArray, n2:pSe|NDArray, vn2:pSe|NDArray
    ) -> pSe | NDArray:
        """Welch's test Degrees of Freedom."""
        vn = (vn2 + vn1)**2
        vd = ( vn2**2 / (n2-1) ) + ( vn1**2 / (n1-1) )
        with np.errstate(invalid='ignore', divide='ignore'):
            return (vn / vd).round(1)

    @staticmethod
    def _ensure_nan(d:NDArray|pSe) -> NDArray:
        return np.where(pd.isna(d), np.nan, d).astype(float)
    
    @classmethod
    def _p_val(cls, val:NDArray, df:NDArray) -> NDArray:
        f = lambda x: cls._ensure_nan(x)
        return t.sf(f(val), f(df)) * 2
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def welch_se(self, data:pDF, std:str=None) -> pDF:
        """
        Welch's t-test of 2 sample means statistical difference.
        Supported data design → dataframe with rows: bins | columns: 'mean" and 'count'.

        When no std is provided, defaults to binary/probability std = m*(1-m).
        
        The test will compare `two neighbor bins` means difference.

        *Statistical Result between two neighbor bins (rows) given alpha:*
        - True  → means are statistically different
        - False → means are statistically same

        >>>
        -----------------
        |Bins|Mean|Count| 
        -----------------
        | B1 | 0.1| 520 | 
        | B2 | 0.2| 800 |
        | .. | .. | ..  |
        -----------------

        Parameters
        ----------
        data (pDF) : data stats containing "count" and "mean" of stats to evaluate
        std (str, opt.) : optional column to use as `populational` std for the test

        Returns
        -------
        data (pDF) : welch t-test result with cols 
        ['significant', 'mean', 'std', 'd.f.', 'test_value', 't_value']
        ▼
        'significant' : True|False - True → means are stat. diff. | False otherwise
        'mean'        : input data mean per data row used to compute `test_value`
        'std'         : calculated std  per data row used to compute `test_value`
        'd.f.'        : calculated test's degrees of freedom used to compute `t_value`
        'test_value'  : calulated Welch t-test value per data row
        't_value'     : calculated test's t-value → `result = test_value > t_value`
        """
        m1, n1, vn1 = self._get_stats(data, std)
        m2, n2, vn2 = self._get_stats_above(m1, n1, vn1)
        
        test = self._welch_test(m1, vn1, m2, vn2)
        wdf  = self._welch_df(n1, vn1, n2, vn2)
        
        t_val  = t.ppf((1-self.alpha/2), wdf)
        t_test = test > t_val

        return pDF({
            self.RES:   t_test, 
            self.MEAN:  m1, 
            self.STD:   np.sqrt(vn1), 
            self.DF:    wdf, 
            self.TEST:  test, 
            self.T_VAL: t_val
        })
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def welch_mx(self, 
        m1: NDArray, 
        n1: NDArray, 
        m2: NDArray, 
        n2: NDArray,
        s1: NDArray = None, 
        s2: NDArray = None,
    ) -> NDArray:
        """
        Welch's t-test of 2 sample statistical difference. 

        When no v1/v2 (std) is provided, defaults to binary/probability std = m*(1-m).
        
        Supported data design:
        - m1, n1: the first sample of matrices/arrays of Means and Counts respectively.
        - m2, n2: the second sample of matrices/arrays of Means and Counts respectively.
        - s1, s2: 
            - None → only valid mean is probability | binary based E(X) → s = np.sqrt( (m)*(1-m) )
            - otherwise matrix/array of `populational` standard deviation must be provided

        Returns bool array of the same shape as the input.

        Welch t-test verdict True/False for every cell given alpha:
        - True:  H0 rejected → means in corresponding cells are statistically different
        - False: otherwise   → means are statistically same
        """
        v1_ = self._get_variance(m1, s1)
        v2_ = self._get_variance(m2, s2)
        
        n1_ = self._welch_adj(m1, n1, v1_, (s1 is None))
        n2_ = self._welch_adj(m2, n2, v2_, (s2 is None))
        
        vn1 = v1_ / n1_
        vn2 = v2_ / n2_      

        test  = self._welch_test(m1, vn1, m2, vn2)
        wdf   = self._welch_df(n1_, vn1, n2_, vn2)
        p_val = self._p_val(test, wdf)
        
        return p_val < self.alpha
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def t_test(self, 
        means:      NDArray, 
        n_samples:  NDArray | int, 
        stds:       NDArray = None, 
        test_mean:  float = 0.0
    ):
        """
        Vectorized t-test to determine significance against a fixed mean.
        
        Parameters
        ----------
        means (array)         : Array of observed means
        n_samples (array|int) : Sample size per group
        stds (array|None)     : Array of standard deviations | None → means represents proba
        test_mean (float)     : The value to compare against
        
        Returns
        -------
        np.array: Boolean array where True indicates significant difference
        """
        s, n = stds, n_samples
        if stds is None:
            s = np.sqrt(means * (1-means))

        if isinstance(n_samples, int):
            n = np.array([n_samples]*len(means))

        s = np.where(s == 0, np.nan, s)
        n = np.where(n < 2 , np.nan, n)

        test  = np.abs(means - test_mean) / (s / np.sqrt(n))
        df    = n - 1
        p_val = self._p_val(test, df)
        
        return p_val < self.alpha    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬