"""
Cross-model transition engine for predictive divergence and auditing analysis.

This module implements a cross-opinion framework designed to determine if 
a secondary 'auditor' model can provide new, statistically significant 
information regarding the binning of a primary baseline model. 
By analyzing the conditional target rates across intersecting model bins, 
it identifies regions of divergence where the auditor discovers sub-segments 
with predictive power that differs significantly from the baseline expectation.
"""

from ...deps.data               import  np, pd, pDF, pSe
from ...deps.common             import  warnings
from ...deps.types              import  NDArray
from ...core.factory            import  (
                                        Dataclass, 
                                        DictClass
                                        )
from ...core.interop            import  Pd
from ...utils.stats             import  Stats
from ..tests                    import  TestSampleDiff


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# MODEL CROSS TRANSITION ASSESSMENT
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    S  = 'Total'
    B  = 'ProbBIN'
    T  = 'Target'
    C  = 'Count'
    W  = 'Weight'
    L  = 'Lift'
    U  = 'Underestimated'
    O  = 'Overestimated'
    AE = 'Audited'
    AO = 'Auditor'

@Dataclass.base
class CrossTab:
    mean:   pDF
    count:  pDF
    weight: pDF
    total:  pDF

@Dataclass.base
class CrossAlign:
    is_a: bool
    max:  int = None
    len:  dict[int|str, int] = None
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ModelsCrossTransition:
    """
    Models cross transition by bins to answer question: 
    Can cross model provide new information to reorder bin of the second model?
    
    Parameters
    ----------
    alpha : float
        Significance level to eliminate noise from statistically significant transition.
    
    align_bins: bool, default False
        True gurantees to return square matricies. This is done by injecting nan to mean 
        matrix and 0 vectors to count matrix, effectively not impacting any result.
        False doesn't verify square property.        
    """    

    def __init__(self, alpha:FloatingPointError, align_bins:bool = False):
        self.alpha = alpha
        self.align = align_bins

    def _set_alignment(self, bins: dict[int, pSe]) -> None:
        if not self.align:
            self._ca = CrossAlign(is_a = False)
            return 
        
        bu = {a: b.value_counts() for a, b in bins.items()}
        bl = {a: len(b) for a, b in bu.items()}
        mx = max(bl.values())

        if all([mx == l for l in bl.values()]):
            self._ca = CrossAlign(is_a = False)
            return

        self._ca = CrossAlign(
            is_a = True,
            max  = mx,
            len  = bl
        )
    
    def _align_bins(self, c:pDF, pair:list, is_count:bool) -> pDF:
        if not self._ca.is_a:
            return c
        
        for i, a in enumerate(pair):
            bl = self._ca.len[a]
            if bl == self._ca.max:
                continue
            
            i0  = ( i == 0 ) # adj. on rows (index); otherwise cols

            if i0: x0 = c.iloc[0,:]
            else:  x0 = c.iloc[:,0]
            
            v = np.nan            
            if is_count:
                v = 0

            x0 = {a: v for a in x0.keys()}
            m  = self._ca.max - bl

            d = pDF([x0]*m, index = [bl + i for i in range(m)])
            if not i0:
                d = d.T
                nc, ni = c.columns.name, c.index.name
                
            c = Pd.concat([c, d], axis=i)

            if not i0:
                c.columns.name = nc
                c.index.name = ni

        return c
        
    def _pd_cross(self, pair:list, *args, is_count:bool = False, **kwargs) -> pDF:
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            c = pd.crosstab(*args, **kwargs, dropna=False)
        return self._align_bins(c, pair, is_count)

    @staticmethod
    def _total(m:pDF, c:pDF) -> pDF:
        w0 = (c.T / c.sum(axis=1)).T
        w1 = c / c.sum(axis=0)
        s = m.copy()
        s.loc[:, LBL.S] = (s*w0).sum(axis=1)
        s.loc[LBL.S, :] = (s*w1).sum(axis=0)
        s.iloc[-1,  -1] = (m*c).sum().sum() / c.sum().sum()
        return s
            
    def _crosstab(self, pair:list, a_x:pSe, a_y:pSe, target:pSe) -> CrossTab:
        """
        Models cross comparision by bins to answer question: 
        Can cross model provide new information to reorder bin of the second model?
        """
        m = self._pd_cross(pair, a_x, a_y, values=target, aggfunc=np.average)
        c = self._pd_cross(pair, a_x, a_y, is_count=True)
        s = self._total(m, c)
        return CrossTab(
            mean   = m, 
            count  = c,
            weight = c / len(target),
            total  = s
        )

    @staticmethod
    def _transpose(tab:CrossTab) -> CrossTab:
        return CrossTab(**{k: t.T for k,t in tab.items()})
           
    def process(self, 
        target: pSe,
        bins:   dict[int, pSe]
    ) -> dict[int, dict[int, CrossTab]]:
        """Returns all cross combinations across all models."""
        self._set_alignment(bins)
        x_tab = {a: {} for a in bins}
        for pair in Stats.combination(list(bins), 2):
            tab = self._crosstab(pair, *[bins[a] for a in pair], target)
            x_tab[pair[0]].update( {pair[1]: tab} )
            x_tab[pair[1]].update( {pair[0]: self._transpose(tab)} )
        return x_tab
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _sig_diff(self, m:pDF, n:pDF, base_m:pSe, base_n:pSe, in_rows:bool=True) -> pDF:
        """
        Bin of Model in rows (in_rows=True) / cols (in_rows=False) 
        being compared to opinion by bins of model on the second axis.        
        """
        mb = base_m.to_numpy()
        nb = base_n.to_numpy()
        if in_rows:
            mb = mb[:,np.newaxis]
            nb = nb[:,np.newaxis]
        
        mask = TestSampleDiff(self.alpha).welch_mx(m, n, mb, nb)
        
        f = m.where(mask, np.nan) # significance mask - keep only sig diff
        return f - mb  
    
    @staticmethod
    def _get_triu_mx(x_tab: CrossTab) -> tuple[NDArray, NDArray]:
        """Ensure true positives only → auditor in columns"""
        a = len(x_tab.mean.loc[:,0])
        b = len(x_tab.mean.loc[0,:])
        upper = np.triu(np.ones((a, b), dtype=bool), k=1)   # j > i → if true signal, must be + sign
        lower = np.tril(np.ones((a, b), dtype=bool), k=-1)  # j < i → if true signal, must be - sign
        return upper, lower
            
    def get_sig_diff(self, x_tab: dict[int, dict[int, CrossTab]]) -> dict[int, dict[int, pDF]]:
        """
        Returns cross tab filtered to only statistically significant different
        fields of model in rows assessed by model in cols (cross opinion).
        
        Model in rows acts as a audited base: 
        - mean of each bin as total E(T) 
        - obs of each bin as total count of a bin
        - cross tab compare significance of difference by auditor in columns:
          - e.g. for a bin 0 of model in rows we compare all cross E(T) by cross model in cols (0-9)
          - for each cross we measure whether signif. diff occurs agains total E(T|bin0)

        True positive signal → Audited model in rows, auditor model in columns:
        - Thus we keep only `true positive` signal by masking any "ghost claims" about a finding by the auditor.
        - Upper triangle (from diagonal) is a true signal if auditor finds higher risk (+) within audited bin.
        - Vice versa for lower triangle where risk by auditor must be with (-).
          
        In sum, we compare whether opinion of the model in columns (cross model) 
        differ significantly to model label (bin) under the assessment (in rows).

        By design, model in columns should not have significant information to 
        reorder agents bin. If this condition is violated, model in cols
        found new information valueinside a given bin of row model.
        """
        diff = {a: {} for a in x_tab}
        for a, tab in x_tab.items():
            for c, cross in tab.items():
                upper, lower = self._get_triu_mx(cross)
                m = cross.total.iloc[:-1,-1]    # mean base as model in rows E(T) of a bin
                n = cross.count.sum(axis=1)     #  obs base as model in rows total bin's count
                d = self._sig_diff(cross.mean, cross.count, m, n)
                d = d.where( ((d > 0) & upper) | ((d < 0) & lower) ) # enforce true positive signal
                diff[a].update( {c: d} )
        return diff
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def get_totals(
        x_tab: dict[int, dict[int, CrossTab]],
        mapping: dict[int, dict],                      
    ) -> dict[int, pDF]:
        """Extract model's totals with scoring lift to one table."""        
        cut = {}
        for a, tab in x_tab.items():
            cross = tab[next(iter(tab))]
            bins_ = mapping[a]
            cut[a] = pDF({
                LBL.T: cross.total[LBL.S][:-1],
                LBL.C: cross.count.sum(axis=1),
                LBL.W: cross.weight.sum(axis=1),                
            }).set_index(pSe({v: k for k, v in bins_.items()}, name = LBL.B))            
            cut[a][LBL.L] = cut[a][LBL.T] / ( cut[a][LBL.T] * cut[a][LBL.W] ).sum()
        return cut
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ModelCrossDivergence:
    """Measures significant divergance between model in rows given cross model opinion in cols."""
 
    @staticmethod
    def process(
        sig_diff: dict[int, dict[int, pDF]], 
        x_tab:    dict[int, dict[int, CrossTab]]
    ) -> dict[int, dict[int, pDF]]:
        """
        Measures significant divergance between the main model given cross model opinion.
        
        Audited model in rows, auditor model in columns:
        - Thus we keep only `true positives` signal by masking any "ghost claims" about a finding by the auditor.
        - Upper triangle (from diagonal) is a true signal if auditor finds higher risk within audited bin.
        - Vice versa for lower triangle.

        *Math Formulation +/-:*
        - +div(Bin_i) = Sum_over_j( P(Agent_Y = j | Agent_X = i) * (E_hat(T | i, j) - E(T | Bin_i)) | > 0 ),
        - -div(Bin_i) = Sum_over_j( P(Agent_Y = j | Agent_X = i) * (E_hat(T | i, j) - E(T | Bin_i)) | < 0 ),
        
        where:
        - Bin_i       : The specific bucket of the baseline model (Model 0) being audited,
        - j           : The sub-segments generated by the advanced model (Model 3),
        - P(Y=j | X=i): The conditional probability/distribution of Model 3 given Model 0,
        - E_hat(T)    : The actual target rate observed within sub-segment (i, j),
        - E(T)        : The baseline expected target rate predicted by Model 0 for Bin_i.
        """
        div = {a: {} for a in sig_diff}
        for a, tab in sig_diff.items():
            for c, cross in tab.items():
                w = x_tab[a][c].weight
                x = cross.fillna(0)*w
                x_p = x.where(x > 0, 0).sum(axis=1)  # Risky → E(T|i,j) - E(T|Bini) > 0 → Adverse Selection Trap
                x_m = x.where(x < 0, 0).sum(axis=1)  # Miss  → E(T|i,j) - E(T|Bini) < 0 → Missed Opportunity Trap
                div[a].update( {c: pDF({LBL.U: x_p, LBL.O: x_m})} )
        return div
    
    @staticmethod
    def get_summary(divergance: dict[int, dict[str, pDF]]) -> dict[int, dict[str, pDF]]:
        """
        Divergance summary over each model in a role:
        - as an Auditor - capturing missing significant information in cross model bins
        - being Audited by a cross model - how much significant information audited model miss?
        """
        audited = {a: [] for a in divergance}
        auditor = {a: [] for a in divergance}

        for a, tab in divergance.items():
            for b, stat in tab.items():
                s = {
                    LBL.U: stat[LBL.U].sum(),
                    LBL.O: stat[LBL.O].sum(),
                    LBL.S: stat.abs().sum(axis=1).sum()
                }
                audited[a].append(s | {LBL.AO: b})
                auditor[b].append(s | {LBL.AE: a})

        div = {}
        for a in divergance:
            div[a] = {
                LBL.AE: pDF(audited[a]).set_index(LBL.AO),
                LBL.AO: pDF(auditor[a]).set_index(LBL.AE)
            }
        return div    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬