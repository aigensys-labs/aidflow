"""
Evaluation framework for feature / model performance and separation diagnostics.

This module implements a rigorous suite of metrics designed to quantify the 
predictive power and discriminative capacity of ML models, with a specialized 
focus on binary target separation. It provides high-precision implementations 
of industry-standard risk metrics—including Gini, IV (Information Value), KS 
(Kolmogorov-Smirnov), and Lift—alongside advanced diagnostics for variable 
importance and probability distribution analysis.
"""

from ..deps.data        import  np, pDF, pSe
from ..deps.ml.stats    import  (
                                auc, 
                                roc_curve, 
                                confusion_matrix
                                )
from ..deps.plots       import  go, make_subplots
from ..deps.types       import  (
                                FuncType,
                                ArrayLike, 
                                NDArray
                                )
from ..core.factory     import  DictClass, Dataclass
from ..core.interop     import  Pd

from ..utils.builtins   import  List
from ..display.plots    import  Plot

from .features.woe      import  IVL, IV, WoE
from .features.encoding import  Encoding


class LBL(DictClass):
    G   = 'gini' 
    KS  = 'KS'
    FPR = 'false_positive_rate'
    TPR = 'true_positive_rate'
    IPR = 'pr_cutpoints_stats'
    PC  = 'prob_cutoff'
    P   = 'proba'
    T   = 'target'
    TR  = 'target_remain'
    C   = 'count'
    CS  = 'cumsum'
    L   = 'lift'
    W   = 'weight'

@Dataclass.base
class DiagnosticsResult:
    obs:    int
    events: int
    gini:   float
    IV:     float 
    KS:     float 
    PR:     dict
    lift:   pDF = None


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ML METRICS OF DISCRIMINATION POWER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class MetricsBase:
    
    def __init__(self, incl_miss:bool = True, th_discrete:int = 20, return_stats:bool = False):
        self.incl_miss    = incl_miss
        self.th_discrete  = th_discrete    
        self.return_stats = return_stats  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Gini(MetricsBase):
    """
    Univariate Gini Performance for Binary Target and any dtype of a Variable.    
    
    Parameters
    ----------
    incl_miss : bool = True
        Gini calculation will include missing part by transforming variable to WOE form.
    
    th_discreate : int = 20
        Threshold above which variable is considered as continuous only relevant for 
        num type variable and binning on ( bins > 1 ).

    return_stats : bool = False
        When True, next to the gini coefficent, FPR and TPR is returned.
    
    show_plt : bool = False
        Graphical Output → show Gini Coefficent plot - distribution and probability behavior.

    plt_title : str = 'Gini Coefficient & Target behavior by groups of X' 
        Graphical Output → title above the Graphs.

    plt_ratio : float|int = 1
        Graphical Output → ratio of graph, lower the value, smaller the graph.
    """

    def __init__(self,
        incl_miss:   bool = True, 
        th_discrete: int = 20,                 
        show_plt:    bool = False, 
        plt_title:   str = 'Gini Coefficient & Target behavior by groups of X', 
        plt_ratio:   float|int = 1
    ):
        super().__init__(incl_miss, th_discrete)
        self.show_plt    = show_plt
        self.title       = plt_title
        self.r           = plt_ratio
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _none_x(return_pr:bool) -> tuple:
        if not return_pr:
            return 0.0        
        return 0.0, np.array([0,1]), np.array([0,1])

    @staticmethod
    def _gini(auc:float) -> float:
        return np.round(2 * auc - 1, 4)
    
    @classmethod
    def fast_calc(cls,
        y_true: ArrayLike, 
        y_pred: ArrayLike | None, 
    ) -> float:
        """
        O(N) Gini computation via histogram binning ~20x faster than
        *`calc()`*, yet Gini coef. is identical or very close to 4 dec. place.
        
        Parameters:        
        -----------
        y_true : Binary targe array.
        y_pred : The prediction array to evaluate; None → returns 0.0.

        Returns:
        --------
        float | tuple[float, NDArray, NDArray] :
            Gini coefficient statistic.
            If return_stats = True, next to Gini coeff., the PR statistics are returned.          
        """
        if y_pred is None:
            return cls._none_x(False)
                
        y_t = np.asarray(y_true, dtype=np.int8)
        y_p = np.asarray(y_pred)

        min_val, max_val = y_p.min(), y_p.max()
        if min_val == max_val:
            return 0.0

        # Quantize predictions into discrete integer bins [0, n_bins - 1]
        n_bins = int(min(len(y_true), 5e4))
        bins = ((y_p - min_val) / (max_val - min_val) * (n_bins - 1)).astype(
            np.int32
        )

        # Bin counts for events and non-events
        pos_counts = np.bincount(bins[y_t == 1], minlength=n_bins)
        neg_counts = np.bincount(bins[y_t == 0], minlength=n_bins)

        # Reverse bin order (sweep from HIGHEST predicted score to LOWEST)
        cum_pos = np.cumsum(pos_counts[::-1])
        cum_neg = np.cumsum(neg_counts[::-1])

        total_pos = cum_pos[-1]
        total_neg = cum_neg[-1]

        if total_pos == 0 or total_neg == 0:
            return 0.0

        # Prepend 0.0 so the ROC curve originates at (FPR=0, TPR=0)
        tpr = np.concatenate(([0.0], cum_pos / total_pos))
        fpr = np.concatenate(([0.0], cum_neg / total_neg))

        # Vectorized trapezoidal integration (AUC)
        auc = 0.5 * np.sum((tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1]))

        return cls._gini(auc)

    @classmethod
    def calc(cls,
        y_true: ArrayLike, 
        y_pred: ArrayLike | None, 
        return_pr: bool = False
    ) -> float | tuple[float, NDArray, NDArray]:
        """
        Base Gini calculation.
        
        Parameters:
        -----------
        y_true : Binary targe array.
        y_pred : The prediction array to evaluate; None → returns 0.0.

        Returns:
        --------
        float | tuple[float, NDArray, NDArray] :
            Gini coefficient statistic.
            If return_stats = True, next to Gini coeff., the PR statistics are returned.        
        """
        if y_pred is None:
            return cls._none_x(return_pr)
    
        fpr, tpr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        
        gini = cls._gini(roc_auc)
        
        if not return_pr:
            return gini
        
        return gini, fpr, tpr
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def univar(self, 
        data:   pDF, 
        target: str, 
        x:      str, 
        bins:   int = None
    ) -> float | tuple[float, NDArray, NDArray]:
        """
        Univariate Gini Performance for Binary Target and any dtype of a Feature.    
        
        Parameters
        ----------
        data   : Input Data containing target & x on which basis Gini will be computed.
        target : Target Variable present in the data. 
        x      : Explanatory Feature for which Gini is computed. 
        bins   : Num var x allows to approach gini by 2 options: 
                 - bins > 0: var is firstly discretized to number of chosen bins 
                 - bins = None: gini computed on values of x in the data
    
        Returns
        -------
        (float | tuple[float, NDArray, NDArray]) : Gini Coefficient | Gini with pr stats
        """        
        
        # y_true, x_fe
        y = data[target]
        x_, woe, length = Encoding(
            n_bins      = bins, 
            incl_miss   = self.incl_miss, 
            th_discrete = self.th_discrete
        ).process(y, data[x])

        # no data to compute gini
        if x_ is None or length < 2:
            return self._none_x(self.return_stats)
        
        # opt. missing exluded (incl_miss = False)
        m = x_.isnull()
        if m.sum():
            y  = y[~m]
            x_ = x_[~m]

        gini, fpr, tpr = self.calc(y, x_, True)

        # Gini Graph
        if self.show_plt:
            if woe is None:
                x_  = Pd.qcut(x_, 10, duplicates='drop')
                woe = WoE.encode(y, x_)

            self._graph(x, woe, gini, fpr, tpr)
        
        if not self.return_stats:
            return gini
        
        return gini, fpr, tpr
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def _graph(self,
        x:    str, 
        woe:  pDF, 
        gini: float, 
        fpr:  NDArray, 
        tpr:  NDArray, 
    ):    
        bx = woe[x].astype(str)
        titles = [f'Gini Coefficient {round(gini,2):.2f}', 'Target Rate by Intervals']
        
        fig = make_subplots(
            rows  = 1, cols  = 2,
            specs=[ [{"secondary_y": False}, {"secondary_y": True}] ],
            subplot_titles = titles
        )
        fig.add_trace(go.Scatter(
            x = [0,1], y = [0,1], showlegend = False
            ), row=1, col=1
        )
        fig.add_trace(go.Scatter(
            x = fpr, y = tpr, fill = 'tonexty', name = 'ROC'
            ), row=1, col=1
        )
        fig.add_trace(go.Bar(    
            x = bx, y = woe['count'], 
            name = 'Counts', 
            marker = dict(color='rgba(50, 100, 200, 0.3)')
            ), row=1, col=2
        )
        fig.add_trace(go.Scatter(
            x = bx, y = woe['mean'],  
            name = 'Target Rate', 
            mode = 'lines', 
            line = dict(color='rgb(200, 50, 50)')
            ), secondary_y=True, row=1, col=2
        ) 
        font = lambda x=15: dict(
            font=dict(size=x)
        )
        fig.update_annotations(font(13))

        title = {
            'text': self.title, 'y':0.92, 'x':0.45, 
            'xanchor': 'center','yanchor': 'top',
            
        } | font(15)

        t_axis = lambda t, x=12, y=11: dict( 
            title=dict(text=t, font=dict(size=x)), tickfont=dict(size=y) 
        )
        grid  = lambda w=0.5: dict(
                gridwidth = w, 
                gridcolor = f'rgba(255, 255, 255, {w})',
        )        
        fig.update_layout( 
            height = int(430*self.r), 
            width  = int(1050*self.r), 
            title  = title, 
            showlegend = False,

            xaxis1 = t_axis("False Positive Rate"),
            yaxis1 = t_axis("True Positive Rate"),
            xaxis2 = t_axis(f"Bins of {x}") | dict(tickangle=30),
            yaxis2 = t_axis("Count") | grid(0),
            yaxis3 = t_axis("Target Rate %")
        )
        fig.show()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SeparationMetrics(MetricsBase):

    """
    Separation Metrics: gini | ks | iv | lift.    
    
    Parameters
    ----------
    incl_miss : bool = True
        Calculation will include missing part by transforming variable to WOE form.
    
    th_discreate : int = 20
        Threshold above which variable is considered as continuous only relevant for 
        num type variable and binning on ( bins > 1 ).

    return_stats : bool = False
        When True, next to the key metric, extra statistics are returned.
    """

    def gini(self,
        y_true: pSe|NDArray, 
        y_pred: pSe|NDArray,
    ) -> float | tuple[float, NDArray, NDArray]:
        """
        Calculate the Gini coefficient statistic.
        
        Parameters:
        -----------
        y_true : Binary targe array.
        y_pred : The prediction array to evaluate.

        Returns:
        --------
        float | tuple[float, NDArray, NDArray] :
            Gini coefficient statistic.
            If return_stats = True, next to Gini coeff., the PR statistics are returned.
        """        
        return Gini.calc(y_true, y_pred, self.return_stats)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def ks(self,            
        y_true: pSe|NDArray, 
        y_pred: pSe|NDArray,
        sample_weight: pSe|NDArray = None
    ) -> float | tuple[float, int, dict[str, NDArray]]:
        """
        Calculate the Kolmogorov-Smirnov (KS) statistic.

        Parameters:
        -----------
        y_true : Binary targe array.
        y_pred : The prediction array to evaluate.
        sample_weight : Optional array of sample weights. 

        Returns:
        --------
        float | tuple[float, int, dict[str, NDArray]] :
            Kolmogorov-Smirnov (KS) statistic.
            If return_stats = True, index point of max KS and dictionary with KS statistics.
        """

        # 1. Standardize inputs
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if sample_weight is None:
            sample_weight = np.ones_like(y_true, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight)
        
        # 2. Sorting: Highest risk first
        desc_idx = np.argsort(y_pred, kind='stable')[::-1]
        y_true_sorted  = y_true[desc_idx]
        weights_sorted = sample_weight[desc_idx]

        # Boolean masks to avoid integer comparison overhead in loops
        is_bad = (y_true_sorted == 1)
        is_good = ~is_bad

        # 3. Weighted Cumulative Sums
        bad_weights  = weights_sorted * is_bad
        good_weights = weights_sorted * is_good
        
        total_bad  = bad_weights.sum()
        total_good = good_weights.sum()

        if total_bad == 0 or total_good == 0:
            return 0.0

        # 4. Calculate CDFs
        cum_bad_pct  = np.cumsum(bad_weights) / total_bad
        cum_good_pct = np.cumsum(good_weights) / total_good

        # 5. Maximum Separation
        diff     = np.abs(cum_bad_pct - cum_good_pct)
        ks_idx   = np.argmax(diff)
        ks_value = diff[ks_idx]

        if not self.return_stats:
            return ks_value
        
        return ks_value, int(ks_idx + 1), {
            'cum_bad_pct':  cum_bad_pct,
            'cum_good_pct': cum_good_pct,
            'diff':         diff,
            'threshold':    y_pred[desc_idx][ks_idx]
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def iv_total(woe: pDF | None) -> float:
        """
        Calculates the Overall Information Value (IV) for a characteristics.

        Parameters
        ----------
        woe : pDF | None
            - The calculated Weight of Evidence mapping.
            - Input woe as None will return 0.0.

        Returns
        -------
        float
            The overall Information Value of the feature.
        """
        if woe is None:
            return 0.0
        if IVL.IV not in woe.columns:
            woe = IV.extend(woe.copy())
        return float(woe[IVL.IV].sum())

    def iv(self,
        y_true: pSe|NDArray, 
        x:      pSe|NDArray,
        bins:   int = None
    ) -> float | tuple[float, pDF|None]:
        """
        Calculates the Information Value (IV) and Weight of Evidence (WoE) for a characteristic.
        
        Parameters:
        -----------
        y_true  : Binary targe array.
        x       : The feature array to evaluate.
        bins    : Num var x allows to approach IV by 2 options 
                  - bins > 0: var is firstly discretized to number of chosen bins 
                  - bins = None: IV computed on values of x in the data
            
        Returns:
        --------
        float | tuple[float, pd.DataFrame | None]:
            The overall Information Value of the feature.
            If return_stats = True, pDF as structured audit table containing WoE and IV per bucket.
        """        
        woe = Encoding(
            n_bins      = bins, 
            transform   = False, 
            ensure_enc  = True, 
            add_iv      = True, 
            incl_miss   = self.incl_miss, 
            th_discrete = self.th_discrete
        ).process(y_true, x)[1]

        iv_total = self.iv_total(woe)
        
        if not self.return_stats:
            return iv_total
        
        return iv_total, woe
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def lift(self, 
        y_true: pSe|NDArray, 
        y_pred: pSe|NDArray,
        bins:   int = 10
    ) -> pDF: 
        """
        Returns E(target) with target Lift by discretized pred to quantile bins.

        Parameters:
        -----------
        y_true : Binary targe array.
        y_pred : The prediction array to evaluate.
        bins   : y_pred discretization to distinct bins.
            
        Returns:
        --------
        pd.DataFrame : The lift statistics.
        """

        p = Pd.qcut(pSe(y_pred), bins, min_left=0, max_right=1, duplicates='drop')
        t = pSe(y_true).groupby(p).agg(**{LBL.T: 'mean', LBL.C: 'count'})
        t.index.name = LBL.P

        t[LBL.W] = t['count'] / len(y_true)
        t[LBL.L] = t[LBL.T] / np.mean(y_true)

        return t[[LBL.T, LBL.L, LBL.W, LBL.C]]
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SeparationDiagnostics:
    """
    Evaluates the separation power of a predictive model by gini, iv, ks, pr, lift.

    Parameters
    ----------
    roc_bins : int
        Probability discretization to generate ROC with detailed desc.

    lift_bins : int | None, optional
        Probability discretization to calculate E(T) and Lift stats.
    """

    def __init__(self, roc_bins: int = 100, lift_bins: int | None = None):
        self.roc_bins  = roc_bins
        self.lift_bins = lift_bins

    @staticmethod
    def _to_array(*array) -> list[NDArray]:
        r = []
        for a in array:
            if not isinstance(a, np.ndarray):
                a = np.asarray(a)
            r.append(a)
        return r
    
    def evaluate(self, 
        y_true: pSe|NDArray, 
        y_pred: pSe|NDArray,
    ) -> DiagnosticsResult:
        """
        Evaluates the separation power of the model and returns key metrics.

        Returns
        -------
        DiagnosticsResult :
        - float : Gini coefficient
        - float : Information Value (IV)
        - float : Kolmogorov-Smirnov statistic (KS)
        - dict  : [False Positive Rate, True Positive Rate, ROC PCT Metrics]
        - pDF | None : optional, (lift_bins != None) → Lift statistic | None            
        """
        lift = None

        y_true, y_pred = self._to_array(y_true, y_pred)
        p, ipr = self.compute_roc_metrics(y_true, y_pred)

        _, fpr, tpr = SeparationMetrics(return_stats=True).gini(y_true, p)
        pr = {
            LBL.FPR: fpr, 
            LBL.TPR: tpr,
            LBL.IPR: ipr
        }
        
        sp = SeparationMetrics()

        gini = sp.gini(y_true, y_pred)
        ks   = (tpr - fpr).max()
        iv   = sp.iv(y_true, p)

        if self.lift_bins:
            lift = sp.lift(y_true, y_pred, self.lift_bins)
        
        return DiagnosticsResult(
            obs     = int(len(y_true)),
            events  = int(sum(y_true)),
            gini    = gini, 
            IV      = iv,
            KS      = ks,
            PR      = pr,
            lift    = lift
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ipr(y_true: pSe|NDArray, y_pred: pSe|NDArray, is_interval: bool=True) -> pDF:
        """
        Converts probability to percentile and compute additional FPR-TPR info.
        - sort prob by descending order with respect to ROC curve fpr-tpr design
        - given prop. cutoff, compute cut of pop. and E(target) on remaining pop.                
        """
        ipr = pSe(y_true).groupby(y_pred).agg(['count', 'mean']).reset_index()
        ipr.columns = [LBL.PC, LBL.CS, LBL.T]

        ipr[LBL.CS] = ipr[LBL.CS] / ipr[LBL.CS].sum()
        ipr[LBL.T]  = ipr[LBL.T] * ipr[LBL.CS]
        
        if is_interval:
            ipr[LBL.PC] = [0.0] + [x.left for x in ipr[LBL.PC][1:]]
        ipr.loc[ipr.index[-1]+1, :] = [1.0, 0.0, 0.0]

        ipr = ipr.sort_values(LBL.PC, ascending=False)
        
        ipr[[LBL.CS, LBL.T]] = ipr[[LBL.CS, LBL.T]].cumsum()
        ipr[LBL.TR] = ipr[LBL.T].iloc[-1] - ipr[LBL.T]
        
        return ipr.round(4)
    
    @staticmethod
    def mark_pr_optim_point(pr: dict) -> int:
        """
        Returns (close to) optim point after which the trade-off between TPR/FPR becomes < 1.
        """
        dt = np.gradient(pr[LBL.TPR])
        df = np.gradient(pr[LBL.FPR])
        df[df==0] = np.inf
        
        # sweep for opt from both sides + Youden's J
        s  = np.abs((dt / df) - 1)
        m1 = s.argmin()
        m2 = len(dt)-1 - s[::-1].argmin()
        m3 = (pr[LBL.TPR] - pr[LBL.FPR]).argmax()
        return int((m1+m2+m3)/3)

    @classmethod
    def get_pr_optim_cutoff(cls, pr: dict) -> float:
        """
        Returns (close to) optim proba cutoff after which the trade-off between TPR/FPR becomes < 1.
        """
        m = cls.mark_pr_optim_point(pr)
        ipr: pDF = pr[LBL.IPR]
        i = len(ipr)-1 - m
        return ipr.loc[i, LBL.PC]

    def compute_roc_metrics(self, 
        y_true: pSe|NDArray, 
        y_pred: pSe|NDArray,
    ) -> tuple[pSe, pDF]:
        """
        Converts probability to percentile and compute additional FPR-TPR info.
        - sort prob by descending order with respect to ROC curve fpr-tpr design
        - given prop. cutoff, compute cut of pop. and E(target) on remaining pop.
        
        Returns
        -------
        tuple:
        - Series: Probability pct codes | obs < 200 orig proba
        - Dataframe: FPR-TPR target cutoff behavior by proba pct
        """                
        y_true_, y_pred_ = self._to_array(y_true, y_pred)

        b = min(self.roc_bins, len(y_true_) // 20)   
        if b < 10:
            return y_pred, self._ipr(y_true_, y_pred_, False)
        
        p_ = Pd.qcut(pSe(y_pred_), b, duplicates='drop')
        return p_.cat.codes, self._ipr(y_true_, p_)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class PlotTheme:
    """RGBA color 0-39 map to 10 distinct themes of 4 colors each."""    
    
    COLOR = {
        # --- Cyberpunk ---
        0: 'rgb(  5, 175, 120)', # Laser Emerald
        1: 'rgb(165,  75, 155)', # Electric Orchid
        2: 'rgb( 40, 140, 170)', # Baltic Cyan
        3: 'rgb( 85,  85, 100)', # Dark Obsidian  

        # --- High-Contrast Tech ---
        4: 'rgb(  0, 210, 150)', # Electric Mint
        5: 'rgb(255, 105,  95)', # Hi-Fi Coral
        6: 'rgb( 85,  80, 160)', # Cyber Indigo
        7: 'rgb(110, 120, 130)', # Charcoal Slate                          

        # --- Quantum ---
        8:  'rgb( 15,  90, 185)', # Cobalt Blue
        9:  'rgb(190,  55,  90)', # Crimson Rose
        10: 'rgb( 95, 125,  65)', # Olive Forest
        11: 'rgb(185, 190, 195)', # Pale Platinum                

        # --- Warm ---
        12: 'rgb( 35, 120, 100)', # Imperial Jade
        13: 'rgb(215, 140,  45)', # Spiced Amber
        14: 'rgb(165,  80,  70)', # Venetian Red
        15: 'rgb(160, 155, 150)', # Sandstone Gray                

        # --- Earth/Terracotta ---
        16: 'rgb(210, 105,  55)', # Terracotta Copper
        17: 'rgb( 30,  65, 110)', # Deep Ocean Navy
        18: 'rgb( 90, 145, 170)', # Sea Glass Blue
        19: 'rgb(170, 175, 180)', # Soft Pebble                

        # --- Tech ---
        20: 'rgb( 33, 115, 115)', # Deep Teal
        21: 'rgb( 70, 120, 150)', # Steel Blue
        22: 'rgb(120, 135, 145)', # Slate Gray
        23: 'rgb(175, 185, 190)', # Light Ash

        # --- Muted ---
        24: 'rgb(195, 125,  60)', # Burnished Amber
        25: 'rgb( 70,  90, 130)', # Deep Indigo
        26: 'rgb(105, 120, 150)', # Twilight Blue
        27: 'rgb(140, 145, 160)', # Muted Ink

        # --- Monochromatic ---
        28: 'rgb( 25,  50,  90)', # Midnight Navy
        29: 'rgb( 60,  95, 140)', # Royal Slate
        30: 'rgb(110, 140, 180)', # Cadet Slate
        31: 'rgb(180, 195, 215)', # Ice Glaze

        # --- Emerald & Earth Academic ---
        32: 'rgb( 45, 110,  85)', # Racing Green
        33: 'rgb(115, 135, 105)', # Olive Sage
        34: 'rgb(150, 135, 120)', # Warm Taupe
        35: 'rgb(175, 175, 170)', # Driftwood Gray
        
        # --- Neutral/Extra Fallbacks ---
        36: 'rgb( 80,  80,  80)', # Dark Gray
        37: 'rgb(120, 120, 120)', # Medium Gray
        38: 'rgb(160, 160, 160)', # Light Gray
        39: 'rgb(200, 200, 200)', # Pale Gray
    }

    def __init__(self, colors: list = None):
        if colors:
            self.COLOR = {i: c for i,c in enumerate(colors)}

    @staticmethod
    def _add_opacity(c: str, opacity: float) -> str:
        c_ = c.replace('rgb(', 'rgba(')
        return c_.replace(')', f', {opacity})')

    def get_color(self, a: int, opacity: float = 1.0) -> str:
        """
        Returns an RGBA color string based on index 'a'.
        Indices 0-39 map to 10 distinct themes of 4 colors each.
        """
        l = len(self.COLOR)-1
        if a > l:
            return self._add_opacity(self.COLOR[l], opacity)
        return self._add_opacity(self.COLOR[a], opacity)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class GiniLiftPlot:
    """
    Constructs Gini and Lift plots with detailed ROC behavior description. 
    Natively supports multiple models as input for comparison and analytics.

    Parameters
    ----------
    eval_stats : DiagnosticResult
        Evaluation stats output from SeparationDiagnostic.evaluate.
        - evaluate() direct output for 1 model
        - dict(str, evaluate()) for n models

    visible_agent : int| str | list, optional, default None
        When eval_stats contains multiple models, this variable controls which
        models should be visible, the rest as legend only - user iteraction.

    theme : type | list, optional, default PlotTheme
        Color theme for plots:
        - list as list of rgb colors in order of eval_stats
        - when custom class is provided, the class contract must follow:
        - implemented method > get_color(a: int, opacity: float), 
        where *a* is model position in provided eval_stats
    
    plot_titles : list, default None
        Optional titles for subplots Gini and Lift.

    update_layout : dict, optional, default None
        Final figure layout update by plotly syntax.
    """

    def __init__(self, 
        eval_stats:     DiagnosticsResult | dict[str, DiagnosticsResult], 
        visible_agent:  str|list = None,
        theme:          PlotTheme|type|list = PlotTheme,
        plot_titles:    list = None,
        update_layout:  dict = None
    ):
        self._set_input(eval_stats, visible_agent)
        self._theme = self._theme_adapter(theme)
        self.plot_titles = plot_titles
        self._update_layout = update_layout
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def title_helper(
        diagnostics: dict[str, DiagnosticsResult], 
        title_key_fx: FuncType = lambda x: x.split(' ')[0],
        div: str = ' | '
    ) -> dict:
        """Helper to provide formatted title with gini and lift values by diagnostics keys."""
        t = title_key_fx
        g = lambda x: div.join([f'{t(s)}: {d[x]:.2%}' for s, d in diagnostics.items()])
        l = lambda x: div.join([f'{t(s)}: {d[x][LBL.L].max():.3f}' for s, d in diagnostics.items()])
        s = Dataclass.schema(DiagnosticsResult)
        return {
            LBL.G: g(s.gini),
            LBL.L: l(s.lift)
        }  
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def _set_input(self, stats:dict, visible_agent:int|str|list) -> None:

        if isinstance(stats, DiagnosticsResult):
            stats = {'Model': stats}

        if not all([isinstance(x, DiagnosticsResult) in x for x in stats.values()]):
            raise ValueError(
            'AIG SAFETY\n► Valid Input is either direct output of SeparationDiagnostic.evaluate'
            '\n► Or dictionary of various models defined as dict(str, evaluate) ► '
            'each sub-dict must be output of SeparationDiagnostic.evaluate for a given model'
            )
        
        self._names = list(stats.keys())
        self._stats: dict[int, DiagnosticsResult] = {i:s for i,s in enumerate(stats.values())}
        
        vis = visible_agent if visible_agent is not None else self._names
        self._vis = [self._names.index(x) for x in List.ensure(vis) if x in self._names]
    
    @staticmethod
    def _theme_adapter(theme: PlotTheme|type|list) -> PlotTheme:
        if theme is PlotTheme:
            return theme()
        if isinstance(theme, list) :
            return PlotTheme(theme)
        return theme        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_fpr_tpr(self, a:int) -> dict:
        return self._stats[a].PR

    def _get_gini_vector(self, a:int) -> NDArray:
        s = self._stats[a]
        g = s.gini
        ipr = s.PR[LBL.IPR]
        return np.array([g]*len(ipr))
    
    def _get_lift(self, a:int) -> pSe:
        return self._stats[a].lift
    
    def _get_name(self, a:int) -> str:
        return self._names[a]
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _axis(
        title:  str, 
        grid_w: float = 0.5, 
        font:   dict  = None, 
        extras: dict  = {}
    ) -> dict:
        font = font or dict(family="Arial", size=12, color="black")

        return dict(title = dict(text = title, font = font)) | dict(
            gridwidth = grid_w, 
            gridcolor = f'rgba(255, 255, 255, {grid_w})',
        ) | extras
    
    def _trace_props(self, a:int) -> dict:
        return dict(
            name = self._get_name(a),                
            fill = 'tonexty',
            line = dict(
                color = self._theme.get_color(a, 1.0),
                width = 2 
            ),
            fillcolor = self._theme.get_color(a, 0.4),
            visible = True if a in self._vis else 'legendonly',
        )
    
    def _mark_max(self, a:int) -> dict:
        """
        Highlight (close to) optim point after which the trade-off between TPR/FPR becomes < 1.
        """
        pr = self._get_fpr_tpr(a)
        m  = SeparationDiagnostics.mark_pr_optim_point(pr)
        
        r = range(len(pr[LBL.FPR]))
        marker_colors = [self._theme.get_color(a, 0.9) if i == m else 'rgba(0,0,0,0)' for i in r]
        marker_sizes  = [10 if i == m else 0 for i in r]

        return dict(
            color  = marker_colors,
            size   = marker_sizes,
            symbol = 'star',
            line   = dict(width=1, color=self._theme.get_color(a, 1.0))
        )
    
    def _legend(self) -> dict:
        return dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)

    def _lift_mean(self) -> str:
        m = []
        for a in self._stats:
            d = self._get_lift(a)
            t = ( d[LBL.T] * d[LBL.W] ).sum()
            m.append( f'{t:.1%}')
        return ' | '.join(list(dict.fromkeys(m)))        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def gini(self, is_grid:bool=False) -> go.Figure:

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = [0,1], 
            y = [0,1], 
            name = '45',
            line = dict(color = 'black', dash = 'dot', width = 1),
            showlegend = False
        ))
    
        for a in reversed(self._stats):
            pr = self._get_fpr_tpr(a)
            fig.add_trace(go.Scatter(
                x = pr[LBL.FPR], 
                y = pr[LBL.TPR], 
                mode = 'lines+markers',
                marker = self._mark_max(a),
                legend = "legend1",
                showlegend = True,
                customdata = np.column_stack((
                    pr[LBL.IPR][[LBL.PC, LBL.CS, LBL.T, LBL.TR]].values,
                    self._get_gini_vector(a)
                )),
                hovertemplate = (
                    'Gini: %{customdata[4]:.1%}<br>'
                    '---------------------------<br>'
                    'TPR: %{y:.1%}<br>'
                    'FPR: %{x:.1%}<br>'
                    '---------------------------<br>'
                    'Prob. Cutoff: %{customdata[0]:.3f}<br>'
                    '---------------------------<br>'
                    'Pop. Cut: %{customdata[1]:.1%}<br>'
                    'E(T) Cut: %{customdata[2]:.1%}<br>'
                    'E(T) Remain: %{customdata[3]:.1%}<br>'
                ),                
                **self._trace_props(a)
            ))

        fig.update_layout(
            height = 600,
            width  = 600,
            xaxis  = self._axis("False Positive Rate"),
            yaxis  = self._axis("True Positive Rate"), 
            legend = self._legend(),    
        )

        return fig
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def lift(self, is_grid:bool=False) -> go.Figure:

        lx = max([len(self._get_lift(a)) for a in self._stats])
        x  = list(range(lx))

        def _align_len(d: pDF) -> pDF:
            l = len(d)
            if l == lx:
                return d
            z = lx - l
            m = d.median()
            i = np.isclose(d[LBL.T], m[LBL.T]).argmax()
            n = pDF([m]*z, index = [d.iloc[i].name]*z)
            n[[LBL.W, LBL.C]] = 0, 0
            return Pd.concat([d,n]).sort_index()
        
        def _legend():
            return dict(
                legend = "legend1" if not is_grid else 'legend2',
                showlegend = True
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = x, 
            y = [1]*len(x), 
            name = 'E(T)',
            line = dict(color = 'black', dash = 'dot', width = 1),
            hovertemplate = self._lift_mean(),
            **_legend()
        ))

        ticks = []
        y_max = 0
        for a in reversed(self._stats):
            d = _align_len( self._get_lift(a) )
            l = d[LBL.L]
            fig.add_trace(go.Scatter(
                x = x, 
                y = l.values, 
                customdata = d[[LBL.T, LBL.W]].values,
                hovertemplate = (
                    'E(Lift): %{y:.3f}<br>'
                    'E(T|Bin): %{customdata[0]:.1%}<br>'
                    'Bin Share: %{customdata[1]:.1%}'
                ),
                **_legend(),
                **self._trace_props(a),
            ))
            ticks.append(d[[LBL.T]].values)
            y_max = max(l.max(), y_max)
        
        ticks = np.average(ticks, axis=0).round(2)
        xaxis = dict(tickvals = x, ticktext = ticks)
        yaxis = dict(range = [-0.2, y_max + 0.2])

        fig.update_layout(
            height = 600,
            width  = 600,
            xaxis  = self._axis("E(Target | Bin)", extras = xaxis),
            yaxis  = self._axis("E(Lift)", extras = yaxis),
            legend = self._legend(),            
        )

        return fig
    # ─────────────────────────────────────────────────────────────────────────────────────────────    

    def create(self, show:bool = True) -> go.Figure | None:
        
        graph = lambda x: self.gini(True) if not x else self.lift(True)
        t_def = lambda x: 'Gini Performance' if not x else 'Lift Performance'
        title = lambda x: self.plot_titles[x] if self.plot_titles is not None else t_def(x)

        fig = Plot.create_subplots(
            [0, 1], 
            graph, 
            title,
            grid_cols = 2,
            height = 450,
            width  = 530,
            horizontal_spacing = 0.17,
            update_layout = dict(
                margin = dict(t=20, b=20, l=0, r=0),
                showlegend = True,
                legend1 = dict(
                    orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.22
                ),
                legend2 = dict(
                    orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.77
            )) | (self._update_layout or {})
        )        

        if not show:
            return fig
        
        fig.show()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ConfusionMatrixPlot:
    """
    Generates a confusion matrix.
    Available themes: light, dark, violet, warm, ocean.
    """

    THEMES = {
        "light": {
            "font_color": "#0f172a",
            "colorscale": ["#f8fafc", "#e0e7ff", "#6366f1", "#312e81"],
            "grid_gap": 3,
            "table_header_bg": "#DADCED",
            "table_row_even": "#ffffff",
            "table_row_odd": "#EBEDF5",
            "table_row_optim": "#dbeafe"
        },
        "dark": {
            "font_color": "#f4f4f5",
            "colorscale": ["#18181b", "#064e3b", "#10b981", "#a7f3d0"],
            "grid_gap": 4,
            "table_header_bg": "#27272a",
            "table_row_even": "#18181b",
            "table_row_odd": "#09090b",
            "table_row_optim": "#374151"
        },
        "violet": {
            "font_color": "#e5e5e5",
            "colorscale": ["#171717", "#4c1d95", "#8b5cf6", "#ddd6fe"],
            "grid_gap": 4,
            "table_header_bg": "#4c1d95",
            "table_row_even": "#000000",
            "table_row_odd": "#0f0f12",
            "table_row_optim": "#5b21b6"
        },
        "warm": {
            "font_color": "#1c1917",
            "colorscale": ["#fef3c7", "#fde047", "#f59e0b", "#9a3412"],
            "grid_gap": 3,
            "table_header_bg": "#fef3c7",
            "table_row_even": "#fffbeb",
            "table_row_odd": "#fef9c3",
            "table_row_optim": "#fcd34d"
        },
        "ocean": {
            "font_color": "#334155",
            "colorscale": ["#f0fdfa", "#5eead4", "#0ea5e9", "#0c4a6e"],
            "grid_gap": 2,
            "table_header_bg": "#ccfbf1",
            "table_row_even": "#f0fdfa",
            "table_row_odd": "#f8fafc",
            "table_row_optim": "#7dd3fc"
        }
    }

    def __init__(self,
        names: tuple = (0, 1), 
        theme: str   = "light",
        title: str   = None,
        plot_titles: list = None
    ):
        self._verify(theme)
        self.names = names
        self.title = title                
        self.subtt = plot_titles
        self.theme = self.THEMES[theme]

    def _verify(self, theme:str) -> None:        
        if theme not in self.THEMES:
            raise ValueError(f"Theme '{theme}' not found. Choose from {list(self.THEMES.keys())}")
    
    @staticmethod
    def get_cmatrix(y_true: pSe|NDArray, y_pred: pSe|NDArray) -> NDArray:
        lbl = y_true.unique()
        lbl.sort()
        return confusion_matrix(y_true, y_pred, labels=lbl)

    def heatmap(self, confusion_mx:NDArray, showscale:bool = False, relat_width:float=0.4) -> go.Figure:
        """Generates a confusion matrix."""
        row_sums = confusion_mx.sum(axis=1)[:, np.newaxis]
        cm_pct = np.divide(
            confusion_mx, row_sums, 
            out=np.zeros_like(confusion_mx, dtype=float), where=row_sums != 0
        ) * 100

        text_matrix = [
            [f"<b>{count:,}</b><br>({pct:.1f}%)" for count, pct in zip(row_count, row_pct)]
            for row_count, row_pct in zip(confusion_mx, cm_pct)
        ]
        t = self.theme

        fig = go.Heatmap(
            z            = confusion_mx,
            x            = self.names,
            y            = self.names,
            text         = text_matrix,
            texttemplate = "%{text}",
            textfont     = dict(size=15),
            showscale    = showscale,
            colorscale   = t["colorscale"],
            xgap         = t["grid_gap"], 
            ygap         = t["grid_gap"],
            colorbar = dict(
                title = dict(
                    text = "Volume",
                    font = dict(color=t["font_color"])
                ),
                len          = 0.75, 
                thickness    = 10, 
                outlinewidth = 0, 
                x            = relat_width+0.02, 
                xanchor      = "left",
                tickfont     = dict(size=11, color=t["font_color"])
        ))
        return fig
    
    def table(self, table_data: dict, row_highlight: int = None) -> go.Figure:
        """Generates a data table."""
        t = self.theme
        num_rows = len(next(iter(table_data.values())))
        
        row_colors = [
            t["table_row_even"] if i % 2 == 0 else t["table_row_odd"] 
            for i in range(num_rows)
        ]
        if row_highlight is not None:
            row_colors[row_highlight] = t["table_row_optim"]

        return go.Table(
            header=dict(
                values = list(table_data.keys()),
                font   = dict(color = t["font_color"], size = 11.5, weight = 600),
                align  = "center",
                height = 40,
                fill_color = t["table_header_bg"],
            ),
            cells=dict(
                values = [list(v) for v in table_data.values()],
                font   = dict(color = t["font_color"], size = 12),
                align  = "right",
                height = 30,
                fill_color = [row_colors] * len(table_data),
            ),
            columnwidth=[90] * len(table_data)
        )
    
    def update_layout(self, fig: go.Figure, width:int = 1000, height:int = 400) -> None:
        """Applies formatting to confusion matrix."""
        fig.update_layout(
            width  = width, 
            height = height,
            title  = {
                "text": f"<b>{self.title}</b>",
                "y": 0.92, "x": 0.5, "xanchor": "center", "yanchor": "top",
                "font": dict(size=20)
            } if self.title else None,
            
            annotations = [go.layout.Annotation({'font':dict(family='Arial', size=12)})],

            yaxis = dict(autorange="reversed", type="category", tickfont=dict(size=13)),
            xaxis = dict(type="category", side="bottom", tickfont=dict(size=13)),
            xaxis_title = "Predicted Class",
            yaxis_title = "Actual Class",
            
            margin = dict(t=20, r=0, b=40, l=0),
            font   = dict(family="Arial", color=self.theme["font_color"]),
            
            plot_bgcolor  = "rgba(0,0,0,0)",
            paper_bgcolor = "rgba(0,0,0,0)",            
        )

    def create(self, 
        y_true: pSe|NDArray, 
        y_pred: pSe|NDArray, 
        table_data:   dict = None, 
        row_highlight: int = None
    ) -> go.Figure:
        """Generates a confusion matrix with optinal data table."""

        cm = self.get_cmatrix(y_true, y_pred)
        cw = [0.4, 0.6]
        hs = 0.12

        fig = make_subplots(
                rows = 1, cols = 2,
                column_widths = cw,
                specs = [[{"type": "heatmap"}, {"type": "table"}]],
                subplot_titles = self.subtt,
                horizontal_spacing = hs
        )

        fig.add_trace(self.heatmap(cm, showscale=True, relat_width=cw[0]-hs/2), row=1, col=1)
        
        if table_data is not None:
            fig.add_trace(self.table(table_data, row_highlight), row=1, col=2)

        self.update_layout(fig, 1070, 400)

        return fig
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬