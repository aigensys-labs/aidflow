"""
Weight of Evidence (WoE) and Information Value (IV) implementation engine.

This module provides the fundamental transformation logic required for GLM-based 
binary classification, specifically tailored for industrial scorecard development. 
It implements a robust WoE encoding framework that converts categorical and binned 
numerical features into a linear scale relative to the target probability, while 
employing data-driven imputation strategies to handle zero-event bins without 
introducing artificial bias.

In addition to encoding, the library provides an Information Value (IV) suite for 
quantifying the predictive strength of individual features through symmetric 
divergence analysis. This is complemented by a comprehensive overview tool that 
assesses the stability of WoE values across different data samples, enabling 
developers to detect potential distribution drift and ensure that feature 
predictivity remains consistent between training and validation environments.
"""

from ...deps.data           import np, pd, pDF, pSe
from ...deps.plots          import go, make_subplots
from ...deps.types          import NDArray

from ...core.factory        import DictClass
from ...core.base.adapter   import ModelAdapterBase
from ...core.interop        import Pd
from ...core.logging        import Logger
from ...core.accessor       import Spinner
from ...core.standard       import VarStd

from ...utils.builtins      import List, Str
from ...display.plots       import Plot
from ...display.rgb         import RgbANSI

from .binning               import Bins, EncodingMap
from ..tests                import TestSampleDiff


logger = Logger(__name__)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LABELS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class STATS(DictClass):
    """mean - count - sum"""
    M = 'mean'
    C = 'count'
    S = 'sum'

class FLAG(DictClass):
    BADS  = 'FLAG'
    GOODS = 'NONF'

class IVL(DictClass):
    """IV Labels"""
    IV  = 'IV'
    STD = 's(IV)'

class OV(DictClass):
    """WoE Overview Labels"""
    T    = lambda x: f'E(Target|{x})'
    IV   = lambda x: f'E(IV|{x})'
    SIV  = lambda x: f's(IV|{x})'
    C    = lambda x: f'Count|{x}'
    D    = 'IV_DriftΔ'
    IS   = 'SignificantΔ'
    FE   = 'Feature'
    DATA = 'data'
    SUM  = 'summary'

class LBL(DictClass):
    """gini - data - verdict - gmx - graph"""
    GINI    = 'gini'
    DATA    = 'data'
    VERDICT = 'verdict'
    GMX     = 'gmx'
    GRAPH   = 'graph'

class DASH(DictClass):
    """data - fig - order"""
    DATA  = 'data'
    FIG   = 'fig'
    ORDER = 'order'

_WOE = VarStd.WOE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class Predictor:
    """
    Lazy import wrap of Inference Engine — Production Scoring API.
    - `Applicable for projects with stored (save=1) models only.`    
    """
    def __init__(self):
        from ...predictor import Predictor
        self.predictor = Predictor

    def __call__(self, *args, **kwds):
        return self.predictor(*args, **kwds)

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# WEIGHT OF EVIDENCE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class WoE:
    """
    Weight of Evidence encoding class - fundamental for GLM with binary objective.
    
    Methodology
    -----------
    (1) 
    WOE = FLAG / NONFLAG, 
    where FLAG: target = 1; NONFLAG: target = 0.

    - Positive WOE for higher target (default) rate → positive beta signs in GLM
    - Higher the WOE, higher the chance of the FLAG Occurrence in a prediction          

    (2)
    For Bins where FLAG = 0 | NONFLAG = 0 → Assigning best found data option:
    - Imputation of WOE value without affecting original WOE of filled bins
    - Statistically better treatment than assigning random small value (e.g. +0.5)
    - This approach preffers data driven behavior of relevant data neighbors and assumes that pure 0 is lack of data
    - Imputed WOE value is data-wise reasonably lowest[b==0]/highest[g==0] WOE value        

    (3) 
    Adjust Missing in WoE.extend():
    if   m_n < 50:  M_WOE = 0                   → For low number of obs → average target rate
    elif m_b == 0:  M_WOE = data[_woe].min()    → If  TARGET = 0 → lowest  WOE found in the data             
    elif m_g == 0:  M_WOE = data[_woe].max()    → If ~TARGET = 0 → highest WOE found in the data     
    """
        
    @staticmethod
    def _verify_grouped_data(data, is_woe: bool = False):
        cols = STATS.list()
        if is_woe:
            cols = cols + [_WOE]
        if [c for c in cols if c not in data.columns]:
            raise ValueError(f'\n\
        {RgbANSI.bold_red("AIG SAFETY")}\n\
        ► Provided data in a grouped form to WOE calculation must contain following colums: {cols}\n\
        ► Count represents total number of observation in a bin (group)\n\
        ► Sum represents sum of TARGET variable in a bin (group)'
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _group_data(y_true: pSe, x: pSe) -> pDF:
        return ( 
            y_true
            .groupby(x, observed=True)
            .agg(STATS.list())
            .reset_index().replace(np.nan,0) 
        )
    
    @staticmethod
    def _align_input(y_true: pSe|NDArray, x: pSe|NDArray) -> tuple[pSe, pSe]:
        f  = lambda s: s if isinstance(s, pSe) else pSe(s)
        y_ = f(y_true)
        x_ = f(x)
        if not y_.name:
            y_.name = 'target'
        if not x_.name:
            x_.name = 'feature'
        if y_.name == x_.name:
            y_ = y_.copy()
            y_.name = y_.name + '_'
        return y_, x_
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _impute(b:pSe, g:pSe, bs:float, gs:float) -> tuple[pSe, pSe]:
        """Impute WOE stats needed to compute WOE.

        For Bins where TARGET = 0 | NONTARGET = 0 → Assigning best found data option:
        - Imputation of WOE value without affecting original WOE of filled bins
        - Statistically better treatment than assigning random small value
        - This approach preffers data driven behavior of relevant data neighbors and assumes that pure 0 is lack of data
        - Imputed WOE value is data-wise reasonably lowest[b==0]/highest[g==0] WOE value
        
        Parameters
        ----------
        b :  Sum of Bads  by bins
        g :  Sum of Goods by bins
        bs : Total Bads
        gs : Total Goods
        """

        m_b_0 = ( b == 0 )
        m_g_0 = ( g == 0 )

        def _imp(b_g):
            m  = ( b_g != 0 )
            b_ = b[m] / sum(b[m])
            g_ = g[m] / sum(g[m])
            return b_, g_
                    
        if m_b_0.sum():         
            b_, g_ = _imp(b)            
            imp = ( b_ / g_ ).min() 
            inv = ( bs / gs ) * g[m_b_0]
            b.loc[m_b_0] = imp * inv
        
        if m_g_0.sum():
            b_, g_ = _imp(g)
            imp = ( g_ / b_ ).min() 
            inv = ( gs / bs ) * b[m_g_0]
            g.loc[m_g_0] = imp * inv

        return b, g
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _adj_missing(data:pDF, x:str) -> pDF:
        """
        Missing" category special treatment to maximize quality of derived WOE.

        Parameters
        ----------
        data : Input Data in a form of -> grouped <- data by explanatory variable
            with total "count", and "sum" of Target variable.
        x : Variable in data to be Encoded to WOE Form.

        Returns
        -------
        DataFrame : Data with calculated WOE
        """
        m = ( data[x] == VarStd.MISSING )
        if not m.sum():
            return data
        
        m_w = data[m].iloc[0]
        
        M_WOE, ORG = [m_w[_WOE]]*2

        m_b = m_w[STATS.S]
        m_n = m_w[STATS.C]
        m_g = m_n - m_b

        if   m_n < 50:  M_WOE = 0                   # For low number of obs → average target rate
        elif m_b == 0:  M_WOE = data[_WOE].min()    # If  TARGET = 0 → lowest  WOE found in the data             
        elif m_g == 0:  M_WOE = data[_WOE].max()    # If ~TARGET = 0 → highest WOE found in the data            
        
        if M_WOE is not ORG:
            data.loc[m, _WOE] =  M_WOE

        return data
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    #     
    @classmethod
    def _encode(cls, data:pDF) -> pDF:
        """Extends data by WOE and stats needed to compute WOE.

        Parameters
        ----------
        data : Input Data in a form of -> grouped <- data by explanatory variable    
            with total "count", and "sum" of Target variable    
        
        Returns
        -------
        DataFrame : Data with calculated WOE            
        """
        d = data.copy()
        
        # Clean bins where both FLAG & NONFLAG = 0
        d_out  = d[( d[STATS.S] == 0 ) & ( d[STATS.C] == 0 )]
        is_out = ( len(d_out) != 0 )

        if is_out:
            d = d.drop(d_out.index, axis=0)

        # Features     
        b  = d[STATS.S].astype(float)   # bads
        n  = d[STATS.C].astype(float)   # total obs
        g  = n - b                      # goods
        bs = b.sum()
        gs = g.sum()

        b,g = cls._impute(b, g, bs, gs)

        # Relative dst by FLAG
        d[FLAG.BADS]  = ( b / bs )
        d[FLAG.GOODS] = ( g / gs )  
        
        # Ecoded as WOE Representation
        d[_WOE] = round(np.log( d[FLAG.BADS] / d[FLAG.GOODS] ), 4)             

        if is_out:
            d_out = d_out.assign(**{_WOE:0})
            d = pd.concat([d, d_out], axis=0).sort_index()  

        return d
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def transform(cls, 
        y_true: pSe|NDArray, 
        x:      pSe|NDArray, 
        return_encoding: bool = False
    ) -> pSe | tuple[pSe, pDF]:
        """
        Feature x Series Transformation to Weight of Evidence.
        
        Parameters
        ----------
        y_true : pSe
            Target variable series on which basis WOE encoding is computed.

        x : pSe
            Feature series to be Encoded to WOE Form.
        
        return_encoding : bool = False
            Extend return by WOE encoding.

        Returns
        -------
        Series: Encoded input datapoints of x mapped to WOE.
        - If `return_encoding = True`
            - tuple: (Series, DataFrame) 
            - where df is calculated WOE by groups (bins/unique vals.) of x.
        """
        y_,x_ = cls._align_input(y_true, x)
        data_ = cls._group_data(y_, x_)
        woe   = cls._encode(data_)
        d_e   = Bins.data_mapping(x_, woe, x_.name, _WOE)
        
        if return_encoding: 
            return d_e, woe

        return d_e
    
    @classmethod
    def encode(cls, y_true: pSe|NDArray, x: pSe|NDArray) -> pDF:
        """
        Returs WOE encoding by groups (bins/unique vals.) of feature *x*.
        
        Parameters
        ----------
        y_true : pSe
            Target variable series on which basis WOE encoding is computed.

        x : pSe
            Feature series to be Encoded to WOE Form.
        
        Returns
        --------
        Dataframe : WOE encoding by groups (bins/unique vals.) of x.
        """
        y_, x_ = cls._align_input(y_true, x)
        d = cls._group_data(y_, x_)
        return cls._encode(d)          
        
    @classmethod
    def extend(cls, data: pDF, x: str = None, adjust_missing: bool = False) -> pDF:
        """
        Extends grouped data by WOE.
        
        Parameters
        ----------
        data : pDF
            Data in a form of grouped data by explanatory feature *x*.
            Data must contain `mean`, `count`, and `sum` of Target variable by bins of *x*.
        
        x : str, optional = None
            Required when adjust_missing = True.

        adjust_missing : bool, optional = False
            Apply missing value treatment.  
            - if   m_n < 50:  M_WOE = 0                 → For low number of obs → average target rate
            - elif m_b == 0:  M_WOE = data[_woe].min()  → If  TARGET = 0 → lowest  WOE found in the data             
            - elif m_g == 0:  M_WOE = data[_woe].max()  → If ~TARGET = 0 → highest WOE found in the data  

        Returns
        -------
        Dataframe : Extended data by WOE encoding.
        """
        cls._verify_grouped_data(data)
        woe = cls._encode(data)          
        if adjust_missing: 
            return cls._adj_missing(woe, x) 
        return woe
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class IV:
    """
    Information Value as Total distance between FLAG vs. NONFLAG distributions 
    (Symmetric Divergence).

    Methodology
    -----------
    IV = (FLAG - NONFLAG) * WOE

    +-------------------+---------------------------+----------------------------+
    | IV Value          | Predictive Power           | Action                    |
    +-------------------+---------------------------+----------------------------+
    | < 0.02            | Useless / Not predictive   | Drop the feature          |
    | 0.02 to 0.1       | Weak predictor             | Use with caution          |
    | 0.1 to 0.3        | Medium predictor           | Good feature              |
    | 0.3 to 0.5        | Strong predictor           | Excellent feature         |
    | > 0.5             | Too strong / Suspicious    | Check for data leakage    |
    +-------------------+---------------------------+----------------------------+    
    """

    @staticmethod
    def _get_flags(woe: pDF) -> tuple[pSe, pSe]:
        if all([x in woe.columns for x in FLAG.list()]):
            return woe[FLAG.BADS], woe[FLAG.GOODS]
        b = woe[STATS.S]
        g = woe[STATS.C] - b
        return b / b.sum(), g / g.sum()
    
    @staticmethod
    def _get_weight(woe: pDF) -> pSe:
        c = woe[STATS.C]
        return c / c.sum()

    @staticmethod
    def _get_proba(woe: pDF) -> tuple[float, pSe]:
        p = woe[STATS.S].sum() / woe[STATS.C].sum()
        return p, woe[STATS.M]
        
    @staticmethod
    def _get_lifts(p: float, pi: pSe) -> tuple[pSe, pSe]:
        return pi / p, (1-pi) / (1-p) 
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def _std(cls, p: float, pi:pSe, woe:pSe, weight:pSe) -> pSe:
        """
        IV populational standard deviation estimate by The Delta Method.
        
        Parameters
        ----------
        p : float
            Sample target mean E(T).

        pi : pSe
            Target mean by Bins E(T|bin).

        woe : pSe
            Weight of Evidence.

        weight : pSe
            Weight of bin.
        
        Returns
        -------
        Series : Standard deviation of IV per each bin.                
        """
        lb, lg = cls._get_lifts(p, pi)

        # The Derivative f'(p)
        var: float = p * (1 - p)
        vari:  pSe = (pi * (1 - pi)).replace(0, np.nan)

        term1 = -(1 / var) * woe
        term2 = (lb - lg) * (-1 / vari)
        f_prime = weight * (term1 + term2)
        
        return f_prime.abs() * np.sqrt(vari)  
            
    @classmethod
    def std(cls, woe: pDF) -> pSe:
        """
        IV populational standard deviation estimate by The Delta Method.

        Var(f(p)) ≈ [f′(p)]2 × Var(p)

        Parameters
        ----------
        woe : pDF
            WOE encoding data produced by WOE.encode().
        
        Returns
        -------
        Series : Standard deviation of IV per each bin.
        """
        WoE._verify_grouped_data(woe, is_woe=True)

        w     = cls._get_weight(woe)
        p, pi = cls._get_proba(woe)

        return cls._std(p, pi, woe[_WOE], w)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    @classmethod
    def extend(cls, woe: pDF, include_std: bool = False) -> pDF:
        """
        Extends WOE encoding by IV.
        
        Parameters
        ----------
        woe : pDF
            WOE encoding data produced by WOE.encode().
        
        include_std : bool
            Extends by IV populational standard deviation estimate by The Delta Method.
        
        Returns
        -------
        Dataframe : Extended data by IV, and optionally by s(IV).
        """        
        WoE._verify_grouped_data(woe, is_woe=True)
        fb, fg = cls._get_flags(woe)
        
        woe[IVL.IV] = (fb - fg) * woe[_WOE].to_numpy() 

        if include_std:
            woe[IVL.STD] = cls.std(woe)

        return woe
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class WoEOverview:
    """
    Overview of WoE features predictive behavior on train-test samples.

    Parameters
    ----------
    target : str 
        Target variable, present in samples provided to create().

    features : list
        List of features to evaluate, present in samples provided to create().

    alpha : float = 0.05
        Statistical level of significance - alpha value - to measure drift between samples.
    """

    def __init__(self, target:str, features: list, alpha: float = 0.05):
        self.target = target
        self.fes    = features
        self.alpha  = alpha

        self._cols  = {
            'mean':  {},
            'count': {},
            'IV':    {},
            'sIV':   {}
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _populate_meta(self, samples: dict[str, pDF]) -> None:
        for s in samples:
            s_ = s.split(' ')[0]
            self._cols['mean'][s]  = OV.T(s_)
            self._cols['count'][s] = OV.C(s_)
            self._cols['IV'][s]    = OV.IV(s_)
            self._cols['sIV'][s]   = OV.SIV(s_)

    def _get_cols(self, *c:str, sample:str=None, as_list:bool=False) -> list[dict|str] | str:        
        c_ = [self._cols[x] for x in c]
        if as_list:
            return [list(x.values()) for x in c_]
        if not sample:
            return c_
        return [x[sample] for x in c_]
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _group_data(self, d:pDF, fe:str) -> pDF:
        x = WoE._group_data(d[self.target], d[fe])
        x = WoE.extend(x).set_index(fe)
        return IV.extend(x, include_std = True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _is_significant(self, x:pDF) -> None:
        iv, si, c = self._get_cols('IV', 'sIV','count')
        
        t = {}
        for i, s in enumerate(iv, start=1):
            t[f'm{i}'] = x[iv[s]]
            t[f's{i}'] = x[si[s]]
            t[f'n{i}'] = x[c[s]]

        x[OV.IS] = TestSampleDiff(self.alpha).welch_mx(**t)
    
    def _rename_cols(self, d:pDF, sample:str) -> pDF:
        m, c, i, s = self._get_cols('mean', 'count', 'IV', 'sIV', sample=sample)
        return d.rename(columns={
            STATS.M: m,
            STATS.C: c,
            IVL.IV:  i,
            IVL.STD: s,
        })[[m, i, s, c]]

    def _create(self, samples: dict[str, pDF]) -> pDF:
        x = None
        for s, d in samples.items():
            if x is None:
                x = self._rename_cols(d, s)
            else:
                x = x.join(self._rename_cols(d, s))
        
        m, iv, si = self._get_cols('mean', 'IV', 'sIV', as_list=True)

        x[OV.D] = x[iv[1]] - x[iv[0]]

        self._is_significant(x)
        Pd.move_column_inplace(
            df  = x, 
            col = m + iv + si + [OV.D], 
            pos = list(range(7))
        )
        return x
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_total_iv(self, x: pDF) -> dict[str, float]:
        iv = self._get_cols('IV', as_list=True)[0]
        t  = x[iv].sum()
        d  = t[iv[1]] - t[iv[0]]
        return t.to_dict() | {OV.D: d}

    def _table(self, fes: dict[str, pDF]) -> pDF:
        t = []
        for fe, x in fes.items():
            t.append({
                OV.FE: fe, 
                **self._get_total_iv(x),
                OV.IS: x[OV.IS].any()
            })
        return pDF(t)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def create(self, samples: dict[str,pDF]) -> dict[str, dict[str, list[pDF] | pDF]]:
        """
        Overview of WoE features preditive behavior on train-test samples.

        Parameters
        ----------
        samples: dict[str, pDF]
            Dictionary of labels and dataframe samples with features encoded to WoE form.
            All Features present in the data will be evaluated on both samples.

        Returns
        -------
        dict[str, dict[str, list[pDF] | pDF]]
            - data → Predictive effect by feature.
            - summary → Summary table of main effects.
        """
        self._populate_meta(samples)

        df = {}
        for fe in self.fes:
            x = {}
            for s, d in samples.items():
                x[s] = self._group_data(d, fe)

            df[fe] = self._create(x)

        return {OV.DATA: df, OV.SUM: self._table(df)}
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
class WoEGetter:
    """
    Helper class to provide eithere:
    - saved trained WOE encoding 
    - create encoding for new data based on trained WOE bins
    
    Parameters
    ----------
    project : str
        Project name under which saved trained WOE encoding will be loaded.

    encoder : ModelAdapterBase
        Encoder (ModelAdapterBase) direct input of trained WOE encoder.

    path : str
        If PROJECT=None then PATH from which saved trained WOE encoding will be loaded.

    apply_data_quality : bool
        Apply trained data_quality phase from Predict class.

    apply_woe_encoding : bool
        Apply trained woe encoding phase from Predict class.

    Note
    ----
    project | encoder : either must be provided
        - project (str | None) under which WOE encoding model was trained and saved
        - encoder (ModelAdapterBase | None) direct input of trained WOE encoder    
    """    

    def __init__(self,
        project: str = None, 
        encoder: ModelAdapterBase = None,
        path:    str = None,

        apply_data_quality: bool = True,
        apply_woe_encoding: bool = True        
    ):
        self.project = project
        self.encoder = encoder
        self.path    = path

        self.apply_data_quality = apply_data_quality,
        self.apply_woe_encoding = apply_woe_encoding
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_trained_encoding(encoder: Predictor | ModelAdapterBase) -> dict[str, EncodingMap]:
        """Provides saved trained WOE encoding using Predict class."""
        p = Predictor()
        if isinstance(encoder, p.predictor):
            return encoder.woe_encode()  
        return encoder.cols.encoding

    @staticmethod
    def _get_stats_gini(stats:dict):
        """
        Stored trained / create encoding standardization.
        Train encoding → link to woe_transformer → stats design in mapping.
        """
        stats = dict(stats)             # ensures dict type from dataclass
        if stats.get(LBL.GINI) is not None:
            return stats[LBL.GINI]
        if stats.get('gini_woe') is not None:
            return stats['gini_woe']     # standard woe_transfomer → GiniPerf dataclass
        
        m = '"woe" gini performance has not been found in stats → fallback to the first value'
        print('AIG SAFETY ►', m)
        logger.warning(m)
        return list(stats.values())[0] 
    
    @staticmethod
    def _apply_trained_encoding(
        encoder:    Predictor | ModelAdapterBase,
        data:       pDF,
        subset:     list = None,
        apply_data_quality: bool = True,
    ) -> pDF:        
        """Applies saved trained WOE encoding to new data."""
        if isinstance(encoder, ModelAdapterBase):
            return encoder.transform(data, preprocess = apply_data_quality, subset = subset).data
        if apply_data_quality: 
            data = encoder.fe_clean(data).data
        return encoder.woe_encode(data, subset = subset).data
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def _standardize_to_orig_enc_fes(cls, 
        data:      pDF, 
        features:  list|str, 
        train_enc: dict[str, EncodingMap]
    ) -> list:
        """Standardize features to original variables names → Encoder mapping standard."""
        e = (
            '\n► Variables origin cannot be safely determined '
            '\n► Feature name can be provided either in standard of original name or WOE name'
        )
        if features is None: 
            features = list(data.columns)    
        fes = List.ensure(features)
        
        # Orig vars provided → we enforce either orig vars or woe vars; not a mix 
        fe_org = [x for x in train_enc if x in fes]

        # Encoded vars provided
        # → WOE has several suffixes → we can be statistically confident that features are WOE
        fe_woe = [x for x in fes if VarStd.is_woe_fe(x)]

        if fe_org and fe_woe:
            raise ValueError(
            'AIG SAFETY\n► Ambiguity in Features naming ► both original and WOE types present'+e                
            )

        if fe_org: 
            return fe_org
        
        if fe_woe:
            fe = [VarStd.fe_to_var(x) for x in fe_woe]
            return [x for x in fe if x in train_enc]

        raise ValueError(
            'AIG SAFETY\n► Neither original nor WOE feature name standard determined'+e
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _standardize_to_train_enc_order(enc:pDF, train_enc:EncodingMap, x_w:str) -> pDF:
        """Standardize created encoding to train order → Encoder mapping standard."""
        o = train_enc.map[x_w]
        return Pd.sort_values_by_array(enc, x_w, o)   
        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _get_unique_encoded_X(features:list, train_enc:dict[str, EncodingMap]):
        """
        Returns feature encoded pair:
        - fe_W:   WOE as interval for graph axis desc 
        - fe_WOE: WOE as num for gini - 5x faster compute than providing interval/str type.
        """        
        return [train_enc[x].vars[1:] for x in features]
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _verify_create_input(data:pDF, target:str, features:list, apply_woe_encoding:bool) -> None:
        def verify(c,m):
            if c: raise ValueError(f'{RgbANSI.bold_red("AIG SAFETY")} ► To create encoding, {m}.')

        verify( (data is None), 'data and target variable must be provided')                
        verify( (target not in data.columns), 'target variable must be specified and present in data')

        if apply_woe_encoding:
            c = [x for x in features if x not in data.columns]
            verify(c, 'when "apply_woe_encoding=True", original variables must be present in input data')
    # ─────────────────────────────────────────────────────────────────────────────────────────────
             
    def _create_trained_based_encoding(self, 
        encoder:  Predictor | ModelAdapterBase,
        data:     pDF, 
        target:   str, 
        features: list|str = None, 
    ) -> dict:
        """Creates WOE encoding for new data based on saved trained WOE encoding."""

        from ..metrics import Gini 

        enc = self._get_trained_encoding(encoder)
        fes = self._standardize_to_orig_enc_fes(data, features, enc)
        self._verify_create_input(data, target, fes, self.apply_woe_encoding)

        data_ = data
        if self.apply_woe_encoding:
            data_ = self._apply_trained_encoding(
                        encoder = encoder,
                        data    = data,  
                        subset  = fes,
                        apply_data_quality = self.apply_data_quality 
            )       
            if target not in data_.columns: 
                data_[target] = data[target]

        fes_w = self._get_unique_encoded_X(fes, enc)
        gini  = Gini()

        encode = {}
        for x_w, x_woe in fes_w:
            x_org = VarStd.fe_to_var(x_w)  
            e = WoE.encode(data_[target], data_[x_w])
            e = self._standardize_to_train_enc_order(e, enc[x_org], x_w)
            g = gini.univar(data_, target, x_woe) 
            encode[x_org] = EncodingMap(map = e, stats = {LBL.GINI: g}) # train encoding format
        
        return encode
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _verify_get_create_input(get_trained:bool, data:pDF|None, target:str|None) -> None:        
        if get_trained:
            return
        verify = []
        for c in [data, target]:
            verify.append(c is None)
        m = 'please either\n\
        ► set get_trained=True to obtain saved trained WOE encoding or\n\
        ► specify:\n\
            → data on which basis encoding will be created based on trained enc\n\
            → target in data\n\
            → features, optionally fe subset in data for encoding creation'
        if any(verify): 
            raise ValueError(f'{RgbANSI.bold_red("AIG SAFETY")} ► To get or create encoding, {m}.')
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def process(self, 
        data:         pDF = None, 
        target:       str = None, 
        features:    list = None, 
        get_trained: bool = False,
    ) -> dict:
        """
        Eithere returns saved trained WOE encoding 
        << or >>
        creates encoding for new data based on trained WOE bins.
        
        Parameters
        ----------
        data : pDF
            `If get_trained = False` → data on which basis encoding will be created based on trained.

        target : str
            `If get_trained = False` → Target in data.

        features : list
            Optionally subset of variables in data for encoding creation.

        get_trained : bool
            True returns saved trained WOE encoding in simplified standard {'data', 'gini'}.

        Returns
        ----------
        dict : Encoding in standardized format {x1: {'data':pDF, 'gini':float}, x2:... }
        """

        self._verify_get_create_input(get_trained, data, target)

        enc = self.encoder
        if enc is None:
            pre = Predictor()
            enc = pre(self.project, self.path)
        
        if get_trained:
            encoding = self._get_trained_encoding(enc)
        else:
            encoding = self._create_trained_based_encoding(enc, data, target, features)

        if features is None:
            features = encoding.keys()

        encode = {}
        for x in features:
            x = VarStd.fe_to_var(x)
            encode[x] = {LBL.DATA:None, LBL.GINI:0}
            w:EncodingMap = encoding.get(x)

            if w is None: 
                continue   
            e = w.map
            g = self._get_stats_gini(w.stats)
            
            encode[x] = {LBL.DATA: e, LBL.GINI: g}

        return encode
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class WoEPlot:
    """
    WOE Features Plotting Functinality.

    Parameters
    ----------
    project : str
        Project name under which saved trained WOE encoding will be loaded.

    encoder : ModelAdapterBase
        Encoder (ModelAdapterBase) direct input of trained WOE encoder.

    apply_data_quality : bool
        Apply trained data_quality phase from Predict class.

    apply_woe_encoding : bool
        Apply trained woe encoding phase from Predict class.
    
    Note
    ----
    project | encoder : either must be provided
        - project (str | None) under which WOE encoding model was trained and saved
        - encoder (ModelAdapterBase | None) direct input of trained WOE encoder            
    """

    def __init__(self,
        project: str = None, 
        encoder: ModelAdapterBase = None,
        apply_data_quality: bool  = True,
        apply_woe_encoding: bool  = True
        
    ):
        self.project = project
        self.encoder = encoder
        self.apply_data_quality = apply_data_quality
        self.apply_woe_encoding = apply_woe_encoding
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _graph_data(data:pDF):
        x = data.columns[0]                          # WOE encoding design
        i = [x,     STATS.C,  STATS.M,     _WOE]     # WOE encoding design
        o = ['BIN', 'DST',   'E(TARGET)', 'WOE']     # custom graph design
        
        d = data[i].copy()
        m = ( ( d[x] == VarStd.MISSING ) & ( d[_WOE] == 0 ) & ( len(d) != 1 ) )
        d = d[~m] # for graph purpose we are not interested in 0 woe added to train data for missing
        d.columns = o
        
        _x, _c, _t, _w = o
        
        d[_x] = d[_x].astype(str).apply(lambda x: Str.max_len(x, 28, 20))
        d[_c] = (d[_c] / d[_c].sum()).round(3)
        d[_t] = d[_t].round(3)
        
        x_ = d[_x].to_list()
        y1 = d[_w].to_list()
        y2 = d[_c].to_list()        
        
        return d, x, o, x_, y1, y2
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _graph_update(fig, row, col, show, show_table, x):
        x_, y1, y2 = Plot.get_sub_axes(fig, row, col).values()

        size = 14
        if not show: size = 11
        font = dict(family="Arial", size=size, color="black")
        
        adjust = dict(tickangle=20)
        if show_table: adjust = dict(tickangle=10)
        
        grid_axes = dict(gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.5)')

        fig.layout[x_].update(tickfont=font,  **grid_axes, **adjust)
        fig.layout[y1].update(tickfont=font, showgrid=False)
        fig.layout[y2].update(tickfont=font, **grid_axes)

        # Show help axes desc only for the first graph in a row (otherwise too noisy in the grid)
        if ( col == 1 ):
            standoff = {}
            if not show: standoff = dict(standoff=5)
            fig.layout[y1].update(title=dict(text="Bin Distribution (%)", font=font, **standoff))
            fig.layout[y2].update(title=dict(text="Bin WOE", font=font, **standoff))    
        
        fig.update_layout(
            height = 500, width = 1050,
            title  = {'text':x, 'x':0.5, 'xanchor': 'center','yanchor': 'top', 'font':font},
            showlegend = False
        )          
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def graph(
        cls,
        data: pDF, 
        fig:  go.Figure = None, 
        row:  int = 1, 
        col:  int = 1, 
        show: bool = False, 
        show_table: bool = False 
    ) -> go.Figure|None:
        """For WOE.encoding() standard WOE graph."""

        # Dynamic fig → allows to create dash-like figure in loop 
        # ────────────────────────────────────────────────────────
        if ( fig is None ):
            r,c = 1, 1
            mat = int(r*c) 
            m = [{'secondary_y': True} for i in range(mat)]
            s = [m]
            if show_table:
                r = 2
                t = [{'type':'table'} for i in range(mat)]
                s = mat*(m, t)

            fig = make_subplots(
                    rows  = r, 
                    cols  = c,
                    specs = s,
                ) 
                    
        if ( data is None ) or ( not len(data) ):
            NA = fig.add_trace(go.Scatter(
                    x=[0.5], y=[0.5],  
                    mode='text',text=['No data'],
                ), row=row, col=col)
            return NA 

        # Data prep.
        # ────────────────────────────────────────────────────────
        df, x, o, x_, y1, y2 = cls._graph_data(data)    
        is_missing_present = (x_[-1]==VarStd.MISSING) 
        # NOTE missing at -1 → WOE encoding design

        # 1 WOE
        # ────────────────────────────────────────────────────────
        def split_missing(x):             
            if is_missing_present: 
                  return x[:-1] + [None] + x[-1:]
            else: return x

        fig.add_trace(
            go.Scatter(
                x = split_missing(x_),
                y = split_missing(y1),
                mode = 'lines+markers',
                name = 'WOE',  
                hovertemplate = '%{x}<br>WOE: %{y:.3}',      
                line = dict(color='rgb(200, 50, 50)', width=2)
        ), row=row, col=col, secondary_y=True)

        # 2 Distribution
        # ────────────────────────────────────────────────────────
        fig.add_trace(
            go.Bar(
                x = x_, y = y2,
                name = 'DST',
                hovertemplate = '%{x}<br>Share: %{y:.0%}',
                marker = dict(color='rgba(100, 150, 220, 0.7)'),
        ), row=row, col=col)

        # 3 WOE Table
        # ───────────────────────────────────────────────────────
        if show_table:
            fig.add_trace(
                go.Table(
                    columnwidth = [100,50,50,50],            
                    header = dict(values = o, font = dict(size=11), align="left"),
                    cells  = dict(values = [df[c] for c in o], font_size=12, height=30, align = "left")
            ), row=row+1, col=col)        
            
        # Axes Updates
        # ───────────────────────────────────────────────────────────                
        cls._graph_update(fig, row, col, show, show_table, x)

        if not show:
            return fig

        fig.show()  
    # ─────────────────────────────────────────────────────────────────────────────────────────────

 
    @staticmethod
    def _dash_order(data:dict, features:list, ascending=True):
        """
        Defaults to sort by Gini when X_sorted None.
        Otherwise use provided order.
        """
        if features is None:
            a = 1 if ascending else -1
            I = np.argsort([a*d[LBL.GINI] for d in data.values()])
            return [list(data)[i] for i in I]
        return list(data)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _dash(self, 
        grid_title:      str = 'WOE', 
        grid_cols:       int = 4, 
        features:       list = None, 
        ascending:      bool = False,
        update_layout:  dict = None,
        **kwargs
    ) -> dict:
        """Contructs WOE encoding data and figures created for each variable."""
        
        # DATA
        # ────────────────────────────────────────────────────────────────────
        data = WoEGetter(
            self.project, self.encoder, 
            apply_data_quality = self.apply_data_quality,
            apply_woe_encoding = self.apply_woe_encoding
        ).process(features = features, **kwargs)

        fes  = self._dash_order(data, features, ascending)

        # DASH
        # ────────────────────────────────────────────────────────────────────
        def graph(x):
            return self.graph( data[x][LBL.DATA] )
        
        def update(x, g):
            d = data[x][LBL.DATA]
            data[x][LBL.GRAPH]   = g
            data[x][LBL.VERDICT] = verdict(x)     # extras → woe_stability
            data[x][LBL.GMX]     = min_max(d)     # extras → woe_stability 

        def verdict(x):
            g = data[x][LBL.GINI]
            if g == 0 : return 'NA'
            if g > 0.4: return 'STRONG'
            if g > 0.2: return 'MEDIUM'
            else:       return 'WEAK'

        def min_max(d):
            if d is None: return
            return d[_WOE].agg(['min','max']).to_list()

        def title(x):
            return f'{x} → {verdict(x)} | {round(data[x][LBL.GINI],2)}'
        
        fig = Plot.create_subplots(
                fes,
                graph_fx       = graph,
                title_fx       = title,
                update_fx      = update,
                secondary_y    = True,
                grid_title     = grid_title,
                grid_cols      = grid_cols,
                update_layout  = update_layout,
                desc_axes_by_first_only = True
        )
        
        return {
            DASH.DATA:  data, 
            DASH.FIG:   fig, 
            DASH.ORDER: fes
        }
    
    def dash(self, 
        grid_title:      str = 'WOE', 
        grid_cols:       int = 4, 
        features:       list = None, 
        ascending:      bool = False,
        update_layout:  dict = None,
        verbose:        bool = True,
        **kwargs
    ) -> dict:
        """
        Contructs WOE encoding data and figures created for each variable.

        Parameters
        ----------
        grid_title    : Dashboard figure main title.
        grid_cols     : Number of figured (columns) in a row of the fig matrix.
        features      : Prior order / subset of features in dashboard.
        ascending     : Sorting order of features if no order provided.
        update_layout : Optional grid layout update.
        verbose       : True shows waiting spinner.

        Returns
        -------
        (dict of) : ▼ data (dict), fig (plotly.Figure), order (list) ▼
        data : dictionary by X with consistency stats:
                - data['graph']      = plotly figure             # plt.Figure
                - data['gmx']        = figure's min-max          # tuple[float,float]
                - data['verdict']    = NA|STRONG|MEDIUM|WEAK     # str
                - data['gini']       = univariate gini of x      # float
                - data['data']       = WOE encoding              # pDF | None
        fig : figure matrix of generated plotly graphs
        order : list ordering of figure - useful for sync order with test data
        """
        with Spinner('Generating WOE Dashboard').progress(verbose):
            return self._dash(
                grid_title  = grid_title,
                grid_cols   = grid_cols,
                features    = features,
                ascending   = ascending,
                update_layout = update_layout,
                **kwargs
            )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬