"""
A specialized suite of machine learning utilities focused on data discretization, 
binning operations, and optimal representation search. 

This module implements high-performance logic for merging numerical and categorical 
bins, managing missing value standards, and executing mapping transformations 
between different binning scales.
"""

from ...deps.common     import  ast
from ...deps.data       import  np, pd, pDF, pSe
from ...deps.types      import  Literal, Any

from ...core.factory    import  (
                                DictClass, 
                                Dataclass
                                )
from ...core.interop    import  (
                                Pd,
                                TypeStandard
                                )
from ...core.standard   import  (
                                VarType,
                                VarStd, 
                                TypeDef
                                )
from ...utils.builtins  import  Str
from ..tests            import  TestModelDiff


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# Encoding Map → Result of applied binnning
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class EncodingMap:
    """
    Standard encoding result bundle — variable, type, map, stats.

    Provides a unified container for binning output:
    the variable name(s), data type, the encoding mapping,
    and associated statistics.

    Attributes
    ----------
    vars : list | str | None
        Variable name(s) the encoding applies to.
    type : list | str | None
        Data type indicator(s) (e.g., `'N'` for numerical,
        `'C'` for categorical).
    map : dict | Any
        The bin mapping (original values → bin labels).
    stats : dict | Any
        Per-bin statistics (counts, target rates, WOE, etc.).
    """
    vars:  list|str = None
    type:  list|str = None
    map:   dict|Any = None
    stats: dict|Any = None
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# Bin Ops
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class BinsUtils:
    """
    Low-level bin manipulation utilities.

    Internal helpers for:
    - Merging adjacent numerical interval bins.
    - Merging categorical group bins.
    - Extracting and transforming data against bin definitions.
    """

    @staticmethod
    def _merge_cat_std(v:str|list|None, div:str) -> list:
        """
        Normalize a categorical bin value to a list of components.

        Parameters
        ----------
        v : str | list | None
            The bin value (string with delimiter, list, or None).
        div : str
            The delimiter used to split string values.

        Returns
        -------
        list
            List of component values (empty if `v` is None).
        """        
        if ( v is None ):
            return []
        elif isinstance(v, str): 
            return v.split(div)
        return v

    @classmethod
    def merge_cat(cls, v1:str|list, v2:str=None, div:str = VarStd.CAT_DIV) -> str:
        """
        Merge two categorical bin groups into a single joined string.

        Parameters
        ----------
        v1 : str | list
            First categorical group (delimiter-joined string or list).
        v2 : str | list | None
            Second categorical group (optional).
        div : str
            Join delimiter.

        Returns
        -------
        str
            The merged categorical bin (all components joined by
            `div`).
        """        
        v1 = cls._merge_cat_std(v1, div)
        v2 = cls._merge_cat_std(v2, div)
        return div.join(v1 + v2)
        
    @staticmethod
    def _merge_num_verify(v1, v2, B=None):
        v = lambda: f'v1:{v1}, v2:{v2}\nAIG SAFETY ► '
        is_not_B_check = B is None
        if is_not_B_check:
            if v2 is not None: 
                raise ValueError(
                    v()+'Bin Merge params either as list of bins to merge v1=[...], '
                    'v2=None or v1=str, v2=str'
                )
            return
        if len(B) != 4: 
            raise ValueError(v()+'DATA TRANSFORM ► INCORRECT INTERVAL VALUES EXTRACTION')

    @classmethod
    def merge_num(cls, v1:str|list[str], v2:str=None) -> str:
        """
        Merge adjacent numerical interval bins into a single
        interval string.

        Handles both modes:
        - **Two-bin mode**: `v1='(0, 10]', v2='(10, 20]'` →
          `'(0, 20]'`.
        - **List mode**: `v1=['(0, 10]', '(10, 20]', '(20, 30]']`
          → `'(0, 30]'`.

        Parameters
        ----------
        v1 : str | list[str]
            First bin(s) (interval string or list of intervals).
        v2 : str | None
            Second bin (interval string). None in list mode.

        Returns
        -------
        str
            The merged interval string.
        """        
        if not isinstance(v1, str):
            cls._merge_num_verify(v1, v2)
            v1, v2 = v1[0], v1[-1]              # space of intervals to join            

        l,r = v1[0], v2[-1]                     # left, right interval bounds
        B = Str.get_interval_numbers(v1 + v2)   # 4 numbers must be returned
        cls._merge_num_verify(v1, v2, B)
        
        return l + B[0] + ", " + B[-1] + r
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_data_bins(data:pDF|pSe, bins:pDF|pSe, x:str, xT:str=None
    ) -> tuple[pSe, pSe]:
        """
        Extract the data series and bin series from
        container objects (DataFrame or Series).

        Parameters
        ----------
        data : pDF | pSe
            Data source (extracts column `x` if DataFrame).
        bins : pDF | pSe
            Bin definition source (extracts column `xT` if
            DataFrame).
        x : str
            Data variable name.
        xT : str | None
            Bin variable name (defaults to `x` if None).

        Returns
        -------
        tuple[pSe, pSe]
            (data_series, bin_series)
        """        
        d, b = data, bins
        if xT is None: xT = x
        if isinstance(data, pDF): d = data[x]
        if isinstance(bins, pDF): b = bins[xT]
        return d, b
    
    @staticmethod
    def transform(
        cond: list, 
        bins: np.ndarray, 
        index: pSe, 
        missing: Any = None
    ) -> pd.Categorical:
        """
        Apply a set of boolean conditions to assign bin labels
        to data points.

        Parameters
        ----------
        cond : list
            List of boolean masks (one per bin), in priority
            order.
        bins : np.ndarray
            Bin label values (assigned in condition order).
        index : pSe
            Index for the output series.
        missing : Any
            Default value for points not matching any condition.

        Returns
        -------
        pd.Categorical
            The binned categorical series.
        """        
        d = pSe(np.select(cond, bins, default=missing), index=index)
        d = TypeStandard.any_to_cat(d)
        return TypeStandard.cat_fill_missing(d)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
        
class Binning:
    """
    Binning operations — merge, transform, and map.

    Public API for bin manipulation:
    - **merge**: Combine adjacent bins (numerical or categorical).
    - **mapping**: Map original bins to merged bins.
    - **transform**: Apply bin definitions to raw data values.
    - **min_bin_share**: Automatically merge small bins.
    - **num_bin_search**: Find optimal bin count via DeLong test.
    """

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # ML STANDARD → BINS
    # ─────────────────────────────────────────────────────────────────────────────────────────────   
    @staticmethod
    def ensure_missing(data:pDF, x:str, Missing = VarStd.MISSING) -> pDF:
        """
        Ensure a missing-value bin is present in the data.

        If no row with the missing standard exists in variable
        `x`, a synthetic row is created (with zeros for all
        other columns) to represent the missing bin.

        Parameters
        ----------
        data : pDF
            Binning statistics table.
        x : str
            The bin variable name.
        Missing : str
            The missing value standard string.

        Returns
        -------
        pDF
            The data with a guaranteed missing-value row.
        """
        M = data[data[x] == VarStd.MISSING].copy()    
        if not len(M):
            c = [c for c in data.columns if c!=x]
            i = max(data.index[-1], 99)
            M = pDF({x:[Missing]}, index = [i])
            M[c] = [0]*len(c)
        else:
            i = max(M.index[0], 99)
            M.rename(index={M.index[0]:i}, inplace=True)
        return M 
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def to_interval_type(s) -> pd.Interval:
        """
        Parse a bin interval string (e.g., `'(0, 10]'`) into
        a `pd.Interval` object.

        Handles:
        - Bracket types: `[`/`]` (closed) vs. `(`/`)`
          (open).
        - `inf` values (converted to large finite numbers for
          safe arithmetic).

        Parameters
        ----------
        s : str | pd.Interval
            The interval string or an existing Interval.

        Returns
        -------
        pd.Interval
            The parsed interval with correct closure.
        """
        if not isinstance(s, str):
            return s
        
        table = str.maketrans({'[': '(', ']': ')'})
        left_closed  = s.startswith('[')
        right_closed = s.endswith(']')
        s = s.replace('inf','1e1000')   # > ~1.8e308 → inf

        left, right = ast.literal_eval(s.translate(table))

        t = 'neither'
        if left_closed and right_closed: t = 'both'
        elif left_closed:                t = 'left'
        elif right_closed:               t = 'right'        

        return pd.Interval(left, right, closed=t)     
    # ─────────────────────────────────────────────────────────────────────────────────────────────
            
    @staticmethod
    def merge(
        v1: str|list, 
        v2: str=None, 
        vtype: VarType = VarType.N, 
        div: str = VarStd.CAT_DIV
    ) -> str:
        """
        Merge two bins (or a list of bins) into one, respecting
        the variable type.

        Parameters
        ----------
        v1 : str | list
            First bin value(s). For numerical: an interval string
            or list of intervals. For categorical: a joined string
            or list of categories.
        v2 : str | None
            Second bin value. Required for two-bin numerical
            merge; ignored for list-mode and categorical merge.
        vtype : 'N' | 'C', default='N'
            Variable type: Numerical (interval merge) or
            Categorical (set union).
        div : str
            Categorical join delimiter.

        Returns
        -------
        str
            The merged bin representation.
        """
        if vtype == VarType.C:            
            return BinsUtils.merge_cat(v1, v2, div)        
        return BinsUtils.merge_num(v1, v2)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def mapping(
        Ox:      pSe, 
        Dx:      pSe, 
        vtype:   VarType, 
        missing: str = VarStd.MISSING, 
        div:     str = VarStd.CAT_DIV
    ) -> list:
        """
        Map original bins to their corresponding merged bins.

        For each bin in `Ox`, finds the bin in `Dx` that
        contains it (numerical: interval containment;
        categorical: set membership) and returns the merged
        label.

        Parameters
        ----------
        Ox : pSe
            Original bin labels (full resolution).
        Dx : pSe
            Merged bin labels (reduced resolution).
        vtype : 'N' | 'C' -  VarType standard
            Variable type (determines matching logic).
        missing : str, default=VarStd.MISSING
            Missing value standard (bypasses matching).
        div : str
            Categorical delimiter (for set membership check).

        Returns
        -------
        list
            Merged bin label for each original bin (same length
            as `Ox`, with missing appended if not present).
        """

        def match(o, d):
            if vtype == VarType.N:
                if o in d: 
                    return True                
            else:
                o = o.split(div)
                d = d.split(div)            
                if all(m in d for m in o): 
                   return True
            return False
        
        MAP = []
        for o in Ox:
            if o == missing:
                continue
            for d in Dx:
                if match(o, d):
                    o = d
                    break
            MAP.append(o)
        
        if missing not in MAP:
            return MAP + [missing]
        
        return MAP
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def data_mapping(data:pDF|pSe, mapping:pDF, x:str, xT:str) -> pSe:
        """
        Map data values to target bin labels using a mapping table.

        Parameters
        ----------
        data : pDF | pSe
            Input data (Series of `x` values, or DataFrame
            containing column `x`).
        mapping : pDF
            Mapping table with columns `x` (source values)
            and `xT` (target bin labels).
        x : str
            Source variable name (in both data and mapping).
        xT : str
            Target variable name (the desired bin label column
            in mapping).

        Returns
        -------
        pSe
            Series of `categorical dtype` and the same length as input, 
            where each value of `x` is replaced by its mapped `xT` bin label.
        """
        d = data
        if isinstance(data, pDF): 
            d = data[x]
        
        X = [x, xT]
        m = mapping[X].to_dict()
        m = dict(zip(*[m[v].values() for v in X]))  # M = {value1: new_value1, value2: new_value2, ...}
        
        t = TypeDef.category[0]        
        return d.map(m).astype(t)
    
    @staticmethod
    def transform_numerical_data(
        data:    pDF|pSe, 
        mapping: pDF|pSe, 
        x:       str, 
        xT:      str = None
    ) -> pd.Categorical:
        """
        Apply numerical bin definitions to raw data values.

        Assigns each data point to the bin whose interval
        contains it, respecting the interval's closure
        (left-closed, right-closed, both, or neither).

        Parameters
        ----------
        data : pDF | pSe
            Input data (Series of numeric values, or DataFrame
            containing column `x`).
        mapping : pDF | pSe
            Bin definitions (Series of interval objects, or
            DataFrame with column `x`/`xT`).
        x : str
            Variable name in data (and in mapping if DataFrame).
        xT : str | None, default=None
            Variable name in mapping for bin definitions
            (defaults to `x`).

        Returns
        -------
        pd.Categorical
            Binned categorical series (same length as input).
            Points outside all bins receive NaN.
        """
        d, b = BinsUtils.extract_data_bins(data, mapping, x, xT)
        bins = np.unique( b[b!=VarStd.MISSING] )
        cond = [
            ((d >  b.left)  if b.closed in ['right', 'both'] else (d >= b.left)) & 
            ((d <= b.right) if b.closed in ['right', 'both'] else (d <  b.right))
            for b in bins
        ]
        return BinsUtils.transform(cond, bins, d.index, np.nan)
    
    @staticmethod
    def transform_categorical_data(
        data:    pDF|pSe, 
        mapping: pDF|pSe, 
        x:       str, 
        xT:      str = None, 
        div:     str = VarStd.CAT_DIV
    ) -> pd.Categorical:
        """
        Apply categorical bin definitions to raw data values.

        Assigns each data point to the bin group that contains
        its category (set membership check via delimiter-
        split components).

        Parameters
        ----------
        data : pDF | pSe
            Input data (Series of category values, or DataFrame
            containing column `x`).
        mapping : pDF | pSe
            Bin group definitions (Series of joined category
            strings, or DataFrame with column `x`/`xT`).
        x : str
            Variable name in data (and in mapping if DataFrame).
        xT : str | None, default=None
            Variable name in mapping for bin definitions
            (defaults to `x`).
        div : str
            Categorical join delimiter.

        Returns
        -------
        pd.Categorical
            Binned categorical series (same length as input).
            Points not in any bin receive the missing standard.
        """         
        d, b = BinsUtils.extract_data_bins(data, mapping, x, xT)
        bins = np.unique(b)
        cond = [d.isin(b.split(div)) for b in bins]
        return BinsUtils.transform(cond, bins, d.index, VarStd.MISSING)
    # ─────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # MINIMAL SHARE OF BIN 
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def min_bin_share( 
        data:   pDF, 
        target: str, 
        x:      str, 
        th:     float = 0.03, 
        vtype:  Literal['N','C'] = 'N', 
        xG:     str = None
    ) -> "MinBinResult":
        """
        Treat bins with fewer than a threshold share of
        observations by merging them into a neighboring bin
        with a similar target rate.

        Ensures statistical reliability of each bin — a bin with
        1% of observations has an unstable target rate estimate.
        Merging to the nearest-neighbor by target mean preserves
        the monotonic relationship while eliminating noise.

        Parameters
        ----------
        data : pDF
            Input data (must contain target and variable `x`).
        target : str
            Target variable (used to find the nearest-neighbor
            bin by target rate).
        x : str
            Variable to bin (source of small bins).
        th : float, default=0.03
            Minimum bin share threshold. Bins with share < `th`
            are merged. Use > 1 for absolute count threshold.
        vtype : 'N' | 'C', default='N'
            Variable type (determines merge semantics).
        xG : str | None, default=None
            Name for the transformed variable. Defaults to
            AIG standard (`x + '_G'`).

        Returns
        -------
        MinBinResult
            Mapping table + processed data + original stats.
        """
        return MinBinShare(target, x, th, vtype, xG).process(data)
    # ─────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # NUM OPTIMAL BINNING
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def num_bin_search(        
        y_true:  pSe, 
        x:       pSe, 
        bins_to: int = 10,
        alpha: float = 0.05, 
        return_stats: bool = True
    ):
        """
        Search the bin space to find the optimal number of bins
        for a numerical variable.

        Evaluates candidate bin counts (2 to `bins_to`) by:
        1. Creating quantile bins for each candidate count.
        2. Computing WOE encoding and Gini for each.
        3. Applying DeLong significance testing between
           consecutive bin versions.
        4. Selecting the bin count that maximizes predictive
           power (min Bias) while minimizing complexity
           (min Variance / degrees of freedom).

        Parameters
        ----------
        y_true : pSe
            Target binary vector (0/1).
        x : pSe
            Exogenous numerical variable to bin.
        bins_to : int, default=10
            Maximum number of bins to evaluate (searches
            range 2 to `bins_to`).
        alpha : float, default=0.05
            Significance level for the DeLong test.
        return_stats : bool, default=True
            If True, includes the full search table in the
            result.

        Returns
        -------
        tuple[int, float] | tuple[int, float, pDF]
            (optimal_bins, best_gini) or
            (optimal_bins, best_gini, search_stats) depending
            on `return_stats`.
        """   
        return NumBinSearch(bins_to, alpha, return_stats).process(y_true, x)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@Dataclass.base
class MinBinResult:
    """
    Minimum bin share treatment output.

    Attributes
    ----------
    mapping : pDF
        Transformation map: original bins → merged bins
        (with the new `xG` variable name).
    dt : pDF
        Processed (merged) bin statistics table.
    do : pDF
        Original bin statistics table (before merging).
    xG : str
        The new variable name for the transformed bins.
    """
    mapping: pDF
    dt:      pDF
    do:      pDF
    xG:      str

class MinBinShare:
    """
    Minimum Bin Share (MBS) Treatment.

    Iteratively merges bins whose observation share falls below
    a threshold into their nearest neighbor (by target rate).
    This eliminates statistically unreliable bins while
    preserving the overall target monotonicity.

    Algorithm:
    1. Compute per-bin statistics (count, share, target mean).
    2. Identify bins with share < threshold.
    3. Merge each small bin into its nearest neighbor (by
       target mean difference).
    4. Repeat until no bins are below threshold.
    5. Generate the original→merged mapping table.

    Parameters
    ----------
    target : str
        Target variable name (used for neighbor selection).
    x : str
        Variable to treat (source of bins).
    th : float, default=0.03
        Minimum bin share threshold (0–1 for relative, > 1 for
        absolute count).
    vtype : 'N' | 'C', default='N'
        Variable type (determines merge semantics).
    xG : str | None, default=None
        Output variable name (defaults to `x + '_G'`).
    """
    _s = 'share'
    _u = '__u__'
    _c = ['count', 'sum', _s, _u]

    def __init__(self,
        target: str, 
        x:      str, 
        th:     float = 0.03, 
        vtype:  VarType = VarType.N, 
        xG:     str = None
    ):
        self.target = target
        self.x      = x
        self.vtype  = vtype
        self.th     = th if th> 0 else 0.03
        self.xG     = VarStd.fe_g(x) if xG is None else xG
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _data_to_stats(self, data:pDF, target:str, x:str) -> pDF:
        """
        Compute per-bin statistics: target mean, count, sum,
        and share (count / total).        
        """
        d = data.loc[:,[x, target]].reset_index(drop=True)
        d = (
            d.groupby(x, observed=True)
             .agg(['mean','count','sum'])
             .droplevel(0, axis=1).reset_index().replace(np.nan,0)
        )        
        d[self._s] = d['count'] / d['count'].sum()  # Extending by bins share             
        return d
    # ─────────────────────────────────────────────────────────────────────────────────────────────    

    def _data_std_cat(self, d:pDF) -> pDF:
        """
        Standardize categorical data for MBS processing —
        sort by descending target mean (ensures nearest-neighbor
        lookup works on a rate-ordered table).
        """
        if self.vtype == VarType.C:
            d = d.sort_values(by=['mean'], ascending=False)
        return d

    def _data_std_num(self, d:pDF, to_or_from_std:Literal[0,1]=0) -> pDF:
        """Standardize numerical bin labels for MBS processing."""
        if to_or_from_std == 0:
            d[self.x] = d[self.x].astype(str)           # Processing as string with reverse to orig. type after transformation
        elif self.vtype == VarType.N: 
            d[self.x] = d[self.x].apply(Binning.to_interval_type)   
        return d
    
    def _data_std_dynamic_org(self, data:pDF) -> tuple[pDF, pDF]:
        """
        Prepare data for MBS processing: create a dynamic table
        (mutated during merging) and an original table (for
        mapping back).
        """
        d = self._data_std_cat(data)
        d = d[d[self.x] != VarStd.MISSING].reset_index(drop=True)   # A Dynamic Tables - length is being chnaged with every merged bin 
        o = d.copy()                                                # B Original Table - used for join for data by original values        
        return self._data_std_num(d), o
    # ─────────────────────────────────────────────────────────────────────────────────────────────
            
    @staticmethod
    def _find_to(data:pDF, u:pSe) -> np.ndarray:
        """
        Find the best merge partner for the first under-threshold
        bin.

        Selects the neighbor (left or right) with the smallest
        target mean difference — preserving the target
        monotonicity.        
        """        
        m, ix = data['mean'], data.index
        i, fm = next(m[u].items())
        if i == ix[0]:  return i, ix[1]                 # special case of the first pos.                  
        if i == ix[-1]: return i, ix[-2]                # special case of the last pos.                
        return i, (m[[i-1, i+1]] - fm).abs().idxmin()   # find the best neighbor → min target diff.
    
    @staticmethod
    def _ft(data:pDF, g:pSe) -> dict:
        """
        Extract the "from-to" pair (the two bins to merge) and
        their data rows.        
        """
        f,t = min(g), max(g)                            # ensure to have the right order
        return {'ft':data.loc[f:t,:], 'f':f, 't':t}     # from - to data

    def _merge_bins(self, ft:dict) -> str:        
        v1,v2 = ft['ft'][self.x]
        return Binning.merge(v1, v2, self.vtype)  

    @staticmethod
    def _merge_mean(ft:dict) -> float:
        """
        Compute the weighted target mean of the merged bin
        (weighted by bin counts).   
        """
        ft = ft['ft']
        div = max(1, ft['count'].sum())                 # Zero Division Treatment
        return (ft['mean']*ft['count']).sum() / div     # Target mean of merged Bin
        
    def _update_merged(self, d:pDF, ft:dict, b:str, m:float) -> None:
        """
        Compute the weighted target mean of the merged bin
        (weighted by bin counts).        
        """
        ft,_,t = ft.values()
        cols = [self.x, 'mean'] + self._c
        d.loc[t, cols] = [b, m] + ft[self._c].sum().to_list()

    @staticmethod
    def _remove_merged(data:pDF, ft:dict) -> pDF:
        """
        Remove the "from" bin (already merged into "to") from
        the dynamic table.        
        """
        return data.drop(ft['f']).reset_index(drop=True)

    def _add_u_to_dynamic_data(self, data:pDF) -> None :
        """
        Populate the under-threshold indicator column (`_u`).

        For relative threshold (0 < th ≤ 1): uses bin share.
        For absolute threshold (th > 1): uses bin count.        
        """        
        if self.th > 1: 
            data[self._u] = data['count']               # TH → Absolute / Relative
        else:      
            data[self._u] = data[self._s]

    def _update_vector(self, d:pDF) -> pSe:
        """Compute the boolean mask of bins below the threshold."""
        return d[self._u] < self.th
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _mbs_treatment(self, d:pDF) -> pDF:
        """
        Execute the iterative MBS merge loop.

        Repeatedly finds the first under-threshold bin, merges
        it into its best neighbor, and re-evaluates — until no
        bins remain below threshold (or only one bin remains).        
        """
        u = self._update_vector(d)
        while ( sum(u) > 0 ) and ( len(u) > 1 ):

            it = self._find_to(d,u)        
            ft = self._ft(d, it)
            b  = self._merge_bins(ft)
            m  = self._merge_mean(ft)      
            self._update_merged(d,ft,b,m)          

            d = self._remove_merged(d,ft)
            u = self._update_vector(d)
        
        return d.loc[:,[self.x,'mean','count','sum',self._s]]             
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _mapping(self, d:pDF, o:pDF, miss:pDF) -> tuple[pDF,pDF,pDF]:
        """
        Generate the original→merged mapping table and final
        data tables.        
        """
        do = pd.concat([o,miss])
        dt = pd.concat([d,miss])    

        mapping = do.copy()
        mapping[self.xG] = Binning.mapping(o[self.x], d[self.x], self.vtype)
        mapping[self.xG] = TypeStandard.any_to_cat(mapping[self.xG])
        return mapping, dt, do
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def process(self, data:pDF) -> MinBinResult:
        """
        Execute the full MBS treatment pipeline.

        Steps:
        1. Compute per-bin statistics.
        2. Ensure missing-value bin presence.
        3. Standardize data (sort, type conversion).
        4. Iteratively merge under-threshold bins.
        5. Generate original→merged mapping.

        Parameters
        ----------
        data : pDF
            Input data (must contain target and variable `x`).

        Returns
        -------
        MinBinResult
            Mapping table + processed data + original stats.
        """          
        stats = self._data_to_stats(data, self.target, self.x)
        miss  = Binning.ensure_missing(stats, self.x) 
        d, o  = self._data_std_dynamic_org(stats)
        self._add_u_to_dynamic_data(d)
        d = self._mbs_treatment(d)
        d = self._data_std_num(d,1)
        m, dt, do = self._mapping(d,o,miss)        
        return MinBinResult(
            mapping = m,
            dt      = dt,
            do      = do,
            xG      = self.xG
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class NUM(DictClass):
    """Column labels for the numerical bin search results table."""
    bins   = 'Bins'
    gini   = 'Gini'
    woe    = 'Woe'
    p_val  = 'p-value' 
    p_base = 'p-base'
    bd     = 'Binsδ'
    gd     = 'Giniδ'
    sig    = 'Significantδ'
    choice = 'Choice'

class SIG(DictClass):
    """Significance indicator symbols (display)."""
    Y = "✅"
    N = "❌"

class NumBinSearch:
    """
    Optimal Bin Count Search via Bias-Variance Trade-off.

    Searches the bin space (2 to N bins) to identify the
    optimal discretization of a continuous variable.

    Methodology:
    1. **Encode**: Create quantile bins for each candidate count;
       compute WOE encoding and Gini.
    2. **Significance Test**: Apply DeLong test between
       consecutive bin versions (sorted by Gini). A significant
       p-value indicates the additional bins provide real
       predictive improvement.
    3. **Optimize**: Among significant improvements, select the
       highest-performing version. If no significant improvement
       exists, select the minimum-bin version (minimum variance).

    The result balances:
    - **Bias** (underfitting): Too few bins → lose signal.
    - **Variance** (overfitting): Too many bins → noisy WOE
      estimates, unstable scores.

    Parameters
    ----------
    bins_to : int, default=10
        Maximum bins to evaluate (searches 2 to `bins_to`).
    alpha : float, default=0.05
        DeLong test significance level.
    return_stats : bool, default=True
        If True, include the full search table in the output.
    """
    _B = [NUM.bins,  NUM.gini]
    _P = [NUM.p_val, NUM.p_base]        

    def __init__(self, bins_to:int=10, alpha:float=0.05, return_stats:bool=True):
        self.bins_to = bins_to
        self.alpha = alpha
        self.return_stats = return_stats
    # ─────────────────────────────────────────────────────────────────────────────────────────────    

    def _is_significant(self, p_val:float|pSe) -> bool|pSe:
        return p_val < self.alpha
    
    @staticmethod
    def _search_dynamics(s:dict) -> tuple[pSe, int]:
        return s[NUM.woe], s[NUM.bins]
    
    @classmethod
    def _search_res(cls, s:dict, *p_res) -> list:
        return dict(zip( cls._B + cls._P, (s[NUM.bins], s[NUM.gini], *p_res) ))
    
    def _search_run(self, search:list, y_true:pSe) -> list:
        """
        Execute the DeLong significance test chain across all
        candidate bin versions.

        Iterates through bins (sorted by Gini ascending), testing
        each version against the last significant baseline.
        Updates the baseline whenever a significant improvement
        is found.        
        """        
        res = []
        for i, s in enumerate(search):
            if ( i == 0 ):
               p_val, p_base = [-1,0]       # ► init vals to set dynamic xs0 as y_pred1 (woe1)
            else:
                xs1   = s[NUM.woe]          # y_pred2 (woe2) 
                p_val = TestModelDiff(y_true, xs0, xs1).delong()            
            res.append( self._search_res(s, p_val, p_base) )
            if self._is_significant(p_val):    # ► Dynamic update - the last sig. to the next
                xs0, p_base = self._search_dynamics(s)
        return res

    def _search_format(self, result:list) -> pDF:
        """
        Format the raw search results into a structured table
        with delta columns and significance flags.        
        """
        res = pDF(result)
        res[NUM.bd]  = res[NUM.bins] - res[NUM.p_base]
        res[NUM.gd]  = res[NUM.gini] - res[NUM.p_base].map(res.set_index(NUM.bins)[NUM.gini])
        res[NUM.sig] = self._is_significant(res[NUM.p_val])
        return res

    def _search_vector(self, search:list, y_true:pSe) -> pDF:
        """Execute the full search: sort, run DeLong chain, format."""
        search.sort(key=lambda x: x[NUM.gini]) # ► Gini sort → important for search algo to work properly
        res = self._search_run(search, y_true)
        return self._search_format(res)    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _p_min(b:pDF) -> float: return b[NUM.p_val].min()
    
    @staticmethod
    def _max_sig_ix(b:pDF) -> int: return b[b[NUM.sig]].index[-1]
    @classmethod
    def _max_perform(cls, b:pDF) -> pDF: return b.loc[cls._max_sig_ix(b): , :]
    
    @staticmethod
    def _min_bins(b:pDF) -> pSe: return b[NUM.bins] == b[NUM.bins].min()
    @classmethod
    def _min_degree(cls, b:pDF)  -> pSe: return b.loc[cls._min_bins(b), cls._B].iloc[0]

    def _optim_space(self, b:pDF) -> pDF:
        """
        Identify the optimization space: rows where the DeLong
        test is significant OR where adding bins *decreases*
        Gini (negative Binsδ — indicating the extra complexity
        is harmful).        
        """
        return b.loc[ self._is_significant(b[NUM.p_val]) | ( b[NUM.bd] < 0 ), :]   
        # points of interests → break points in performance or decrease in degrees

    def _optim_search(self, search:pDF) -> pDF:
        """ 
        Apply the optimization space restriction.

        - **Min Bias**: Keep only bins with significant Gini
          performance improvements.
        - **Min Variance**: Keep bins with fewer degrees of
          freedom but same performance (negative Binsδ).

        If the restricted space is empty, returns the full
        search table (no restriction).
        """
        b = self._optim_space(search)
        if not len(b):
            return search
        return b
    
    def _bin_search(self, search:pDF) -> tuple[int, float]:            
        """ 
        Select the optimal bin count from the search table.

        Decision logic:
        1. If any bin version is significantly better (DeLong
           p < alpha): select the highest-Gini version among
           significant ones (max performance = min Bias).
        2. Among that restricted set: select the minimum bin
           count (min degrees of freedom = min Variance).
        3. If no significant improvement: select the minimum
           bin count from the full table.
        """         
        b = self._optim_search(search)

        if self._is_significant( self._p_min(b) ):
            b = self._max_perform(b) # significant models diff goes to max perf.
        
        bins, gini = self._min_degree(b)
        
        return int(bins), gini
    
    @staticmethod
    def _update_search_choice(search:pDF, bin:int) -> None:
        """Mark the chosen bin in the search table (visual flag)."""
        search[NUM.choice] = np.where(search[NUM.bins].to_numpy() == bin, SIG.Y, SIG.N)    
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _encode_bin(y_true:pSe, x:pSe, b:int) -> dict:
        """Create a quantile bin encoding for a candidate bin count."""
        from ..metrics import Gini, WoE
        y = y_true
        q = Pd.qcut(x, b, duplicates='drop')
        e = WoE.transform(y, q)
        g = Gini.fast_calc(y, e)
        return {NUM.bins:b, NUM.gini:g, NUM.woe:e}

    @classmethod
    def _create_bin_space(cls, y_true:pSe, x:pSe, bins:int) -> list[dict]:
        """
        Generate the full search space: all candidate bin
        encodings from 2 to `bins+1`.        
        """        
        bin_st = 2
        bin_to = max(2, bin_st + bins - 1)
        space  = []
        for b in range(bin_st, bin_to):
            space.append( cls._encode_bin(y_true, x, b) )
        return space
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def process(self, y_true: pSe, x: pSe) -> tuple[int, float] | tuple[int, float, pDF]:
        """
        Execute the full optimal bin search.

        Steps:
        1. Create the bin space (2 to `bins_to`).
        2. Run DeLong significance chain (sorted by Gini).
        3. Select the optimal bin (min Bias + min Variance).
        4. Mark the choice in the search table.

        Parameters
        ----------
        y_true : pSe
            Target binary vector.
        x : pSe
            Exogenous numerical variable.

        Returns
        -------
        tuple[int, float] | tuple[int, float, pDF]
            (optimal_bins, best_gini) or
            (optimal_bins, best_gini, search_table) depending
            on `return_stats`.
        """
        space  = self._create_bin_space(y_true, x, self.bins_to)
        search = self._search_vector(space, y_true)        
        b,g = self._bin_search(search)
        self._update_search_choice(search, b)      
        return (b, g, search) if self.return_stats else (b, g)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬