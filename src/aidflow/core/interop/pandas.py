"""
Low-level infrastructure pandas data ops and structural manipulation tools.

Type-Aware Data Manipulation with specialized helpers designed specifically for 
ML pipeline requirements.
"""

from ...deps.common      import  (
                                math,
                                wraps, 
                                contextmanager
                                )
from ...deps.data       import  (
                                np, 
                                pd, pDF, pSe,
                                pd_Styler
                                )
from ...deps.types      import  ArrayLike
from ...utils.builtins  import  List
from ..standard         import  (
                                VarStd, 
                                TypeDef
                                )

class GetValue:
    """
    Fast O(1) DataFrame row lookup with loop caching.

    Provides constant-time access to values at a specific row
    condition (`df[at_x] == x`), replacing the O(N) scan of
    `df.loc[df[at_x] == x, of_col]`.

    Performance
    -----------
    - **Unique index + scalar col**: `.at` lookup — 10–100x
      faster than conditional `.loc`.
    - **Unique index + vector cols**: `.loc` on indexed
      DataFrame — 2–5x faster.
    - **Non-unique index**: Falls back to standard `.loc`
      (no performance penalty, but no speedup).

    Caching Modes
    -------------
    - **Direct use**: Create an instance and call it repeatedly
      in a loop. The indexed DataFrame is built once.
    - **Instance-attached** (`_value`): Cached on a worker
      instance for automatic reuse across method calls.
      Auto-refreshes if the DataFrame object changes.
    - **Decorator** (`at_x_of_col`): Attaches the lookup
      behavior directly to a worker method signature.

    Parameters
    ----------
    df : pDF
        Source DataFrame.
    at_x : str
        Column name used as the row condition key.
    no_match : any, default=None
        Value returned when the key is not found.
    verbose : bool, default=True
        Print a safety warning when non-unique keys are detected.
    to_raise : bool, default=True
        If True, raise `ValueError` when the key is non-unique
        (return type would change from scalar to Series/DataFrame).
        If False, return the multi-row result.

    Examples
    --------
    >>> # Direct use in a loop
    >>> getter = GetValue(df, at_x='feature_name')
    >>> for name in feature_names:
    ...     val = getter(name, 'coefficient')

    >>> # One-shot lookup (no caching needed)
    >>> val = GetValue.value(df, at_x='feature_name', x='revenue', of_col='coefficient')

    >>> # Worker method decorator
    >>> @Pd.GetValue.at_x_of_col(at_x='feature_name')
    ... def get_coefficient(self, df, x):
    ...     return x  # x is the feature name, returns the coefficient
    """
    attr_name = "__get_df_val_of_col_at_x_fast"
    registry  = "__getatloop_name_space__"

    def __init__(self, df:pDF, at_x:str, no_match=None, verbose:bool=True, to_raise:bool=True):
        self.id = id(df)
        self.at_x = at_x            

        self.no_match = no_match
        self.verbose  = verbose
        self.to_raise = to_raise

        self.is_unique = df[at_x].is_unique            
        if self.is_unique:
                self.dx = df.set_index(at_x)
        else: self.dx = df
    
    def __call__(self, x, of_col:str|list|None):
        """
        Retrieve value(s) where `at_x == x`.

        Parameters
        ----------
        x : hashable
            Key value to match against the `at_x` column.
        of_col : str | list | None
            Column(s) to retrieve. If None, all columns.

        Returns
        -------
        scalar | pd.Series | pDF
            - `str` col + unique key → scalar.
            - `list` cols + unique key → `pd.Series`.
            - Non-unique key → `pd.Series` or `pDF`
              (or raises if `to_raise=True`).
        """
        if of_col is None:
            of_col = self.dx.columns            
        if self.is_unique:
            return self._unique(x, of_col)            
        return self._not_unique(x, of_col)
    
    def _unique(self, x:str, of_col:str|list|None):
        """
        Unique values provides fast match via .at (1col) - %timeit on small/medium df 20-100x.
        For list of cols → .loc with ~2x-5x (given .set_index) while returning scalar/vector.
        """
        if isinstance(of_col, str): 
                fx = self.dx.at
        else: fx = self.dx.loc            
        try:    
            return fx[x, of_col]
        except: 
            return self.no_match
            
    def _not_unique(self, x, of_col:str|list|None):
        """Non-unique value provides standard .loc match while returning vector/matrix."""
        if isinstance(of_col, str): 
                m = 'pd.Series instead of scalar'
        else: m = 'pd.DataFrame instead of pd.Series'
        m = f'AIG SAFETY ► Pd.GetValue operates on non-unique values → Returning object is {m}.'
        if self.to_raise: raise ValueError(m)
        if self.verbose:  print(m)
        return self.dx.loc[self.dx[self.at_x]==x, of_col]                    

    
    @classmethod
    def value(cls, 
        df:pDF, at_x:str, x:str, of_col:str|list|None=None, to_raise:bool=True
    ):
        """
        Helper to get one value(s) only from df → mainly for checks. 
        `Otherwise use instance of GetValue to obtain fast look-up by caching.`
        
        Instance use case leveraging performance boost:
        >>> get_fast = GetValue(df, at_x)
        >>> for x in values_of_X:
        >>>     value_of_col_at_x = get_fast(x, of_col)

        Applied logic   
        -------------         
        a : Scalar → df.set_index(at_x).at[x, of_col]  → `of_col >> str  >> at_x has unique values only >> 10-100x boost`.
        b : Vector → df.set_index(at_x).loc[x, of_col] → `of_col >> list >> at_x has unique values only >> 2-5x boost`.
        c : Vector | Matrix → df.loc[df[at_x]==x, of_col] → `NON-unique values in at_x` with standard performance.

        Parameters
        ----------
        df :        DataFrame from which value(s) is retrieved.
        at_x :      Column which serves the row condition as at_x==x.
        x :         Value for a row condition as at_x == x.
        of_col :    Column(s) which value(s) is returned.
        to_raise :  True → raise when non-unique at_x==x is present 
                    thus different type would be returned 
                    → Series instead of Scalar Value | DataFrame instead of Series.

        Returns
        -------
        a : of_col is str → `Scalar Value` of of_col from row at_x==x.
        b : of_col is a list of cols → `Vector of Values` (pd.Series).
        c : `Vector (of_col is str) | Matrix (pd.DataFrame - of_col as list)` if 
            `non-unique at_x==x occurs <<>> and  to_raise=False`.

        Examples
        --------
        >>> # Single lookup
        >>> val = GetValue.value(df, at_x='feature', x='revenue', of_col='coeff')

        >>> # In a loop (use instance instead for speed)
        >>> getter = GetValue(df, at_x='feature')
        >>> for f in features:
        ...     getter(f, 'coeff')
        """
        return cls(df, at_x, to_raise=to_raise)(x, of_col)
    
    # ─── API for AIG Worker Instances ───
    def _is_df_same(self, df:pDF) -> bool:
        """Check whether cached DataFrame is the same object in memory."""
        return self.id == id(df)
    
    @classmethod
    def _get_name(cls, method_name:str=None) -> str:
        """Generate a unique cached-lookup attribute name."""
        return f"{cls.attr_name}_{method_name or 'default'}" 
    
    @classmethod
    def _register_method(cls, instance, name) -> None:
        """Register a cached lookup name in the instance's namespace tracker."""
        if not hasattr(instance, cls.registry):
            setattr(instance, cls.registry, set())
        getattr(instance, cls.registry).add(name)
    
    @classmethod
    def _set_lookup(cls, instance, name:str, df:pDF, at_x:str, to_raise:bool):
        """Initialize and attach a cached `GetValue` instance to the worker."""
        fast_lookup = cls(df, at_x, to_raise=to_raise)
        setattr(instance, name, fast_lookup)
        return fast_lookup
    
    @classmethod
    def _register_lookup(cls, instance, name:str, df:pDF, at_x:str, to_raise:bool):
        """Retrieve or create the cached lookup for the worker instance."""
        if not hasattr(instance, name):
            cls._register_method(instance, name)
            return cls._set_lookup(instance, name, df, at_x, to_raise)
        return getattr(instance, name)
    
    @classmethod
    def _value(cls, 
        instance, 
        df:pDF, at_x:str, x:str, of_col:str|list|None, 
        _name:str=None, to_raise:bool=True
    ):
        """
        <O(1)> vs. <O(N)> `df.loc[df[at_x]==x, of_col].iloc[0]`.

        Wrap for worker instance of GetAtLoop for repeated lookups in loops.
        Cached data auto-refreshes if the DataFrame changes.

        Memory cost: keeps a per-instance cached df.set_index(at_x).      
        """
        name = cls._get_name(_name)
        fast_lookup = cls._register_lookup(instance, name, df, at_x, to_raise)
        if not fast_lookup._is_df_same(df):
            fast_lookup = cls._set_lookup(instance, name, df, at_x, to_raise)                 
        return fast_lookup(x, of_col)

    @classmethod
    def at_x_of_col(cls, at_x:str, of_col:str|list|None=None, to_raise:bool=True):
        """
        Decorator: attach Pd.GetValue behavior directly to worker methods.
        Otherwise, can be passed dynamically when calling the method using GetValue._value.

        Usage
        -----
        The decorator applies as: 
        >>> @Pd.GetValue.at_x_of_col(at_x:str, of_col <optional>)
        >>> def(df: pDF, x: str, of_col <optional>):
        >>>     ...

        Return logic   
        -------------         
        a : Scalar → df.set_index(at_x).at[x, of_col]  
            → `of_col >> str  >> at_x has unique values only >> 10-100x boost`.

        b : Vector → df.set_index(at_x).loc[x, of_col] 
            → `of_col >> list >> at_x has unique values only >> 2-5x boost`.
            
        c : Vector | Matrix → df.loc[df[at_x]==x, of_col] 
            → `NON-unique values in at_x` with standard performance.

        Parameters
        ----------
        at_x :      Column which serves the row condition as at_x==x.
        of_col :    Column(s) which value(s) is returned.
        to_raise :  True → raise when non-unique at_x==x is present 
                    thus different type would be returned 
                    → Series instead of Scalar Value | DataFrame instead of Series.

        Returns
        -------
        a : of_col is str → `Scalar Value` of of_col from row at_x==x.
        b : of_col is a list of cols → `Vector of Values` (pd.Series).
        c : `Vector (of_col is str) | Matrix (pd.DataFrame - of_col as list)` if 
            `non-unique at_x==x occurs <<>> and  to_raise=False`.
        """
        _of_col = of_col
        def decor(fx):
            name = fx.__name__
            @wraps(fx)
            def wrapper(instance, df:pDF, x:str, of_col:str|list|None=None):
                target_col = of_col or _of_col
                return cls._value(instance, df, at_x, x, target_col, name, to_raise)
            return wrapper
        return decor

    @classmethod
    def clear_cache(cls, instance) -> None:
        """
        Clear all cached `GetValue` objects attached to the given instance.

        Use after a pipeline stage completes to free the indexed
        DataFrames from memory.
        """
        reg = getattr(instance, cls.registry, None)
        if not reg:
            return
        for name in list(reg):
            if hasattr(instance, name):
                delattr(instance, name)
        reg.clear()      

    @classmethod
    @contextmanager
    def loop_scope(cls, instance):
        """
        Context manager: auto-clear `GetValue` cache on exit.

        Scopes the cache lifetime to a loop block, ensuring the
        indexed DataFrames are released when the loop completes.

        Examples
        --------
        >>> with GetValue.loop_scope(worker):
        ...     for x in values:
        ...         worker.get_value(df, x)
        """
        try:
            yield
        finally:
            cls.clear_cache(instance)        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Pd:
    """
    Pandas operations utility — hardened data manipulation for ML pipelines.

    Provides a collection of type-aware, edge-case-resistant pandas
    operations that are either performance-critical, error-prone with
    raw pandas, or specific to AIDflow's encoding and training stages.

    Operation Groups
    ----------------
    - **Sorting**: Deterministic mergesort, custom-order sorting.
    - **Concatenation**: Null/empty-safe DataFrame concatenation.
    - **Null masking**: Multi-column `isnull` / `notnull` masks.
    - **Column management**: In-place column repositioning.
    - **Time resolution**: Inference of temporal granularity.
    - **Interval correction**: Precision and empty-bin fixes for
      `pd.qcut` / `pd.cut` boundaries.
    - **Categorical tools**: Code-based categorical construction,
      mode-to-numeric transformation.
    - **Display**: DataFrame styling for reports and diagnostics.
    """
    
    @staticmethod
    def stable_sort(data:pDF, sortby:str|list, **kwargs) -> pDF:
        """
        Deterministic sort using stable mergesort algorithm.

        Guarantees that rows with equal sort keys retain their
        original relative order — critical for reproducibility
        in ML feature pipelines where row order must be stable
        across runs.

        Parameters
        ----------
        data : pDF
            DataFrame to sort.
        sortby : str | list
            Column name(s) to sort by.
        **kwargs
            Passed to `pd.DataFrame.sort_values`.

        Returns
        -------
        pDF
            Sorted DataFrame (new object; original unchanged).
        """
        return data.sort_values(by=sortby, kind='mergesort', **kwargs)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    @staticmethod
    @wraps(pd.concat)
    def concat(dfs:list[pDF|pSe|None], **kwargs) -> pDF|None:
        """
        Null/empty-safe DataFrame concatenation.

        Filters out `None` entries and empty frames before
        concatenation. If all inputs are empty, returns a
        column-aligned empty DataFrame (preserving the union of
        all column names) rather than an error.

        Parameters
        ----------
        dfs : list of pDF | pSe | None
            Frames to concatenate. `None` and empty frames
            are silently excluded.
        **kwargs
            Passed to `pd.concat`.

        Returns
        -------
        pDF | None
            Concatenated frame. If all inputs are empty, a
            column-aligned empty frame. Returns `None` if
            no columns exist at all.

        Notes
        -----
        This is the standard concatenation primitive used by
        the data encoder to merge WOE features with raw data.
        """        
        data = [d for d in dfs if (d is not None) and (not d.empty)]
        if data or isinstance(dfs[0], pSe):  
            return pd.concat(data, **kwargs)    
        
        E = pDF()    
        for d in dfs:
            if d is None: 
                continue
            cols = [c for c in d.columns if c not in E.columns]
            if cols:
                E[cols] = None
                
        if not len(E.columns):
            return None
        return E        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def notnull(df:pDF, *args, all_=True) -> pSe:
        """
        Multi-column `notnull` mask.

        Parameters
        ----------
        df : pDF
            Source DataFrame.
        *args : str
            Column name(s) to check.
        all_ : bool, default=True
            - True  → AND logic (all columns must be non-null).
            - False → OR logic  (any column non-null).

        Returns
        -------
        pSe
            Boolean mask.
        """        
        m = df[args[0]].notnull()    
        for x in args[1:]:
            if all_:
                m &= df[x].notnull()    
            else:
                m |= df[x].notnull()
        return m
    
    @staticmethod
    def isnull(df:pDF, *args, all_=True) -> pSe:
        """
        Multi-column `isnull` mask.

        Parameters
        ----------
        df : pDF
            Source DataFrame.
        *args : str
            Column name(s) to check.
        all_ : bool, default=True
            - True  → AND logic (all columns must be null).
            - False → OR logic  (any column is null).

        Returns
        -------
        pSe
            Boolean mask.
        """        
        m = df[args[0]].isnull()    
        for x in args[1:]:
            if all_:
                m &= df[x].isnull()    
            else:
                m |= df[x].isnull()
        return m
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def move_column_inplace(df:pDF, col:list|str, pos:list|int) -> pDF:
        """
        Reposition column(s) within the DataFrame in-place.

        Parameters
        ----------
        df : pDF
            DataFrame to modify.
        col : str | list
            Column name(s) to move.
        pos : int | list
            Target position(s). Negative indices count from the end.
        """        
        if isinstance(col, str):
            col = [col]
        if isinstance(pos, int):
            pos = [pos]
        lcs = len(df.columns)
        for i,p in enumerate(pos):
            if p < 0: 
                pos[i] = lcs + p
        for i in range(len(col)):
            c = df.pop(col[i])
            df.insert(pos[i], c.name, c)

        return df
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def get_time_resolution(t:pSe) -> str:
        """
        Infer the temporal resolution of a time Series.

        Determines the finest time unit at which the data is
        sampled by examining the median gap between consecutive
        observations.

        Parameters
        ----------
        t : pSe
            Time/datetime Series (monotonic or not).

        Returns
        -------
        str
            One of `'D'`, `'h'`, `'min'`, `'s'`.

        Raises
        ------
        ValueError
            If the time resolution cannot be determined
            (unsupported or irregular spacing).

        Notes
        -----
        Panel data (duplicate timestamps) is handled by filtering
        zero-difference gaps before inference.
        """
        d  = t.sort_values().diff().dropna()
        d  = d[d > pd.Timedelta(0)] # panel data → elimination of same tu
        
        # space data → irrelevant tu
        if not len(d):
            return 'D'

        ts = d.median().total_seconds()
        
        units = {
            'D': 86400,
            'h':  3600,
            'min':  60,
            's':     1
        }
        
        tolerance = 0.05

        # 1. Standard TU tolerance
        for u, s in units.items():
            lb = s * (1 - tolerance)
            ub = s * (1 + tolerance)
            
            if lb <= ts <= ub:
                return u
        
        # 2. If it doesn't fit any standard, try to check if it's a multiple 
        # of one (e.g., data every 4 hours)
        for u, s in units.items():
            if ts % s < (s * tolerance): 
                return u
        
        raise ValueError('AIG SAFETY'
            '\n► Time variable resolution cannot be determined.'
            '\n► Supported time formats: day, hour, minute, second.'
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def get_num_precision(x:pSe|float|int) -> int:
        """
        Determine the maximum decimal precision of a numeric Series or scalar.

        Parameters
        ----------
        x : pSe | float | int
            Numeric data to inspect.

        Returns
        -------
        int
            Maximum number of decimal places (0 for integers).

        Raises
        ------
        ValueError
            If the input is not a numeric dtype.
        """        
        if isinstance(x, float):
            return len( str(x).rstrip('0').split('.')[-1] )
        if isinstance(x, int):
            return 0 
        
        dtype = x.dtype.name.lower()
        if any(x in dtype for x in ('float', 'double')):
            return int( x.fillna(0).astype(str).str.rstrip('0').str.split('.').str[-1].str.len().max() )
        elif 'int' in dtype:
            return 0
        else:
            raise ValueError('AIG SAFETY ► ONLY NUMERIC DATA TYPE CAN BE PASSED TO PRECISION DETERMINATION')

    @classmethod
    def cut_correct_precision(cls, 
        d:pSe, p:int, min_left:int|float = None, max_right:int|float = None
    ) -> pSe:
        """
        Correct interval boundary precision errors introduced by `pd.qcut` / `pd.cut`.

        Pandas internal floating-point arithmetic can produce interval
        endpoints with spurious extra decimal digits (e.g., `0.33000000000000002`).
        This method snaps boundaries back to the source data's precision and
        handles open-ended (first/last) boundaries.

        Parameters
        ----------
        d : pSe
            Categorical Series with `pd.Interval` categories.
        p : int
            Target precision (decimal places).
        min_left : int | float, optional
            Explicit left boundary for the first interval.
            If None, uses `-inf`.
        max_right : int | float, optional
            Explicit right boundary for the last interval.
            If None, uses `+inf`.

        Returns
        -------
        pSe
            Series with corrected interval categories.
            Non-categorical inputs are returned unchanged.
        """
        cat = d.dtype
        
        if cat.name != 'category':
            return d
            
        def correct(x):
            if   fl: return -np.inf if min_left is None else min_left  # r = ( x-a )
            elif lr: return np.inf if max_right is None else max_right
            pa = p+1
            a  = math.pow(10,-pa)
            return round(x+a, pa)
        
        def closed(int_:pd.Interval):
            c = int_.closed
            if not lr:    return c 
            if c=='left': return c
            return 'neither'            
        
        intervals = cat.categories           
        intervals = [int_  for int_ in intervals if isinstance(int_, pd.Interval)]
        mapping   = {}        
        
        l = len(intervals)-1
        
        for i, int_ in enumerate(intervals):
            adj = False        
            ilr = {'l': int_.left, 'r':int_.right}        

            for k,x in ilr.items():
                fl = ( i == 0 and k == 'l' )    # first left
                lr = ( i == l and k == 'r' )    # last right

                if ( cls.get_num_precision(x) > p ) or any([fl,lr]):
                    ilr[k] = correct(x)
                    adj = True        
            if adj: 
                mapping[int_] = pd.Interval(ilr['l'], ilr['r'], closed=closed(int_))
            
        if mapping:
            d = d.cat.rename_categories(mapping)        
        return d    

    @staticmethod
    def _cut_correct_empty_map(v:pSe) -> dict:
        """
        Build a category remapping for empty bins produced by `pd.cut`.

        Identifies intervals with zero observations and merges them
        into the nearest non-empty neighbor by adjusting boundaries.

        Parameters
        ----------
        v : pSe
            Value counts of the categorical intervals.

        Returns
        -------
        dict
            Mapping of original → corrected category.
        """        
        v:pDF = v.sort_index().reset_index()
        v.columns = ['x','c']
        
        o = v['x'].to_dict()
        m = {}

        e = v.loc[v['c']==0,'x']
        v = v.loc[v['c']!=0,'x']        
        vi= v.index
        v = v.to_dict()
        
        for i in e.index:                        
            i_ = vi[np.argmin(abs(vi-i))]
            v_ = v[i_]
            e_ = e[i]
            if i > i_:
                  l,r = v_, e_
            else: l,r = e_, v_
            v[i_] = pd.Interval(l.left, r.right, closed=r.closed)
            m[o[i_]] = v[i_]        
        
        a = {o[i]:o[i] for i in o}
        return a | m
    
    @classmethod
    def _cut_correct_empty(cls, d:pSe, v:pSe) -> pSe:
        """
        Apply empty-bin correction to a categorical Series.

        Parameters
        ----------
        d : pSe
            Categorical Series to correct.
        v : pSe
            Value counts (from `d.value_counts()`).

        Returns
        -------
        pSe
            Series with empty intervals merged into neighbors.
        """        
        m = cls._cut_correct_empty_map(v)
        t = d.dtype        
        if t.name != 'category':
            return d.map(m)
        return d.cat.rename_categories(m)
    
    @classmethod
    def cut_correct_empty(cls, d:pSe) -> pSe:
        """
        Correct empty intervals (zero observations) caused by `pd.cut`.

        Empty bins are merged into their nearest non-empty neighbor,
        eliminating degenerate categories that would confuse downstream
        WOE encoding or model training.

        Parameters
        ----------
        d : pSe
            Categorical Series from `pd.cut`.

        Returns
        -------
        pSe
            Corrected Series. Unchanged if no empty bins exist.
        """
        v = d.value_counts()
        if not any(v==0):
            return d
        return cls._cut_correct_empty(d,v)
    
    @staticmethod
    def _is_cut_lbl_str(kwargs:dict) -> bool:
        """Check if `labels` kwarg contains string labels (skip correction)."""
        l = kwargs.get('labels')
        return (l is not None) and isinstance(l[0], str)
    
    @classmethod
    def qcut(cls, 
        d: pSe, 
        bins: int,
        min_left: int|float = None, 
        max_right:int|float = None, 
        **kwargs
    ) -> pSe:
        """
        Quantile-based binning with corrected interval boundaries.

        Wraps `pd.qcut` and post-processes the result to fix:
        - Floating-point precision artifacts on interval edges.
        - Open-ended boundary normalization.

        Parameters
        ----------
        d : pSe
            Numeric Series to bin.
        bins : int
            Number of quantile bins.
        min_left : int | float, optional
            Explicit left boundary override for the first bin.
        max_right : int | float, optional
            Explicit right boundary override for the last bin.
        **kwargs
            Passed to `pd.qcut` (e.g., `duplicates`, `labels`).

        Returns
        -------
        pSe
            Categorical Series with corrected interval categories.
        """
        p = cls.get_num_precision(d)+1
        d = pd.qcut(d, bins, **kwargs)
        if cls._is_cut_lbl_str(kwargs):
            return d
        return cls.cut_correct_precision(d, p, min_left, max_right)

    @classmethod
    def cut(cls, 
        d: pSe, 
        bins: int|ArrayLike, 
        min_left: int|float = None, 
        max_right: int|float = None, 
        **kwargs
    ) -> pSe:
        """
        Fixed-width binning with corrected interval boundaries.

        Wraps `pd.cut` and post-processes the result to fix:
        - Floating-point precision artifacts on interval edges.
        - Open-ended boundary normalization.
        - Empty bins (merged into nearest neighbor).

        Parameters
        ----------
        d : pSe
            Numeric Series to bin.
        bins : int | array-like
            Number of bins or explicit bin edges.
        min_left : int | float, optional
            Explicit left boundary override for the first bin.
        max_right : int | float, optional
            Explicit right boundary override for the last bin.
        **kwargs
            Passed to `pd.cut` (e.g., `duplicates`, `labels`).

        Returns
        -------
        pSe
            Categorical Series with corrected, non-empty interval categories.
        """
        p = cls.get_num_precision(d)+1
        d = pd.cut(d, bins, **kwargs)
        if cls._is_cut_lbl_str(kwargs):
            return d
        d = cls.cut_correct_precision(d, p, min_left, max_right)
        return cls.cut_correct_empty(d)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def get_len_unique_values(d:pSe, include_missing=False) -> int:
        """
        Count unique values in a Series, with optional missing-value exclusion.

        For categorical dtypes, uses the category count (faster).
        For other dtypes, uses `np.unique`.

        Parameters
        ----------
        d : pSe
            Source Series.
        include_missing : bool, default=False
            If False, missing values (NaN / `VarStd.MISSING`)
            are excluded from the count.

        Returns
        -------
        int
            Number of unique values.
        """
        if ( d.dtype.name.lower() == 'category' ):
            d = d.cat.categories
        else: 
            d = np.unique(d)
        l = len(d)
        if include_missing:
            return l
        if ( VarStd.MISSING in d ) or ( np.nan in d ):
            return l-1        
        return l
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def df_zip(cols, vals, index=None) -> pDF:
        """
        Construct a DataFrame from parallel column names and value arrays.

        Parameters
        ----------
        cols : sequence
            Column names.
        vals : sequence of array-like
            Value arrays (one per column).
        index : array-like, optional
            Index to assign.

        Returns
        -------
        pDF
            Constructed DataFrame.
        """        
        return pDF(dict(zip(cols, vals)), index=index)
    
    @staticmethod
    def style(
        df: pDF, 
        hide_axis: int|list = None, 
        bold_cols: str|list = None, 
        format_cols:   dict = None,
        format:        dict = None,
        color:         dict = None,
        properties:    dict = None
    ) -> pDF|pd_Styler:
        """
        Apply display styling to a DataFrame for reports and diagnostics.

        - hide_axis = [0,1] → 0 index | 1 cols
        - bold_cols = [col1, col2, ...]
        - format_cols = {col:'{:.3f}', ...}
        - format      = {precision=3, thousands=".", decimal=","}
        - color       = {'subset':col, 'condition':fx | None for bool, 'color': ['green','red'] | 'green'}
        - properties  = {'subset':col, 'width':'400px', 'text-align':'left'}
        
        Parameters
        ----------
        df : pDF
            Source DataFrame.
        hide_axis : int | list, optional
            Axes to hide. 0 = index, 1 = columns.
        bold_cols : str | list, optional
            Column(s) to render in bold.
        format_cols : dict, optional
            Per-column format specifiers, e.g. `{'revenue': '{:.2f}'}`.
        format : dict, optional
            Global format kwargs, e.g. `{'precision': 3}`.
        color : dict, optional
            Conditional coloring spec:
            `{'subset': col, 'condition': fx, 'color': 'red' | ['green', 'red']}`.
        properties : dict, optional
            CSS properties, e.g. `{'subset': col, 'width': '200px'}`.

        Returns
        -------
        pDF | pd_Styler
            Styled output. Original `pDF` if no args provided.            
        """
        if not any([hide_axis is not None, bold_cols, format_cols, format, color, properties]):
            return df
        if not isinstance(df, pd_Styler):
            df = df.style        
        if hide_axis is not None:      
            for h in List.ensure(hide_axis):
                df = df.hide(axis=h)
        if bold_cols is not None:      
            df = df.map(lambda x:'font-weight: 600', subset=bold_cols)
        if format_cols is not None: 
            df = df.format(format_cols)
        if format is not None:
            df = df.format(**format)
        if color is not None:        
            cfx = color.pop('condition', lambda x: x)
            col = color.pop('color', None)
            if col is None:             col = [None, 'color: #B50064;']
            elif isinstance(col, str):  col = [None, f'color:{col};']
            else:                       col = [f'color:{col[0]}', f'color:{col[1]}']
            df = df.map(lambda v: col[1] if cfx(v) else col[0], **color)
        if properties is not None:
            df = df.set_properties(**properties)
            align = properties.get('text-align')
            if align:
                ca = dict(selector='th', props=[('text-align', align)])
                df = df.set_table_styles([ca])
        return df
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def sort_values_by_array(
        df: pDF, 
        x:  str, 
        order_array: pSe|list|tuple, 
        reset_index: bool=True, 
        **kwargs
    ) -> pDF:
        """
        Sort a DataFrame by a column using an explicit value order.

        Useful for enforcing domain-specific ordering (e.g.,
        bin labels, severity levels) that does not match
        alphabetical or numeric sort.

        Parameters
        ----------
        df : pDF
            DataFrame to sort.
        x : str
            Column to sort by.
        order_array : pSe | list | tuple
            Explicit ordered list of values (defines the sort order).
        reset_index : bool, default=True
            Reset the index after sorting.
        **kwargs
            Passed to `pd.DataFrame.sort_values`.

        Returns
        -------
        pDF
            Sorted DataFrame.

        Notes
        -----
        All values in `df[x]` must be present in `order_array`.
        Missing values will be placed at the end.
        """
        sort = lambda x: pd.Categorical(
            x, categories=order_array, ordered=True
        )
        return df.sort_values(x, key=sort, ignore_index=reset_index, **kwargs)
    
    @staticmethod
    def create_cat_from_codes(d:pSe, categories:list|tuple=None) -> pSe:
        """
        Sort a DataFrame by a column using an explicit value order.

        Useful for enforcing domain-specific ordering (e.g.,
        bin labels, severity levels) that does not match
        alphabetical or numeric sort.

        Parameters
        ----------
        df : pDF
            DataFrame to sort.
        x : str
            Column to sort by.
        order_array : pSe | list | tuple
            Explicit ordered list of values (defines the sort order).
        reset_index : bool, default=True
            Reset the index after sorting.
        **kwargs
            Passed to `pd.DataFrame.sort_values`.

        Returns
        -------
        pDF
            Sorted DataFrame.

        Notes
        -----
        All values in `df[x]` must be present in `order_array`.
        Missing values will be placed at the end.
        """
        if d.dtype in TypeDef.bool:
            d = d.astype(int)
        if categories is None:
            categories = d.unique()
        return pSe(pd.Categorical.from_codes(d, categories=categories), index = d.index)

    @staticmethod
    def cat_mode_to_num(data:pDF) -> pDF:
        """
        Transform categorical (object-dtype) columns to binary indicator
        of the mode value.

        For each object-dtype column, produces a 0/1 indicator where
        1 marks the most frequent value. Useful for quick numeric
        proxy encoding in diagnostic pipelines.

        Parameters
        ----------
        data : pDF
            Input DataFrame.

        Returns
        -------
        pDF
            Copy with object columns replaced by mode indicators.
            Non-object columns are unchanged.
        """
        vars = data.dtypes         
        vari = vars.index        
        cat  = [vari[i] for i,x in enumerate(vars) if x in TypeDef.object]
        
        if not cat:
            return data

        c = data[cat].describe(include=[object])
        mode = c.loc['top']
        
        d = data.copy()
        d[cat] = (d[cat] == mode).astype(int)
                                
        return d
    
    @staticmethod
    def str_max_len(d:pSe, max_len=35, extension='...') -> pSe:        
        """
        Enforce maximum string length with truncation indicator.

        Parameters
        ----------
        d : pSe
            String Series.
        max_len : int, default=35
            Maximum allowed string length.
        extension : str, default='...'
            Suffix appended to truncated strings.

        Returns
        -------
        pSe
            Series with strings truncated to `max_len` where needed.
            Non-object dtypes are returned unchanged.
        """
        if d.dtype not in TypeDef.object:
            return d
        if d.str.len().max() > max_len:
            d = d.apply(lambda x: x[:max_len] + extension if len(x) > max_len else x)
        return d
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DataFrame:
    """
    Read-only DataFrame proxy. Transparent delegation to pd.DataFrame for all read access; 
    writes raise RuntimeError. 
    Distinct type identity → isinstance against pd.DataFrame returns False.

    **Not Accepted by Offload.**
    """

    def __init__(self, df: pDF):
        self._internal_dataframe = df

    def __getattr__(self, name):
        return getattr(self._internal_dataframe, name)

    def __setattr__(self, name, value):
        if name == '_internal_dataframe':
            object.__setattr__(self, name, value)
        else:
            setattr(self._internal_dataframe, name, value)
    
    def __delitem__(self, key):
        del self._internal_dataframe[key]

    def __setitem__(self, key, value):
        self._internal_dataframe[key] = value
    
    def __getitem__(self, key):
        return self._internal_dataframe[key]
    
    def __len__(self):
        return len(self._internal_dataframe)

    def __iter__(self):
        return iter(self._internal_dataframe)

    def __contains__(self, key):
        return key in self._internal_dataframe

    def __repr__(self):
        return repr(self._internal_dataframe)

    def __str__(self):
        return str(self._internal_dataframe)

    def __eq__(self, other):
        other_df = other._internal_dataframe if isinstance(other, type(self)) else other
        return self._internal_dataframe.equals(other_df)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        return self._internal_dataframe.__getstate__()

    def __setstate__(self, state):
        df = pDF()
        df.__setstate__(state)
        self._internal_dataframe = df
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬