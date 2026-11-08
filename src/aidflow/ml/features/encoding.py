"""
Deterministic Feature Processing and Binning Gateway.

Serves as the critical decision engine for WOE-based discrimination
metrics. It analyzes the distribution and type of a feature to
determine the optimal transformation strategy (Binned vs. Continuous).

Decision Logic
--------------
1. **Type Detection**: Identifies if the feature is numeric or
   categorical.
2. **Sparsity Check**: Evaluates the number of unique values
   against the `th_discrete` threshold to decide between
   `cut` (equal-width) and `qcut` (equal-frequency) binning.
3. **Missing Value Integration**: If `incl_miss` is enabled,
   injects a standardized missing category to ensure nulls are
   captured in the model's discrimination power.
4. **Encoding Path**: Routes the feature to the WOE engine if
   categorical or binned, or retains raw values if continuous.
"""

from ...deps.data       import  pDF, pSe
from ...deps.types      import  NDArray
from ...core.interop    import  Pd, TypeStandard
from ...core.standard   import  TypeDef, VarType
from .binning           import  Binning
from .woe               import  IV, WoE


class Encoding:
    """
    Deterministic Feature Processing and Binning Gateway.

    Analyzes the distribution and type of a feature to determine
    the optimal transformation strategy (Binned vs. Continuous),
    then applies WOE encoding.

    The `process` method implements a strict decision flow:
    1. **Type Detection**: Numeric or Categorical.
    2. **Sparsity Check**: Unique values vs. `th_discrete`
       threshold → selects `cut` (equal-width) or `qcut`
       (equal-frequency).
    3. **Missing Integration**: If `incl_miss` is enabled,
       injects a standardized missing category so nulls are
       captured in the discrimination power.
    4. **Encoding Path**: Routes to WOE if categorical/binned,
       or retains raw values if continuous.

    Parameters
    ----------
    n_bins : int | None, default=None
        Requested number of bins for numerical discretization.
        If None, the method dynamically determines the binning
        strategy based on `th_discrete` and data cardinality.
    transform : bool, default=True
        If True, transforms the feature series to WOE form.
        If False, skips transformation (encoding only).
    ensure_trf : bool, default=False; **`encode applicable`**
        - process: If True, ensures a WOE transformation is returned
        regardless of feature quality (e.g., single-value
        features that would normally return None).
        - encode: If True, also returns first transformation.
    ensure_enc : bool, default=False
        If True, ensures a WOE encoding table is returned
        regardless of feature type or parameter setup.
    add_iv : bool, default=False
        If True, extends the WOE table with Information Value
        per bin.
    incl_miss : bool, default=True; **`encode applicable`**
        If True, missing values are treated as a distinct
        category and included in the WOE encoding.
    th_discrete : int, default=20
        Maximum unique values below which a numeric feature
        is treated as discrete (eligible for `cut` binning
        instead of `qcut`). Only relevant for numeric
        features with `bins > 1`.
    """
    def __init__(self, 
        n_bins:     int = None,
        transform:  bool = True,
        ensure_trf: bool = False,
        ensure_enc: bool = False,
        add_iv:     bool = False,                 
        incl_miss:  bool = True, 
        th_discrete: int = 20
    ):
        self.bins        = n_bins
        self.transform   = transform
        self.ensure_trf  = ensure_trf
        self.ensure_enc  = ensure_enc
        self.add_iv      = add_iv
        self.incl_miss   = incl_miss
        self.th_discrete = th_discrete    
    # ─────────────────────────────────────────────────────────────────────────────────────────────        
    
    @staticmethod
    def _woe_pdf(
        y_true: pSe|NDArray, 
        x:      pSe|NDArray, 
        transform: bool = True,
        add_iv:    bool = False
    ) -> tuple[pSe, pDF, int]:
        """
        Compute WOE encoding and optionally transform the feature.

        Parameters
        ----------
        y_true : pSe | NDArray
            Target variable on which WOE encoding is computed.
        x : pSe | NDArray
            Feature series to be encoded (categorical or binned).
        transform : bool, default=True
            If True, also returns the WOE-transformed series.
            If False, returns only the encoding table.
        add_iv : bool, default=False
            If True, extends the WOE table with IV per bin.

        Returns
        -------
        tuple
            - **pSe | None**: WOE-transformed series (None if
              `transform=False`).
            - **pDF**: WOE encoding table (bin → WOE mapping).
            - **int**: Number of unique bins/categories.
        """
        if transform:
            x_t, woe = WoE.transform(y_true, x, return_encoding=True)
        else:
            x_t = None
            woe = WoE.encode(y_true, x)
        if add_iv:
            woe = IV.extend(woe)
        return x_t, woe, len(woe)

    def _transform(self, 
        y_true: pSe|NDArray, 
        x: pSe|NDArray, 
        is_missing: bool, 
        transform: bool = None
    ):
        """
        Apply missing-value handling and delegate to WOE encoding.

        If the feature contains missing values and `incl_miss`
        is enabled, injects the standardized missing category
        before encoding. Then delegates to `_woe_pdf` for the
        actual WOE computation.

        Parameters
        ----------
        y_true : pSe | NDArray
            Target variable for WOE computation.
        x : pSe | NDArray
            Feature series (categorical or binned).
        is_missing : bool
            Whether to inject the standardized missing category.
        transform : bool | None, default=None
            Override for the class-level `transform` setting.
            If None, resolves from `self.transform`.

        Returns
        -------
        tuple
            - **pSe | None**: WOE-transformed series.
            - **pDF**: WOE encoding table.
            - **int**: Number of bins/categories.
        """
        if is_missing:
            x = TypeStandard.cat_fill_missing(x)
        trf = self.transform if transform is None else transform        
        return self._woe_pdf(y_true, x, trf, self.add_iv)
    
    def _return_continuous(self, y_true: pSe|NDArray, x: pSe|NDArray, u:int):
        """
        Return the feature as-is for continuous (unbinned) path.

        The feature is passed through without transformation.
        A WOE encoding table is only computed if `ensure_enc`
        is enabled (for diagnostic purposes).

        Parameters
        ----------
        y_true : pSe | NDArray
            Target variable (used only if `ensure_enc=True`).
        x : pSe | NDArray
            Continuous feature series.
        u : int
            Number of unique values in the feature.

        Returns
        -------
        tuple
            - **pSe**: Original (untransformed) feature series.
            - **pDF | None**: WOE table (only if `ensure_enc`).
            - **int**: Unique value count.
        """
        woe = None
        if self.ensure_enc:
            woe = self._woe_pdf(y_true, x, False, self.add_iv)[1]
        return x, woe, u

    @staticmethod
    def _to_se(y_true: pSe|NDArray, x: pSe|NDArray) -> pSe:
        """
        Ensure the feature is a Series with aligned index.

        If `x` is already a Series, returns it unchanged.
        Otherwise, wraps it in a Series using the index from
        `y_true` (if available) for proper alignment.

        Parameters
        ----------
        y_true : pSe | NDArray
            Target series (source of index if needed).
        x : pSe | NDArray
            Feature to convert.

        Returns
        -------
        pSe
            Feature as a Series.
        """
        if isinstance(x, pSe):
            return x
        i = y_true.index if isinstance(y_true, pSe) else None
        return pSe(x, index=i)
    # ─────────────────────────────────────────────────────────────────────────────────────────────        
    
    def process(self, 
        y_true: pSe|NDArray, 
        x:      pSe|NDArray,
    ) -> tuple[pSe|None, pDF|None, int|None]:
        """
        Analyze the feature and return WOE-encoded output.

        Determines the optimal transformation strategy based on
        feature type, cardinality, and missing values, then
        applies the appropriate WOE encoding path.

        Decision Flow
        -------------
        1. **Insufficient variance**: If unique values < 2 and
           no missing values → return (None, None, None) unless
           `ensure_trf` is set (in which case returns a
           zero-filled series).
        2. **Categorical / Object**: Direct WOE encoding on
           unique values (with missing integration if applicable).
        3. **Numeric, no bins requested, has missing**: Treat
           as discrete (`bins = min(th_discrete, unique)`)
           to capture the missing category.
        4. **Numeric, continuous**: Return as-is (no encoding).
        5. **Numeric, binned**: Apply `cut` or `qcut`
           binning, then WOE encode.

        Parameters
        ----------
        y_true : pSe | NDArray
            Binary target used to compute WOE discrimination.
        x : pSe | NDArray
            Feature series to be analyzed and encoded.

        Returns
        -------
        tuple
            - **pSe | None**: Transformed feature (WOE values)
              or original (continuous path).
            - **pDF | None**: WOE encoding table.
            - **int | None**: Number of bins/categories.
            - **(None, None, None)**: Feature has insufficient
              variance to be useful.
        """          
        x = self._to_se(y_true, x)        
        bins = self.bins

        is_num_type = ( x.dtype in TypeDef.numeric )
        is_missing  = self.incl_miss and ( x.isnull().sum() > 0 )
        
        # NUM Vars discretization condition
        # ─────────────────────────────────────────────────
        u = Pd.get_len_unique_values(x)
        c = ( u < self.th_discrete ) # discreate NUM vars → use cut instead of qcut

        if u < 2 and ( not is_missing ):
            if not (self.ensure_trf or self.ensure_enc):
                return tuple([None]*3)
                        
            return self._transform(y_true, x, is_missing)
        
        # ─── (a) Categorical/Object FE 
        # → Encoding to be able directly compute Gini
        # ─────────────────────────────────────────────────
        if ( not is_num_type ):
            return self._transform(y_true, x, is_missing)

        # ─── (b) NUM FE 
        # → Requested orig. values with missing part 
        # → Approximate orig.values by woe + missing
        # ─────────────────────────────────────────────────
        if ( bins is None ) and ( is_missing ):
            bins = min(self.th_discrete, u)
            c = ( bins == u )

        # ─── (b.1) orig 
        # → Continuous fe
        # ─────────────────────────────────────────────────
        if not bins:
            return self._return_continuous(y_true, x, u)

        # ─── (b.2) bins
        # → Binning Applied 
        # → Encoding thus Missing part can be included directly
        # ─────────────────────────────────────────────────
        if c: x = Pd.cut(x,  bins, duplicates='drop') 
        else: x = Pd.qcut(x, bins, duplicates='drop') 

        return self._transform(y_true, x, is_missing)
    # ─────────────────────────────────────────────────────────────────────────────────────────────        

    def encode(self, d:pSe|pDF, encoding:pDF, x:str, xT:str, vtype:str=None
    ) -> pSe | tuple[pSe, pSe]:
        """
        Apply a pre-computed WOE encoding to new data.

        Routes the data to the appropriate binning transform
        (numerical interval-based or categorical set-membership)
        and maps values to their target bin labels.

        Parameters
        ----------
        d : pSe | pDF
            New data to encode. If a DataFrame, the column
            `x` is extracted.
        encoding : pDF
            Pre-computed WOE encoding table (from `process`).
            Contains bin definitions and target labels.
        x : str
            Feature column name (source).
        xT : str
            Target column name (encoded WOE label).
        vtype : str = None
            Optionally, vtype by VarType standard can be provided to
            enforce application of the first transformation by vtype.

        Returns
        -------
        pSe | tuple[pSe, pSe]
            Series of target WOE labels (categorical dtype),
            same length as input.
            Additional series from first data 
            transformation when ensure_trf = True.
        """
        if isinstance(d, pDF):
            d = d[x]

        if not self.incl_miss:
            d = d[~d.isna()]

        return self._encode(d, encoding, x, xT, vtype)

    def _encode(self, d:pSe, encoding:pDF, x:str, xT:str, vtype:str = None
    ) -> pSe | tuple[pSe, pSe]:
        """
        Core binning transform — routes to numerical or categorical.

        Selects the appropriate binning strategy based on the
        data type:
        - **Numeric**: Interval-based assignment (each value is
          assigned to the bin whose interval contains it,
          respecting closure boundaries).
        - **Categorical**: Set-membership assignment (each value
          is assigned to the bin group that lists it as a
          component).
        """
        is_num = (
            vtype == VarType.N 
            if vtype is not None else 
            d.dtype in TypeDef.numeric
        )
        if is_num:
            trf = Binning.transform_numerical_data(d, encoding, x)
        else:
            trf = Binning.transform_categorical_data(d, encoding, x)  

        enc = Binning.data_mapping(trf, encoding, x, xT)
        
        if not self.ensure_trf:
            return enc
        
        return enc, trf