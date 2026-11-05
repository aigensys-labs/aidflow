"""
Data Quality Screening and Type Enforcement for the ML Pipeline.

This module implements the pre-modelling data standardisation stage, responsible
for:

1. **Feature Screening** (fit): Detects and removes features that fail
   quality gates — blank, excessive missing, non-informative, noisy,
   over-correlated, or of unsupported type.
2. **Type Enforcement** (transform): Aligns new (inference) data to the
   exact dtypes and format learned during fit, preventing silent type
   drift between train and test.

Design Principles
-----------------
- **Fit-Transform contract**: `fit` learns the canonical type mapping
  and quality verdicts from the training data. `transform` enforces
  that mapping on new data, raising or warning on inconsistency.
- **Auto-repair over failure**: Mixed-type columns (numeric stored as
  string, timedelta, bool) are repaired automatically when the dominant
  type is clear. The user is notified but not blocked.
- **Conservative drops**: A feature is only removed when it clearly
  fails a threshold. Borderline cases are kept and flagged.
- **Flow vars are sacred**: Variables designated as `flow_vars` (IDs,
  time keys) are preserved throughout the pipeline and excluded from
  screening.
"""

from ....deps.data                  import  np, pDF, pSe
from ....deps.struct                import  dataclass
from ....deps.display               import  display
from ....deps.types                 import  Callable

from ....core.base                  import  (
                                            FitterDataBase,
                                            FitterModelBase
                                            )
from ....core.factory               import  Dataclass
from ....core.foundation            import  (
                                            AigParam, 
                                            AigStop
                                            )
from ....core.interop               import  Pd
from ....core.display               import  Display
from ....core.interop               import  TypeStandard
from ....core.standard              import  (
                                            TypeDef, 
                                            TypeDefinitions
                                            )
from ....core.accessor              import  Rgb, Format
from ....core.logging               import  Logger

from ....ml.features                import  Correlation

from ....utils.builtins             import  List, Str
from ....display.verbosity          import  Verbosity


logger = Logger(__name__)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → CLEANING → FIT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FitStandards:
    """
    Type conversion and enforcement functions applied during fit and transform.

    Attributes
    ----------
    aig_time : Callable
        Parse and normalise the pipeline time variable to a standard format.
    time_num : Callable
        Convert timedelta/timedelta-like columns to numeric (days or seconds).
    bool_num : Callable
        Convert boolean columns to 0/1 numeric.
    obj_num : Callable
        Convert object/string columns containing numeric data to numeric dtype.
    obj_bool : Callable
        Convert object columns containing boolean-like data to proper bool.
    num_cast : Callable
        Safe downcast of float64 → float32 (preserves precision where needed).
    """
    aig_time = TypeStandard.aig_time
    time_num = TypeStandard.time_to_num
    bool_num = TypeStandard.bool_to_num
    obj_num  = TypeStandard.obj_to_num
    obj_bool = TypeStandard.obj_to_bool
    num_cast = TypeStandard.safe_num_downcast

@Dataclass.base
class FitParamsBase:
    """
    Data quality screening thresholds.

    All thresholds define the boundary at which a feature is **removed**
    from the pipeline. Values are inclusive upper bounds — a feature is
    kept when its metric is ≤ the threshold.

    Attributes
    ----------
    max_miss : float, default=0.7
        Maximum allowed missing rate (fraction of NaN). Features with
        > 70% missing are dropped as they provide insufficient signal.
    max_1val : float, default=0.9
        Maximum allowed share of the single most frequent value. Features
        where > 90% of values are identical carry no discriminative
        information and are dropped.
    max_corr : float, default=0.9
        Maximum allowed absolute correlation with another feature.
        When two features exceed this threshold, the one with higher
        missing rate is dropped (redundancy elimination).
    max_var : float, default=0.9
        Maximum allowed "noise ratio" for numerical features — the
        fraction of values that are unique when rounded to 0 decimal
        places. A ratio > 90% suggests the column is an ID or timestamp
        rather than a meaningful feature.
    cat_max : int, default=50
        Maximum allowed number of distinct categories. Categorical
        features with > 50 categories are likely IDs or high-cardinality
        encodings unsuitable for standard ML treatment.
    downcast : bool | int, default=True
        Enable safe float64 → float32 downcast for memory efficiency.
        Set to False to preserve full precision (e.g., for regulatory
        reproducibility requirements).
    """      
    max_miss: float = 0.7
    max_1val: float = 0.9 
    max_corr: float = 0.9 
    max_var:  float = 0.9 
    cat_max:  int   = 50
    downcast: bool|int = True

@Dataclass.base(update_specs=dict(slots=False))
class FitParamsCfg:
    """
    Pipeline configuration parameters.

    Attributes
    ----------
    time_var : str
        Name of the raw time column in the input data.
    time_format : str
        Format string for parsing ``time_var``.
    new_time_var : str
        Standardised time column name used throughout the pipeline.
    flow_vars : list
        Column names to preserve as-is (IDs, keys, indices). These are
        excluded from quality screening and type conversion.
    verbose : bool | Verbosity, default=True
        Enable progress display, warnings, and result output.
    """    
    time_var:     str
    time_format:  str
    new_time_var: str
    flow_vars:   list
    verbose: bool|Verbosity = True

@Dataclass.base
class FitParams(FitParamsBase, FitParamsCfg):
    """Fitter params input design."""

@Dataclass.base
class FitResult:
    """
    Fit output: the learned type mapping and screening verdicts.

    This object is passed to ``transform`` to enforce consistency
    between training and inference data.

    Attributes
    ----------
    vars_type : pSe
        Final dtype per feature (after auto-repair), indexed by feature name.
    vars_all : list
        All feature names (flow_vars + screened features), in canonical order.
    numeric : list
        Features classified as numeric (post-repair).
    object : list
        Features classified as categorical/object (post-repair).
    t_type_fix : list
        Features that were auto-repaired (e.g., numeric-as-string → numeric).
    timedelta : list
        Features converted from timedelta to numeric.
    bool : list
        Features converted from bool to numeric.
    vars_n_o : list
        Union of numeric + object features (the model-ready set).
    vars_dropped : list
        Features removed during screening, with reason.
    missing : pSe
        Missing rate per feature in the training data (for drift detection).
    """
    vars_type:      pSe 
    vars_all:       list 
    numeric:        list
    object:         list
    t_type_fix:     list
    timedelta:      list       
    bool:           list
    vars_n_o:       list
    vars_dropped:   list
    missing:        pSe   

@Dataclass.base
class FitStats:
    """
    Diagnostic statistics from the fit stage.

    Used for user-facing display and audit trail.

    Attributes
    ----------
    vars_org : pSe
        Original dtypes before any conversion (pre-repair).
    d_typefix : list
        Detailed repair log: which features were mixed-type, the
        numeric/string split ratio, and the auto-repair decision.
    corr_upper : pDF
        Upper-triangle correlation matrix (post-screening numeric features).
    corr_removed : int
        Number of features removed due to correlation redundancy.
    """
    vars_org:     pSe
    d_typefix:    list  
    corr_upper:   pDF
    corr_removed: int

SCHEMA = Dataclass.schema(TypeDefinitions)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class FitBase:
    """
    Base class for data quality screening and type management.

    Provides the shared infrastructure for:
    - Feature tracking (type registry, drop list, repair log)
    - Data manipulation (adjacent column updates, consolidation)
    - Quality gate evaluation (blank, missing, non-informative, noisy)
    - Type conversion dispatch (timedelta, bool, mixed numeric/string)
    """

    @dataclass(slots=True)
    class VarData:
        x:   str    # var name
        xs:  pSe    # var data series
        xc:  int    # var non-miss count
        xvc: pSe    # var non-miss value counts

    def __init__(self, params:FitParams, verbose=True):
        self._assert()

        self.prm = params
        self.verbose = verbose

        self.adj_cols       = {}
        self.vars_dropped   = []
        self.t_tdelta       = []     # train timedelta type
        self.t_bool         = []     # train bool type
        self.t_typefix      = []     # train typefix obj -> num
        self.d_typefix      = []     # dash info of typefix

    @staticmethod
    def _assert():
        """
        Verify that FitResult schema matches the canonical TypeDefinitions.

        Raises
        ------
        ValueError
            If any required type key is missing from the FitResult schema.
        """        
        fr = Dataclass.schema(FitResult)
        if any(
            x not in fr.keys() for x in [
                SCHEMA.numeric, 
                SCHEMA.object, 
                SCHEMA.timedelta, 
                SCHEMA.bool, 
                SCHEMA.missing
        ]):
            raise ValueError(
                'AIG SAFETY ► FitResult must match TypeDefinitions types exactly.'
            )

    # Data ops
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def set_remove_flow_vars_type(self, vars_types:pSe) -> pSe:
        """
        Partition features into "screen" and "preserve" sets.

        Flow vars (IDs, keys) and the time variable are excluded from
        quality screening. Their types are recorded separately so they
        can be re-appended to the final result.

        Parameters
        ----------
        vars_types : pSe
            Full dtype mapping (all features in the data).

        Returns
        -------
        pSe
            Dtypes of exogenous features to screen (flow vars removed).
        """        
        self.flow_vars_types = None
        f = self.prm.flow_vars
        v = vars_types.index
        n = v.isin(f)
        if sum(n):
            self.flow_vars_types = vars_types[n]    # keep types of vars needed in the flow
        r = v.isin( f + [self.prm.time_var] )       # vars to remove from processing
        return self.data.dtypes[v[~r]]              # exogenous vars-types for processing
    
    def add_flow_vars_type(self, vars_types:pSe) -> pSe:
        """Re-append preserved flow var types to the screened result."""
        return Pd.concat([self.flow_vars_types, vars_types])
    
    def set_vars_type_to_process(self, vars_types: pSe) -> None:
        """Register the active type mapping for the current screening pass."""
        self.vars_types_ = vars_types.copy()

    def get_vars_type(self, sort=True) -> pSe:
        """
        Retrieve the current type mapping.

        Parameters
        ----------
        sort : bool
            Return alphabetically sorted (True) or original order (False).

        Returns
        -------
        pSe
            Feature name → dtype mapping.
        """        
        vt = self.vars_types_
        if sort: 
            return vt.sort_index()       
        return vt 

    def update_vars_type(self) -> None:
        """Refresh the dtype entry for the current feature after conversion."""
        self.vars_types_[self.x] = self.var.xs.dtype

    def get_num_vars(self, sort=True) -> list:
        """List feature names classified as numeric."""
        vt = self.get_vars_type(sort)
        return [x for x in vt.index if vt[x] in TypeDef.numeric]
        
    def get_obj_vars(self, sort=True) -> list:
        """List feature names classified as object/categorical."""
        vt = self.get_vars_type(sort)
        return [x for x in vt.index if vt[x] in TypeDef.object]
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
        
    def set_data(self, data:pDF) -> None:
        """Set the working data frame for the current pass."""
        self.data = data

    def _update_data(self, du:pDF | list[pDF], cols:list) -> None:     
        if not cols:
            return
        data = Pd.concat(
            List.ensure(du) + [self.data.drop(cols, axis=1)],
            axis=1
        )
        self.data = data[self.data.columns]

    def update_data(self, du:pDF) -> None:
        """Replace specific columns in the working data with transformed versions."""            
        cols = list(du.columns)
        self._update_data(du, cols)

    def consolidate_data(self) -> None:
        """Merge all pending column adjustments back into the main data frame."""
        du = list( self.adj_cols.values() )
        self._update_data(du, self.adj_cols)

    def apply_adj(self, fx: Callable) -> None:
        """
        Apply a transformation function to the current feature column.

        The result is stored in the adjustment buffer (`adj_cols`) and
        consolidated later. This defers writes to avoid repeated DataFrame
        reconstruction during per-feature processing.

        Parameters
        ----------
        fx : Callable
            Transformation function `(data, col, inplace) → data`.
        """        
        d = self.adj_cols.get(self.x)
        d = self.data if d is None else d

        d: pDF = fx(d, self.x, inplace=False)
        if not len(d.columns):
            return
        self.adj_cols[self.x] = d
    
    def downcast_float_vars(self, x:str=None):
        """
        Apply safe float64 → float32 downcast.

        Parameters
        ----------
        x : str, optional
            Specific feature to downcast. If None, downcasts all numeric columns.
        """        
        if not self.prm.downcast:
            return
        if x:
            self.apply_adj(FitStandards.num_cast)
        else:
            dn = FitStandards.num_cast(self.data, inplace=False)
            self.update_data(dn)          


    # Var ops
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    @staticmethod
    def _ensure_se(d:pDF|pSe) -> pSe:
        if isinstance(d, pSe):
            return d
        return d.squeeze()
    
    @staticmethod
    def _counts(xs:pSe) -> tuple[int, pSe]:
        xc  = xs.count()
        xvc = xs.value_counts()  
        return xc, xvc
    
    def get_col(self, x: str) -> pSe:
        """Retrieve a feature column, preferring the adjustment buffer."""        
        if x in self.adj_cols:
            return self._ensure_se( self.adj_cols[x] )
        return self.data[x]
    
    def set_var(self, x:str) -> None:   
        """Load a feature into the working state for quality evaluation."""        
        self.x = x
        xs = self.get_col(x)
        self.var = self.VarData(x, xs, *self._counts(xs))
    
    def update_var(self) -> None:
        """Re-read the current feature and refresh its type in the registry."""
        self.set_var(self.x)
        self.update_vars_type()
   
    def remove_var(self, reason:str, x:str=None) -> None:
        """Mark a feature as dropped and record the reason."""
        if x is None: x = self.x
        self.vars_dropped.append(
            {'Feature':x, 'Type':self.vars_types_[x], 'Reason':reason}
        )
        del self.vars_types_[x]

    def is_var_type(self, t:str) -> bool:
        """Check if the current feature belongs to a type group (e.g., 'numeric', 'bool')."""
        return self.vars_types_[self.x] in TypeDef[t]
    
    # Quality Checks def
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def is_blank(self) -> bool:
        """True if the feature has zero non-missing values (100% NaN)."""
        return not self.var.xc
    
    def is_missing(self) -> bool:
        """True if missing rate exceeds `max_miss` threshold."""
        c = ( self.var.xs.isnull().sum() / len(self.var.xs) )
        return c  > self.prm.max_miss
    
    def is_non_informative(self) -> bool:
        """True if the most frequent value exceeds `max_1val` share (near-constant)."""
        c = ( self.var.xvc.iloc[0] / self.var.xc )
        return c > self.prm.max_1val
    
    def is_num_noisy(self) -> bool:
        """
        True if the numerical feature is likely an ID or timestamp.

        Computed as the fraction of values that become unique when rounded
        to 0 decimal places. A ratio > `max_var` (default 90%) indicates
        the column carries per-row identity rather than signal.
        """        
        c = ( len( self.var.xvc.index.round(0).value_counts() ) / self.var.xc )
        return c > self.prm.max_var
    
    def is_cat_noisy(self) -> bool:
        """True if the number of distinct categories exceeds `cat_max`."""
        return len(self.var.xvc) > self.prm.cat_max
    
    # Data Type Convertors by fx
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def timedelta_to_num(self):
        """
        Convert timedelta-type feature to numeric (if applicable).

        No-op if the feature is not of timedelta type.
        """        
        if not self.is_var_type(SCHEMA.timedelta):
            return 
        self.apply_adj(FitStandards.time_num)
        self.update_var()
        self.t_tdelta.append(self.x)

    def bool_to_num(self):
        """
        Convert boolean-type feature to 0/1 numeric (if applicable).

        No-op if the feature is not of bool type.
        """        
        if not self.is_var_type(SCHEMA.bool):
            return
        self.apply_adj(FitStandards.bool_num)
        self.update_var()
        self.t_bool.append(self.x)        
        
    def fix_numerical_and_string(self):
        """
        Detect and auto-repair mixed numeric/string or numeric-as-string columns.

        Two cases are handled:
        - **Mixed**: column contains both numeric-parseable and string values.
          Decision: if numeric count > string count → cast to numeric; else keep as categorical.
        - **Numeric-as-string**: column is typed object but all non-missing
          values are parseable as numeric → cast to numeric.

        A repair log entry is recorded for user visibility.
        """        
        num = TypeStandard.safe_to_numeric(self.var.xs).notnull().sum()
        
        if not num:
            return
        
        str_ = (self.var.xvc.index.map(type)==str).sum()
        
        if (            
            str_ or
            ( ( not str_ ) and self.is_var_type(SCHEMA.object) )
        ):
            self._fix_numerical_and_string(num, 'Mixed Data Type - Numerical and String')

    def _fix_numerical_and_string(self, num:int, w:str):   
        str_= self.var.xc - num
        dcs = str_ / num
        decision = 'Categorical'
        if dcs < 1:
            self.apply_adj(FitStandards.obj_num)
            self.downcast_float_vars(self.x)
            decision = 'Numerical'
        d = {
            'Feature':self.x, 'Non-Missing': f'{self.var.xc:,}', 
            'NUM Events': f'{num:,}','CAT Events': f'{str_:,}', 
            'CAT/NUM': f'{dcs:.1%}', 'Reason':w + ' → ' + decision
        }
        logger.warning(
            f'Conversion → {decision} → ' +
            ' | '.join(f'{k}: {v}' for k, v in d.items() if k!='Reason')
        )
        self.d_typefix.append(d)
        self.update_var()
        self.t_typefix.append(self.x)   
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def corr(self):
        """
        Screen numeric features for correlation redundancy.

        Features exceeding `max_corr` absolute correlation with a more
        populated feature are dropped. The less-populated feature is
        removed (higher missing rate = less reliable).

        Results are stored in `corr_upper` (matrix) and `corr_removed`
        (count) for the stats output.
        """
        num = self.get_num_vars()
        self.corr_upper, self.corr_removed = [None]*2
        if not num:
            return
        corr, to_drop = (
            Correlation(self.prm.max_corr, verbose=self.verbose).process(self.data[num])
        )
        self.corr_upper = corr.astype(np.float32).round(5)
        self.corr_removed = len(to_drop)
        for x in to_drop:                
            self.remove_var(f'Correlation over {self.prm.max_corr:.0%} with more populated Feature', x)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def verify_types(self):
        """
        Final cross-check: ensure all remaining features are numeric or object.

        Any feature with an unexpected dtype (e.g., datetime leaking through,
        int categories) is dropped and the user is warned.
        """
        current_types = self.get_vars_type(False)
        allowed_types = self.get_num_vars(False) + self.get_obj_vars(False)
        if len(current_types) == len(allowed_types):
            return
        miss = [x for x in current_types.index if x not in allowed_types]
        type_= current_types[miss]
        for x in miss: 
            self.remove_var('Unknown Feature Type', x)
        if not self.verbose:
            return
        Format.splitter()
        Format.display(f'WARNING: Unexpected Data Types of variables: {', '.join(miss)}')
        Format.display('Please check data structure of these variables before data import ► '
              'AIG will ignore listed variables for the next Phases')
        display(type_)    
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class FitWorker(FitBase):
    """
    Data quality screening: learns canonical types and removes unreliable features.

    Executes the per-feature screening pipeline in a fixed order:

    1. **Blank check** — drop features with 100% missing values.
    2. **Time feature check** — drop unregistered datetime columns.
    3. **Type repair** — auto-fix mixed numeric/string, timedelta, bool.
    4. **Missing rate** — drop if > `max_miss`.
    5. **Non-informative** — drop if single value dominates > `max_1val`.
    6. **Numeric noise** — drop if likely ID/timestamp (> `max_var` unique).
    7. **Categorical noise** — drop if > `cat_max` distinct categories.
    8. **Correlation** — drop redundant highly-correlated numeric features.
    9. **Type verification** — drop any remaining unexpected dtypes.

    The output is a `FitResult` (learned type mapping + drop list) and
    `FitStats` (diagnostic data for display/audit).
    """             

    def __init__(self,params:FitParams, verbose:bool=True):
        super().__init__(params, verbose)

    # Quality Checks by Var
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def var_quality_check(self, x):
        """
        Execute the full screening pipeline for a single feature.

        Checks are evaluated in priority order; the first failing gate
        determines the drop reason. Type repairs (steps 3–5) are applied
        before quality gates to ensure the correct type is being evaluated.

        Parameters
        ----------
        x : str
            Feature name to screen.
        """        
        self.set_var(x)
        
        # 1 Missing
        if self.is_blank():
            self.remove_var('100% Blank Feature')
            return        
        
        # 2 Time Features
        if self.is_var_type(SCHEMA.datetime):
            self.remove_var('Unspecified Static Time Feature')
            return 
        
        # 3 Convert to proper type Num/Obj   
        self.fix_numerical_and_string()
        self.timedelta_to_num()
        self.bool_to_num()

        # 4 (Any) Excessive Missing Rate
        if self.is_missing():
            self.remove_var(f'Missing > {self.prm.max_miss:.0%}')

        # 5 (Any) Non-Informative Feature
        elif self.is_non_informative():
            self.remove_var(f'Only 1 Value over {self.prm.max_1val:.0%} of all Values')

        # 6 (Num)
        elif self.is_var_type(SCHEMA.numeric):
            # 6.1 Noisy Feature - ID or Time VAR most likely  
            if self.is_num_noisy():
                self.remove_var('Suspiciously High Variability - ID/Time Feature?')

        # 7 (Cat)
        elif self.is_var_type(SCHEMA.object):
            # 7.1 Noisy Feature - Max Number of Categories    
            if self.is_cat_noisy():
                self.remove_var(f'Number of Categories over Max {self.prm.cat_max}')    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_fit_result(self) -> tuple[FitResult, FitStats]:
        num = self.get_num_vars()
        obj = self.get_obj_vars()        
        n_o = num + obj                       # sorted num-cat    

        v_t = self.get_vars_type()[n_o]      # vars dtypes sorted num-cat
        v_t = self.add_flow_vars_type(v_t)
        v_all = self.prm.flow_vars + n_o

        missing = self.data[n_o].isna().sum()/len(self.data)

        return (
        FitResult(
            vars_type    = v_t,
            vars_all     = v_all,
            vars_n_o     = n_o,   
            numeric      = num,
            object       = obj,
            vars_dropped = self.vars_dropped,
            t_type_fix   = self.t_typefix,
            timedelta    = self.t_tdelta,
            bool         = self.t_bool,
            missing      = missing,
        ), 
        FitStats(
            vars_org     = self.vars_types_org,
            d_typefix    = self.d_typefix,
            corr_upper   = self.corr_upper,
            corr_removed = self.corr_removed
        ))
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @AigParam.with_setattr_runtime(verbose=None)    
    def process(self, data:pDF, vars_types:pSe, **kwargs) -> tuple[FitResult, FitStats]:
        """
        Execute the full data quality screening pipeline.

        Parameters
        ----------
        data : pDF
            Training data (all columns).
        vars_types : pSe
            Pre-determined feature types (from upstream type inference).
        kwargs : dict
            Forwarded to progress display.

        Returns
        -------
        tuple[FitResult, FitStats]
            Learned type mapping + screening diagnostics.
        """
        
        self.set_data(data)
        self.downcast_float_vars()
        # NOTE: Downcasting is positioned here (not at data load) to preserve
        # full-precision input/output reproducibility and regulatory traceability.
        # If memory is constrained during prototyping, moving to load step is safe.

        vars_types_ = self.set_remove_flow_vars_type(vars_types)
        self.set_vars_type_to_process(vars_types_)
        self.vars_types_org = vars_types_

        to_scan, progress = Display.processing_progress(
            vars_types_.index, verbose=self.verbose, desc='Data Quality Check'
        )
        for x in to_scan:
            Display.update_progress(progress)
            self.var_quality_check(x)          

        self.consolidate_data()

        self.corr()
        self.verify_types()  

        return self._get_fit_result()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → CLEANING → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols(FitResult):
    """Transfomer output design."""
    vars_all:   list 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class TransWorker:
    """
    Enforce learned type standards on new (inference) data.

    Applies the type mapping learned during `fit` to incoming data,
    detecting and handling type drift between train and test.

    Enforcement Pipeline (in order):
    1. **Time alignment** — parse and normalise the time variable.
    2. **Timedelta** — convert to numeric (raise if new data disagrees with train).
    3. **Bool** — convert to 0/1 (attempt repair if new data disagrees).
    4. **Numeric** — ensure numeric type (attempt repair from string if needed).
    5. **Object/Categorical** — verify type consistency (info only).
    6. **Downcast** — apply float64 → float32 where trained.
    7. **Exact format** — force exact dtype match with training.
    8. **Missing drift** — warn if missing rates diverge significantly.

    Parameters
    ----------
    params : FitParams
        Pipeline configuration (time vars, downcast, flow vars).
    fit_result : FitResult
        Learned type mapping and screening output from the fit stage.
    """       

    def __init__(self, params: FitParams, fit_result:FitResult):
        self.t_prm   = params
        self.t_res   = fit_result
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    def set_data(self, data:pDF) -> None:
        """Align data with trained variables."""
        time = (
            self.t_prm.time_var, 
            self.t_prm.time_format,
            self.t_prm.new_time_var
        )        
        cols   = data.columns
        subset = [x for x in self.t_res.vars_all if x in cols]

        if not subset:
            raise AigStop.custom(
                'AIG SAFETY ► Input data has to contain known columns from training, '
                'provided data has overlap gap.'
            )

        if all([True if x in subset else False for x in time[:-1]]):
            FitStandards.aig_time(data,*time)
        
        self.data = data[subset]
        self.vars_types = self.get_vars_types()

    def update_data(self, du: pDF | list[pDF|pSe], cols: list=None) -> None:
        cols = cols or du.columns
        data = Pd.concat(
            List.ensure(du) + [self.data.drop(cols, axis=1)], 
            axis=1
        )
        self.data = data[self.data.columns]

    def get_data(self) -> pDF:
        return self.data

    def get_cols(self, trained_kind:list) -> list:
        return [x for x in trained_kind if x in self.data.columns]
    
    def get_vars_types(self) -> pSe:
        return self.data.dtypes
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _info(self, wrong_x: list, text: Callable[[str], str]):
        for x in wrong_x:
            w = f'Feature {x} {text(x)}'
            logger.warning(w)

    def _info_exact(self, fcc:list):
        head = 'Applying Trained Types to New Data → Inconsistency present → Forcing Trained Type'
        data  = {
            'TRAIN' : self.t_res.vars_type[fcc], 
            'NEW'   : self.vars_types[fcc]
        }
        al  = Str.fx_align(list(data))
        msg = lambda x: f'{x} DATA {al(x)}→ {data[x].to_dict()}'

        for x in (head, msg('TRAIN'), msg('NEW')):
            logger.warning(x)
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    def apply_type(self, fx: Callable, cols:list) -> None:
        dt = fx(self.data, cols, inplace = False)                
        self.update_data(dt)

    def ensure_type(self, 
        expected_type: str, 
        fx_transform: Callable = None, 
        fx_fix_wrong_type: Callable = None, 
        to_raise: bool = False
    ) -> None:        
        """
        Enforce a trained type on all features of that type.

        Parameters
        ----------
        expected_type : str
            Type group to enforce (e.g., 'numeric', 'bool', 'timedelta').
        fx_transform : Callable, optional
            Standard conversion to apply (e.g., bool → 0/1).
        fx_fix_wrong_type : Callable, optional
            Repair function for features whose dtype disagrees with training.
        to_raise : bool
            If True, raise on unrepairable mismatch (blocks execution).
            If False, warn and attempt repair.

        Behaviour
        ---------
        - Features with correct type → apply standard transform (if provided).
        - Features with wrong type but repaired during train → apply fix silently.
        - Features with wrong type and NOT repaired during train → warn user.
          If `to_raise`, stop execution.
        """        
        vars_fitted, type_def = self.t_res[expected_type], TypeDef[expected_type]
        
        vars_fitted = self.get_cols(vars_fitted)
        if not vars_fitted:
            return
        
        wrong_type  = [x for x in vars_fitted if self.vars_types[x] not in type_def]

        if wrong_type:                
            not_fixed_on_train = [x for x in wrong_type if x not in self.t_res.t_type_fix]
            # NOTE warn only if different behavior from train
            
            i = lambda x: (
                f'was determined as {expected_type} during TRAIN '
                f'but NEW DATA type is {self.vars_types[x]} → Auto Fixed Attempt'
            )
            self._info(not_fixed_on_train, i)
            
            if not_fixed_on_train and to_raise:
                raise ValueError(
                    '\n ► AIG EXECUTION HAS BEEN STOPPED'
                    '\n ► Scoring Data Types must be fixed prior AIG prediction'
                )
            
            if fx_fix_wrong_type is not None:
                self.apply_type(fx_fix_wrong_type, wrong_type)

        if fx_transform is not None:
            self.apply_type(fx_transform, vars_fitted)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def get_downcast_float64(self) -> list:
        if not self.t_prm.downcast:
            return []
        
        v_fitted = [
            x for x in self.t_res[SCHEMA.numeric] 
            if self.t_res.vars_type[x] in TypeDef.float32
        ]
        v_fitted = self.get_cols(v_fitted)
        v_types  = self.get_vars_types()

        return [x for x in v_fitted if v_types[x] in TypeDef.float64]
        
    def apply_downcast(self) -> None:
        """Apply float64 → float32 downcast to identified columns."""
        cols = self.get_downcast_float64()
        if not cols:
            return        
        self.apply_type(FitStandards.num_cast, cols)
    # ───────────────────────────────────────────────────────────────────────────────────────────── 
    
    def ensure_exact_format(self):
        """
        Force exact dtype match with training for all model features.

        Even if a feature passes the type-group check (e.g., both train and
        test are "numeric"), the exact dtype may differ (int64 vs float32).
        This step enforces the precise training dtype.

        Mismatches are logged in the info report for user visibility.
        """
        vars_type = self.get_vars_types()  # update after applied standards
        cols = self.get_cols(self.t_res.vars_n_o)
        if not cols:
            return
        fcc = ( self.t_res.vars_type[cols] != vars_type[cols] )
        if not sum(fcc):
            return
        fcc = fcc[fcc].index.to_list()           
        du  = []
        for x in fcc:
            du.append(
                self.data[x].astype(self.t_res.vars_type[x])
            )
        self.update_data(du, fcc)
        self._info_exact(fcc)                  
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def verify_missing(self):
        """
        Detect missing-rate drift between train and new data.

        Flags two cases:
        - **Blank**: a feature that was non-missing in train is 100% missing
          in new data (likely a data pipeline issue).
        - **Significant drift**: missing rate in new data exceeds training
          rate by > 15 percentage points.

        Warnings are logged for user visibility. Execution continues.
        """
        cols = self.get_cols(self.t_res.vars_all)
        miss = self.data[cols].isna().sum() / len(self.data)
        blank = list(miss[miss == 1].index)            

        cols_ = self.get_cols(self.t_res.vars_n_o)
        signi = {}
        for x in cols_:
            diff = miss[x] - self.t_res.missing[x]
            if ( diff > 0.15 ) and ( x not in blank ): 
                signi[x] = diff
        
        i = lambda x: 'is 100% missing in NEW DATA but was non-missing in TRAIN '
        self._info(blank, i)     
        
        i =  lambda x: f'has {signi[x]:.1%}% higher Missing rate in NEW DATA → TRAIN missing rate was {self.t_res.missing[x]:.1%}'
        self._info(signi, i)   
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @AigParam.with_setattr_runtime(verbose=None)    
    def process(self, data:pDF, **kwargs) -> pDF:
        """
        Execute the full type-enforcement pipeline on new data.

        Parameters
        ----------
        data : pDF
            New/inference data to transform.
        kwargs : dict
            Forwarded to runtime config.

        Returns
        -------
        pDF
            Transformed data with types enforced to match training.

        Raises
        ------
        AigStop
            If data has zero overlap with trained features.
        ValueError
            If a critical type mismatch is detected (timedelta) and
            auto-repair fails.
        """        
        self.set_data(data)        

        self.ensure_type(SCHEMA.timedelta, FitStandards.time_num, to_raise=True)      
        self.ensure_type(SCHEMA.bool,      FitStandards.bool_num, FitStandards.obj_bool)      
        self.ensure_type(SCHEMA.numeric,   fx_fix_wrong_type = FitStandards.obj_num)      
        self.ensure_type(SCHEMA.object)    # info only → obj to cat is reponsibility of higher layer

        self.apply_downcast()

        self.ensure_exact_format()
        self.verify_missing()                 

        return self.get_data()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → CLEANING → DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Fitter(FitterDataBase):
    """Learn and Transform Data Features to expected trained dtypes and format."""
    
    def __init__(self, params: FitParams):
        self.prm = params
        
    def fit(self, data:pDF, fe_types:list):
        """Learn-Fix dtypes and format standards of dataset."""
        fitter = FitWorker(self.prm, self.prm.verbose)        
        result = fitter.process(data, fe_types)
        self.result = result[0]
        self.stats  = result[1]
        return self
    
    def transform(self, data:pDF):        
        """Transformation of dataset based on fitted standards."""
        worker = TransWorker(self.prm, self.result)
        return worker.process(data)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeatureStandardizer(FitterModelBase[Fitter]):
    """
    Data Quality Screening and Type Enforcement.

    Screens features for quality (missing, non-informative, noisy,
    correlated) and enforces type consistency between training and
    inference data. This is the first gate before any modelling begins —
    it ensures the ML pipeline receives clean, well-typed features.

    Parameters
    ----------
    max_miss : float, default=0.7
        Maximum allowed missing rate (fraction of NaN). Features with >
        70% missing are dropped as they provide insufficient signal.
    max_1val : float, default=0.9
        Maximum allowed share of the single most frequent value. Features
        where > 90% of values are identical carry no discriminative
        information and are dropped.
    max_corr : float, default=0.9
        Maximum allowed absolute correlation with another feature.
        When two features exceed this threshold, the one with higher
        missing rate is dropped (redundancy elimination).
    max_var : float, default=0.9
        Maximum allowed "noise ratio" for numerical features — the
        fraction of values that are unique when rounded to 0 decimal
        places. A ratio > 90% suggests the column is an ID or timestamp
        rather than a meaningful feature.
    cat_max : int, default=50
        Maximum allowed number of distinct categories. Categorical
        features with > 50 categories are likely IDs or high-cardinality
        encodings unsuitable for standard ML treatment.
    downcast : bool | int, default=True
        Enable safe float64 → float32 downcast for memory efficiency.
        Set to False to preserve full precision (e.g., for regulatory
        reproducibility requirements).

    time_var : str
        Name of the raw time column in the input data.
    time_format : str
        Format string for parsing `time_var`.
    new_time_var : str
        Standardised time column name used throughout the pipeline.
    flow_vars : list
        Column names to preserve as-is (IDs, keys, indices). These are
        excluded from quality screening and type conversion.
    verbose : bool | Verbosity, default=True
        Enable progress display, warnings, and result output.

    Notes
    -----
    **Auto-repair**: Mixed-type columns (numeric stored as string,
    timedelta, bool) are repaired automatically when the dominant type
    is clear. The user is notified via the repair log but not blocked.

    **Fit-Transform contract**: The type mapping learned during `fit`
    is enforced during `transform`. Type drift between train and test
    is detected and reported. Critical mismatches (e.g., timedelta)
    raise and stop execution.

    **Downcast**: Float64 → float32 is applied for memory efficiency.
    Position is intentionally after screening to preserve full-precision
    input/output for regulatory traceability.

    Examples
    --------
    >>> from aidflow.fitters import FeatureStandardizer
    >>> std = FeatureStandardizer(
    ...     time_var='date',
    ...     time_format='%Y-%m',
    ...     new_time_var='AIG_YY_MM',
    ...     flow_vars=['id', 'entity_id'],
    ...     max_miss=0.7,
    ... )
    >>> std.fit(data_train, fe_types)
    >>> data_test = std.transform(data_test)
    """

    def __init__(self, *args, params: dict|FitParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(FitParams, params, args, kwargs)

    def fit(self, data:pDF, fe_types:list):
        """
        Screen training data and learn canonical type standards.

        Parameters
        ----------
        data : pDF
            Training data.
        fe_types : list
            Feature type assignments from upstream inference.

        Returns
        -------
        self
            Fitted instance ready for `transform`.
        """
        self._fitter = Fitter(self._params)
        self._fitter.fit(data, fe_types)
        return self
    
    def transform(self, data:pDF):
        """
        Apply fitted type standards to new data.

        Parameters
        ----------
        data : pDF
            New/inference data.

        Returns
        -------
        pDF
            Type-aligned data ready for modelling.
        """
        return self.fitter.transform(data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬