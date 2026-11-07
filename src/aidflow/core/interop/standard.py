"""
AIDflow Data Governance & Standardization Engine.

This module implements the fundamental deterministic layer of AIDflow. It provides 
the low-level utilities required to enforce strict data types, standardize temporal 
resolutions, and ensure dimensional consistency across the entire pipeline. 

By centralizing these operations, AIDflow eliminates 'silent casting' errors and 
guarantees that data flowing into fitters is identical in structure regardless of 
the source ingestion method.
"""

from ...deps.data           import  (
                                    pa, pa_pc,
                                    pd, pDF, pSe
                                    )
from ...deps.types          import  (
                                    Literal, 
                                    Callable
                                    )
from ...core.foundation     import  AigStop

from ...display.rgb         import  RgbANSI
from ...utils.builtins      import  List

from ..standard             import  (
                                    VarStd, 
                                    TypeDef
                                    )
from .arrow                 import  Arrow
from .pandas                import  Pd


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# AIG FLOW DATA STANDARDS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class DataStandard:
    """
    Deterministic Data Manipulation Utility.
    
    Provides a suite of static methods to enforce structural standards on 
    Pandas objects. This class ensures that all data entering the AIDflow 
    ecosystem adheres to required dimensions and types before being passed 
    to downstream workers.
    """
    _error = lambda x: AigStop.custom(f'AIG ► DATA STANDARD ► {x}')

    @classmethod
    def _cols(cls, var:str|list) -> list:
        if isinstance(var, str): 
            var = [var]
        elif not isinstance(var, list):
            cls._error(
                'INPUT COLUMNS ► Data Standardization '
                'for specified column(s) either as str|list'
            )
        return var
            
    @classmethod
    def _pdf_cols(cls, data:pDF, var:str|list):
        """Enforce data type standards to AIG flow."""
        if not isinstance(data, pDF):
            cls._error(
                'INPUT DATA ► Data provided to Standardization '
                'has to be pd.DataFrame instance'
            )
        return cls._cols(var)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def move_column(data: pa.Table|pDF, name:str, i:int) -> pa.Table|pDF:
        """
        Move column in data specified by name to the ith position.

        Parameters
        ----------
        data : pa.Table | pDF
            The dataset for column reorder.
        name : str
            The column to move.
        i : int
            The target position of the column in the data.
            
        Returns
        -------
        pa.Table | pDF
            Data with the column in the position.        
        """
        if isinstance(data, pDF):
            return Pd.move_column_inplace(data, name, i)    
        return Arrow.move_column_inplace(data, name, i)
            
    @classmethod
    def sort(cls, data:pa.Table|pDF, sortby:str|list, ascending:bool=True, **kwargs
    ) -> pa.Table|pDF:
        """
        Deterministic Dataset Sorting.
        
        Performs a stable sort on the dataset to ensure that row indices remain 
        consistent across different execution environments—a critical requirement 
        for auditability and reproducibility in regulatory scoring.

        Parameters
        ----------
        data : pa.Table | pDF
            The dataset to be sorted.
        sortby : str | list
            The column(s) to sort by.
        **kwargs : dict
            Additional sorting parameters passed to the internal pandas engine.

        Returns
        -------
        pa.Table | pDF
            A deterministically sorted DataFrame.
        """
        if isinstance(data, pa.Table):
            return Arrow.stable_sort(data, sortby, ascending)

        cols = cls._pdf_cols(data, sortby)
        return Pd.stable_sort(data, cols, ascending = ascending, **kwargs)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def get_cols(data: pa.Table|pDF) -> list:
        """Return data columns."""
        if isinstance(data, pa.Table):
            return data.column_names
        return data.columns

    @staticmethod
    def subset_data(data: pa.Table|pDF, cols:list[str]) -> pa.Table|pDF:
        """Subset data by cols."""
        if isinstance(data, pa.Table):
            return data.select(cols)
        return data[cols]
    
    @classmethod
    def columns(cls, data: pa.Table|pDF, cols:list, flow_vars:list|dict) -> pa.Table|pDF:
        """
        Dimensionality Control & Column Filtering.
        
        Manages the exact subset of features that enter a specific pipeline step. 
        Supports both 'keep' and 'remove' strategies via dictionaries, ensuring 
        that critical flow variables are always preserved while noise is pruned.

        Parameters
        ----------
        data : pa.Table | pDF
            The source dataset.
        cols : list | str | dict, optional
            Selection criteria: a list of columns to keep, a string for single column, 
            or a dictionary with 'keep' or 'remove' keys.
        flow_vars : list | dict, optional
            High-priority variables that must be preserved regardless of the filter.

        Returns
        -------
        pa.Table | pDF
            A DataFrame containing only the selected and validated dimensions.
        """
        cols_ = cls.get_cols(data)
        if cols is None:
            cols = [] 
        if isinstance(cols, dict):
            if cols.get('remove'):
                cols = [c for c in cols_ if c not in List.ensure(cols['remove'])]
            elif cols.get('keep'):
                cols = List.ensure(cols['keep'])
            else:
                cls._error(
                    'COLUMNS CHOICE ► Dictionary type must have '
                    'either "keep" or "remove" as an action key'
                )
        elif isinstance(cols, str):
            cols = [] if not cols else [cols]
        elif not isinstance(cols, list):
            cls._error(
                'COLUMNS CHOICE ► List or Dictionary type '
                'must be provided for Input Columns choice.'
            )
        if not cols: 
            return data
        if flow_vars is None: 
            flow_vars = []
        cols = list(dict.fromkeys(flow_vars + cols)) # dedup. while keeping the req. order
        cols = [c for c in cols if c in cols_]
        return cls.subset_data(data, cols)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def _add_column(cls, data: pa.Table|pDF, name:str, array:pSe, position:int=None) -> pa.Table|pDF:
        if name in cls.get_cols(data):
            return data
                
        if isinstance(data, pDF):
            array.name = name
            data = Pd.concat([data, array], axis=1)
            if position is not None:
                Pd.move_column_inplace(data, name, position)
            return data
        
        return Arrow.add_column(data, name, array, position)     
      
    @classmethod
    def _add_dummy_time(cls, data: pa.Table | pDF, yy_mm:str, aig_time:str) -> pa.Table | pDF:
        """Ensures time variable presence in the data."""
        if yy_mm in cls.get_cols(data):
            return data
        
        end_t = pd.Timestamp.now().normalize()
        dummy = pd.date_range(
            end=end_t, periods=len(data), freq='min'
        ).to_period('h').astype(str)

        return cls._add_column(data, aig_time, dummy, 0)

    @classmethod
    def _aig_time(cls, 
        data: pa.Table | pDF, 
        yy_mm: str, 
        yy_mm_format: str = None, 
        aig_time: str = 'AIG_YY_MM'
    ) -> pSe | pa.ChunkedArray:
        """Extends Data with Standard Time Variable Used Across the Flow."""
        cols = cls.get_cols(data)

        if aig_time in cols:
            return data[aig_time]
        
        t = data[yy_mm]
        t = Arrow.to_pandas(t)
        
        if yy_mm_format: 
              t_:pSe = pd.to_datetime(t, errors='raise', format=yy_mm_format)
        else: t_:pSe = pd.to_datetime(t, errors='raise', format='ISO8601')
        
        p = Pd.get_time_resolution(t_)
        return t_.dt.to_period(p).astype(str) 
        # NOTE str type to avoid error "Object of type Period is not serializable"
    
    @classmethod
    def aig_time(cls, 
        data: pa.Table | pDF, 
        yy_mm: str, 
        yy_mm_format: str = None, 
        aig_time: str = 'AIG_YY_MM'
        ) -> pa.Table | pDF:
        """
        Standardized Temporal Alignment.

        Extends the dataset with a standardized time variable (`AIG_YY_MM`). 
        This is a core requirement for AIDflow's temporal stability checks and 
        Out-of-Time (OOT) validation, ensuring all dates are cast to a consistent 
        serializable period format.

        Parameters
        ----------
        data : pa.Table | pDF
            The dataset to be extended with the standard time variable.
        yy_mm : str
            The name of the original date/time column in the input data.
        yy_mm_format : str, optional
            Explicit format for parsing dates. Defaults to ISO8601 if not provided.
        aig_time : str, optional
            The target name for the standardized time variable.

        Raises
        ------
        ValueError
            If the input date column does not meet the required format.
        """         
        # Verify presence
        data = cls._add_dummy_time(data, yy_mm, aig_time)
        
        try: 
            time = cls._aig_time(data, yy_mm, yy_mm_format, aig_time)
            data = cls._add_column(data, aig_time, time, 0)

        except Exception as e:
            AigStop.custom((
                f'\n► {RgbANSI.bold('TIME VARIABLE')} {RgbANSI.bold_red(yy_mm)} '
                f'{RgbANSI.bold('DOES NOT MEET PROVIDED FORMAT')} {RgbANSI.bold_red(yy_mm_format)}'
                '\n► PLEASE CHECK INPUT DATA AND STANDARDIZE TIME VARIABLE FORMAT'
                f'\n\n{e}'
            ), red=False, bold=False)
        
        return data
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def time_to_period(
        data:   pSe|pDF, 
        period: Literal['day','week','month'] = 'month', 
        yy_mm:  str = 'AIG_YY_MM',
        s:      str = '-'
    ) -> pSe:
        """
        Temporal Resolution Converter.

        Transforms time-like series into standardized string periods (Year-Month, 
        Year-Week, etc.). This allows the pipeline to aggregate data and 
        calculate stability metrics across different temporal granularities.
        """
        if isinstance(data, pDF): data = data[yy_mm]
        d = pd.to_datetime(pSe(data))
        x = lambda t: t.astype(str).str.zfill(2)

        if period == 'week' : 
            k = d.dt.isocalendar()
            return k.year.astype(str) +s+ x(k.week)

        ym = x(d.dt.year) +s+ x(d.dt.month)
        if period == 'month': 
            return ym
        return ym +s+ x(d.dt.day)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class TypeStandard(DataStandard):
    """
    Strict Variable Type Enforcement.

    Extends DataStandard to provide a set of explicit casting utilities. In regulated 
    industries, 'Object' types are a primary source of model failure; this class 
    eliminates that risk by forcing variables into specific numerical, boolean, 
    or categorical states before they reach the fitters.
    """
    _error = lambda x: AigStop.custom(f'AIG ► VARS STANDARD ► {x}')

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # ► CONVERTORS OBJ-NUM-BOOL-TIME VARS
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_func(fx: Callable, data:pDF, var:list, inplace:bool) -> pDF | None:
        df = data[var].apply(fx)
        if inplace:
            data[var] = df
            return
        return df

    @classmethod
    def safe_num_downcast(cls,
        data: pDF, 
        x: str | list | None = None, 
        keep_arrow: bool = True, 
        inplace: bool = True
    ) -> pDF | None:
        """
        Safely downcasts 64-bit float columns to 32-bit float across both NumPy
        ('float64') and PyArrow ('double[pyarrow]') dtypes.

        Parameters
        ----------
        df : pDF
            Input DataFrame.
        x : str | list | None = None
            Subset of variables present in data. When None - all variables are considered.
        keep_arrow : bool 
            If True, Arrow columns become 'float32[pyarrow]'.
            If False, converts Arrow columns to standard NumPy 'float32'.        
        """
        dtypes = data.dtypes
        if x:
            dtypes = dtypes[List.ensure(x)]

        def fx(se: pSe):
            dt = dtypes[se.name]

            if isinstance(dt, pd.ArrowDtype):
                target_dtype = "float32[pyarrow]" if keep_arrow else "float32"
                return se.astype(target_dtype)
            
            return se.astype("float32")

        cols = [x for x, dt in dtypes.items() if dt in TypeDef.float64]

        return cls._apply_func(fx, data, cols, inplace)

    @classmethod
    def safe_to_numeric(cls, data: pSe) -> pSe:
        """
        Backend-agnostic numeric coercion with guaranteed null-mask integrity.
        Ensure any resulting IEEE NaNs have validity bit set to 0 (true Null)

        Handles both native Pandas (NumPy/object) and ArrowDtype inputs cleanly,
        ensuring .notnull() and .isna() evaluate correctly on coerced NaNs.
        """
        d = pd.to_numeric(data, errors="coerce")

        if (
            isinstance(d.dtype, pd.ArrowDtype) or 
            hasattr(getattr(d, "array", None), "_pa_array")
        ):
            col = d.array._pa_array
            if pa_pc.any(pa_pc.is_nan(col)).as_py():
                clean_pa_arr = pa_pc.if_else(pa_pc.is_nan(col), None, col)
                return pd.Series(
                    pd.arrays.ArrowExtensionArray(clean_pa_arr),
                    index = d.index,
                    name  = d.name
                )

        return d

    @classmethod
    def obj_to_num(cls, data:pDF, var:list|str, inplace:bool=True) -> pDF | None:
        """Cast object-type variables to numerical types using coercion."""
        var = cls._pdf_cols(data,var)
        if not var: 
            return                
        return cls._apply_func(cls.safe_to_numeric, data, var, inplace)

    @classmethod
    def obj_to_bool(cls, data:pDF, var:list|str, inplace:bool=True) -> pDF | None:
        """Cast object-type variables to boolean type."""
        var = cls._pdf_cols(data,var)
        if not var:
            return
        
        fx = lambda x: x.astype(bool)
        return cls._apply_func(fx, data, var, inplace)

    @classmethod
    def bool_to_num(cls, data:pDF, var:list|str, inplace:bool=True) -> pDF | None:
        """Standardize boolean flags to integer (0/1)."""
        var = cls._pdf_cols(data,var)
        if not var:
            return
        
        fx = lambda x: x.astype(int)
        return cls._apply_func(fx, data, var, inplace)

    @classmethod    
    def time_to_num(cls, data:pDF, var:list|str, inplace:bool=True) -> pDF | None:
        """Convert Timedelta durations into total seconds (int)."""
        var = cls._pdf_cols(data,var)
        if not var:        
            return
        if not all(data[x].dtype in TypeDef.timedelta for x in var):
            cls._error(
                'TIMEDELTA TO NUM ► Not all provided Variables are timedelta dtype'
            )
        
        fx = lambda x: x.dt.total_seconds()
        return cls._apply_func(fx, data, var, inplace)        
                    
    @classmethod
    def any_to_obj(cls, data:pDF, var:list|str, inplace:bool=True) -> pDF | None:
        """Force variables to object type (standardizing strings/mixed types)."""
        var = cls._pdf_cols(data,var)
        if not var:        
            return
        
        fx = lambda x: x.astype(TypeDef.object[0])
        return cls._apply_func(fx, data, var, inplace) 

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # ► CONVERTORS CATEGORICAL VARS
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def get_verify_subset(cls, 
        data: pSe | pDF, 
        var: list | str = None, 
        e_message:  str = ''
    ) -> pSe | pDF:
        """
        Input Validation for Categorical Operations.
        
        Ensures that categorical transformations are only applied to compatible 
        Pandas types (DataFrame or Series) and that variables are correctly specified.
        """
        if isinstance(data, pSe):
            return data
        
        if not isinstance(data, pDF):
            cls._error(
                'CAT ► Data Categorical operations can be'
                'applied on DataFrame or Series type only'
            )
        if var is None:
            cls._error(
                'CAT ► DataFrame input requires specifying '
                f'Variable(s) {e_message}'
            )
        var = cls._pdf_cols(data, var)
        return data[var]
    
    @classmethod
    def check_cat_series(cls, data:pSe) -> None:
        """
        Verify that a series is explicitly cast as the 'category' dtype.
        """
        if not isinstance(data, pSe):
            cls._error(
                'CAT ► Changing and/or Adding Categories '
                'can be applied on Series type only'
            )        
        if not data.dtype.name in TypeDef.category:
            cls._error('CAT ► Provided Series is not Categorical type')    

    @staticmethod
    def get_dtypes_cat(data:pDF) -> pSe:
        """
        Returns stable version of data.dtypes[data.dtypes.isin('category')]; 
        avoding pd hash mismatch behavior.

        Reasoning example
        -----------------
        pandas isin uses hash, thus, wrongly matches expected equality, e.g.:
        
        >>> dc  = data.astype('category')
        ... dt  = dc.dtypes
        ... cat = dt[dt.isin(['category'])]
        ... len(cat) == len(dt)
        ... [output]: False

        Even when all vars are 'category' dtype, likely isin result will be a few columns.        
        """
        return pSe({x:d for x,d in data.dtypes.items() if isinstance(d, pd.CategoricalDtype)})

    @staticmethod
    def is_cat_missing(data:pSe) -> bool:
        """Check for the existence of a standardized 'MISSING' category in categorical types."""
        dtype = data.dtype
        return ( dtype.name in TypeDef.category ) and ( VarStd.MISSING not in dtype.categories )    
    
    @classmethod
    def any_to_cat(cls, data:pSe|pDF, var:list|str=None) -> pDF | pSe:
        """
        Force cast variables to pandas Categorical type for memory efficiency and speed.
        """
        m = 'for Categorical conversion.'
        d = cls.get_verify_subset(data, var, m) 
        return d.astype(TypeDef.category[0])        

    @classmethod
    def cat_add_not_in_train(cls, data:pSe) -> pSe:
        """Categorical type variable add standard not in train category."""
        cls.check_cat_series(data)
        return data.cat.add_categories(VarStd.NA_TRAIN)
    
    @classmethod
    def cat_add_missing(cls, data:pSe) -> pSe:
        """Inject the standard 'MISSING' category into a categorical variable."""
        cls.check_cat_series(data)  
        return data.cat.add_categories(VarStd.MISSING)

    @classmethod
    def cat_fill_missing(cls, data:pSe|pDF, var:list|str=None) -> pSe | pDF:
        """
        Standardize all nulls/NaNs in categorical variables to the 'MISSING' label.
        """
        m = 'for Missing fill.'
        d = cls.get_verify_subset(data, var, m)
        if isinstance(d, pSe):
            if cls.is_cat_missing(d):
                d = cls.cat_add_missing(d)
        return d.fillna(VarStd.MISSING)

    @classmethod
    def obj_to_cat(cls, 
        data: pDF, 
        var: list|str, 
        verify_obj_type: bool = False, 
        inplace: bool = True
    ) -> pDF | None:
        """
        Complete Object → Categorical Pipeline.
        
        A high-level utility that strips whitespace, handles missing value 
        standardization, and casts variables to categorical types in a 
        single atomic operation.
        """
        var = cls._pdf_cols(data,var)
        if not var:
            return
        
        if (
            verify_obj_type and 
            ( sum(data[var].dtypes == TypeDef.object[0]) != len(var) )
        ):
            cls._error(
                'CAT ► All Variables must be Object dtype'
                'for Standardization to Categorical dtype'
            )
        
        empty = data[var].count()==0
        empty = empty[empty].index.to_list()
        
        dc = cls.cat_fill_missing(data, var)
        dc = dc.astype(str)

        cat = [x for x in var if x not in empty]          
        for x in cat:
            dc[x] = dc[x].str.strip()
        if cat: 
            dc[cat] = dc[cat].replace({x:VarStd.MISSING for x in TypeDef.missing})

        dc = cls.any_to_cat(dc, var)

        if inplace:
            data[var] = dc
            return
        
        return dc
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬