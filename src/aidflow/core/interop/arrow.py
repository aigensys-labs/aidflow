"""
Low-level infrastructure Arrow ops providing high-performance data bridges 
and structural manipulation tools.

It serves as the primary engine for optimizing memory usage and ensuring 
data integrity during the hand-off between different execution layers.

Facilitates the use of PyArrow Tables for large-scale dataset handling, reducing 
memory overhead during high-volume data transformations.

Design Rationale:
    PyArrow's columnar format offers superior memory efficiency for
    large-scale operations (scanning, filtering, aggregating). However,
    the Python ML ecosystem (scikit-learn, LightGBM, XGBoost, etc.)
    operates on row-oriented Pandas DataFrames. This module bridges
    that gap with minimal overhead, ensuring that:
    - Inputs arriving as `pa.Table` are transparently converted to
      Arrow-backed `pDF` for the target function [9].
    - Outputs can be optionally converted back to `pa.Table` for
      downstream pipeline steps.
    - The round-trip conversion strips pandas metadata that would
      confuse external libraries (e.g., `ArrowDtype` markers).
"""

from ...deps.common     import  (
                                wraps, 
                                )
from ...deps.data       import  (
                                np, 
                                pa, 
                                pd, pDF, pSe,
                                )
from ...deps.types      import  (
                                Any, 
                                Literal,
                                Callable,
                                FuncType
                                )
from ...utils.builtins  import  List
from ..factory          import  (
                                Dataclass
                                )
from ..standard         import  TypeDef

class Arrow:
    """
    Apache Arrow ↔ Pandas interoperability engine.

    Provides a set of methods for seamless data transitions between
    PyArrow's columnar format and Pandas' row-oriented format. 
    
    The primary interface is the `transition` decorator, which creates 
    a transparent bridge for any function that needs to work with Arrow 
    Tables at the boundary but standard DataFrames internally.
    """
    
    @Dataclass.base
    class Contract:
        """
        Configuration contract for Arrow ↔ Pandas transitions.

        Defines the rules for how data is converted at the input
        and output boundaries of a wrapped function. Used as the
        parameter for `Arrow.transition`.

        Parameters
        ----------
        safe_conversion : bool = False 
            When True, the resulting DataFrame is guaranteed to have 
            standard numpy dtypes (no `ArrowDtype` markers), at the 
            cost of an additional memory copy. When False, the conversion
            is zero-copy but may retain Arrow-backed dtypes.

        input_columns : list = None 
            Optional list of expected column names. If provided, 
            only these columns are extracted from the input Arrow 
            Table (others are dropped). If `validate_input` is True, 
            missing columns raise a `KeyError`.

        convert_output : bool | list = None  
            Controls output conversion back to `pa.Table`. If `True`, 
            the entire result is converted. If a list of ids [string|int], 
            only those results as res[id] in the output are converted to
            Arrow arrays (e.g. result object is dict or list with inner dfs). 
            If `None` or `False`, the output is returned as-is (Pandas).

        preserve_index : bool = True 
            Whether to preserve the DataFrame index during the 
            Pandas → Arrow conversion. 

        validate_input : bool = True 
            Whether to raise an error when expected columns are missing from 
            the input Arrow Table. 

        Example:
            >>> contract = Arrow.Contract(
            ...     input_columns   = ['feature_a', 'feature_b', 'target'],
            ...     convert_output  = True,
            ...     safe_conversion = True,
            ...     validate_input  = True
            ... )
        """        
        safe_conversion: bool = False
        input_columns:   list = None
        convert_output:  bool | list = None
        preserve_index:  bool = True
        validate_input:  bool = True

    @staticmethod
    def _get_cols(d: pa.Table | pDF) -> list:
        if isinstance(d, pa.Table):
            return d.column_names
        return d.columns.to_list()

    def _get_subset(d: pa.Table | pDF, cols: list) -> pa.Table | pDF:
        if isinstance(d, pa.Table):
            return d.select(cols)
        return d.loc[cols, :]
    
    @classmethod
    def convert_input(cls, 
        obj: Any, 
        contract: Contract, 
        func: FuncType
    ) -> Any:
        """
        Convert an input object to a Pandas DataFrame if it is an Arrow Table.

        Checks whether the input is a `pa.Table` and, if so, applies
        the column selection and validation rules defined in the contract
        before converting to a standard Pandas DataFrame.

        Non-Arrow inputs (e.g., already a `pDF`, a list, a scalar)
        are passed through unchanged.

        Parameters:
            obj (Any): The input object to potentially convert.
            contract (Contract): The transition contract defining
                expected columns and validation rules.
            func (FuncType): The wrapped function (used for error
                message context).

        Returns:
            Any: The converted `pDF` if input was a `pa.Table`,
                otherwise the original object unchanged.

        Raises:
            KeyError: If `contract.validate_input` is True and any
                column in `contract.input_columns` is missing from
                the Arrow Table. The error message includes the
                function name and the list of missing columns.

        Note:
            When `contract.safe_conversion` is True, the resulting
            DataFrame is guaranteed to have standard numpy dtypes
            (no `ArrowDtype` markers), at the cost of an additional
            memory copy. When False (default), the conversion is
            zero-copy but may retain Arrow-backed dtypes.
        """
        if (
            ( not isinstance(obj, (pDF, pa.Table)) ) or
            ( not contract.safe_conversion and isinstance(pDF) )
        ):
            return obj
        
        table: pa.Table | pDF = obj
        cols = cls._get_cols(table)

        if contract.input_columns is not None:
            missing = set(contract.input_columns) - set(cols)

            if missing and contract.validate_input:
                raise KeyError(
                    f"{func.__name__}: missing columns \n► {sorted(missing)}"
                )

            cols  = [c for c in contract.input_columns if c in cols]            
            table = cls._get_subset(table)

        if isinstance(table, pa.Table):
            table = table.to_pandas()

        if contract.safe_conversion:
            return cls.to_safe_datatype(table)
            
        return table

                
    @staticmethod
    def convert_output(obj: Any, contract: Contract) -> Any:
        """
        Convert a Pandas output object to PyArrow format.

        Applies the output conversion rules defined in the contract:
        - If `convert_output` is `True`: the entire result (or
          each element of a tuple/list) is converted to `pa.Table`.
        - If `convert_output` is a list: only the specified columns
          in the output DataFrame are replaced with Arrow-backed arrays.
        - If `convert_output` is `None` or `False`: the output
          is returned unchanged.

        Parameters:
            obj (Any): The output object (DataFrame, tuple, list, or
                other) to potentially convert.
            contract (Contract): The transition contract defining
                output conversion rules.

        Returns:
            Any: The converted object. Arrow-backed columns or full
                Arrow Tables if conversion was requested; otherwise
                the original object.

        Note:
            The index preservation behavior is controlled by
            `contract.preserve_index`. When set to False, the
            resulting Arrow Table will have a default range index.
        """        
        c = contract.convert_output
        
        if c is None:
            return obj
        
        from_pd = lambda x: pa.Table.from_pandas(
            x, preserve_index = contract.preserve_index
        )
                
        if isinstance(c, list):
            for x in c:
                obj[x] = from_pd(obj[x])

        else:
            if isinstance(obj, (tuple, list)):
                no = []
                for v in obj:
                    no.append( from_pd(v) )
                obj = type(obj)(no)
            else:
                obj = from_pd(obj)

        return obj

    @classmethod
    def _wrapper(cls, 
        func:       Callable, 
        args:       tuple, 
        kwargs:     dict,
        contract:   Contract, 
    ) -> Any:                
        ci = lambda x: cls.convert_input(x, contract, func)
        
        new_args = []
        for arg in args:
            new_args.append(ci(arg))

        new_kwargs = {}
        for k, v in kwargs.items():
            new_kwargs[k] = ci(v)

        res = func(*new_args, **new_kwargs)
        del new_args, new_kwargs

        return cls.convert_output(res, contract)
        
    @classmethod
    def transition(cls, contract: Contract = Contract()) -> Callable:
        """
        Decorator that bridges PyArrow Tables to standard Pandas functions.

        Wraps a target function so that:
        1. Input Conversion:
        - Any `pa.Table | pDF` arguments (positional or keyword) are
          automatically converted to `pDF` before the function
          executes.
        - When `contract.safe_conversion is True`, the resulting
          DataFrame is guaranteed to have standard numpy dtypes
          (no `ArrowDtype` markers), at the cost of an additional
          memory copy. When False (default), the conversion is
          zero-copy but may retain Arrow-backed dtypes.

        2. Output Conversion:
        - The function's return value is optionally converted back
           to `pa.Table` based on the `contract.convert_output`.

        Non-Arrow arguments pass through unchanged, making this
        decorator safe to apply to functions that accept mixed
        input types.

        Parameters:
            contract (Contract): The transition rules governing
                input validation, column selection, and output
                conversion. Defaults to an empty contract (no
                validation, no output conversion).

        Returns:
            Callable: A decorator that wraps the target function.

        Example:
            >>> @Arrow.pa_transition(Arrow.Contract(
            ...     input_columns   = ['age', 'income', 'target'],
            ...     safe_conversion = True,
            ...     convert_output  = False,
            ... ))
            ... def train_model(df: pDF) -> pDF:
            ...     # converted input df is guaranteed to have only chosen  
            ...     # input_columns and be a pandas standard DataFrame with 
            ...     # `standard dtypes`
            ...     return df.describe()
            ...
            >>> @Arrow.pa_transition(Arrow.Contract(
            ...     input_columns   = ['age', 'income', 'target'],
            ...     safe_conversion = False,
            ...     convert_output  = True,
            ... ))
            ... def train_model(df: pDF) -> pDF:
            ...     # converted input df is guaranteed to have only chosen 
            ...     # input_columns and be a pandas DataFrame with Arrow backend
            ...     return df.describe()            
            >>>
            >>> # Accepts both:
            >>> result = train_model(pandas_df)      # Passthrough
            >>> result = train_model(arrow_table)    # Auto-converted
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):                
                return cls._wrapper(
                    func, args, kwargs, contract
                )
            return wrapper
        return decorator
        
    @classmethod
    def to_safe_datatype(cls, data: pDF | pSe | np.ndarray) -> pDF | pSe | np.ndarray:
        """
        Ensures that pd DataFrame has all values in pd/np native standards.
        If no ArrowDtype columns are present returns data as-is.   

        Executes from_pandas - to_pandas while stripping pandas metadata
        → critical for standard py libs (numpy, scikit, lgb, ...).

        The original index is preserved.

        Parameters:
            data (pDF  | pSe | np.ndarray): The source DataFrame, potentially 
                containing Arrow-backed columns or non-standard metadata.

        Returns:
            pDF | pSe | np.ndarray: A clean DataFrame with standard numpy dtypes 
                and no Arrow-specific metadata. Safe for consumption by any standard 
                ML library.
        """
        if isinstance(data, np.ndarray):
            return data

        if isinstance(data, pSe):
            if not isinstance(data.dtype, pd.ArrowDtype):
                return data

            s = cls._conversion(data).iloc[:, 0]            
            s.name = data.name
            return s

        x = [x for x,t in data.dtypes.items() if isinstance(t, pd.ArrowDtype)] 
        if not x:
            return data    
        
        d = cls._conversion(data, x)
        c = data.columns
        r = [z for z in c if z not in x]

        d[r] = data[r]
        return d[c]

    @staticmethod
    def _conversion(data:pDF|pSe, x:list=None) -> pDF|pSe:
        if isinstance(data, pSe):
            d = data.to_frame()
        else: 
            d = data[x]
        d = (
            pa.Table.from_pandas(d, preserve_index=False)
            .replace_schema_metadata(None)
            .to_pandas()
        )
        d.index = data.index
        return d  

    def to_pandas(data: pa.Table | pa.ChunkedArray | pDF | pSe) -> pDF | pSe:
        """
        Converts arrow Table to pd dataframe while keeping same mem 
        object by using data.to_pandas(`types_mapper = pd.ArrowDtype`) 
        → native pyarrow dtypes present in the output pDF.

        Pandas input is returned as-is.
        """
        if isinstance(data, (pa.Table, pa.ChunkedArray)):
            return data.to_pandas(types_mapper = pd.ArrowDtype)
        
        if not isinstance(data, (pDF, pSe)):
            raise ValueError(
                'AIG SAFETY ► to_pandas() accepts Arrow or Pandas data objects.'
            )
        return data

    @staticmethod
    def take(table:pa.Table, indicies:int|list = 100, random_state:int = 42) -> pDF:
        """
        Extract a subset of rows from an Arrow Table as a Pandas DataFrame.

        Supports two modes:
        - **Random Sample**: If `indicies` is an integer, selects
          that many random rows (without replacement) using a
          reproducible RNG.
        - **Explicit Indices**: If `indicies` is a list, selects
          rows at those specific positions.

        Parameters:
            table (pa.Table): The source Arrow Table.
            indicies (int | list): Either the number of rows to sample
                (int) or an explicit list of row indices to select.
                Defaults to 100 (random sample).
            random_state (int): Seed for the random number generator.
                Ensures reproducible sampling. Defaults to 42.

        Returns:
            pDF: A Pandas DataFrame containing the selected rows.

        Example:
            >>> # Random sample of 50 rows:
            >>> sample = Arrow.take(arrow_table, 50, random_state=7)
            >>>
            >>> # Specific rows:
            >>> rows = Arrow.take(arrow_table, [0, 5, 10, 50])
        """        
        if isinstance(indicies, int):
            idx = table.num_rows
            rng = np.random.default_rng(random_state)
            indicies = rng.choice(list(range(idx)), size=indicies, replace=False, shuffle=False)
            indicies.sort()
        return table.take(indicies).to_pandas()

    @staticmethod
    def move_column_inplace(table:pa.Table, col:list|str, pos:list|int) -> pa.Table:
        """
        Reorder a column to a new position in the Arrow Table.

        Removes the target column from its current position and
        re-inserts it at the specified index. All other columns
        maintain their relative order.

        Parameters:
            table (pa.Table): The source Arrow Table.
            col (list | str): The column name (or list of names)
                to move. For a single column, pass a string.
            pos (list | int): The target position (0-indexed) where
                the column should be inserted.

        Returns:
            pa.Table: A new Arrow Table with the column at the
                requested position.

        Example:
            >>> # Move 'target' to be the first column:
            >>> table = Arrow.move_column_inplace(table, 'target', 0)
            >>>
            >>> # Move 'id' to the last position:
            >>> table = Arrow.move_column_inplace(table, 'id', len(table.column_names) - 1)
        """        
        cols = list(table.column_names)
        cols.remove(col)
        cols.insert(pos, col)
        return table.select(cols) 

    @staticmethod
    def stable_sort(table:pa.Table, sortby:str|list, ascending:bool=True) -> pa.Table:
        """
        Perform a deterministic (stable) sort on an Arrow Table.

        Uses PyArrow's stable sort algorithm, guaranteeing that rows
        with equal sort-key values retain their original relative
        order. This is critical for reproducible pipeline execution
        where non-deterministic ordering could affect downstream
        sampling or binning.

        Parameters:
            table (pa.Table): The source Arrow Table.
            sortby (str | list): Column name(s) to sort by. If a list
                is provided, sorting is applied hierarchically
                (first element is primary key, second is tiebreaker, etc.).
            ascending (bool): Sort direction. True for ascending
                (default), False for descending.

        Returns:
            pa.Table: A new Arrow Table with rows in sorted order.

        Example:
            >>> # Sort by a single column:
            >>> table = Arrow.stable_sort(table, 'date', ascending=False)
            >>>
            >>> # Multi-key sort (date desc, then id asc):
            >>> table = Arrow.stable_sort(table, ['date', 'id'], ascending=False)
        """
        kind = 'descending'
        if ascending:
            kind = 'ascending'
        return table.sort_by([(x, kind) for x in List.ensure(sortby)])

    @staticmethod
    def add_column(
        data:pa.Table, name:str, array:pSe|pa.ChunkedArray, position:int=None
    ) -> pa.Table:
        """
        Add a new column to an Arrow Table.

        Appends or inserts a column at a specified position. The input
        array is automatically converted to a PyArrow Array if it is
        not already one.

        Parameters:
            data (pa.Table): The source Arrow Table.
            name (str): The name for the new column.
            array (pSe | pa.ChunkedArray): The values for the new
                column. Accepts Pandas Series, NumPy arrays, or
                existing PyArrow arrays.
            position (int): The 0-indexed position where to insert
                the column. If None, the column is appended to the
                end (default).

        Returns:
            pa.Table: A new Arrow Table with the added column.

        Example:
            >>> # Append to the end:
            >>> table = Arrow.add_column(table, 'score', scores_series)
            >>>
            >>> # Insert as the first column:
            >>> table = Arrow.add_column(table, 'id', id_array, position=0)
        """        
        array = pa.array(array)
        if position is not None:
            return data.add_column(position, name, array)        
        return data.append_column(name, array)

    @staticmethod
    def add_columns(
        data: pa.Table, 
        new_data: pa.Table, 
        cols: list[str] = None,
        keys: str | list[str] = None,
        join_type: Literal[
            "left semi", "right semi", "left anti", "right anti", 
            "inner", "left outer", "right outer", "full outer"] = 'left outer'
    ) -> pa.Table:
        """
        Merge columns from a source Arrow Table into a target Arrow Table.

        Supports two alignment strategies:

        - **Key-based (join)**: When `keys` is provided, performs a PyArrow
          join between the two tables and extracts the requested columns.
          Existing columns in the target that overlap with `cols` are
          dropped first to avoid duplicates. The join type is configurable
          (defaults to `'left outer'`).

        - **Positional (append)**: When `keys` is None, columns are
          appended directly from the source table, assuming both tables
          have the same number of rows and are row-aligned.

        Parameters:
            data (pa.Table): The target Arrow Table to enrich.
            new_data (pa.Table): The source Arrow Table containing
                the column values to extract.
            cols (list[str]): Column names to extract from `new_data`.
                If None, all columns in `new_data` are used (default).
            keys (str | list[str]): Join key column name(s). If provided,
                a PyArrow join is performed. If None, positional append
                is used (default).
            join_type (str): The join strategy when using key-based merge.
                One of PyArrow's supported join types.
                Defaults to `'left outer'`.

        Returns:
            pa.Table: A new Arrow Table with the merged columns.

        Raises:
            Exception: If a join key is not present in both tables,
                or if a requested column is not found in `new_data`.

        Example:
            >>> # Positional: append all columns from scores table
            >>> result = Arrow.add_columns(base_table, scores_table)
            >>>
            >>> # Key-based: join on 'user_id', take specific columns
            >>> result = Arrow.add_columns(
            ...     data=base_table,
            ...     new_data=enriched_table,
            ...     cols=['score', 'flag'],
            ...     keys='user_id',
            ...     join_type='inner'
            ... )
            >>>
            >>> # Key-based: multi-key join
            >>> result = Arrow.add_columns(
            ...     data=base_table,
            ...     new_data=lookup_table,
            ...     keys=['account_id', 'period'],
            ...     join_type='left outer'
            ... )
        """
        key_list = [keys] if isinstance(keys, str) else (keys or [])

        if not cols:
            cols = new_data.column_names
        cols = [c for c in cols if c not in key_list]

        existing = [c for c in cols if c in data.column_names]
        if existing:
            data = data.drop_columns(existing)      

        if keys:
            if isinstance(keys, str):
                keys = [keys]
            return data.join(new_data.select(keys + cols), keys=keys, join_type=join_type)
        
        for col_name in cols:
            col_array = new_data.column(col_name)
            data = data.append_column(col_name, col_array)
        return data    