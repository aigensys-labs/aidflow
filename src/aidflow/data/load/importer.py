"""
Data Ingestion and Standardization — Pipeline Entry Point.

This module implements the first stage of the AIDflow pipeline, responsible
for importing data from external sources and standardizing it into the
canonical format expected by all downstream stages.

Supported Sources
-----------------
- **File**: Parquet, CSV, or any format readable by PyArrow (via file path).
- **Database**: Any DB with a PyArrow-compatible connector (PostgreSQL,
  DuckDB, etc.) via connection string or raw SQL.
- **Spark**: Spark tables (catalog.db.table) or SQL queries via an active
  PySpark session.

All sources are ingested into a PyArrow Table (columnar, zero-copy), then
converted to a Pandas DataFrame with Arrow-backed dtypes for the pipeline.

Standardization Steps
---------------------
1. **Verify**: Target column is present, non-null, and binary (int/bool).
2. **Time normalize**: Parse the raw time column and produce the standard
   pipeline time variable (`new_time_var`, default `AIG_YY_MM`).
3. **Column subset**: Retain only specified columns (plus flow vars).
4. **Sort**: Order rows chronologically by the standard time variable.
5. **Target first**: Move the target column to position 0.

The output is a clean, sorted, Arrow-backed DataFrame ready for
feature type inference and quality screening.
"""

from ...deps.common         import  Path
from ...deps.data           import  (
                                    pa, 
                                    pDF, 
                                    pg_dbapi, 
                                    spark_session
                                    )
from ...deps.types          import  Literal

from ...core.structure      import  (
                                    SID, 
                                    STRUCTURE
                                    )
from ...core.factory        import  (
                                    Dataclass,
                                    DictClass
                                    )
from ...core.foundation     import  AigParam, AigStop
from ...core.interop        import  (
                                    Arrow, 
                                    DataStandard
                                    )
from ...core.accessor       import  Spinner
from ...core.standard       import  TypeDef
from ...core.logging        import  Logger

from ...storage.file        import  ArrowIO
from ...storage.folder      import  Folder


logger = Logger(__name__)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → LOAD → IMPORT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    """Data source type identifiers for the ingestion connector."""
    F = ('file', 'local')
    S = ('spark',)
    D = ('db', 'database')

@Dataclass.base
class ImpParamsBase:
    """
    Pipeline-standardisation parameters.

    Attributes
    ----------       
    time_var : str, default='AIG_YY_MM'
        Standard time column name produced after parsing. All downstream
        stages reference this column as the temporal index.
    columns_subset : list | str, default=None
        Columns to retain after import. If None, all columns are kept.
        Accepts a single column name (str) or a list of names.
    flow_vars : list | str, default=None
        ID-like columns to preserve (excluded from feature screening
        but always carried through the pipeline).  
    """     
    time_var:       str = 'AIG_YY_MM'
    columns_subset: list | str = None
    flow_vars:      list | str = None

@Dataclass.base(update_specs=dict(slots=False))
class ImpParamsCfg:
    """
    Data source configuration.

    Attributes
    ----------
    data_name : str
        Identifier for the data source. For file sources, this is the
        file name (without extension) resolved against the project
        directory. For DB/Spark, this is the table name or SQL query.
    data_source : 'file' | 'spark' | 'db', default='file'
        Source type determining the ingestion connector.
    default_path : str | Path, default=None
        Base directory for file-based imports. If the file is not found
        in the project directory, this path is used as fallback.
    target : str
        Column name of the target variable (must be present, non-null,
        and binary/int).
    yy_mm : str
        Column name of the raw time/date in the source data.
    yy_mm_format : str
        Format string for parsing `yy_mm` (e.g., '%Y-%m-%d').
    """     
    data_name:    str
    data_source:  Literal['file', 'spark', 'db'] = 'file'
    default_path: str|Path = None
    target:       str
    yy_mm:        str
    yy_mm_format: str

@Dataclass.base
class ImpParams(ImpParamsBase, ImpParamsCfg):
    """Importer params input design."""       
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ArrowConnector:
    """
    Multi-source data ingestion via PyArrow.

    Provides a unified interface to load data from files, databases,
    or Spark into a PyArrow Table — the common columnar format that
    all downstream pipeline stages operate on.

    Supported Sources
    -----------------
    - **file/local**: File path (Parquet, CSV, ORC, etc.) read via PyArrow.
    - **db/database**: Connection string or raw SQL query. Uses the
      database's native Arrow fetch (zero-copy where supported).
    - **spark**: Table reference (`catalog.db.table`) or SQL query
      executed via an active PySpark session.

    SQL Detection
    -------------
    A location string is detected as SQL (vs. a table/file name) if it
    contains whitespace/newlines or starts with a SQL keyword
    (SELECT, WITH, SHOW, DESCRIBE, EXPLAIN, FROM).
    """

    @staticmethod
    def _is_sql(location: str) -> bool:
        """
        Heuristic: determine if a location string is a SQL query.

        Returns True if the string contains whitespace (multi-word)
        or starts with a common SQL keyword.
        """        
        return (
            any(w in location for w in (" ", "\n", "\t")) or 
            location.strip().lower().startswith(
                ("select", "with", "show", "describe", "explain", "from")
            )
        )
    
    @classmethod
    def ingest(cls, source_type: str, location: str) -> pa.Table:
        """
        Ingest data from the specified source into a PyArrow Table.

        Parameters
        ----------
        source_type : str
            Source type: 'file', 'local', 'db', 'database', 'spark'.
        location : str
            For file: file path.
            For db: connection string or SQL query.
            For spark: table reference (catalog.db.table) or SQL query.

        Returns
        -------
        pa.Table
            Columnar data in Arrow format.

        Raises
        ------
        ValueError
            If `source_type` is not a recognised source identifier.
        """
        kind = source_type.strip().lower()
        loc  = str(location).strip()
        
        if kind in LBL.F:
            return ArrowIO.load(loc)
            
        if kind in LBL.D:
            # location = "postgresql://user:pass@localhost:5432/mydb"
            db  = pg_dbapi()
            sql = loc if cls._is_sql(loc) else f"SELECT * FROM {loc}"

            with db.connect(loc) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return cur.fetch_arrow_table()            
            
        if kind in LBL.S:
            # location = "catalog.db.table_name" or active PySpark Session
            session = spark_session()
            spark = session.builder.getOrCreate()
            
            spark_df = spark.sql(loc) if cls._is_sql(loc) else spark.table(loc)
            
            return pa.Table.from_batches(spark_df._collect_as_arrow())

        raise ValueError(f"Unsupported data source engine: {source_type}")
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    
@Dataclass.verify
class DataImporter:
    """
    Data Ingestion and Standardization — Pipeline Entry Point.

    Imports data from an external source (file, database, or Spark),
    verifies mandatory constraints, and standardizes the output into
    the canonical format expected by downstream AIDflow stages.

    Pipeline Position
    -----------------
    This is the **first** stage. Its output feeds directly into
    feature type inference and quality screening.

    Parameters
    ----------
    data_name : str
        Data source identifier (file name, table name, or SQL query).
    data_source : 'file' | 'spark' | 'db', default='file'
        Source type.
    default_path : str | Path, default=None
        Fallback directory for file-based imports.
    target : str
        Target column name (must be binary, non-null, present).
    time_var : str
        Raw time column name in the source data.
    time_format : str
        Parsing format for the time column.
    new_time_var : str, default='AIG_YY_MM'
        Standard time column name for the pipeline.
    columns_subset : list | str, default=None
        Columns to retain. None = all.
    flow_vars : list | str, default=None
        ID-like columns to always preserve.

    Examples
    --------
    >>> from aidflow.io import DataImporter
    >>> imp = DataImporter(
    ...     data_name='sales_2024',
    ...     data_source='file',
    ...     target='is_churn',
    ...     time_var='month',
    ...     time_format='%Y-%m',
    ...     flow_vars=['entity_id'],
    ... )
    >>> data = imp.process()
    >>> data.columns[:5]  # ['is_churn', 'AIG_YY_MM', 'entity_id', ...]  # doctest: +SKIP
    """

    def __init__(self, *args, params: dict|ImpParams = None, **kwargs):
        self.params = AigParam.param_ar_kw(ImpParams, params, args, kwargs)
        
    def _get_path(self):
        """User-custom-path or default PROJECT data import directory."""
        path = Folder.standard(self.params.data_name)        
        if path.exists():
            return path
        path, file = Folder.path_file(path)
        if not path: 
            path = self.params.default_path or '.'
        return Folder.standard( f'{path}/{file}' )       
    
    @staticmethod
    def _verify_path(path: Path) -> None:
        if path.exists():
            return
        e = f'AIG SAFETY ► File {path} has not been found.'
        logger.error(e)
        AigStop.custom(e)

    def _import(self) -> pa.Table:        
        """Standard data import for known sources and types."""    
        path = self.params.data_name
        if self.params.data_source in LBL.F:
            path = self._get_path()
            self._verify_path(path)
        return ArrowConnector.ingest(self.params.data_source, path)        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _is_pa(d: pa.Table | pDF) -> bool:
        return isinstance(d, pa.Table)
    
    @staticmethod
    def _desc(name:str, n:str=None, e:str=None) -> str:
        m1 = lambda x, y: f'\n► SPECIFIED {x} VARIABLE BY USER PANEL {y} '
        m2 = lambda x: f'\n► PLEASE CHECK INPUT DATA OR CHANGE {x} VARIABLE DURING SETUP'
        
        n = n or 'WAS NOT FOUND IN THE INPUT DATA'
        return m1(name, n) + (e or '') + m2(name) 

    @classmethod
    def _present(cls, data: pa.Table | pDF, col: str, name: str) -> None:
        t = cls._desc(name)
        if col not in DataStandard.get_cols(data):
            AigStop.custom(t)

    @classmethod
    def _notnull(cls, data: pa.Table | pDF, col: str, name: str) -> None:
        n = 'HAS NULL VALUES'
        e = f'\n► SUPERVISED LEARNING IS EXTRA DEPENDENT ON {name}'  
        t = cls._desc(name, n, e)
        if (
            data[col].null_count 
            if cls._is_pa(data)
            else data[col].isna().sum()
        ):
            AigStop.custom(t)

    @classmethod
    def _dtypes(cls, 
        data: pa.Table | pDF, col: str, name: str, types: tuple[type]
    ) -> None:
        n = f'IS NOT SUPPORTED DTYPE'
        e = f'\n► SUPPORTED DTYPES FOR {name} ARE {types}'
        t = cls._desc(name, n, e)

        dtype = data[col].type if cls._is_pa(data) else data[col].dtype
        if dtype not in (*TypeDef.bool, *TypeDef.integer):
            AigStop.custom(t)

    def _verify(self, data: pa.Table) -> None:
        """
        Verify mandatory constraints on the imported data.

        Checks (in order, stops at first failure):
        1. Target column is **present**.
        2. Target column has **no null values**.
        3. Target column is **binary** (int or bool type).

        Raises
        ------
        AigStop
            With a descriptive message if any check fails.
        """  
        self._present(data, self.params.target, 'TARGET')              
        self._notnull(data, self.params.target, 'TARGET')
        self._dtypes(data,  self.params.target, 'TARGET', (int, bool))
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _standardize(self, data: pa.Table | pDF) -> pDF:
        """Transform the raw Arrow / pd data into the pipeline-standard format."""
        flow_vars = self.params.flow_vars or []
        yy_mm = self.params.yy_mm
        drop = [yy_mm] if yy_mm not in flow_vars and yy_mm in data.columns else []

        data = DataStandard.move_column(data, self.params.target, 0)
        data = DataStandard.aig_time(data, yy_mm, self.params.yy_mm_format, self.params.time_var)  
        data = DataStandard.columns(data, self.params.columns_subset, flow_vars)
        data = DataStandard.sort(data, self.params.time_var)

        data = Arrow.to_pandas(data)
        data = data.drop(drop, axis=1)

        if data.index.is_monotonic_increasing:
            return data
        return data.reset_index(drop=True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def pipeline(self, direct_data:pDF=None) -> pDF:
        """
        Execute the full ingestion pipeline: import → verify → standardize.
        
        Parameters
        ----------
        direct_data: pDF = None
            Sideload data already present in memory. When provided,
            the import step (input_source / input_data) is bypassed
            entirely — the data is used as-is for downstream processing.  

        Returns
        -------
        pDF
            Standardised DataFrame ready for the AIDflow pipeline.

        Raises
        ------
        AigStop
            If any verification check fails (missing target, nulls, bad type).
        ValueError
            If the data source type is unsupported.
        """
        di = direct_data
        if di is None:
            di = self._import()
        self._verify(di)
        return self._standardize(di)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def process(self, direct_data: pDF = None, verbose: int|bool = True) -> pDF:
        """
        Execute the ingestion pipeline with progress display.
        
        Steps applied in order:
        0. Ingest data by Arrow backend.
        1. Move target column to position 0.
        2. Parse and normalise the time variable (`time_var` → `new_time_var`).
        3. Retain only specified columns (plus flow vars and time).
        4. Sort rows chronologically by the standard time variable.
        5. Convert to Pandas with Arrow-backed dtypes.

        Parameters
        ----------
        direct_data: pDF = None
            Sideload data already present in memory. When provided,
            the import step (input_source / input_data) is bypassed
            entirely — the data is used as-is for downstream processing.        
        verbose : int | bool, default=True
            Enable animated progress display during import.

        Returns
        -------
        pDF
            Standardised DataFrame ready for the AIDflow pipeline.
        """
        title = f'Importing {self._get_path()}'
        with Spinner(title, show_timer=True).progress(verbose):
            return self.pipeline(direct_data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬