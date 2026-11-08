"""
Unified Persistence Layer for standardized data and model serialization.

This module implements a high-level I/O abstraction that decouples the library's 
internal data structures from their physical storage formats. By utilizing an 
inference-based approach to file types, it provides a single, consistent API 
for saving and loading diverse assets—ranging from CSVs and Pickles to compressed 
mgzip archives—while ensuring that all paths are standardized through the 
global Folder management system.
"""

from ..deps.common import (
    wraps, 
    io
)
from ..deps.data import (
    pd, 
    pDF, 
    pa, 
    pa_csv, 
    pa_ds, 
    pa_pq, 
    pa_ft,
    pa_pc
)
from ..deps.common import (
    pickle, 
    mgzip, 
    csv, 
    Path
)
from ..deps.types import (
    Any, 
    Literal
)
from ..core.standard import (
    TypeDef
)
from ..core.logging import (
    Logger
)
from ..display.rgb import (
    RgbANSI
)
from .thread import (
    AsyncIO
)
from .folder import (
    Folder
)


logger = Logger(__name__)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FILE OPS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ArrowIO:
    """
    High-performance layer utilizing Apache Arrow for large datasets.
    Standardized interface for high-level object persistence.
    
    Provides a simplified 'save/load' API that abstracts away the underlying 
    compression logic.
    """

    EXT = dict(
        pq  = (".parquet", ".pq"),
        csv = (".csv", ".tsv"),
        zip = (".gzip", ".gz"),
        ipc = (".arrow", ".feather", ".ipc"),
    )

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    @staticmethod
    def _sanitize_col(field: pa.Field, col: pa.ChunkedArray) -> pa.ChunkedArray:
        """Sanitizes an individual column in C++ if NaNs are present."""
        if pa.types.is_floating(field.type) and pa_pc.any(pa_pc.is_nan(col)).as_py():
            return pa_pc.if_else(pa_pc.is_nan(col), None, col)
        return col

    @classmethod
    def _postprocess(cls, table: pa.Table) -> pa.Table:
        """Batch sanitizes IEEE NaNs and reconstructs the Table."""                
        if not any(pa.types.is_floating(field.type) for field in table.schema):
            return table

        new_columns = [
            cls._sanitize_col(field, table.column(i))
            for i, field in enumerate(table.schema)
        ]

        return pa.Table.from_arrays(new_columns, schema=table.schema)

    @classmethod
    @wraps(pa_csv.read_csv)
    def read_csv(cls, input_file: str | Path, **kwargs) -> pa.Table:
        """Read a Table from a stream of CSV data."""
        
        ro = kwargs.pop("read_options", None)
        read_options = ro or pa_csv.ReadOptions(
            use_threads = True, 
            block_size  = 1048576,  # 1MB chunks for thread distribution
            skip_rows   = 0
        )
                
        co = kwargs.pop("convert_options", None)
        convert_options = co or pa_csv.ConvertOptions(
            null_values = TypeDef.missing,
            strings_can_be_null = True,            
        )
        
        d = pa_csv.read_csv(
            input_file,
            read_options    = read_options,
            convert_options = convert_options,
            **kwargs
        )
        return cls._postprocess(d)
    
    @classmethod
    def read_gzip(cls, path: str | Path, **kwargs) -> pa.Table:
        """
        Read a Table from a stream of GZIP data.

        Parameters
        ----------
        path : string or path
            The location of GZIP data. Recognized file extension is ".gzip", ".gz".
        """
        d = MgzipIO.load(path)
        p = lambda d: cls._postprocess(d)

        if isinstance(d, pa.Table):
            return p(d)
        
        if isinstance(d, pDF):
            return p( pa.Table.from_pandas(d, preserve_index = True) )

        if isinstance(d, (bytes, bytearray, pa.Buffer)):
            if d.startswith(b"PAR1"):
                return p(pa_pq.read_table(pa.BufferReader(d), **kwargs))
            # Fallback binary to CSV reader via zero-copy BufferReader
            return cls.read_csv(pa.BufferReader(d), **kwargs)

        if isinstance(d, io.IOBase):
            return cls.read_csv(d)        
                
        if isinstance(d, str):
            if "\n" in d or "\r" in d or "," in d:
                return cls.read_csv(pa.BufferReader(d.encode("utf-8")), **kwargs)
            return cls.read_csv(d, **kwargs)

        if isinstance(d, dict):
            return p( pa.Table.from_pydict(d) )
        
        if isinstance(d, list):
            return p( pa.Table.from_pylist(d) )
        
        raise TypeError(f"Unsupported payload type returned by MgzipIO: {type(d).__name__}")
    
    @classmethod
    @wraps(pa_pq.read_table)
    def read_parquet(cls, source: str | Path, **kwargs) -> pa.Table:
        """Read a Table from Parquet format."""
        d = pa_pq.read_table(source, **kwargs)
        return cls._postprocess(d)

    @staticmethod
    @wraps(pa_ft.read_table)
    def read_feather(source: str | Path | pa.NativeFile, **kwargs) -> pa.Table:
        """Read a pyarrow.Table from Feather / IPC format with memory mapping."""
        # mmap=True is default in read_table for path strings (zero-copy memory mapped)
        return pa_ft.read_table(source, **kwargs)

    @classmethod
    @wraps(pa_ds.Dataset.to_table)
    def read_dataset(cls, source: str | Path, **kwargs) -> pa.Table:
        """Read the dataset to an Arrow table."""
        d = pa_ds.dataset(source, **kwargs).to_table()
        return cls._postprocess(d)
    
    @classmethod
    def load(cls,
        path: str | Path, 
        format: Literal['parquet', 'csv', 'gzip', 'feather', 'dataset'] = None, 
        **kwargs
    ) -> pa.Table:
        """
        Unified Arrow entrypoint factory supporting files, multi-file datasets, and SQL.        

        Parameters
        ----------
        path : str | Path
            Path to the source file.

        format : Literal['parquet', 'csv', 'gzip', 'feather', 'dataset'] = None
            Optional explicit format of the file to load. 
            When `format = None` → format is infered by the file extension .
        
        Returns
        -------
        pa.Table : Loaded arrow Table from the source.            
        """
        # 1. SQL / ADBC Driver Cursor
        if hasattr(path, "fetch_arrow_table"):
            return path.fetch_arrow_table()

        # 2. String Path Auto-Inference
        if isinstance(path, (str, Path)):
            path = str(path)

            if format == "parquet" or path.endswith(cls.EXT['pq']):
                return cls.read_parquet(path, **kwargs)
            
            if format == "csv" or path.endswith(cls.EXT['csv']):
                return cls.read_csv(path, **kwargs)
            
            if format == "gzip" or path.endswith(cls.EXT['zip']):
                return cls.read_gzip(path, **kwargs)
            
            if format in ("feather", "ipc") or path.endswith(cls.EXT['ipc']):
                return cls.read_feather(path, **kwargs)
            
            if format == "dataset" or "partitioning" in kwargs:
                return cls.read_dataset(path, **kwargs)

        # 3. Direct Dataset or Table Pass-through
        if isinstance(path, pa_ds.Dataset):
            return path.to_table()
            
        raise ValueError(f"Unsupported Arrow source or format: {type(path)} / {format}")

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    @wraps(pa_csv.write_csv)
    @Folder.exists
    def to_csv(data: pa.Table, output_file: str | Path, **kwargs) -> None:
        """Write an Arrow Table to CSV format."""
        pa_csv.write_csv(data, output_file, **kwargs)

    @staticmethod
    def to_gzip(data: pa.Table, path: str | Path, **kwargs) -> None:
        """Write an Arrow Table to GZIP format via Pandas conversion.

        Parameters
        ----------
        data : pa.Table
            The input Arrow Table.
        path : str | path
            Destination filepath ending in '.mgzip'.
        """
        d = data.to_pandas()
        MgzipIO.save(d, path, **kwargs)

    @staticmethod
    @wraps(pa_pq.write_table)
    @Folder.exists
    def to_parquet(data: pa.Table, where: str | Path, **kwargs) -> None:
        """Write an Arrow Table to Parquet format."""
        c = kwargs.pop("compression", None) or "zstd"
        l = kwargs.pop("compression_level", None) or 5
        pa_pq.write_table(data, where, compression=c, compression_level=l, **kwargs)

    @staticmethod
    @wraps(pa_ft.write_feather)
    @Folder.exists
    def to_feather(data: pa.Table, dest: str | Path, **kwargs) -> None:
        """Write an Arrow Table to Feather / IPC format."""
        pa_ft.write_feather(data, dest, **kwargs)

    @staticmethod
    @wraps(pa_ds.write_dataset)
    @Folder.exists
    def to_dataset(data: pa.Table, base_dir: str | Path, **kwargs) -> None:
        """Write an Arrow Table to a partitioned multi-file dataset directory."""
        pa_ds.write_dataset(data, base_dir, **kwargs)

    @classmethod
    def save(cls, 
        data: pa.Table, 
        path: str | Path, 
        format: Literal['parquet', 'csv', 'gzip', 'feather', 'dataset'] = None, 
        **kwargs
    ) -> None:
        """
        Unified Arrow save factory supporting individual file formats and
        multi-file partitioned datasets.

        Parameters
        ----------
        data : pa.Table
            Arrow Table to save.

        path : str | Path
            Destination path where to save the data.

        format : Literal['parquet', 'csv', 'gzip', 'feather', 'dataset'] = None
            Optional explicit format of the file to save. 
            When `format = None` → format is infered by the file extension .
        """
        if not isinstance(data, pa.Table):
            raise TypeError(
                f"Expected pyarrow.Table for saving, got {type(data).__name__}"
            )
        path = str(path)

        if format == "parquet" or path.endswith(cls.EXT['pq']):
            return cls.to_parquet(data, path, **kwargs)

        if format == "csv" or path.endswith(cls.EXT['csv']):
            return cls.to_csv(data, path, **kwargs)

        if format == "gzip" or path.endswith(cls.EXT['zip']):
            return cls.to_gzip(data, path, **kwargs)

        if format in ("feather", "ipc") or path.endswith(cls.EXT['ipc']):
            return cls.to_feather(data, path, **kwargs)

        if (
            format == "dataset"
            or "partitioning" in kwargs
            or "partition_cols" in kwargs
        ):
            return cls.to_dataset(data, path, **kwargs)

        raise ValueError(
            f"Unsupported Arrow destination or format: {path} / {format}"
        )    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

    
class BaseIO:
    """
    Standardized interface for high-level object persistence.
    
    Provides a simplified 'save/load' API that abstracts away the underlying 
    compression logic, utilizing either synchronous or asynchronous mgzip 
    backends to ensure consistent serialization of models and structured files.
    """        
    
    @classmethod
    def _exception(cls, o:str, p:str, e:Exception, verbose:bool): 
        m = RgbANSI.bold_red(f'AIG ► {o.upper()} FAILED') + ' ► ' + str(p)
        logger.error(m)
        if verbose: 
            print(m)
        raise e

    @classmethod
    def save(cls, data, path:str|Path, async_io:bool=True, verbose:bool=True):
        """
        Persist a Python object to disk using the standard AIDflow compression.
        
        Parameters
        ----------
        data : Any
            The object to be serialized.
        path : str | Path
            Destination path; will be normalized via Folder.standard().
        async_io : bool, default=True
            If True, performs save in a background thread to prevent UI freezing.
        verbose : bool, default=True
            If True, prints an error message to the console upon failure.
        """
        ops  = 'save'
        path = Folder.standard(path)
        try:
            if async_io:
                MgzipIOAsync.io().ops(ops, path, data)
            else:
                MgzipIO.save(data, path) 
        except Exception as e: 
            cls._exception(ops, path, e, verbose)

    @classmethod
    def load(cls, path:str|Path, async_io:bool=False, verbose:bool=True):
        """
        Load a serialized object from the standard AIDflow storage format.

        Parameters
        ----------
        path : str | Path
            Path to the asset; will be normalized via Folder.standard().
        async_io : bool, default=False
            If True, loads the object asynchronously.
        verbose : bool, default=True
            If True, prints an error message upon failure.

        Returns
        -------
        Any: The deserialized Python object.
        """
        ops  = 'load'
        path =  Folder.standard(path)
        try: 
            if async_io:
                res  = MgzipIOAsync.io().ops(ops, path)
                return MgzipIOAsync.callback(res)                
            return MgzipIO.load(path)
        except Exception as e: 
            cls._exception(ops, path, e, verbose)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class MgzipIO:
    """Low-level implementation of compressed pickle serialization using mgzip."""

    EXT = 'pkl.gz'

    @classmethod
    def suffix(cls, path) -> str:
        return FileUtils.ensure_extension(path, cls.EXT)

    @classmethod
    @Folder.exists
    def save(cls, data: pDF, path: str | Path, **kwargs) -> None:
        """Compresses and dumps object to disk."""
        path = cls.suffix(path)
        with mgzip.open(path, 'wb') as f:
            pickle.dump(data, f, **kwargs)
    
    @classmethod
    def load(cls, path: str | Path, **kwargs) -> Any:
        """Decompresses and loads object from disk."""
        path = cls.suffix(path)        
        return cls._load(path, kwargs)

    @classmethod
    def _load(cls, path: str | Path, kwargs: dict) -> Any:
        """
        Decompresses GZIP/MGZIP files and auto-detects the underlying engine
        (pickle vs raw bytes / text / structured formats).
        """
        with mgzip.open(path, "rb") as f:
            decompressed_data = f.read()

        if decompressed_data is None:
            return None

        # 1. Pickle Deserialization
        if (
            decompressed_data.startswith(b"\x80") or 
            decompressed_data.startswith(b"\x81")
        ):
            try:
                return pickle.loads(decompressed_data, **kwargs)
            except Exception:
                pass

        # 2. Check for Parquet / Arrow magic headers
        # Returns raw bytes buffer to pass directly to C++ readers
        if (
            decompressed_data.startswith(b"PAR1") or 
            decompressed_data.startswith(b"ARROW1")
        ):
            return decompressed_data

        # 3. Fallback: UTF-8 string or raw bytes for CSV / JSON parsing
        try:
            return decompressed_data.decode("utf-8")
        except UnicodeDecodeError:
            return decompressed_data

class MgzipIOAsync:
    """Asynchronous wrapper for MgzipIO operations to prevent thread blocking."""

    _async: AsyncIO | None = None
    _io = {'save': MgzipIO.save, 'load':MgzipIO.load}

    @classmethod
    def io(cls) -> AsyncIO:
        """Returns an AsyncIO instance mapped to mgzip save/load functions."""
        if cls._async is None:
            cls._async = AsyncIO(cls._io)
        return cls._async

    @staticmethod
    def callback(result:tuple) -> Any:
        """Extracts the result from an asynchronous future/tuple."""
        if result[1] is not None:
            raise result[1]
        return result[0]
    
    @classmethod
    def shutdown(cls) -> None:
        cls.io().stop(wait=True)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class FileIO:
    """Atomic utility methods for low-level file format manipulation."""

    @staticmethod
    def save_pickle(data:pDF, path:str|Path, **kwargs):
        """Standard python pickle save wrapper."""
        with open(path, 'wb') as f: 
            pickle.dump(data, f, protocol=-1, **kwargs)   

    @staticmethod
    def load_pickle(path:str|Path, **kwargs):
        """Standard python pickle load wrapper."""
        with open(path, 'rb') as f:
            return pickle.load(f, **kwargs)
    
    @staticmethod
    def save_dict_to_csv(
        data: dict, 
        path: str|Path, 
        fields: tuple = ("Attribute", "Value"),
        err: str = 'AIG WARNING ► Saving csv failed!'
    ) -> None:
        """Specialized method to transform a dictionary into 2-column CSV (Attr, Val)."""
        try:
            if not isinstance(data, dict):
                raise ValueError(
                    '► Save to csv supports only dictionary data input ► '
                    f'provided data is {type(data)}'
                )
            
            path = FileUtils.ensure_extension(path, 'csv')
            
            with open(path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(fields)
                for attr, value in data.items():
                    writer.writerow([attr, value])        
        except:
            logger.error(f'{err} ► {path}')

    @staticmethod
    def save_csv(df:pDF, path:str|Path, **kwargs) -> str:
        """Simplified pandas csv save interface. Returns save path string."""
        path = FileUtils.ensure_extension(path, 'csv')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, **kwargs)
        return path

    @staticmethod
    def load_csv(path:str|Path, **kwargs) -> pd.DataFrame:
        """Simplified pandas csv load interface."""
        return pd.read_csv(path, **kwargs)
    
    @classmethod        
    def load_csv_as_dict(cls, 
        path: str|Path, fields: tuple[str] = ("Attribute", "Value"), **kwargs
    ) -> dict:
        """Simplified pandas csv load interface → return as dictionary by fields."""
        return (
            cls.load_csv(path, **kwargs)
            .set_index(fields[0])[fields[1]]
            .to_dict()
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class FileUtils:
    """Path manipulation and filename sanitization utilities."""
    
    @staticmethod
    def file_extension(file:str|Path) -> str:
        """Extracts the extension from a path string."""
        return Path(file).suffix.lstrip('.')
    
    @classmethod    
    def infer_type(cls, file_name:str|Path, base:list=None) -> str:
        """Determines the data format based on extension or a provided fallback list."""
        file_type = cls.file_extension(file_name)
        if ( base is not None ) and ( file_type not in base[:-1] ): 
            file_type = base[-1]
        return file_type
   
    @classmethod
    def ensure_extension(cls, file:str|Path, ftype:str) -> str:
        """Guarantees that a path has the required extension for its type."""
        file_type = cls.infer_type(file)
        if file_type != ftype.split('.')[-1]: 
            file = str(file) + '.' + ftype
        return str(file)
    
    @staticmethod
    def local_path(path, *args):
        """Joins paths while ensuring they are normalized via Folder.standard()."""
        path = Folder.standard(path)
        for a in args: 
            path = path / str(a)
        return path
    
    @classmethod
    def path_extension(cls, path, *args):
        """Append args to filename with underscores."""
        path = str(path)
        path = '_'.join([path]+list(args))
        return cls.local_path(path)

    @staticmethod
    def obs_to_K(name, reverse=False):
        """Converts observation integers (e.g., '000') to K-notation ('K') and vice versa."""
        o,k = '000','K'
        OK  = (str, o, k)     # from obs int to 'K'
        KO  = (int, k, o)     # from 'K' to obs int

        if not reverse:
              t, f, a = OK    # to type, find, add        
        else: t, f, a = KO   
        
        l = str(name).rfind(f)    
        if l != -1:
            n = str(name)[:l] + a
            try: name = t(n)
            except: pass
        return name    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬