"""
System resource orchestration and memory-aware data partitioning engine.

This module implements the capacity planning logic for AIG, providing 
predictive memory analysis to prevent Out-of-Memory (OOM) failures during 
large-scale ML processing. By analyzing DataFrame schemas and system 
availability in real-time, it enables the framework to automatically 
determine optimal batch sizes for data fitting and transformation.

Key resource management capabilities:
1. Predictive Memory Profiling: Utilizes a granular dtype-to-byte mapping 
   (`DTYPE_MEMORY`) to calculate the exact memory footprint of datasets 
   before they are loaded or processed.

2. Dynamic Batch Partitioning: Implements `fit_df_rows` and `fit_data_id_batch` 
   to compute safe processing windows, automatically slicing data into batches 
   that fit within available system RAM while maintaining a configurable safety buffer.

3. Resource-Aware Orchestration: Integrates with `psutil` to monitor local server 
   memory (RSS/USS/PSS), ensuring that the pipeline adapts its throughput based on 
   the actual hardware constraints of the deployment environment.

4. Adaptive ID-Based Slicing: Supports complex panel data structures by calculating 
   weighted average row lengths per unique identifier, allowing for memory-safe 
   batching across grouped datasets.
"""

from ..deps.common  import os, sys, gc, psutil, tracemalloc
from ..deps.data    import pa, np, pDF, pSe
from ..deps.struct  import dataclass, field
from ..deps.display import wg
from ..deps.types   import Optional


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# MEMORY UTILS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Memory:
    """Utility for memory availability check and data adjustment."""
    
    DTYPE_MEMORY = {
        'int8':    1,
        'int16':   2,
        'int32':   4,
        'int64':   8,
        'float32': 4,
        'float64': 8,
        'bool':    1,
        'object': 50,  # Exp. for average string length
        '<m8':     8, 
        '<m8[ns]': 8,
        'datetime64[ns]':  8,
        'timedelta64[ns]': 8,
    }
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_local_system(verbose=False):    
        """Returns local available memory in bytes."""        
        memory_in_bytes = psutil.virtual_memory().available
        if verbose: 
            print(f"Total Local Server Memory: {memory_in_bytes / (1024 * 1024):.2f} MB")
        return memory_in_bytes
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _flat_data_input_type(df:pDF|dict|list) -> list[pDF]:
        if isinstance(df, dict):      
            return list(df.values())
        if not isinstance(df, list): 
            return [df]
        return df

    @staticmethod
    def _revert_data_input_type(T:pDF|dict|list, result:dict) -> dict|list:
        if isinstance(T,dict): 
            return {k:result[i] for i,k in enumerate(T)}
        if not isinstance(T,list):
            return result[0]    
        return result
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def fit_df_rows(cls, 
        df:pDF|dict|list, 
        *, 
        mem_buffer:float = 0.3, 
        total_worker_memory_gb:int = None, 
        verbose:bool = False
    ) -> int:
        """        
        Screen all columns with respect to their type and expected memory needs 
        across provided data (across all tables → df as dict). 

        Returned value represents max rows which will fit into available memory 
        for all columns present in the input.

        NOTE that df as dict[str, pDF] → returned value acts as if data tables
        across dict is one dataframe → useful when batches of tables in the dict
        go to batch processing.
        Thus, each of the tables can be a returned rows subset for a batch run.
        """        
        df_ = cls._flat_data_input_type(df)

        available_memory = cls.get_local_system(verbose)
        if total_worker_memory_gb is not None:
            available_memory = min(available_memory, total_worker_memory_gb*(1024**3))

        memory_per_row = 0
        for d in df_:                                
            for dtype in d.dtypes:
                dtype_str = str(dtype).lower()                                        
                memory_per_col = cls.DTYPE_MEMORY.get(dtype_str, 50)                                        
                memory_per_row += memory_per_col
        
        if not memory_per_row:
            return 0
        
        buffer = max( 0.15, min(mem_buffer, 0.99) )          # Safety buffer of x% for data processing, operations, overhead
        available_memory = available_memory * (1-buffer)     # Expected available purely for pDF
                        
        possible_rows = available_memory / memory_per_row    # Compute the number of rows that can fit within the worker memory
            
        if verbose: 
            print(f'Total expected rows which can be safely processed is {int(possible_rows):,}'
                  ' | given worker memory {total_worker_memory_mb//1024:,}GB')
        return int(possible_rows)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _ids_len(ID:pSe|list=None, Group:str=None) -> tuple[int|None, str|None]:
        if ID is None:
            return None, None        
        if Group is None:
            if not isinstance(ID, pSe):
                raise TypeError('When Group is None provided, ID vector must be pd.Series type to infer name.')
            Group = ID.name
        return len(np.unique(ID)), Group
    
    @classmethod
    def _data_len(cls, T:pDF|dict|list, group:str=None) -> int:
        """
        Weighted expected data len across tables.
        For ID (panel) data returned value is expected rows per unique ID.
        For tabular data returned value is expected rows in data tables.
        """
        D = cls._flat_data_input_type(T)
        avg_len = {'l':[],'w':[]}
        for d in D:
            l = len(d)               # total rows per d as length
            if group is None:        
                w = len(d.columns)   # total cols per d as weight
            else:
                if group in d.columns:
                    x = d[group]
                else:
                    assert d.index.name == group, f"Index name '{d.index.name}' does not match expected group name '{group}'"
                    x = d.index
                w = len(x.unique())  # total ids per d as weight
                l = l/w              # avg. rows per id
            avg_len['l'].append(l)
            avg_len['w'].append(w)
        return int( np.average(avg_len['l'], weights=avg_len['w']) )

    @staticmethod
    def _per_partes(n_parts:float, batch_size:float, total_len:int) -> list:
        """Create data parts as batches of indicies."""
        if n_parts <= 1.05:                  # 5% buffer to allow go at once
            return [[0, total_len]]
        
        batch_size = int(batch_size)
        rows_ids = [[0, batch_size]]         # go in parts init batch
        R = range( int(n_parts+0.5) )        # number of batch runs
        for i in R:
            b0 = rows_ids[i][1]              # get previous val. to ensure that all ids/rows are processed
            b1 = b0 + batch_size             # batch size to process
            if i == R[-1]: b1 = total_len    # last position is given by size of ID/row
            rows_ids.append([b0, b1])        # batch of given part
        return rows_ids
    
    @classmethod
    def _row_batches(cls, T:pDF|dict|list, data_exp_len:int, mem_max_rows:int) -> dict|list:
        row_parts = data_exp_len / mem_max_rows     # parts needed to process all rows in batches
        row_batches = []
        for d in cls._flat_data_input_type(T):            
            total_len = len(d)
            batch_size = total_len / row_parts
            row_batches.append( cls._per_partes(row_parts, batch_size, total_len) )        
        return cls._revert_data_input_type(T, row_batches)
    
    @classmethod
    def _id_batches(cls, ids_len:int, data_exp_len_per_id:int, mem_max_rows:int) -> list:
        batch_id  = mem_max_rows / data_exp_len_per_id  # batch size of ids given fitting in mem
        id_parts  = ids_len / batch_id                # parts needed to process all unique IDs 
        ids_batches = cls._per_partes(id_parts, batch_id, ids_len)
        return ids_batches
    
    @classmethod
    def fit_data_id_batch(cls, 
        data:       pDF|dict|list, 
        id_vector:  pSe|list|tuple = None, 
        group:      str = None, 
        mem_buffer: float = 0.5, 
        verbose:    bool = False
    ) -> dict[str, list] | list[list]:
        """
        Measures Data memory needs vs. available system memory and creates batches of data 
        slices to ensure smooth processing. If ID is provided the return is batches of ID 
        slices expecting user to follow up on slicing ID vector by batches and subsetting 
        data based on batch id.

        Args:
            data (pDF | dict | list) : pandas or pyspark.pandas data either directly or 
                in container dict |list.

            id_vector (pDF, optional) :  Defaults to None.
                ID dataframe vector - if provided then return is batches of ID slices.

            group (str, optional) : Defaults to None. 
                Group for data and ID dataframes - if not provided, yet ID provided - infer based on ID. 

            mem_buffer (float, optional) : Defaults to 0.5.
                Part of memory left free for overhead, processing etc.
            
            verbose (bool, optional) : Defaults to False.
                print information during memory inference.

        Returns:
            (dict | list): Either indicies of batches slices of each data provided | 
                or batches of ID indicies. 
        """
        ids_len, Group_ = cls._ids_len(id_vector, group)        
        data_len = cls._data_len(data, Group_)
        max_rows = cls.fit_df_rows(data, mem_buffer=mem_buffer, verbose=verbose)        
        
        if Group_ is not None:
            return cls._id_batches(ids_len, data_len, max_rows)
        
        return cls._row_batches(data, data_len, max_rows)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class MemoryCheck:

    @staticmethod
    def rss() -> None:
        """Returns the MEM profile by USS | RSS | PSS."""
        gc.collect()
        
        p = psutil.Process(os.getpid())
        m = p.memory_full_info()
        l = lambda x: round(x/1024**3,1)

        print("USS:", l(m.uss), "GB")
        print("RSS:", l(m.rss), "GB")
        print("PSS:", l(m.pss), "GB")

    @staticmethod
    def ctx_offload(lbl:str='Worker Offload'):
        """
        Context manager to trace, clean up, and diagnose worker offloads.

        Usage
        -----
        with MemoryCheck.ctx_offload("Worker Offload") as t:
            # 1. Save PyArrow dataset/table to disk and null it
            fx()
            
        print(t.report)
        """    
        return OffloadTracer(lbl)

    @staticmethod
    def ctx_tracer():
        """
        Lightweight memory diagnostics for pipeline workers.

        Usage
        -----
        with MemoryCheck.ctx_tracer() as t:
            fx()    
        
        print(t.report)
        """        
        return MemDiag().tracer()
    
    @staticmethod
    def trace_instance(instance_name:str) -> None:
        ui_instances = [o for o in gc.get_objects() if type(o).__name__ == instance_name]
        if len(ui_instances) <= 1:
            print(f"No old {instance_name} instances found. Re-run your notebook cell first.")
            return

        old_ui = ui_instances[0]
        cells = [r for r in gc.get_referrers(old_ui) if type(r).__name__ == 'cell']
        print(f"Tracing {len(cells)} closure cell(s) for old {instance_name} (id={id(old_ui)}):\n")

        for i, cell in enumerate(cells):

            # Step 1: Unwrap cell -> __closure__ tuple
            closure_tuples = [r for r in gc.get_referrers(cell) if isinstance(r, tuple)]            
            for ct in closure_tuples:

                # Step 2: Unwrap tuple -> function
                funcs = [r for r in gc.get_referrers(ct) if callable(r)]
                for fn in funcs:
                    code = getattr(fn, '__code__', None)
                    loc = f"{code.co_filename}:{code.co_firstlineno}" if code else "unknown"
                    qname = getattr(fn, '__qualname__', str(fn))
                    print(f"Cell [{i}] captured by -> Function: {qname} ({loc})")

                    # Step 3: Find holding widget or container
                    for holder in gc.get_referrers(fn):
                        h_type = type(holder).__name__
                        if h_type in ('list', 'dict', 'tuple'):
                            for outer in gc.get_referrers(holder):
                                o_type = type(outer).__name__
                                if o_type not in ('frame', 'list', 'dict', 'tuple'):
                                    print(f"  └── Held by: {o_type}")
                        elif h_type not in ('frame', 'code'):
                            print(f"  └── Held directly by: {h_type}")        

    @staticmethod
    def get_instance(instance_id:int):
        for obj in gc.get_objects():
            if id(obj) == instance_id:
                return obj

    @staticmethod
    def inspect_arrow_ref(max_objects: int = 5):
        """
        Finds all active PyArrow objects, sorts them by memory size (largest first),
        and prints their reference holders.
        """
        gc.collect()

        def get_arrow_size(obj) -> int:
            """Helper to get accurate byte size across Tables, Arrays, Batches, and Buffers."""
            if hasattr(obj, "nbytes"):
                return obj.nbytes
            if hasattr(obj, "size"):
                return obj.size
            return 0

        # 1. Collect all living PyArrow objects
        arrow_objs = [
            obj
            for obj in gc.get_objects()
            if isinstance(
                obj, (pa.Table, pa.ChunkedArray, pa.RecordBatch, pa.Buffer)
            )
        ]

        total_bytes = sum(get_arrow_size(o) for o in arrow_objs)
        print(f"🔍 Found {len(arrow_objs)} active PyArrow objects in Python GC.")
        print(
            f"📊 Total tracked Arrow memory: {total_bytes / 1024 / 1024:.2f} MB ({total_bytes / 1e9:.2f} GB)\n"
        )

        # 2. Sort by memory size descending (largest first)
        arrow_objs.sort(key=get_arrow_size, reverse=True)

        # 3. Inspect referrers for top largest objects
        for i, target in enumerate(arrow_objs[:max_objects]):
            sz_bytes = get_arrow_size(target)
            sz_mb = sz_bytes / 1024 / 1024

            print(
                f"━" * 60
                + f"\n Target [{i+1}]: {type(target).__name__} | Size: {sz_mb:.2f} MB ({sz_bytes:,} bytes)"
            )
            if hasattr(target, "shape"):
                print(f"  Shape: {target.shape}")

            # Get everything pointing to this target
            referrers = gc.get_referrers(target)
            print(f"  Referenced by {len(referrers)} container(s):")

            for ref in referrers:
                # Filter out local iterator and temporary reference lists
                if ref is arrow_objs or ref is referrers:
                    continue

                if isinstance(ref, dict):
                    # Find who owns this dictionary namespace
                    owners = [
                        o
                        for o in gc.get_referrers(ref)
                        if not isinstance(o, (list, dict))
                    ]
                    owner_name = (
                        f"instance of {type(owners[0]).__name__}"
                        if owners
                        else "unnamed dict/namespace"
                    )
                    keys_preview = [str(k) for k in list(ref.keys())[:5]]
                    print(f"    ├─ 🗂️  [{owner_name}] Keys: {keys_preview}")

                elif isinstance(ref, (list, tuple, set)):
                    print(
                        f"    ├─ 📦 [{type(ref).__name__}] containing {len(ref)} items"
                    )

                elif isinstance(ref, pDF):
                    print(
                        f"    ├─ 🐼 [pDF] shape={ref.shape}, cols={list(ref.columns[:3])}"
                    )

                else:
                    print(f"    ├─ ⚙️  [{type(ref).__module__}.{type(ref).__name__}]")
                    
    @staticmethod
    def inspect_widget_counts():
        """Finds all active widgets objects and provide stats by type."""        
        gc.collect()

        # 1. Registered widgets connected to the frontend Comm channel
        registry_widgets = list(wg.Widget.widgets.values())

        # 2. Total Widget instances physically surviving in heap RAM
        heap_widgets = [o for o in gc.get_objects() if isinstance(o, wg.Widget)]

        print(f"Kernel Registry Count (Widget.widgets) : {len(registry_widgets)}")
        print(f"RAM Heap Instance Count (gc.get_objects): {len(heap_widgets)}")

        if registry_widgets:
            print("\nRegistered Widget Breakdown:")
            counts = {}
            for w in registry_widgets:
                name = type(w).__name__
                counts[name] = counts.get(name, 0) + 1

            for w_type, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  ├── {w_type}: {count}")  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@dataclass
class MemSnapshot:
    """A single point-in-time memory diagnostic."""

    label: str
    pid: int
    rss_gb: float
    pss_gb: float
    private_clean_gb: float
    private_dirty_gb: float
    python_object_count: int
    numpy_array_count: int
    numpy_total_gb: float
    pandas_df_count: int
    pandas_total_gb: float
    top_objects: list[dict] = field(default_factory=list)
    top_arrays: list[dict] = field(default_factory=list)
    tracemalloc_top: list[dict] = field(default_factory=list)

class MemDiag:
    """Lightweight memory diagnostics for pipeline workers."""

    MIN_ARRAY_BYTES = 50 * 1024 * 1024  # 50 MB
    MIN_OBJECT_BYTES = 10 * 1024 * 1024  # 10 MB

    def take(
        self, label: str, *, tracemalloc_top: int = 10
    ) -> tuple[MemSnapshot, set[int]]:
        """Capture a memory snapshot. Returns (MemSnapshot, set_of_gc_object_ids)."""
        pid = os.getpid()
        proc = self._read_smaps_rollup(pid)

        gc.collect()
        all_objects = gc.get_objects()
        gc_ids = {id(o) for o in all_objects}

        py_obj_count = len(all_objects)
        numpy_arrays: list[dict] = []
        pandas_dfs: list[dict] = []

        np_count, np_total = 0, 0
        pd_count, pd_total = 0, 0
        top_objects: list[dict] = []

        # Single-pass scan over all GC-tracked objects
        for obj in all_objects:
            # Check Pandas DataFrames
            if isinstance(obj, pDF):
                pd_count += 1
                try:
                    sz = int(obj.memory_usage(deep=True).sum())
                except Exception:
                    sz = 0
                pd_total += sz
                if sz >= self.MIN_ARRAY_BYTES:
                    pandas_dfs.append(
                        {
                            "type": "DataFrame",
                            "size_mb": round(sz / 1024 / 1024, 1),
                            "shape": list(obj.shape),
                            "cols": len(obj.columns),
                        }
                    )

            # Check NumPy arrays (only catches GC-tracked ndarrays)
            elif isinstance(obj, np.ndarray):
                np_count += 1
                sz = obj.nbytes
                np_total += sz
                if sz >= self.MIN_ARRAY_BYTES:
                    numpy_arrays.append(
                        {
                            "type": "ndarray",
                            "size_mb": round(sz / 1024 / 1024, 1),
                            "shape": list(obj.shape),
                            "dtype": str(obj.dtype),
                        }
                    )

            # Generic large container check
            else:
                try:
                    sz = sys.getsizeof(obj)
                    if sz >= self.MIN_OBJECT_BYTES:
                        top_objects.append(
                            {
                                "size_mb": round(sz / 1024 / 1024, 1),
                                "type": type(obj).__name__,
                            }
                        )
                except (TypeError, SystemError):
                    continue

        # Sort descriptors descending
        top_arrays = sorted(
            numpy_arrays + pandas_dfs, key=lambda x: x["size_mb"], reverse=True
        )[:10]
        top_objects = sorted(
            top_objects, key=lambda x: x["size_mb"], reverse=True
        )[:10]

        tl_top: list[dict] = []
        if tracemalloc.is_tracing():
            snap = tracemalloc.take_snapshot()
            for stat in snap.statistics("lineno")[:tracemalloc_top]:
                tl_top.append(
                    {
                        "size_mb": round(stat.size / 1024 / 1024, 1),
                        "count": stat.count,
                        "trace": str(stat),
                    }
                )

        snapshot = MemSnapshot(
            label=label,
            pid=pid,
            rss_gb=round(proc.get("Rss", 0) / 1024 / 1024, 3),
            pss_gb=round(proc.get("Pss", 0) / 1024 / 1024, 3),
            private_clean_gb=round(
                proc.get("Private_Clean", 0) / 1024 / 1024, 3
            ),
            private_dirty_gb=round(
                proc.get("Private_Dirty", 0) / 1024 / 1024, 3
            ),
            python_object_count=py_obj_count,
            numpy_array_count=np_count,
            numpy_total_gb=round(np_total / 1024 / 1024 / 1024, 3),
            pandas_df_count=pd_count,
            pandas_total_gb=round(pd_total / 1024 / 1024 / 1024, 3),
            top_objects=top_objects,
            top_arrays=top_arrays,
            tracemalloc_top=tl_top,
        )
        return snapshot, gc_ids

    def compare(self, before: MemSnapshot, after: MemSnapshot) -> str:
        """Human-readable diff between two snapshots."""
        lines = [
            f"{'Metric':<25} {'Before':>10} {'After':>10} {'Δ':>10}",
            "─" * 60,
            self._row(
                "Private (GB)",
                before.private_clean_gb + before.private_dirty_gb,
                after.private_clean_gb + after.private_dirty_gb,
            ),
            self._row("RSS (GB)", before.rss_gb, after.rss_gb),
            self._row("PSS (GB)", before.pss_gb, after.pss_gb),
            "─" * 60,
            self._row(
                "Python objects",
                before.python_object_count,
                after.python_object_count,
                int,
            ),
            self._row(
                "numpy tracked (GB)", before.numpy_total_gb, after.numpy_total_gb
            ),
            self._row(
                "pandas total (GB)",
                before.pandas_total_gb,
                after.pandas_total_gb,
            ),
            self._row(
                "pandas DF count",
                before.pandas_df_count,
                after.pandas_df_count,
                int,
            ),
            "─" * 60,
        ]
        return "\n".join(lines)

    class tracer:
        """Memory execution tracer."""

        def __init__(self, enable_tracemalloc: bool = False):
            self._diag = MemDiag()
            self._enable_tracemalloc = enable_tracemalloc
            self.before: Optional[MemSnapshot] = None
            self.after: Optional[MemSnapshot] = None
            self.report: str = ""
            self._before_ids: set[int] = set()

        def __enter__(self):
            gc.collect()
            if self._enable_tracemalloc and not tracemalloc.is_tracing():
                tracemalloc.start()

            self.before, self._before_ids = self._diag.take("before")
            return self

        def __exit__(self, *exc):
            gc.collect()
            self.after, _ = self._diag.take("after")
            self._build_report()

            # Release before_ids immediately to free memory
            self._before_ids.clear()
            return False

        def _build_report(self):
            lines: list[str] = [
                self._sep("SUMMARY"),
                self._diag.compare(self.before, self.after),
            ]

            # Collect metadata 
            new_dfs = []
            new_objs = []

            for obj in gc.get_objects():
                if id(obj) in self._before_ids:
                    continue

                if isinstance(obj, pDF):
                    try:
                        sz = obj.memory_usage(deep=True).sum() / 1024 / 1024
                        new_dfs.append(
                            (
                                sz,
                                list(obj.shape),
                                [str(c) for c in obj.columns[:4]],
                            )
                        )
                    except Exception:
                        continue
                else:
                    try:
                        sz = sys.getsizeof(obj)
                        if sz >= self._diag.MIN_OBJECT_BYTES:
                            new_objs.append((sz, type(obj).__name__))
                    except (TypeError, SystemError):
                        continue

            # Sort metadata
            new_dfs.sort(key=lambda x: x[0], reverse=True)
            lines.append(self._sep(f"NEW DataFrames ({len(new_dfs)})"))
            for sz, shape, cols in new_dfs[:20]:
                lines.append(
                    f"  {sz:>8.1f} MB  shape={shape}  cols[:4]={cols}"
                )

            if len(new_dfs) > 20:
                rest_total = sum(sz for sz, _, _ in new_dfs[20:])
                lines.append(
                    f"  ... +{len(new_dfs) - 20} more  ({rest_total:.1f} MB total)"
                )

            new_objs.sort(key=lambda x: x[0], reverse=True)
            lines.append(self._sep(f"NEW large objects ({len(new_objs)})"))
            for sz, name in new_objs[:15]:
                lines.append(f"  {sz / 1024 / 1024:>8.1f} MB  {name}")

            if tracemalloc.is_tracing():
                snap = tracemalloc.take_snapshot()
                top_stats = snap.statistics("lineno")[:10]
                lines.append(self._sep("ALLOCATION ORIGINS (tracemalloc)"))
                for stat in top_stats:
                    lines.append(
                        f"  {stat.size / 1024 / 1024:>8.1f} MB  {stat.count:>5} blocks  {stat}"
                    )

            self.report = "\n".join(lines)

        @staticmethod
        def _sep(title: str) -> str:
            bar = "─" * 60
            return f"\n{bar}\n  {title}\n{bar}"

    @staticmethod
    def _read_smaps_rollup(pid: int) -> dict[str, int]:
        """Reads /proc/{pid}/smaps_rollup with psutil fallback."""
        try:
            with open(f"/proc/{pid}/smaps_rollup") as f:
                result: dict[str, int] = {}
                for line in f:
                    if ":" in line:
                        key, val = line.split(":", 1)
                        parts = val.strip().split()
                        if parts:
                            result[key.strip()] = int(parts[0])
                return result
        except (OSError, ValueError):
            # Fallback for non-Linux / macOS environments
            try:
                import psutil

                process = psutil.Process(pid)
                mem = process.memory_info()
                return {"Rss": mem.rss // 1024, "Pss": mem.rss // 1024}
            except Exception:
                return {}

    @staticmethod
    def _row(
        name: str, before: float, after: float, cast: type = float
    ) -> str:
        delta = after - before
        if cast is int:
            return f"{name:<25} {before:>10,.0f} {after:>10,.0f} {delta:>+10,.0f}"
        return f"{name:<25} {before:>10.2f} {after:>10.2f} {delta:>+10.2f}"
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class OffloadTracer:
    """
    Context manager to trace, clean up, and diagnose worker offloads.

    Usage
    -----
    with OffloadTracer(f"Worker [{worker.name}] Offload") as t:
        # 1. Save PyArrow dataset/table to disk and null it
        fx()
        
    print(t.report)
    """

    def __init__(self, label: str = "Worker Offload"):
        self.label = label
        self.pid = os.getpid()

        # Step metrics
        self.pa_before_mb: float = 0.0
        self.pa_after_gc_mb: float = 0.0
        self.pa_after_release_mb: float = 0.0

        self.private_before_gb: float = 0.0
        self.private_after_gb: float = 0.0

        self.rss_before_gb: float = 0.0
        self.rss_after_gb: float = 0.0

        self.diagnosis: str = ""
        self.report: str = ""

    def __enter__(self):
        # Initial cleanup to get clean baseline
        gc.collect()

        pool = pa.default_memory_pool()
        self.pa_before_mb = pool.bytes_allocated() / 1024 / 1024

        mem = self._read_smaps(self.pid)
        self.private_before_gb = (
            mem.get("Private_Dirty", 0) / 1024 / 1024
        )  # MB to GB
        self.rss_before_gb = mem.get("Rss", 0) / 1024 / 1024

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False  # Do not suppress exceptions raised inside the block

        pool = pa.default_memory_pool()

        # Step 1: Force Python GC (Destroys dead wrappers & drops C++ ref counts)
        gc.collect()
        self.pa_after_gc_mb = pool.bytes_allocated() / 1024 / 1024

        # Step 2: Ask C++ pool to return empty arenas back to Linux
        pool.release_unused()
        self.pa_after_release_mb = pool.bytes_allocated() / 1024 / 1024

        # Final OS Snapshot
        mem = self._read_smaps(self.pid)
        self.private_after_gb = mem.get("Private_Dirty", 0) / 1024 / 1024
        self.rss_after_gb = mem.get("Rss", 0) / 1024 / 1024

        # Build Diagnosis & Report
        self._diagnose()
        self._build_report()
        return False

    def _diagnose(self):
        pa_freed = self.pa_before_mb - self.pa_after_release_mb
        private_freed_mb = (
            self.private_before_gb - self.private_after_gb
        ) * 1024

        if self.pa_after_release_mb > (self.pa_before_mb * 0.8) and pa_freed < 10:
            self.diagnosis = (
                "❌ REFERENCE LEAK DETECTED:\n"
                "   PyArrow C++ memory did NOT drop when 'table = None' was set.\n"
                "   A downstream view, slice, column, Pandas DataFrame, or cache\n"
                "   is still holding strong references to the PyArrow C++ buffers."
            )
        elif pa_freed > 10 and private_freed_mb < (pa_freed * 0.2):
            self.diagnosis = (
                "⚠️ C++ ARENA FRAGMENTATION / PAGE CACHE:\n"
                "   PyArrow C++ logically freed the memory, but Linux Private_Dirty RAM\n"
                "   did not drop proportionally. Cause: C++ allocator arena fragmentation,\n"
                "   unflushed write buffers, or OS file page cache."
            )
        else:
            self.diagnosis = (
                "✅ SUCCESSFUL OFFLOAD:\n"
                "   PyArrow C++ buffers cleared and physical Linux RAM returned to kernel."
            )

    def _build_report(self):
        lines = [
            f"\n{'='*60}",
            f" 🧹 OFFLOAD DIAGNOSTIC REPORT: {self.label}",
            f"{'='*60}",
            f"1. PyArrow C++ Allocated Pool:",
            f"   - Before offload:         {self.pa_before_mb:>8.1f} MB",
            f"   - After gc.collect():     {self.pa_after_gc_mb:>8.1f} MB  (Δ {self.pa_after_gc_mb - self.pa_before_mb:+8.1f} MB)",
            f"   - After release_unused(): {self.pa_after_release_mb:>8.1f} MB  (Δ {self.pa_after_release_mb - self.pa_before_mb:+8.1f} MB)",
            f"\n2. Linux OS Process Memory:",
            f"   - Private_Dirty RAM:      {self.private_before_gb:>8.2f} GB -> {self.private_after_gb:>8.2f} GB  (Δ {self.private_after_gb - self.private_before_gb:+6.2f} GB)",
            f"   - Total RSS RAM:          {self.rss_before_gb:>8.2f} GB -> {self.rss_after_gb:>8.2f} GB  (Δ {self.rss_after_gb - self.rss_before_gb:+6.2f} GB)",
            f"{'-'*60}",
            f"DIAGNOSIS:",
            f"{self.diagnosis}",
            f"{'='*60}\n",
        ]
        self.report = "\n".join(lines)

    @staticmethod
    def _read_smaps(pid: int) -> dict[str, int]:
        try:
            with open(f"/proc/{pid}/smaps_rollup") as f:
                res = {}
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        parts = v.strip().split()
                        if parts:
                            res[k.strip()] = int(parts[0])
                return res
        except (OSError, ValueError):
            return {}
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬