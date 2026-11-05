"""
High-dimensional correlation engine with dynamic memory optimization and parallel orchestration.

This module implements an intelligent correlation framework designed to handle datasets 
that exceed available system RAM. By analyzing the quadratic memory requirements of 
correlation matrices, it dynamically determines the optimal execution strategy—switching 
between single-threaded processing and a multi-worker parallel fold approach based on 
available CPU cores and feature density.
"""

from ...deps.common         import  (
                                    os, 
                                    random,
                                    batched
                                    )
from ...deps.struct         import  dataclass
from ...deps.data           import  np, pDF
from ...deps.data_view      import  itables_show
from ...deps.plots          import  go
from ...deps.display        import  display
from ...deps.types          import  (
                                    NDArray,
                                    Literal
                                    )
from ...core.factory        import  DictClass
from ...core.accessor       import  (
                                    Parallel, 
                                    Spinner
                                    )
from ...core.standard       import  TypeDef
from ...core.interop        import  Pd

from ...utils.memory        import  Memory
from ...display.style       import  ItablesStyle


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# CORRELATION REDUCTION WITH PARALLEL ENGINE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬   
class MULTI(DictClass):
    value: str = 'max_corr_value'
    var:   str = 'max_corr_var'

@dataclass
class Plan:
    process: Literal['single', 'parallel', 'serial']
    n_jobs:     int
    batch_size: int
    reason:     str
    
class CorrProcess:
    """
    Creates Processing plan for Correlation.
    a) Evaluates if correlation can be computed in memory at once → single process.
    b) If not and cpu cores sufficient, calculates the optimal chunk size → parallel process.
    c) If parallel cannot be executed due to cores limitation → serial process.
    """
    FLOAT_BYTES = Memory.DTYPE_MEMORY['float64']

    def __init__(self, 
        memory_reserve:  float = 0.3,
        memory_limit_gb: float = None,
        available_cores:   int = None
    ):
        self.available_cores = available_cores        
        self.safe_mem = self._safe_mem(memory_limit_gb, memory_reserve)

        self.plan = Plan(
            process    = 'single',
            n_jobs     = 1,
            batch_size = None,
            reason     = None
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _safe_mem(memory_limit_gb:float, safety_margin:float) -> int:
        """Server/System available MEM."""
        available_mem = Memory.get_local_system()
        if memory_limit_gb is not None:
            available_mem = min(available_mem, memory_limit_gb*(1024**3))
        return int( available_mem * (1 - safety_margin) )
    
    @classmethod
    def _compute_mem(cls, n_samples:int, n_features:int) -> int:
        """
        Estimate memory required for FULL computation.
        Intermediate centered data + Final C x C output matrix.
        """
        sample_mem   = n_samples  * n_features * cls.FLOAT_BYTES
        corr_res_mem = n_features * n_features * cls.FLOAT_BYTES
        return 2*sample_mem + corr_res_mem  # n*2 → safety buffer for numpy internal data copies
    # ─────────────────────────────────────────────────────────────────────────────────────────────
            
    @classmethod
    def _find_max_worker_fes(cls, n_samples:int, worker_mem:int) -> int:
        """
        If OOM risk, calculate max features per chunk (c)
        If running in parallel, available MEM is split among n_jobs workers.

        Thus, solve the quadratic equation for c.
        sample_mem + corr_res_mem <= ram_per_worker:
         - (n_samples * c * dtype) + (c^2 * dtype) <= ram_per_worker
         - 1*c^2 + N*c - (ram_per_worker / dtype) = 0
         - a*x^2 + b*x + c = 0 --> D = ( b^2 - 4ac )
        """        
        C_term = worker_mem / cls.FLOAT_BYTES
        b = n_samples * 2 # NOTE n*2 → safety buffer for numpy internal data copies
        discriminant = (b ** 2) + (4 * C_term)
        
        # a is positive → (-b+D^0.5)/2 root is the max
        max_c = (-b + np.sqrt(discriminant)) / 2 
        return int(np.floor(max_c))
    
    def _find_optimal_n_workers(self, n_samples:int, n_features:int) -> tuple[int, int]:
        """
        Calculate minimum workers needed to keep each chunk under the RAM limit.
        If the server can handle 1000 features, and we have 5k, we need 5^2 workers.

        1. We would like to `find min workers` so that corr chunks have max 
        overlapping features.

        2. By the quadratic mem needs given CxC output matrix, split to `n`
        workers leads to mem reduction by exactly `n` e.g.:

         - orig. 1000x1000 = 10^6 vs. split to 10 workers 100x100 = 10^4 → 10 * 10^4.
         - Vice versa, having 10x more FEs than max mem means 100x more mem reqs.

        How to be fast yet fit to MEM when data_fes >> max_system_fes:
        --------------------------------------------------------------
        Our goal is to find such parallel split which will fit into safe mem
        at those feature splits corr can be computed in parallel. 

        e.g.:
        - Total Features      : 3 x max_fes; thus k_factor = 3
        - Workers needed      : 3^2; (k_factor^2)
        - Features per Worker : (3 x max_fes) / (3^2) = 1/3 max_fes
        
        - Since memory is quadratic, the RAM used by each worker is:
        (1/3 max_fes)^2 = 1/9 max_fes^2
        
        `- The System View : 9x(1/9 safe_ram) = safe_ram`
        """        
        # Optim to find `min mem safe n` workers | cpu cores
        max_fes  = self._find_max_worker_fes(n_samples, self.safe_mem)
        k_factor = n_features / max_fes
        n_jobs   = int(np.ceil(k_factor) ** 2)  # 4, 9, 16, ...
        cores    = self.available_cores or os.cpu_count() or 1

        # KO → have to go slow mode in serial processing
        if cores < n_jobs:            
            return 0, self._verify_chunk(max_fes)  

        # OK → we have resources to run everything in parallel
        mem_worker = self.safe_mem / n_jobs
        chunk_size = self._find_max_worker_fes(n_samples, mem_worker)            

        return n_jobs, self._verify_chunk(chunk_size, mem_worker)

    def _verify_chunk(chunk_size:int, mem_per_worker:int=None) -> int:
        if chunk_size >= 2:
            return chunk_size
        if not mem_per_worker:
            mem_per_worker = Memory.get_local_system()
        raise MemoryError(
            f"Available memory per worker ({mem_per_worker / (1024**3):.2f} GB) "
            "is too low to correlate even 2 features."
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def set_plan(self, data:pDF|NDArray) -> None:
        """Set execution processing plan with respect to system capabilities."""
        
        n_samples, n_features = data.shape
        compute_mem = self._compute_mem(n_samples, n_features)

        m = ( f"Compute Requires: {compute_mem / (1024**3):.2f} GB | "
              f"Available Safe Mem: {self.safe_mem / (1024**3):.2f} GB." )

        # (a) single processing ok
        if compute_mem <= self.safe_mem:
            self.plan = Plan(
                process    = 'single',
                n_jobs     = 1,
                batch_size = n_features,
                reason = f"Sufficient Memory ► {m}"
            )
            return

        n_jobs, batch_size = self._find_optimal_n_workers(n_samples, n_features)
        process = 'parallel' if n_jobs else 'serial'

        # (b|c) serial|parallel processing needed
        self.plan = Plan(
            process    = process,
            n_jobs     = n_jobs,
            batch_size = batch_size,
            reason = f"Split to {process} process ► {m} | "
                     f"Folds Compute : Splitting into {n_jobs} chunks of ~{batch_size} features."
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def get_plan(self) -> Plan:
        return self.plan
    
    @property
    def is_single(self) -> bool:
        return self.plan.process == 'single'
    
    @property
    def is_serial(self) -> bool:
        return self.plan.process == 'serial'

    @property
    def is_parallel(self) -> bool:
        return self.plan.process == 'parallel'    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬   

class Correlation:
    """
    Computes correlation of numeric type variables.

    `(a) len(vars) <= available mem`: Standard one pass correlation matrix.
    `(b) len(vars) > available mem` : Computing correlations by folds in parallel.

    Corr by Folds need certain stop factor → parallel_th is an approx. how to reach uncorrelated 
    full data even though corr is computed in Folds → fold data doesn't see other folds.

    `parallel_th`
     - the value represents requested min stability factor in vars_in between run_t+1 / run_t
     - if new interation result across all folds will not cause change in vars_in (given th), we stop
    
    Parallel corr in sum:
     - Until parallel_th is exceeded we are continuing correlation compute by batches (folds)
     - i.e. after n-parallel interations, len(vars_in_t+1) / len(vars_in_t) >= parallel_th:
       → no reasonable to compute more folds as we are not changing vars_in/vars_out significantly
    """

    def __init__(self, 
        max_corr:        float = 0.90, 
        parallel_th:     float = 0.95, 
        memory_reserve:  float = 0.3,
        memory_limit_gb: float = None,
        available_cores:   int = None,
        verbose:          bool = True
    ):
        self.max_corr        = max_corr
        self.parallel_th     = parallel_th
        self.verbose         = verbose
        
        self.corr_process = CorrProcess(memory_reserve, memory_limit_gb, available_cores)
        self._set_result()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_result(self) -> None:
        self.corr_matrix = None
        self.vars_out = []

    def _align_multi_result(self, X:pDF) -> None:
        if self.corr_matrix is None or not len(self.corr_matrix):
            return
        cols = [x for x in X.columns if x in self.corr_matrix.index]        
        self.corr_matrix = self.corr_matrix.loc[cols]

    def get_result(self) -> tuple[pDF, list]:
        """
        Provides Correlation Matrix or Max Vector for parallel|serial processing
        with vector of vars_out → variable > max_corr.
        """
        return self.corr_matrix, self.vars_out
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def upper(corr_matrix: pDF, k: int = 0) -> pDF:
        """Returns the Upper Triangle of correlation matrix."""
        return corr_matrix.where( np.triu(np.ones(corr_matrix.shape), k=k).astype(bool) )
    
    def test(self, data:pDF, triu:bool=False) -> tuple[pDF, list]:
        """
        Correlation Compute with respect to missing-rate.
        Keep corr vars with the highest data hit-rate.
        """
        data_sorted = data[data.count().sort_values(ascending=False).index]
        corr_matrix = data_sorted.corr().abs()
        corr_upper  = self.upper(corr_matrix, k = 1)
        vars_out    = [x for x in corr_upper.columns if any(corr_upper[x] > self.max_corr)]
        if triu: 
            corr_matrix = corr_upper
        return corr_matrix, vars_out
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def single(self, data:pDF, triu:bool=True) -> tuple[pDF, list]:
        """
        Correlation Compute with respect to missing-rate.
        Keep corr vars with the highest data hit-rate.
        """
        x = len(data.columns)
        v = (x > 100) and self.verbose
        m = f'Computing Correlations of {x} Features'
        with Spinner(m, show_timer=True).progress(v):
            return self.test(data, triu) 
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _vars_res(vars_all:list|pDF, vars_out:list) -> tuple[list, list]:
        """Returns vars_ok, vars_out with respect to orig. data order."""
        if isinstance(vars_all, pDF):
            vars_all = vars_all.columns
        if isinstance(vars_out[0], list):
            vars_out = [x for X in vars_out for x in X]
        vars_ok  = [x for x in vars_all if x not in vars_out]
        vars_out = [x for x in vars_all if x in vars_out]        
        return vars_ok, vars_out
    
    @staticmethod
    def _get_corr_max_pair(corr_matrix: pDF) -> pDF:
        """Returns pairs of x-y with max corr."""
        corr = ( corr_matrix
                .where(~np.eye(len(corr_matrix), dtype=bool), None) # null diagonal
                .fillna(-1) )
        null  = (corr == -1).sum()
        null  = null[null==len(corr)].index
        c_max = corr.max()
        c_var = corr.idxmax()
        corr  = pDF({
            MULTI.value: c_max, 
            MULTI.var:   c_var  
        })
        corr.loc[null,:] = None
        return corr
            
    def _test(self, data:pDF, fes:list) -> tuple[pDF, list]:
        """Correlation processed in parallel/series returns pair-var with max corr."""
        corr, vars_out = self.test(data[fes], False)
        return self._get_corr_max_pair(corr), vars_out
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def parallel(self, data:pDF, n_jobs:int) -> tuple[pDF, list, list]:
        """
        Parallel Correlation Compute with respect to missing-rate.
        Keep corr vars with the highest data hit-rate.
        """
        x = len(data.columns)
        m = f'Correlation Parallel Processing of {x} Features'
        with Spinner(m).progress(self.verbose):
           return self._test_parallel(data, n_jobs)
    
    @staticmethod
    def _parallel_th(vars_all:list):
        vars_out = []
        def fx(vars_ok_prev:list, vars_ko:list):
            vars_out.extend(vars_ko)
            vars_ok_new = [x for x in vars_all if x not in vars_out]
            parallel_th = len(vars_ok_new) / len(vars_ok_prev)
            return vars_ok_new, parallel_th
        return fx

    def _parallel_fx(self, n_jobs:int) -> _test:
        return Parallel(
            data_and_fe=1, pd_axis=0, th=0, n_jobs=n_jobs, verbose=False
        ).process(self._test)
    
    def _test_parallel(self, data:pDF, n_jobs:int) -> tuple[pDF, list, list]:
        test_ = self._parallel_fx(n_jobs)

        vars_all = data.columns
        vars_ok  = vars_all
        run_res  = []

        run_th = self._parallel_th(vars_all)
        th = np.clip(self.parallel_th, 0.8, 1)

        while True:
            max_corr, vars_ko = test_(data, vars_ok)
            run_res.append([max_corr, vars_ko])
            
            vars_ok, ok_ratio = run_th(vars_ok, vars_ko)

            if ok_ratio >= th:
                break
            random.Random(42).shuffle(vars_ok) 
            # random ensures that each parallel run has random fe sets for corr
        
        max_corr, vars_out = zip(*run_res)
        return Pd.concat(max_corr), *self._vars_res(vars_all, vars_out)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def serial(self, data:pDF, max_fes_batch:int) -> tuple[pDF, list, list]:
        """
        Parallel Correlation Compute with respect to missing-rate.
        Keep corr vars with the highest data hit-rate.
        """
        verbose_ = self.verbose
        self.verbose = False

        x = len(data.columns)
        m = f'Correlation Series Processing by {max_fes_batch} from {x} Total Features'
        with Spinner(m).progress(verbose_):            
           max_corr, vars_ok, vars_out = self._test_serial(data, max_fes_batch)
        
        self.verbose = verbose_               
        return max_corr, vars_ok, vars_out
    
    def _test_serial(self, data:pDF, max_fes_batch:int) -> tuple[pDF, list, list]:
        vars_all = data.columns
        batch_res = []
        for batch in batched(vars_all, max_fes_batch):
            batch_res.append( 
                self._test(data, list(batch)) 
            )
        max_corr, vars_out = zip(*batch_res)
        return Pd.concat(max_corr), *self._vars_res(vars_all, vars_out)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _subset(self, data:pDF, subset:list) -> pDF:
        x = [x for x,t in data.dtypes.items() if t in TypeDef.numeric]
        x = x if not subset else [s for s in subset if s in x]
        X = data[x]
        
        xv = len(X.columns)
        if not subset:
              dv = len(data.columns)
        else: dv = len(subset)
        
        if xv != dv and self.verbose:
            print(f'WARNING: Subset to Numerical types applied → input vars: {dv} | subset: {xv}')

        return X
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _align_with_multi_corr(self, corr:pDF, is_multi_process:bool) -> pDF:
        if not is_multi_process: 
            return corr
        # final single pass after parallel|serial            
        return self._get_corr_max_pair(corr)

    def _merge_multi_corr(self, corr_multi:pDF) -> pDF:
        corr = Pd.concat([self.corr_matrix, corr_multi])
        corr = ( corr.reset_index()
                .sort_values(['index', MULTI.value], ascending=False)
                .drop_duplicates(subset=['index'])
                .set_index('index') )
        corr.index.name = None
        return corr

    def _update_corr(self, corr:pDF) -> None:
        if self.corr_matrix is None: 
            self.corr_matrix = corr
            return        
        self.corr_matrix = self._merge_multi_corr(corr)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _update_result(self, corr:pDF, vars_ko:list) -> None:
        self._update_corr(corr)
        self.vars_out.extend(vars_ko)
    
    def _run_serial(self, X:pDF) -> list:
        plan = self.corr_process.get_plan()
        corr, vars_ok, vars_ko = self.serial(X, plan.batch_size)
        self._update_result(corr, vars_ko)
        return vars_ok
        
    def _run_parallel(self, X:pDF) -> list:
        plan = self.corr_process.get_plan()
        corr, vars_ok, vars_ko = self.parallel(X, plan.n_jobs)
        self._update_result(corr, vars_ko)
        return vars_ok
   
    def _run_single(self, X:pDF, triu:bool=True, is_multi_process:bool=False) -> pDF:
        corr, vars_ko = self.single(X, triu)
        corr = self._align_with_multi_corr(corr, is_multi_process)
        self._update_result(corr, vars_ko)
    
    def _update_process(self, X_updated:pDF) -> None:
        self.corr_process.set_plan(X_updated)
        # final cleaning on reduced data by parallel/serial pre-processed
        if self.corr_process.is_single:            
            self._run_single(X_updated, triu=False, is_multi_process=True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def process(self, data:pDF, subset:list=None) -> tuple[pDF, list]:
        """Computes correlation of numeric type variables."""
        
        self._set_result()
        X = self._subset(data, subset)
        
        if len(X.columns) < 2:
            return X.corr(), []

        self.corr_process.set_plan(X)
        
        # (a) Processing Mem OK → direct end by Single corr CxC
        if self.corr_process.is_single:
            self._run_single(X)
            return self.get_result()
        
        # # # (b) Processing OOM → slow-mode by Serial processing Mem KO
        if self.corr_process.is_serial:
            vars_ok = self._run_serial(X)
            self._update_process(X[vars_ok])        
        else:
            vars_ok = X.columns.to_list()

        # (c) Processing OOM → fast-mode by Parallel processing Mem OK 
        if self.corr_process.is_parallel:
            vars_ok = self._run_parallel(X[vars_ok])
            self._update_process(X[vars_ok])
        
        self._align_multi_result(X)
        return self.get_result()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def show( 
        corr_mx:  pDF, 
        max_corr: float = None, 
        kind:     Literal['pd', 'itables'] = 'itables'
    ) -> None:
        """Correlation Matrix View with optional highlight of value > max_corr."""
        CorrView.show(corr_mx, max_corr, kind)

    @staticmethod
    def heatmap(
        *,
        data:   pDF = None, 
        xy:    list = None, 
        upper:  pDF = None,         
        text:   str = None,
        update_layout: dict = None, 
    ) -> go.Figure:
        """Correlation Heatmap."""
        return CorrView.heatmap(
            data = data, xy = xy, upper = upper, text = text, update_layout = update_layout
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class CorrView:

    @staticmethod
    def _itables(data:pDF, max_corr:float=None) -> None:
        j1 = ItablesStyle.replace_nan()
        j2 = ItablesStyle.number_color(max_corr)
        js = ItablesStyle.combine(j1, j2)
        itables_show(
            data, 
            maxBytes   = "5MB",
            columnDefs = [{"targets": "_all", "createdCell": js}]
        )

    @staticmethod
    def _pd(data:pDF, max_corr:float=None) -> None:
        def cond(val):
            if max_corr is None:
                return ''
            is_th  = val > max_corr
            color  = '#B50064' if is_th else 'black'
            weight = '500' if is_th else 'normal'
            return f'color: {color}; font-weight: {weight}'
        
        display( data.style.map(cond).format(na_rep="") )

    @classmethod
    def show(cls, 
        corr_mx:  pDF, 
        max_corr: float = None, 
        kind:     Literal['pd', 'itables'] = 'itables'
    ) -> None:
        """Correlation Matrix View with optional highlight of value > max_corr."""
        d = corr_mx.T
        if kind == 'itables':
            cls._itables(d, max_corr)
        else:
            cls._pd(d, max_corr)

    @staticmethod
    def heatmap(
        *,
        data:   pDF = None, 
        xy:    list = None, 
        upper:  pDF = None,         
        text:   str = None,
        update_layout: dict = None, 
    ) -> go.Figure:    
        """Returns Correlation Heatmap figure."""

        if upper is None:
            corr  = data.corr().abs()
            upper = Correlation.upper(corr)
                
        if xy is None:
            if data is not None: xy = data.columns
            else: xy = upper.columns
            
        fig = go.Figure(data=go.Heatmap(
                z = upper.values,
                x = xy,
                y = xy,
                text = text,
                hoverongaps = False,
                colorscale  = 'Jet' 
        ))    

        fig.update_layout(**(dict(
            title  = None, 
            height = 600, 
            margin = {'t':20, 'b':20, 'l':10, 'r':10}
            ) | (update_layout or {})
        ))

        return fig    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬