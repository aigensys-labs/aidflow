"""
High-performance compute orchestration layer for distributed data processing.

This module provides a unified interface for executing CPU and GPU intensive 
ML tasks in parallel, abstracting the underlying backend (Multiprocessing or Ray). 
It implements an intelligent distribution engine that optimizes workload splitting 
based on available hardware resources, ensuring maximum throughput while 
minimizing memory fragmentation.

Key architectural capabilities:
1. Backend Agnostic Execution: Seamlessly switches between `ParallelMulti` 
   (local multiprocessing) and `ParallelRay` (distributed cluster computing) 
   via a single configuration toggle, allowing the same pipeline to scale from 
   a laptop to a cloud cluster without code changes.
   
2. Resource-Aware Data Distribution: Implements an optimized splitting logic 
   (`ParallelDist`) that calculates the most efficient data shards based on 
   CPU core counts and dataset size, preventing memory overflow during parallel 
   execution.

3. Hybrid Memory Management: Integrates with Ray's object store to handle large-scale 
   dataset sharing across workers, eliminating redundant data copying and 
   reducing I/O overhead in high-dimensional feature engineering.

4. Smart Result Aggregation: Features a type-aware merging system (`ParallelMerge`) 
   that automatically reconstitutes split results into their original format 
   (DataFrames, Dicts, or Lists) based on the function's return signature.
"""
from ...deps.common     import  (
                                os,
                                wraps
                                )
from ...deps.execution  import  (
                                ray, 
                                Pool, 
                                RLock
                                )
from ...deps.data       import  np, pDF, pSe
from ...deps.display    import  tqdm
from ...deps.types      import  (
                                FuncType, 
                                Literal
                                )
from ..factory          import  (
                                DictClass,
                                Dataclass, 
                                DictDataclass
                                )
from ..foundation       import  AigStop
from ..interop          import  Pd, DictMerger
from ..logging          import  Logger

from ...utils.ignore    import  Warnings


logger = Logger(__name__)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PARALLEL DATA PROCESSING
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class KW(DictClass):
    EXE:str = '__parallel_exe'
    POS:str = '__parallel_pos'

@Dataclass.base
class Params:
    data_and_fe:  int = 0 
    th:     float|int = 100
    pd_axis:      int = 1
    n_jobs:       int = None
    fe_is_group: bool = False
    kwargs_pos:  bool = False
    verbose:     bool = True
    worker_opt:  dict = None

    def __post_init__(self):
        self._verify()
    
    def _verify(self):
        if not (self.fe_is_group and not self.data_and_fe):
            return
        AigStop.custom(
        f'AIG SAFETY ► PARALLEL EXE'
        '\n► Parameter "data_and_fe" set to 1'
        '\n► "fe_is_group" parameter was set to True but parameter "data_and_fe" is 0.'
        '\n► "fe_is_group" represents special case to split data by unique group present in the data.'
        '\n► If you would like to process data in parallel by groups, please also set data_and_fe=1.'
        )        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ParallelGroup:
    """Utils for data multiprocessing distribution by group ids."""
    
    @staticmethod
    def _verify_data(data:pDF, group:str) -> None:
        if group == data.index.name:
            return
        if data.get(group) is not None:
            return
        raise ValueError(
            '"fe_is_group" was set to True but no group was found in the data.'
            'Please make sure that data-group link exists to operate join on group data.'
        )        

    @classmethod
    def _verify_group(cls, data:pDF, group:pSe|str) -> None:
        g = None
        if   isinstance(group, str): g = group
        elif hasattr(group, 'name'): g = group.name
        if not g:
            raise TypeError(
            'Group parameter provided as non string but not as a Series'
            '→ Only Series of unique ids can be provided to infer group name'
            )
        cls._verify_data(data, g)
    
    @staticmethod
    def get_d_group(args:tuple) -> tuple[pDF, list|str]:
        return args[0], args[1]

    @classmethod
    def get_unique_ids(cls, args:tuple) -> list|pSe:
        data, group = cls.get_d_group(args)
        cls._verify_group(data, group)
        if not isinstance(group, str): 
            return group
        if data.index.name == group:
            return data.index.unique().to_list()
        return data[group].unique().to_list()  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ParallelDist:
    """Multiprocessing data distribution and unpacking."""

    @staticmethod
    def _most_distributed(L:int, S:int) -> int:
        """Find the most distributed data split of len L with respect to split ceiling S."""
        S = int(np.clip(S, 2, L))   # Split Standardization
        R = range(2, S+1)           # Split Exploration Space
        
        f = L % S                   # Init Residual Value
        r = max(2, int(L/S*0.05))   # Init Max Allowed Residual  
        s = S                       # Init Return Split Value
        
        while f > r:                # Search for min residual
            r+= 1     
            z = [L % s for s in R]
            p = [R[i] for i,v in enumerate(z) if v < r]
            if p:
                s = max(p) # Higher position means higher split → closer to orig. S
                break
        return s

    @staticmethod
    def _get_split(max_splits:int|None) -> int:
        """Parallel splits driven by resorces or user request."""
        if max_splits is None:
            cpu = os.cpu_count()        
            max_splits = (cpu-1 if cpu else 4) 
            # -1 buffer for processing, memory management, I/O, OS...
        return max_splits
    
    @classmethod
    def distribute(cls, 
        X: list | tuple | dict | pSe, 
        max_splits: int = None
    ) -> tuple[list, range]:
        """Find the most distributed data split which will utilize max available resources."""
        
        is_dict = isinstance(X, dict)
        
        X_ = X        
        if is_dict:
            X_ = list(X)
        
        l = len(X_)
        s = cls._get_split(max_splits)
        p = cls._most_distributed(l,s)
        
        c  = int(l/p)
        r  = list( range(0,l,c) )
        xs = [X_[start:end] for start, end in zip(r, r[1:] + [None])]

        if is_dict:
            xs = [{x:X[x] for x in xs_} for xs_ in xs]
            
        return xs, range(len(xs))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _verify_setup(args:tuple, prm:Params) -> None:
        if not prm.data_and_fe:
            return
        if prm.data_and_fe and isinstance(args[0], pDF):
            return
        AigStop.custom(
            'AIG SAFETY ► PARALLEL EXE'
            '\n► Parameter "data_and_fe" set to 1'
            '\n► The first argument must be pd.DataFrame, the second one list of Features | Group.'
        )

    @classmethod
    def _get_fes(cls, args:tuple, prm:Params) -> list | tuple | dict | pSe:
        """
        Standard:
         - The split factor is given by prm.data_and_fe.
         - prm.data_and_fe → [0] Dict-keys or [1] Variables to split by.
        
        Group → fe_is_group = True: 
         - Special case where second arg is a group.
         - ids obtained either directly from the data or from the provided list|pSe.
        """
        cls._verify_setup(args, prm)
        if prm.fe_is_group:
            return ParallelGroup.get_unique_ids(args)
        return args[prm.data_and_fe]
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def distribute_fes(cls, args:tuple, prm:Params, n_jobs:int=None) -> tuple[list, range]:
        """Split Features and or Data (data_and_fe) by best distributed split."""
        X = cls._get_fes(args, prm)
        return cls.distribute(X, n_jobs or prm.n_jobs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ParallelMerge:

    @staticmethod
    def _merge_returned_by_type(t, batches_result:list, pd_axis:int):
        """Merge fx results from the multi-processing."""
        if isinstance(t, (list, tuple)):
            merged = [r for res in batches_result for r in res]
        elif isinstance(t, (dict, DictDataclass)):
            merged = DictMerger.merge_list(batches_result, pd_axis)
        elif isinstance(t, set):
            merged = set().union(*batches_result)
        elif isinstance(t, pDF):
            merged = Pd.concat(batches_result, axis=pd_axis)
        elif isinstance(t, pSe):
            merged = Pd.concat(batches_result, axis=0)
        else: 
            merged = batches_result                
        return merged    


    @staticmethod
    def _is_multiple_result(returned_value) -> bool:
        """
        Checks if a function returns a single value or multiple values (tuple), 
        with refined edge-case handling.

        Parameters
        ----------
        returned_value : The function returned object(s) to inspect.
        """
        if not isinstance(returned_value, tuple):
            return False
        
        # different types → likely multiple objects returned
        if ( 
            ( len(returned_value) > 1 ) and
            ( any(isinstance(item, type(returned_value[0])) == False for item in returned_value) )
        ):
            return True
        
        # single long tuple → likely single tuple returned
        if len(returned_value) > 10:
            return False
        
        # tuple with reasonable return length       
        return len(returned_value) > 1  
        
    @classmethod
    def results(cls, p_res:tuple, pd_axis:int):
        """Unpack and merge fx results from the multi-processing."""
        res0 = p_res[0]                             # fx returned obj. of 0th parallel batch
        if not cls._is_multiple_result(res0):        # check whether fx returns single or multiple objects         
            return cls._merge_returned_by_type(res0, p_res, pd_axis)
    
        res = {}                            
        for i,x in enumerate(res0):                 # loop over each returned object (i) and check its type (r)
            batches = [rx[i] for rx in p_res]       # ith object in return of orig. FX for all parallel batches
            res[i]  = cls._merge_returned_by_type(x, batches, pd_axis)
        return tuple(res.values())   
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ParallelBase:
    """
    Standard Parallel Wrapper for FX.
    
    If ( data_and_fe = 0 ) → FX has to have first argument either:
    (data,..) or (FEs,...) 
        - Either is used for split to parallel batches and FX executed with a batch part.
        - `data type: dict type only`.
        - `FEs type: list/tuple/set`.

    If ( data_and_fe = 1 ) → first two arguments must be in order:
    (data, FE, ...)
        - Both are split → logic first split FEs to batches and then data batches are created to hold only     
    FEs of the same batch to optimize memory usage.
        - `data type: pDF type only`.
        - `FEs type: list/tuple/set`.

    Parameters
    ----------
    data_and_fe : int = 0
        0 | 1 → When FX has first two arguments data and FE and split is done by FE.

    th : int = 100
        Threshold above which FX is executed parallel, otherwise orig FX w/o parallel.

    pd_axis : int = 1
        0 | 1 → dataframe(s) in result(s) - orientation to concat multi-res by pd.concat.
    
    n_jobs : int = None
          Parallel data splits → If default None then available CPUS.

    fe_is_group : bool = False
        Special case to indicate that second argument is a Group → split done by group.
    
    kwargs_pos : bool = False
        Inject parallel position kwargs: (parallel_pos, parallel_exe) to fx when 
        parallel is executed.

    verbose : bool = True
        Show spinner progress.

    worker_opt : dict = None
        When "ray" is the backend, custom options for ray.remote worker can be provided.
    
    spin_title : str = 'Multiprocessing in progress'
        Title shown next to the spinner animation.
    
    Returns
    ----------
    (Any) : Base Function outputs merged for all parallel batches.
    """   

    def __init__(self,        
        *,
        data_and_fe:  int = 0, 
        th:           int = 100, 
        pd_axis:      int = 1,
        n_jobs:       int = None,
        fe_is_group: bool = False,
        kwargs_pos:  bool = False,
        verbose:     bool = True,
        worker_opt:  dict = None,
        spin_title:   str = 'Multiprocessing in progress'
    ):
        self.prm = Params(
            data_and_fe = data_and_fe,
            th          = th,
            pd_axis     = pd_axis,
            n_jobs      = n_jobs,
            fe_is_group = fe_is_group,
            kwargs_pos  = kwargs_pos,
            verbose     = verbose,
            worker_opt  = worker_opt
        )
        self._msg = spin_title

    def _p_wrap(self, fx:FuncType, args:tuple, kwargs:dict) -> list:
        raise NotImplementedError

    def _p_worker(self, p_wrap:list):
        raise NotImplementedError
    
    def process(self, fx:FuncType):
        """
        Accepts target function which processes data in parallel.
        Target Function must meet expected design:
        
        def fx(ar0, ar1, *args, **kwargs):
           ...

        The arguments must be either:
          - data_and_fe = 1 → mandatory ar0 and ar1:
            • ar0: data as DataFrame 
            • ar1: list of fes or group id to create distributed batches

          - data_and_fe = 0 → mandaotry ar0 only:
            • ar0: when dict provided, dict is split into distributed batches
                   otherwise list/tuple/set is split to distributed batches
        """   
        @wraps(fx)  
        def wrapper(*args, **kwargs):
            from ..accessor import Spinner

            X = args[self.prm.data_and_fe] # Data or Variables to split by
            
            if ( len(X) < self.prm.th ) and ( not self.prm.fe_is_group ):  
                return fx(*args, **kwargs)                    
            
            al = Spinner(self._msg, show_timer=True)
            wp = al.with_progress(self.prm.verbose)

            p_wrap = self._p_wrap(fx, args, kwargs)
            p_res  = wp(self._p_worker, p_wrap)

            return ParallelMerge.results(p_res, self.prm.pd_axis)
        return wrapper
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PARALLEL BY MULTIPROCESSING
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class _MultiData:
    
    @staticmethod
    def get_d_fes(args:tuple) -> tuple[pDF, list|str]:
        return args[0], args[1]
    
    @classmethod
    def make_group_data(cls, args:tuple, splits:list) -> list:
        data, group = cls.get_d_fes(args)
        if not isinstance(group, str):
            group = group.name
        return [data.join(
                    pDF({group:x}).set_index(group), on=group, how='inner'
                ) for x in splits ] 

    @classmethod
    def make_fe_data(cls, args:tuple, splits:list):
        d, fes = cls.get_d_fes(args)
        oc = [c for c in d.columns if c not in fes]  # other cols
        return [d[oc + list(x)] for x in splits]
# ─────────────────────────────────────────────────────────────────────────────────────────────────
    
class ParallelMulti(ParallelBase):
    """
    Parallel Engine by multiprocessing.
    
    If ( data_and_fe = 0 ) → FX has to have first argument either:
    (data,..) or (FEs,...) 
        - Either is used for split to parallel batches and FX executed with a batch part.
        - `data type: dict type only`.
        - `FEs type: list/tuple/set`.

    If ( data_and_fe = 1 ) → first two arguments must be in order:
    (data, FE, ...)
        - Both are split → logic first split FEs to batches and then data batches are created to hold only     
    FEs of the same batch to optimize memory usage.
        - `data type: pDF type only`.
        - `FEs type: list/tuple/set`.

    Parameters
    ----------
    data_and_fe : int 
        0 | 1 → When FX has first two arguments data and FE and split is done by FE.

    th : int
        Threshold above which FX is executed parallel, otherwise orig FX w/o parallel.

    pd_axis : int
        0 | 1 → dataframe(s) in result(s) - orientation to concat multi-res by pd.concat.
    
    n_jobs : int
          Parallel data splits → If default None then available CPUS.

    fe_is_group : bool
        Special case to indicate that second argument is a Group → split done by group.
    
    kwargs_pos : bool
        Inject parallel position kwargs: (parallel_pos, parallel_exe) to fx when 
        parallel is executed.

    verbose : bool
        Show animated in progress.
    
    spin_title : str = 'Multiprocessing in progress'
        Title shown next to the spinner animation.

    Returns
    ----------
    (Any) : Base Function outputs merged for all parallel batches.
    """   

    def __init__(self,
        data_and_fe: int = 0, 
        th:          int = 100, 
        pd_axis:     int = 1,
        n_jobs:      int = None,
        fe_is_group:bool = False,
        kwargs_pos: bool = False,
        verbose:    bool = True,
        spin_title:  str = 'Multiprocessing in progress',
        **kwargs
    ):
        super().__init__(
            data_and_fe = data_and_fe,
            th          = th,
            pd_axis     = pd_axis,
            n_jobs      = n_jobs,
            fe_is_group = fe_is_group,
            kwargs_pos  = kwargs_pos,
            verbose     = verbose,
            spin_title  = spin_title
        )

    def _distribute_data(self, args:tuple, xs:list) -> None | list[pDF]:
        if not self.prm.data_and_fe:
            return None
        if self.prm.fe_is_group:
            return _MultiData.make_group_data(args, xs)
        return _MultiData.make_fe_data(args, xs)

    def _p_wrap(self, fx:FuncType, args:tuple, kwargs:dict) -> list:
        """"Helper to provide a standard parallel args-kwargs input to pool.map."""
        
        xs, r = ParallelDist.distribute_fes(args, self.prm)
        ds = self._distribute_data(args, xs)

        def arp(p):
            if ds is None:
                return [xs[p], *args[1:]]
            return [ds[p], xs[p], *args[2:]]

        def kwp(p):
            if self.prm.kwargs_pos:
                return kwargs | {KW.POS:p, KW.EXE:True}
            return kwargs
        
        return [(fx, arp(i), kwp(i)) for i in r]        
    
    @staticmethod
    def _p_unpack(p_wrap:tuple):
        """"Helper to provide a standard parallel args-kwargs input to pool.map."""
        Warnings.local_redundant()
        func, pos_args, kwargs = p_wrap
        return func(*pos_args, **kwargs)

    @classmethod
    def _p_worker(cls, p_wrap:list):
        """Multiprocessing worker."""
        with Pool(
            processes   = len(p_wrap), 
            initializer = tqdm.set_lock, 
            initargs    = (RLock(),)
        ) as pool:
            return pool.map(cls._p_unpack, p_wrap)  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PARALLEL BY RAY
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@ray.remote
def _ray_worker(p_wrap:list):
    """Ray worker."""
    return ParallelRay._p_unpack(p_wrap)    
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class ParallelRay(ParallelBase):
    """
    Parallel Engine by Ray.
    
    If ( data_and_fe = 0 ) → FX has to have first argument either:
    (data,..) or (FEs,...) 
        - Either is used for split to parallel batches and FX executed with a batch part.
        - `data type: dict type only`.
        - `FEs type: list/tuple/set`.

    If ( data_and_fe = 1 ) → first two arguments must be in order:
    (data, FE, ...)
        - Both are split → logic first split FEs to batches and then data batches are created to hold only     
    FEs of the same batch to optimize memory usage.
        - `data type: pDF type only`.
        - `FEs type: list/tuple/set`.

    Parameters
    ----------
    data_and_fe : int 
        0 | 1 → When FX has first two arguments data and FE and split is done by FE.

    th : int
        Threshold above which FX is executed parallel, otherwise orig FX w/o parallel.

    pd_axis : int
        0 | 1 → dataframe(s) in result(s) - orientation to concat multi-res by pd.concat.
    
    n_jobs : int
          Parallel data splits → If default None then available CPUS.

    fe_is_group : bool
        Special case to indicate that second argument is a Group → split done by group.
    
    kwargs_pos : bool
        Inject parallel position kwargs: (parallel_pos, parallel_exe) to fx when 
        parallel is executed.

    verbose : bool
        Show animated in progress.

    worker_opt : dict
        Custom options for ray.remote worker can be provided.
    
    spin_title : str = 'Multiprocessing in progress'
        Title shown next to the spinner animation.
            
    Returns
    ----------
    (Any) : Base Function outputs merged for all parallel batches.
    """   

    def __init__(self,
        data_and_fe: int = 0, 
        th:          int = 100, 
        pd_axis:     int = 1,
        n_jobs:      int = None,
        fe_is_group:bool = False,
        kwargs_pos: bool = False,
        verbose:    bool = True,
        worker_opt: dict = None,
        spin_title:  str = 'Multiprocessing in progress',
        **kwargs
    ):
        super().__init__(
            data_and_fe = data_and_fe,
            th          = th,
            pd_axis     = pd_axis,
            n_jobs      = n_jobs,            
            fe_is_group = fe_is_group,
            kwargs_pos  = kwargs_pos,
            verbose     = verbose,
            worker_opt  = worker_opt,
            spin_title  = spin_title
        )      
        self._ray_init()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def _ray_init(cls):
        if ray.is_initialized():           
            return   
        # Fix: Suppress "Future Warning" about GPU visibility.
        # Safe for mixed workloads. GPU tasks will still get correct isolation.
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
        os.environ["RAY_LOG_LEVEL"] = "ERROR"
        
        cls._ray_single()

        # try:    cls._ray_cluster()
        # except: cls._ray_single()
    
    @classmethod
    def _ray_cluster(cls):        
        conf = dict(
            ignore_reinit_error = True, 
            include_dashboard = True,
            dashboard_host = "0.0.0.0",
            log_to_driver = True,            
            runtime_env = {
                "working_dir": ".",
                # Exclude local data/venv to keep the 'zip and ship' fast
                "excludes": ["*.csv", "*.parquet", "venv/", "__pycache__/"],
            }
        )
        # We use address='auto' to join the head node started via 'ray start --head'
        ray.init(address='auto', **conf, connection_retries=0, request_timeout_ms=1000)

        # Validation: Print how many nodes we actually joined
        nodes = ray.nodes()
        active_nodes = [n for n in nodes if n['Alive']]
        logger.info(f"Ray Cluster Mode: {len(active_nodes)} Nodes Connected")

    @staticmethod
    def _ray_single():
        os.environ["RAY_LOG_LEVEL"] = "CRITICAL"
        
        conf = dict(
            ignore_reinit_error = True, 
            include_dashboard = False, 
            log_to_driver = False,
            _system_config = {
                "metrics_report_interval_ms": int(1e8), 
                "enable_metrics_collection": False
            }            
        )
        ray.init(**conf)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_available_cpus(self) -> int:
        r_cpu  = ray.cluster_resources().get("CPU", 0)
        n_jobs = self.prm.n_jobs or r_cpu
        return int(min(n_jobs, r_cpu))
    
    @staticmethod
    def _get_available_gpus(n_gpus:int|None) -> int:
        n_gpus = n_gpus or 0
        r_gpus = ray.cluster_resources().get("GPU", 0)
        if n_gpus and not r_gpus:
            logger.warning("GPU requested but no GPU found in Ray cluster. Falling back to CPU.")
        return int(min(n_gpus, r_gpus))
    
    def _worker_setup(self) -> dict:
        options = self.prm.worker_opt or {}
        if options.get('num_cpus'):
            logger.warning(
                "CPU count requested but design of Parallel processing is 1 CPU per batch."
            )
        gpus = self._get_available_gpus(options.get('num_gpus'))
        return options | dict(
            num_cpus = 1,   # given by the split to batches by available cpus we need 1 for a batch
            num_gpus = gpus
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _distribute_data(self, args:tuple) -> None | pDF:
        if not self.prm.data_and_fe:
            return None
        return ray.put(args[0])
    
    def _p_wrap(self, fx:FuncType, args:tuple, kwargs:dict) -> list:                
        """"Helper to provide a standard parallel args-kwargs input to ray worker."""
        
        n_jobs = self._get_available_cpus()        
        xs, r  = ParallelDist.distribute_fes(args, self.prm, n_jobs)        
        d_ref  = self._distribute_data(args)

        def arp(p):
            if d_ref is None:
                return [xs[p], *args[1:]]
            return [d_ref, xs[p], *args[2:]]

        def kwp(p):
            if self.prm.kwargs_pos:
                return kwargs | {KW.POS:p, KW.EXE:True}
            return kwargs
        
        return [(fx, arp(i), kwp(i)) for i in r]  

    @staticmethod
    def _p_unpack(p_wrap:list):
        """"Helper to provide a standard parallel args-kwargs input to ray worker."""
        func, pos_args, kwargs = p_wrap
        d = pos_args[0]
        if isinstance(d, ray.ObjectRef):
            pos_args = (ray.get(d), *pos_args[1:])
        return func(*pos_args, **kwargs)
    
    def _p_worker(self, p_wrap:list):
        """Multiprocessing worker."""        
        res = [
            _ray_worker.options(
                **self._worker_setup()
            ).remote(w) for w in p_wrap
        ]
        return ray.get(res)  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PARALLEL OFF
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ParallelOff(ParallelBase):
    """Dummy class to support turning off parallel backend."""   

    def process(self, fx:FuncType):
        """Dummy wrapper to support turning off parallel backend."""   
        @wraps(fx)  
        def wrapper(*args, **kwargs):
            return fx(*args, **kwargs)                    
        return wrapper
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PARALLEL BACKEND
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Parallel:
    """
    The central Factory for parallel processing.
    
    Provides the `get_backend` method to dynamically select the 
    appropriate backend (`multi` or `ray` or None) based on the current user environment.
    """

    @staticmethod
    def get_backend(backend: Literal['ray', 'multi', None] = 'ray'):
        if backend == 'multi':  return ParallelMulti
        if backend == 'ray':    return ParallelRay
        if backend == None:     return ParallelOff
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬