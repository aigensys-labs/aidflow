"""
Deterministic sampling engine for robust data splitting and cross-validation orchestration.

This module provides a unified framework for managing all stochastic operations 
within the AIDflow pipeline, including stratified k-fold splits, Monte Carlo 
cross-validation, bootstrap resampling, and chronological time-series partitioning. 
By centralizing random state management and sampling logic, it ensures absolute 
reproducibility of training and validation sets across different execution environments.
"""

from ...deps.common     import  wraps
from ...deps.data       import  np, pSe, pDF
from ...deps.struct     import  dataclass
from ...deps.types      import  (
                                Union,
                                Literal,
                                Iterable,
                                NDArray, 
                                ArrayLike,
                                GeneratorType
                                )
from ...core.accessor   import  Rgb, Format
from ...core.standard   import  Scalar
from ...core.interop    import  Pd
from ...utils.builtins  import  List


type SampleBase     = pDF | pSe
type SampleArray    = NDArray | ArrayLike
type SampleType     = SampleBase | SampleArray | dict | list | tuple
type SampleSet      = list[ SampleBase | SampleArray ]
type SampleGroup    = SampleSet | list[ SampleSet ]
type SampleData     = SampleBase | SampleArray | SampleGroup

type SampleIxBase   = pSe | ArrayLike | range[int] | list[int] | tuple[int]
type SampleIxSet    = list[ SampleIxBase ] 
type SampleIxGroup  = SampleIxSet | list[ SampleIxSet ] 
type SampleIndex    = SampleIxBase | SampleIxGroup



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SAMPLING BASE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Bootstrap:
    """
    Helper class for statistical bootstrap resampling strategy.
    
    Generates multiple training sets where samples are drawn with replacement,
    and the corresponding out-of-bag (OOB) samples form the test set.

    Parameters
    ----------
    n_splits : int, default 200.
        The number of bootstrap iterations (splits) to generate. 

    random_state : int, optional, default 42.
        Seed for the random number generator to ensure reproducibility.
    
    Returns
    ------
    Initializes the internal RNG and stores the split count.    
    """

    def __init__(self, n_splits=200, random_state=42):
        self.n_splits = n_splits
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def _test_ix_fast(n:int, train_idx:np.ndarray) -> np.ndarray:
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[train_idx] = False
        return np.where(oob_mask)[0]
    
    def split(self, X, y=None):
        """
        Generator that yields train and test indices for bootstrap resampling.
        
        For each iteration:
        1. Generates a training set of size `n` by sampling with replacement.
        2. Identifies the test set as indices NOT present in the training set (Out-of-Bag).
        
        Parameters
        ----------
        X : array-like
            The input dataset. The length determines the sample size `n`.
        y : array-like, optional
            Target labels (ignored in index generation, but included for API consistency).
        
        Yields
        ------
        train_idx : ndarray
            Indices for the bootstrap training set (size n, with duplicates).
        test_idx : ndarray
            Indices for the out-of-bag test set (indices not in train_idx).
        """        
        n = len(X)
        for _ in range(self.n_splits):
            train_idx = self.rng.integers(0, n, size=n)
            test_idx  = self._test_ix_fast(n, train_idx)
            yield train_idx, test_idx
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Traverse:
    
    @staticmethod
    def _sampleable(data: SampleData | SampleIndex | GeneratorType) -> SampleData | SampleIndex:
        if isinstance(data, GeneratorType):
            return tuple(data)
        return data

    @classmethod
    def uni(cls, func):
        """Travers provided structure and applies target function to each df."""
        @wraps(func)
        def wrapper(self, data: SampleData | SampleIndex, *args, **kwargs):
            data = cls._sampleable(data)

            if not List.is_iterable(data):
                return func(self, data, *args, **kwargs)        
            
            return type(data)(wrapper(self, d, *args, **kwargs) for d in data)            
        
        return wrapper
    
    @classmethod
    def set(cls, func):
        """Travers provided structure and applies target function on final sets (sibling groups)."""
        @wraps(func)
        def wrapper(self, data: SampleData | SampleIndex, *args, **kwargs):
            data = cls._sampleable(data)
            
            if (
                not List.is_iterable(data) or
                not List.is_iterable(data[0])
            ):
                return func(self, data, *args, **kwargs)  
              
            return type(data)(wrapper(self, d, *args, **kwargs) for d in data)
        
        return wrapper

    @staticmethod
    def _get_data_from_set(data:SampleBase | SampleSet) -> SampleBase:
        """
        1. If data is a single data type -> returns data
        2. If [leaf data set] -> provides the first data from a leaf as obs size representative.
        """
        if List.is_iterable(data):
            return data[0]
        return data    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SamplingBy:
    """
    The key utility class for slicing and indexing data structures (NumPy and Pandas objects).
    Handles logic for both single and multiple index sets across single or grouped datasets.
    
    In theory, "infinite" number of data sets and ix sets can be provided → Traverse will 
    return properly created objects.
    """

    @staticmethod
    def _array_subset(d:SampleArray, ix:ArrayLike) -> NDArray: 
        return d[ix]

    @staticmethod
    def _index_subset(d:SampleBase, ix:SampleIxBase) -> SampleBase: 
        return d.loc[ix]

    @classmethod
    def _indicies_subset(cls, d:SampleBase | SampleArray, ix:SampleIxBase) -> SampleBase | SampleArray: 
        if not hasattr(d, 'iloc'):
            return cls._array_subset(d, ix)
        return d.iloc[ix]

    @staticmethod
    def _is_multi_ix(ix:SampleIxBase | SampleIxSet) -> bool:
        return (
            ( not isinstance(ix, (pSe,pDF)) ) 
            and
            ( List.is_iterable(ix) and not Scalar.is_scalar(ix[0]) )  
        )

    @classmethod
    def _multi_ix_worker(cls, 
        ix:   SampleIxBase | SampleIxSet, 
        data: SampleBase | SampleSet, 
        fx:   Union[_index_subset, _indicies_subset, _array_subset]
    ) -> SampleData:
        """
        Core worker method that executes the slicing logic based on data and index structure.
        
        Handles four scenarios:
        1. Single data + Single index: Returns single slice.
        2. Single data + Multi index: Returns list of slices (one per index set).
        3. Grouped data + Single index: Returns list of slices (one per dataset).
        4. Grouped data + Multi index: Returns nested list of slices (list of lists).
        
        Parameters
        ----------
        ix : The index or set of indices.
        data : The input data (single or grouped).
        fx : The slicing function to apply (_index_subset, etc.).
        
        Returns
        ------
        SampleData: The sliced result, structured as a single object or list(s) of objects.
        """                 
        is_multi_ix = cls._is_multi_ix(ix)
        
        if List.is_iterable(data):
            if not is_multi_ix:
                return [fx(d, ix) for d in data]
            return [[fx(d, i) for d in data] for i in ix]
        
        if is_multi_ix:
            return [fx(data, i) for i in ix]
        return fx(data, ix)
            
    @classmethod
    @Traverse.set
    def _multi_ix(cls, 
        ix:   SampleIndex, 
        data: SampleBase | SampleArray | SampleSet, 
        fx:   Union[_index_subset, _indicies_subset, _array_subset]
    ) -> SampleData:
        """
        Decorated wrapper for `_multi_ix_worker` to handle traversal logic.
        
        Parameters
        ----------
        ix : The index input.
        data : The input data structure.
        fx : The specific slicing function to apply.
        
        Returns
        ------
        The result of the `_multi_ix_worker` execution.
        """
        return cls._multi_ix_worker(ix, data, fx)

    @classmethod
    @Traverse.set
    def index(cls, data:SampleData, ix:SampleIndex) -> SampleData:
        """           
        Selects rows from data using label-based indexing (`.loc`).

        Handles four scenarios:
        1. Single data + Single index: Returns single slice.
        2. Single data + Multi index: Returns list of slices (one per index set).
        3. Grouped data + Single index: Returns list of slices (one per dataset).
        4. Grouped data + Multi index: Returns nested list of slices (list of lists).
        
        Parameters
        ----------
        ix : The index or set of indices.
        data : The input data (single or grouped).
        
        Returns
        ------
        SampleData: The sliced result, structured as a single object or list(s) of objects.
        """         
        return cls._multi_ix(ix, data, cls._index_subset)

    @classmethod
    @Traverse.set
    def indicies(cls, data:SampleData, ix:SampleIndex) -> SampleData:
        """           
        Selects rows from data using integer-position indexing (`.iloc` or array slicing).

        Handles four scenarios:
        1. Single data + Single index: Returns single slice.
        2. Single data + Multi index: Returns list of slices (one per index set).
        3. Grouped data + Single index: Returns list of slices (one per dataset).
        4. Grouped data + Multi index: Returns nested list of slices (list of lists).
        
        Parameters
        ----------
        ix : The index or set of indices.
        data : The input data (single or grouped).
        
        Returns
        ------
        SampleData: The sliced result, structured as a single object or list(s) of objects.
        """           
        return cls._multi_ix(ix, data, cls._indicies_subset)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeAdapter:
    """Features Helper to provide compatible type standard for Traverse and Sampling design."""

    CONVERT = List.ITERABLES

    def __init__(self, fe_list:tuple[SampleType]):
        self.fe_list_type = self.convert(fe_list)

    def __iter__(self) -> int:
        return iter(self.fe_list_type)
    
    @classmethod
    def _verify_type(cls, fes:SampleType) -> None:
        if isinstance(fes, Iterable):
            return
        raise TypeError(Rgb.bold(
            '\n ► PLEASE PROVIDE FEATURES VECTOR IN FORM OF:'
            ' pd.Series | np.array | dict | list | tuple | set'
        ))        
    
    @classmethod
    def convert(cls, fe_list:tuple[SampleType]) -> dict[int, dict[str, SampleBase | SampleArray]]:
        """Converts input Features to compatible processing-sampling type."""
        fe_list_std = {}
        for i, fes in enumerate(fe_list):
            t = type(fes)
            cls._verify_type(fes)
            if isinstance(fes, cls.CONVERT):
                if isinstance(fes, set): fes = list(fes)
                fes = pSe(fes)
            fe_list_std[i] = {'fes':fes, 'type':t}
        return fe_list_std

    def get_fes(self, i:int) -> SampleBase | SampleArray:
        """Returns standardized input Features."""
        return self.fe_list_type[i]['fes']

    def get_type(self, i:int) -> SampleType:
        """Returns original input Features."""
        return self.fe_list_type[i]['type']    

    def update_type(self, i:int, fe_sample: SampleBase | SampleArray) -> SampleType:
        """Update Sampled Features to orignal input type."""
        type_ = self.get_type(i)
        if type_ not in self.CONVERT:
            return fe_sample
        return type_(fe_sample) 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@dataclass
class SamplingChoiceParams:
    sample_size:    int|float
    stratify:       pSe|np.ndarray = None
    positive_ratio: float = None
    random_state:   int  = 42
    shuffle:        bool = False

@dataclass
class SamplingParams(SamplingChoiceParams):
    return_index:   bool = False
    verbose:        bool = False
    info:           str  = None
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
        
class SamplingInfo:
    """
    Sampling split optional infotext when verbosity is requested.
    
    Parameters
    ----------
    sample : int
        The calculated number of samples selected.
    learning_size : int
        The total number of observations or variables in the original dataset.
    params : SamplingParams
        Configuration object containing info messages, verbosity settings, etc.        
    """
    
    def __init__(self, sample:int, learning_size:int, params:SamplingParams):
        self.sample = sample
        self.learning_size = learning_size
        self.prm = params  
    
    def _info(self, desc:str, extra:str=None) -> str:
        s = lambda x: str(x).strip() if x else ''
        j = lambda *x: ' '.join([s for s in x if s])
        desc = Rgb.bold( j(s(desc), s(extra)) )
        return j(self.prm.info, desc)

    def _sample(self, extra:str=None, sample:str=None) -> str:
        """
        Constructs the core sampling message components.
        
        Parameters
        ----------
        extra : str, optional
            Additional text to append to the header (e.g., 'CV 5x').
        sample : str, optional
            Override the default sample number string.
            
        Returns
        ------
        tuple: A tuple containing (header_string, context_string).
        The context string is empty if the sample size equals the total size.
        """
        smp  = sample or self.sample        
        info = self._info('Data Sampling', extra)
        i1 = f'► {info} → {smp:,} '
        i3 = f' from {self.learning_size:,} Total observations'
        return i1, i3 if smp != self.learning_size else ''
    
    def data(self, extra:str=None) -> str:
        """
        Generates the summary string for general data sampling.
        
        Parameters
        ----------
        extra : str, optional
            Additional text to describe the specific sampling type.
            
        Returns
        -------
        str: A formatted string describing the random subsample size vs total size.
        """        
        i1, i3 = self._sample(extra)
        i2 = f'Randomly Sampled Observations' if i3 else 'Original Input Size'
        return i1 + i2 + i3        
        
    def cv(self, kind:str, n_splits:int) -> str:
        """
        Generates the summary string for Cross-Validation splits.
        
        Calculates approximate train/valid split sizes based on the total sample
        and the number of splits requested.
        
        Parameters
        ----------
        kind : str
            The type of cross-validation (e.g., 'KFold', 'Stratified').
        n_splits : int
            The number of folds to generate.
            
        Returns
        -------
        str: A formatted string showing the train/valid observation split per fold.
        """        
        valid = int(self.sample/n_splits)
        train = int(self.sample-valid)
        i1, _ = self._sample(f' {n_splits}x CV {kind.upper()} → TRAIN → {train} | VALID', valid)
        return i1

    def split(self, desc:str=None) -> str:
        """
        Generates the summary string for a Train/Validation split.
        
        Explicitly mentions the number of observations allocated to Training
        versus the remainder allocated to Validation.
        
        Parameters
        ----------
        desc : str, optional
            Custom description text to include in the header.
            
        Returns
        -------
        str: A formatted string detailing the train/validation observation split.
        """        
        i1, i3 = self._sample(desc)
        i2 = f'obs split to Train and {self.learning_size-self.sample:,} to Validation part'
        return i1 + i2 + i3  
    
    @staticmethod
    def data_note() -> None:
        """
        Generates the summary string for a Train/Validation split.
        
        Explicitly mentions the number of observations allocated to Training
        versus the remainder allocated to Validation.
        
        Parameters
        ----------
        desc : str, optional
            Custom description text to include in the header.
            
        Returns
        -------
        str: A formatted string detailing the train/validation observation split.
        """        
        Format.display(Rgb.bold_red(
            'WARNING: Invalid setting in data sampling'
            '\n ► return_index=True is ambiguous with reset_index=True'
            '\n ► returning sampled data index object → reset_index=True is ignored'
        ))

    def fe(self) -> str:
        """
        Generates the summary string for Feature (Variable) Sampling.
        
        Parameters
        ----------
        None (uses self.sample and self.learning_size).
            
        Returns
        -------
        str: A formatted string describing the number of explanatory variables selected.
        """        
        info = self._info('Vars Sampling')
        m = ''
        if self.sample != self.learning_size:
            m = f' randomly chosen from {self.learning_size:,} Total Variables'
        return (
            f'► {info} → {self.sample:,} Explanatory Variables' + m
        )
    
    def fe_note(self) -> None:
        """
        Prints a note regarding Feature Sampling on Series.
        
        Alerts the user that if `return_index=True` is used with a Pandas Series,
        the output will be the sampled indices of the original series, not the values.
        
        Only prints if `verbose` is enabled in the params.
        """        
        if not self.prm.verbose: 
            return
        Format.display(Rgb.bold(
            'NOTE: Feature sampling on Series with return_index=True '
            'returned vector is sampled index of original series.' 
        ))

    def no_obs(self, sample:int=None) -> None:
        """
        Prints an error message if there are no variables to sample from.
        
        Checks if `learning_size` is zero and prints a bold red warning if so.
        
        Returns
        -------
        str: The error message string (even if printed).
        """
        m = lambda x: Rgb.bold_red(f'{x} HAS 0 SIZE')
        if not self.learning_size: Format.display(m('INPUT DATA'))
        if not sample: Format.display(m('RETURNING INPUT DATA AS SAMPLE'))
        return m
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SamplingChoice:
    """
    Logic for determining sample sizes and generating indices.
    
    This class handles the calculation of how many samples to take based on 
    user configuration (absolute count or percentage) and generates the 
    actual numpy index arrays for training and validation sets.
    
    Parameters
    ----------
    params (SamplingChoiceParams) : Configuration object containing parameters 
        sample_size (int), random_state (int), and shuffle (bool) flags.
    """    
    def __init__(self, params:SamplingChoiceParams):
        self.prm = params

    def sampling_size(self, X:SampleType) -> tuple[int, int]:  
        """
        Calculates the target sample size based on input data length and config.
        
        Handles three cases:
        1. `sample_size == 1`: Take all data (100%).
        2. `sample_size < 1`: Treat as percentage (e.g., 0.5 = 50%).
        3. `sample_size > 1`: Treat as absolute count (capped at data length).        
    
        Parameters
        ----------
        X : The input dataset (DataFrame, Series, list, etc.).
            
        Returns
        -------
        A tuple of (calculated_sample_size, original_learning_size).
        """          

        learning_size = len(X)
        sampling_size = self.prm.sample_size

        if sampling_size == 1:
            sample = learning_size
        
        elif sampling_size < 1:
            sample = int(learning_size*sampling_size)
        
        elif sampling_size > 1:
            sample = min(sampling_size, learning_size)

        return int(sample), learning_size
    
    @staticmethod
    def _verify_size(data:SampleType) -> None:
        """Verifies group set consistency length, i.e. X-y obs match."""
        if not List.is_iterable(data):
            return
        if len(set(len(df) for df in data)) > 1:
            raise ValueError(
                "AIG SAFETY ► When [group of data] is provided"
                ", same obs size within the leaf is mandatory."
            )
    
    def _size(self, data:SampleType) -> tuple[int, int]:
        d = Traverse._get_data_from_set(data)
        return self.sampling_size(d)
    
    @Traverse.set
    def size(self, X:SampleType) -> tuple[int, int]:  
        """
        Calculates sampling and learning sizes.
        
        Performs safety checks on grouped data before calculating sizes.
        
        Parameters
        ----------
        X : Input data or tuple of data (e.g., (X, y)).
            
        Returns
        -------
        A tuple of (sample_size, learning_size).
        """        
        self._verify_size(X)
        return self._size(X)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_rng(self, random:int=None):
        rnd = random if random is not None else self.prm.random_state
        return np.random.default_rng(rnd)
    
    def _sample(self, size:tuple[int, int|ArrayLike], random:int=None) -> np.ndarray[int]:
        """Generates a random index array based on sample and learning sizes."""
        sample_size, learning_size = size
        rng = self._get_rng(random)
        return rng.choice(learning_size, size=sample_size, replace=False)

    def _sample_order(self, smpl: np.ndarray[int]) -> np.ndarray[int]:
        if not self.prm.shuffle:
            return np.sort(smpl)
        return smpl

    @staticmethod
    def _sample_folds(sample:np.ndarray[int], fold_sizes:tuple[int]) -> list[np.ndarray[int]]:
        """Splits sample indicies to list of sample indicies given fold_sizes."""
        folds = []
        i = 0
        for s in fold_sizes:
            i_1 = i + s
            folds.append(sample[i:i_1])
            i = i_1
        return folds

    @staticmethod
    def _strat_join(sample:list[np.ndarray[int]]) -> np.ndarray[int]:
        return np.concatenate(sample)

    def _strat_join_order(self, sample:list[np.ndarray[int]]) -> np.ndarray[int]:
        smp = self._strat_join(sample)
        if not self.prm.shuffle:
            return self._sample_order(smp)
        rng = self._get_rng()
        return rng.permutation(smp)
    
    @staticmethod
    def _strat_get_fold_sizes(sample:np.ndarray[int], n_folds:int) -> tuple[int]:
        l  = len(sample)
        fs = np.full(n_folds, l // n_folds, dtype=int)
        fs[:l % n_folds] += 1
        return tuple(fs)
    
    def _strat_fold(self, sample:list[np.ndarray[int]], n_folds:int) -> np.ndarray[int]:
        """Generates sample indicies by folds given n_folds."""
        f_smpl  = []        
        for smp in sample:
            fs = self._strat_get_fold_sizes(smp, n_folds)
            f_smpl.append(self._sample_folds(smp, fs))

        folds = []    
        for fold in zip(*f_smpl):
            fs = self._strat_join_order(fold)
            folds.append(fs)

        return self._strat_join(folds)
    
    def _balance_sample_size(self, c:NDArray, sample_size:int) -> NDArray:
        """Balances sample with respect to the positive_ratio value."""
        i = c.argmin()

        d = c[i] / c.sum() or 1e-2
        r = np.clip(self.prm.positive_ratio, d, 0.5)
        s = int(sample_size * r)

        if s < c[i]:
            c[i] = s

        x = c.argmax()
        c[x] = sample_size - c[i]

        return c

    @staticmethod
    def _strat_sample_size(c:NDArray, sample_size:int, learning_size:int) -> NDArray:
        """Stratified sample size by target values."""
        r = c / learning_size
        c = ( r * sample_size ).astype(int)
        
        i = c.argmin()
        x = c.argmax()
        d = int(sample_size - c.sum())

        if d: # rounding error ~ 1obs diff might occur
            if r[x] * learning_size > c[x]:
                c[x] = c[x] + d
                
            elif r[i] * learning_size > c[i]:
                c[i] = c[i] + d

        return c

    def _stratified_balance(self, size: tuple[int, int]):
        """Stratifies, and optionally balances, sample by target values."""
        se  = self.prm.stratify
        v,c = np.unique(se, return_counts=True)

        sample_size, learning_size = size
        
        if self.prm.positive_ratio:
            c = self._balance_sample_size(c, sample_size)
        else:
            c = self._strat_sample_size(c, sample_size, learning_size)

        return v, c
    
    def _stratified_sample(self, 
        size: tuple[int, int], 
        random:  int = None, 
        n_folds: int = None,
    ) -> np.ndarray[int]:
        """
        Generates a random index array based on the calculated size by stratified series 
        unique values. 

        Optionally, stratified sampling is executed by folds when n_folds is provided.
        """
        se  = self.prm.stratify
        v,c = self._stratified_balance(size)
        
        smpl = []
        for v_, c_ in zip(v,c):
            idx = np.where(se==v_)[0]
            smpl.append( self._sample((c_, idx), random) )

        if n_folds is not None:
            return self._strat_fold(smpl, n_folds)

        return self._strat_join_order(smpl)

    @Traverse.set
    def sample(self, size: tuple[int, int], random:int=None, n_folds:int=None
    ) -> np.ndarray[int]:
        """
        Generates a random index array based on the calculated size.
        
        Creates a NumPy array of unique indices representing the selected rows.
        Respects the `shuffle` configuration: if False, indices are sorted.
        
        Parameters
        ----------
        size: Tuple of (target_sample_count, total_population_size).
        random: Optional seed for the random number generator. 
                If None, uses the global `random_state` from params.
        n_folds: Optional number of a folds to distribute stratified sample.
            
        Returns
        -------
        A NumPy array of integers (indices) for the selected sample.
        """
        if self.prm.stratify is not None:
            return self._stratified_sample(size, random, n_folds)
        
        sample = self._sample(size, random)
        return self._sample_order(sample)
    
    @staticmethod
    def _complement_sample(X: SampleType, train_ix:list) -> np.ndarray[int]:
        d = Traverse._get_data_from_set(X)
        return np.setdiff1d(np.arange(len(d)), train_ix)

    @staticmethod
    def complement_sample(X: SampleType, sample_ix:list) -> np.ndarray[int]:
        """
        Efficiently calculates complement indices (100x faster than setdiff1d).        
        Uses boolean masking to identify indices not present in the input sample.
        
        Parameters
        ----------
        X : Input data.
        sample_ix : List or array of indices of a sampled data.
            
        Returns
        -------
        A NumPy array of integers representing the residual set indices.
        """
        d = Traverse._get_data_from_set(X)
        mask = np.ones(len(d), dtype=bool)
        mask[sample_ix] = False
        return np.where(mask)[0]
    # ─────────────────────────────────────────────────────────────────────────────────────────────
   
    @Traverse.set
    def _is_sampling(self, size: tuple[int, int]) -> bool:
        sample, learning_size = size
        return ( learning_size ) and ( sample )
    
    def is_sampling(self, size:tuple) -> bool | tuple[bool]:
        """
        Checks if sampling is active for one or more datasets.
        
        Parameters
        ----------
        size: Tuple of (target_sample_count, total_population_size).        
        """
        is_smp = self._is_sampling(size)
        if List.is_iterable(is_smp) and not sum(is_smp): 
            return False
        return is_smp
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SamplingCV:
    """
    Handles various sampling strategies for Cross-Validation (CV) and random splits.
    
    This class provides methods to generate stratified or random train/validation 
    folds, supporting both sequential splitting and random shuffling (Monte Carlo CV).
    
    Attributes:
        n_splits (int)           : Number of CV splits → folds or monte carlo samples.
        sample_size (int|float)  : Used for random MC only → controls validation size of each round.
        stratify (pSe | NDArray) : Series on which basis stratified sampling will be conducted.        
        random_state (int)       : Random seed for reproducibility.
        shuffle (bool)           : Whether to shuffle the data before splitting.
    """
    def __init__(self, 
        n_splits:       int = 3, 
        sample_size:    int|float = 0.33,
        stratify:       pSe|ArrayLike = None,
        random_state:   int = 42, 
        shuffle:        bool = False
    ):
        self.ncv      = n_splits
        self.size     = sample_size
        self.stratify = stratify
        self.random   = random_state
        self.shuffle  = shuffle
    
    def _verify_sample_size(self):        
        if self.size > 0 and self.size != 1:
            return
        raise ValueError(
            'AIG SAFETY ► Monte Carlo CV requires "sample_size" parameter'
            ' to determine validation sample size either as:'
            '\n ► float, where sample_size > 0 and sample_size < 1 or'
            '\n ► int to set validation sample size directly'
        )
        
    def _get_dist_valid_sizes(self, X:SampleType | SampleSet) -> tuple[int]:
        """
        Calculates the size of the validation set for each fold.
        
        Distributes the total number of samples as evenly as possible across 
        the specified number of folds, assigning the remainder to the first folds.
        """
        d = Traverse._get_data_from_set(X)
        n_samples = len(d)

        fold_sizes = np.full(self.ncv, n_samples // self.ncv, dtype=int)
        fold_sizes[:n_samples % self.ncv] += 1
        
        return tuple(fold_sizes)
    
    @Traverse.set
    def _sample(self, X:SampleType, n_folds:int=None) -> SamplingChoice.sample:
        d = Traverse._get_data_from_set(X)

        if n_folds is not None: 
              sample_size = len(d)
        else: sample_size = self.size

        smp = SamplingChoice( SamplingChoiceParams(
                sample_size = sample_size, 
                stratify    = self.stratify,
                shuffle     = self.shuffle
        ))
        sl_size = smp.size(d)
        return lambda random: smp.sample(sl_size, random, n_folds)

    @Traverse.set
    def folds(self, X:SampleType | SampleSet) -> list[ tuple[np.ndarray, np.ndarray] ]:
        """
        Generates sequential train/validation index pairs for cross-validation.
        
        Splits the data into `n_splits` folds. If `shuffle` is enabled, the indices 
        are shuffled before being distributed into sequential chunks.
        
        Args:
            X: The input data or sample set.
            
        Returns:
            A list of tuples, where each tuple contains (train_indices, valid_indices).
        """
        d = Traverse._get_data_from_set(X)        
        fold_sizes = self._get_dist_valid_sizes(d)
        
        sample = self._sample(d, self.ncv)
        idx = sample(self.random)

        valid = SamplingChoice._sample_folds(idx, fold_sizes)
        train = [SamplingChoice.complement_sample(d, v) for v in valid]

        return list( zip(train, valid) )

    @Traverse.set
    def mc_splits(self, X:SampleType | SampleSet) -> list[ tuple[np.ndarray, np.ndarray] ]:
        """
        Generates random train/validation splits for cross-validation.
        Also known as Monte Carlo Cross-Validation (MCCV).
        
        Unlike `folds` which uses sequential chunks, this method randomly samples 
        a training set `n_splits` times, each time deriving the training set as the 
        complement of the validation set.
        
        Args:
            X: The input data or sample set.
            
        Returns:
            A list of tuples, where each tuple contains (train_indices, valid_indices).
        """
        sample = self._sample(X)
        valid  = [sample(self.random+i) for i in range(self.ncv)]
        return [(SamplingChoice.complement_sample(X, v), v) for v in valid]

    @Traverse.set
    def create_data_folds(self, X:SampleType | SampleSet) -> SampleGroup:
        """
        Creates a sample object by sequential CV folds.
        
        Generates `n_splits times` the index pairs using the `folds` method 
        and wraps them into a SampleGroup using SamplingBy indices.
        
        Args:
            X: The input data or sample set.
            
        Returns:
            A SampleGroup containing the data split into sequential folds.
        """
        folds = self.folds(X)
        return SamplingBy.indicies(X, folds)

    @Traverse.set
    def create_data_mc_splits(self, X:SampleType | SampleSet) -> SampleGroup:
        """
        Creates a sampled object by Monte Carlo random CV splits.
        
        Generates `n_splits times` the index pairs using the `sample_size` 
        and wraps them into a SampleGroup using SamplingBy indices.
        
        Args:
            X: The input data or sample set.
            
        Returns:
            A SampleGroup containing the data split into random splits.
        """
        self._verify_sample_size()
        splits = self.mc_splits(X)
        return SamplingBy.indicies(X, splits)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Sampling:
    """
    High-level interface for deterministic data sampling within the AIG flow.
    
    This class encapsulates the logic for selecting subsets of data, managing 
    cross-validation splits, and feature selection. It supports various sampling 
    strategies (fixed size, percentage), deterministic seeding, and optional 
    verbose logging of sampling statistics.
    
    Parameters
    -----------
    sample_size (int | float) : Target size for the sample. 
        - If int, exact count. If float (0.0-1.0), percentage of total data.
        - When Sampling is used with split methods, sample_size represents the 
        hold out value → i.e. train_valid_split → sample_size = valid size.
    
    stratify (pSe | NDArray) : Series on which basis stratified sampling will be conducted.        
        Providing stratify series balance sampling to a statistically 
        same ratio of 1s for ML training as in the full sample; unless 1s ratio specified 
        by `stratify_ratio`. Default None.
    
    stratify_ratio (float) : Custom ratio of 1s in stratified sampling. Maximum 0.5.
        Default None → follow the orig. sample ratio.

    return_index (bool) : If True, return the indices of the selected samples
        instead of the data itself. Default is False.

    random_state (int) : Seed for the random number generator to ensure 
        reproducibility. Default is 42.

    shuffle (bool) : If True, shuffle the data before sampling. Default is False.

    verbose (bool) : If True, return additional information strings alongside 
        the data. Default is False.

    info (str) : Optional custom information string to be included in output.
    """
    def __init__(self,
        sample_size:    int|float, 
        stratify:       pSe   = None,
        stratify_ratio: float = None,
        return_index:   bool  = False, 
        random_state:   int   = 42, 
        shuffle:        bool  = False, 
        verbose:        bool  = False,
        info:           str   = None
    ):
        self.prm = SamplingParams(
            sample_size    = sample_size,
            stratify       = stratify,
            positive_ratio = stratify_ratio,
            return_index   = return_index,
            random_state   = random_state,
            shuffle        = shuffle,
            verbose        = verbose,
            info           = info
        )
        self.smp = SamplingChoice(self.prm)
    # ─────────────────────────────────────────────────────────────────────────────────────────────         
    
    @Traverse.set
    def _sampling(self, data:SampleType | SampleData, return_size:bool=False
    ) -> tuple[list | int | None, SamplingInfo]:  
        """Sampling standard process."""           
        s_l  = self.smp.size(data)
        is_s = self.smp.is_sampling(s_l)
        info = SamplingInfo(*s_l, self.prm)
        size = self._get_sample_size(s_l)
        info.no_obs(size)
        if not is_s:
            return None, info
        if return_size:
            return size, info        
        return self.smp.sample(s_l), info
    # ─────────────────────────────────────────────────────────────────────────────────────────────      
    
    @Traverse.set
    def _get_sample_size(self, size:tuple) -> int:
        return size[0]

    @Traverse.uni
    def _reset_idx(self, data:pDF) -> pDF:
        return data.reset_index(drop=True)  
    
    @staticmethod
    def _get_data(data: SampleBase | SampleSet ) -> SampleBase | SampleSet:
        """Unpacks a single-element SampleSet into its underlying data."""
        if len(data) == 1: 
            return data[0]
        return data
    
    def _get_cv_splits(self, data:SampleBase | SampleSet, n_splits:int, stratify:pSe, kind='fold'
    ) -> SampleData:
        """Generates Cross-Validation splits using SamplingCV."""
        cv = SamplingCV(
            n_splits     = n_splits, 
            sample_size  = self.prm.sample_size,
            stratify     = stratify,
            shuffle      = self.prm.shuffle, 
            random_state = self.prm.random_state
        )
        if kind == 'fold':
            return cv.create_data_folds(data)
        return cv.create_data_mc_splits(data)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _data_return_index(self, data:SampleData, ix:SampleIndex) -> pDF|list[pDF]:
        if self.prm.return_index:
            return ix
        return SamplingBy.indicies(data, ix)

    def _data_reset_index(self, data:SampleData, reset_index:bool) -> SampleData:
        if not reset_index:
            return data
        if self.prm.return_index: 
            SamplingInfo.data_note()
            return data
        return self._reset_idx(data)
    
    def _data_process(self, data:SampleData, ix:SampleIndex, reset_index:bool) -> SampleData:
        """Orchestrates the sampling data processing steps."""
        data_ = self._data_return_index(data, ix)
        return self._data_reset_index(data_, reset_index)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @Traverse.set
    def _train_valid_split(self, data:SampleBase | SampleArray, reset_index:bool=False
    ) -> SampleSet | tuple[SampleSet, str]:      

        sample, info = self._sampling(data)        
        if sample is None: 
            return data if not self.prm.verbose else (data, info.no_obs())
        
        t_ix  = self.smp.complement_sample(data, sample)
        v_ix  = sample
        
        data_ = self._data_process(data, [t_ix, v_ix], reset_index)
        
        return data_ if not self.prm.verbose else ( data_, info.split() )

    def train_valid_split(self, *data:SampleData, reset_index:bool=False
    ) -> SampleData | tuple[SampleData, str]:
        """
        Performs a single train/validation split.        

        Training part as the main sample and the remainder as a validation part.
        Config value → `sample_size = validation size`.

        Parameters
        ----------
        *data       : Input data arguments.
        reset_index : Whether to reset indices in the output.
            
        Returns
        -------
        The train/validation split or (split, info).
        """
        return self._train_valid_split(self._get_data(data), reset_index)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @Traverse.set
    def _cv_splits(self, data:SampleBase | SampleArray, n_splits:int=3, reset_index:bool=False, 
                   kind: Literal['fold', 'random'] = 'fold'
    ) -> SampleSet | tuple[SampleSet, str]:      

        sample, info = self._sampling(data)
        if sample is None: 
            return data if not self.prm.verbose else (data, info.no_obs())
        
        if kind == 'random':
            sample = range(len(Traverse._get_data_from_set(data)))

        stratify = self.prm.stratify
        if kind == 'fold' and self.prm.stratify is not None:
            stratify = self._data_process(self.prm.stratify, sample, reset_index)
        
        data_ = self._data_process(data, sample, reset_index)        
        data_ = self._get_cv_splits(data_, n_splits, stratify, kind)

        info_ = lambda: info.data('PRIOR') +' '+ info.cv(kind, n_splits)

        return data_ if not self.prm.verbose else ( data_, info_() )

    def cv_folds(self, *data:SampleData, n_splits:int=3, reset_index:bool=False
    ) -> SampleData | tuple[SampleData, str]:
        """
        Generates sequential non-overlapping cross-validation folds (k-folds).
        
        1. Input data
            - `sample_size < 1` → optionally subsample the input data prior the next step
            - `sample_size = 1` → the orig. input data are used for the next step
        2. Original or subsampled set from (1) is used to create 
        `n_splits-times non-overlapping sequential CV folds`.
        
        Parameters
        ----------
        *data       : Input data arguments.
        n_splits    : Number of folds.
        reset_index : Whether to reset indices in the output.
            
        Returns
        -------
        The folds or (folds, info).
        """
        return self._cv_splits(self._get_data(data), n_splits, reset_index, kind='fold')

    def cv_splits(self, *data:SampleData, n_splits:int=3, reset_index:bool=False
    ) -> SampleData | tuple[SampleData, str]:
        """
        Monte Carlo Cross-Validation (Repeated Random Sub-sampling).
        Generates n_splits of (train, valid) indices without replacement.        
        
        1. `sample_size` config value determines validation size of each train-valid round. 
        2. `n_splits` controls number of generated MC samples (MC rounds). 
        
        Parameters
        ----------
        *data       : Input data arguments.
        n_splits    : Number of MC samples.
        reset_index : Whether to reset indices in the output.
            
        Returns
        -------
        The random splits or (splits, info).
        """
        return self._cv_splits(self._get_data(data), n_splits, reset_index, kind='random')      
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @Traverse.uni
    def _subsample(self, data:SampleBase | SampleArray, reset_index:bool=False
    ) -> SampleBase | tuple[SampleBase, str]:      

        sample, info = self._sampling(data)
        if sample is None:
            return data if not self.prm.verbose else (data, info.no_obs())
        
        data_ = self._data_process(data, sample, reset_index)
        
        return data_ if not self.prm.verbose else ( data_, info.data() )

    def subsample(self, *data:SampleData, reset_index:bool=False
    ) -> SampleData | tuple[SampleData, str]:
        """
        Returns a subset of the data based on the configured sample_size.

        Parameters
        ----------
        *data: Input data.
        reset_index: Whether to reset indices.
            
        Returns
        -------
        The subsampled data or (data, info).
        """        
        return self._subsample(self._get_data(data), reset_index)    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @Traverse.uni
    def _time_split(self, data:SampleBase, time_var:str, reset_index:bool=False
    ) -> SampleSet | tuple[SampleSet, str]:
        
        size, info = self._sampling(data, return_size=True)        
        if size is None:
            return data if not self.prm.verbose else (data, info.no_obs())
        
        data_  = Pd.stable_sort(data, time_var)
        
        t_end = int( len(data_) - size )
        t_ix  = range(t_end)  
        v_ix  = range(t_end, len(data_))
        data_ = self._data_process(data_, [t_ix, v_ix], reset_index)
        
        return data_ if not self.prm.verbose else ( data_, info.split('BY TIME') )
    
    def time_split(self, *data:SampleData, time_var:str, reset_index:bool=False
    ) -> SampleData | tuple[SampleData, str]:
        """
        Splits data chronologically to train-test sample where the split is given by 
        the config → `sample_size = test size`.

        Parameters
        ----------
        *data: Input data.
        time_var: Name of the time column.
        reset_index: Whether to reset indices.
            
        Returns
        -------
        The time split or (split, info).
        """
        return self._time_split(self._get_data(data), time_var, reset_index)      
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @Traverse.uni
    def _features(self, i:int, fe_adapter:FeAdapter) -> SampleType | tuple[SampleType, str]:
        
        fe_list = fe_adapter.get_fes(i)
        sample, info = self._sampling(fe_list)
        
        if sample is None: 
            return fe_list if not self.prm.verbose else (fe_list, info.no_obs())
        
        fe_list_: SampleBase | SampleArray = SamplingBy.indicies(fe_list, sample)
    
        if self.prm.return_index and fe_adapter.get_type(i) in (pSe, pDF):
            fe_list_ = fe_list_.index
            info.fe_note()
        
        fe_list_ = fe_adapter.update_type(i, fe_list_)
        
        return fe_list_ if not self.prm.verbose else ( fe_list_, info.fe() )
    
    def features(self, *fe_list:SampleData) -> SampleData | tuple[SampleData, str]:
        """
        Subsamples a list of features.
        
        Selects a subset of features (columns) based on the configured sample size.
        Supports pandas Series/DataFrames, numpy Arrays, dicts, lists, tuples.
        
        Parameters
        ----------
        fe_list: The list of features to sample from.
            
        Returns
        -------
        The sampled features or (features, info).
        """
        fea = FeAdapter(fe_list)
        return self._get_data( self._features(list(fea), fea) )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬