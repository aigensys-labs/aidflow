"""
ML dataset management utilities for cross-validation orchestration and data alignment.

This module provides high-performance helpers for preparing datasets during the 
model optimization phase, focusing on the creation and management of K-Fold 
cross-validation samples. It ensures strict data integrity through standardized 
alignment routines that handle both pandas DataFrames and numpy arrays, allowing 
for seamless transitions between feature subsets and internal ML backend requirements.

Further, the library implements a "shadowing" framework used to generate 
synthetic datasets by independently shuffling columns. This mechanism is critical 
for establishing baseline performance metrics and verifying the significance of 
feature selections, providing an automated way to detect overfit or spurious 
correlations within the pipeline's data flow.
"""

from ...deps.data           import  np, pDF, pSe
from ...deps.types          import  (
                                    Any, 
                                    Literal,
                                    NDArray, 
                                    FuncType
                                    )
from ...core.foundation     import  (
                                    AigParam, 
                                    AigStop
                                    )
from ...core.interop        import  Pd
from ...core.accessor       import  Format

from .sampling              import  Sampling
from .subsampling           import  Subsampling
                                    

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ML DATA PROCESSING HELPERS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class DatasetCV:
    """
    Dataset helper to store data and CV samples during ML optim.

    Parameters
    ----------
    cv : int = 3
        Number of Cross-Validation data splits.

    random_state : int = 42
        Random state of sampling and shuffle.

    subsample : int = None
        Providing int value will result to prior subsampling of input *X* and *y*
        in a balanced fashion; closer desc in the *Subsample chapter* below.
    
    shuffle : bool = False
        Shuffle data by random factor.

    add_constant_to_X : bool = False
        - when `add_constant_to_X = True` → X is extended by "const" column of 1s
        - when `cv > 1` → instance stores CV fold samples in cv_samples

    Subsample
    ---------
    Constructs sample of at most `subsample` rows regardless of input size. 
    The sample is built as a balanced (50:50) subset drawn directly from 
    the full dataset.

    e.g. subsample = 20,000:
    
    - If total positives ≥ 10,000:
        sample = 10,000 positives + 10,000 negatives = 20,000 rows
    - If total positives < 10,000:
        sample = all positives + fill up to 20K by negatives
        (total = n_pos + n_neg → 20,000 rows)

    If the input has fewer than 20,000 rows, all observations are used
    (no subsampling).

    This balanced construction guarantees a maximum 50:50 ratio under
    all circumstances, maximising the number of positive events in the
    sample while maintaining class balance for numerical stability.

    Statistical Validity
    --------------------
    For the WOE-logit selector: balanced sampling shifts only the
    intercept; slope coefficients and their tests are exactly invariant
    to the class ratio. Feature selection is unbiased.

    For tree-based selectors (e.g. LightGBM): the split criterion is
    approximately invariant to class ratio — first-round split ranking
    is exactly invariant, subsequent rounds are near-invariant in
    practice. Feature selection results are effectively unchanged;
    importance magnitudes and tree structure may differ slightly.

    In both cases, balanced sampling is safe for feature selection.
    """

    def __init__(self, 
        cv:                  int = 3, 
        random_state:        int = 42,
        subsample:     int|float = None,
        shuffle:            bool = False,
        add_constant_to_X:  bool = False,
    ):
        self.cv = cv      
        self.random_state = random_state
        self.subsample    = subsample
        self.shuffle      = shuffle
        self.add_constant = add_constant_to_X

    @property
    def Logit(self):
        """Logit model to adjust constant (intercept) setup."""
        from ..models import Logit
        return Logit

    def balance_sample(self, X:pDF, y:pSe) -> tuple[pDF, pSe]:
        """
        Constructs sample of at most `subsample` rows regardless of input size. 
        The sample is built as a balanced (50:50) subset drawn directly from 
        the full dataset.
        """
        if not self.subsample or len(X) <= self.subsample:
            return X, y
        return Subsampling(
            X, y, random_state = AigParam.di(self.random_state, 42)
        ).glm(sample_size = self.subsample)
        
    def set_data(self, X:pDF, y:pSe|pDF) -> None:        
        """
        Sets X and y data to the instance.
         - when `add_constant_to_X = True` → X is extended by "const" column of 1s
         - when `cv > 1` → instance stores CV fold samples in cv_samples
         - y is always stored as numpy vector
         - X is always stored as DataFrame to manipulate with FEs subset during ops        
        """
        DataUtils.is_valid_df(X, (pDF,))
        if isinstance(y, pDF): 
            y = y.squeeze()

        X_, y_ = self.balance_sample(X, y)
            
        if isinstance(y_, pSe): 
            y_ = y_.to_numpy()
        if self.add_constant:
            X_:pDF = self.Logit._add_constant(X_)
        
        self.y = y_
        self.X = X_
        if self.is_cv:
            self.set_cv(self.X, self.y)
    
    def get_data(self, kind:Literal['X, y, cv']='cv') -> dict | pDF | pSe:
        if kind == 'cv' and self.is_cv:
            return self.cv_samples
        if kind == 'X':
            return self.X
        if kind =='y':
            return self.y

    def _get_X_subset(self, fes:list, X:pDF, add_const:bool=None):
        if not add_const:
            return X[fes]

        if not self.add_constant:
            raise ValueError(
                "AIG SAFETY ► constant doesn't exist in the data"
                " ► When constant is needed, DataSetCV(add_constant_to_X = True).set_data with()"
            )
        return X[[self.Logit.CONST]+fes]

    def get_X_subset(self, 
        fes:        list, 
        X:          pDF  = None, 
        to_numpy:   bool = True,
        add_const:  bool = None 
    ) -> pDF|NDArray:
        """
        Returns fes subset of X.
         - X is None → defaults to self.X provided to set_data()
         - add_const → extend fes by 'const' created during set_data()
         - to_numpy  → returned data type is numpy NDArray instead of DataFrame
        """
        if X is None:
            X = self.X
        if add_const is None:
            add_const = self.add_constant
        X = self._get_X_subset(fes, X, add_const)
        if to_numpy:
            X = X.to_numpy()
        return X

    def set_cv(self, X:pDF, y:pSe) -> None:
        """
        Sets dict of KFold CV samples train-test.
         - each fold `i` → (X_train, y_train, X_valid, y_valid)
        """
        self.cv_samples = {}
       
        Xy_folds = Sampling(
            sample_size  = 1, 
            stratify     = y,
            random_state = self.random_state,
            shuffle      = self.shuffle
        ).cv_folds(X, y, n_splits = self.cv)
        
        for i, Xy in enumerate(Xy_folds):

            Xy_train, Xy_valid = Xy

            self.cv_samples[i] = dict(
                X_train = Xy_train[0], 
                y_train = Xy_train[1],
                X_valid = Xy_valid[0],  
                y_valid = Xy_valid[1]
            )
                
    def get_cv_range(self) -> range:
        return range(self.cv)
    
    def get_ith_cv_sample(self, i:int) -> dict:
        return self.cv_samples[i]
    
    def get_cv_sample(self, i_cv_sample:int):
        """
        Returns cv sample 
         - i_cv_sample - i-th CV sample from created number of cv samples
         - returned tuple - train-valid → (X_train, y_train, X_valid, y_valid)
        """
        sample = self.get_ith_cv_sample(i_cv_sample)
        return (
            sample['X_train'], sample['y_train'], 
            sample['X_valid'], sample['y_valid'], 
        )

    def get_cv_subset(self, i_cv_sample:int, fes:str) -> tuple:
        """
        Returns cv sample 
         - i_cv_sample - i-th CV sample from created number of cv samples
         - returned tuple - train-valid → (X_train, y_train, X_valid, y_valid)
        """
        sample = self.get_ith_cv_sample(i_cv_sample)
        return (
            self.get_X_subset(fes, sample['X_train']), sample['y_train'], 
            self.get_X_subset(fes, sample['X_valid']), sample['y_valid'], 
        )
    
    def get_cv_subset_by_custom_X(self, fx:FuncType, i_cv_sample:dict, fes:str) -> tuple:
        """
        Returns cv sample 
         - fx - representing custom get_X_subset()
         - i_cv_sample - i-th CV sample from created number of cv samples
         - returned tuple - train-valid → (X_train, y_train, X_valid, y_valid)
        """
        sample = self.get_ith_cv_sample(i_cv_sample)
        return (
            fx(fes, sample['X_train']), sample['y_train'],  
            fx(fes, sample['X_valid']), sample['y_valid']
        )

    @property
    def is_cv(self):
        return self.cv > 1
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
        
class DataUtils:
    """Dataset helper to provide standard functionality during ML optim."""
    
    FEATURE = 'feature'
    SHADOW  = 'shadow'

    @staticmethod
    def is_valid_df(X:Any, valid_type:tuple=(pDF, np.ndarray)) -> None:
        """Verifies accepted data types."""
        if not isinstance(valid_type, tuple):
            raise TypeError("'valid_type' param must be tuple type.")
        if not isinstance(X, valid_type):
            m = {pDF: "pd.Dataframe", np.ndarray: "np.ndarray"}
            m = ' | '.join([x for t,x in m.items() if t in valid_type])
            raise AigStop.custom(
                f"AIG SAFETY ► Only {m} input is allowed due to optim search."
            )
                
    @classmethod
    def get_fes(cls, X:pDF, subset:list=None) -> list:
        """Verifies and returns list of features with respect to a subset."""
        cls.is_valid_df(X, (pDF,))
        fes = list(X.columns)
        if subset is not None:
            fes = [x for x in fes if x in subset]
        if not len(fes):
            raise AigStop.custom(
                'AIG SAFETY ► NO FEATURES AVAILABLE / SELECTED '
                '► PLEASE ADJUST HYPERPARMS OR TUNE INPUT DATA.'
            )
        return fes

    @classmethod
    def align_data(cls, X:pDF|NDArray, verbose:bool=True) -> pDF:
        """
        Aligns data for internal ML processing.
        Returning data is a always a Dataframe with features depending on input data type.
         - pDF: actual data fes 
         - NDArray: dummy fe{i}.
        """
        cls.is_valid_df(X)
        if isinstance(X, np.ndarray):
            X = pDF(X, columns=[f'{cls.FEATURE}{i}' for i in range(X.shape[1])])
            if verbose: 
                print(
        'AIG SAFETY ► X is np.ndarray: Selection search will be processed.'
        '\n→ However, for the production safety please transform your X to Dataframe prior.'
        '\n→ Selectors objective is to identify significant features among space of FEs.'
        '\n→ Feature names ensure selected features are chosen correctly from the new datasets.'
                )
                Format.splitter()
        return X
    
    @classmethod
    def align_data_with_fes(cls, X:pDF|NDArray, subset:list=None, verbose:bool=True
    ) -> tuple[pDF, list]:
        """
        Aligns data and input features with respect to data type.
         - Returning data is a always a Dataframe type as a subset of fes: data[fes].
         - Returning fes are always a list[str] of either pDF: actual data fes | NDArray: dummy fe{i}.
        """
        X_ = cls.align_data(X, verbose)
        fe = cls.get_fes(X_, subset)
        return X_[fe], fe

    @classmethod
    def align_data_with_result(cls, X:pDF|NDArray, fes:list[str]
    ) -> tuple[pDF|NDArray, list[str|int]]:
        """
        Aligns data and result features with respect to data type.
         - Returning data is in orig. data type as a subset of fes: data[fes].
         - Returning fes are aligned with data type pDF: list[str] | NDArray: list[int].
        """
        if not fes:
            return X, fes
        if isinstance(X, pDF):
            return X[fes], fes
        type_ = [type(x) is str for x in fes] 
        if sum(type_) == len(type_):
            fes = [int(x[len(cls.FEATURE):]) for x in fes]
            return X[:,fes], fes
        if not sum(type_):
            return X[:,fes], fes
        if sum(type_) != len(type_):
            raise TypeError(
            'AIG SAFETY ► Mixed type of features → either int for np.ndarray or str for Dataframe.'
            )    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    @classmethod
    def _shadow_col(cls, c:str='', i:int=0) -> str:
        return f"{cls.SHADOW}_{('_')*i}{c}"
    
    @classmethod
    def _shadow_verify(cls, cols:list[str]) -> None:
        s = cls._shadow_col()
        v = lambda c: c[:len(s)]==s
        if len(cols) == sum([v(c) for c in cols]):
            cls.SHADOW = cls.SHADOW+'_'

    @classmethod
    def shadow_col(cls, c:str, cols:list[str]) -> str:       
        """Provides safe shadow name for a column c with respect to all data cols."""
        s = lambda i: cls._shadow_col(c,i)
        return next(
            s(i) for i in range(100) if s(i) not in cols
        )  
    
    @classmethod
    def create_shadow_cols(cls, cols:list[str]) -> dict:
        """Returns input cols mapping → {col:shadow for cols}."""
        cls._shadow_verify(cols)
        return {c: cls.shadow_col(c, cols) for c in cols}
        
    @classmethod
    def get_cols_with_shadow(cls, cols:list[str]) -> list:
        """Returns list of original columns extended by their shadow: [orig + shadow]."""
        c = cls.create_shadow_cols(cols)
        return list(c) + list(c.values())
    
    @classmethod
    def remove_shadow_cols(cls, cols:list[str]) -> list:
        """
        Returns original columns → removes shadow cols if present in cols.        
        
        NOTE: Valid only when orig. data and shadow data were created 
        by extend_by_shadow_df() without any column reposition during data usage.
        """
        s_ = cls._shadow_col()
        s  = sum([x.startswith(s_) for x in cols])
        if not s: 
            return cols
        c = len(cols)   # shadow df design → c = orig + shadow
        o = c - s       # shadow df design → o = cols - shadow
        if o > 0 and s / o >= 1:
            return cols[:o]
        return cols     # not shadow → some cols just had name as prefix

    @classmethod
    def create_shadow_df(cls, X:pDF|NDArray, random_state:int=42) -> pDF|NDArray:
        """Create a shadow dataset by shuffling each column independently."""
        is_pd = isinstance(X, pDF)
        is_np = isinstance(X, np.ndarray)
        
        if not any([is_pd, is_np]):
            raise ValueError("Input must be pandas DataFrame or numpy array")
        
        rng = np.random.default_rng(random_state)

        if is_pd:
            cols = X.columns
            c_s  = cls.create_shadow_cols(cols)            
            return pDF({
                c_s[c]: rng.permutation(X[c].values) for c in cols
                }, index=X.index
            )        
        
        shadow = X.copy()
        for i in range(shadow.shape[1]):
            rng.shuffle(shadow[:, i])
        return shadow     
    
    @classmethod
    def extend_by_shadow_df(cls, X:pDF|NDArray, random_state:int=42) -> pDF|NDArray:
        """Horizontally stacks the original data with its shadow."""
        shadow = cls.create_shadow_df(X, random_state)        
        if isinstance(X, pDF):
            return Pd.concat([X, shadow], axis=1)            
        return np.hstack([X, shadow])        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬