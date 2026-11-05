"""
Data Sampling and Temporal Train-Test Split.

This module implements the second ingestion stage: after raw data is
imported and standardized, it is partitioned into training and
out-of-time (OOT) samples via a **temporal** (chronological) split.

Pipeline Position
-----------------
Runs immediately after data import (`DataImporter`) and before
feature type inference. Its output is the train/test data pair that
all downstream stages (screening, encoding, modelling) operate on.

Temporal Split Principle
------------------------
The split is **chronological**, not random:
- **Train**: the earlier portion of the time window (for learning).
- **Test**: the later portion (out-of-time, for validation).

This mirrors production reality: the model is trained on past data
and must generalise to future data. Random splitting would leak
future information into training, producing optimistic metrics that
do not reflect production performance.

Optional Sampling (Prototyping)
-------------------------------
Both observation-level and feature-level subsampling are available
for memory-constrained prototyping. These are disabled by default
(fraction = 1.0) and only active when explicitly configured.
"""

from ...deps.data           import  np, pDF, pSe
from ...deps.types          import  Literal

from ...core.factory        import  (
                                    Dataclass,
                                    DictClass
                                    )
from ...core.interop        import  (
                                    Pd,
                                    DataStandard
                                    )
from ...core.accessor       import  Rgb

from ...ml.sampling         import  Sampling


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FX(DictClass):
    """
    Field name generators for stats output.

    Each lambda takes a split label ('train', 'test', 'total', 'data')
    and produces the corresponding stats dictionary key.
    """
    obs     = lambda x: f'obs_{x}'
    target  = lambda x: f'target_{x}'
    time_n  = lambda x: f'time_n_{x}'
    time_x  = lambda x: f'time_x_{x}'
    time_f  = lambda x: f'full_time_{x}'

class TT(DictClass):
    """
    Train/test split identifiers and pre-computed stats keys.

    Provides the canonical labels used throughout the module
    for referencing train and test data in stats output.
    """
    TT        = ['train', 'test']
    OBS_TRAIN = FX.obs(TT[0])
    OBS_TEST  = FX.obs(TT[1])
    TARGET_TR = FX.target(TT[0])
    TARGET_TE = FX.target(TT[1])
    TIME_N_TR = FX.time_n(TT[0])
    TIME_N_TE = FX.time_n(TT[1])
    TIME_X_TR = FX.time_x(TT[0])
    TIME_X_TE = FX.time_x(TT[1])

class LBL(TT):
    """
    Stats output labels: data composition and temporal dynamics.

    Used for user-facing display of the data split summary.
    Inherits train/test keys from `TT`.
    """
    OBS_T    = FX.obs('total')
    TARGET_T = FX.target('total')
    F_TIME_D = FX.time_f('data')
    F_TIME_U = FX.time_f('unique')
    F_TIME_S = FX.time_f('obs_start')
    F_TIME_E = FX.time_f('obs_end')
    D_SUBSET = FX.obs('data')
    VARS_CNT = 'variables_cnt'
    T_SUBSET = 'temporal_subset'
    TIME_N_D = FX.time_n('data')
    TIME_X_D = FX.time_x('data')
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → LOAD → TRANSFORM DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class DataProcParamsBase:
    """
    Data processing and sampling parameters.

    Attributes
    ----------
    import_start : float
        Start of the import window as a fraction of unique time periods
        (e.g., 0.0 = from the beginning, 0.1 = skip the first 10%).
        Used to exclude early unstable data if needed.
    import_end : float
        End of the import window as a fraction of unique time periods
        (e.g., 1.0 = to the end, 0.9 = exclude the last 10%).
        Used to exclude recent incomplete periods if needed.
    train_test : int | float
        Temporal train/test split ratio. If float (0 < x < 1), it is
        the fraction allocated to train (e.g., 0.8 = 80% train, 20% test).
        If int, it is the number of periods allocated to test.
    smp_train : int | float
        Observation sampling fraction for training data (1.0 = no sampling).
        Values < 1.0 enable stratified subsampling for prototyping.
    smp_feature : int | float
        Feature sampling fraction (1.0 = no sampling).
        Values < 1.0 enable random feature subsampling for prototyping.
    """
    import_start: float
    import_end:   float
    train_test:   int|float
    smp_train:    int|float
    smp_feature:  int|float

@Dataclass.base(update_specs=dict(slots=False))
class DataProcParamsCfg:
    """
    Pipeline configuration (references from the import stage).

    Attributes
    ----------
    target : str
        Target column name.
    time_var : str
        Standard pipeline time column name.
    flow_vars : list
        ID-like columns to preserve (excluded from feature counting).
    """
    target:     str
    time_var:   str
    flow_vars:  list    

@Dataclass.base
class DataProcParams(DataProcParamsBase, DataProcParamsCfg):
    """Transfomer params input design."""       

@Dataclass.base
class DataProcResult:
    """
    Output of the data processing stage.

    Attributes
    ----------
    train_data : pDF
        Training data (earlier temporal portion), sampled if configured.
    test_data : pDF
        Test/OOT data (later temporal portion), unsampled.
    vars_type : pSe
        Feature type mapping (post-sampling if feature sampling is active).
    """
    train_data: pDF
    test_data:  pDF
    vars_type:  pSe

@Dataclass.base
class DataProcStats:
    """
    Diagnostic statistics for the data split (user-facing display).

    Attributes
    ----------
    data_stats : dict
        Aggregated stats: observation counts, target rates, time ranges
        for input data, train, and test.
    train_sample : pDF
        A 100-row stratified sample of the training data for inspection.
    sampling_info : list
        Human-readable messages about active sampling (feature/observation).
    """
    data_stats:     dict
    train_sample:   pDF
    sampling_info:  list
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class ProcUtils:
    """
    Statistics and display utilities for the data processing stage.

    Computes and assembles the data split summary:
    - Observation counts (total, subset, train, test).
    - Target rates (positives per split).
    - Time ranges (min/max period per split).
    - A stratified sample view of the training data.
    """  

    def __init__(self, params:DataProcParams):
        self.prm = params
    
    def _vars_cnt(self, vars_type:pSe) -> int:
        """Count the number of feature variables (excluding flow vars and time)."""
        out = self.prm.flow_vars
        return sum([True for x in vars_type.index if x not in out])

    def _time_nx(self, data:pDF):
        """Return the time range (min, max) of a dataset."""        
        return (
            data[self.prm.time_var].min(),
            data[self.prm.time_var].max()
        )
    def _target_sum(self, data:pDF):
        """Count total target positives in a dataset."""        
        return data[self.prm.target].sum()
    
    def _input_data_stats(self, data:pDF) -> dict:
        """
        Compute stats for the full input data (pre-split).

        Returns
        -------
        dict
            Observation count, target sum, unique time periods,
            and the import window indices (start/end).
        """        
        obs    = len(data)
        target = self._target_sum(data)
        time_d = data[self.prm.time_var]
        time_u = np.unique(time_d)
        time_s = int(len(time_u)*self.prm.import_start)
        time_e = int(len(time_u)*self.prm.import_end)-1        
        return {
            LBL.OBS_T:    obs,
            LBL.TARGET_T: target,
            LBL.F_TIME_D: time_d,
            LBL.F_TIME_U: time_u,
            LBL.F_TIME_S: time_s,
            LBL.F_TIME_E: time_e,
        }
    
    def _data_stats(self, data:pDF, vars_type:pSe, time_se:tuple) -> dict:
        """
        Compute stats for the temporally-subset data.

        Returns
        -------
        dict
            Observation count, feature count, and time range.
        """        
        tnd, txd = self._time_nx(data)
        return {
            LBL.D_SUBSET: len(data),
            LBL.VARS_CNT: self._vars_cnt(vars_type),
            LBL.T_SUBSET: time_se,
            LBL.TIME_N_D: tnd,
            LBL.TIME_X_D: txd
        }    
         
    def _tt_stats(self, train:pDF, test:pDF) -> dict:
        """
        Compute train/test split stats.

        Returns
        -------
        dict
            Per-split: observation count, target sum, time range.
        """        
        tntr, txtr = self._time_nx(train)
        tnte, txte = self._time_nx(test)
        return {
            LBL.OBS_TRAIN: len(train),
            LBL.OBS_TEST:  len(test),
            LBL.TARGET_TR: self._target_sum(train),
            LBL.TARGET_TE: self._target_sum(test),
            LBL.TIME_N_TR: tntr,
            LBL.TIME_X_TR: txtr,
            LBL.TIME_N_TE: tnte,
            LBL.TIME_X_TE: txte,
        }    

    def _final_stats(self, stats:dict, train:pDF, fe_info:str, data_info:str):
        """
        Assemble the final DataProcStats output.

        Parameters
        ----------
        stats : dict
            Accumulated stats from all stages.
        train : pDF
            Final training data (for the sample view).
        fe_info : str
            Feature sampling message (empty if no sampling).
        data_info : str
            Observation sampling message (empty if no sampling).

        Returns
        -------
        DataProcStats
            Complete stats object for display.
        """        
        sample = self._make_view_sample(train)
        return DataProcStats(
            data_stats    = stats,
            train_sample  = sample,
            sampling_info = [fe_info] + [data_info]
        )     
    
    def _make_view_sample(self, data:pDF):
        """
        Generate a 100-row stratified sample for user inspection.

        The sample is sorted chronologically and stratified by target
        to show a representative view of the training data.
        """        
        return DataStandard.sort( 
            Sampling(100, stratify=data[self.prm.target]).subsample(data), 
            self.prm.time_var
        )  

    @staticmethod
    def _info(d:Literal['data','train','test']) -> str:
        """
        Generate a styled label for the data section in sampling info messages.

        Parameters
        ----------
        d : 'data' | 'train' | 'test'
            Section identifier.

        Returns
        -------
        str
            Styled label string.
        """        
        if d=='data':  return 'Input Data'
        if d=='train': return Rgb.bold_blue('TRAIN')
        if d=='test':  return Rgb.bold('TEST')    
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class DataProcessor:
    """
    Data Processor — Temporal Train/Test Split and Optional Sampling.

    Partitions the imported data into training and OOT (out-of-time)
    samples via a chronological split, with optional feature and
    observation subsampling for prototyping.

    Processing Order
    ----------------
    1. Compute input data stats (total observations, time range).
    2. Apply the import window (`import_start` → `import_end`).
    3. Split temporally into train (earlier) and test (later).
    4. Feature sampling (optional, for prototyping).
    5. Observation sampling (optional, for prototyping).
    6. Assemble final stats and result.

    Parameters
    ----------
    import_start : float
        Start fraction of the import window (0.0 = from beginning).
    import_end : float
        End fraction of the import window (1.0 = to end).
    train_test : int | float
        Split ratio (float) or test periods (int).
    smp_train : int | float
        Observation sampling fraction (1.0 = no sampling).
    smp_feature : int | float
        Feature sampling fraction (1.0 = no sampling).

    Attributes
    ----------
    stats : DataProcStats
        Diagnostic summary (available after `process`).

    Notes
    -----
    **Chronological split**: The train/test boundary is always
    temporal (earlier = train, later = test). This prevents
    information leakage and mirrors the production scenario where
    the model must predict future outcomes from past data.

    **Test is always unsampled**: Sampling applies only to training
    data. The test set is preserved in full for unbiased OOT
    evaluation.

    **Flow vars preserved**: ID-like columns are excluded from
    feature sampling and counting but always carried through.

    Examples
    --------
    >>> from aidflow.fitters import DataProcessor
    >>> proc = DataProcessor(
    ...     import_start=0.0,
    ...     import_end=1.0,
    ...     train_test=0.8,
    ...     smp_train=1.0,
    ...     smp_feature=1.0,
    ...     target='is_churn',
    ...     time_var='month',
    ...     new_time_var='AIG_YY_MM',
    ...     flow_vars=['entity_id'],
    ... )
    >>> result = proc.process(data)
    >>> len(result.train_data), len(result.test_data)  # doctest: +SKIP
    """
    
    stats: DataProcStats = None
    
    def __init__(self, params: DataProcParams):
        self.prm = params
    
    def _get_temporal_ix(self, stats:dict) -> tuple[str, str]:
        """
        Resolve the import window boundaries to actual time values.

        Converts the fractional import_start/import_end (computed in
        stats) to the corresponding time period values.        
        """
        t = stats[LBL.F_TIME_U]
        return t[stats[LBL.F_TIME_S]], t[stats[LBL.F_TIME_E]]
    
    def _temporal_subset(self, ts:pSe, stats:dict):
        """Resolve the import window to data indices."""
        s,e = self._get_temporal_ix(stats)
        st  = ts[ts==s].index.min()
        et  = ts[ts==e].index.max()
        return st,et
    
    def _data_temporal_subset(self, data:pDF, stats:dict) -> pDF:
        """Apply the import window to subset the data temporally."""
        s,e = self._temporal_subset(data[self.prm.time_var], stats)
        return data.loc[s:e,:].reset_index(drop=True), (s, e)
    
    def _vars_types(self, data:pDF) -> pSe:
        """Extract the dtype mapping from the data columns."""
        return data.dtypes
    
    def _set_remove_flow_vars_types(self, vars_type:pSe) -> pSe:
        """
        Separate flow variables from feature types.

        Flow variables (ID columns) are removed from the active
        feature list but preserved internally for re-attachment
        after sampling.        
        """        
        self._flow_vars_types = None
        if not self.prm.flow_vars:
            return vars_type    
        m = vars_type.index.isin(self.prm.flow_vars)
        self._flow_vars_types = vars_type[m]
        return vars_type[~m]
    
    def _add_flow_vars_types(self, vars_types:pSe) -> pSe:
        """Re-attach flow variable types after sampling (preserves them in the mapping)."""
        return Pd.concat([self._flow_vars_types, vars_types])
    
    def _fe_sampling(self, data:pDF, vars_type:pSe) -> tuple[pDF, pSe, str]:
        """
        Apply feature subsampling (optional, for prototyping).

        If `smp_feature` is 1.0, returns data unchanged.
        Otherwise, randomly samples a fraction of features (preserving
        flow vars and target) and trims the data accordingly.        
        """        
        if self.prm.smp_feature == 1:
            return data, vars_type, ''
        
        features, info = Sampling(
            self.prm.smp_feature, 
            verbose = True,
            info = ProcUtils._info('train')
        ).features(vars_type)

        if len(vars_type) != len(features):
            vars_type = features
            vars_all  = self.prm.flow_vars + list(vars_type.index)
            data = data[vars_all] 
        return data, vars_type, info

    def _train_sampling(self, data:pDF) -> tuple[pDF, str]:
        """
        Apply observation subsampling to training data (optional).

        If `smp_train` is 1.0, returns data unchanged.
        Otherwise, applies a stratified random subsample (preserving
        target class balance) and sorts chronologically.
        """
        if self.prm.smp_train == 1:
            return data, ''
        
        data_sample, info = Sampling(
            self.prm.smp_train,
            verbose = True, 
            info = ProcUtils._info('train')
        ).subsample(data, reset_index = True)
        
        if len(data) != len(data_sample):
            data = DataStandard.sort(data_sample, self.prm.time_var).reset_index(drop=True)
        return data, info

    def _train_test_split(self, data:pDF) -> tuple[pDF, pDF]:
        """
        Apply the temporal train/test split.

        Splits the data chronologically: earlier periods → train,
        later periods → test.        
        """
        return (
            Sampling(self.prm.train_test)
            .time_split(
                data, 
                time_var = self.prm.time_var, 
                reset_index = True
        ))

    def process(self, input_data:pDF) -> DataProcResult:
        """
        Execute the full data processing pipeline.

        Runs in order:
        1. Compute input stats.
        2. Separate flow vars from features.
        3. Apply import window (temporal subset).
        4. Temporal train/test split.
        5. Feature sampling (optional).
        6. Observation sampling (optional).
        7. Re-attach flow vars.
        8. Assemble final stats.

        Parameters
        ----------
        input_data : pDF
            Standardized data from `DataImporter`.

        Returns
        -------
        DataProcResult
            Train/test data pair and feature type mapping.

        Notes
        -----
        After this call, `self.stats` contains the full diagnostic
        summary for display.
        """
        t_utl = ProcUtils(self.prm)
        stats = t_utl._input_data_stats(input_data)
        
        vars_type = self._vars_types(input_data)
        vars_type = self._set_remove_flow_vars_types(vars_type)

        data, time_se  = self._data_temporal_subset(input_data, stats)
        stats = stats | t_utl._data_stats(data, vars_type, time_se)
    
        train, test = self._train_test_split(data)        
        stats = stats | t_utl._tt_stats(train, test)

        train, vars_type, fe_info = self._fe_sampling(train, vars_type)        
        train, d_info  = self._train_sampling(train)        

        vars_type = self._add_flow_vars_types(vars_type)

        self.stats = t_utl._final_stats(stats, train, fe_info, d_info)
        
        return DataProcResult(
            train_data = train,
            test_data  = test,
            vars_type  = vars_type,
        )   
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬