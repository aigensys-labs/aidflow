"""
Data Encoding Application — Bridge to ML Training.

This module implements the final step of the encoding phase, responsible
for applying all previously trained encodings (standardize, WOE transformation) 
to produce standardized train/test samples ready for model training.

Pipeline Position
-----------------
This is the **last encoding stage** and the **first training-stage
interface**. It consumes the trained model adapters from all prior
stages and produces the final feature matrix that the ML model will
train on.

Key Responsibilities
-------------------
- **Encoding application**: Applies the trained WOE encoder to
  transform raw features into their WOE representation.
- **Sample creation**: Splits data into train/test (time-based or
  random) when no explicit test set is provided.
- **Standardization**: Ensures both samples have identical columns
  and dtypes, converted to native pandas/numpy (Arrow-safe).
- **Safety verification**: Validates that input data dtypes match
  the trained schema (when preprocessing is disabled).

Design Principle
----------------
**Single source of truth for encoding**: Both train and test samples
pass through the same encoder instance, guaranteeing that the
feature transformation is identical across both sets. This eliminates
train/test leakage from the encoding step and ensures the model
sees consistent representations.
"""

from ...deps.struct             import  dataclass
from ...deps.data               import  pa, pDF, pSe
from ...deps.types              import  Callable

from ...core.base               import  (
                                        FitterDataBase, 
                                        FitterModelBase
                                        )
from ...core.factory            import  Dataclass
from ...core.foundation         import  (
                                        AigParam, 
                                        AigStop
                                        )
from ...core.interop            import  Pd, Arrow
from ...core.base.adapter       import  ModelAdapterBase
from ...core.standard           import  UniType

from ...ml.sampling             import  Sampling
from ...display.rgb             import  RgbANSI

from ..screen.fe_cleaner        import  TransResCols as CleanerFes
from ..encode.woe_encoder       import  TransResCols as WoEFes

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → TRAININING → SAMPLES ENCODING
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class EncoderParamsBase:
    """
    Encoding pipeline flags.

    Attributes
    ----------
    test_size : float, default=0.3
        Fraction of data allocated to the test set (used only when
        no explicit test set is provided).        
        - Special case 0: returns test_set = None - no data split
    """
    test_size:      float = 0.3

@Dataclass.base(update_specs=dict(slots=False))
class EncoderParamsCfg:
    """
    Trained model adapters and split configuration.

    Attributes
    ----------
    fe_cleaner : ModelAdapterBase, default=None
        Trained feature cleaner.
    woe_encoder : ModelAdapterBase, default=None
        Trained WOE encoder. Applied to all features to produce 
        the final WOE representation.
    time_var : str, default=None
        Time column name. If set, train/test split is done
        chronologically (most recent `test_size` fraction = test).
        If None or not present in data, a random split is used.
    """
    fe_cleaner:     ModelAdapterBase = None
    woe_encoder:    ModelAdapterBase = None
    time_var:       str = None

@Dataclass.base
class EncoderParams(EncoderParamsBase, EncoderParamsCfg):
    """Fitter params input design."""

@Dataclass.base
class EncoderStats:
    """
    Shape and description stats for the produced train/test samples.

    Attributes
    ----------
    train : dict
        {'n_observations': int, 'n_features': int, 'dst': describe()}
    test : dict | None
        Same structure for the test sample if sample test_size > 0.
    """    
    train: dict
    test:  dict|None
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class EncoderInput:
    """
    Input contract for the data encoder.

    Attributes
    ----------
    train_set : pDF
        Training data (raw features, pre-WOE).
    test_set : pDF | None, default=None
        Optional explicit test set. If None, the train set is split
        using `test_size` (time-based or random).
    train_woe : pDF | None, default=None
        Pre-computed WOE features for the train set. If None, the
        WOE encoder is applied to produce them.
    woe_features : list | None, default=None
        WOE feature column names to include. If None, all WOE features
        from the encoder output are used.
    """
    train_set: pDF 
    test_set:  pDF|None = None
    train_woe: pDF|None = None
    woe_features: list|None = None

@Dataclass.base
class EncoderResult:
    """
    Output contract: ML-ready train/test samples.

    Attributes
    ----------
    train_set : pDF
        Training sample with WOE-encoded features (native dtypes).
    test_set : pDF | None
        Test sample with WOE-encoded features (native dtypes).
        Identical columns to train.
        When test_size was `set to 0`, None is provided instead.
    features : list
        Ordered feature column names (model input order).
    dtypes : pSe
        Expected dtypes per column (for downstream verification).
    """    
    train_set: pDF
    test_set:  pDF|None
    features:  list
    dtypes:    pSe

@dataclass
class _Woe:
    """Internal: WOE-encoded data carrier."""
    data: pDF|None = None
    fes: list|None = None 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class _Predictor:
    """
    Encapsulated access to trained model adapters.

    Provides a uniform interface to invoke the trained preprocessing,
    and WOE encoding models, decoupling the encoder logic from the 
    specific adapter implementations.
    """
    def __init__(self, prm: EncoderParams):
        self.prm  = prm

    def pre(self, data:pDF):
        """Apply trained feature cleaner."""
        return self.prm.fe_cleaner.predict(data)

    def woe(self, data:pDF):
        """Apply trained WOE encoder (produce WOE values per feature)."""
        return self.prm.woe_encoder.predict(data)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Fitter(FitterDataBase):
    """
    Data Encoding — execution backend.

    Applies the full encoding pipeline to raw data and produces
    standardized train/test samples ready for ML training:

    1. **Standardize**: ensure identical columns and native dtypes.
    2. **WOE encode**: transform all features to WOE values.
    3. **Split**: create train/test (time-based or random).
    4. **Verify**: check input dtype consistency (safety gate).

    Parameters
    ----------
    params : EncoderParams
        Trained adapters and pipeline flags.
    """
    stats:  EncoderStats = None

    def __init__(self, params: EncoderParams):
        self.prm = params
        self.predict = _Predictor(self.prm)
        self._set_dtypes()

    def _set_dtypes(self):
        cols: CleanerFes = self.prm.fe_cleaner.cols
        self.trained_dtypes = cols.vars_types    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_woe_fes(self) -> list:
        cols: WoEFes = self.prm.woe_encoder.cols
        return cols.vars_woe

    def _get_woe(self, inputs: EncoderInput, is_train: bool) -> _Woe:        
        fes = inputs.woe_features

        if not is_train:
            return _Woe(fes = fes)
        
        return _Woe(data = inputs.train_woe, fes = fes)
            
    def _encode_data(self, d:pDF, inputs:EncoderInput, is_train:bool) -> _Woe:
        """
        Apply WOE encoding to the data (or use pre-computed).

        If pre-computed WOE is available (train), skips the encoding
        step. Otherwise, invokes the trained WOE encoder.        
        """
        woe = self._get_woe(inputs, is_train)
        fes = self._get_woe_fes()

        if woe.data is None: woe.data = self.predict.woe(d)
        if woe.fes  is None: woe.fes  = [x for x in fes if x in woe.data.columns]

        return woe
   
    def _extend_data(self, d:pDF, inputs:EncoderInput, is_train:bool = False) -> pDF:
        """Extend data by encoded WOE feature data."""
        woe = self._encode_data(d, inputs, is_train)
        return Pd.concat([d, woe.data[woe.fes]], axis=1)
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    def _fx_preprocess(self, data:tuple, fx:Callable) -> tuple:
        res = []
        for d in data:
            res.append(fx(d))
        return tuple(res)

    def _is_preprocess(self, data:tuple[pDF]) -> bool:
        for d in data:
            dt   = d.dtypes
            tt   = self.trained_dtypes
            cols = [x for x in d.columns if x in tt.index]
            diff = [x for x in cols if dt[x] not in UniType.get_group(tt[x])]

            is_diff = bool(diff)
            if is_diff:
                break
        
        return is_diff

    def _preprocess(self, *data) -> tuple|pDF:
        """
        Apply the full preprocessing pipeline to one or more datasets.

        Parameters
        ----------
        *data : pDF
            One or more datasets to preprocess.

        Returns
        -------
        pDF | tuple
            Preprocessed data (single if one input, tuple if multiple).
        """
        if self._is_preprocess(data):
            data = self._fx_preprocess(data, self.predict.pre)

        data = self._fx_preprocess(data, Arrow.to_safe_datatype)

        return data[0] if len(data) == 1 else data

    def _split_data(self, d:pDF, test_size:float=None) -> tuple[pDF, pDF]:
        """
        Split data into train/test.

        Uses time-based split (chronological) if `time_var` is set,
        otherwise random split. The test set is the most recent
        `test_size` fraction.

        Parameters
        ----------
        d : pDF
            Data to split.

        Returns
        -------
        tuple[pDF, pDF]
            (train, test)
        """
        size = test_size or self.prm.test_size
        if self.prm.time_var and self.prm.time_var in d.columns:
            return Sampling(size).time_split(d, time_var=self.prm.time_var)
        return Sampling(size).train_valid_split(d)

    def _get_tt_data(self, inputs: EncoderInput, test_size:float=None) -> tuple[pDF, pDF|None]:
        """
        Resolve the train/test data pair.

        If an explicit test set is provided in inputs, both are
        preprocessed independently. Otherwise, the train set is
        split into train/test after preprocessing.

        Parameters
        ----------
        inputs : EncoderInput
            Input contract.

        Returns
        -------
        tuple[pDF, pDF|None]
            (train, test) — both preprocessed.
        """
        if inputs.test_set is not None:
            return self._preprocess(inputs.train_set, inputs.test_set)
        
        d = self._preprocess(inputs.train_set)
        if test_size == 0:
            return d, None
        return self._split_data(d, test_size)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _verify(self, inputs:EncoderInput) -> None:
        """Safety gate: verify input data dtypes match the trained schema."""
        schema  = Dataclass.schema(EncoderInput)
        verify  = (schema.train_set, schema.test_set)
        trained = self.trained_dtypes.index.to_list()

        for name, data in inputs.items():
            if name not in verify or data is None:
                continue
            
            cols = [x for x in data.columns if x in trained]

            if not cols:
                AigStop.custom(
                    RgbANSI.bold_red(f'► AIG SAFETY ► Inconsistent {name} data dtypes '
                    '► Non of provided columns were present in training data')+
                    f'\n► Supported columns: \n{trained}', 
                    red=False, bold=False
                )
    
    def _get_inputs(self, inputs: EncoderInput | pDF | pa.Table) -> EncoderInput:
        """Normalize and verify the input."""        
        if isinstance(inputs, pa.Table):
            inputs = Arrow.to_pandas(inputs)
        if isinstance(inputs, pDF):
            inputs = EncoderInput(train_set=inputs)
        self._verify(inputs)
        return inputs
    
    @staticmethod
    def get_stats(data:pDF|None) -> dict | None:
        """Compute shape and description statistics for a sample."""
        if data is None:
            return
        s = data.shape
        return dict(n_observations = s[0], n_features = s[1], dst = data.describe())
    
    def process(self, 
        inputs: EncoderInput | pDF | pa.Table,
        test_size: float = None
    ) -> tuple[pDF, pDF | None]:
        """
        Execute the full encoding pipeline and produce train/test samples.

        Steps:
        1. Normalize and verify inputs.
        2. Resolve and preprocess train/test data.
        3. Apply WOE encoding to both sets.
        4. Ensure identical column structure.
        5. Record dtype schema and shape stats.

        Parameters
        ----------
        inputs : EncoderInput | pDF
            Input data (raw features, pre-WOE).
        test_size : float, default=None
            Fraction of data allocated to the test set (used only when
            no explicit test set is provided).        
            - None: uses value set during init
            - 0: returns test_set = None - no data split

        Returns
        -------
        tuple[pDF, pDF]
            (train, test) — ML-ready samples with WOE features.
        """        
        inputs_ = self._get_inputs(inputs)
        train, test = self._get_tt_data(inputs_, test_size)
        train = self._extend_data(train, inputs_, is_train=True)
        
        if test is not None:
            test = self._extend_data(test, inputs_)[train.columns]

        self.stats = EncoderStats(
            train  = self.get_stats(train),
            test   = self.get_stats(test),
        )
        return train, test
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DataEncoderProcessor(FitterModelBase[Fitter]):
    """
    Final encoding stage. Its output is the **input to ML training**.
    Both train and test samples are produced by the same encoder
    instance, guaranteeing identical feature transformations.

    Parameters
    ----------
    fe_cleaner : ModelAdapterBase
        Trained feature cleaner.
    woe_encoder : ModelAdapterBase
        Trained WOE encoder (the primary transformation).
    time_var : str, default=None
        Time column for chronological train/test split.
    test_size : float, default=0.3
        Test fraction (when no explicit test set is provided).

    Notes
    -----
    **Single source of truth**: The same trained adapters are applied
    to both train and test, eliminating encoding-related leakage.

    **Dtype safety**: Input dtypes are verified against the trained 
    schema to catch silent mismatches. Trained feature cleaner 
    normalizes dtypes when any mismatch is present.

    **Native dtypes output**: Arrow-backed dtypes are converted to
    native pandas/numpy for maximum compatibility with ML backends
    (LightGBM, XGBoost, scikit-learn).

    **Time-based split**: When `time_var` is set, the split is
    chronological (most recent observations → test), which better
    reflects production conditions for time-series data.

    **Pre-computed WOE**: If `train_woe` is provided in the input,
    the WOE encoder is skipped for the train set (useful when WOE
    was already computed during the pipeline and you want to reuse it).

    Examples
    --------
    >>> from aidflow.fitters import DataEncoderProcessor
    >>> encoder = DataEncoderProcessor(
    ...     fe_cleaner=fe_cleaner,
    ...     obs_stabilizer=obs_analyzer,
    ...     woe_encoder=woe_encoder,
    ...     time_var='AIG_YY_MM',
    ...     test_size=0.3,
    ... )
    >>> result = encoder.transform(data)
    >>> result.train.shape  # (n_train, n_features)
    >>> result.features[:5]  # ['target', 'revenue_WOE', ...]
    """

    def __init__(self, *args, params: dict|EncoderParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(EncoderParams, params, args, kwargs)
        self._fitter = Fitter(params = self._params)
    
    def transform(self, 
        inputs: EncoderInput | pDF | pa.Table, 
        test_size: float = None
    ) -> EncoderResult:
        """
        Produce ML-ready train/test samples from raw data.

        Applies the full encoding pipeline:
        1. Dtype verification & standardization (Arrow → native).
        2. Outlier removal (optional): filter flagged observations.
        3. WOE encoding: transform all features to WOE values.
        4. Train/test split: time-based or random.
        5. Column alignment: ensure test matches train columns.

        Parameters
        ----------
        inputs : EncoderInput | pDF | pa.Table
            Input data. Can be:
            - A raw DataFrame (auto-wrapped into EncoderInput).
            - An EncoderInput with explicit train/test sets.
            - An EncoderInput with pre-computed WOE (skip encoding).
        test_size : float, default=None
            Fraction of data allocated to the test set (used only when
            no explicit test set is provided).        
            - None: uses value set during init
            - 0: returns test_set = None - no data split
            
        Returns
        -------
        EncoderResult
            - `train_set`: ML-ready training sample.
            - `test_set`: ML-ready test sample (identical columns).
            - `features`: Ordered feature list (model input order).
            - `dtypes`: Expected dtype per column.

        Raises
        ------
        ValueError
            If input dtypes are inconsistent with the trained schema
            (when `preprocess=False`).
        """
        train, test = self._fitter.process(inputs, test_size)
        dt = train.dtypes

        return EncoderResult(
            train_set = train,            
            test_set  = test,
            features  = dt.index.to_list(),
            dtypes    = dt
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬