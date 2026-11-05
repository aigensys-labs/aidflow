"""
ML Data Encoder — Final Encoding Application for Training.

This module implements the final encoding stage of the AIDflow
pipeline. It applies all previously trained encodings (standardize,
and WOE transformation) to produce standardized train/test samples 
ready for model training.

What It Does
------------
1. **Standardize**: Ensure both samples have identical columns
   and native dtypes (Arrow-safe). Trained feature cleaner is 
   applied to input data.
2. **WOE encode**: Transform all features to their WOE
   representation using the trained encoder.
3. **Split**: Create train/test samples (time-based or random)
   when no explicit test set is provided.

Role in the Pipeline
--------------------
This is the **last encoding stage** and the **first
training-stage interface**. It consumes the trained model
adapters from all prior stages and produces the final feature
matrix that all training agents will use.

Design Principle
----------------
**Single source of truth for encoding**: Both train and test
samples pass through the same encoder instance, guaranteeing
that the feature transformation is identical across both sets.
This eliminates train/test leakage from the encoding step.

Key Guarantees
--------------
- **Dtype safety**: Native pandas/numpy dtypes (Arrow-backed
  types are converted) for full compatibility with LightGBM,
  XGBoost, and scikit-learn backends.
- **Column alignment**: Test sample columns are guaranteed to
  match the train sample exactly.
- **Reproducible split**: When `time_var` is set, the split is
  chronological (most recent observations → test), reflecting
  production conditions.

Dependencies
------------
Requires the following stages to be completed:
- Feature Cleaner (`fe_cleaner`)
- WOE Encoder (`woe_encoder`)
"""

from ...deps.data               import  pa, pDF, pSe

from ...core.parameters         import  AIGFLOW, PRM
from ...core.registry           import  (
                                        FitRegister, 
                                        FitRegOpt,   
                                        RegGet
                                        )
from ...core.base               import  (
                                        FitterBase,
                                        TransformerBase,
                                        ModelProcessorBase,
                                        ModelEnvelope,
                                        ModelInput,
                                        ModelResult,
                                        ModelBase,
                                        )
from ...core.base.adapter       import  ModelAdapterBase
from ...core.factory            import  (
                                        Dataclass,
                                        DictClass
                                        )
from ...core.foundation         import  AigParam
from ...core.io                 import  ModelIO

from .processor                 import  (
                                        DataEncoderProcessor,
                                        EncoderParamsBase,
                                        EncoderParamsCfg,
                                        EncoderInput,
                                        EncoderResult,
                                        EncoderStats
                                        )

MODEL = AIGFLOW.Encoding.data_encoder
FREG  = FitRegister(MODEL)
DREG  = None
DESC  = 'AIG TRAINING STAGE ► ML DATA ENCODER ► APPLY TRAINED ENCODING TO TRAIN AND TEST SAMPLES'
DEPS =  (
        AIGFLOW.Data.fe_cleaner,
        AIGFLOW.Encoding.woe_encoder,
        )


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG = 'DataSamplesEncoder'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → ENCODER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class EncoderParamsMapping:
    """Direct assignment → no global flow params."""
    params = Dataclass.schema(EncoderParamsBase)
    setup  = {
        PRM.LOAD.test.param : (params.test_size, 'Test Part')
    }
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class FitOptions(FitRegOpt):
    """
    Model-to-Fitter Registration Layer.
    
    This class acts as the bridge between the high-level Model and the low-level Fitter.
    It utilizes a registration pattern to ensure that adding new prediction 
    algorithms only requires a new entry here, keeping the top-layer API stable.
    """     
    
    @FREG.register(REG.AIG)
    class TrainingDataEncoder(EncoderParamsBase):
        
        config: dict|EncoderParamsCfg = EncoderParamsCfg(
            fe_cleaner      = None,
            woe_encoder     = None,
            time_var        = None,
        )

        def _build(self, params:EncoderParamsBase, config:dict) -> DataEncoderProcessor:            
            setup  = dict(params) | config
            return DataEncoderProcessor(**setup) 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Fitter(FitterBase[EncoderParamsBase]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """        
    
    FREG = FREG
    DREG = DREG
    
    def fit(self, data:EncoderInput): 
        """
        Encodes data by trained project data encoding.

        Parameters
        ----------
        data : Input data samples for encoding.
        """   
        self.result: EncoderResult = self.model.transform(data)
        self.stats:  EncoderStats  = self.model.get_stats()
        return self
    
    def transfrom(self, data:pDF|EncoderInput, test_size:float=None) -> EncoderResult:
        """
        Encodes data by trained project data encoding.
        
        Parameters
        ----------
        data : New data for encoding / preprocessing.
        test_size : float, default=None
            Fraction of data allocated to the test set (used only when
            no explicit test set is provided).        
            - None: uses value set during init
            - 0: returns test_set = None - no data split     
        """
        return self.model.transform(data, test_size=test_size)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬        



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → ENCODER → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    features:   list
    dtypes:     pSe

@Dataclass.base
class TransResData(EncoderResult):
    """Transfomer data output design."""

class Transformer(TransformerBase[Fitter, EncoderStats, TransResCols]):
    """
    Transformer meta-processing between Fitter-Model.
    
    Handles the post-processing of Fitter results. Its primary role is to standardize 
    the raw output from various fitters into a unified TransResCols, TransResData
    format, enabling seamless transition regardless of the underlying fitter choice.
    """
    
    ModelFitter = Fitter

    def _set_cols(self):
        res = self.fitter.result
        return TransResCols(
            features = res.features,
            dtypes   = res.dtypes
        )        

    def transform(self, data:pDF|EncoderInput, test_size:float=None) -> TransResData: 
        """Model meta-transform."""
        return TransResData(
            **self.fitter.transfrom(data, test_size=test_size)
        )     
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → DATA ENCODER → PROCESSOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Processor(ModelProcessorBase[Transformer, TransResCols, TransResData]):
    """
    Logic Processor.
    
    Coordinates the lifecycle between FitChoice selection and Parameter resolution. 
    It ensures that params and dependencies are correctly injected into the 
    Transformer - Fitter before execution begins.
    """
    
    MODEL  = MODEL
    DESC   = DESC

    Config  = EncoderParamsCfg
    Mapping = EncoderParamsMapping

    ModelTransformer  = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_choice(self, fx_choice:str, inputs:"DataEncoderInput"):
        encoders = self.get_model_dependecy(DEPS)
        adapter  = lambda x: ModelAdapterBase(encoders[x])

        defaults = self.Config(
            fe_cleaner      = adapter(AIGFLOW.Data.fe_cleaner),
            woe_encoder     = adapter(AIGFLOW.Encoding.woe_encoder),
            time_var        = AigParam.di(inputs.time_var, self.setup.time_var)
        )  
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)   
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def get_params(self, fx_choice:str, inputs:"DataEncoderInput", params:dict):
        params: EncoderParamsBase = self.get_fit_params(fx_choice, inputs.fx_params, params)
        return params, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → DATA ENCODER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class DataEncoderInput(ModelInput):
    """
    Data Encoder fit input specification.

    Defines the raw data samples required to apply the full
    encoding pipeline and produce ML-ready train/test features.

    Attributes
    ----------
    train_set : pDF
        Training data (raw features, pre-WOE). Must contain
        the target column and all features to be encoded.
    test_set : pDF | None, default=None
        Optional explicit test set. If None, the train set is
        split using the configured `test_size` (time-based
        or random).
    woe_features : list | None, default=None
        WOE feature column names to include in the output.
        If None, all WOE features from the encoder are used.
    """
    train_set: pDF 
    test_set:  pDF|None = None
    woe_features: list|None = None

@Dataclass.base(dynamic_params=True)
class DataEncoderEnvelopeCfg:
    train_woe:   pDF | None

@Dataclass.base(dynamic_params=True)
class DataEncoderEnvelope(ModelEnvelope):
    """
    Extended input envelope carrying pre-computed WOE data.

    When the WOE-encoded training data is already available
    from a previous pipeline stage, it can be passed via this
    envelope to avoid redundant encoding.

    Attributes
    ----------
    inputs : DataEncoderInput
        Primary input specification.
    config : DataEncoderEnvelopeCfg
        Optional pre-computed WOE data.
    """
    inputs: DataEncoderInput 
    config: DataEncoderEnvelopeCfg

@Dataclass.base(dynamic_params=True)
class DataEncoderResult(ModelResult):
    """
    Data Encoder output — ML-ready train/test samples.

    The final result of the encoding application. Contains
    standardized training and test samples with WOE-encoded
    features, ready for consumption by all training agents.

    Attributes
    ----------
    data : pDF
        Data (training) sample with WOE-encoded features (native
        dtypes). Identical column structure to `test_set`.
    test_set : pDF
        Data test/holdout sample with WOE-encoded features 
        (native dtypes). Identical columns to `data`.
    features : list
        Ordered feature column names (model input order).
        This is the definitive feature list that all training
        agents will consume.
    dtypes : pSe
        Expected dtype per column. Provides the schema for
        downstream verification (ensuring new data matches
        the training schema).

    Notes
    -----
    **Native dtypes**: All Arrow-backed dtypes are converted
    to native pandas/numpy, ensuring full compatibility with
    LightGBM, XGBoost, and scikit-learn backends without
    additional conversion overhead during training.

    **Column alignment**: The `test` sample is guaranteed
    to have the same columns (and order) as `train`. This
    eliminates any column-mismatch errors downstream.

    **Feature lineage**: The `features` list is the
    authoritative reference for the model input order. Use
    this list when passing data to training agents.
    """
    data:      pDF
    test_set:  pDF
    features:  list
    dtypes:    pSe
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class DataEncoder(Processor, ModelBase):   
    """
    ML Data Encoder — Final Encoding Application for Training.

    The bridge between the encoding phase and the training
    phase. Applies all previously trained encodings to raw data
    and produces synchronized, ML-ready train/test samples.

    What It Does
    ------------
    1. **Standardize**: Apply the trained feature cleaner 
       to raw data - ensures native dtypes and identical
       column structure across train-test samples.
    2. **WOE encode**: Transform all features to WOE
       representation using the trained encoder.
    3. **Split**: Create train/test samples (time-based or
       random) when no explicit test set is provided.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Apply the full encoding pipeline to
      produce train/test samples. Produces the ML-ready data.
    - **transform(data)**: Apply the same encoding to new data
      (e.g., for scoring or re-encoding on a different sample).

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **Single source of truth**: The same trained adapters are
    applied to both train and test, eliminating encoding-related
    leakage. The feature transformation is guaranteed identical
    across both sets.

    **Dtype safety**: Native pandas/numpy dtypes are enforced.
    Arrow-backed types are converted to ensure full compatibility
    with LightGBM, XGBoost, and scikit-learn without additional
    conversion overhead during training.

    **Time-based split**: When `time_var` is set, the split
    is chronological (most recent observations → test), which
    better reflects production conditions for time-series data.

    Examples
    --------
    >>> from aidflow.training import DataEncoder
    >>> encoder = DataEncoder(ctx.setup)
    >>>
    >>> # Encode train + test data
    >>> encoder.fit(DataEncoderInput(
    ...     train_set = train_raw,
    ...     test_set  = test_raw
    ... ))
    >>>
    >>> # Access the ML-ready samples
    >>> result = encoder.fitted
    ... result.train        # Encoded train data for the Training phase
    ... result.test         # Encoded test  data for the Selection phase
    ... result.features     # All features for the next phases
    ... result.dtypes       # Dtypes overview
    """

    def _save(self, save_path) -> None:
        features = {
            'features': self.cols.features,
            'fe_types': self.cols.dtypes.to_dict()
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _get_inputs(inputs: DataEncoderEnvelope | DataEncoderInput) -> DataEncoderInput:
        if isinstance(inputs, DataEncoderEnvelope):
            return inputs.inputs
        return inputs 
    
    @staticmethod
    def _get_encoder_input(inputs:  DataEncoderEnvelope | DataEncoderInput) -> EncoderInput:
        train_woe = None
        if isinstance(inputs, DataEncoderEnvelope):
            train_woe = inputs.config.train_woe
            inputs = inputs.inputs
        return EncoderInput(
            train_set    = inputs.train_set,
            test_set     = inputs.test_set,
            train_woe    = train_woe,
            woe_features = inputs.woe_features,
        )
    
    def _get_result(self, res: TransResData):
        return DataEncoderResult(
            data     = lambda: res.train_set, 
            test_set = lambda: res.test_set, 
            features = lambda: res.features,
            dtypes   = lambda: res.dtypes
        )   
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs: DataEncoderInput, fx_choice: str = REG.AIG, **params):
        """
        Apply the full encoding pipeline to produce ML-ready samples.

        Executes dtypes standardization (if mismatch), WOE encoding, 
        and train/test split.

        Parameters
        ----------
        inputs : DataEncoderInput
            Input configuration including train_set, test_set,
            and woe_features.
        fx_choice : str, optional
            The encoding strategy identifier. Defaults to REG.AIG.
        **params : dict
            Runtime parameters passed to the internal fitter
            (test_size).

        Returns
        -------
        DataEncoder
            The fitted encoder instance with ML-ready samples.
        """                       
        inputs_ = self._get_inputs(inputs)
        verify  = len(inputs_.train_set)
        
        self.choice = self.get_choice(fx_choice, inputs_)
        self.params = self.get_params(fx_choice, inputs_, params)

        self.processor.fit(
            inputs_, self._get_encoder_input(inputs),
            fx_choice = self.choice,
            params    = self.params, 
            verify    = verify
        )
        self._save(inputs_.save_path)
        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def transform(self, 
        data: pDF | pa.Table | DataEncoderInput, 
        test_size: float = None
    ) -> DataEncoderResult:
        """
        Training phase utility for automatic feature encoding, preprocessing, 
        and train/test splits.

        - Native Dtype Enforcement:
            - Converts datasets to native NumPy/Pandas dtypes to ensure full 
              compatibility with LightGBM, XGBoost, and Scikit-Learn.
            - Prevents repeated N-step conversions during training, minimizing 
              peak pipeline RAM footprint.

        - Train/Test Partitioning:
            - Retains user-supplied `test_set` if explicitly provided.
            - Automatically splits `data` / train_set by `test_size` if no 
              test set is present.

        - Feature Encoding Extension:
            - Extends train and test sets by WOE features given project-level 
              fitted encoding.

        Parameters
        ----------
        data : pDF | pa.Table | DataEncoderInput
            The raw or input-config data to be encoded for ML sampling.
        test_size : float, default=None
            Fraction of data allocated to the test set (used only when
            no explicit test set is provided).        
            - None: uses value set during init
            - 0: returns test_set = None - no data split

        Returns
        -------
        DataEncoderResult
            An object containing synchronized train and test sets ready for agents.
        """
        if isinstance(data, DataEncoderInput):
            data = self._get_encoder_input(data)
        res = self.processor.transform(data, test_size = test_size)
        return self._get_result(res)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬