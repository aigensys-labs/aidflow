"""
Data Loader — Pipeline Ingestion & Standardization.

This module implements the first stage of the AIDflow pipeline.
It ingests raw data from an external source, verifies mandatory
constraints, standardizes the output into the canonical pipeline
format, and performs the time-based train/test split that all
downstream agents operate on.

What It Does
------------
1. **Ingest**: Import data from file, database, or Spark via
   PyArrow (zero-copy columnar format).
2. **Verify**: Target is present, non-null, and binary.
3. **Standardize**: Parse time column, sort chronologically,
   move target to position 0, subset columns.
4. **Split**: Partition data time-wise into TRAIN (in-sample)
   and TEST (OOT) segments.
5. **Classify**: Infer variable types (numeric, categorical,
   flow, target) for downstream feature screening.

The output is the canonical dataset that feeds every subsequent
stage — feature encoding, agent training, comparison, and
deployment.

Supported Sources
-----------------
- **File**: Parquet, CSV, ORC, or any PyArrow-readable format.
- **Database**: Any DB with a PyArrow-compatible connector
  (PostgreSQL, DuckDB, etc.) via connection string or raw SQL.
- **Spark**: Spark tables or SQL queries via an active
  PySpark session.

Pipeline Position
-----------------
This is the **entry point**. It has no upstream dependencies.
Its output is consumed by:
- Feature Type Inference & Quality Screening.
- WOE Encoding (categorical → ordinal).
- Agent Training (GLM / GBM).
- All downstream selection and deployment stages
"""

from ...deps.common         import  wraps
from ...deps.data           import  pa, pDF, pSe

from ...core.structure      import  (
                                    SID, 
                                    STRUCTURE
                                    )
from ...core.parameters     import  (
                                    PRM,
                                    AIGFLOW,
                                    Parameters
                                    )
from ...core.registry       import  (
                                    FitRegister, 
                                    FitRegOpt,   
                                    DashRegister,  
                                    RegGet
                                    )
from ...core.base           import  (
                                    FitterBase,
                                    TransformerBase,
                                    ModelProcessorBase,
                                    ModelInput,
                                    ModelResult,
                                    ModelBase,  
                                    ModelUIBase
                                    )
from ...core.factory        import  (
                                    Dataclass,
                                    DictClass
                                    )
from ...core.foundation     import  AigParam
from ...core.display        import  DashInterface
from ...core.io             import  (
                                    ModelIO, 
                                    ReportIO, 
                                    )
from ...core.accessor       import  Format

from .processor             import  (
                                    DataProcParamsBase,
                                    DataProcParams,
                                    DataProcResult,
                                    DataProcStats,
                                    DataProcessor
                                    )
from .importer              import  (
                                    ImpParamsBase,
                                    ImpParamsCfg,
                                    DataImporter
                                    )
from .utils_ui              import  PanelUtilsUI


MODEL = AIGFLOW.Data.loader
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG DATA STAGE ► DATA LOAD ► Import and Standardize Data with Variables Types inference'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG = 'AigImporter'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → LOAD → IMPORT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ImportOptions(FitRegOpt):
    """Module Fit Options - Each Fitter is Auto Registered to the main REGISTRY."""       
    
    @FREG.register(REG.AIG)
    class AigDataImporter(ImpParamsBase):
        
        config: dict|ImpParamsCfg = ImpParamsCfg(            
            data_name    = None,
            default_path = STRUCTURE[SID.D],
            data_source  = 'file',
            target       = None,
            yy_mm        = None,
            yy_mm_format = None
        )

        def _build(self, params:ImpParamsBase, config:dict) -> DataImporter:            
            setup  = dict(params) | config
            return DataImporter(**setup) 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → LOAD → TRANSFORM DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class TransParamsPanelMapping:
    """Transfomer params mapping to Config defaults."""    
    params = Dataclass.schema(DataProcParamsBase)
    setup  = {
        PRM.LOAD.start.param      : params.import_start,
        PRM.LOAD.end.param        : params.import_end,
        PRM.LOAD.test.param       : params.train_test,
        PRM.LOAD.sample.param     : params.smp_train,
        PRM.LOAD.features.param   : params.smp_feature
    }
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class ImporterResult:
    """Importer params input design."""
    data: pDF

class Importer:
    """Importer engine responsible to ingest data to the pipeline."""      

    fitter = None

    def __init__(self, fx_choice: RegGet.Choice | str | dict):
        self.fx_choice = fx_choice
        self.fx = FREG.get(self.fx_choice)

    def process(self, 
        params: ImpParamsBase, 
        direct_data: pDF | pa.Table = None, 
        verbose: bool = True
    ) -> pDF:
        self.fitter = self.fx.build(params=params)
        return self.fitter.process(direct_data=direct_data, verbose=verbose)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class FitResult(DataProcResult):
    """Fitter output design."""

class Fitter(FitterBase[DataProcParams]):
    """
    Splits input data time-wise fashion to create TRAIN part for training and TEST 
    part as OOT sample.
    """    

    FREG = FREG
    DREG = DREG

    def _process(self, data:pDF) -> tuple[DataProcResult, DataProcStats]:
        transformer = DataProcessor(self.params)
        result = transformer.process(data)
        stats  = transformer.stats
        return result, stats
    
    def fit(self, data:pDF): 
        """Transform data to standardized input to AIG flow."""
        result, stats = self._process(data)

        self.result: FitResult = FitResult(**result)
        self.stats: DataProcStats = stats

        self._init_dash()
        return self

    def transform(self, data:pDF) -> DataProcResult:
        """Transform data to standardized input to AIG flow."""
        return self._process(data)[0]
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    vars_type: pSe

@Dataclass.base
class TransResData(DataProcResult):
    """Transfomer data output design."""

class Transformer(TransformerBase[Fitter, DataProcStats, TransResCols]):
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
            vars_type = res.vars_type
        )     
               
    def transform(self, data:pDF) -> TransResData: 
        """Model meta-transform."""
        res = self.fitter.transform(data)
        return TransResData(
            train_data = res.train_data,
            test_data  = res.test_data,
            vars_type  = res.vars_type
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → NUM → PROCESSOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Processor(ModelProcessorBase[Transformer, TransResCols, TransResData]):
    """Transformer - Importer Choice-Params - Processor."""    
    
    MODEL  = MODEL
    DESC   = DESC

    Config  = ImpParamsCfg
    Mapping = TransParamsPanelMapping

    ModelTransformer = Transformer

    importer = None
    _choice  = None
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    @property
    def fitter(self) -> ImportOptions: 
        if self.importer is None:
            return
        return self.importer.fitter   
    
    @property
    def report(self) -> DashInterface: 
        if self.importer is None:
            return
        d = self.fx.fitter._dashboard
        if d is not None:
            return d        
        return ReportIO(self.setup).load(self.MODEL)      
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def _is_set_new_importer(self, new_choice:RegGet.Choice):
        if self._choice == new_choice.name:
            return False
        self.importer = Importer(new_choice)            
        self._choice  = new_choice.name
        return True
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_choice(self, fx_choice:str, inputs:"DataLoaderInput"):
        cfg = self.setup
        defaults  = self.Config(
            data_name       = cfg.input_data,
            data_source     = cfg.input_source,
            default_path    = cfg.STRUCTURE[SID.D],
            target          = AigParam.di(inputs.target, cfg.target),
            yy_mm           = cfg.yy_mm,
            yy_mm_format    = cfg.yy_mm_format
        )
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
                                       
    def get_params(self, fx_choice:str, inputs:"DataLoaderInput", params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params) 
        
        cfg = self.setup
        imp = ImpParamsBase(            
            columns_subset  = cfg.input_cols,
            flow_vars       = cfg.flow_vars,
            time_var        = AigParam.di(inputs.time_var, cfg.time_var),
        )      
        trn = DataProcParams(
            target          = AigParam.di(inputs.target, cfg.target),
            time_var        = AigParam.di(inputs.time_var, cfg.time_var),
            flow_vars       = cfg.flow_vars,            
            **params
        )
        return imp, trn, cfg
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → LOAD → API
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class DataLoaderInput(ModelInput):
    """
    DataLoader fit input specification.

    Defines the configuration required to execute the data
    ingestion and standardization pipeline. All fields are
    resolved from the flow context (`ctx.setup`) and user
    panel inputs.

    `Following attribues are present for all pipeline models.`

    Attributes
    ----------
    direct_data: pDF | pa.Table = None
        Sideload data already present in memory. When provided,
        the import step (input_source / input_data) is bypassed
        entirely — the data is used as-is for downstream processing.
    target: str = None
        Defaults to ctx setup value. For prototyping, during ML dev,
        testing behavior of specific model on various targets might 
        be beneficial.
    time_var: str = None
        Defaults to ctx setup value. For prototyping, during ML dev,
        testing behavior of specific model on various targets might 
        be beneficial.
    fx_choice: str = None
        Defaults to ctx setup value. Choice of model's internal 
        fitter.
    fx_params: dict = None
        Defaults to ctx setup values. ML parametrization of model's
        internal fitter.
    fx_config: dict = None
        Defaults to ctx setup values. Config parametrization of model's
        internal fitter.
    save_path: str = None
        Optional custom save path.
    verbose: int = None
        Defaults to ctx setup values. Each step can be overriden to 
        customize verbosity level.
    """
    direct_data: pDF | pa.Table = None

@Dataclass.base(dynamic_params=True)
class DataLoaderResult(ModelResult):
    """
    DataLoader output — the canonical pipeline dataset.

    The final result of the ingestion and standardization
    stage. Contains the train/test split and variable type
    classification that all downstream stages consume.

    Attributes
    ----------
    input_data : pDF
        The full imported dataset (before split). Contains
        all columns as ingested, with the target at position 0
        and the standard time variable in place.
    train_data : pDF
        Training segment (in-sample). All downstream agents
        train on this data.
    test_data : pDF
        Test segment (OOT / out-of-time). Used for evaluation
        and comparison — never used for training.
    dtypes : pSe
        Variable type classification per column (e.g.,
        'numeric', 'categorical', 'flow', 'target').
        Consumed by the feature encoding and quality
        screening stages.

    Notes
    -----
    **Train/Test Split**: The split is time-based (not
    random) — the most recent observations form the TEST
    segment. This mirrors production conditions where the
    model is deployed and evaluated on future data.

    **Variable Types**: The `dtypes` series drives all
    downstream decisions:
    - Target → the prediction objective.
    - Flow → preserved for joins, never modeled.
    - Numeric → GLM with WOE encoding | GBM passthrough.
    - Categorical → GLM with WOE encoding | GBM passthrough.
    """
    input_data: pDF
    train_data: pDF   
    test_data:  pDF
    dtypes:     pSe 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class DataLoader(Processor, ModelBase):
    """
    Data Loader — Pipeline Ingestion & Standardization.

    The entry point of the AIDflow pipeline. Ingests raw data
    from an external source, verifies mandatory constraints,
    standardizes the output into the canonical format, and
    performs the time-based train/test split.

    What It Does
    ------------
    1. **Ingest**: Import data from file, database, or Spark
       via PyArrow (zero-copy columnar).
    2. **Verify**: Target present, non-null, binary.
    3. **Standardize**: Parse time, sort chronologically,
       move target to position 0, subset columns.
    4. **Split**: Time-based partition into TRAIN and TEST.
    5. **Classify**: Infer variable types for downstream use.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full ingestion + standardization +
      train/test split. Produces the canonical dataset.
    - **transform(data)**: Re-split an existing dataset using
      the configured time boundaries. Use for re-partitioning
      or OOT monitoring.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **No Upstream Dependencies**: This is the first stage of
    the pipeline. It requires no prior fitted models.

    **Time-Based Split**: The train/test split is strictly
    time-based (not random). The most recent observations form
    the TEST segment. This mirrors production conditions where
    the model is evaluated on future data.

    **Arrow-Backed Output**: The resulting DataFrames use
    Arrow-backed dtypes for zero-copy compatibility with
    downstream PyArrow operations.

    **Idempotent**: Re-fitting with the same configuration
    produces the same output. The ingestion is deterministic.

    Examples
    --------
    >>> from aidflow.data import DataLoader
    >>> loader = DataLoader(ctx.setup)
    >>>
    >>> # Full ingestion + split
    >>> loader.fit(DataLoaderInput())
    >>>
    >>> # Access the split datasets
    >>> result = loader.fitted
    >>> result.train_data.shape
    >>> result.test_data.shape
    >>> result.dtypes  # Variable type per column
    """

    def _save(self, save_path:str) -> None:
        features = {
            'fe_types': self.cols.vars_type.to_dict()
        }
        ModelIO(self.setup).save(
            MODEL, self, features, init_fe=True, 
            custom_path = save_path
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_result(self, res: TransResData, data: pDF|None = None):
        return DataLoaderResult(
            input_data = lambda: data if data is not None else self._result.data,
            train_data = lambda: res.train_data,
            test_data  = lambda: res.test_data,
            dtypes     = lambda: res.vars_type,
        )    
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs: DataLoaderInput, fx_choice: str = REG.AIG, **params):
        """
        Execute the full data ingestion and standardization pipeline.

        Ingests data from the configured source, verifies
        constraints, standardizes the format, and performs the
        time-based train/test split.

        Parameters
        ----------
        inputs : DataLoaderInput
            Flow context configuration (resolved from ctx.setup).
        fx_choice : str, optional
            The fitting strategy identifier. Defaults to REG.AIG.
        **params : dict
            Runtime parameters passed to the internal fitter
            (import_start, import_end, train_test, smp_learn, smp_feature).

        Returns
        -------
        DataLoader
            The fitted instance with train/test split and
            variable type classification available.
        """
        
        self.choice = self.get_choice(fx_choice, inputs)
        self.params = self.get_params(fx_choice, inputs, params)
        
        pi, *pt = self.params

        if self._is_set_new_importer(self.choice):
            di = self.importer.process(pi, inputs.direct_data, inputs.verbose)
            self._result = ImporterResult(data = di)
        
        self.processor.fit(
            inputs, self._result.data,
            fx_choice = self.choice,
            params = pt
        )

        self._save(inputs.save_path)
        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def transform(self, data:pDF) -> DataLoaderResult:
        """
        Re-partition a dataset using the configured time boundaries.

        Parameters
        ----------
        data : pDF
            Input data to split (must contain the standard time
            variable).

        Returns
        -------
        DataLoaderResult
            The split datasets (train, test) + variable types.
        """        
        res = self.processor.transform(data)
        return self._get_result(res, data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → LOAD → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI:
    """UI Dynamic Parameters Panel."""

    title = 'DATA LOAD OVERVIEW'

    def __init__(self, config:Parameters, inputs:DataLoaderInput):
        prm = TransParamsPanelMapping(config, default=True)        
        self.wg = PanelUtilsUI(prm)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def panel(self, model:DataLoader):
        """Shows Main UI Options for ML fx fit."""        
        self._panel = self.wg.panel(model.fx.stats.data_stats)

    def import_title(self, *args, **kwargs):        
        Format.title(self.title, style = Format.StyleTitle(center=False))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_ui_setting(self) -> DataProcParamsBase:
        if not hasattr(self, '_panel'):
            return {}
        return self.wg.get_ui_setting()        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → LOAD → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class DataLoaderUI(ModelUIBase[DataLoader, OptionUI]):
    """Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs."""
    
    Model  = DataLoader
    Option = OptionUI
      
    def static_panel(self, *args, **kwargs):
        """Update Panel with Imported Timeline Update option."""                        
        self.option_ui.panel(self.model)
    
    @wraps(DataLoader.fit)
    def fit(self, *args, **kwargs): 
        """Loader fit-import UI wrapper."""                                
        self.model_ui.fit(    
            *args, 
            ui_params = self.model_ui.UIParams(
                fit_transform_pipeline      = False,
                gui_static_before_confirm_1 = self.option_ui.import_title,
                dynamic_to_args_kwargs_FX_4 = self.inject_to_params,    
                gui_static_w_fitted_5       = self.static_panel,
                is_button_last              = True, 
                output_wait                 = True, 
            ),
            **kwargs
        )
        return self
    
    @wraps(DataLoader.transform)
    def transform(self, data: pDF) -> DataLoaderResult:
        return self.model.transform(data)

    @property
    def fitted(self) -> DataLoaderResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬