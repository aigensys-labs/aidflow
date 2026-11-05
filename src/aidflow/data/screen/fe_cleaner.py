"""
Feature Cleaner — Data Quality Screening & Type Enforcement.

This module implements the data quality stage of the AIDflow
pipeline. It screens features against quality gates (missing rate,
information content, noise, correlation redundancy), auto-repairs
common type issues, and enforces type consistency between training
and inference data.

What It Does
------------
1. **Screen**: Remove features that fail quality gates:
   - Blank (100% missing).
   - Excessive missing (> threshold).
   - Non-informative (near-constant).
   - Noisy numeric (likely ID/timestamp).
   - High-cardinality categorical (likely ID).
   - Correlation-redundant (highly correlated with a more
     populated feature).
2. **Auto-repair**: Fix common type issues:
   - Numeric stored as string → cast to numeric.
   - Mixed numeric/string → decide by dominant type.
   - Timedelta → numeric.
   - Boolean → 0/1 numeric.
3. **Enforce**: On inference data, apply the exact type mapping
   learned during training. Detect and report type drift.
4. **Classify**: Produce the final numeric/categorical partition
   that drives all downstream encoding and modelling.

Pipeline Position
-----------------
This is the **second** stage (after Data Loader). It consumes the
standardized dataset and the variable type classification from
the loader, and produces the clean, well-typed feature set.

Dependencies
------------
Requires the Data Loader stage to be completed (provides the
standardized dataset + variable type inference).
"""

from ...deps.common             import  wraps
from ...deps.data               import  pa, pDF, pSe

from ...core.parameters         import  (
                                        PRM,
                                        AIG,
                                        AIGFLOW
                                        )
from ...core.registry           import  (
                                        FitRegister, 
                                        FitRegOpt,   
                                        DashRegister,  
                                        RegGet
                                        )
from ...core.base               import  (
                                        FitterBase,
                                        TransformerBase,
                                        ModelProcessorBase,
                                        ModelInput,
                                        ModelResult,
                                        ModelBaseDual, 
                                        OptionUIBase,
                                        ModelUIBase
                                        )
from ...core.factory            import  (
                                        Dataclass,
                                        DictClass
                                        )
from ...core.foundation         import  AigParam
from ...core.interop            import  (
                                        Arrow,
                                        DataFrame,
                                        TypeStandard
                                        )
from ...core.io                 import  ModelIO
from ...core.standard           import  TypeDef

from .fitters.fe_standardizer   import  (
                                        FeatureStandardizer,
                                        FitParamsBase,
                                        FitParamsCfg,
                                        FitResult,
                                        FitStats                                    
                                        )

MODEL = AIGFLOW.Data.fe_cleaner
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG DATA STAGE ► QUALITY STANDARDS ► Learn and Ensure Features Standards & Types'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG = 'FeStandardizer'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → CLEANING → FIT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class CleanParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(FitParamsBase)
    setup  = {
        PRM.SCREEN.miss.param       : (params.max_miss,  'Maximum Missing'      ),
        PRM.SCREEN.mode.param       : (params.max_1val,  'Maximum 1 Value'      ),
        PRM.SCREEN.var.param        : (params.max_var,   'Maximum Variability'  ),
        PRM.SCREEN.corr.param       : (params.max_corr,  'Maximum Correlation'  ),
        PRM.SCREEN.cat.param        : (params.cat_max,   'Maximum Categories'   ),
        PRM.SCREEN.downcast.param   : (params.downcast,  'Float64 Downcast'     ),
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
    class FeatureStandardizerFitter(FitParamsBase):

        config: dict|FitParamsCfg = FitParamsCfg(
            time_var     = None,
            time_format  = None,
            new_time_var = AIG.time_var.value,
            flow_vars    = ['TARGET', AIG.time_var.value],
            verbose      = True
        )

        @property
        def _model(self) -> FeatureStandardizer:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> FeatureStandardizer:            
            setup  = dict(params) | config
            return FeatureStandardizer(**setup)

        def get_result(self) -> FitResult:
            return self._model.result

        def get_stats(self) -> FitStats:
            return self._model.stats
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class FitParams(FitParamsBase):
    """Fitter params input design."""
    fe_types: pSe

class Fitter(FitterBase[FitParams]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """  
    
    FREG = FREG
    DREG = DREG
    
    def fit(self, data:pDF):
        """
        Learns variables types and standards.

        Parameters
        ----------
        data :    Input Data for quality screening.
        verbose : bool → True shows dashboard and stats.
        """       
        self.model.fit(data, self.params.fe_types)
        self.result: FitResult = self.model.get_result()
        self.stats:  FitStats  = self.model.get_stats()
        self._init_dash()
        return self

    def transform(self, data:pDF) -> pDF:
        """ 
        Data Quality between Train/OOT/Production may change format or 
        missing vars in new data compared to Train.

        Parameters
        ----------
        data :    New Data to apply trained dtypes and behavior.
                
        Returns
        -------
        Transformed Data Standardized to expected variables types 
        """            
        return self.model.transform(data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → CLEANING → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer output design."""
    vars_types:     pSe
    vars_all:       list
    vars_num:       list
    vars_cat:       list
    vars_dropped:   list

@Dataclass.base
class TransResData:
    """Transfomer data output design."""
    data: pDF
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Transformer(TransformerBase[Fitter, FitStats, TransResCols]):
    """
    Transformer meta-processing between Fitter-Model.
    
    Handles the post-processing of Fitter results. Its primary role is to standardize 
    the raw output from various fitters into a unified TransResCols, TransResData
    format, enabling seamless transition regardless of the underlying fitter choice.
    """ 

    ModelFitter = Fitter

    def _fit_data(self, inputs:"FeCleanerInput"):
        return inputs.train_set
    
    def _set_cols(self):
        res = self.fitter.result
        return TransResCols(     
            vars_types   = self._get_dtypes(res),       
            vars_all     = res.vars_all,
            vars_num     = res.numeric,
            vars_cat     = res.object,
            vars_dropped = res.vars_dropped
        ) 
    
    @staticmethod
    def _get_dtypes(res: FitResult) -> pSe:
        """Aligns dtypes with transform() → object to category."""
        dt  = res.vars_type.copy()
        obj = [x for x,t in dt.items() if t in TypeDef.object and x in res.object]
        dt[obj] = TypeDef.category[0]
        return dt
    
    def transform(self, data:pDF) -> TransResData:
        """
        1. Applies trained standard to all columns
        2. Transforms all obj cols to category 
        """
        data = self.fitter.transform(data)
        cols = [x for x in self.cols.vars_cat if x in data.columns]
        TypeStandard.obj_to_cat(data, cols)      
        return TransResData(data = data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → CLEANING → PROCESSOR
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
    
    Config  = FitParamsCfg
    Mapping = CleanParamsMapping

    ModelTransformer = Transformer    
    # ─────────────────────────────────────────────────────────────────────────────────────────────          
        
    def get_choice(self, fx_choice:str, inputs:"FeCleanerInput"):
        defaults  = self.Config(
            time_var     = self.setup.yy_mm,
            time_format  = self.setup.yy_mm_format,
            new_time_var = AigParam.di(inputs.time_var, self.setup.time_var),
            verbose      = AigParam.di(inputs.verbose, self.setup.verbose),
            flow_vars    = self.setup.flow_vars,
        ) 
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)     
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
    
    def get_params(self, fx_choice:str, inputs:"FeCleanerInput", params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params)
        fit = FitParams(fe_types = inputs.dtypes, **params)        
        return fit, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → CLEANING → API
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class FeCleanerInput(ModelInput):
    """
    FeatureCleaner fit input specification.

    Defines the training and test datasets required to execute
    the data quality screening and type enforcement pipeline.

    Attributes
    ----------
    train_set : pDF
        Training data (in-sample). The primary dataset on which
        quality screening and type learning are performed.
    test_set : pDF
        Test data (OOT). Used to verify type consistency against
        the training standards during the dual transformation.
    dtypes : pSe
        Variable type classification per column (from the Data
        Loader stage). Drives the screening logic — numeric
        features are screened for noise, categorical for
        cardinality, etc.

    Notes
    -----
    **Both Sets Required**: The dual-set design allows the
    cleaner to verify type consistency on OOT data during
    the same pass — catching drift early before any modelling
    begins.
    """
    train_set:  pDF
    test_set:   pDF
    dtypes:     pSe

@Dataclass.base
class FeCleanerResult(ModelResult):
    """
    FeatureCleaner output — the clean, well-typed feature set.

    The final result of the data quality screening stage.
    Contains the cleaned data and the complete feature
    classification that drives all downstream encoding and
    modelling.

    Attributes
    ----------
    data : pDF
        Cleaned training data. Features that failed quality
        gates are removed, types are enforced, and the result
        is sorted in canonical order (target first, then
        numeric, then categorical).
    dtypes : pSe
        Final dtype per feature (post-repair), indexed by
        feature name. This is the authoritative type mapping
        enforced on all downstream inference data.
    dropped : list
        Features removed during screening, with reason
        (e.g., 'Missing > 70%', 'Only 1 Value over 90%',
        'Correlation over 90%').
    features : list
        All surviving features (flow vars + numeric +
        categorical), in canonical pipeline order.
    num_features : list
        Features classified as numeric. These are passed
        directly to tree-based models and WOE-encoded for
        GLM models.
    cat_features : list
        Features classified as categorical/object. These
        undergo WOE encoding before GLM training, and are
        handled natively by GBM models.

    Notes
    -----
    **Numeric/Categorical Partition**: This partition drives
    all downstream decisions:
    - Numeric → WOE encoding (GLM) or native handling (GBM).
    - Categorical → WOE encoding (GLM) or native handling (GBM).
    - Flow vars → preserved for joins, never modeled.

    **Dropped Features**: The `dropped` list provides a
    complete audit trail. Each entry records the feature name,
    its original type, and the specific reason for removal.
    """
    data:         pDF
    dtypes:       pSe
    dropped:      list
    features:     list
    num_features: list
    cat_features: list

@Dataclass.base
class FeCleanerResultTest:
    """Model Output Test Data Design."""
    data: pDF

@Dataclass.base(dynamic_params=True)
class FeCleanerResultDual(FeCleanerResult):
    """Model Output Dual (Train-Test) Design."""
    test_data: DataFrame
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class FeatureCleaner(Processor, ModelBaseDual):     
    """
    Feature Cleaner — Data Quality Screening & Type Enforcement.

    The data quality gate of the AIDflow pipeline. Screens
    features against quality thresholds, auto-repairs common
    type issues, enforces type consistency between training
    and inference data, and produces the final numeric/
    categorical partition that drives all downstream modelling.

    What It Does
    ------------
    1. **Screen**: Remove features that fail quality gates:
       - Blank, excessive missing, non-informative, noisy,
         high-cardinality, or correlation-redundant.
    2. **Auto-repair**: Fix common type issues (numeric-as-string,
       timedelta, bool, mixed types).
    3. **Enforce**: Apply the trained type mapping to inference
       data. Detect and report type drift.
    4. **Classify**: Produce the final numeric/categorical
       partition for downstream encoding and modelling.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full screening on training data + type
      enforcement on test data. Produces the clean feature set
      and type mapping.
    - **transform(data)**: Apply the trained type standards to
      any new dataset. Use for OOT scoring or production
      inference.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **Auto-repair over Failure**: Mixed-type columns (numeric
    stored as string, timedelta, bool) are repaired
    automatically when the dominant type is clear. The user is
    notified via the repair log but not blocked.

    **Conservative Drops**: A feature is only removed when it
    clearly fails a threshold. Borderline cases are kept and
    flagged in the `dropped` list for user review.

    **Fit-Transform Contract**: The type mapping learned during
    `fit` is strictly enforced during `transform`. Type
    drift between train and test is detected and reported.
    Critical mismatches (e.g., timedelta) raise and stop
    execution.

    **Flow Vars Are Sacred**: Variables designated as
    `flow_vars` (IDs, time keys) are preserved throughout
    the pipeline and excluded from screening.

    **Memory Management**: When `offload` is enabled in the
    flow context, data is converted to native Pandas dtypes
    after screening to sever Arrow reference chains and prevent
    memory spikes during model fitting.

    Examples
    --------
    >>> from aidflow.data import FeatureCleaner
    >>> cleaner = FeatureCleaner(ctx.setup)
    >>>
    >>> # Full screening + type enforcement
    >>> cleaner.fit(FeCleanerInput(
    ...     train_set = train_data,
    ...     test_set  = test_data,
    ...     dtypes    = dtypes
    ... ))
    >>>
    >>> # Apply to new data
    >>> result = cleaner.transform(data)
    >>>
    >>> # Access results
    >>> result.data          # Data with same dtypes as train_set
    >>> result.features      # All surviving features
    >>> result.num_features  # Numeric features
    >>> result.cat_features  # Categorical features
    >>> result.dropped       # Removed features + reasons    
    """

    def _save(self, save_path) -> None:
        features = {
            'features':     self.cols.vars_all, 
            'num_features': self.cols.vars_num, 
            'cat_features': self.cols.vars_cat
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)

    def _to_pd(self, data:pDF) -> pDF:
        """
        Manages memory allocation and data offloading during the screening phase.

        Behavior:
        - offload == False:
            Retains data in PyArrow format for as long as possible (until after 
            `obs_stabilizer` drops unnecessary rows). Test data remains in Arrow 
            format to maintain a minimal footprint throughout the pipeline.
        - offload == True:
            Optimized for aggressive memory recovery. Immediately after data cleaning, 
            data is converted to native Pandas dtypes, and all underlying Arrow arrays 
            are cleared (`None` + GC).
            Especially effective when cleaner drops a large number of columns 
            or applying aggressive precision downcasting (e.g., float64 to float32/16), 
            as it severs references to the uncompressed parent arrays.

        Rationale:
            Downstream ML libraries (e.g., LightGBM, Scikit-Learn) do not natively 
            support `ArrowDtype` and trigger on-the-fly conversions during training. 
            Converting data *after* observation filtering/subsampling and severing 
            the Arrow reference chains prevents severe memory spikes during model fitting.
        """
        if self.setup.offload:
            return Arrow.to_safe_datatype(data)
        return data
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs:FeCleanerInput, fx_choice:str = REG.AIG, **params):
        """
        Execute the full data quality screening and type enforcement.

        Screens training features against quality gates,
        auto-repairs type issues, and enforces the trained
        type mapping on the test set.

        Parameters
        ----------
        inputs : FeCleanerInput
            Training + test datasets and variable type
            classification.
        fx_choice : str, optional
            The fitting strategy identifier. Defaults to REG.AIG.
        **params : dict
            Runtime parameters passed to the internal fitter
            (max_miss, max_1val, max_var, max_corr, cat_max, downcast).

        Returns
        -------
        FeatureCleaner
            The fitted instance with clean feature set and
            type mapping available.
        """
        verify = ( 
            len(inputs.train_set) and 
            len(inputs.test_set)  and 
            len(inputs.dtypes) 
        )

        self.choice = self.get_choice(fx_choice, inputs)
        self.params = self.get_params(fx_choice, inputs, params)

        self.processor.fit(
            inputs, 
            fx_choice = self.choice,
            params    = self.params, 
            verify    = verify
        )
        self._save(inputs.save_path)
        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def transform(self, data: pDF | pa.Table, subset:list=None) -> FeCleanerResult:
        """
        Apply the trained type standards to new data.

        Parameters
        ----------
        data : pDF
            Input data to clean and type-enforce.
        subset : list, optional
            Specific features to process and return. 
            Default = None: all trained features. 

        Returns
        -------
        FeCleanerResult
            Cleaned data + feature classification.
        """
        data = Arrow.to_pandas(data)

        c_all = self.cols.vars_all
        c_sub = subset or data.columns
        cols  = [x for x in c_sub if x in c_all]

        miss = c_all if not cols else None
        df = self.processor.transform(data[cols], verify = miss)

        return FeCleanerResult(
            data         = df.data, 
            dtypes       = self.cols.vars_types, 
            dropped      = self.cols.vars_dropped,
            features     = self.cols.vars_all, 
            num_features = self.cols.vars_num, 
            cat_features = self.cols.vars_cat, 
        )    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def transform_dual(self, inputs:FeCleanerInput) -> FeCleanerResultDual:
        """
        Applies the cleaning logic to a provided dataframe.
        
        Mandatory match of signature methods fit - transform_dual for dual-set processing.

        Parameters
        ----------
        inputs : FeCleanerInput
            Configuration containing both train and test sets.

        Returns
        -------
        FeCleanerResultDual
            A result object containing processed data for both train and test sets.
        """
        df = self.processor.transform(inputs.train_set)
        dt = self.processor.transform(inputs.test_set)

        self._result = FeCleanerResultTest(
            data = self._to_pd(dt.data)
        )
        
        return FeCleanerResultDual(
            data         = self._to_pd(df.data), 
            test_data    = lambda: DataFrame(self._result.data),
            dtypes       = self.cols.vars_types, 
            dropped      = self.cols.vars_dropped,
            features     = self.cols.vars_all, 
            num_features = self.cols.vars_num, 
            cat_features = self.cols.vars_cat, 
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → CLEANING → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI(OptionUIBase):
    title   = 'DATA CLEANING PANEL'
    split   = 3
    Mapping = CleanParamsMapping
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → CLEANING → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FeatureCleanerUI(ModelUIBase):
    """
    AIG Flow Data Quality Screening.    
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    Responsible for cleaning the raw datasets, enforcing data types, 
    and identifying feature sets (numerical vs categorical).    
    """
    
    Model  = FeatureCleaner
    Option = OptionUI

    @wraps(FeatureCleaner.fit)
    def fit(self, *args, **kwargs):
        """Cleaner fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(), **kwargs)
        return self

    @wraps(FeatureCleaner.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> FeCleanerResultDual: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬