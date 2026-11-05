"""
Feature Stabilizer — Temporal Stationarity Screening.

This module implements the temporal stability analysis stage of the
AIDflow pipeline. It screens features (and optionally the target)
for time-series stationarity using the Augmented Dickey-Fuller
unit-root test, identifying features whose statistical properties
drift over time.

What It Does
------------
1. **Transform**: Convert features to time-series form:
   - Numeric: mean per period.
   - Categorical: dominant-category share proxy (continuous [0,1]).
2. **Test**: Apply the ADF unit-root test per feature (constant +
   constant/trend detrending, conservative p-value).
3. **Classify**: Label each feature as STABLE, UNSTABLE, or
   UNASSESSABLE (insufficient observations).
4. **Exclude**: Produce the default exclusion set (UNSTABLE +
   UNASSESSABLE) for downstream modelling.

Why Temporal Stability Matters
------------------------------
Non-stationary features violate a fundamental assumption of most
ML models: that the training distribution generalises to the
prediction distribution. A feature that is drifting (unit root)
may appear predictive in-sample but produce systematically wrong
predictions out-of-time.

Screening for stationarity **before** training catches this
failure mode at the source — it is cheaper to remove an unstable
feature during screening than to debug a model that silently
degrades in production.

Classification Logic
--------------------
- **STABLE** ('Y'): ADF p-value < α (default 0.10). The null
  hypothesis of a unit root is rejected.
- **UNSTABLE** ('N'): ADF p-value ≥ α. The null is not rejected.
- **UNASSESSABLE** ('NaN'): Fewer than 60 non-NaN observations.
  The ADF test has insufficient power (validated via Monte Carlo),
  so no verdict is issued. This is a data-sufficiency gate, not
  a quality judgment.

Pipeline Position
-----------------
This is the **third** stage (after Data Loader + Feature
Cleaner). It consumes the clean, well-typed feature set and
produces the stationarity-verified subset.

The stage can also be applied to the **target** variable to
verify that the prediction objective itself is stationary.

Dependencies
------------
Requires the Feature Cleaner stage to be completed (provides
the clean feature set with numeric/categorical partition).
"""

from ...deps.common             import  wraps
from ...deps.data               import  pDF

from ...core.parameters         import  (
                                        PRM,
                                        AIG,
                                        AIGFLOW,
                                        Parameters
                                        )
from ...core.registry           import  (
                                        ProjectModelRegistry,
                                        FitRegister, 
                                        FitRegOpt,   
                                        DashRegister,  
                                        RegGet
                                        )
from ...core.base               import  (
                                        FitterBase,
                                        TransformerBase,
                                        ProcessorBase,
                                        ModelProcessorBase,
                                        ModelInput,
                                        ModelResult,
                                        ModelBase,  
                                        OptionUIBase,
                                        ModelUIBase
                                        )
from ...core.factory            import  (
                                        Dataclass,
                                        DictClass
                                        )
from ...core.foundation         import  AigParam
from ...core.io                 import  (
                                        ModelIO, 
                                        DashIO
                                        )
from .fitters.fe_analyzer       import  (
                                        FeatureTemporalAnalyzer, 
                                        TimeStationarity,
                                        TAParamsBase,
                                        TAParamsCfg,
                                        TAResult
                                        )
from .utils_ui                  import  PanelUtilsUI


MODEL = AIGFLOW.Data.fe_stabilizer
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG DATA STAGE ► TEMPORAL STABILITY ► Learn and Ensure Features Temporal Stationarity'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG = 'FeTemporalAnalyzer'

class LBL(DictClass):
    TT = ('TRAIN', 'TEST')

class VAR(DictClass):
    """Fitter-Transformer variables link."""
    NUM = 'vars_num'
    CAT = 'vars_cat'
    OUT = 'vars_dropped'
    N_O = 'vars_n_o'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → TEMPORAL STABILITY → FITTER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class TAParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(TAParamsBase)
    setup  = {
        PRM.SCREEN.th_st.param  : params.th_stationarity
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
    class TimeSeriesFitter(TAParamsBase):
        
        config: dict|TAParamsCfg = TAParamsCfg(
            time_var = AIG.time_var.value,
            verbose  = True
        )

        def _build(self, params:TAParamsBase, config:dict) -> FeatureTemporalAnalyzer:            
            setup  = dict(params) | config
            return FeatureTemporalAnalyzer(**setup)

        def get_extra(self, ts_stat:pDF) -> tuple[list, list]:
            return TimeStationarity.get_status_unstable(ts_stat)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class FitParams(TAParamsBase):
    """Fitter params input design."""       
    num_features: list
    cat_features: list
    target:       str
    is_target:    bool = False

@Dataclass.base
class FitResult:
    """Fitter output design."""
    th:       float
    TRAIN: TAResult
    TEST:  TAResult

@Dataclass.base
class FitStats:
    target:   FitResult
    features: FitResult = None
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Fitter(FitterBase[FitParams]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """                     

    FREG = FREG
    DREG = DREG

    def _get_params(self):
        """API fork: exo or target TS."""
        if not self.params.is_target:
            return self.params.num_features, self.params.cat_features
        return [self.params.target], []
    
    def _process_ta(self, train:pDF, test:pDF) -> dict:
        """Time Series data processing and analysis."""
        num, cat = self._get_params()
        return {
            tx: TAResult( **self.model.process(d, num, cat) ) 
            for tx, d in zip(LBL.TT, [train, test])
        }    
             
    def _get_ta_result_update(self, th_ts:float) -> dict:
        """Update given new th based obtained TS analysis."""
        return {
            tx: TAResult( 
                **self.model.update( self.result[tx], th_ts )
            ) for tx in LBL.TT 
        }  

    def _get_target_stats(self) -> None | FitResult:
        t: ModelBase = ProjectModelRegistry.get_model(
            self.setup.project_signature, MODEL
        )
        if t is None:
            return None
        return t.stats.target
    
    def _set_stats(self) -> None:        
        if self.params.is_target:
            self.stats = FitStats(target = self.result)
            return
        
        self.stats = FitStats(
            target = self._get_target_stats(),
            features = self.result
        )

    def _update_stats(self) -> None:
        self.stats = FitStats(
            target = self.stats.target,
            features = self.result
        )

    def fit(self, train:pDF, test:pDF):
        """
        Analyze and clean variables of temporal instability.

        Parameters
        ----------
        train :   Input Data for temporal analysis
        test :    Test data part for temporal analysis
        verbose : bool → True shows dashboard and stats
        """                 
        self.result: FitResult = FitResult(
            th = self.params.th_stationarity, 
            **self._process_ta(train, test)
        )
        self._set_stats()
        self._init_dash()
        return self

    def update(self, th_ts:float):        
        """Update TS stats given stationarity TH update."""
        self.result = FitResult(
            th = th_ts, 
            **self._get_ta_result_update(th_ts)
        )
        self._update_stats()
        self._init_dash(ipy_header = self.dashboard._get_ipy_header())
        return self

    def _update_vars_by_ui(self) -> None: 
        """Fitted by VarStabilizerUI for reporting purposes."""        
        self.dashboard._update_vars_out_user()
    
    def _get_ui_vars_out(self) -> list|None:
        if self.verbose:
            return self.dashboard._get_vars_out_user()
        io = DashIO(self.setup)
        return io.load(custom_path=io.get_custom_selection_path(MODEL))
    
    def _get_result_num_cat(self, out:list) -> tuple[list, list]:
        """Provide Numerical-Categorical Vars from TS with respect to input sorting."""
        num = [x for x in self.params.num_features if x not in out]
        cat = [x for x in self.params.cat_features if x not in out]
        return num, cat
    
    def get_updated_vars(self) -> dict:
        """Get fitter result with respect to user dynamic changes in UI."""
        ts_stat = self.result.TRAIN.ts_stat
        vars_ko = self._get_ui_vars_out()
        
        if vars_ko is None:
            vars_ko = self.model.get_extra(ts_stat)

        num, cat = self._get_result_num_cat(vars_ko)
        return {VAR.NUM: num, VAR.CAT: cat, VAR.N_O: num+cat, VAR.OUT: vars_ko}      
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → TEMPORAL STABILITY → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransParams:
    """Transformer params input design."""       
    flow_vars: list

@Dataclass.base
class TransResData:
    """Transfomer data output design."""
    data: pDF

@Dataclass.base
class TransResCols:
    """Transfomer output design."""
    vars_all:       list 
    vars_num:       list
    vars_cat:       list
    vars_n_o:       list
    vars_dropped:   list    
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class TransUtils:
    """Data Temporal Stability Application."""       

    def __init__(self, trans_params:TransParams, fitter:Fitter):
        self.t_prm  = trans_params
        self.fitter = fitter

    def get_cols(self) -> TransResCols:
        """Temporal Stability Transformer Variables Result."""    
        res = self.fitter.get_updated_vars()
        va  = self.t_prm.flow_vars + res[VAR.N_O]
        return TransResCols(
            vars_all = va, 
            vars_num = res[VAR.NUM],
            vars_cat = res[VAR.CAT],
            vars_n_o = res[VAR.N_O],
            vars_dropped = res[VAR.OUT],
        )
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Transformer(TransformerBase[Fitter, FitResult, TransResCols]):
    """
    Transformer meta-processing between Fitter-Model.
    
    Handles the post-processing of Fitter results. Its primary role is to standardize 
    the raw output from various fitters into a unified TransResCols, TransResData
    format, enabling seamless transition regardless of the underlying fitter choice.
    """

    ModelFitter = Fitter

    def _fit_data(self, inputs: "FeStabilizerInput"):
        return inputs.train_set, inputs.test_set
    
    def _set_cols(self):
        return TransUtils(self.t_prm, self.fitter).get_cols()    
    
    def transform(self, data:pDF) -> TransResData:
        """Model meta-transform."""
        cols = [x for x in self.cols.vars_all if x in data.columns]
        return TransResData(
            data = data[cols]
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → TEMPORAL STABILITY → PROCESSOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ExtendedProcessor(ProcessorBase):
    """Transformer Extended Processor."""  

    fx: Transformer

    def update(self, th_ts:float) -> None:
        """Update fitted function."""
        if th_ts is None: 
              self.fx.trans_update()
        else: self.fx.fit_update(th_ts)        
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Processor(ModelProcessorBase[Transformer, TransResCols, TransResData]):
    """
    Logic Processor.
    
    Coordinates the lifecycle between FitChoice selection and Parameter resolution. 
    It ensures that params and dependencies are correctly injected into the 
    Transformer - Fitter before execution begins.
    """  
    
    MODEL  = MODEL
    DESC   = DESC

    Config  = TAParamsCfg
    Mapping = TAParamsMapping

    ModelTransformer  = Transformer
    ExtendedProcessor = ExtendedProcessor
    
    def __init__(self, setup:Parameters, is_target:bool):
        super().__init__(setup)
        self.is_target = is_target
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_choice(self, fx_choice:str, inputs:"FeStabilizerInput"):
        defaults = self.Config(
            time_var = AigParam.di(inputs.time_var, self.setup.time_var),
            verbose  = AigParam.di(inputs.verbose,  self.setup.verbose)
        )  
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)   
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
        
    def get_params(self, fx_choice:str, inputs:"FeStabilizerInput", params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params)
        
        fit = FitParams(
            num_features = inputs.num_features,
            cat_features = inputs.cat_features,
            target       = AigParam.di(inputs.target, self.setup.target),
            is_target    = self.is_target,
            **params
        )
        trn = TransParams(
            flow_vars = self.setup.flow_vars,
        )
        return fit, self.setup, trn
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬    
# DATA → TEMPORAL STABILITY
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬   
@Dataclass.base(dynamic_params=True)
class FeStabilizerInput(ModelInput):
    """
    FeatureStabilizer fit input specification.

    Defines the datasets and feature lists required to execute
    the temporal stationarity screening.

    Attributes
    ----------
    train_set : pDF
        Training data (in-sample). The primary dataset on which
        stationarity is assessed.
    test_set : pDF | None, default=None
        Test data (OOT). Used to verify stationarity on the
        out-of-time segment. If None, only train is assessed.
    num_features : list | None, default=None
        Numerical features to screen. If None, all numeric
        features from the upstream cleaner are used.
    cat_features : list | None, default=None
        Categorical features to screen. If None, all categorical
        features from the upstream cleaner are used.
    """
    train_set:      pDF
    test_set:       pDF |None = None
    num_features:   list|None = None
    cat_features:   list|None = None

@Dataclass.base
class FeStabilizerResult(ModelResult):
    """
    FeatureStabilizer output — the stationarity-verified feature set.

    The final result of the temporal stability screening stage.
    Contains the stationarity-filtered data and the complete
    feature partition that drives all downstream modelling.

    Attributes
    ----------
    data : pDF
        Stationarity-verified data. Features classified as
        UNSTABLE or UNASSESSABLE are removed. The result is
        in canonical order (flow vars first, then stable
        numeric, then stable categorical).
    features : list
        All surviving features (flow vars + stable numeric +
        stable categorical), in canonical pipeline order.
    num_features : list
        Stable numerical features.
    cat_features : list
        Stable categorical features.
    dropped : list
        Features removed due to temporal instability (UNSTABLE
        or UNASSESSABLE). Each entry includes the ADF p-value
        and the reason for exclusion.

    Notes
    -----
    **Conservative Exclusion**: A feature is excluded if it is
    UNSTABLE or UNASSESSABLE in on the train segment. Test part
    serves the context for the user decision.

    **User Override**: All auto-decisions are reversible. The
    user can adjust the significance threshold interactively
    (via `update`) or manually opt features back in through
    the UI panel.

    **Categorical Proxy**: Categorical features are screened
    using a continuous proxy (dominant-category temporal share).
    A stable proxy implies the categorical distribution is
    not drifting over time.
    """
    data:           pDF
    features:       list 
    num_features:   list
    cat_features:   list
    dropped:        list 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class FeatureStabilizer(Processor, ModelBase):
    """
    Feature Stabilizer — Temporal Stationarity Screening.

    The temporal stability gate of the AIDflow pipeline.
    Screens features for time-series stationarity using the
    Augmented Dickey-Fuller unit-root test, identifying features
    whose statistical properties drift over time and are therefore
    unreliable for production prediction.

    What It Does
    ------------
    1. **Transform**: Convert features to time-series form
       (numeric: mean per period; categorical: dominant-category
       share proxy).
    2. **Test**: Apply ADF unit-root test per feature (constant +
       constant/trend, conservative p-value).
    3. **Classify**: STABLE / UNSTABLE / UNASSESSABLE per feature.
    4. **Exclude**: Produce the default exclusion set (UNSTABLE +
       UNASSESSABLE).

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full stationarity screening on train + test.
      Produces the stable feature set and exclusion list.
    - **transform(data)**: Apply the trained feature filter to new
      data. Use for OOT scoring or production inference.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **Verifiability Gate**: Features with fewer than 60 non-NaN
    observations are classified as UNASSESSABLE (not UNSTABLE).
    At n < 60, the ADF test statistic has not converged to its
    asymptotic distribution, so no reliable verdict can be issued.
    This is a data-sufficiency gate, not a quality judgment.

    **Interactive Override**: The significance threshold can be
    adjusted interactively without re-running the ADF test
    (O(n) reclassification vs O(n × T) re-computation). The
    user can also manually opt any feature back in.

    **Categorical Features**: Categorical variables are reduced
    to a continuous proxy (temporal share of dominant category
    set) before ADF testing, enabling uniform treatment of all
    feature types.

    **Dual-Segment**: Stationarity is assessed on both the
    training and test segments. A feature is excluded if it is
    unstable in either segment (conservative).

    **Target Analysis**: The stage can also be applied to the
    target variable to verify that the prediction objective
    itself is stationary.
    
    Examples
    --------
    >>> from aidflow.data import FeatureStabilizer
    >>> stabilizer = FeatureStabilizer(ctx.setup)
    >>>
    >>> # Full stationarity screening
    >>> stabilizer.fit(FeStabilizerInput(
    ...     train_set    = train_data,
    ...     test_set     = test_data,
    ...     num_features = num_fes,
    ...     cat_features = cat_fes
    ... ))
    >>>
    >>> # Adjust threshold interactively (no re-computation)
    >>> stabilizer.update(th_st=0.05)
    >>>
    >>> # Apply to new data
    >>> result = stabilizer.transform(new_data)
    >>>
    >>> # Access results
    >>> result.data          # Data subset to stable fes
    >>> result.features      # Stable features
    >>> result.num_features  # Stable numeric
    >>> result.cat_features  # Stable categorical
    >>> result.dropped       # Unstable + unassessable    
    """    

    def _save(self, save_path) -> None:
        features = {
            'features':     self.cols.vars_all, 
            'num_features': self.cols.vars_num, 
            'cat_features': self.cols.vars_cat
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path) 
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs:FeStabilizerInput, fx_choice:str = REG.AIG, **params):
        """
        Execute the full temporal stationarity screening.

        Assesses all features on both the training and test
        segments using the ADF unit-root test, and produces
        the stable feature set.

        Parameters
        ----------
        inputs : FeStabilizerInput
            Training + test datasets and feature lists.
        fx_choice : str, optional
            The fitting strategy identifier. Defaults to REG.AIG.
        **params : dict
            Additional parameters passed to the internal fitter
            (th_stationarity).

        Returns
        -------
        FeatureStabilizer
            The fitted instance with the stable feature set
            and exclusion list available.
        """
        verify = ( 
            len(inputs.train_set) and 
            ( len(inputs.num_features or []) + 
              len(inputs.cat_features or []) )
        )
        
        self.choice = self.get_choice(fx_choice, inputs)
        self.params = self.get_params(fx_choice, inputs, params)

        self.processor.fit(
            inputs,
            fx_choice = self.choice,
            params    = self.params,
            verify    = verify
        )
        self.custom_path = inputs.save_path
        self._save(self.custom_path)        
        self._set_update_data(lambda: inputs.train_set)
        return self  
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def update(self, *args, **kwargs):
        """
        Update the stationarity threshold and refresh the feature set.

        Re-classifies features under the new threshold using
        pre-computed ADF p-values (no re-testing).

        Parameters
        ----------
        *args : Any
            Positional arguments for the update process.
        **kwargs : dict
            Keyword arguments including the new threshold
            (th_st).

        Returns
        -------
        FeatureStabilizer
            The updated instance with the new feature set.
        """
        self.processor.update(kwargs.get(PRM.SCREEN.th_st.param))
        self._save(self.custom_path)
        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def transform(self, data:pDF) -> FeStabilizerResult:
        """
        Apply the trained stationarity filter to new data.

        Parameters
        ----------
        data : pDF
            Input data to filter.

        Returns
        -------
        FeStabilizerResult
            Stationarity-verified data + feature classification.
        """    
        df = self.processor.transform(data)
        return FeStabilizerResult(
            data         = df.data if not self._is_train_data(data) else None, 
            features     = self.cols.vars_all, 
            num_features = self.cols.vars_num, 
            cat_features = self.cols.vars_cat, 
            dropped      = self.cols.vars_dropped,
        )         
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬    



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → TEMPORAL STABILITY → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI(OptionUIBase):
    """UI Dynamic Parameters Panel."""

    title = 'TEMPORAL STABILITY'

    def __init__(self, setup:Parameters, input:FeStabilizerInput, is_target:bool):
        prm = TAParamsMapping(setup, default=True)        
        self.wg = PanelUtilsUI(prm, self._title(is_target), is_ta = True, desc = not is_target) 

    @property
    def slider(self):
        return self.wg.slider()

    def _title(self, is_target:bool) -> str:
        return f"{self.title} OF {'TARGET' if is_target else 'FEATURES'}"
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → TEMPORAL STABILITY → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FeatureStabilizerUI(ModelUIBase[FeatureStabilizer, OptionUI]):
    """
    AIG Flow Data Temporal Stability.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    Identifies and removes unstable features based on temporal 
    consistency checks to prevent data leakage and overfitting.    
    """
    
    Model  = FeatureStabilizer
    Option = OptionUI

    def __init__(self, 
        setup:  Parameters, 
        inputs: FeStabilizerInput, 
        is_target: bool=False
    ): 
        super().__init__(setup, inputs, runner_wg_id = int(is_target), is_target = is_target)
        self.is_target = is_target

    def __repr__(self):
        if self.is_target:
            return "<GraphView>"
        return repr(self.model)

    def custom_control(self, fx):
        """Custom UI widget."""
        return self.option_ui.slider.observe(fx, names='value')
        
    def custom_update_vars(self, *args, **kwargs):
        """User Custom Variables Selection and propagation."""
        return self.model.update_transform()    
    
    def chosen_vars_update(self, *args, **kwargs):
        """User Custom Variables Selection Save and Summary."""
        self.model.fx.fitter._update_vars_by_ui()
        
    @wraps(FeatureStabilizer.fit)
    def fit(self, *args, **kwargs): 
        """Temporal Stability fit UI wrapper."""                        
        self.model_ui.fit(   
            *args,
            ui_params = self._get_ui_fit_defaults(
                output_wait                 = True,
                parent_has_update_FX_method = True,
                gui_static_before_confirm_1 = None,
                custom_control_widget       = self.custom_control,
                custom_update_fitted_8      = self.custom_update_vars if not self.is_target else None,
                gui_dynamic_w_fitted_8      = self.chosen_vars_update if not self.is_target else None                
            ),
            **kwargs
        )
        return self

    @wraps(FeatureStabilizer.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)
    
    @property
    def fitted(self) -> FeStabilizerResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬