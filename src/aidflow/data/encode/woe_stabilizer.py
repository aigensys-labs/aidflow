"""
WOE Feature Stability Analysis — Time-Wise Consistency Gate.

This module implements the final feature quality gate of the AIDflow
pipeline. It verifies that each WOE-encoded feature maintains
consistent predictive behaviour over time, excluding features whose
signal is temporally unstable before they enter model training.

What It Does
------------
1. **Groups** observations by time unit (week, month, or native
   resolution).
2. **Tests** each WOE feature's bin-level target-rate consistency
   across time units.
3. **Scores** the overall feature-level stability (proportion of
   time units where the feature maintains its separation).
4. **Gates** features on the stability cutoff (default 0.65).
   Features below the threshold are dropped.
5. **Outputs** the final feature set — the WOE features that are
   both predictive and time-stable.

What is Measured?
-----------------
- **Bin-level consistency**: For each WOE bin, is the target
  rate consistent over time? A bin whose event rate fluctuates
  wildly is capturing noise, not a stable signal.
- **Feature-level stability**: Is the overall predictive power
  (separation direction and magnitude) stable over time? A
  feature whose discriminative power degrades or flips over
  time is unreliable.

Why This Matters
----------------
A WOE feature can have excellent average Gini yet be driven by
a few anomalous time periods. If the feature's signal is not
temporally consistent, it will fail in production when the
underlying process shifts. This gate catches such "unstable"
features and prevents them from entering the model.

Role in the Pipeline
--------------------
This is the **final feature quality gate**. It runs after WOE
transformation and before model training. Its output is the
definitive feature set for the GLM models.

The stabilizer also supports **interactive threshold tuning**:
after `fit`, the user can call `update` with a new stability
cutoff to re-partition features without re-running the full
consistency test. This enables real-time feature selection in
the dashboard.

Dependencies
------------
Requires the WOE Encoder stage to be completed (the stabilizer
operates on WOE-encoded features).
"""

from ...deps.common             import  wraps
from ...deps.data               import  pDF

from ...core.parameters         import  (
                                        GEN,
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
                                        FIT, 
                                        FitterBase,
                                        TransformerBase,
                                        ProcessorBase, 
                                        ModelProcessorBase,
                                        ModelEnvelope,
                                        ModelInput,
                                        ModelResult,
                                        ModelBase,  
                                        OptionUIBase,
                                        ModelUIBase                                                                            
                                        )
from ...core.base.adapter       import  ModelAdapterBase
from ...core.factory            import  (
                                        Dataclass,
                                        DictClass
                                        )
from ...core.interop            import  Pd
from ...core.foundation         import  AigParam
from ...core.io                 import  (
                                        ModelIO, 
                                        DashIO
                                        )
from ...core.standard           import  VarStd

from .woe_encoder               import  WoeEncoder
from .fitters.woe_analyzer      import  (
                                        WoeStabilityAnalyzer,
                                        FitParamsBase,
                                        FitParamsCfg,
                                        FitResult,
                                        FitStats
                                        )


MODEL = AIGFLOW.Encoding.woe_stabilizer
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG DATA STAGE ► WOE STABILIZE ► Analyze and Prune WOE Features on Predictive Stability'
DEPS  = AIGFLOW.Encoding.woe_encoder


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG = 'WoeStabilizer'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → FIT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class WoeParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(FitParamsBase)
    setup  = {
        PRM.ENCODE.woe_stability.param   : (params.stability_cutoff, 'Stability Cutoff'     ),
        PRM.ENCODE.woe_alpha.param       : (params.alpha,            'Stability Bins alpha' ),
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
    class WoeStabilityAnalyzerFitter(FitParamsBase):

        config: dict|FitParamsCfg = FitParamsCfg(
            target          = None,
            time_var        = AIG.time_var.value,
            time_unit       = None,
            max_time_units  = None,
            verbose         = True
        )

        @property
        def _model(self) -> WoeStabilityAnalyzer:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> WoeStabilityAnalyzer:            
            setup = dict(params) | config
            return WoeStabilityAnalyzer(**setup)

        def get_result(self) -> FitResult:
            return self._model.result
        
        def get_stats(self) -> FitStats:
            return self._model.stats      
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@Dataclass.base
class FitParams(FitParamsBase):
    """Fitter params input design."""    
    woe_features: list
    woe_encoder:  ModelAdapterBase = None

@Dataclass.base
class FitUpdateData:
    """Fitter update data design."""
    train_woe: pDF
    test_woe:  pDF

@Dataclass.base
class FitUpdatePrm:
    """Fitter update data design."""
    th:             float
    alpha:          float
    is_th_only:     bool = False

@Dataclass.base
class DashData:
    """Dash-Fitter vars connection design."""
    th:            float
    target:        str
    consistency:   FitStats
    test_data_woe: pDF
    is_th_only:    bool
    woe_encoder:   WoeEncoder
    dash_instance: object = None
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FitBase(FitterBase[FitParams]):    

    FREG = FREG
    DREG = DREG
    
    @property
    def _woe_encoder(self) -> ModelAdapterBase:
        return self.params.woe_encoder
    
    @property
    def _update_data(self) -> FitUpdateData:
        return self._relay

    @property
    def _update_prm(self) -> FitUpdatePrm:
        return self._update_params
    
    def _set_relay_data(self, data:FitUpdateData) -> None:
        self._relay = data

    def _set_update_params(self, data:FitUpdatePrm) -> None:
        self._update_params = data
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_model(self, th:float, alpha:float) -> None:
        self.model = self.fx.build(
            params = self.params, **FitParamsBase(stability_cutoff=th, alpha=alpha)
        )

    def _fit_model(self, train_woe:pDF, test_woe:pDF) -> None:
        self.model.fit(train_woe, test_woe, self.params.woe_features)   

    def _update_model(self, stats:FitStats) -> None:
        self.model.update(stats) 
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_mapped_model_cfg(self) -> dict:
        c = self.model.get_params()['config']
        p = Dataclass.schema(FitParamsCfg)
        return {
            GEN.target.param:   c[p.target],
            AIG.time_var.param: c[p.time_var],
        }
    
    def _get_model_cfg_vals(self, cols:list|str = None) -> list|str:
        mc = self._get_mapped_model_cfg()
        if cols is None:
            return list( mc.values() )
        if isinstance(cols, str):
            return mc[cols]
        return [mc[x] for x in cols]    
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _data_encoder(self, data:pDF) -> pDF:
        fes = VarStd.fe_to_var(self.params.woe_features)
        return self._woe_encoder.predict(data, subset = fes)
    
    @staticmethod
    def _is_data_woe_misaligned(data:pDF, data_woe:pDF) -> bool:
        return (
            ( data_woe is None ) or 
            ( len(data.index) != len(data_woe.index) ) or
            ( data.index != data_woe.index ).sum()
        )

    def _validate_data(self, data:pDF) -> None:
        cols = self._get_model_cfg_vals()
        if ( 
            ( data is not None ) and
            ( len(data) ) and
            all([x in data.columns for x in cols])
        ): return
        raise ValueError('AIG SAFETY'
        '\n► Input data passing to stability test are internally encoded to WOE by trained WOE encoder.'
        '\n► Data must contain orig. features needed to encode selected woe_features.'
       f'\n► Stability test needs extra columns present in the data to conduct the test: {', '.join(cols)}.'
        )

    def _data_preprocess(self, data:pDF, data_woe:pDF=None) -> pDF|None:
        self._validate_data(data)

        if self._is_data_woe_misaligned(data, data_woe):
            data_woe = self._data_encoder(data)

        cols = self._get_model_cfg_vals()
        denc = data_woe     

        cols_na = [x for x in cols if x not in denc.columns]
        if cols_na:
            dcfg = data.loc[denc.index, cols_na]
            return Pd.concat([dcfg, denc], axis=1)

        return denc
    
    def _data_to_woe(self, data:dict[str,tuple]) -> tuple[pDF,pDF]:
        return (
            self._data_preprocess(*data['train']),
            self._data_preprocess(*data['test'])
    )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _set_update_data(self, train_woe:pDF, test_woe:pDF) -> None:
        self._set_relay_data(FitUpdateData(
            train_woe = train_woe,
            test_woe  = test_woe
        ))

    def _set_update_prm_init(self) -> None:
        self._set_update_params(FitUpdatePrm(
            th    = self.params.stability_cutoff,
            alpha = self.params.alpha
        ))

    def _set_update_prm(self, th:float, alpha:float) -> None:
        self._set_update_params(FitUpdatePrm(
            th         = th,
            alpha      = alpha,
            is_th_only = self._is_th_update_only(th, alpha)
        ))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _is_th_update_only(self, th:float=None, alpha:float=None) -> bool:
        return (
            self._update_prm.th != th and 
            self._update_prm.alpha == alpha
        ) or not self._is_update(th, alpha) # no change → dashboard will reappear

    def _is_update(self, th:float, alpha:float) -> bool:        
        return (
            self._update_prm.th != th or 
            self._update_prm.alpha != alpha
        )
    
    def _update_fit(self, th:float, alpha:float) -> None:
        if not self._is_update(th, alpha):
            return
        self._set_model(th, alpha)
        if self._is_th_update_only(th, alpha):
            self._update_model(self.stats)
        else:
            self._fit_model(
                self._update_data.train_woe,
                self._update_data.test_woe
            )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_result_dash(self, 
        th: float = None, 
        dash_instance: object = None, 
    ) -> None:
        self.result: FitResult = self.model.get_result()
        self.stats:   FitStats = self.model.get_stats()

        self._init_dash(dash_data = DashData(
            th            = th,
            target        = self._get_model_cfg_vals(GEN.target.param),
            consistency   = self.stats,
            test_data_woe = self._update_data.test_woe,
            is_th_only    = self._update_prm.is_th_only,
            woe_encoder   = self._woe_encoder.model, 
            dash_instance = dash_instance
        ))
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Fitter(FitBase):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """     

    def fit(self, 
        train_set:  pDF, 
        test_set:   pDF, 
        train_woe:  pDF = None, 
    ): 
        """
        Analyze and Learn Features WOE predictive stability.

        Parameters
        ----------
        train_set : Input orig. train data for woe stability analysis.
        test_set  : Input orig. test data for woe stability analysis.
        train_woe : Optional, Input train data encoded as woe for woe stability analysis.
        test_woe  : Optionalm Input test data encoded as woe for woe stability analysis.
        """
        d0, d1 = self._data_to_woe({
            'train': (train_set, train_woe), 
            'test':  (test_set,  None)
        })
        self._fit_model(d0, d1)
        self._set_update_prm_init()
        self._set_update_data(d0, d1)
        self._set_result_dash(self.params.stability_cutoff)
        return self
    
    def update(self, th:float, alpha:float):
        """Updates WOE Features which suffer of Unstable Prediction."""   
        self._update_fit(th, alpha) 
        self._set_update_prm(th, alpha)                        
        self._set_result_dash(th, self.dashboard)
        return self
        
    def _update_vars_by_ui(self) -> None: 
        """Fitted by WoeStabilizerUI for reporting purposes."""        
        self.dashboard._update_vars_out_user()
    
    def _get_ui_vars_out(self) -> list|None:
        if self.verbose:        
            return self.dashboard._get_vars_out_user()
        io = DashIO(self.setup)
        return io.load(custom_path=io.get_custom_selection_path(MODEL))    
    
    def _get_result_woe(self, out:list) -> list:
        """Provide WOE Chosen Features."""
        woe = [x for x in self.params.woe_features if x not in out]
        return woe
    
    def get_updated_result(self) -> FitResult:
        """Get fitter result with respect to user dynamic changes in UI."""
        out = self._get_ui_vars_out()

        if out is None:
            out = self.result.vars_dropped
        
        woe = self._get_result_woe(out)
        
        return FitResult(
            vars_woe     = woe,
            vars_dropped = out,
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    vars_woe:       list
    vars_dropped:   list

@Dataclass.base
class TransResData:
    """Transfomer data output design."""
    data: pDF
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.verify
class TransUtils:
    """Transformer Utilities."""       

    def __init__(self, fitter:Fitter):
        self.fitter = fitter

    def get_cols(self) -> TransResCols:
        """Temporal Stability Transformer Features Result."""    
        res = self.fitter.get_updated_result()
        return TransResCols(
                vars_woe      = res.vars_woe,
                vars_dropped  = res.vars_dropped,
            )
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Transformer(TransformerBase[Fitter, FitStats, TransResCols]):
    """
    Transformer meta-processing between Fitter-Model.
    
    Handles the post-processing of Fitter results. Its primary role is to standardize 
    the raw output from various fitters into a unified TransResCols, TransResData
    format, enabling seamless transition regardless of the underlying fitter choice.
    """

    ModelFitter = Fitter

    def _fit_data(self, inputs: "WoeStabilizerInput", train_woe:pDF): 
        return inputs.train_set, inputs.test_set, train_woe
    
    def _set_cols(self):
        return TransUtils(self.fitter).get_cols()
            
    def transform(self, data:pDF) -> TransResData: 
        """Model meta-transform."""   
        cols = [x for x in self.cols.vars_woe if x in data.columns]
        return TransResData(
            data = data[cols]
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → PROCESSOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ExtendedProcessor(ProcessorBase):
    """Transformer Extended Processor."""  

    fx: Transformer

    def update(self, th:float, alpha:float) -> None:
        """Update fitted function."""
        if th is None: 
              self.fx.trans_update()
        else: self.fx.fit_update(th, alpha)  
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

    Config  = FitParamsCfg
    Mapping = WoeParamsMapping

    ModelTransformer  = Transformer
    ExtendedProcessor = ExtendedProcessor    
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_choice(self, fx_choice:str, inputs:"WoeStabilizerInput"):
        defaults = self.Config(
            target    = AigParam.di(inputs.target,   self.setup.target),
            verbose   = AigParam.di(inputs.verbose,  self.setup.verbose),
            time_var  = AigParam.di(inputs.time_var, self.setup.time_var),
            time_unit = None,
            max_time_units = None,
        )        
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_params(self, fx_choice:str, inputs:"WoeStabilizerInput", params:dict):
        params  = self.get_fit_params(fx_choice, inputs.fx_params, params)        
        encoder = self.get_model_dependecy(DEPS)
        
        fit = FitParams(
            woe_features = inputs.woe_features, 
            woe_encoder  = ModelAdapterBase(encoder),
            **params
        )
        return fit, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → API
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class WoeStabilizerInput(ModelInput):
    """
    WOE Stabilizer fit input specification.

    Defines the training and test data required to execute the
    temporal stability screening on WOE features.

    Attributes
    ----------
    train_set : pDF
        Training data. Must contain the target column, the time
        variable, and WOE-encoded features (or original features
        for internal WOE encoding).
    test_set : pDF
        Out-of-time test data. Same structure as `train_set`.
        Used for OOS stability verification.
    woe_features : list | None, default=None
        WOE features to test. If None, all WOE-suffixed columns
        in the data are used.

    Notes
    -----
    **WOE-Encoded Required**: The stability test operates on
    WOE-encoded features. If the input data is not yet WOE-
    transformed, the model internally encodes it using the
    trained WOE Encoder from the previous stage.

    **Time Variable Required**: The data must contain the
    specified time variable column (e.g., `'AIG_YY_MM'`) to
    group observations into time units for the consistency test.
    """
    train_set:      pDF
    test_set:       pDF
    woe_features:   list|None = None  

@Dataclass.base(dynamic_params=True)
class WoeStabilizerEnvelopeCfg:
    train_woe:   pDF | None

@Dataclass.base(dynamic_params=True)
class WoeStabilizerEnvelope(ModelEnvelope):
    """
    Extended input envelope carrying pre-computed WOE data.

    When the WOE-encoded training data is already available
    (e.g., from a previous pipeline step), it can be passed
    via this envelope to avoid redundant encoding.

    Attributes
    ----------
    inputs : WoeStabilizerInput
        Primary input specification.
    config : WoeStabilizerEnvelopeCfg
        Optional pre-computed WOE data.
    """
    inputs: WoeStabilizerInput 
    config: WoeStabilizerEnvelopeCfg

@Dataclass.base
class WoeStabilizerFeatureResult:
    """
    WOE Stabilizer output features by two naming conventions.

    Attributes
    ----------
    woe : list
        Features by woe naming convention.
    raw : list
        Original (raw input) feature names for the
        woe features — the business-facing identifiers.    
    """
    woe: list
    raw: list

@Dataclass.base
class WoeStabilizerResult(ModelResult):
    """
    WOE Stabilizer output — the stabilized feature set.

    The final result of the temporal stability screening.
    Contains the stabilized WOE feature data along with full
    feature lineage metadata for audit and downstream consumption.

    Attributes
    ----------
    data : pDF
        WOE-encoded data restricted to the stable features only.
        (May be None for in-flow pipeline steps where the data
        is passed forward without materialization.)
    features : WoeStabilizerFeatureResult
        Stable features: WOE names + original business names.
    dropped : WoeStabilizerFeatureResult
        Unstable (dropped) features: WOE names + original
        business names. The implicit reason is temporal
        instability — the feature's target-separation was not
        consistent across time units.
    """    
    data:       pDF
    features:   WoeStabilizerFeatureResult
    dropped:    WoeStabilizerFeatureResult
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class WoeStabilizer(Processor, ModelBase):     
    """
    WOE Feature Stabilizer — Temporal Consistency Quality Gate.

    The final feature quality gate of the AIDflow pipeline.
    Screens WOE-encoded features for temporal stability, ensuring
    that only time-consistent signals proceed to model training.

    What It Does
    ------------
    1. **Tests** each WOE feature's bin-level target rate
       consistency across time units (train + OOS test).
    2. **Scores** each feature's overall stability (proportion
       of time units where the feature maintains its
       target-separation direction and magnitude).
    3. **Screens** features against `stability_cutoff` —
       those below the threshold are dropped.
    4. **Selects** the stable WOE feature subset for
       downstream model training.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full stability screening on train + test
      data. Produces the stable feature verdict.
    - **transform(data)**: Selects the stable WOE columns from
      the input data (pass-through — no value transformation).

    Update Protocol
    ---------------
    After `fit`, the `update` method allows interactive
    threshold tuning without re-running the full consistency
    test. Adjust `stability_cutoff` or `alpha` to
    re-partition features into stable/unstable. This is the
    basis for the UI-driven feature curation workflow.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **What "stability" means**: A WOE feature is stable if,
    across time units, each bin consistently separates the
    target (event rate direction and magnitude remain
    consistent). Unstable features may have excellent average
    Gini but are driven by a few anomalous periods — they will
    fail in production when the regime shifts.

    **Train + Test verification**: Stability is checked on both
    the training sample and the out-of-time test sample. A
    feature must pass on training to be kept; the test result
    provides additional OOS confidence.

    **Transform is a selection**: The `transform` method does
    not modify values — it selects only the stable WOE columns
    from the input. The WOE values themselves were computed in
    the previous (WOE encoder) stage.

    **Final Gate**: This is the last stage that can modify the
    feature set before model training. All downstream agents
    consume the output of this stage.

    Examples
    --------
    >>> from aidflow.encoding import WoeStabilizer
    >>> stabilizer = WoeStabilizer(ctx.setup)
    >>>
    >>> # Fit WOE encoding on all encoded features
    >>> stabilizer.fit(WoeStabilizerInput(
    ...     train_set = train_data, 
    ...     test_set  = test_data
    ... ))
    >>>
    >>> # Apply to new data (all features)
    >>> result = stabilizer.transform(test_data)
    ... result.data         # Data subset to stable WOE fes
    ... result.features     # Stable features  .woe / .org
    ... result.dropped      # Dropped features .woe / .org
    """
    
    def _save(self, save_path) -> None:
        features = {
            'features': self.cols.vars_woe,
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)         
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_inputs(inputs: WoeStabilizerEnvelope | WoeStabilizerInput) -> WoeStabilizerInput:
        if isinstance(inputs, WoeStabilizerEnvelope):
            return inputs.inputs
        return inputs   
    
    def _get_train_woe(self, inputs: WoeStabilizerEnvelope | WoeStabilizerInput) -> pDF:
        if isinstance(inputs, WoeStabilizerEnvelope):
            return inputs.config.train_woe
        enc = self.get_model_dependecy(DEPS)
        return enc.transform(inputs.train_set).data 
    
    @staticmethod
    def _verify(inputs:WoeStabilizerInput) -> bool:
        return ( 
              len(inputs.train_set) and 
              len(inputs.test_set)  and 
            ( inputs.woe_features is None or len(inputs.woe_features) )
        )    
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs: WoeStabilizerInput, fx_choice: str = REG.AIG, **params):
        """
        Execute the temporal stability screening on WOE features.

        Tests each WOE feature's bin-level consistency across
        time units on both the training and OOS test samples,
        scores overall feature stability, and screens against
        the `stability_cutoff` threshold.

        Parameters
        ----------
        inputs : WoeStabilizerInput
            Input configuration including train/test sets and
            WOE features to screen.
        fx_choice : str, optional
            The stability fitting strategy identifier. Defaults
            to REG.AIG.
        **params : dict
            Runtime parameters passed to the internal fitter
            (woe_stability, woe_alpha).

        Returns
        -------
        WoeStabilizer
            The fitted stabilizer instance.
        """
        inputs_ = self._get_inputs(inputs)
        
        self.choice = self.get_choice(fx_choice, inputs_)
        self.params = self.get_params(fx_choice, inputs_, params)
        
        train_woe = self._get_train_woe(inputs)

        self.processor.fit(
            inputs_, train_woe,
            fx_choice = self.choice,
            params    = self.params, 
            verify    = self._verify(inputs_)
        )
        self.custom_path = inputs_.save_path
        self._save(inputs_.save_path)

        _inputs: WoeStabilizerEnvelope = inputs
        update_data = (
            (lambda: _inputs.config.train_woe) 
            if _inputs.config.train_woe is not None
            else (lambda: train_woe)
        )
        self._set_update_data(update_data)

        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def update(self, *args, **kwargs):
        """
        Re-evaluate stability verdicts with updated thresholds.

        Adjusts the `woe_stability` and/or `woe_alpha` and
        re-partitions features without re-running the full
        consistency test (threshold-only updates) or re-fits
        (alpha changes).

        Parameters
        ----------
        *args : Any
            Positional arguments for update.
        **kwargs : dict
            Keyword arguments including the new threshold values
            (`woe_stability`, `woe_alpha`).

        Returns
        -------
        WoeStabilizer
            The updated stabilizer instance.
        """
        self.processor.update(
            th = kwargs.get(PRM.ENCODE.woe_stability.param), 
            alpha = kwargs.get(PRM.ENCODE.woe_alpha.param)
        )
        self._save(self.custom_path)
        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def transform(self, data:pDF) -> WoeStabilizerResult:
        """
        Select the stable WOE features from the input data.

        This is a pass-through selection (no value
        transformation) — the WOE values were already computed
        in the previous stage. Only the stable features (per
        the fit/update verdict) are retained.

        Parameters
        ----------
        data : pDF
            Data containing WOE features.

        Returns
        -------
        WoeStabilizerResult
            Stable WOE data + full feature lineage metadata.
        """
        df = self.processor.transform(data)
        return WoeStabilizerResult(
            data = df.data if not self._is_train_data(data) else None, 
            features = WoeStabilizerFeatureResult(
                woe = self.cols.vars_woe, 
                raw = VarStd.fe_to_var(self.cols.vars_woe)
            ),
            dropped = WoeStabilizerFeatureResult(
                woe = self.cols.vars_dropped,
                raw = VarStd.fe_to_var(self.cols.vars_dropped)
            )
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI(OptionUIBase):
    title   = 'WOE STABILITY PANEL'
    split   = 1
    Mapping = WoeParamsMapping
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → GUI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class WoeStabilizerUI(ModelUIBase[WoeStabilizer, OptionUI]):
    """
    AIG Flow Data WOE Features Stabilizer.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    Screens Weight of Evidence (WOE) encoded features for stability 
    to ensure that predictive behavior remains consistent across 
    temporal splits or different data folds.    
    """
    
    Model  = WoeStabilizer
    Option = OptionUI

    def custom_update_vars(self, *args, **kwargs):
        """User Custom Features Selection and propagation."""    
        return self.model.update_transform()    

    def chosen_vars_update_info(self, *args, **kwargs):
        """User Custom Features Selection Save and Summary."""
        self.model.fx.fitter._update_vars_by_ui()

    @staticmethod
    def _fit_transform_data_link() -> str:
        """Returns name of data which are passed to transform(data) after fit()."""
        # s = Dataclass.schema(WoeStabilizerInput)
        # return s.train_woe
        return FIT.UPDATE
    
    @wraps(WoeStabilizer.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(
            *args,
            ui_params = self._get_ui_fit_defaults(
                fit_transform_inputs_link   = self._fit_transform_data_link(),
                parent_has_update_FX_method = True,
                custom_update_fitted_8      = self.custom_update_vars,
                gui_dynamic_w_fitted_8      = self.chosen_vars_update_info,
            ),
            **kwargs
        )  

    @wraps(WoeStabilizer.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)
        
    @property
    def fitted(self) -> WoeStabilizerResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬