"""
WOE (Weight of Evidence) Encoding — Final Feature Representation.

This module implements the WOE encoding stage of the AIDflow
pipeline. It transforms all encoded features (numerical,
categorical, and interaction) into their optimal WOE
representation — the final feature format consumed by GLM
models for inference.

What It Does
------------
1. **Collects** all encoded features from prior stages:
   - Numerical features (binned/discretized).
   - Categorical features (label-encoded).
   - Interaction features (binary 0/1 rules).
2. **Prunes bins** per feature using a statistical strategy
   (Gini-impact or t-test significance) to find the optimal
   bin structure.
3. **Computes WOE** for each bin:
   WOE = ln(Distribution of Goods / Distribution of Bads)
4. **Gates** each feature on Gini. Features below the threshold
   are dropped (insufficient predictive signal after encoding).
5. **Encodes** surviving features as a single numeric WOE value
   per observation.

Why WOE?
--------
- **Interpretability**: Each WOE value has direct business
  meaning (risk direction and magnitude per segment).
- **Stability**: WOE bins are less sensitive to small data
  changes than raw values.
- **Linearity**: Transforms non-linear feature-target
  relationships into approximately linear ones, benefiting
  logistic regression and linear scoring models.
- **Comparability**: All features are on a common risk scale.
  Coefficients are directly comparable across features.

Role in the Pipeline
--------------------
This is the **GLM-only** encoding path. GBM models consume
stable numerical and categorical features directly without
WOE transformation. The WOE encoder produces the final
feature representation for:
- Logistic Regression (Sparse, Lasso, ElasticNet, PLS).
- Scorecard deployment (each WOE bin = a point on the scale).

The WOE encoder is the convergence point of all feature
engineering: num-encoded, cat-encoded, and ixn-encoded
features all pass through this stage to produce the unified
model input.

Bin Pruning Strategies
----------------------
- **Gini (S1)**: Sequential merge, DeLong-gated. Produces
  fewer bins (lower variance, higher bias). Best for
  stability-critical features.
- **T-Test (S2)**: Alpha search, Welch-merged, DeLong-selected.
  Preserves more signal, slightly higher variance. Best for
  signal-critical features.

Dependencies
------------
Requires the following stages to be completed:
- Numerical Feature Encoder (binned features).
- Categorical Feature Encoder (label-encoded features).
- Interaction Encoder (optional — binary rule features).
"""

from ...deps.common             import  wraps
from ...deps.data               import  pDF

from ...core.parameters         import  (
                                        PRM,
                                        AIGFLOW,
                                        Parameters
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
                                        ModelEnvelope,
                                        ModelInput,
                                        ModelResult,
                                        ModelBase,  
                                        OptionUIBase,
                                        ModelUIBase
                                        )
from ...core.base               import  FitterModelBase
from ...core.factory            import  (
                                        Dataclass,
                                        DictClass
                                        )
from ...core.foundation         import  (
                                        AigParam, 
                                        AigStop
                                        )
from ...core.io                 import  ModelIO
from ...core.interop            import  Pd

from .fitters.woe_transformer   import  (
                                        WoeTransformerGini,
                                        WoeTransformerTest,
                                        FitParamsBase,
                                        FitParamsCfg,
                                        FitResult,
                                        FitStats                                          
                                        )
from .ixn_encoder               import  IxnEncoder


MODEL = AIGFLOW.Encoding.woe_encoder
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG DATA STAGE ► WOE ENCODING ► Transform Encoded Num and Cat FEs to WOE Representation'
DEP1  = AIGFLOW.Encoding.num_encoder
DEP2  = AIGFLOW.Encoding.cat_encoder
DEP3  = AIGFLOW.Encoding.ixn_encoder
DEPS  = DEP1 + DEP2 + DEP3


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    GINI = 'WoeGini'
    TEST = 'WoeTest'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → FIT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class WoeParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(FitParamsBase)
    setup  = {
        PRM.ENCODE.gini.param        : (params.gini_cutoff,  'Gini Feature Cutoff'   ),
        PRM.ENCODE.woe_alpha.param   : (params.alpha,        'Bins Pruning alpha'    ),
    }  
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class InnerFitterBase(FitParamsBase):

    config: dict|FitParamsCfg = FitParamsCfg(
        target   = None,
        verbose  = True
    )

    @property
    def _model(self) -> FitterModelBase:
        return self.model        
    
    def get_result(self) -> FitResult:
        return self._model.result

    def get_stats(self) -> FitStats:
        return self._model.stats
    
class FitOptions(FitRegOpt):
    """
    Model-to-Fitter Registration Layer.
    
    This class acts as the bridge between the high-level Model and the low-level Fitter.
    It utilizes a registration pattern to ensure that adding new prediction 
    algorithms only requires a new entry here, keeping the top-layer API stable.
    """   
    
    @FREG.register(REG.GINI)
    class WoeTransformerGiniFitter(InnerFitterBase):

        @property
        def _model(self) -> WoeTransformerGini:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> WoeTransformerGini:            
            setup  = dict(params) | config
            return WoeTransformerGini(**setup)

    @FREG.register(REG.TEST)
    class WoeTransformerTestFitter(InnerFitterBase):

        @property
        def _model(self) -> WoeTransformerTest:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> WoeTransformerTest:            
            setup  = dict(params) | config
            return WoeTransformerTest(**setup)            
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class FitParams(FitParamsBase):
    """Fitter params input design."""    
    num_features: list
    cat_features: list
    ixn_encoder: IxnEncoder

class FitIxn:
    """
    IXN vars as special type of CAT → provided ixn to gen interactions.
    Same philosophy as WOE/CAT: direct woe from input data.
    """    
    def __init__(self, cfg:Parameters, prm:FitParams):
        self.cfg = cfg
        self.prm = prm
        self.ixn = self.get_uni_ixn()

    def get_fes(self) -> tuple[list, list]:
        num = self.prm.num_features or []
        cat = self.prm.cat_features or []
        return num, cat
    
    def get_uni_ixn(self) -> list:        
        """Unique IXN mapping → user can define req. ixn subset via num-cat."""
        if self.prm.ixn_encoder is None:
            return []
        
        ixn_mapping = self.prm.ixn_encoder.cols.encoding        
        if not ixn_mapping:
            return []
        
        num, cat = self.get_fes()
        ixn = [x for x in num + cat if x in ixn_mapping] 
        if ixn:
            return ixn
        return list(ixn_mapping)
    
    def get_uni_fes(self) -> tuple[list|None, list|None]:
        """Unique WOE-CAT mapping → user can define req. ixn subset via num-cat."""
        num, cat = self.get_fes()
        if not num + cat + self.ixn:
            return None, None       # fallback to the target model fes handling
        num = [x for x in num if x not in self.ixn]
        cat = [x for x in cat if x not in self.ixn]
        return num, cat + self.ixn  # ixn vars are cat types

    def create_ixn(self, data:pDF, subset:list=None) -> pDF:
        if self.prm.ixn_encoder is None:
            return pDF(index=data.index)
        return self.prm.ixn_encoder.transform(
            data, subset = subset or self.ixn
        ).data
                    
    def extend_by_ixn(self, data:pDF, subset:list=None) -> pDF:
        if not self.ixn:
            return data  
        ixn = [x for x in self.ixn if x in subset] if subset else self.ixn
        if not ixn:
            return data
        d = data.copy()        
        d[ixn] = self.create_ixn(data, ixn)
        return d
    
    def update_fitted_ixn(self, result:FitResult) -> None:
        self.ixn = [x for x in self.ixn if x in result.encoding]
 
class Fitter(FitterBase[FitParams]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """  

    FREG = FREG
    DREG = DREG
    
    def _set_ixn(self) -> None:
        self.ixn = FitIxn(self.setup, self.params)

    def _data_preprocess(self, data) -> tuple[pDF, list, list]:
        num, cat = self.ixn.get_uni_fes()
        return self.ixn.extend_by_ixn(data), num, cat

    def fit(self, data:pDF): 
        """
        Learns WOE Encoding

        Parameters
        ----------
        data : Exogenous Data for woe encoding.
        """        
        self._set_ixn()
        data_, num, cat = self._data_preprocess(data)

        self.model.fit(data_, num, cat)
        self.result: FitResult = self.model.get_result()
        self.stats:  FitStats  = self.model.get_stats()
        self.ixn.update_fitted_ixn(self.result)
        self._init_dash()
        
        return self
    
    def transform(self, data:pDF, subset:list=None) -> pDF:
        """ 
        Transform Features to WOE Encoding.

        Parameters
        ----------
        data : New Data to apply trained transformation.
        subset : Feature subset to encode.
                
        Returns
        -------
        Transformed Data with transformed woe features.
        """         
        data_ = self.ixn.extend_by_ixn(data, subset)
        cols  = subset or data_.columns
        return self.model.transform(data_[cols])
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    encoding:       dict
    vars_woe:       list
    vars_selected:  list
    vars_dropped:   list
    vars_map:       dict

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

    def _set_cols(self):
        res = self.fitter.result
        return TransResCols(
            encoding        = res.encoding,
            vars_woe        = res.vars_woe,
            vars_selected   = list(res.vars_map),
            vars_dropped    = res.vars_dropped,
            vars_map        = res.vars_map
        )
        
    def transform(self, data:pDF, subset:list=None) -> TransResData: 
        """Model meta-transform."""
        return TransResData(
            data = self.fitter.transform(data, subset)
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → PROCESSOR
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
    Mapping = WoeParamsMapping

    ModelTransformer = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_choice(self, fx_choice:str, inputs:"WoeEncoderInput"):
        defaults = self.Config(
            target   = AigParam.di(inputs.target,  self.setup.target),
            verbose  = AigParam.di(inputs.verbose, self.setup.verbose),
        )        
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_params(self, fx_choice:str, inputs:"WoeEncoderInput", params:dict):
        params  = self.get_fit_params(fx_choice, inputs.fx_params, params)
        encoder = self.get_model_dependecy(DEP3, is_req=False)

        fit = FitParams(
            num_features = inputs.num_features, 
            cat_features = inputs.cat_features, 
            ixn_encoder  = encoder, 
            **params
        )
        return fit, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → API
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class WoeEncoderInput(ModelInput):
    """
    WOE encoder fit input specification.

    Defines the training data and feature lists required to
    execute the WOE encoding pipeline.

    Attributes
    ----------
    train_set : pDF
        Training data. Must contain the target column and
        all features to be WOE-encoded. This is the primary
        dataset on which bin structures are learned.
    num_features : list | None, default=None
        Numerical (already binned) features to WOE-encode.
        If None and cat_features is also None, all non-target
        columns are used.
    cat_features : list | None, default=None
        Categorical features to WOE-encode.

    Notes
    -----
    **Encoded Features Required**: WOE encoding operates on
    already-encoded features (from the num/cat/ixn encoder
    stages). Raw continuous or label-encoded features are
    not directly supported — they must pass through the
    respective encoder first.

    **Interaction Features**: Interaction (binary 0/1) features
    from the ixn encoder are handled automatically — they are
    treated as a special category type and WOE-encoded
    alongside the other categorical features.

    **GLM-Only**: This encoding path is exclusively for GLM
    models. GBM models consume stable numerical and categorical
    features directly without WOE transformation.
    """
    train_set:      pDF
    num_features:   list|None = None
    cat_features:   list|None = None

@Dataclass.base(dynamic_params=True)
class WoeEncoderEnvelopeCfg:
    train_num:   pDF | None
    train_cat:   pDF | None

@Dataclass.base(dynamic_params=True)
class WoeEncoderEnvelope(ModelEnvelope):
    """Model Fit Input Design."""
    inputs: WoeEncoderInput 
    config: WoeEncoderEnvelopeCfg

@Dataclass.base
class WoeEncoderResult(ModelResult):
    """
    WOE encoder output — final feature representation.

    The final result of the WOE encoding stage. Contains the
    WOE-encoded features, full encoding maps, and feature
    lineage metadata for downstream GLM consumption.

    Attributes
    ----------
    data : pDF
        WOE-encoded features — one numeric WOE column per
        encoded feature. Each value represents the log-odds
        contribution of that feature's bin relative to the
        target distribution.
    features : list
        WOE feature names (e.g., `['revenue__g_WOE']`).
    selected : list
        Original feature names that passed the WOE gate
        (and received encoding).
    dropped : list
        Features that were candidates but failed the Gini
        threshold (dropped after WOE transformation).
    fe_map : dict
        Mapping of original feature name → [bin_name, woe_name].
        The full naming lineage for each feature.
    encoding : dict
        Per-feature encoding records (bin structure, WOE values,
        performance stats). Contains the full encoding definition
        for audit and transform.

    Notes
    -----
    **Final Representation**: This is the model-facing format.
    Each WOE value is a single number per observation, directly
    usable as a logistic regression input.

    **Interpretable**: Each WOE bin has a direct business
    meaning — positive = above-average event rate, negative =
    below-average. The magnitude reflects the strength of
    separation.
    """
    data:       pDF
    features:   list
    selected:   list
    dropped:    list
    fe_map:     dict
    encoding:   dict
# ─────────────────────────────────────────────────────────────────────────────────────────────────
    
class WoeEncoder(Processor, ModelBase):     
    """
    WOE (Weight of Evidence) Encoder — Final Feature Representation.

    The convergence point of all feature engineering. Transforms
    encoded numerical, categorical, and interaction features into
    their WOE representation — the final format consumed by GLM
    models for inference.

    What It Does
    ------------
    1. **Collects** encoded features from prior stages (num, cat,
       ixn).
    2. **Prunes bins** per feature using a statistical strategy
       (Gini-impact or t-test significance).
    3. **Computes WOE** for each bin (log-odds of goods vs. bads).
    4. **Gates** each feature on Gini (below threshold → dropped).
    5. **Encodes** surviving features as a single numeric WOE value
       per observation.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full WOE encoding pipeline — bin pruning +
      WOE computation + gating. Produces the final feature set.
    - **transform(data, subset)**: Applies the learned WOE mappings
      to new data. Optionally restricts to a subset of features.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **WOE Benefits**:
    - Unified scale across all features (comparable magnitudes).
    - Monotonic relationship with risk (higher WOE = higher event rate).
    - Approximately linear in logistic models (coefficients = risk
      direction).
    - Robust to scale changes and outliers.

    **Bin Pruning**: Prevents overfitting of bin structure to
    training noise. The two strategies (Gini / t-test) offer
    different bias-variance trade-offs — choose based on whether
    stability or signal preservation is the priority.

    **Missing Handling**: Missing values are treated as a distinct
    bin with its own WOE value, preserving their predictive signal.

    **GLM-Only**: This encoding path is exclusively for GLM
    models. GBM models consume stable numerical and categorical
    features directly without WOE transformation.

    **Interaction Support**: Interaction (binary 0/1) features
    from the ixn encoder are automatically detected and WOE-encoded
    as a special category type.

    Examples
    --------
    >>> from aidflow.encoding import WoeEncoder
    >>> encoder = WoeEncoder(ctx.setup)
    >>>
    >>> # Fit WOE encoding on all encoded features
    >>> encoder.fit(WoeEncoderInput(train_set = train_data))
    >>>
    >>> # Apply to new data (all features)
    >>> result = encoder.transform(test_data)
    >>> result.data         # Data encoded to WOE representation
    ... result.features     # WOE features
    >>>
    >>> # Apply subset only
    >>> partial = encoder.transform(test_data, subset=['revenue__g_WOE'])
    """

    def _save(self, save_path) -> None:
        features = {
            'features': self.cols.vars_woe,
            'map': self.cols.vars_map
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _set_vars_req(self) -> None:
        selected = self.cols.vars_selected
        self.ixn_mapping = {}
        
        ixn_encoder: IxnEncoder = self.get_model_dependecy(DEP3, is_req=False)

        if ixn_encoder is None or not ixn_encoder.cols.encoding:
            self._vars_req = selected
            return

        ixn_mapping = ixn_encoder.cols.encoding
        
        vars_req = [x for x in selected if x not in ixn_mapping]
        for x in ixn_mapping:
            ixn = ixn_mapping[x].vars # orig data vars needed to gen ixn vars
            self.ixn_mapping[x] = ixn
            ixn = [x for x in ixn if x not in vars_req]
            vars_req.extend(ixn) 
        self._vars_req = vars_req
    
    def _get_vars_req(self, subset:list=None) -> list:
        if subset is None:
            return self._vars_req
        subset = [x for x in subset if x in self._vars_req]
        if not self.ixn_mapping:
            return subset
        vars_req = [] 
        for x in subset:
            if x in self.ixn_mapping:
                vars_req.extend(self.ixn_mapping[x])
            else:
                vars_req.append(x)
        return vars_req
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_inputs(inputs: WoeEncoderEnvelope | WoeEncoderInput) -> WoeEncoderInput:
        if isinstance(inputs, WoeEncoderEnvelope):
            return inputs.inputs
        return inputs

    def _set_encoders(self):
        """
        WOE encoder is a encoding check-point → wraps all trained encoders.
        Intended for champion agent inference.
        """
        self._num_enc = self.get_model_dependecy(DEP1)
        self._cat_enc = self.get_model_dependecy(DEP2)

    def _get_data(self, inputs: WoeEncoderEnvelope | WoeEncoderInput) -> tuple[pDF, pDF, pDF]:
        if isinstance(inputs, WoeEncoderInput):
            return (
                inputs.train_set,
                self._num_enc.transform(inputs.train_set).data,
                self._cat_enc.transform(inputs.train_set).data
            )
        d = (
            inputs.inputs.train_set,
            inputs.config.train_num,
            inputs.config.train_cat
        )
        d = [x for x in d if len(x)]
        if any(len(d[i]) != len(d[i+1]) for i in range(len(d)-1)):
            AigStop.custom(
                'AIG SAFETY ► Input data has different length ► NUM & CAT '
                'encoders must be re-fitted prior WOE after Outliers removal update.'                
            )
        return d
    
    def _join_data(self, inputs: WoeEncoderInput | WoeEncoderInput) -> pDF:
        data = self._get_data(inputs)
        return Pd.concat(data, axis=1)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs: WoeEncoderInput, fx_choice:str = REG.GINI, **params):
        """
        Fit WOE encoding on all encoded features.

        For each feature: prunes bins using the selected strategy
        (Gini or t-test), computes WOE values, gates on predictive
        power, and stores the final encoding.

        Parameters
        ----------
        inputs : WoeEncoderEnvelope | WoeEncoderInput
            Input configuration containing train_set, target, and
            feature lists.
        fx_choice : str, optional
            The bin pruning strategy. Defaults to REG.GINI.
        **params : dict
            Runtime parameters passed to the internal fitter
            (gini_cutoff, woe_alpha).

        Returns
        -------
        WoeEncoder
            The fitted encoder instance.
        """
        inputs_ = self._get_inputs(inputs)
        verify  = len(inputs_.train_set)
        
        self.choice = self.get_choice(fx_choice, inputs_)
        self.params = self.get_params(fx_choice, inputs_, params)

        self._set_encoders()

        self.processor.fit(
            inputs_, 
            self._join_data(inputs),
            fx_choice = self.choice,
            params    = self.params, 
            verify    = verify
        )
        self._set_vars_req()
        self._save(inputs_.save_path)
        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def transform(self, data:pDF, subset:list=None) -> WoeEncoderResult:
        """
        Apply fitted WOE encodings to new data.

        For each encoded feature, produces the WOE numeric value
        based on the bin assignment of the input observation.
        Unknown values are mapped to the 'Missing' category.

        Parameters
        ----------
        data : pDF
            New data containing the original (pre-WOE) features.
        subset : list, optional
            Specific features to process and return (`including IXN`). 
            Default = None: all trained features.         

        Returns
        -------
        WoeEncoderResult
            WOE-encoded features + full feature lineage metadata.
        """
        miss = [x for x in self._get_vars_req(subset) if (x not in data.columns)]
        df = self.processor.transform(data, subset = subset, verify = miss)
        return WoeEncoderResult(
            data     = df.data, 
            features = self.cols.vars_woe, 
            selected = self.cols.vars_selected,
            dropped  = self.cols.vars_dropped,
            fe_map   = self.cols.vars_map,
            encoding = self.cols.encoding
        )    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI(OptionUIBase):
    title   = 'WOE ENCODING PANEL'
    split   = 1
    Mapping = WoeParamsMapping
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → WOE → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class WoeEncoderUI(ModelUIBase):
    """
    AIG Flow Features WOE Encoder.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    Implemented to perform Weight of Evidence (WOE) encoding on features, 
    converting categorical/discrete features into a continuous scale 
    relative to a binary target.    
    """

    Model  = WoeEncoder
    Option = OptionUI
        
    @wraps(WoeEncoder.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(), **kwargs)

    @wraps(WoeEncoder.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)
        
    @property
    def fitted(self) -> WoeEncoderResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬