"""
Feature Interaction Extraction — Tree Mining & Binary Encoding.

This module implements the interaction discovery stage of the AIDflow
pipeline. It extracts significant non-linear feature interactions
by training a shallow gradient boosting model and mining its tree
structure for interpretable rules, encoding each as a binary (0/1)
categorical variable.

What It Does
------------
1. **Trains** a shallow GB model (constrained depth, ≤31 leaves)
   on the input features. The model is a *search mechanism*, not
   a predictive model.
2. **Mines** interaction rules from the tree leaves that satisfy
   minimum support, target-rate, and correlation thresholds.
3. **Gates** each rule on Gini/IV. Rules below the threshold
   are dropped.
4. **Encodes** surviving rules as binary (0/1) features —
   1 if the observation matches the rule condition, 0 otherwise.

Why Shallow Trees?
------------------
Deep trees produce highly specific rules (many conditions
combined) that overfit to training noise. A shallow tree
(≤31 leaves) produces broader, more stable interaction patterns
that generalise to OOT data. The constraint acts as a
regulariser on the interaction complexity.

Why Binary Encoding?
--------------------
Each interaction is a discrete *event* (the condition is either
met or not). Binary encoding is the natural representation:
it is category-agnostic, stable under monotone transforms,
and directly consumable by the WOE encoder.

Role in the Pipeline
--------------------
This is a **feature engineering** stage. It generates new
binary interaction features that capture non-linear
relationships between existing features. The output binary
columns are fed into the WOE encoding stage alongside the
standardized numerical and categorical features for GLM.

The interactions are discovered automatically — the user does
not specify which pairs to search. The tree structure of the
GB model provides the search space.

Dependencies
------------
Requires the Feature Standardizer stages (numerical binning
and/or categorical encoding) to be completed. Interaction
mining operates on the standardized feature set.
"""

from ...deps.common             import  wraps
from ...deps.data               import  pDF

from ...core.parameters         import  (
                                        PRM,
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
                                        ModelBase,  
                                        OptionUIBase,
                                        ModelUIBase
                                        )
from ...core.base               import  FitterModelBase
from ...core.factory            import  (
                                        Dataclass,
                                        DictClass,
                                        )
from ...core.foundation         import  AigParam
from ...core.io                 import  ModelIO

from .fitters.ixn_extractor     import  (
                                        InteractionExtractorXGB,
                                        InteractionExtractorLGB,
                                        FitParamsBase,
                                        FitParamsCfg,
                                        FitResult,
                                        FitStats
                                        )

MODEL = AIGFLOW.Encoding.ixn_encoder
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG DATA STAGE ► INTERACTION ENCODING ► Extract Non-Linear Interactions to New Features'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    LGB: str = 'IxnLGB'
    XGB: str = 'IxnXGB'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → IXN → FIT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class IxnParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(FitParamsBase)
    setup  = {
        PRM.ENCODE.ixn_leaves.param     : (params.gb_max_leaves,        'Tree Maximum Leaves'   ),
        PRM.TRAIN.gb_estimators.param   : (params.gb_estimators,        'Tree Estimators'       ),
        PRM.TRAIN.gb_learning.param     : (params.gb_learning_rate,     'Tree Learning Rate'    ),
        PRM.TRAIN.gb_min_child.param    : (params.gb_min_child_weight,  'Tree Min Child Weight' ),
        PRM.ENCODE.ixn_target.param     : (params.ixn_min_target,       'Ixn Min Target Rate'   ),
        PRM.ENCODE.mbs.param            : (params.ixn_min_share,        'Ixn Min Pop Share'     ),
        PRM.SCREEN.corr.param           : (params.ixn_max_corr,         'Ixn Max Correlation'   ),
        PRM.ENCODE.gini.param           : (params.gini_cutoff,          'Gini Feature Cutoff'   )
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

    @FREG.register(REG.LGB)
    class InteractionExtractorLGBFitter(InnerFitterBase):
        
        @property
        def _model(self) -> InteractionExtractorLGB:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> InteractionExtractorLGB:            
            setup  = dict(params) | config
            return InteractionExtractorLGB(**setup)
            
    @FREG.register(REG.XGB)
    class InteractionExtractorXGBFitter(InnerFitterBase):

        @property
        def _model(self) -> InteractionExtractorXGB:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> InteractionExtractorXGB:            
            setup  = dict(params) | config
            return InteractionExtractorXGB(**setup)          
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class FitParams(FitParamsBase):
    """Fitter params input design."""
    num_features: list
    cat_features: list

class Fitter(FitterBase[FitParams]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """       
    
    FREG = FREG
    DREG = DREG
    
    _is_dummy = False    

    def _fit_no_model(self, data:pDF) -> bool:
        if len(data.columns) > 3:
            return False
        self._is_no_model = True
        self.result = None
        if self.verbose:
            print('► Not Enough Features to Generate Interactions ► Skipping') 
        return True
        
    def fit(self, data:pDF): 
        """
        Analyze and learn interactions encoding.

        Parameters
        ----------
        data : Input Data for interactions learning.
        """
        if self._fit_no_model(data):
            return

        self.model.fit(
            data, 
            num_features = self.params.num_features, 
            cat_features = self.params.cat_features
        )
        self.result: FitResult = self.model.get_result()
        self.stats:  FitStats  = self.model.get_stats()
        self._init_dash()
        return self
    
    def transform(self, data:pDF, ixn_subset:list=None) -> pDF:
        """ 
        Transform Features to Interactions.

        Parameters
        ----------
        data : New Data to apply trained transformation.
        ixn_subset : list, optional
            Specific interactions to produce. 
            Default = None: all trained are produced.
                            
        Returns
        -------
        Transformed Data with transformed numerical features.
        """         
        if self._is_no_model:
            return pDF()        
        return self.model.transform(data, ixn_subset)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → IXN → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    encoding:       dict
    vars_ixn:       list
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

    def _fit_data(self, inputs:"IxnEncoderInput"): 
        return inputs.train_set

    def _set_cols(self):
        res = self.fitter.result        
        return TransResCols(
            encoding      = res.encoding,
            vars_ixn      = res.vars_ixn,
            vars_selected = res.vars_org,
            vars_dropped  = res.vars_dropped,
            vars_map      = res.vars_map
        )    
        
    def transform(self, data:pDF, ixn_subset:list=None) -> TransResData: 
        """Model meta-transform."""
        return TransResData(
            data = self.fitter.transform(data, ixn_subset)
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → IXN → PROCESSOR
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
    Mapping = IxnParamsMapping

    ModelTransformer = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_choice(self, fx_choice:str, inputs:"IxnEncoderInput"):
        defaults = self.Config(
            target   = AigParam.di(inputs.target,  self.setup.target),
            verbose  = AigParam.di(inputs.verbose, self.setup.verbose),
            val_size = PRM.LOAD.valid.value
        )        
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_params(self, fx_choice:str, inputs:"IxnEncoderInput", params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params)
        fit = FitParams(
            num_features = inputs.num_features, 
            cat_features = inputs.cat_features, 
            **params
        )
        return fit, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → IXN → API
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class IxnEncoderInput(ModelInput):
    """
    Interaction encoder fit input specification.

    Defines the training data, feature lists, and optional
    subset selection required to execute the interaction
    mining pipeline.

    Attributes
    ----------
    train_set : pDF
        Training data. Must contain the target column and
        the features to search for interactions. This is the
        primary dataset on which the GB model is trained and
        rules are mined.
    num_features : list | None, default=None
        Numerical features to include in the interaction search.
        If None, all non-target columns are used.
    cat_features : list | None, default=None
        Categorical features to include in the interaction search.
        If None, none are used.
    ixn_subset : list | None, default=None
        Specific interaction names to produce in `transform`.
        If None, all mined interactions are produced.

    Notes
    -----
    **Target Required**: Interactions are mined against the
    target. The target column must be present in `train_set`.

    **Feature Resolution**: If both `num_features` and
    `cat_features` are None, all non-target columns are used
    as numerical features (the typical flow — the standardizer
    stage resolves the feature type split).

    **Minimum Features**: At least 3 columns (including target)
    are required for interaction mining. Fewer than this
    produces no meaningful interactions.
    """
    train_set:      pDF
    num_features:   list|None = None
    cat_features:   list|None = None
    ixn_subset:     list|None = None

@Dataclass.base
class IxnEncoderResult(ModelResult):
    """
    Interaction encoder output — mined rules and feature lineage.

    The final result of the interaction extraction. Contains the
    binary interaction columns, the rule conditions, and full
    feature lineage metadata for downstream consumption.

    Attributes
    ----------
    data : pDF
        Binary (0/1) interaction columns — one column per mined
        rule. 1 if the observation matches the rule condition,
        0 otherwise.
    features : list
        All mined interaction names (e.g., `['ixn_0', 'ixn_1', ...]`).
    selected : list
        Original features that contribute to at least one kept
        interaction. These features are "in use" by the interaction
        stage and should not be independently dropped downstream.
    dropped : list
        Interaction names that were mined but failed the Gini gate.
    fe_map : dict
        Mapping of interaction name → list of contributing feature
        names. Useful for understanding which original features
        drive each interaction.
    encoding : dict
        Per-interaction encoding records (rule condition, type,
        stats). Contains the full rule definition for audit.

    Notes
    -----
    **Binary Output**: Each interaction is a discrete event
    (rule met / not met). These binary columns are directly
    consumable by the WOE encoder.

    **Interpretable**: Each interaction has a human-readable rule
    condition (e.g., `revenue > 500 AND segment == 'B'`),
    stored in the `encoding` dict.
    """
    data:       pDF
    features:   list
    selected:   list
    dropped:    list
    fe_map:     dict
    encoding:   dict
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class IxnEncoder(Processor, ModelBase):     
    """
    Feature Interaction Extraction — Tree Mining & Binary Encoding.

    Discovers significant non-linear feature interactions by
    training a shallow gradient boosting model and mining its
    tree structure for interpretable rules. Each surviving rule
    is encoded as a binary (0/1) categorical variable.

    What It Does
    ------------
    1. **Trains** a shallow GB model (≤31 leaves) on the input
       features. The model is a search mechanism, not a
       predictive model.
    2. **Mines** interaction rules from tree leaves that satisfy
       minimum support, target-rate, and correlation thresholds.
    3. **Gates** each rule on Gini/IV (below threshold → dropped).
    4. **Encodes** surviving rules as binary (0/1) features.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full mining pipeline — train GB + mine
      rules + gate. Produces the binary interaction columns.
    - **transform(data, subset)**: Applies the mined rules to
      new data. Optionally restricts to a subset of interactions.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **Search, not prediction**: The GB model is a search
    mechanism whose tree structure is decomposed into
    interpretable rules. The model itself is never used to
    predict.

    **Shallow by design**: Constrained tree depth (≤31 leaves)
    ensures broad, stable interaction patterns rather than
    over-specific training noise.

    **Binary output**: Each interaction is a discrete event
    (rule met / not met). Binary encoding is category-agnostic,
    stable, and directly consumable by the WOE encoder.

    **Redundancy control**: Highly correlated rules (above
    `ixn_max_corr`) are pruned to avoid capturing the same
    pattern multiple times.

    **Supported Backends**: LightGBM (default) or XGBoost.
    The tree mining protocol is identical across backends.

    Examples
    --------
    >>> from aidflow.encoding import IxnEncoder
    >>> encoder = IxnEncoder(ctx.setup)
    >>>
    >>> # Mine interactions
    >>> encoder.fit(IxnEncoderInput(train_set = train_data))
    >>>
    >>> # Apply to new data (all interactions)
    >>> result = encoder.transform(test_data)
    >>> result.data         # Created data with interaction features
    >>>
    >>> # Apply subset only
    >>> partial = encoder.transform(test_data, subset=['ixn_0', 'ixn_2'])
    """

    def _save(self, save_path) -> None:
        features = {
            'features': self.cols.vars_ixn,
            'map': self.cols.vars_map
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)
    
    def _vars_req(self, ixn_subset:list|None) -> list:
        if self.cols.encoding is None:
            return []
        if ixn_subset is None:
            return self.cols.vars_selected
        req_vars = []
        for x in ixn_subset:
            req_vars.extend(self.cols.encoding[x].vars)
        return list(dict.fromkeys(req_vars))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs:IxnEncoderInput, fx_choice: str = REG.LGB, **params):
        """
        Analyze the feature set and mine interaction rules.

        Trains the shallow GB model, mines interaction rules
        from the tree structure, gates on predictive power,
        and stores the surviving binary encodings.

        Parameters
        ----------
        inputs : IxnEncoderInput
            Configuration including train_set, num_features,
            and cat_features.
        fx_choice : str, optional
            The interaction mining backend. Defaults to REG.LGB.
        **params : dict
            Runtime parameters passed to the internal fitter
            (gb_max_leaves, gb_estimators, gb_learning_rate,
            gb_min_child_weight, ixn_min_target, ixn_min_share,
            ixn_max_corr, gini_cutoff).

        Returns
        -------
        IxnEncoder
            The fitted encoder instance.
        """
        verify = ( 
            len(inputs.train_set) and (
            ( inputs.num_features is None or len(inputs.num_features) ) or
            ( inputs.cat_features is None or len(inputs.cat_features) )
        ))
        
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

    def transform(self, data:pDF, subset:list=None) -> IxnEncoderResult:
        """
        Apply mined interaction rules to new data.

        Evaluates each rule condition against the input data,
        producing binary (0/1) interaction columns.

        Parameters
        ----------
        data : pDF
            New data containing the original features referenced
            by the mined rules.
        subset : list, optional
            Specific interaction names to produce. If None, all
            trained interactions are produced.

        Returns
        -------
        IxnEncoderResult
            Binary interaction columns + feature lineage metadata.
        """
        miss = [x for x in self._vars_req(subset) if (x not in data.columns)]                
        df = self.processor.transform(data, subset, verify = miss)
        return IxnEncoderResult(
            data     = df.data, 
            features = self.cols.vars_ixn, 
            selected = self.cols.vars_selected, 
            dropped  = self.cols.vars_dropped,
            fe_map   = self.cols.vars_map,
            encoding = self.cols.encoding
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → IXN → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI(OptionUIBase):
    title   = 'INTERACTIONS ENCODING PANEL'
    split   = 4
    Mapping = IxnParamsMapping
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → IXN → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class IxnEncoderUI(ModelUIBase):
    """
    AIG Flow Data Feature Interaction Encoder.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    Responsible for identifying and creating interaction features 
    between different variable types to capture complex relationships.    
    """

    Model  = IxnEncoder
    Option = OptionUI
        
    @wraps(IxnEncoder.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(), **kwargs)  
    
    @wraps(IxnEncoder.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> IxnEncoderResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬