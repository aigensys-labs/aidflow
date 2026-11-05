"""
Categorical Feature Encoding — Rare Category Merging & WOE Preparation.

This module implements the categorical encoding stage of the AIDflow
pipeline. It transforms raw categorical features into stable,
well-supported category groups optimized for predictive discrimination
against the target.

What It Does
------------
1. **Gates**: Measures each feature's raw Gini/IV. Features below
   the threshold are dropped (insufficient predictive signal).
2. **Missing handling**: NaN/missing values are converted to a
   distinct 'Missing' category, preserving their predictive signal.
3. **Merges**: Categories with fewer than the minimum share of
   observations are merged with their neighbour (by target
   response behaviour) to ensure statistical reliability.
4. **Re-gates**: The merged feature's Gini/IV is re-measured.
   If merging destroyed the signal, the feature is dropped.

Why Merge Rare Categories?
--------------------------
Categories with very few observations produce unstable WOE
estimates (high variance) and are prone to overfitting. Merging
them into neighbouring categories (by response behaviour) reduces
variance while preserving the overall discriminative structure.
This is equivalent to regularisation in the categorical space.

Unlike numerical features (which require a bin search),
categorical features already have discrete values. The encoding
here is a **stability optimization** rather than a structure
search — the categories are given, and the question is how many
are stable enough to retain.

Role in the Pipeline
--------------------
This is a **data encoding** stage. It consumes raw categorical 
features and produces merged, stable category features, which 
are then fed into the WOE encoding stage (next in the pipeline). 
The output is the category encoding maps that all downstream 
GLM agents operate on.

Dependencies
------------
Requires the Feature Cleaner stage to be completed (categorical
features should be type-standardized before encoding).
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
from ...core.factory            import  (
                                        Dataclass,
                                        DictClass
                                        )
from ...core.foundation         import  AigParam
from ...core.io                 import  ModelIO

from .fitters.cat_standardizer  import  (
                                        CategoricalEncoder,
                                        FitParamsBase,
                                        FitParamsCfg,
                                        FitResult,
                                        FitStats
                                        )

MODEL = AIGFLOW.Encoding.cat_encoder
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG DATA STAGE ► CATEGORICAL ENCODING ► Transform Categorical Features to Encoded Standard'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG: str = 'CatEncoder'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → NUM → FIT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class CatParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(FitParamsBase)
    setup  = {
        PRM.ENCODE.mbs.param  : (params.min_bin_share, 'Minimum Bin Size'    ),
        PRM.ENCODE.gini.param : (params.gini_cutoff,   'Gini Feature Cutoff' ),
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
    class CategoricalEncoderFitter(FitParamsBase):

        config: dict|FitParamsCfg = FitParamsCfg(
            target   = None,
            verbose  = True
        )

        @property
        def _model(self) -> CategoricalEncoder:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> CategoricalEncoder:            
            setup  = dict(params) | config
            return CategoricalEncoder(**setup)

        def get_result(self) -> FitResult:
            return self._model.result

        def get_stats(self) -> FitStats:
            return self._model.stats   
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class FitParams(FitParamsBase):
    """Fitter params input design."""
    cat_features: list

class Fitter(FitterBase[FitParams]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """       
    
    FREG = FREG
    DREG = DREG    

    def fit(self, data:pDF): 
        """
        Analyze and learn categorical encoding.

        Parameters
        ----------
        data : Input Data for categorical learning.
        """
        if self._fit_no_model(MODEL, self.params.cat_features, FitResult):
            return
        
        self.model.fit(data, self.params.cat_features)        
        self.result: FitResult = self.model.get_result()
        self.stats:  FitStats  = self.model.get_stats()
        self._init_dash()

        return self
    
    def transform(self, data:pDF) -> pDF:
        """ 
        Transform Categorical Features to Encoding Space.

        Parameters
        ----------
        data : New Data to apply trained transformation.
                
        Returns
        -------
        Transformed Data with transformed numerical features.
        """
        if self._is_no_model:
            return pDF() 
                
        return self.model.transform(data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → NUM → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    encoding:       dict
    vars_cat:       list
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

    def _fit_data(self, inputs:"CatEncoderInput"): 
        return inputs.train_set

    def _set_cols(self):
        res = self.fitter.result
        return TransResCols(
            encoding        = res.encoding,
            vars_cat        = res.vars_cat,
            vars_selected   = list(res.vars_map),
            vars_dropped    = res.vars_dropped,
            vars_map        = res.vars_map
        )
        
    def transform(self, data:pDF) -> TransResData: 
        """Model meta-transform."""
        return TransResData(
            data = self.fitter.transform(data)
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → NUM → PROCESSOR
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
    Mapping = CatParamsMapping

    ModelTransformer = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────         

    def get_choice(self, fx_choice:str, inputs:"CatEncoderInput"):
        defaults = self.Config(
            target  = AigParam.di(inputs.target,  self.setup.target),
            verbose = AigParam.di(inputs.verbose, self.setup.verbose)
        )     
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_params(self, fx_choice:str, inputs:"CatEncoderInput", params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params)        
        fit = FitParams(cat_features = inputs.cat_features, **params)
        return fit, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → NUM → API
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class CatEncoderInput(ModelInput):
    """
    Categorical encoder fit input specification.

    Defines the training data and feature set required to
    execute the categorical encoding pipeline.

    Attributes
    ----------
    train_set : pDF
        Training data. Must contain the target column and
        the categorical features to be encoded. This is the
        primary dataset on which gating and MBS merging are
        performed.
    cat_features : list | None, default=None
        Categorical features to encode. If None, all non-target
        columns in `train_set` are used as candidates.

    Notes
    -----
    **Target Required**: Encoding is optimized against the
    target (MBS merging uses target response to determine
    neighbour categories). The target column must be present
    in `train_set`.

    **Feature Resolution**: If `cat_features` is None, the
    full column set (excluding target) is used. In practice,
    the Feature Cleaner stage typically resolves the categorical
    feature list before this stage.
    """
    train_set:      pDF
    cat_features:   list|None = None

@Dataclass.base
class CatEncoderResult(ModelResult):
    """
    Categorical encoder output — category structure and feature lineage.

    The final result of the categorical encoding. Contains the
    category encoding maps, the transformed data, and full
    feature lineage metadata for downstream consumption.

    Attributes
    ----------
    data : pDF
        Transformed data — one column per encoded feature
        (merged categories, including 'Missing' category
        where applicable).
    features : list
        All encoded categorical features (merged category names).
    selected : list
        Original feature names that passed all gates and were
        encoded.
    dropped : list
        Features that failed a gate (insufficient signal at
        original or encoded level).
    fe_map : dict
        Mapping of original feature name → final encoded feature
        name (after min-share grouping).
    encoding : dict
        Per-feature encoding maps (original category → merged
        category). Used by the WOE encoder for probability
        computation.
    """
    data:       pDF
    features:   list
    selected:   list
    dropped:    list
    fe_map:     dict
    encoding:   dict
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class CatEncoder(Processor, ModelBase):     
    """
    Categorical Feature Encoding — Rare Category Merging & WOE Preparation.

    Transforms raw categorical features into stable, well-supported
    category groups by merging rare categories. The resulting features
    are designed for OOT stability and feed into the WOE encoding step.

    What It Does
    ------------
    1. **Gates**: Features with Gini below threshold are
       dropped (insufficient predictive signal).
    2. **Handles Missing**: NaN → 'Missing' category (preserves
       predictive signal of the absence of a value).
    3. **Merges**: Rare categories (below minimum share) are
       merged with their neighbour by target response.
    4. **Re-gates**: Merged features must maintain predictive
       power (else dropped).

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full encoding pipeline — gating +
      missing handling + MBS merging. Produces the category
      structure.
    - **transform(data)**: Applies the learned category mappings
      to new data, producing merged categorical features.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **Stability Optimization**: Unlike numerical features (which
    require a bin search), categorical features already have
    discrete values. The optimization is purely in merging rare
    categories to ensure WOE stability — equivalent to
    regularisation in the categorical space.

    **Dual Gate**: Features must pass a Gini threshold both
    before (original signal) and after (merged signal) the MBS
    step. This prevents the merge from destroying discriminative
    power.

    **Missing as a Category**: Missing values are treated as a
    distinct category, preserving their predictive signal in
    the WOE computation. This is critical for features where
    absence of a value is itself informative (e.g., no
    interaction in the last 30 days).

    **MBS Merging**: Categories below the minimum share are
    merged with their neighbour (by target response behaviour).
    This ensures each group has sufficient sample size for
    stable WOE computation in production.

    Examples
    --------
    >>> from aidflow.encoding import CatEncoder
    >>> encoder = CatEncoder(ctx.setup)
    >>>
    >>> # Learn category structure
    >>> encoder.fit(CatEncoderInput(train_set = train_data))
    >>>
    >>> # Apply to new data
    >>> result = encoder.transform(test_data)
    >>> result.data         # Encoded categorical data
    >>> result.selected     # ['region', 'segment', 'channel', ...]
    >>> result.dropped      # ['noise_cat', 'irrelevant', ...]
    """

    def _save(self, save_path) -> None:
        features = {
            'features': self.cols.vars_cat,
            'map': self.cols.vars_map
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)         
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs:CatEncoderInput, fx_choice:str = REG.AIG, **params):
        """
        Analyze and learn categorical encoding.

        Executes the predictive gates, missing handling, and
        min-bin-share merging for each categorical feature.
        Results are stored as encoding maps ready for
        `transform`.

        Parameters
        ----------
        inputs : CatEncoderInput
            Input configuration including train_set and
            categorical feature lists.
        fx_choice : str, optional
            The categorical encoding strategy. Defaults to REG.AIG.
        **params : dict
            Runtime parameters passed to the internal fitter
            (gini_cutoff, min_bin_share).

        Returns
        -------
        CatEncoder
            The fitted encoder instance.
        """
        verify = len(inputs.train_set)
        
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

    def transform(self, data:pDF, subset:list=None) -> CatEncoderResult:
        """
        Apply fitted category mappings to new data.

        Maps each categorical feature to its merged (post-MBS)
        category values using the learned encoding maps.

        Parameters
        ----------
        data : pDF
            New data containing the original categorical features.
        subset : list, optional
            Specific features to process and return. 
            Default = None: all trained features. 

        Returns
        -------
        CatEncoderResult
            Transformed data + feature lineage metadata.
        """
        c_all = self.cols.vars_selected
        c_sub = subset or data.columns
        cols  = [x for x in c_sub if x in c_all]

        miss = c_all if not cols else None
        df = self.processor.transform(data[cols], verify = miss)

        return CatEncoderResult(
            data     = df.data, 
            features = self.cols.vars_cat,
            selected = self.cols.vars_selected,
            dropped  = self.cols.vars_dropped,
            fe_map   = self.cols.vars_map,
            encoding = self.cols.encoding
        )    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → NUM → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI(OptionUIBase):
    
    title   = 'CATEGORICAL ENCODING PANEL'
    split   = 1
    Mapping = CatParamsMapping
        
    def _min_share_constrain(self, value):
        if value < 1: # Driven by param design in PRM
            return 0  # shared param - either float or int
        return value
    
    def options(self):
        super().options()

        mbs = PRM.ENCODE.mbs.param
        desc = {
            mbs: ['Minimum Bin Size', [0], self._min_share_constrain]
        }     
        self.wg.params_to_values(desc)        
        self.wg.shared_param_link( {mbs: self.wg.shared_relat_abs} )        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FE → CAT → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class CatEncoderUI(ModelUIBase):
    """
    AIG Flow Data Categorical Features Encoder.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    Analyzes categorical features and applies the chosen encoding 
    strategy to convert categories into numerical representations.    
    """

    Model  = CatEncoder
    Option = OptionUI
        
    @wraps(CatEncoder.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(), **kwargs)  
    
    @wraps(CatEncoder.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)
        
    @property
    def fitted(self) -> CatEncoderResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬