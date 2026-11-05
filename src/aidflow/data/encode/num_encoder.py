"""
Numerical Feature Discretization — Bin Search & Encoding.

This module implements the numerical encoding stage of the AIDflow
pipeline. It transforms continuous numerical features into
categorical interval features (bins) optimized for predictive
discrimination against the target, using a non-greedy,
stability-optimized bin search.

What It Does
------------
1. **Gates**: Measures each feature's raw Gini/IV. Features below
   the threshold are dropped (insufficient signal).
2. **Searches**: Finds the optimal bin count via a statistical
   alpha-based stability search (not greedy tree splitting).
3. **Bins**: Cuts the feature into the discovered number of bins
   (quantile-based for continuous, value-based for low-cardinality).
4. **Merges**: Bins with fewer than the minimum share of
   observations are merged with their neighbours.
5. **Re-gates**: Re-measures the encoded feature's Gini/IV.
   If encoding destroyed the signal, the feature is dropped.

Why Non-Greedy?
---------------
Greedy algorithms (e.g., decision tree splitting) maximise impurity
reduction at each step, which captures training noise in bin
boundaries. The non-greedy approach finds the simplest bin
structure that preserves predictive power — lower variance,
better OOT stability. This aligns with the bias-variance
trade-off: fewer free parameters → less overfitting → more
robust production performance.

Role in the Pipeline
--------------------
This is a **data encoding** stage. It consumes raw
numerical features and produces categorical bin features,
which are then fed into the WOE encoding stage (next in the
pipeline). The output is the bin structure (encoding maps)
that all downstream GLM agents operate on.

Dependencies
------------
Requires the Feature Cleaner stage to be completed (numerical
features should be type-standardized before binning).
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

from .fitters.num_discretizor   import  (
                                        NumericalDiscretizor,
                                        FitParamsBase,
                                        FitParamsCfg,
                                        FitResult,
                                        FitStats                                        
                                        )


MODEL = AIGFLOW.Encoding.num_encoder
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG DATA STAGE ► NUMERICAL ENCODING ► Encode Numerical Features to Bin Representation'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG: str = 'NumDiscretizor'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → NUM → FIT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class NumParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(FitParamsBase)
    setup  = {
        PRM.ENCODE.bins.param      : (params.max_num_bins,   'Bin Search Space'      ),
        PRM.ENCODE.gini.param      : (params.gini_cutoff,    'Gini Feature Cutoff'   ),
        PRM.ENCODE.num_alpha.param : (params.alpha,          'Bins Difference alpha' ),
        PRM.ENCODE.mbs.param       : (params.min_bin_share,  'Minimum Bin Size'      ),
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
    class NumericalDiscretizorFitter(FitParamsBase):

        config: dict|FitParamsCfg = FitParamsCfg(
            target   = None,
            discrete = 10,
            verbose  = True
        )

        @property
        def _model(self) -> NumericalDiscretizor:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> NumericalDiscretizor:            
            setup  = dict(params) | config
            return NumericalDiscretizor(**setup)

        def get_result(self) -> FitResult:
            return self._model.result

        def get_stats(self) -> FitStats:
            return self._model.stats
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class FitParams(FitParamsBase):
    """Fitter params input design."""       
    num_features: list

class Fitter(FitterBase[FitParams]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """    

    FREG = FREG
    DREG = DREG
    
    def fit(self, data:pDF): 
        """
        Analyze and learn numerical encoding.

        Parameters
        ----------
        data : Input Data for numerical learning.
        """
        if self._fit_no_model(MODEL, self.params.num_features, FitResult):
            return
                
        self.model.fit(data, self.params.num_features)        
        self.result: FitResult = self.model.get_result()        
        self.stats:  FitStats  = self.model.get_stats()
        self._init_dash()

        return self
    
    def transform(self, data:pDF) -> pDF:
        """ 
        Transform Numerical Features to Discrete Space.

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
    vars_num:       list
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

    def _fit_data(self, inputs: "NumEncoderInput"):
        return inputs.train_set

    def _set_cols(self):
        res = self.fitter.result        
        return TransResCols(
            encoding      = res.encoding,
            vars_num      = res.vars_num,
            vars_selected = list(res.vars_map),
            vars_dropped  = res.vars_dropped,
            vars_map      = res.vars_map
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
    Mapping = NumParamsMapping

    ModelTransformer = Transformer 
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_choice(self, fx_choice:str, inputs:"NumEncoderInput"):
        defaults = self.Config(
            target   = AigParam.di(inputs.target,  self.setup.target),
            verbose  = AigParam.di(inputs.verbose, self.setup.verbose),
            discrete = 10,
        )        
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_params(self, fx_choice:str, inputs:"NumEncoderInput", params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params)        
        fit = FitParams(num_features = inputs.num_features, **params)
        return fit, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → NUM → API
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class NumEncoderInput(ModelInput):
    """
    Numerical encoder fit input specification.

    Defines the training data and feature set required to
    execute the non-greedy binning pipeline.

    Attributes
    ----------
    train_set : pDF
        Training data. Must contain the target column and
        the numerical features to be encoded. This is the
        primary dataset on which bin search and gating are
        performed.
    num_features : list | None, default=None
        Numerical features to encode. If None, all non-target
        columns in `train_set` are used as candidates.

    Notes
    -----
    **Target Required**: Binning is optimized against the
    target. The target column must be present in `train_set`.

    **Feature Resolution**: If `num_features` is None, the
    full column set (excluding target) is used. In practice,
    the Feature Cleaner stage resolves the numerical feature 
    list before this stage.
    """
    train_set:      pDF
    num_features:   list|None = None

@Dataclass.base
class NumEncoderResult(ModelResult):
    """
    Numerical encoder output — bin structure and feature lineage.

    The final result of the numerical discretization. Contains
    the bin encoding maps, the transformed data, and full
    feature lineage metadata for downstream consumption.

    Attributes
    ----------
    data : pDF
        Transformed data — one column per encoded feature
        (grouped bin values, including 'Missing' category
        where applicable).
    features : list
        All encoded numerical features (grouped bin names).
    selected : list
        Original feature names that passed all gates and were
        encoded.
    dropped : list
        Features that failed a gate (insufficient signal at
        original or encoded level).
    fe_map : dict
        Mapping of original feature name → final encoded
        feature name (after min-bin-share grouping).
    encoding : dict
        Per-feature encoding maps (original value → bin →
        grouped bin). Used by the WOE encoder for probability
        computation.
    """
    data:       pDF
    features:   list
    selected:   list
    dropped:    list
    fe_map:     dict
    encoding:   dict
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class NumEncoder(Processor, ModelBase):     
    """
    Numerical Feature Discretization — Non-Greedy Binning & Encoding.

    Transforms continuous numerical features into categorical
    interval features using a non-greedy, stability-optimized
    bin search. The resulting bins are designed for OOT
    stability and feed into the WOE encoding step.

    What It Does
    ------------
    1. **Gates**: Features with Gini below threshold are
       dropped (insufficient predictive signal).
    2. **Searches**: Finds the optimal bin count via alpha-based
       stability search (not greedy tree splitting).
    3. **Bins**: Cuts the feature into the discovered number of
       bins (quantile or value-based).
    4. **Merges**: Bins below minimum share are merged with
       neighbours.
    5. **Re-gates**: Encoded features must maintain predictive
       power (else dropped).

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full binning pipeline — search + gating
      + encoding. Produces the bin structure.
    - **transform(data)**: Applies the learned bin mappings to
      new data, producing categorical interval features.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **Non-Greedy Strategy**: Unlike decision-tree-based binning
    (which captures training noise in boundaries), this approach
    finds the simplest stable bin structure that preserves
    predictive power. Result: lower variance, better OOT
    performance.

    **Dual Gate**: Features must pass a Gini threshold both
    before (original signal) and after (encoded signal) binning.
    This prevents encoding from destroying discriminative power.

    **Missing as a Bin**: Missing values are treated as a
    distinct category in the encoding, preserving their
    predictive signal in the WOE computation.

    **Discrete Features**: Features with ≤ `discrete` unique
    values use value-based binning (each unique value is a
    candidate bin) rather than quantile-based splitting. This
    avoids forcing artificial splits on inherently categorical
    numerical columns (e.g., counts, age groups).

    Examples
    --------
    >>> from aidflow.encoding import NumEncoder
    >>> encoder = NumEncoder(ctx.setup)
    >>>
    >>> # Learn bin structure
    >>> encoder.fit(NumEncoderInput(train_set = train_data))
    >>>
    >>> # Apply to new data
    >>> result = encoder.transform(test_data)
    >>> result.data         # Encoded numerical data
    >>> result.selected     # ['revenue', 'tenure', ...]
    >>> result.dropped      # ['noise_var', 'irrelevant', ...]
    """
    
    def _save(self, save_path) -> None:
        features = {
            'features': self.cols.vars_num,
            'map': self.cols.vars_map
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs:NumEncoderInput, fx_choice: str = REG.AIG, **params):
        """
        Analyze and learn numerical bin encoding.

        Executes the non-greedy bin search, min-bin-share
        merging, and predictive gates for each numerical
        feature. Results are stored as encoding maps ready for
        `transform`.

        Parameters
        ----------
        inputs : NumEncoderInput
            Input configuration including training set and
            numerical feature lists.
        fx_choice : str, optional
            The numerical encoding strategy. Defaults to REG.AIG.
        **params : dict
            Runtime parameters passed to the internal fitter
            (max_num_bins, gini_cutoff, num_alpha, min_bin_share).

        Returns
        -------
        NumEncoder
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

    def transform(self, data:pDF, subset:list=None) -> NumEncoderResult:
        """
        Apply fitted bin encodings to new data.

        Maps each numerical feature to its binned (grouped)
        categorical values using the learned encoding maps.

        Parameters
        ----------
        data : pDF
            New data containing the original numerical features.
        subset : list, optional
            Specific features to process and return. 
            Default = None: all trained features.            

        Returns
        -------
        NumEncoderResult
            Transformed data + feature lineage metadata.
        """
        c_all = self.cols.vars_selected
        c_sub = subset or data.columns
        cols  = [x for x in c_sub if x in c_all]

        miss = c_all if not cols else None
        df = self.processor.transform(data[cols], verify = miss)

        return NumEncoderResult(
            data     = df.data, 
            features = self.cols.vars_num, 
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
    
    title   = 'NUMERICAL DISCRETIZATION PANEL'
    split   = 2
    Mapping = NumParamsMapping
        
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
# DATA → NUM → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class NumEncoderUI(ModelUIBase):
    """
    AIG Flow Data Numerical Features Encoder.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    Responsible for analyzing numerical features and transforming them 
    into a discrete space based on the selected fitting strategy.    
    """

    Model  = NumEncoder
    Option = OptionUI
        
    @wraps(NumEncoder.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(), **kwargs)
    
    @wraps(NumEncoder.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> NumEncoderResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬