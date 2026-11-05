"""
Agent 4 — GBM Training with Non-Linear Feature Selection.

This module implements the fourth and final training agent of the
AIDflow pipeline. It applies Gradient Boosting (LightGBM or XGBoost)
to capture non-linear effects and feature interactions that linear
models cannot express.

What It Does
------------
1. **Feature Selection**: Two-stage pipeline:
   - Stage 1: Tree-based selection with early stopping.
     Identifies features that contribute to predictive power
     in a non-linear context (split importance + validation metric).
   - Stage 2: Shadow pruning (permutation testing) to remove
     features whose importance is not significantly above random.
2. **Model Fitting**: GBM (LightGBM/XGBoost) on the surviving
   features with early stopping.
3. **Diagnostics**: Validation metrics (Gini, AUC, KS, lift)
   on the held-out set.
4. **Importance**: Feature ordering by contribution.

Strategy: Non-Linear Performance
--------------------------------
Agent 4 is the "Performance Powerhouse" of the learning ensemble,
it captures:
- **Non-linear effects**: Threshold behaviors, saturation points.
- **Interactions**: Feature pairs/triples that are predictive
  only in combination (no explicit engineering needed).
- **Categorical handling**: Native support.

When to Use Agent 4
-------------------
- The relationship between features and target is non-linear.
- Feature interactions are important (e.g., income × tenure).
- Categorical features are present and important.
- Maximum predictive accuracy is the priority (and interpretability
  can be addressed via SHAP post-hoc).

Role in the Ensemble
--------------------
Agent 4 provides the **performance ceiling** — the highest
achievable AUC/Gini from the feature set. If Agent 4
significantly outperforms Agents 1–3, it indicates that
non-linear effects or interactions are driving the signal.
The Comparator stage will surface this gap.
"""

from ...deps.common             import  wraps
from ...deps.data               import  pDF
from ...deps.types              import  NDArray

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
                                        DictClass
                                        )
from ...core.foundation         import  AigParam                                        
from ...core.io                 import  ModelIO

from .fitters.gbm_selector      import  (
                                        LightBinaryGBM,
                                        ExtremeBinaryGBM,
                                        FitParamsBase,
                                        FitParamsCfg,
                                        FitResult,
                                        FitStats,
                                        FitDashLink
                                        )

MODEL = AIGFLOW.Training.agent4
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG TRAINING STAGE ► AGENT 4 TRAINING ► GBM Optimization'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    LGB: str = 'LGB'
    XGB: str = 'XGB'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → FIT DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class AgentParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(FitParamsBase)
    setup  = {
        PRM.TRAIN.gb_leaves.param       : (params.max_leaves,       'Max Leaves'                ),
        PRM.TRAIN.gb_learning.param     : (params.learning_rate,    'Learning Rate'             ),
        PRM.TRAIN.gb_min_child.param    : (params.min_child_weight, 'Min Child Weight'          ),
        PRM.TRAIN.gb_estimators.param   : (params.n_estimators,     'N Estimators'              ),
        PRM.TRAIN.shadow_quantile.param : (params.shadow_quantile,  'Shadow Pruning Quantile'   ),
        PRM.TRAIN.cv.param              : (params.cv,               'Cross-Valid Shadow Folds'  ),
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
    
    def get_extra(self) -> FitDashLink:
        return self._model.dash

class FitOptions(FitRegOpt):
    """
    Model-to-Fitter Registration Layer.
    
    This class acts as the bridge between the high-level Model and the low-level Fitter.
    It utilizes a registration pattern to ensure that adding new prediction 
    algorithms only requires a new entry here, keeping the top-layer API stable.
    """  

    @FREG.register(REG.LGB)
    class LightBinaryGBMFitter(InnerFitterBase):

        @property
        def _model(self) -> LightBinaryGBM:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> LightBinaryGBM:            
            setup  = dict(params) | config
            return LightBinaryGBM(**setup)

    @FREG.register(REG.XGB)
    class ExtremeBinaryGBMFitter(InnerFitterBase):

        @property
        def _model(self) -> ExtremeBinaryGBM:
            return self.model        
        
        def _build(self, params:FitParamsBase, config:dict) -> ExtremeBinaryGBM:            
            setup  = dict(params) | config
            return ExtremeBinaryGBM(**setup)                
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class FitParams(FitParamsBase):
    """Fitter params input design."""
    num: list
    cat: list

@Dataclass.base
class DashLink(FitDashLink):
    """Fitter dash link."""
    data: pDF

class Fitter(FitterBase[FitParams]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """     
    
    FREG = FREG
    DREG = DREG
           
    def fit(self, data:pDF): 
        """
        ML Agent training.
        
        Parameters
        ----------
        data: pDF
            Input Data for Agent training.
        verbose: bool
            True shows dashboard and stats.
        """   
        self.model.fit(data, num_features=self.params.num, cat_features=self.params.cat)
        self.result: FitResult = self.model.get_result()
        self.stats:  FitStats  = self.model.get_stats()
        self._init_dash(
            fit_link = DashLink(data = data, **self.model.get_extra())
        )
        return self
    
    def predict_proba(self, data:pDF) -> NDArray:
        """
        Predict response variable of a model given chosen significant exogenous features.
        
        Parameters
        ----------
        data : pDF
            Input DataFrame containing the features required by the model.
            
        Returns
        ------
        NDArray
            2-D Numpy array of probabilities corresponding to the target class 0 / 1.
        """        
        return self.model.predict_proba(data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → TRANSFORMER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    features:   list
    selected:   list
    dropped:    list
    raw:        list

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

    def _fit_data(self, inputs:"Agent4Input"): 
        return inputs.train_set 
        
    def _set_cols(self):
        res = self.fitter.result        
        return TransResCols(
            features = [self.setup.target],
            selected = res.features_selected,
            dropped  = res.features_dropped,
            raw      = res.features_selected
        )
        
    def transform(self, data:pDF) -> TransResData: 
        """Model meta-transform."""
        d = self.fitter.predict_proba(data[self.cols.selected])[:, 1]
        return TransResData(
            data = pDF(d, columns = self.cols.features, index = data.index)
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → PROCESSOR
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
    Mapping = AgentParamsMapping

    ModelTransformer = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_choice(self, fx_choice:str, inputs:"Agent4Input"):
        defaults = self.Config(
            target   = AigParam.di(inputs.target, self.setup.target),
            val_size = PRM.LOAD.valid.value,
            verbose  = AigParam.di(inputs.verbose, self.setup.verbose)
        )        
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

    def get_params(self, fx_choice:str, inputs:"Agent4Input", params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params)
        fit = FitParams(
            num = inputs.num_features, 
            cat = inputs.cat_features, 
            **params
        )
        return fit, self.setup
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → API
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base(dynamic_params=True)
class Agent4Input(ModelInput):
    """
    Agent 4 fit input specification.

    Defines the training data and feature sets (numerical +
    categorical) required to execute the GBM training pipeline.

    Attributes
    ----------
    train_set : pDF
        Training data. Must contain the target column, numerical
        features, and optionally categorical features.
    num_features : list | None, default=None
        Numerical features to consider for selection. If None,
        all non-target columns are treated as numerical.
    cat_features : list | None, default=None
        Categorical features (natively handled by the GBM, no
        encoding required). If None, no categorical features
        are used.

    Notes
    -----
    **Feature Resolution**: If `num_features` is None, all
    non-target, non-categorical columns are used. If both are
    None, all non-target columns are treated as numerical.
    """
    train_set:      pDF
    num_features:   list|None = None
    cat_features:   list|None = None

@Dataclass.base
class Agent4Result(ModelResult):
    """
    Agent 4 output — the trained GBM prediction contract.

    The final result of the GBM training. Contains the probability
    predictions along with full feature lineage metadata for
    audit and downstream consumption.

    Attributes
    ----------
    data : pDF
        Prediction data — a single-column DataFrame of
        P(class=1) probabilities, indexed identically to the
        input data.
    features : list
        Output column name(s) for the prediction
        (e.g., `['score']`).
    selected : list
        Features that survived tree-based selection and shadow
        pruning (both numerical and categorical).
    dropped : list
        Features that were candidates but removed during
        selection.
    raw_features : list
        Original feature names for the selected features.
        Equal to selected (API compatibility with GLM agents).
    """
    data:           pDF
    features:       list
    selected:       list
    dropped:        list
    raw_features:   list
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Agent4Predictor(Processor, ModelBase):     
    """
    Agent 4 — GBM Training with Non-Linear Feature Selection.

    The "Performance Powerhouse" of the learning ensemble. Applies
    Gradient Boosting (LightGBM or XGBoost) to capture non-linear
    effects and feature interactions that linear models cannot
    express, then fits a production-ready boosted tree ensemble.

    What It Does
    ------------
    1. **Selects** features via two-stage pipeline:
       - Stage 1: Tree-based selection with early stopping
         (captures non-linear signal and interactions).
       - Stage 2: Shadow pruning (permutation stability test).
    2. **Fits** GBM (LightGBM/XGBoost) on the surviving features
       with early stopping.
    3. **Diagnoses** on a held-out validation set.
    4. **Orders** features by importance.

    Two Entry Points
    ----------------
    - **fit(inputs, fx_choice)**: Full training pipeline.
      `fx_choice` selects the engine:
      - `REG.LGB` (default): LightGBM — leaf-wise, fast.
      - `REG.XGB`: XGBoost — depth-wise, robust.
    - **transform(data)**: Inference — applies the trained
      model to generate P(class=1) probabilities.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **Non-Linear**: Agent 4 captures threshold effects, saturation, 
    and feature interactions automatically. If the signal is non-linear,
    Agent 4 will typically outperform the GLM agents.

    **Categorical Support**: Native categorical feature handling.
    No encoding required — pass them via `cat_features`.

    **Overfitting Control**: Two-stage selection (tree selector
    + shadow pruning) + early stopping provides multiple
    safeguards against overfitting. The `max_leaves`,
    `min_child_weight`, and `shadow_quantile` parameters
    control the bias/variance trade-off.

    **Interpretability**: Lower than GLM agents (no direct
    coefficients). Use SHAP values for feature-level
    explainability. Suitable for environments where
    """
    
    def _save(self, save_path) -> None:
        features = {
            'features': self.cols.features,
            'selected': self.cols.selected
        }
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs: Agent4Input, fx_choice: str = REG.LGB, **params):
        """
        Train the GBM agent on numerical and/or categorical features.

        Executes the full two-stage feature selection pipeline
        (tree-based selection + shadow pruning), fits the final
        GBM with early stopping, and computes validation
        diagnostics.

        Parameters
        ----------
        inputs : Agent4Input
            Training configuration including train_set,
            num_features, and cat_features.
        fx_choice : str, optional
            The GBM engine: REG.LGB (LightGBM, default)
            or REG.XGB (XGBoost).
        **params : dict
            Runtime parameters passed to the internal fitter
            (gb_max_leaves, gb_learning_rate, gb_estimators,
            gb_min_child_weight, shadow_quantile, cross_val_folds).

        Returns
        -------
        Agent4Predictor
            The fitted agent instance.
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

    def transform(self, data:pDF) -> Agent4Result:
        """
        Apply the trained agent to generate P(class=1) probabilities.

        Parameters
        ----------
        data : pDF
            Input data containing the selected features
            (numerical and/or categorical).

        Returns
        -------
        Agent4Result
            Predictions + full feature lineage metadata.
        """
        vars_miss = [x for x in self.cols.selected if (x not in data.columns)]                
        df = self.processor.transform(data, verify = vars_miss)
        return Agent4Result(
            data         = df.data, 
            features     = self.cols.features,
            selected     = self.cols.selected,
            dropped      = self.cols.dropped,
            raw_features = self.cols.raw
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI(OptionUIBase):    
    title   = 'GBM Optimization Panel'.upper()
    split   = 3
    width   = 150
    Mapping = AgentParamsMapping
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING → AGENT → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Agent4PredictorUI(ModelUIBase):
    """
    Agent 4: Non-Linear Gradient Boosting Machine (GBM).
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    Agent 4 serves as the 'Performance Powerhouse' of the learning ensemble. Unlike 
    the previous agents that rely on linear relationships, Agent 4 utilizes 
    Gradient Boosting Decision Trees (via LightGBM) to capture complex, non-linear 
    interactions and high-order dependencies within the feature set.       
    """

    Model  = Agent4Predictor
    Option = OptionUI
        
    @wraps(Agent4Predictor.fit)
    def fit(self, *args, **kwargs):
        """Encoder fit UI wrapper."""
        self.model_ui.fit(*args, ui_params = self._get_ui_fit_defaults(), **kwargs)  
    
    @wraps(Agent4Predictor.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> Agent4Result: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬