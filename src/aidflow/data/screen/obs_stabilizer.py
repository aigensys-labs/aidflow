"""
Observation Stabilizer — Outlier Detection & Row-Level Reduction.

This module implements the row-level screening stage of the AIDflow
pipeline. It detects anomalous observations (e.g., atypical time
periods) using Isolation Forest, flagging rows that deviate
significantly from the bulk of the data in feature space.

What It Does
------------
1. **Preprocess**: Median imputation (robust to outliers) +
   StandardScaler (zero-mean, unit-variance).
2. **Detect**: Fit Isolation Forest ensemble on the scaled feature
   matrix. Uses contamination rate to calibrate the decision
   threshold.
3. **Score**: Assign each observation a binary outlier flag (IN/OUT)
   and a continuous anomaly score.
4. **Filter**: Remove flagged outliers from the dataset, producing
   the clean observation set for downstream modelling.

Why Observation Screening Matters
---------------------------------
A single anomalous observation (e.g., a one-time economic shock, a
data entry error, a structural break) can disproportionately
influence model fitting — especially in linear models and WOE
encoding. Removing these observations **before** training prevents:
- Coefficient distortion (GLM).
- Split instability (GBM).
- WOE bin contamination (encoding).
- Metric inflation/deflation (evaluation).

Pipeline Position
-----------------
This is the **fourth** stage (after Data Loader → Feature Cleaner →
Temporal Stabilizer). It operates on the fully prepared feature set 
and produces the final clean observation.

Dependencies
------------
Requires the Feature Stabilizer stage to be completed (provides
the stationarity-verified feature set).
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
from ...core.interop            import  Arrow

from .fitters.obs_analyzer      import  (
                                        LBL,
                                        ObservationAnalyzer,
                                        FitParamsBase,
                                        FitParamsCfg,
                                        FitStats
                                        )
from .utils_ui                  import  PanelUtilsUI


MODEL = AIGFLOW.Data.obs_stabilizer
FREG  = FitRegister(MODEL)
DREG  = DashRegister(MODEL)
DESC  = 'AIG DATA STAGE ► OBSERVATION STABILITY ► Learn Outliers and Ensure Observation Stability'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class REG(DictClass):
    """Fit Models Registry Name."""
    AIG = 'IsolationForest'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → OBSERVATION STABILITY → FITTER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.mapping
class OLParamsMapping:
    """Model - Inner Fitter params mapping to Config defaults."""
    params = Dataclass.schema(FitParamsBase)
    setup  = {
        PRM.SCREEN.th_ol.param  : params.th_outliers
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
    class IsolationForestFitter(FitParamsBase):
        
        config: dict|FitParamsCfg = FitParamsCfg(
            bootstrap     = True,     # obs sampling
            max_features  = 0.8,      # < 1 → features sampling
            random_state  = 42,
            n_jobs        = -1
        )

        def _build(self, params:FitParamsBase, config:dict) -> ObservationAnalyzer:
            setup  = dict(params) | config
            return ObservationAnalyzer(**setup)
        
        @property
        def _model(self) -> ObservationAnalyzer:
            return self.model
        
        def get_stats(self) -> FitStats:            
            return self._model.stats
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@Dataclass.base
class FitParams(FitParamsBase):
    """Fitter params input design."""       
    num_features: list
    cat_features: list = None

class Fitter(FitterBase[FitParams]):
    """
    The primary point of ML training. The low-level compute layer responsible 
    for passing parametrization to chosen internal fitter. 
    """   
    
    FREG = FREG
    DREG = DREG

    def _get_th(self, th:None|float) -> float:
        return AigParam.di(th, self.params.th_outliers)

    def fit(self, data:pDF, th: None|float = None):        
        """
        Analyze and clean variables of observation instability.

        Parameters
        ----------
        data :    Input Data for outliers learning.
        verbose : bool → True shows dashboard and stats.     
        """               
        if self._fit_no_model(MODEL, self.params.num_features):
            return
        
        th = self._get_th(th)
        self.model = self.model(th)
        self.model.fit(data, self.params.num_features)
        self.stats: FitStats = self.model.get_stats()
        self._init_dash()

        return self               
    
    def update(self, data:pDF, th_out:float):        
        """Update Outliers by TH update.""" 
        return self.fit(data, th_out)
        # NOTE Several approches have been tried regarding memory occupation vs. recomp.speed vs. UI.
        # Expected usage is that user will not ride the TH slider for fun - thus, CONFIRM button to 
        # submit TH change to data was confusing - therefore update = self.fit.
        # Historically it was combination slider + confirm button - where internally all needed data
        # for update were stored in the RAM - thus changing the slider was immediately propagated to UI.
        
    def transform(self, data: pDF) -> pDF:
        """ 
        Clean variables of observation instability.

        Parameters
        ----------
        data : New Data to apply trained transformation.
                
        Returns
        -------
        Transformed Data without outliers.
        """
        if self._is_no_model:
            return pDF({LBL.OL: [0]}, index=data.index)
        
        return self.model.transform(data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → OBSERVATION STABILITY → TRANSFORM DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class TransResData:
    """Transfomer data output design."""
    data:     pDF
    outliers: pDF

@Dataclass.base
class TransResCols:
    """Transfomer cols output design."""
    numeric: list
    vars_ol: list 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class Transformer(TransformerBase[Fitter, FitStats, TransResCols]):
    """
    Transformer meta-processing between Fitter-Model.
    
    Handles the post-processing of Fitter results. Its primary role is to standardize 
    the raw output from various fitters into a unified TransResCols, TransResData
    format, enabling seamless transition regardless of the underlying fitter choice.
    """

    ModelFitter = Fitter

    def _fit_data(self, inputs:"ObsStabilizerInput"):
        return inputs.train_set
    
    def _set_cols(self):
        prm = self.fitter.params
        return TransResCols(
            numeric = prm.num_features,
            vars_ol = [LBL.OL]
        )
    
    def transform(self, data:pDF) -> TransResData:
        """Model meta-transform."""
        
        o = self.fitter.transform(data)
        d = data.loc[o[LBL.OL]==0, :]

        return TransResData(
            data = d, 
            outliers = o
            )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → OBSERVATION STABILITY → PROCESSOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Processor(ModelProcessorBase[Transformer, TransResCols, TransResData]):
    """
    Logic Processor.
    
    Coordinates the lifecycle between FitChoice selection and Parameter resolution. 
    It ensures that params and dependencies are correctly injected into the 
    Transformer - Fitter before execution begins.
    """   
    
    MODEL = MODEL
    DESC  = DESC

    Config  = FitParamsCfg
    Mapping = OLParamsMapping

    ModelTransformer = Transformer
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_choice(self, fx_choice:str, inputs:"ObsStabilizerInput"):
        defaults  = FitParamsCfg()
        fx_config = self.get_fit_config(fx_choice, inputs.fx_config, defaults)
        return RegGet.choice(fx_choice, fx_config)  
        
    def get_params(self, fx_choice:str, inputs:"ObsStabilizerInput", params:dict):
        params = self.get_fit_params(fx_choice, inputs.fx_params, params)

        fit = FitParams(
            num_features = inputs.num_features,
            cat_features = inputs.cat_features,
            **params
        )                
        return fit, self.setup      
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬    



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬    
# DATA → OBSERVATION STABILITY
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬    
@Dataclass.base(dynamic_params=True)
class ObsStabilizerInput(ModelInput):
    """
    ObservationStabilizer fit input specification.

    Defines the training data and feature set required to execute
    the outlier detection pipeline.

    Attributes
    ----------
    train_set : pDF
        Training data (in-sample). The dataset on which outlier
        detection is performed. Must contain the numeric features
        used for anomaly scoring.
    num_features : list
        Numeric features to use for outlier detection. These are
        the columns that define the feature space in which
        Isolation Forest operates.
    cat_features : list | None, default=None
        Categorical features (informational only). Not used
        directly in outlier scoring (Isolation Forest requires
        numeric input) but preserved for pipeline context.

    Notes
    -----
    **Numeric-Only Scoring**: Isolation Forest operates on
    numeric feature space. Categorical features are not
    included in the anomaly score — the detection is based
    purely on numeric feature deviation.
    """
    train_set:      pDF
    num_features:   list
    cat_features:   list|None = None
    
@Dataclass.base
class ObsStabilizerResult(ModelResult):
    """
    ObservationStabilizer output — the outlier-filtered dataset.

    The final result of the observation-level screening stage.
    Contains the clean data (outliers removed) and the outlier
    flags for audit and inspection.

    Attributes
    ----------
    data : pDF
        Cleaned training data. Observations flagged as outliers
        are removed. The result is in native Pandas/NumPy dtypes
        (Arrow conversion boundary) — ready for downstream ML
        libraries (LightGBM, Scikit-Learn, etc.).
    outliers : pDF
        Outlier flags per observation. Contains the OUTLIER
        column (1 = outlier, 0 = inlier) and the original
        row indices. Used for audit trail and user inspection.

    Notes
    -----
    **Memory Boundary**: The returned data is converted to
    native Pandas dtypes at this stage (the final step of the
    screening phase). This single-conversion boundary minimizes
    peak pipeline memory by ensuring native allocations only
    occur on the finalized, reduced dataset.

    **Flag, Don't Drop**: The outlier flags are preserved in the
    `outliers` field. The decision to exclude is already made
    (reflected in `data`), but the flags remain available for
    user review and potential override.

    **Contamination Rate**: The proportion of removed outliers
    equals the configured `th_outliers` threshold (default
    1%). Users can adjust this interactively via `update`.
    """
    data:     pDF
    outliers: pDF
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class ObservationStabilizer(Processor, ModelBase):
    """
    Observation Stabilizer — Outlier Detection & Row-Level Reduction.

    The row-level screening gate of the AIDflow pipeline.
    Detects anomalous observations using Isolation Forest,
    flagging and removing rows that deviate significantly from
    the bulk of the data in feature space.

    What It Does
    ------------
    1. **Preprocess**: Median imputation + StandardScaler.
    2. **Detect**: Fit Isolation Forest on the scaled feature
       matrix (contamination-calibrated threshold).
    3. **Flag**: Assign IN/OUT verdict + anomaly score per row.
    4. **Filter**: Remove flagged outliers from the dataset.

    Two Entry Points
    ----------------
    - **fit(inputs)**: Full outlier detection + filtering on the
      training set. Produces the clean observation set.
    - **transform(data)**: Apply the trained outlier model to
      new data. Use for OOT scoring or production inference.

    Parameters
    ----------
    setup : Parameters
        Flow context parametrization accessible via ctx.setup.

    Notes
    -----
    **Unsupervised**: Detection is purely based on feature-space
    deviation. No target variable is required.

    **Robust Preprocessing**: Missing values are imputed with
    the training median (robust to outliers) before scoring.
    A single missing cell does not exclude an observation.

    **Interactive Override**: The contamination threshold can
    be adjusted interactively (slider in UI) without re-running
    the full pipeline. The user can increase or decrease the
    outlier proportion to match their risk appetite.

    **Memory Management**: This stage is the Arrow → Pandas
    conversion boundary. Data is converted to native Pandas
    dtypes after row filtering to minimize peak memory
    consumption — downstream ML libraries (LightGBM,
    Scikit-Learn) require native dtypes.

    **Feature Comparison**: The diagnostic output includes a
    per-feature IN vs OUT mean comparison, helping the user
    identify which features drive the outlier classification.

    Examples
    --------
    >>> from aidflow.data import ObservationStabilizer
    >>> stabilizer = ObservationStabilizer(ctx.setup)
    >>>
    >>> # Full outlier detection + filtering
    >>> stabilizer.fit(ObsStabilizerInput(
    ...     train_set    = train_data,
    ...     num_features = num_fes,
    ... ))
    >>>
    >>> # Adjust threshold interactively
    >>> stabilizer.update(th_ol=0.02)
    >>>
    >>> # Apply to new data
    >>> result = stabilizer.transform(new_data)
    >>>
    >>> # Access results
    >>> result.data         # Data subset wo outliers
    >>> result.outliers     # Per-row outlier flags
    """

    def _save(self, save_path) -> None:
        features = {'features': self.cols.vars_ol}
        ModelIO(self.setup).save(MODEL, self, features, custom_path = save_path) 

    def _to_pd(self, data:pDF) -> pDF:
        """
        Executes deferred conversion of Arrow-backed data to native Pandas dtypes 
        at the end of the screening phase.

        Behavior:
        - offload == False:
            Converts data to native Pandas dtypes now (`Arrow.safe_conversion`).
            Dependency: `fe_cleaner` preserved PyArrow dtypes through row filtering.
        - offload == True:
            Passes data through as-is.
            Dependency: `fe_cleaner` already offloaded Arrow memory and converted 
            to native Pandas dtypes upstream.

        Rationale:
            Downstream ML packages (e.g., LightGBM, Scikit-Learn) do not natively 
            support `ArrowDtype` and trigger inefficient on-the-fly conversions during 
            training. Deferring conversion until after row dropping (outliers removal) 
            minimizes overall RAM consumption by ensuring native Pandas allocations 
            only occur on the finalized, reduced dataset.
        """              
        if self.setup.offload:
            return data
        return Arrow.to_safe_datatype(data)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, inputs:ObsStabilizerInput, fx_choice: str  = REG.AIG, **params):
        """
        Execute the full outlier detection and filtering pipeline.

        Trains Isolation Forest on the feature matrix, scores
        all observations, and removes flagged outliers from the
        training set.

        Parameters
        ----------
        inputs : ObsStabilizerInput
            Training data and feature lists.
        fx_choice : str, optional
            The fitting strategy identifier. Defaults to REG.AIG.
        **params : dict
            Runtime parameters passed to the internal fitter
            (th_outliers).

        Returns
        -------
        ObservationStabilizer
            The fitted instance with the clean observation set
            and outlier flags available.
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
        self.custom_path = inputs.save_path
        self._save(self.custom_path)
        self._set_update_data(lambda: inputs.train_set)
        return self
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def update(self, *args, **kwargs):
        """
        Update the contamination threshold and re-filter observations.

        Re-scores all observations under the new threshold using
        the fitted Isolation Forest (no re-fitting of the model).

        Parameters
        ----------
        *args : Any
            Positional arguments for the update process.
        **kwargs : dict
            Keyword arguments including the new threshold
            (th_ol).

        Returns
        -------
        ObservationStabilizer
            The updated instance with the new clean set.
        """
        data = self._get_update_data()
        self.processor.update(data, kwargs[PRM.SCREEN.th_ol.param])
        self._save(self.custom_path)
        return self    
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def transform(self, data:pDF) -> ObsStabilizerResult:
        """
        Apply the trained outlier filter to new data.

        Parameters
        ----------
        data : pDF
            Input data to filter.

        Returns
        -------
        ObsStabilizerResult
            Cleaned data (outliers removed) + outlier flags.

            As the final step of the Screening phase, returned `data` is converted to native 
            Pandas/NumPy dtypes. This single-conversion boundary minimizes peak pipeline 
            memory footprint by preventing redundant downstream allocations and ensuring 
            full compatibility with numpy, LightGBM, XGBoost, and Scikit-Learn.
        """
        miss = [x for x in self.cols.numeric if (x not in data.columns)]                
        df = self.processor.transform(data, verify = miss)
        return ObsStabilizerResult(
            data     = self._to_pd(df.data),
            outliers = df.outliers
        )  
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬    



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → OBSERVATION STABILITY → UI → OPTIONS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class OptionUI(OptionUIBase):
    """UI Dynamic Parameters Panel."""

    title = 'OBSERVATION STABILITY'

    def __init__(self, setup:Parameters, input:ObsStabilizerInput):
        prm = OLParamsMapping(setup, default=True)
        self.wg = PanelUtilsUI(prm, self.title) 
        
    @property
    def slider(self):
        return self.wg.slider()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → OBSERVATION STABILITY → UI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ObservationStabilizerUI(ModelUIBase[ObservationStabilizer, OptionUI]):
    """
    AIG Flow Data Observation Stability.
    Model Wrapper to provide UI as Dynamic User Panel - Dashboard - Data Outputs.
    
    Screens the dataset for outlier observations that may distort 
    model training and ensures observation-level stability.
    """
    
    Model  = ObservationStabilizer
    Option = OptionUI

    def custom_control(self, fx):
        """Custom UI widget."""
        return self.option_ui.slider.observe(fx, names='value')

    @wraps(ObservationStabilizer.fit)
    def fit(self, *args, **kwargs): 
        """Observation Stability fit UI wrapper."""                        
        self.model_ui.fit(   
            *args,
            ui_params = self._get_ui_fit_defaults(
                output_wait                 = True,
                parent_has_update_FX_method = True,
                gui_static_before_confirm_1 = None,
                custom_control_widget       = self.custom_control,
            ),
            **kwargs
        )
        return self

    @wraps(ObservationStabilizer.transform)
    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    @property
    def fitted(self) -> ObsStabilizerResult: return self.model_ui.fitted
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬