"""
Observation Outlier Detection (Row-Dimension Reduction).

This module implements the row-level screening stage of the pipeline,
responsible for detecting and flagging anomalous observations (e.g.,
atypical time periods) before modelling. It uses Isolation Forest to
identify observations that deviate significantly from the bulk of the
data in the feature space.

Design Principles
-----------------
- **Unsupervised**: No target variable is used. Outliers are defined
  purely by feature-space deviation.
- **Contamination-based**: The outlier proportion is set a priori
  (`th_outliers`, default 1%). Isolation Forest uses this to calibrate
  its decision threshold.
- **Robust preprocessing**: Missing values are imputed with the median
  (robust to outliers), then features are standardised (zero-mean,
  unit-variance) before scoring.
- **Flag, don't drop**: Observations are labelled IN/OUT with a
  continuous score. The decision to exclude is made downstream
  (model stage), preserving user agency.

Output
------
- Per-observation: outlier flag (1/0), verdict ('IN'/'OUT'), and
  a scaled anomaly score.
- Aggregate: contamination rate, sample of flagged rows, and a
  per-feature comparison of IN vs OUT means.
"""

from ....deps.data          import  np, pDF
from ....deps.ml.dim        import  (
                                    IsolationForest, 
                                    StandardScaler, 
                                    scale
                                    )
from ....core.base          import  (
                                    FitterDataBase, 
                                    FitterModelBase
                                    )
from ....core.factory       import  (
                                    Dataclass,
                                    DictClass
                                    )
from ....core.foundation    import  AigParam
from ....core.interop       import  Pd

from ....ml.sampling        import  Sampling


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# LOCAL REG-LABELS-CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    """ol-vr-sc labels"""
    OL = 'OUTLIER'
    VR = 'VERDICT'
    SC = 'SCORE'

class IO(DictClass):
    """Outliers IN / OUT"""   
    IN  = 'IN'
    OUT = 'OUT'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → OBSERVATION STABILITY → FITTER DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class FitParamsBase:
    """
    Outlier detection sensitivity parameter.

    Attributes
    ----------
    th_outliers : float, default=0.01
        Expected proportion of outliers (contamination rate).
        Isolation Forest uses this to calibrate its decision threshold:
        observations in the most extreme ``th_outliers`` fraction of the
        anomaly score distribution are flagged as outliers.

        Lower values are more conservative (fewer observations flagged).
        Higher values are more aggressive (more observations flagged).
        Default 0.01 flags the most extreme 1% of observations.
    """     
    th_outliers: float = 0.01

@Dataclass.base(update_specs=dict(slots=False))
class FitParamsCfg:
    """
    Isolation Forest estimator configuration.

    Attributes
    ----------
    bootstrap : bool, default=True
        If True, each tree is built on a random subsample of observations
        (with replacement). Provides variance reduction in the ensemble.
    max_features : float, default=0.8
        Fraction of features to consider at each split. Values < 1
        enable feature subsampling, which decorrelates the trees and
        reduces overfitting to specific feature combinations.
    random_state : int, default=42
        Seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel workers for tree construction (-1 = all cores).
    """ 
    bootstrap:    bool  = True     # obs sampling
    max_features: float = 0.8      # < 1 → features sampling
    random_state: int   = 42
    n_jobs:       int   = -1

@Dataclass.base
class FitParams(FitParamsBase, FitParamsCfg):
    """Outliers params input design."""       

@Dataclass.base
class FitStats:
    """
    Diagnostic output from the outlier detection fit.

    Attributes
    ----------
    th : float
        Contamination rate used (expected outlier proportion).
    len_data : int
        Total number of observations processed.
    sum_ol : int
        Number of observations flagged as outliers.
    sample : pDF
        Random subsample (max 2000 rows) of the scored data with
        per-feature values and outlier verdict — for user inspection.
    compare : pDF
        Per-feature mean comparison of IN vs OUT groups (transposed).
        Highlights which features drive the outlier classification.
    """
    th:         float
    len_data:   int
    sum_ol:     int
    sample:     pDF
    compare:    pDF
# ─────────────────────────────────────────────────────────────────────────────────────────────────
   
class FitDataScaler:
    """
    Preprocessing pipeline for outlier detection.

    Applies median imputation (robust to outliers) followed by
    StandardScaler (zero-mean, unit-variance) to produce a feature
    matrix suitable for Isolation Forest scoring.

    The learned medians and scaler are preserved for consistent
    transformation of new data during `transform`.

    Attributes
    ----------
    medians : pSe
        Column-wise median values from the training data (for imputation).
    scaler : StandardScaler
        Fitted scaler (mean and std per feature).
    """       
    
    medians = None
    scaler  = None    
        
    def _fillna(self, data:pDF) -> pDF:        
        if self.medians is None:
            self.medians = data.median() 
        return data.fillna(self.medians)

    def _fill_data(self, data:pDF) -> pDF:  
        return self._fillna(data)
    
    def _fit_scaler(self, data:pDF) -> None:
        if self.scaler is None:
            self.scaler = StandardScaler()        
            self.scaler.fit(data) 
        
    def _scale_data(self, data:pDF) -> pDF:
        data_sc = self.scaler.transform(data)
        return pDF(data_sc, columns=data.columns, index=data.index)
    
    def data_pipe(self, data: pDF) -> tuple[pDF, pDF]:
        """
        Execute the full preprocessing pipeline: impute → fit scaler → scale.

        Parameters
        ----------
        data : pDF
            Raw feature matrix (may contain NaN).

        Returns
        -------
        tuple[pDF, pDF]
            d_scaled : Scaled data (zero-mean, unit-variance) for model input.
            d_std    : Imputed data (median-filled, unscaled) for display/inspection.
        """       
        d_std = self._fill_data(data)   # relatively cheap to compute
        self._fit_scaler(d_std)         # -- no need to keep extra 2xdata in memory for update  
        return self._scale_data(d_std), d_std
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class FitModel:
    """
    Isolation Forest outlier detection with preprocessing and scoring.

    Orchestrates the full detection pipeline:
    1. Preprocess (impute + scale).
    2. Fit Isolation Forest ensemble.
    3. Score observations (flag, verdict, anomaly score).
    4. Generate diagnostic stats (sample, feature comparison).

    Parameters
    ----------
    th_outliers : float
        Contamination rate (expected outlier proportion).
    config : dict
        Isolation Forest hyperparameters (bootstrap, max_features, etc.).
    """      
    _estimator: IsolationForest = None

    def __init__(self, th_outliers:float, config:dict):
        self.th     = th_outliers 
        self.cfg    = config
        self.scaler = FitDataScaler()

    @property
    def model(self) -> IsolationForest:
        if self._estimator is None:
            raise RuntimeError('Run .fit prior prediction.')
        return self._estimator
    
    def get_setup(self) -> dict:
        th_min = 1e-10
        th_out = dict(
            contamination = max(self.th, th_min),
            random_state = 42
        )
        return th_out | self.cfg

    def fit(self, X:pDF):
        """
        Train the Isolation Forest on preprocessed feature data.

        Parameters
        ----------
        X : pDF
            Scaled feature matrix (output of `FitDataScaler.data_pipe`).
        """        
        setup = self.get_setup()
        self._estimator = IsolationForest(**setup)
        self._estimator.fit(X)
    
    def predict(self, data:pDF, is_scaled:bool=False) -> pDF:
        """
        Classify observations as outlier or inlier.

        Parameters
        ----------
        data : pDF
            Feature matrix to classify.
        is_scaled : bool, default=False
            If True, data is already preprocessed (skips scaling).
            If False, the full preprocessing pipeline is applied.

        Returns
        -------
        pDF
            Single-column DataFrame with `OUTLIER` (1 = outlier, 0 = inlier),
            indexed identically to the input.
        """         
        data_= data
        if not is_scaled:
            data_, _  = self.scaler.data_pipe(data)

        pred = self.model.predict(data_)   
        pred = np.where(pred == -1, 1, 0) 

        return pDF({LBL.OL: pred}, index = data.index) 
            
    def _scale_decision(self, data: pDF) -> np.ndarray:
        """
        Compute the anomaly score for each observation.

        The raw Isolation Forest decision function is normalised
        (z-scored) to produce a comparable continuous score where
        higher values indicate greater anomaly.
        """
        return scale(self.model.decision_function(data))   
        
    def _get_ovs(self, data_scaled:pDF) -> pDF:
        """Compute outlier label, verdict, and score for scaled data."""
        ol = self.predict(data_scaled, True).squeeze() 
        vr = np.where(ol==1, IO.OUT, IO.IN)
        sc = self._scale_decision(data_scaled)
        return Pd.df_zip(
            LBL.list(), [ol, vr, sc], data_scaled.index
        )
    
    def stats(self, X_scaled:pDF, X_std:pDF) -> FitStats:
        """
        Generate diagnostic statistics for user inspection.

        Parameters
        ----------
        X_scaled : pDF
            Scaled data (for model scoring).
        X_std : pDF
            Imputed (unscaled) data (for readable display).

        Returns
        -------
        FitStats
            Contamination rate, outlier count, sample rows, and
            per-feature IN/OUT mean comparison.
        """
        data = X_std.copy()        
        data[LBL.list()] = self._get_ovs(X_scaled)

        sample  = Sampling(2e3).subsample(data, reset_index=True)
        compare = (
            data[ [LBL.VR] + X_std.columns.to_list() ]
            .groupby(LBL.VR).mean().transpose()
        )
        return FitStats(
            th       = self.th,
            len_data = len(data),
            sum_ol   = sum(data[LBL.OL]),
            sample   = sample,
            compare  = compare
        )
    
    def process(self, X:pDF) -> FitStats:
        """
        Run the complete outlier detection pipeline.

        Executes: preprocess → fit model → score observations → generate stats.

        Parameters
        ----------
        X : pDF
            Raw feature matrix (may contain NaN).

        Returns
        -------
        FitStats
            Full diagnostic output.
        """
        d_scaled, d_std = self.scaler.data_pipe(X)                
        self.fit(d_scaled)
        return self.stats(d_scaled, d_std)          

@Dataclass.verify
class Fitter(FitterDataBase):
    """
    Outlier detection — execution backend.

    Manages the fit-transform lifecycle for observation-level anomaly
    detection:

    - **fit**: Preprocess, train Isolation Forest, score all observations,
      and store diagnostic stats.
    - **transform**: Score new observations using the fitted model and
      learned preprocessing.

    Parameters
    ----------
    params : FitParams
        Contamination rate and estimator configuration.
    config : dict
        Isolation Forest hyperparameters.
    """    
    _model: FitModel = None

    def __init__(self, params:FitParams, config:dict):
        self.prm = params
        self.cfg = config

    def fit(self, data:pDF):
        """
        Fit the outlier detection model on the full feature matrix.

        Parameters
        ----------
        data : pDF
            Feature matrix (numeric columns, may contain NaN).

        Returns
        -------
        self
            Fitted instance with `stats`, `features`, and `model` set.
        """
        self._model   = FitModel(self.prm.th_outliers, self.cfg)
        self.stats    = self._model.process(data)        
        self.features = data.columns.to_list()
        self.result   = True

        return self  
        
    def transform(self, data:pDF) -> pDF:
        """
        Predict outlier status for new observations.

        Applies the learned preprocessing (median imputation + scaling)
        and the fitted Isolation Forest to produce outlier labels.

        Parameters
        ----------
        data : pDF
            New observation matrix (same columns as training).

        Returns
        -------
        pDF
            DataFrame with `OUTLIER` column (1 = outlier, 0 = inlier),
            indexed identically to the input.
        """
        return self.model.predict(data[self.features])
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# OBS → OUTLIERS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ObservationAnalyzer(FitterModelBase[Fitter]):
    """
    Observation Outlier Detection (Row-Dimension Reduction).

    Detects anomalous observations (e.g., atypical time periods) using
    Isolation Forest on the screened feature matrix. Observations are
    flagged with a binary outlier label, a categorical verdict (IN/OUT),
    and a continuous anomaly score. The decision to exclude flagged
    observations is made downstream — this module only scores.

    Parameters
    ----------
    th_outliers : float, default=0.01
        Expected proportion of outliers (contamination rate).
        Observations in the most extreme fraction of the anomaly score
        distribution are flagged. Lower = more conservative.
    bootstrap : bool, default=True
        Enable bootstrapping in the Isolation Forest ensemble.
    max_features : float, default=0.8
        Fraction of features per split (feature subsampling).
    random_state : int, default=42
        Seed for reproducibility.
    n_jobs : int, default=-1
        Parallel workers for tree construction.

    Notes
    -----
    **Unsupervised**: Detection is purely based on feature-space deviation.
    No target variable is required.

    **Robust to missing values**: NaN is imputed with the training median
    before scoring, so a single missing cell does not exclude an observation.

    **Flag, don't drop**: The output is a label + score, not a hard exclusion.
    The model stage decides whether to exclude flagged observations,
    preserving user agency and interpretability.

    **Feature comparison**: The `stats.compare` output shows per-feature
    IN vs OUT means, helping the user identify *which* features drive
    the outlier classification.

    Examples
    --------
    >>> from aidflow.fitters import ObservationAnalyzer
    >>> oa = ObservationAnalyzer(th_outliers=0.01, n_jobs=4)
    >>> oa.fit(data_screened, num_features=['revenue', 'costs', 'margin'])
    >>> result = oa.transform(data_new)  # doctest: +SKIP
    >>> result  # OUTLIER | VERDICT | SCORE  # doctest: +SKIP
    """
    _estimator: type = IsolationForest

    def __init__(self, *args, params: dict|FitParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(FitParams, params, args, kwargs)
        self._config = self.get_estimator_config(self._params, kwargs)

    def _get_num(self, data:pDF, num:list|None) -> list:
        if num is not None:
            return num
        return [x for x in data.columns]
    
    def fit(self, data:pDF, num_features:list|None = None):        
        """
        Fit the outlier detection model on the feature matrix.

        Parameters
        ----------
        data : pDF
            Feature matrix (numeric columns, may contain NaN).
        num_features : list | None, default=None
            Specific numeric columns to use. If None, all columns are used.

        Returns
        -------
        self
            Fitted instance ready for `transform`.
        """          
        num = self._get_num(data, num_features)
        self._fitter = Fitter(self._params, self._config)
        self._fitter.fit(data[num])
        return self
    
    def transform(self, data:pDF) -> pDF:
        """
        Score new observations for outlier status.

        Parameters
        ----------
        data : pDF
            New observation matrix (same columns as training).

        Returns
        -------
        pDF
            DataFrame with `OUTLIER` (1/0) and `VERDICT` (IN/OUT)
            for each observation, indexed identically to input.
        """
        return self.fitter.transform(data)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬