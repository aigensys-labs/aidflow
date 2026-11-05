"""
Context-Aware Inference Engine — Production Scoring API.

This module implements the `Predictor`, the primary inference
interface for applying trained AIDflow pipelines to new, unseen
datasets. It provides a symmetrical API that mirrors the training
pipeline's structure, ensuring that all preprocessing — from data
cleaning through feature stabilization to WOE encoding — is
executed with the exact parameters locked during training.

Training-Serving Symmetry
-------------------------
The Predictor guarantees that inference follows the identical
transformation path used during model development:

    raw data → fe_clean → fe_stabilize → obs_stabilize
    → num_encode → cat_encode → ixn_encode
    → woe_encode → woe_stabilize → agent predict

Each stage applies the trained standards (binning thresholds,
imputation values, stability filters, encoding maps) — not
re-derived statistics. This eliminates training-serving skew.

The `preprocess` Parameter
---------------------------
Most predict methods accept a `preprocess` flag (default: True):

- **preprocess=True**: Accepts the original input columns
  (same as training). The full preprocessing pipeline is
  applied automatically before prediction.
- **preprocess=False**: Expects pre-encoded data (output of
  `woe_encode` or equivalent). Useful when chaining stages
  or when preprocessing has already been applied externally.
"""

from ..deps.common import (
    inspect
)
from ..deps.data  import (
    pDF
)
from ..deps.types import (
    Callable
)
from ..core.parameters  import  (
    AIGFLOW, 
    Parameters
)
from ..core.base import (
    ModelBase
)
from ..core.io import  (
    ProjectPaths, 
    ModelIO
)
from ..core.foundation import (
    AigStop
)
from ..core.logging import (
    Logger
)
from ..core.standard import (
    VarStd
)
from ..utils.builtins import (
    Str
)
from ..models.models import (
    FeatureCleaner,
    FeCleanerResult,
    FeatureStabilizer,
    FeStabilizerResult,
    ObservationStabilizer,
    ObsStabilizerResult,
    NumEncoder,
    NumEncoderResult,
    CatEncoder,
    CatEncoderResult,
    IxnEncoder,
    IxnEncoderResult,
    WoeEncoder,
    WoeEncoderResult,
    WoeStabilizer,
    WoeStabilizerResult,
    DataEncoder,
    DataEncoderResult,
    Agent1Predictor,
    Agent1Result,
    Agent2Predictor,    
    Agent2Result,
    Agent3Predictor,
    Agent3Result,
    Agent4Predictor,
    Agent4Result,
    AgentPredictor,
    AgentPredictorResult
)


logger = Logger(__name__)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# CONTEXT AWARE PREDICTOR
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Predictor:
    """
    Inference Engine — Production Scoring API.
    
    The primary user-facing interface for applying trained AIDflow
    pipelines to new data. Provides access to every pipeline stage
    (cleaning, encoding, agent prediction) using the exact
    parameters locked during training.

    > `Applicable for projects with stored (save=1) models only.`
    
    #### What It Does      
    - **Loads** all trained models from a project's storage.
    - **Preprocesses** input data through the full pipeline
      (cleaning → stabilization → encoding) using trained
      standards.
    - **Predicts** with any trained agent or the champion
      production model.
    - **Exposes** intermediate stages for diagnostic or
      custom-chain use.

    #### Pipeline Stages Available
    1. **Data Quality**: `fe_clean` — type/quality/relevance
       validation against trained standards.
    2. **Stability**: `fe_stabilize`, `obs_stabilize` —
       time-stability filtering and outlier labeling.
    3. **Encoding**: `num_encode`, `cat_encode`,
       `ixn_encode`, `woe_encode`, `woe_stabilize`, 
       `data_encode` — full feature transformation pipeline.
    4. **Prediction**: `agent1_predict` … `agent4_predict`,
       `champion_predict` — model inference.

    #### Two Usage Modes
    - **preprocess=True** (default): Pass original input columns.
      The full preprocessing pipeline is applied automatically.
    - **preprocess=False**: Pass pre-encoded data (e.g., WOE
      features). Skips preprocessing — useful for chaining
      stages or when encoding is applied externally.

    Parameters
    ----------
    project : str | Parameters
        Base training project. Accepts a project name (str) or
        a `Parameters` object. Defines which trained models
        are loaded for inference.
    path : str, optional
        Alternative project setup path for comparison. Must
        reside within the base project's `2_Output` folder.

    Notes
    -----
    **Training-Serving Symmetry**: 
    - Every transformation uses the parameters locked during training 
    (binning thresholds, imputation values, encoding maps, stability filters). 
    - No statistics are re-derived at inference time.

    **Chainable Stages**: 
    - Individual stages (`fe_clean`, `woe_encode`, etc.) can be called 
    independently to inspect intermediate results or build custom pipelines.

    **Champion vs. Agents**: 
    - The `champion_agent_predict` method applies the production model 
    (selected during the Selection stage). 
    - Individual agent predictions (`agent1_predict` … `agent4_predict`) 
    are available for comparison or monitoring.

    Examples
    --------
    ```python
    from aidflow.predictor import Predictor
    pred = Predictor(project='my_project')
    
    # Full pipeline prediction (raw data in, probability out)
    result = pred.champion_agent_predict(new_raw_data, preprocess=True)
    result.data  # P(class=1) probabilities
    
    # Individual agent for comparison
    a1 = pred.agent1_predict(new_raw_data, preprocess=True)
    a2 = pred.agent2_predict(new_raw_data, preprocess=True)
    ```
    """
    def _mapping(self) -> dict[str, Callable]:
        return {
            AIGFLOW.Schema.features:            self.get_schema,
            AIGFLOW.Data.fe_cleaner:            self.fe_clean,
            AIGFLOW.Data.fe_stabilizer:         self.fe_stabilize,
            AIGFLOW.Data.obs_stabilizer:        self.obs_stabilize,
            AIGFLOW.Encoding.num_encoder:       self.num_encode,
            AIGFLOW.Encoding.cat_encoder:       self.cat_encode,
            AIGFLOW.Encoding.ixn_encoder:       self.ixn_encode,
            AIGFLOW.Encoding.woe_encoder:       self.woe_encode,
            AIGFLOW.Encoding.woe_stabilizer:    self.woe_stabilize,
            AIGFLOW.Encoding.data_encoder:      self.data_encode,
            AIGFLOW.Training.agent1:            self.agent1_predict,
            AIGFLOW.Training.agent2:            self.agent2_predict,
            AIGFLOW.Training.agent3:            self.agent3_predict,
            AIGFLOW.Training.agent4:            self.agent4_predict,
            AIGFLOW.Selection.champion_agent:   self.champion_predict
        }
    
    def __init__(self, project: str | Parameters, path: str = None):
        e = (
            'AIG SAFETY ► Valid project and optionally path has to be '
            'provided of trained and stored Models for prediction.'
        )
        if isinstance(project, Parameters):
            setup = project
        else: 
            try:
                p = ProjectPaths(project, path, err=e)            
                setup = p.get_setup()
            except:
                PredictorRepr.error(e)

        self.setup    = setup
        self.project  = setup.project
        self.model_io = ModelIO(setup)
        self.mapping  = self._mapping()       
    
    def __repr__(self):
        return PredictorRepr.repr(self.mapping)
    
    def get_predictor(self, model:str) -> Callable:
        """
        Return the predict method for the specified model stage.

        Parameters
        ----------
        model : str
            AIGFLOW model identifier (e.g.,
            `AIGFLOW.Training.agent1`).

        Returns
        -------
        Callable
            The bound predict method for that stage.
        """
        return self.mapping[model]
    
    def get_model(self, model:str) -> ModelBase: 
        """
        Load and return the trained model for the specified stage.

        Parameters
        ----------
        model : str
            AIGFLOW model identifier.

        Returns
        -------
        ModelBase
            The fitted model instance (with `.transform`,
            `.cols`, and `.stats` available).
        """
        try:
            return self.model_io.load(model)
        except:
            PredictorRepr.error(
                f'AIG SAFETY ► Requested {model} has not been found under '
                f'project "{self.project}" ► Ensure that project has been '
                'trained and stored (save=1)'
            )
    
    def get_schema(self) -> dict:
        """
        Return the full feature lineage from the training pipeline.

        Provides all original variables and the created features
        at each training pipeline stage — enabling complete
        auditability of how raw inputs map to final model
        features.

        Returns
        -------
        dict
            Feature lineage by flow step.
        """
        return self.get_model(AIGFLOW.Schema.features)
        
    def fe_clean(self, data:pDF, subset:list=None) -> FeCleanerResult:
        """
        Validate and clean input data against trained standards.

        Applies the trained data-quality pipeline: type
        validation, outlier detection, and relevance filtering.
        Variables that do not match the trained behavioral
        profile are flagged (and optionally removed).

        Parameters
        ----------
        data : pDF
            Input data of shape (n_samples, n_features).
        subset : list, optional
            Specific features to process and return. 
            Default = None: all trained features.   

        Returns
        -------
        FeCleanerResult
        """
        model:FeatureCleaner = self.get_model(AIGFLOW.Data.fe_cleaner)
        return model.transform(data, subset=subset)
    
    def fe_stabilize(self, data:pDF, preprocess:bool=False) -> FeStabilizerResult:
        """
        Apply trained time-stability filtering to input data.

        Removes variables that were detected as unstable
        (time-variant distributions) during training. Ensures
        only temporally stable features propagate downstream.

        Parameters
        ----------
        data : pDF
            Input data of shape (n_samples, n_features).
        preprocess : bool, default=False
            If True, accepts original columns and applies
            `fe_clean` first. If False, expects output
            from `fe_clean`.

        Returns
        -------
        FeStabilizerResult
        """
        if preprocess:
            data = self.fe_clean(data).data
        model:FeatureStabilizer = self.get_model(AIGFLOW.Data.fe_stabilizer)
        return model.transform(data)
            
    def obs_stabilize(self, data:pDF, preprocess:bool=False) -> ObsStabilizerResult:
        """
        Apply trained outlier detection and provide OUTLIER labels.

        Uses the outlier model trained during the pipeline to
        flag anomalous observations in new data.

        Parameters
        ----------
        data : pDF
            Input data of shape (n_samples, n_features).
        preprocess : bool, default=False
            If True, accepts original columns and applies
            `fe_clean` first. If False, expects output 
            from fe_clean / `fe_stabilize`.

        Returns
        -------
        ObsStabilizerResult
        """           
        if preprocess: 
            data = self.fe_clean(data).data           
        model:ObservationStabilizer = self.get_model(AIGFLOW.Data.obs_stabilizer)
        return model.transform(data)

    def num_encode(self, data:pDF|None=None, subset:list=None) -> NumEncoderResult | dict:
        """
        Apply trained numerical binning to input data.

        Parameters
        ----------
        data : pDF | None
            Input data. If None, returns the trained encoding
            map (bin boundaries per variable).
        subset : list, optional
            Specific features to process and return. 
            Default = None: all trained features.            

        Returns
        -------
        NumEncoderResult | dict
            - data provided: data of encoded numerical features
            - data is None: trained bin boundaries
        """        
        model:NumEncoder = self.get_model(AIGFLOW.Encoding.num_encoder)
        if data is None:
            return model.cols.encoding
        return model.transform(data, subset = subset)

    def cat_encode(self, data:pDF|None=None, subset:list=None) -> CatEncoderResult | dict:
        """
        Apply trained categorical encoding to input data.

        Parameters
        ----------
        data : pDF | None
            Input data. If None, returns the trained encoding
            map (category → code per variable).
        subset : list, optional
            Specific features to process and return. 
            Default = None: all trained features.            

        Returns
        -------
        CatEncoderResult | dict
            - data provided: data of encoded categorical features
            - data is None: trained category mappings
        """
        model:CatEncoder = self.get_model(AIGFLOW.Encoding.cat_encoder)
        if data is None:
            return model.cols.encoding
        return model.transform(data, subset = subset)
    
    def ixn_encode(self, data:pDF|None=None, subset:list=None) -> IxnEncoderResult | dict:
        """
        Apply trained interaction encoding to input data.

        Parameters
        ----------
        data : pDF | None
            Input data. If None, returns the trained interaction
            specification.
        subset : list, optional
            Specific interactions to produce. 
            Default = None: all trained are produced.

        Returns
        -------
        IxnEncoderResult | dict
            - data provided: data of interaction features
            - data is None: trained interaction spec
        """
        model:IxnEncoder = self.get_model(AIGFLOW.Encoding.ixn_encoder)        
        if data is None:
            return model.cols.encoding
        return model.transform(data, subset = subset)    
    
    def woe_encode(self, data:pDF|None=None, subset:list=None) -> WoeEncoderResult | dict:
        """
        Apply trained WOE encoding to input data.

        Converts binned features into Weight of Evidence values
        using the bin-level target rates locked during training.

        Parameters
        ----------
        data : pDF | None
            Input data (binned features). If None, returns the
            trained WOE encoding map.
        subset : list, optional
            Specific features to process and return (including IXN). 
            Default = None: all trained features.       

        Returns
        -------
        WoeEncoderResult | dict
            - data provided: data of WOE-encoded features
            - data is None: trained WOE values per bin
        """ 
        model:WoeEncoder = self.get_model(AIGFLOW.Encoding.woe_encoder)
        if data is None:
            return model.cols.encoding
        return model.transform(data, subset = subset)

    def woe_stabilize(self, data:pDF=None, encode:bool=False) -> WoeStabilizerResult | list:
        """
        Apply trained WOE stability filtering.

        Removes WOE features that were flagged as unstable
        (distribution shift) during training.

        Parameters
        ----------
        data : pDF | None
            WOE-encoded data. If None, returns the list of
            stable WOE features.
        encode : bool, default=False
            If True, applies `woe_encode` first (expects
            binned data). If False, expects WOE-encoded input.

        Returns
        -------
        WoeStabilizerResult | list
            - data provided: data subset to stable WOE features only
            - data is None: list of stable WOE feature names
        """      
        model:WoeStabilizer = self.get_model(AIGFLOW.Encoding.woe_stabilizer)
        if data is None:
            return model.cols.vars_woe
        if encode:
            data = self.woe_encode(data).data
        return model.transform(data)    

    def data_encode(self, data:pDF=None, test_size:float=0.0) -> DataEncoderResult:
        """
        Apply full encoding pipeline to produce ML-ready data.

        Executes dtypes standardization (if mismatch), outlier 
        removal (optional), WOE encoding, and train/test split.

        Parameters
        ----------
        data : pDF | None
            Input data - both raw and preprocessed supported.
        test_size : float, default=0.0
            Fraction of data allocated to the test set (used only when
            no explicit test set is provided).
            - Special case for 0 - returns None for test_set.

        Returns
        -------
        DataEncoderResult
        """         
        model:DataEncoder = self.get_model(AIGFLOW.Encoding.data_encoder)
        return model.transform(data, test_size=test_size)
    
    def _woe_data_agent(self, data:pDF, selected:list, raw:list) -> pDF:
        """
        Prepare WOE-encoded data for agent prediction.

        Applies the full preprocessing chain (clean + encode)
        restricted to the agent's selected features.

        Parameters
        ----------
        data : pDF
            Raw input data.
        selected : list
            Agent's selected WOE features.
        raw : list
            Original (pre-WOE) feature names for cleaning.

        Returns
        -------
        pDF
            WOE-encoded data containing only the agent's
            selected features.
        """        
        fes  = VarStd.fe_to_var(selected)
        data = self.fe_clean(data, subset = raw).data
        return self.woe_encode(data, subset = fes).data

    def agent1_predict(self, data:pDF, preprocess:bool=False) -> Agent1Result:
        """
        Predict with Agent 1 on input data.

        Parameters
        ----------
        data : pDF
            Input data.
        preprocess : bool, default=False
            If True, applies the full preprocessing pipeline
            (clean → encode) before prediction. If False,
            expects WOE-encoded data.

        Returns
        -------
        Agent1Result
            P(class=1) + feature lineage metadata.
        """
        model:Agent1Predictor = self.get_model(AIGFLOW.Training.agent1)
        if preprocess:
            data = self._woe_data_agent(data, model.cols.selected, model.cols.raw)
        return model.transform(data)

    def agent2_predict(self, data:pDF, preprocess:bool=False) -> Agent2Result:
        """
        Predict with Agent 2 on input data.

        Parameters
        ----------
        data : pDF
            Input data.
        preprocess : bool, default=False
            If True, applies the full preprocessing pipeline
            (clean → encode) before prediction. If False,
            expects WOE-encoded data.

        Returns
        -------
        Agent2Result
            P(class=1) + feature lineage metadata.
        """
        model:Agent2Predictor = self.get_model(AIGFLOW.Training.agent2)
        if preprocess:
            data = self._woe_data_agent(data, model.cols.selected, model.cols.raw)        
        return model.transform(data)

    def agent3_predict(self, data:pDF, preprocess:bool=False) -> Agent3Result:
        """
        Predict with Agent 3 on input data.

        Parameters
        ----------
        data : pDF
            Input data.
        preprocess : bool, default=False
            If True, applies the full preprocessing pipeline
            (clean → encode) before prediction. If False,
            expects WOE-encoded data.

        Returns
        -------
        Agent3Result
            P(class=1) + feature lineage metadata.
        """
        model:Agent3Predictor = self.get_model(AIGFLOW.Training.agent3)
        if preprocess:
            data = self._woe_data_agent(data, model.cols.selected, model.cols.raw)       
        return model.transform(data)

    def agent4_predict(self, data:pDF, preprocess:bool=False) -> Agent4Result:
        """
        Predict with Agent 4 (GBM) on input data.

        Parameters
        ----------
        data : pDF
            Input data.
        preprocess : bool, default=False
            If True, applies `fe_clean` before prediction
            (GBM agents do not require WOE encoding). If False,
            expects cleaned data.

        Returns
        -------
        Agent4Result
            P(class=1) + feature lineage metadata.
        """
        model:Agent4Predictor = self.get_model(AIGFLOW.Training.agent4)
        if preprocess:
            data = self.fe_clean(data, subset = model.cols.raw).data
        return model.transform(data)

    def champion_predict(self, data:pDF, preprocess:bool=False) -> AgentPredictorResult:
        """
        Predict with the Champion Production Agent on input data.

        The champion is the final production model selected
        during the Selection stage. This is the primary
        inference method for production scoring.

        Parameters
        ----------
        data : pDF
            Input data.
        preprocess : bool, default=False
            If True, applies the champion's full preprocessing
            pipeline before prediction. If False, expects
            pre-processed data.

        Returns
        -------
        AgentPredictorResult
            P(class=1) + production model metadata.
        """
        model:AgentPredictor = self.get_model(AIGFLOW.Selection.champion_agent)
        if preprocess:
            data = model.preprocess(data)
        return model.transform(data)     
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PredictorRepr:
    """Predictor formatted overview of all methods."""

    @staticmethod
    def error(msg:str):
        logger.error(msg)
        AigStop.custom(msg)
        
    @classmethod
    def repr(cls, mapping: dict[str, Callable]) -> str:
        al = Str.fx_align(mapping)
        return (
            'Worker → corresponding Predictor method:'
            '\n- '+'\n- '.join([
                f'{m+al(m)} → {f.__name__}{cls._sig(f)}'
                for m, f in mapping.items()
            ])
        )

    @staticmethod
    def _sig(x: Callable) -> str:
        sig = str(inspect.signature(x)).replace('pandas.','')
        a,u = '-> ', '|'
        fx, res = sig.split(a)
        res = res.split(u)
        cl = ''
        for i, r in enumerate(res):
            cl += u if i > 0 else ''
            cl += r.split('.')[-1]
        return fx + a + cl
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬