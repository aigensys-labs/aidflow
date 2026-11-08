"""
Domain Interface and Phase Orchestration Layer.

This module serves as the primary API surface for end-users, providing a 
semantic interface to the underlying AIDflow worker ecosystem. It transforms 
the generic execution capabilities of the `core/worker` layer into domain-specific 
pipeline phases (e.g., DataPhase, EncodingPhase), allowing users to interact 
with the ML pipeline using meaningful business logic rather than technical 
configurations.

Key architectural roles provided by this module:
1. Worker Registration: Acts as the local namespace where specific model implementations 
are mapped to unique worker identifiers (e.g., `AIGFLOW.Data.data_loader`), enabling 
a pluggable and extensible set of tools for each pipeline phase.

2. State Lineage Management: Explicitly manages the flow of data between workers 
   by defining how results from one step (e.g., `train.raw`) are linked as inputs 
   to the next (e.g., `train.clean`), ensuring a strict and traceable data lineage.
   
3. Input/Output Schema Enforcement: Integrates specific Input and Result dataclasses 
   for every worker, providing the user with IDE-assisted type hinting and 
   guaranteeing that every pipeline step adheres to its required data contract.
"""

from ..core.parameters          import  (
                                        PRM, 
                                        AIGFLOW
                                        )
from ..core.base.worker         import  (
                                        WorkerConfig,
                                        WorkerContract,
                                        DatasetBinding,
                                        WorkerBase,
                                        WorkerPhaseBase,
                                        )
from ..core.factory             import  Dataclass
from ..core.protocol            import  ProjectContextProtocol

from .load                      import  (
                                        DataLoaderInput, 
                                        DataLoaderResult,
                                        DataLoaderUI
                                        )
from .screen                    import  (
                                        FeCleanerInput,
                                        FeCleanerResultDual,
                                        FeatureCleanerUI,
                                        FeStabilizerInput,
                                        FeStabilizerResult,
                                        FeatureStabilizerUI,
                                        ObsStabilizerInput,
                                        ObsStabilizerResult,
                                        ObservationStabilizerUI
                                        )
from .encode                    import  (
                                        NumEncoderInput,
                                        NumEncoderResult,
                                        NumEncoderUI, 
                                        CatEncoderInput,
                                        CatEncoderResult,
                                        CatEncoderUI,
                                        IxnEncoderInput,
                                        IxnEncoderResult,
                                        IxnEncoderUI,
                                        WoeEncoderEnvelopeCfg,
                                        WoeEncoderEnvelope,
                                        WoeEncoderInput,
                                        WoeEncoderResult,
                                        WoeEncoderUI,
                                        WoeStabilizerEnvelopeCfg,
                                        WoeStabilizerEnvelope,
                                        WoeStabilizerInput,
                                        WoeStabilizerResult,
                                        WoeStabilizerUI
                                        )
from .transform                 import  (
                                        DataEncoderEnvelopeCfg,
                                        DataEncoderEnvelope,
                                        DataEncoderInput, 
                                        DataEncoderResult,
                                        DataEncoder
                                        )


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA PHASE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class DataLoaderWorker(WorkerBase[DataLoaderResult]):
    """
    Step 0: Data Ingestion.
    
    Initializes the pipeline by importing raw data and initializing the 
    global dataset context. It establishes the baseline 'raw' state for 
    all subsequent workers.

    Parameters
    ----------
    inputs : DataLoaderInput, optional
        Custom input configuration (source paths, etc.). If None, uses defaults.
    **kwargs : dict
        Additional parameters passed to the DataLoader worker.
    
    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.data_loader.spec`.

    Returns
    -------
    DataLoaderWorker
        The populated instance of DataLoaderWorker.        
        

    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.data_loader
    - inputs    → ctx.workers.data_loader.spec
    - result    → ctx.workers.data_loader.result
    - state     → ctx.workers.data_loader.state
    - report    → ctx.workers.data_loader.report
    - stats     → ctx.workers.data_loader.stats
    - fitter    → ctx.workers.data_loader.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - data      → ctx.datasets.input
    - data      → ctx.datasets.train.raw
    - data      → ctx.datasets.test.raw
    - dtypes    → ctx.datasets.dtypes.raw
    - stats     → ctx.datasets.stats.raw
    

    Examples
    --------
    Defining the Project Context (Control Plane) & Project Scope:
    
    ```python
    import aidflow as aid
    
    ctx = aid.ProjectContext()
    
    ctx.genesis(
        project      = 'test',
        input_data   = 'test.csv',
        target       = 'target',
        yy_mm        = 'decision_date',
        yy_mm_format = '%Y-%m-%d'
    )
    ctx.params()
    ```
    
    ### Worker → Model → Fitter
    
    Accessing worker's inputs schema → parameters & types (*shown in the display output*):

    ```python        
    ctx.workers.data_loader.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.data_loader.fitters
    ``` 

    Executing worker with defaults → auto provided by the context:
    
    ```python        
    ctx.workers.data_loader()
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.data_loader
    inputs   = worker.spec.inputs
    result   = worker.result
    state    = worker.state
    features = worker.features
    params   = worker.params
    stats    = worker.stats
    fitter   = worker.fitter  # jump to chosen internal fitter
    ...
    ```

    View to model report of executed worker:

    ```python
    ctx.workers.data_loader.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.data_loader.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter  = ctx.workers.data_loader.fitter
    params_ = fitter.params
    stats_  = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.data_loader.fitters[0]
    prm = dict(
        flow_vars = ['target1', 'target2'],
        columns_subset = ['fe1', 'fe2']
    )
    data_loader = ctx.workers.data_loader(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various targets might be under testing\
    (*note that 'target2' must be present in the data*):

    ```python
    cfg = dict(
        target = 'target2',
    )
    data_loader = ctx.workers.data_loader(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Data.loader,
        Model   = DataLoaderUI,
        Input   = DataLoaderInput,
        Result  = DataLoaderResult,
        title   = 'DATA LOAD'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(
            defaults = DataLoaderInput(),
            datasets = DatasetBinding(fitted = {
                'input'       : self._result.input_data, 
                'dtypes.raw'  : self._result.dtypes,     
                'train.raw'   : self._result.train_data, 
                'test.raw'    : self._result.test_data, 
                'stats.raw'   : self._processor.stats
            })
        )   
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeatureCleanerWorker(WorkerBase[FeCleanerResultDual]):
    """
    Step 1: Feature Quality Screening.
    
    Enforces data standards by cleaning raw datasets and identifying valid 
    numerical and categorical features. This step creates the 'clean' lineage.

    Parameters
    ----------
    inputs : FeCleanerInput, optional
        Custom cleaning inputs configuration. If None, automatically resolves\
        'train.raw', 'test.raw', and 'dtypes.raw' from preceding step.
    **kwargs : dict
        Additional parameters passed to the FeatureCleaner worker.
    
    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.fe_cleaner.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for fe_cleaner training and feature cleaning
    - test_set  != None → custom test data might be provided to apply dtypes by trained model
    - dtypes    != None → custom dtypes of all variables in train_set to apply on train and test

    Returns
    -------
    FeatureCleanerWorker
        The populated instance of FeatureCleanerWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.fe_cleaner
    - inputs    → ctx.workers.fe_cleaner.spec
    - result    → ctx.workers.fe_cleaner.result
    - state     → ctx.workers.fe_cleaner.state
    - report    → ctx.workers.fe_cleaner.report
    - stats     → ctx.workers.fe_cleaner.stats
    - fitter    → ctx.workers.fe_cleaner.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - data      → ctx.datasets.train.clean
    - data      → ctx.datasets.test.clean
    - dtypes    → ctx.datasets.dtypes.clean
    - dropped   → ctx.datasets.dtypes.dropped
    - features  → ctx.datasets.features.all.clean
    - features  → ctx.datasets.features.num.clean
    - features  → ctx.datasets.features.cat.clean
    - stats     → ctx.datasets.stats.clean


    Examples
    --------
    Defining the Project Context (Control Plane) & Project Scope:
    
    ```python
    import aidflow as aid
    
    ctx = aid.ProjectContext()
    
    ctx.genesis(
        project      = 'test',
        input_data   = 'test.csv',
        target       = 'target',
        yy_mm        = 'decision_date',
        yy_mm_format = '%Y-%m-%d'
    )
    ctx.params()
    ```
    
    ### Worker → Model → Fitter
    
    Accessing worker's inputs schema → parameters & types (*shown in the display output*):

    ```python        
    ctx.workers.fe_cleaner.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.fe_cleaner.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.workers.data_loader()   # 0. import data to the context scope
    ctx.workers.fe_cleaner()    # 1. apply cleaning logic to the imported data (0)
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.fe_cleaner
    inputs   = worker.spec.inputs
    result   = worker.result
    state    = worker.state
    features = worker.features
    params   = worker.params
    stats    = worker.stats
    fitter   = worker.fitter  # jump to chosen internal fitter
    ...
    ```

    View to model report of executed worker:

    ```python
    ctx.workers.fe_cleaner.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.fe_cleaner.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter  = ctx.workers.fe_cleaner.fitter
    params_ = fitter.params
    stats_  = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.fe_cleaner.fitters[0]
    prm = dict(
        max_miss = 0.5,
        max_1val = 0.6,
        max_corr = 0.8,
    )
    fe_cleaner = ctx.workers.fe_cleaner(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various time variables might be under testing\
    (*note that 'time_var2' must be present in the data*):

    ```python
    cfg = dict(
        time_var     = 'time_var2',
        time_format  = '%Y%m%d',
    )
    fe_cleaner = ctx.workers.fe_cleaner(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Data.fe_cleaner,
        Model   = FeatureCleanerUI,
        Input   = FeCleanerInput,
        Result  = FeCleanerResultDual,
        offload = AIGFLOW.Data.loader,
        title   = 'DATA QUALITY'        
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = FeCleanerInput(
                train_set = self.sets.get('train.raw'),
                test_set  = self.sets.get('test.raw'),
                dtypes    = self.sets.get('dtypes.raw'),
            ),

            datasets = DatasetBinding(
                fitted = {
                    'train.clean'           : self._result.data,
                    'dtypes.clean'          : self._result.dtypes,
                    'dtypes.dropped'        : self._result.dropped,
                    'features.all.clean'    : self._result.features,
                    'features.num.clean'    : self._result.num_features,
                    'features.cat.clean'    : self._result.cat_features,
                    'stats.clean'           : self._processor.stats
                },
                direct = {'test.clean' : self._processor.model_result.data}
            ),

            verify = {'train': 'train.clean', 'test': 'test.clean'}
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FeatureStabilizerWorker(WorkerBase[FeStabilizerResult]):
    """
    Step 2: Feature Temporal Stability.
        
    Analyzes features for temporal stationarity to prevent data leakage and 
    overfitting. It filters the 'clean' feature set down to a 'stable' set.

    Parameters
    ----------
    inputs : FeStabilizerInput, optional
        Custom stability configuration. If None, resolves from '.clean' state.
    is_target : bool, optional
        If True, performs stability check on the target variable instead\
        of features (skips registration of result datasets). Defaults to False.
    **kwargs : dict
        Additional parameters passed to the FeatureStabilizer worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.fe_stabilizer.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for fe_stabilizer training and feature temporal cleaning
    - test_set  != None → custom test data might be provided to obtain temporal view and stats
    - num_features != None → optional subset of numerical features
    - cat_features != None → optional subset of categorical features

    Returns
    -------
    FeatureStabilizerWorker
        The populated instance of FeatureStabilizerWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.fe_stabilizer
    - inputs    → ctx.workers.fe_stabilizer.spec
    - result    → ctx.workers.fe_stabilizer.result
    - state     → ctx.workers.fe_stabilizer.state
    - report    → ctx.workers.fe_stabilizer.report
    - stats     → ctx.workers.fe_stabilizer.stats
    - fitter    → ctx.workers.fe_stabilizer.fitter
    
    Registers
    ---------
    Registers data into the datasets context:    
    - features  → ctx.datasets.features.all.stable.dist
    - features  → ctx.datasets.features.num.stable.dist
    - features  → ctx.datasets.features.cat.stable.dist
    - dropped   → ctx.datasets.features.all.dropped.stable.dist
    - dropped   → ctx.datasets.features.num.dropped.stable.dist
    - dropped   → ctx.datasets.features.cat.dropped.stable.dist
    - stats     → ctx.datasets.stats.stable.features

    Examples
    --------
    Defining the Project Context (Control Plane) & Project Scope:
    
    ```python
    import aidflow as aid
    
    ctx = aid.ProjectContext()
    
    ctx.genesis(
        project      = 'test',
        input_data   = 'test.csv',
        target       = 'target',
        yy_mm        = 'decision_date',
        yy_mm_format = '%Y-%m-%d'
    )
    ctx.params()
    ```
    
    ### Worker → Model → Fitter
    
    Accessing worker's inputs schema → parameters & types (*shown in the display output*):

    ```python        
    ctx.workers.fe_stabilizer.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.fe_stabilizer.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.workers.data_loader()    # 0. import data to the context scope
    ctx.workers.fe_cleaner()     # 1. apply cleaning logic to the imported data (0)
    ctx.workers.fe_stabilizer()  # 2. apply temporal stability logic to the cleaned data (1)
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.fe_stabilizer
    inputs   = worker.spec.inputs
    result   = worker.result
    state    = worker.state
    features = worker.features
    params   = worker.params
    stats    = worker.stats
    fitter   = worker.fitter  # jump to chosen internal fitter
    ...
    ```

    View to model report of executed worker:

    ```python
    ctx.workers.fe_stabilizer.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.fe_stabilizer.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter  = ctx.workers.fe_stabilizer.fitter
    params_ = fitter.params
    stats_  = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.fe_stabilizer.fitters[0]
    prm = dict(
        th_stationarity = 0.2
    )
    fe_stabilizer = ctx.workers.fe_stabilizer(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various time variables might be under testing\
    (*note that 'time_var2' must be present in the data*):

    ```python
    cfg = dict(
        time_var = 'time_var2',
    )
    fe_stabilizer = ctx.workers.fe_stabilizer(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Data.fe_stabilizer,
        Model   = FeatureStabilizerUI,
        Input   = FeStabilizerInput,
        Result  = FeStabilizerResult,
        title   = 'TEMPORAL STABILITY'
    )

    def _get_contract(self, is_default: bool = True):
        source = Dataclass.schema(FeCleanerResultDual)
        return WorkerContract(

            defaults = FeStabilizerInput(
                train_set    = self.sets.get('train.clean'),
                test_set     = self.sets.get('test.clean'),
                num_features = self.sets.get('features.num.clean'),
                cat_features = self.sets.get('features.cat.clean'),
            ),

            datasets = DatasetBinding(
                fitted = {
                    'features.all.stable.dist'           : self._result.features,
                    'features.all.dropped.stable.dist'   : self._result.dropped,
                    'features.num.stable.dist'           : self._result.num_features,
                    'features.cat.stable.dist'           : self._result.cat_features,
                    'stats.stable.features'              : self._processor.stats
                },
                subset = (AIGFLOW.Data.fe_cleaner, {
                    'features.num.dropped.stable.dist' : [source.num_features, self._result.dropped],
                    'features.cat.dropped.stable.dist' : [source.cat_features, self._result.dropped],
                })
            )
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ObservationStabilizerWorker(WorkerBase[ObsStabilizerResult]):
    """
    Step 3: Observation Stability (Outlier Screening).
    
    The final guardrail of the Data Phase. It identifies and removes 
    distorting observations (outliers) from the stable training set.

    Parameters
    ----------
    inputs : ObsStabilizerInput, optional
        Custom outlier configuration. If None, resolves from '.stable.dist' state.
    **kwargs : dict
        Additional parameters passed to the ObservationStabilizer worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.obs_stabilizer.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for obs_stabilizer training and outliers cleaning
    - num_features != None → optional subset of numerical features
    - cat_features != None → optional subset of categorical features

    Returns
    -------
    ObservationStabilizerWorker
        The populated instance of ObservationStabilizerWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.obs_stabilizer
    - inputs    → ctx.workers.obs_stabilizer.spec
    - result    → ctx.workers.obs_stabilizer.result
    - state     → ctx.workers.obs_stabilizer.state
    - report    → ctx.workers.obs_stabilizer.report
    - stats     → ctx.workers.obs_stabilizer.stats
    - fitter    → ctx.workers.obs_stabilizer.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - data      → ctx.datasets.train.stable
    - stats     → ctx.datasets.stats.stable.observations

    Examples
    --------
    Defining the Project Context (Control Plane) & Project Scope:
    
    ```python
    import aidflow as aid
    
    ctx = aid.ProjectContext()
    
    ctx.genesis(
        project      = 'test',
        input_data   = 'test.csv',
        target       = 'target',
        yy_mm        = 'decision_date',
        yy_mm_format = '%Y-%m-%d'
    )
    ctx.params()
    ```
    
    ### Worker → Model → Fitter
    
    Accessing worker's inputs schema → parameters & types (*shown in the display output*):

    ```python        
    ctx.workers.obs_stabilizer.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.obs_stabilizer.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.workers.data_loader()    # 0. import data to the context scope
    ctx.workers.fe_cleaner()     # 1. apply cleaning logic to the imported data (0)
    ctx.workers.fe_stabilizer()  # 2. apply temporal stability logic to the cleaned data (1)
    ctx.workers.obs_stabilizer() # 3. apply observation stability logic to the cleaned data (2)
    ```

    Note that steps 0-3 can be run by ExecutionContext one-liner:

    ```python   
    ctx.flow.process('data', verbose = 0) # steps 0 - 3 have been trained within the context scope
    ```
    
    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.obs_stabilizer
    inputs   = worker.spec.inputs
    result   = worker.result
    state    = worker.state
    features = worker.features
    params   = worker.params
    stats    = worker.stats
    fitter   = worker.fitter  # jump to chosen internal fitter
    ...
    ```

    View to model report of executed worker:

    ```python
    ctx.workers.obs_stabilizer.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.obs_stabilizer.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter  = ctx.workers.obs_stabilizer.fitter
    config_ = fitter.config
    params_ = fitter.params
    model_  = fitter.model
    stats_  = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.obs_stabilizer.fitters[0]
    prm = dict(
        th_outliers = 0.05
    )
    obs_stabilizer = ctx.workers.obs_stabilizer(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        bootstrap    = False,
        max_features = 0.7
    )
    obs_stabilizer = ctx.workers.obs_stabilizer(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Data.obs_stabilizer,
        Model   = ObservationStabilizerUI,
        Input   = ObsStabilizerInput,
        Result  = ObsStabilizerResult,
        offload = AIGFLOW.Data.fe_cleaner,
        title   = 'OBSERVATION STABILITY'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = ObsStabilizerInput(
                train_set    = self.sets.get('train.clean'),
                num_features = self.sets.get('features.num.stable.dist'),
                cat_features = self.sets.get('features.cat.stable.dist'),
            ),

            datasets = DatasetBinding(fitted = {
                'train.stable'              : self._result.data,
                'stats.stable.observations' : self._processor.stats
            }),

            verify = 'train.stable'
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
        
class DataPhase(WorkerPhaseBase):
    """
    AIG Flow Data Phase Orchestrator.
    
    The primary interface for the initial stage of the ML pipeline. It coordinates 
    the flow from raw data ingestion through cleaning and stability screening, 
    ensuring a traceable lineage of datasets (raw -> clean -> stable).
    """

    def __init__(self, ctx: ProjectContextProtocol):    
        super().__init__(ctx)
        self._register((
            DataLoaderWorker,
            FeatureCleanerWorker,
            FeatureStabilizerWorker,
            ObservationStabilizerWorker,
        ))
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ENCODING PHASE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class NumEncoderWorker(WorkerBase[NumEncoderResult]):
    """
    Step 4: Numerical Feature Discretization/Encoding.
    
    Transforms stable numerical features into a discrete encoded space 
    based on the selected fitting strategy.

    Parameters
    ----------
    inputs : NumEncoderInput, optional
        Custom encoding configuration. If None, resolves from 'train.stable' and\
        'features.num.stable.dist'.
    **kwargs : dict
        Additional parameters passed to the NumEncoder worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.num_encoder.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for num_encoder training
    - num_features != None → optional subset of numerical features

    Returns
    -------
    NumEncoderWorker
        The populated instance of NumEncoderWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.num_encoder
    - inputs    → ctx.workers.num_encoder.spec
    - result    → ctx.workers.num_encoder.result
    - state     → ctx.workers.num_encoder.state
    - report    → ctx.workers.num_encoder.report
    - stats     → ctx.workers.num_encoder.stats
    - fitter    → ctx.workers.num_encoder.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - data      → ctx.datasets.train.encoded.num
    - features  → ctx.datasets.features.num.encoded
    - features  → ctx.datasets.features.num.selected
    - mapping   → ctx.datasets.features.num.mapping    
    - dropped   → ctx.datasets.features.num.dropped.encoded
    - encodings → ctx.datasets.encodings.num
    - stats     → ctx.datasets.stats.encoded.num

    Examples
    --------
    Defining the Project Context (Control Plane) & Project Scope:
    
    ```python
    import aidflow as aid
    
    ctx = aid.ProjectContext()
    
    ctx.genesis(
        project      = 'test',
        input_data   = 'test.csv',
        target       = 'target',
        yy_mm        = 'decision_date',
        yy_mm_format = '%Y-%m-%d'
    )
    ctx.params()
    ```
    
    ### Worker → Model → Fitter
    
    Accessing worker's inputs schema → parameters & types (*shown in the display output*):

    ```python        
    ctx.workers.num_encoder.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.num_encoder.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.workers.data_loader()    # 0. import data to the context scope
    ctx.workers.fe_cleaner()     # 1. apply cleaning logic to the imported data (0)
    ctx.workers.fe_stabilizer()  # 2. apply temporal stability logic to the cleaned data (1)
    ctx.workers.obs_stabilizer() # 3. apply observation stability logic to the cleaned data (2)
    ctx.workers.num_encoder()    # 4. apply numerical encoding logic to the stable data (3)
    ```
    
    Note that steps 0-3 can be run by ExecutionContext one-liner:

    ```python   
    ctx.flow.process('data', verbose = 0) # steps 0 - 3 have been trained within the context scope
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.num_encoder
    inputs   = worker.spec.inputs
    result   = worker.result
    state    = worker.state
    features = worker.features
    params   = worker.params
    stats    = worker.stats
    fitter   = worker.fitter  # jump to chosen internal fitter
    ...
    ```

    View to model report of executed worker:

    ```python
    ctx.workers.num_encoder.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.num_encoder.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.num_encoder.fitter
    result_   = fitter.result
    features_ = fitter.features
    params_   = fitter.params
    stats_    = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.num_encoder.fitters[0]
    prm = dict(
        max_num_bins  = 20,
        min_bin_share = 0.05,
        gini_cutoff   = 0.01 
    )
    num_encoder = ctx.workers.num_encoder(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target   = 'target2',
        discrete = 15
    )
    num_encoder = ctx.workers.num_encoder(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Encoding.num_encoder,
        Model   = NumEncoderUI,
        Input   = NumEncoderInput,
        Result  = NumEncoderResult,
        title   = 'NUMERICAL DISCRETIZATION'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = NumEncoderInput(
                train_set    = self.sets.get('train.stable'),
                num_features = self.sets.get('features.num.stable.dist')
            ),

            datasets = DatasetBinding(fitted = {
                'train.encoded.num'             : self._result.data,
                'features.num.encoded'          : self._result.features,
                'features.num.selected'         : self._result.selected,
                'features.num.dropped.encoded'  : self._result.dropped,
                'features.num.mapping'          : self._result.fe_map,
                'encodings.num'                 : self._result.encoding,
                'stats.encoded.num'             : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class CatEncoderWorker(WorkerBase[CatEncoderResult]):
    """
    Step 5: Categorical Feature Encoding.
    
    Applies encoding strategies to transform object/categorical features into 
    categorical representations suitable for ML models.

    Parameters
    ----------
    inputs : CatEncoderInput, optional
        Custom encoding configuration. If None, resolves from 'train.stable'\
        and 'features.cat.stable.dist'.
    **kwargs : dict
        Additional parameters passed to the CatEncoder worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.cat_encoder.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for cat_encoder training
    - cat_features != None → optional subset of categorical features

    Returns
    -------
    CatEncoderWorker
        The populated instance of CatEncoderWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.cat_encoder
    - inputs    → ctx.workers.cat_encoder.spec
    - result    → ctx.workers.cat_encoder.result
    - state     → ctx.workers.cat_encoder.state
    - report    → ctx.workers.cat_encoder.report
    - stats     → ctx.workers.cat_encoder.stats
    - fitter    → ctx.workers.cat_encoder.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - data      → ctx.datasets.train.encoded.num
    - features  → ctx.datasets.features.cat.encoded
    - features  → ctx.datasets.features.cat.selected
    - mapping   → ctx.datasets.features.cat.mapping    
    - dropped   → ctx.datasets.features.cat.dropped.encoded
    - encodings → ctx.datasets.encodings.cat
    - stats     → ctx.datasets.stats.encoded.cat

    Examples
    --------
    Defining the Project Context (Control Plane) & Project Scope:
    
    ```python
    import aidflow as aid
    
    ctx = aid.ProjectContext()
    
    ctx.genesis(
        project      = 'test',
        input_data   = 'test.csv',
        target       = 'target',
        yy_mm        = 'decision_date',
        yy_mm_format = '%Y-%m-%d'
    )
    ctx.params()
    ```
    
    ### Worker → Model → Fitter
    
    Accessing worker's inputs schema → parameters & types (*shown in the display output*):

    ```python        
    ctx.workers.cat_encoder.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.cat_encoder.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.workers.data_loader()    # 0. import data to the context scope
    ctx.workers.fe_cleaner()     # 1. apply cleaning logic to the imported data (0)
    ctx.workers.fe_stabilizer()  # 2. apply temporal stability logic to the cleaned data (1)
    ctx.workers.obs_stabilizer() # 3. apply observation stability logic to the cleaned data (2)
    ctx.workers.cat_encoder()    # 4. apply categorical encoding logic to the stable data (3)
    ```
    
    Note that steps 0-3 can be run by ExecutionContext one-liner:

    ```python   
    ctx.flow.process('data', verbose = 0) # steps 0 - 3 have been trained within the context scope
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.cat_encoder
    inputs   = worker.spec.inputs
    result   = worker.result
    state    = worker.state
    features = worker.features
    params   = worker.params
    stats    = worker.stats
    fitter   = worker.fitter  # jump to chosen internal fitter
    ...
    ```

    View to model report of executed worker:

    ```python
    ctx.workers.cat_encoder.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.cat_encoder.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.cat_encoder.fitter
    result_   = fitter.result
    features_ = fitter.features
    params_   = fitter.params
    stats_    = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.cat_encoder.fitters[0]
    prm = dict(
        min_bin_share = 0.05,
        gini_cutoff   = 0.01 
    )
    cat_encoder = ctx.workers.cat_encoder(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target   = 'target2',
    )
    cat_encoder = ctx.workers.cat_encoder(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Encoding.cat_encoder,
        Model   = CatEncoderUI,
        Input   = CatEncoderInput,
        Result  = CatEncoderResult,
        title   = 'CATEGORICAL ENCODING'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = CatEncoderInput(
                train_set    = self.sets.get('train.stable'),
                cat_features = self.sets.get('features.cat.stable.dist')
            ),
            
            datasets = DatasetBinding(fitted = {
                'train.encoded.cat'             : self._result.data,
                'features.cat.encoded'          : self._result.features,
                'features.cat.selected'         : self._result.selected,
                'features.cat.dropped.encoded'  : self._result.dropped,
                'features.cat.mapping'          : self._result.fe_map,
                'encodings.cat'                 : self._result.encoding,
                'stats.encoded.cat'             : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class IxnEncoderWorker(WorkerBase[IxnEncoderResult]):
    """
    Step 6: Interaction Feature Engineering.
    
    Generates new interaction variables by combining stable numerical 
    and categorical features to capture complex data relationships.

    Parameters
    ----------
    inputs : IxnEncoderInput, optional
        Custom interaction configuration. If None, resolves from 'train.stable'\
        and stable feature sets.
    **kwargs : dict
        Additional parameters passed to the IxnEncoder worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.ixn_encoder.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for ixn_encoder training
    - num_features != None → optional subset of numerical features
    - cat_features != None → optional subset of categorical features

    Returns
    -------
    IxnEncoderWorker
        The populated instance of IxnEncoderWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.ixn_encoder
    - inputs    → ctx.workers.ixn_encoder.spec
    - result    → ctx.workers.ixn_encoder.result
    - state     → ctx.workers.ixn_encoder.state
    - report    → ctx.workers.ixn_encoder.report
    - stats     → ctx.workers.ixn_encoder.stats
    - fitter    → ctx.workers.ixn_encoder.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - data      → ctx.datasets.train.encoded.num
    - features  → ctx.datasets.features.ixn.encoded
    - features  → ctx.datasets.features.ixn.selected
    - mapping   → ctx.datasets.features.ixn.mapping    
    - dropped   → ctx.datasets.features.ixn.dropped.encoded
    - encodings → ctx.datasets.encodings.ixn
    - stats     → ctx.datasets.stats.encoded.ixn

    Examples
    --------
    Defining the Project Context (Control Plane) & Project Scope:
    
    ```python
    import aidflow as aid
    
    ctx = aid.ProjectContext()
    
    ctx.genesis(
        project      = 'test',
        input_data   = 'test.csv',
        target       = 'target',
        yy_mm        = 'decision_date',
        yy_mm_format = '%Y-%m-%d'
    )
    ctx.params()
    ```
    
    ### Worker → Model → Fitter
    
    Accessing worker's inputs schema → parameters & types (*shown in the display output*):

    ```python        
    ctx.workers.ixn_encoder.spec
    ```    

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.ixn_encoder.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.workers.data_loader()    # 0. import data to the context scope
    ctx.workers.fe_cleaner()     # 1. apply cleaning logic to the imported data (0)
    ctx.workers.fe_stabilizer()  # 2. apply temporal stability logic to the cleaned data (1)
    ctx.workers.obs_stabilizer() # 3. apply observation stability logic to the cleaned data (2)
    ctx.workers.ixn_encoder()    # 4. apply interactions encoding logic to the stable data (3)
    ```
    
    Note that steps 0-3 can be run by ExecutionContext one-liner:

    ```python   
    ctx.flow.process('data', verbose = 0) # steps 0 - 3 have been trained within the context scope
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.ixn_encoder
    inputs   = worker.spec.inputs
    result   = worker.result
    state    = worker.state
    features = worker.features
    params   = worker.params
    stats    = worker.stats
    fitter   = worker.fitter  # jump to chosen internal fitter
    ...
    ```

    View to model report of executed worker:

    ```python
    ctx.workers.ixn_encoder.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.ixn_encoder.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.ixn_encoder.fitter
    result_   = fitter.result
    features_ = fitter.features
    params_   = fitter.params
    stats_    = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.ixn_encoder.fitters[0]
    prm = dict(
        gb_max_leaves       = 31
        gb_estimators       = 100
        gb_learning_rate    = 0.1
        gb_min_child_weight = 0.1
        ixn_min_target      = 0.15
    )
    ixn_encoder = ctx.workers.ixn_encoder(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target   = 'target2',
        val_size = 0.3
    )
    ixn_encoder = ctx.workers.ixn_encoder(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Encoding.ixn_encoder,
        choice  = PRM.ENCODE.ixn_strategy.param,
        Model   = IxnEncoderUI,
        Input   = IxnEncoderInput,
        Result  = IxnEncoderResult,
        title   = 'INTERACTIONS ENCODING'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = IxnEncoderInput(
                train_set    = self.sets.get('train.stable'),
                num_features = self.sets.get('features.num.stable.dist'),
                cat_features = self.sets.get('features.cat.stable.dist')
            ),

            datasets = DatasetBinding(fitted = {
                'train.encoded.ixn'             : self._result.data,
                'features.ixn.stable.dist'      : self._result.features,    # keeping access ergonomy
                'features.ixn.encoded'          : self._result.features,
                'features.ixn.selected'         : self._result.selected,
                'features.ixn.dropped.encoded'  : self._result.dropped,
                'features.ixn.mapping'          : self._result.fe_map,
                'encodings.ixn'                 : self._result.encoding,
                'stats.encoded.ixn'             : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class WoeEncoderWorker(WorkerBase[WoeEncoderResult]):
    """
    Step 7: Weight of Evidence (WOE) Encoding.
    
    Implements WOE transformation to convert variables into a continuous 
    scale relative to the binary target probability.

    Parameters
    ----------
    inputs : WoeEncoderInput, optional
        Custom WOE configuration. If None, resolves from\
        'train.stable', 'features.encoded.num' and 'features.encoded.cat'.
    **kwargs : dict
        Additional parameters passed to the WoeEncoder worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.woe_encoder.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for woe_encoder training
    - num_features != None → custom subset of numerical encoded vars
    - cat_features != None → custom subset of categorical encoded vars

    Returns
    -------
    WoeEncoderWorker
        The populated instance of WoeEncoderWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.woe_encoder
    - inputs    → ctx.workers.woe_encoder.spec
    - result    → ctx.workers.woe_encoder.result
    - state     → ctx.workers.woe_encoder.state
    - report    → ctx.workers.woe_encoder.report
    - stats     → ctx.workers.woe_encoder.stats
    - fitter    → ctx.workers.woe_encoder.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - data      → ctx.datasets.train.encoded.num
    - features  → ctx.datasets.features.woe.encoded
    - features  → ctx.datasets.features.woe.selected
    - mapping   → ctx.datasets.features.woe.mapping    
    - dropped   → ctx.datasets.features.woe.dropped.encoded
    - encodings → ctx.datasets.encodings.woe
    - stats     → ctx.datasets.stats.encoded.woe.encoding

    Examples
    --------
    Defining the Project Context (Control Plane) & Project Scope:
    
    ```python
    import aidflow as aid
    
    ctx = aid.ProjectContext()
    
    ctx.genesis(
        project      = 'test',
        input_data   = 'test.csv',
        target       = 'target',
        yy_mm        = 'decision_date',
        yy_mm_format = '%Y-%m-%d'
    )
    ctx.params()
    ```
    
    ### Worker → Model → Fitter
    
    Accessing worker's inputs schema → parameters & types (*shown in the display output*):

    ```python        
    ctx.workers.woe_encoder.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.woe_encoder.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.workers.data_loader()    # 0. import data to the context scope
    ctx.workers.fe_cleaner()     # 1. apply cleaning logic to the imported data (0)
    ctx.workers.fe_stabilizer()  # 2. apply temporal stability logic to the cleaned data (1)
    ctx.workers.obs_stabilizer() # 3. apply observation stability logic to the cleaned data (2)
    ctx.workers.num_encoder()    # 4. apply numerical encoding logic to the stable data (3)
    ctx.workers.cat_encoder()    # 5. apply categorical encoding logic to the stable data (3)
    ctx.workers.ixn_encoder()    # 6. apply interactions encoding logic to the stable data (3)
    ctx.workers.woe_encoder()    # 7. apply woe encoding logic using trained encoding in steps (4-5-6)
    ```
    
    Note that steps 0-3 can be run by ExecutionContext one-liner:

    ```python   
    ctx.flow.process('data', verbose = 0) # steps 0 - 3 have been trained within the context scope
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.woe_encoder
    inputs   = worker.spec.inputs
    result   = worker.result
    state    = worker.state
    features = worker.features
    params   = worker.params
    stats    = worker.stats
    fitter   = worker.fitter  # jump to chosen internal fitter
    ...
    ```

    View to model report of executed worker:

    ```python
    ctx.workers.woe_encoder.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.woe_encoder.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.woe_encoder.fitter
    result_   = fitter.result
    features_ = fitter.features
    params_   = fitter.params
    stats_    = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.woe_encoder.fitters[0]
    prm = dict(
            gini_cutoff = 0.05,
            alpha = 0.01
    )
    woe_encoder = ctx.workers.woe_encoder(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase when various targets might be under testing\
    (*note that 'target2' must be present in the data*):

    ```python
    cfg = dict(
        target = 'target2',
    )
    woe_encoder = ctx.workers.woe_encoder(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Encoding.woe_encoder,
        choice  = PRM.ENCODE.woe_strategy.param,
        Model   = WoeEncoderUI,
        Input   = WoeEncoderInput,
        Result  = WoeEncoderResult,
        offload = [
            AIGFLOW.Encoding.num_encoder, 
            AIGFLOW.Encoding.cat_encoder, 
            AIGFLOW.Encoding.ixn_encoder
        ],
        title   = 'WOE ENCODING'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = WoeEncoderEnvelope(
                inputs = WoeEncoderInput(
                    train_set    = self.sets.get('train.stable'),
                    num_features = self.sets.get('features.num.encoded'),
                    cat_features = self.sets.get('features.cat.encoded')
                ),
                config = WoeEncoderEnvelopeCfg(
                    train_num = self.sets.get('train.encoded.num') if is_default else None,
                    train_cat = self.sets.get('train.encoded.cat') if is_default else None
                )
            ),

            datasets = DatasetBinding(fitted = {
                'train.encoded.woe'             : self._result.data,
                'features.woe.encoded'          : self._result.features,
                'features.woe.selected'         : self._result.selected,
                'features.woe.dropped.encoded'  : self._result.dropped,
                'features.woe.mapping'          : self._result.fe_map,
                'encodings.woe'                 : self._result.encoding,
                'stats.encoded.woe.encoding'    : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class WoeStabilizerWorker(WorkerBase[WoeStabilizerResult]):
    """
    Step 8: WOE Feature Stability Validation.
    
    The final transformation guardrail. Screens WOE-encoded features for 
    temporal stability to ensure the encoding is robust across data folds.

    Parameters
    ----------
    inputs : WoeStabilizerInput, optional
        Custom stability configuration. If None, resolves from 'train.stable',\
        'test.clean' and 'features.encoded.woe'.
    **kwargs : dict
        Additional parameters passed to the WoeStabilizer worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.woe_stabilizer.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for woe_stabilizer training
    - test_set  != None → custom test data for woe_stabilizer assessment view
    - woe_features != None → custom subset of woe encoded vars

    Returns
    -------
    WoeStabilizerWorker
        The populated instance of WoeStabilizerWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.woe_stabilizer
    - inputs    → ctx.workers.woe_stabilizer.spec
    - result    → ctx.workers.woe_stabilizer.result
    - state     → ctx.workers.woe_stabilizer.state
    - report    → ctx.workers.woe_stabilizer.report
    - stats     → ctx.workers.woe_stabilizer.stats
    - fitter    → ctx.workers.woe_stabilizer.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - features  → ctx.datasets.features.woe.stable
    - features  → ctx.datasets.features.all.stable.target
    - features  → ctx.datasets.features.num.stable.target
    - features  → ctx.datasets.features.cat.stable.target
    - features  → ctx.datasets.features.ixn.stable.target
    - dropped   → ctx.datasets.features.woe.dropped.stable
    - dropped   → ctx.datasets.features.all.dropped.stable.target
    - dropped   → ctx.datasets.features.num.dropped.stable.target
    - dropped   → ctx.datasets.features.cat.dropped.stable.target
    - dropped   → ctx.datasets.features.ixn.dropped.stable.target
    - stats     → ctx.datasets.stats.encoded.woe.stable

    Examples
    --------
    Defining the Project Context (Control Plane) & Project Scope:
    
    ```python
    import aidflow as aid
    
    ctx = aid.ProjectContext()
    
    ctx.genesis(
        project      = 'test',
        input_data   = 'test.csv',
        target       = 'target',
        yy_mm        = 'decision_date',
        yy_mm_format = '%Y-%m-%d'
    )
    ctx.params()
    ```
    
    ### Worker → Model → Fitter
    
    Accessing worker's inputs schema → parameters & types (*shown in the display output*):

    ```python        
    ctx.workers.woe_stabilizer.spec
    ```    

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.woe_stabilizer.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.workers.data_loader()    # 0. import data to the context scope
    ctx.workers.fe_cleaner()     # 1. apply cleaning logic to the imported data (0)
    ctx.workers.fe_stabilizer()  # 2. apply temporal stability logic to the cleaned data (1)
    ctx.workers.obs_stabilizer() # 3. apply observation stability logic to the cleaned data (2)
    ctx.workers.num_encoder()    # 4. apply numerical encoding logic to the stable data (3)
    ctx.workers.cat_encoder()    # 5. apply categorical encoding logic to the stable data (3)
    ctx.workers.ixn_encoder()    # 6. apply interactions encoding logic to the stable data (3)
    ctx.workers.woe_encoder()    # 7. apply woe encoding logic using trained encoding in steps (4-5-6)
    ctx.workers.woe_stabilizer() # 8. apply woe stabilization logic using trained woe encoding in step (7)
    ```

    Note that steps 0-3 can be run by ExecutionContext one-liner:

    ```python   
    ctx.flow.process('data', verbose = 0) # steps 0 - 3 have been trained within the context scope
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.woe_stabilizer
    inputs   = worker.spec.inputs
    result   = worker.result
    state    = worker.state
    features = worker.features
    params   = worker.params
    stats    = worker.stats
    fitter   = worker.fitter  # jump to chosen internal fitter
    ...
    ```

    View to model report of executed worker:

    ```python
    ctx.workers.woe_stabilizer.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.woe_stabilizer.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.woe_stabilizer.fitter
    result_   = fitter.result
    features_ = fitter.features
    params_   = fitter.params
    stats_    = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.woe_stabilizer.fitters[0]
    prm = dict(
            stability_cutoff = 0.8,
            alpha = 0.01
    )
    woe_stabilizer = ctx.workers.woe_stabilizer(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase when various targets might be under testing\
    (*note that 'target2' and 'time_var2' must be present in the data*):

    ```python
    cfg = dict(
        target   = 'target2',
        time_var = 'time_var2',
    )
    woe_stabilizer = ctx.workers.woe_stabilizer(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Encoding.woe_stabilizer,
        Model   = WoeStabilizerUI,
        Input   = WoeStabilizerInput,
        Result  = WoeStabilizerResult,
        offload = AIGFLOW.Encoding.woe_stabilizer,
        title   = 'WOE STABILIZE'
    )

    def _get_contract(self, is_default: bool = True):
        source1 = Dataclass.schema(FeStabilizerResult)
        source2 = Dataclass.schema(IxnEncoderResult)        
        return WorkerContract(

            defaults = WoeStabilizerEnvelope(
                inputs = WoeStabilizerInput(
                    train_set    = self.sets.get('train.stable'),
                    test_set     = self.sets.get('test.clean'),
                    woe_features = self.sets.get('features.woe.encoded')
                ),
                config = WoeStabilizerEnvelopeCfg(
                    train_woe = self.sets.get('train.encoded.woe') if is_default else None,
                )
            ),

            datasets = DatasetBinding(
                fitted = {
                    'features.woe.stable'           : self._result.features.woe,
                    'features.woe.dropped.stable'   : self._result.dropped.woe,
                    'stats.encoded.woe.stable'      : self._processor.stats,
                },
                subset = (
                    (AIGFLOW.Data.fe_stabilizer, {
                        'features.all.stable.target'         : [source1.features,     self._result.features.raw],
                        'features.num.stable.target'         : [source1.num_features, self._result.features.raw],
                        'features.cat.stable.target'         : [source1.cat_features, self._result.features.raw],
                        'features.all.dropped.stable.target' : [source1.features,     self._result.dropped.raw],
                        'features.num.dropped.stable.target' : [source1.num_features, self._result.dropped.raw],
                        'features.cat.dropped.stable.target' : [source1.cat_features, self._result.dropped.raw],
                    }),
                    (AIGFLOW.Encoding.ixn_encoder, {
                        'features.ixn.stable.target'         : [source2.features, self._result.features.raw],
                        'features.ixn.dropped.stable.target' : [source2.features, self._result.dropped.raw],
                    })                    
                )
            )
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DataEncoderWorker(WorkerBase[DataEncoderResult]):
    """
    Step 9: ML Sample Generation & Synchronization.
        
    Acts as the final synchronization point before agent training. It applies\
    all previously trained encoding transformation logic (Numerical, Categorical, and WOE)\
    to create synchronized 'ml' versions of the training and test sets.
    
    This method ensures that the exact same encoding pipelines trained during\
    the Encoding Phase are applied consistently across both datasets,\
    preventing data leakage and ensuring model validity.
    
    Notes
    -----
    - DataEncoder serves as the entry point to the agent training phase.
    - It applies all fitted encodings to transform input data into ML-ready datasets.
    - Any new data sharing the original training schema can be passed in directly without\
    requiring a separate test split - new input data are split to train and test part.
    - Results are registered under `ctx.datasets` as `train.ml` and `test.ml`.

    Parameters
    ----------
    inputs : DataEncoderInput, optional
        Custom input configuration for sample generation. If None, resolves\
        defaults from the current pipeline state (train_set, test_set, woe_features).
    **kwargs : dict
        Additional parameters passed to the DataEncoder worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.data_encoder.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for encoding application by data_encoder 
    - test_set  != None → custom test for encoding application by data_encoder 
    - woe_features != None → optional subset of woe features

    Returns
    -------
    DataEncoderWorker
        The populated instance of DataEncoderWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.data_encoder
    - inputs    → ctx.workers.data_encoder.spec
    - result    → ctx.workers.data_encoder.result
    - state     → ctx.workers.data_encoder.state
    - stats     → ctx.workers.data_encoder.stats
    - fitter    → ctx.workers.data_encoder.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - data      → ctx.datasets.train.ml
    - data      → ctx.datasets.test.ml
    - dtypes    → ctx.datasets.dtypes.ml
    - features  → ctx.datasets.all.ml
    - stats     → ctx.datasets.stats.ml

    Examples
    --------
    Defining the Project Context (Control Plane) & Project Scope:
    
    ```python
    import aidflow as aid
    
    ctx = aid.ProjectContext()
    
    ctx.genesis(
        project      = 'test',
        input_data   = 'test.csv',
        target       = 'target',
        yy_mm        = 'decision_date',
        yy_mm_format = '%Y-%m-%d'
    )
    ctx.params()
    ```
    
    ### Worker → Model → Fitter
    
    Accessing worker's inputs schema → parameters & types (*shown in the display output*):

    ```python        
    ctx.workers.data_encoder.spec
    ```    

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.data_encoder.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.workers.data_loader()    # 0. import data to the context scope
    ctx.workers.fe_cleaner()     # 1. apply cleaning logic to the imported data (0)
    ctx.workers.fe_stabilizer()  # 2. apply temporal stability logic to the cleaned data (1)
    ctx.workers.obs_stabilizer() # 3. apply observation stability logic to the cleaned data (2)
    ctx.workers.num_encoder()    # 4. apply numerical encoding logic to the stable data (3)
    ctx.workers.cat_encoder()    # 5. apply categorical encoding logic to the stable data (3)
    ctx.workers.ixn_encoder()    # 6. apply interactions encoding logic to the stable data (3)
    ctx.workers.woe_encoder()    # 7. apply woe encoding logic using trained encoding in steps (4-5-6)
    ctx.workers.woe_stabilizer() # 8. apply woe stabilization logic using trained woe encoding in step (7)
    ctx.workers.data_encoder()   # 9. apply trained encodings (4-7) to transform train-test data
    ```
    
    Note that steps 0-3 and steps 4-9 can be run by ExecutionContext one-liner:

    ```python   
    ctx.flow.process('data', verbose = 0)     # steps 0 - 3 have been trained within the context scope
    ctx.flow.process('encoding', verbose = 0) # steps 4 - 9 have been trained within the context scope
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.data_encoder
    inputs   = worker.spec.inputs
    result   = worker.result
    state    = worker.state
    features = worker.features
    params   = worker.params
    stats    = worker.stats
    fitter   = worker.fitter  # jump to chosen internal fitter
    ...
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.data_encoder.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.data_encoder.fitter
    params_   = fitter.params
    stats_    = fitter.stats
    ```

    Use new data to apply fitted encodings to obtain ML-ready datasets:
    
    ```python
    data2 = ... # new pd.dataframe with the orig. training schema
    ctx.workers.data_encoder(train_set = data2)
    ```

    Obtaining created ML-ready sets (auto-provided to agents):

    ```python
    train = ctx.datasets.train.ml
    test  = ctx.datasets.test.ml
    ```

    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.data_encoder.fitters[0]
    prm = dict(
        preprocess = True,
        remove_outliers = False
    )
    data_encoder = ctx.workers.data_encoder(fx_choice = fx, fx_params = prm)
    ```

    ```python
    cfg = dict(
        time_var  = 'time_var2',
        test_size = 0.3
    )
    data_encoder = ctx.workers.data_encoder(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Encoding.data_encoder,
        Model   = DataEncoder,
        Input   = DataEncoderInput,
        Result  = DataEncoderResult,
        offload = [
            AIGFLOW.Data.fe_cleaner, 
            AIGFLOW.Data.obs_stabilizer,
            AIGFLOW.Encoding.woe_encoder
        ],
        title   = 'DATA ENCODER'
    )

    def _get_contract(self, is_default: bool = True):
        is_woe = self._processor.is_fitted(AIGFLOW.Encoding.woe_encoder)
        is_wfe = self._processor.is_fitted(AIGFLOW.Encoding.woe_stabilizer)
        return WorkerContract(

            defaults = DataEncoderEnvelope(
                inputs = DataEncoderInput(
                    train_set    = self.sets.get('train.stable'),
                    test_set     = self.sets.get('test.clean'),
                    woe_features = self.sets.get('features.woe.stable') if is_wfe else None
                ),
                config = DataEncoderEnvelopeCfg(
                    train_woe = self.sets.get('train.encoded.woe') if is_default and is_woe else None
                )
            ),

            datasets = DatasetBinding(fitted = {
                'train.ml'        : self._result.data,
                'test.ml'         : self._result.test_set,
                'dtypes.ml'       : self._result.dtypes,
                'features.all.ml' : self._result.features,
                'stats.ml'        : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class EncodingPhase(WorkerPhaseBase):
    """
    AIG Flow Encoding Phase Orchestrator.
    
    Coordinates the conversion of stable features into ML-ready encoded variables. 
    It maintains a strict dependency on the DataPhase output, ensuring that only 
    stabilized features are transformed.
    """

    def __init__(self, ctx: ProjectContextProtocol):    
        super().__init__(ctx)
        self._register((
            NumEncoderWorker,
            CatEncoderWorker,
            IxnEncoderWorker,
            WoeEncoderWorker,
            WoeStabilizerWorker,
            DataEncoderWorker
        ))
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬