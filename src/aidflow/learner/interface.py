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

from ..core.parameters      import  (
                                    PRM, 
                                    AIGFLOW
                                    )
from ..core.base.worker     import  (
                                    WorkerConfig,
                                    WorkerContract,
                                    DatasetBinding,
                                    WorkerBase,
                                    WorkerPhaseBase,
                                    )
from ..core.protocol        import  ProjectContextProtocol

from .agents                import  (
                                    Agent1Input,
                                    Agent1Result,
                                    Agent1PredictorUI,
                                    Agent2Input,
                                    Agent2Result,
                                    Agent2PredictorUI,
                                    Agent3Input,
                                    Agent3Result,
                                    Agent3PredictorUI,
                                    Agent4Input,
                                    Agent4Result,
                                    Agent4PredictorUI,
                                    )


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TRAINING PHASE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Agent1PredictorWorker(WorkerBase[Agent1Result]):
    """
    Step 10: Agent 1 Training Worker.
    
    Orchestrates the fitting of Learning Agent 1. By default, it consumes 
    'train.ml' and utilizes woe stable features from the global context.

    Parameters
    ----------
    inputs : Agent1Input, optional
        Custom training configuration. If None, resolves defaults from pipeline sets.
    **kwargs : dict
        Additional parameters for the Agent 1 worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.agent1_predictor.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for agent1_predictor training
    - woe_features != None → optional subset of woe features

    Returns
    -------
    Agent1PredictorWorker
        The populated instance of Agent1PredictorWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.agent1_predictor
    - inputs    → ctx.workers.agent1_predictor.spec
    - result    → ctx.workers.agent1_predictor.result
    - state     → ctx.workers.agent1_predictor.state
    - report    → ctx.workers.agent1_predictor.report
    - stats     → ctx.workers.agent1_predictor.stats
    - fitter    → ctx.workers.agent1_predictor.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - predictions → ctx.datasets.predictions.train.agent1
    - features    → ctx.datasets.features.agent1.predicted
    - features    → ctx.datasets.features.agent1.selected
    - dropped     → ctx.datasets.features.agent1.dropped
    - stats       → ctx.datasets.stats.agents.agent1

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
    ctx.workers.agent1_predictor.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.agent1_predictor.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.flow.process('data')        # 1. process workers 0-3 → data load - data stability
    ctx.flow.process('encoding')    # 2. process workers 4-9 → train and apply encoding to train and test samples
    ctx.workers.agent1_predictor()  # 3. train agent1 on stabilized and encoded woe data (2)
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.agent1_predictor
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
    ctx.workers.agent1_predictor.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.agent1_predictor.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.agent1_predictor.fitter
    result_   = fitter.result
    features_ = fitter.features
    config_   = fitter.config
    params_   = fitter.params
    stats_    = fitter.stats
    model_    = fitter.model
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.agent1_predictor.fitters[0]
    prm = dict(
        alpha = 0.05,
        eps = 1e-3,
        tol = 1e-4,
        cv = 3,
        shadow_quantile = 0.9
    )
    agent1 = ctx.workers.agent1_predictor(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target = 'target2',
        val_size = 0.33,
        n_permutations = 3
    )
    agent1 = ctx.workers.agent1_predictor(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Training.agent1,
        Model   = Agent1PredictorUI,
        Input   = Agent1Input,
        Result  = Agent1Result,
        offload = AIGFLOW.Training.agent1,
        title   = 'GLM SPARSE OPTIMIZATION'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = Agent1Input(
                train_set    = self.sets.get('train.ml'),
                woe_features = self.sets.get('features.woe.stable'),
            ),
            
            datasets = DatasetBinding(fitted = {
                'predictions.train.agent1'  : self._result.data,
                'features.agent1.predicted' : self._result.features,
                'features.agent1.selected'  : self._result.selected,
                'features.agent1.dropped'   : self._result.dropped,
                'stats.agents.agent1'       : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Agent2PredictorWorker(WorkerBase[Agent2Result]):
    """
    Step 11: Agent 2 Training Worker.
    
    Orchestrates the fitting of Learning Agent 2. By default, it consumes 
    'train.ml' and utilizes woe stable features from the global context.    

    Parameters
    ----------
    inputs : Agent2Input, optional
        Custom training configuration. If None, resolves defaults from pipeline sets.
    **kwargs : dict
        Additional parameters for the Agent 2 worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.agent2_predictor.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for agent2_predictor training
    - woe_features != None → optional subset of woe features

    Returns
    -------
    Agent2PredictorWorker
        The populated instance of Agent2PredictorWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.agent2_predictor
    - inputs    → ctx.workers.agent2_predictor.spec
    - result    → ctx.workers.agent2_predictor.result
    - state     → ctx.workers.agent2_predictor.state
    - report    → ctx.workers.agent2_predictor.report
    - stats     → ctx.workers.agent2_predictor.stats
    - fitter    → ctx.workers.agent2_predictor.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - predictions → ctx.datasets.predictions.train.agent2
    - features    → ctx.datasets.features.agent2.predicted
    - features    → ctx.datasets.features.agent2.selected
    - dropped     → ctx.datasets.features.agent2.dropped
    - stats       → ctx.datasets.stats.agents.agent2

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
    ctx.workers.agent2_predictor.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.agent2_predictor.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.flow.process('data')        # 1. process workers 0-3 → data load - data stability
    ctx.flow.process('encoding')    # 2. process workers 4-9 → train and apply encoding to train and test samples
    ctx.workers.agent2_predictor()  # 3. train agent2 on stabilized and encoded woe data (2)
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.agent2_predictor
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
    ctx.workers.agent2_predictor.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.agent2_predictor.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.agent2_predictor.fitter
    result_   = fitter.result
    features_ = fitter.features
    config_   = fitter.config
    params_   = fitter.params
    stats_    = fitter.stats
    model_    = fitter.model
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.agent2_predictor.fitters[0]
    prm = dict(
        alpha = 0.05,
        eps = 1e-3,
        tol = 1e-4,
        cv = 3,
        shadow_quantile = 0.9
    )
    agent2 = ctx.workers.agent2_predictor(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target = 'target2',
        val_size = 0.33,
        n_permutations = 3
    )
    agent2 = ctx.workers.agent2_predictor(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Training.agent2,
        choice  = PRM.TRAIN.glm_strategy.param,
        Model   = Agent2PredictorUI,
        Input   = Agent2Input,
        Result  = Agent2Result,
        offload = AIGFLOW.Training.agent2,
        title   = 'GLM DENSE OPTIMIZATION'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = Agent2Input(
                train_set    = self.sets.get('train.ml'),
                woe_features = self.sets.get('features.woe.stable'),
            ),
            
            datasets = DatasetBinding(fitted = {
                'predictions.train.agent2'  : self._result.data,
                'features.agent2.predicted' : self._result.features,
                'features.agent2.selected'  : self._result.selected,
                'features.agent2.dropped'   : self._result.dropped,
                'stats.agents.agent2'       : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Agent3PredictorWorker(WorkerBase[Agent3Result]):
    """
    Step 12: Agent 3 Training Worker.
    
    Orchestrates the fitting of Learning Agent 3. By default, it consumes 
    'train.ml' and utilizes woe stable features from the global context.

    Parameters
    ----------
    inputs : Agent3Input, optional
        Custom training configuration. If None, resolves defaults from pipeline sets.
    **kwargs : dict
        Additional parameters for the Agent 3 worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.agent3_predictor.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for agent3_predictor training
    - woe_features != None → optional subset of woe features

    Returns
    -------
    Agent3PredictorWorker
        The populated instance of Agent3PredictorWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.agent3_predictor
    - inputs    → ctx.workers.agent3_predictor.spec
    - result    → ctx.workers.agent3_predictor.result
    - state     → ctx.workers.agent3_predictor.state
    - report    → ctx.workers.agent3_predictor.report
    - stats     → ctx.workers.agent3_predictor.stats
    - fitter    → ctx.workers.agent3_predictor.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - predictions → ctx.datasets.predictions.train.agent3
    - features    → ctx.datasets.features.agent3.predicted
    - features    → ctx.datasets.features.agent3.selected
    - dropped     → ctx.datasets.features.agent3.dropped
    - stats       → ctx.datasets.stats.agents.agent3

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
    ctx.workers.agent3_predictor.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.agent3_predictor.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.flow.process('data')        # 1. process workers 0-3 → data load - data stability
    ctx.flow.process('encoding')    # 2. process workers 4-9 → train and apply encoding to train and test samples
    ctx.workers.agent3_predictor()  # 3. train agent3 on stabilized and encoded woe data (2)
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.agent3_predictor
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
    ctx.workers.agent3_predictor.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.agent3_predictor.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.agent3_predictor.fitter
    result_   = fitter.result
    features_ = fitter.features
    config_   = fitter.config
    params_   = fitter.params
    stats_    = fitter.stats
    model_    = fitter.model
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.agent3_predictor.fitters[0]
    prm = dict(
        alpha = 0.05,
        eps = 1e-3,
        tol = 1e-4,
        cv = 3,
        shadow_quantile = 0.9
    )
    agent3 = ctx.workers.agent3_predictor(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target = 'target2',
        val_size = 0.33,
        n_permutations = 3
    )
    agent3 = ctx.workers.agent3_predictor(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Training.agent3,
        Model   = Agent3PredictorUI,
        Input   = Agent3Input,
        Result  = Agent3Result,
        offload = AIGFLOW.Training.agent3,
        title   = 'GLM PLS OPTIMIZATION'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = Agent3Input(
                train_set    = self.sets.get('train.ml'),
                woe_features = self.sets.get('features.woe.stable'),
            ),
            
            datasets = DatasetBinding(fitted = {
                'predictions.train.agent3'  : self._result.data,
                'features.agent3.predicted' : self._result.features,
                'features.agent3.selected'  : self._result.selected,
                'features.agent3.dropped'   : self._result.dropped,
                'stats.agents.agent3'       : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Agent4PredictorWorker(WorkerBase[Agent4Result]):
    """
    Step 13: Agent 4 Training Worker.
    
    Orchestrates the fitting of Learning Agent 4. By default, it consumes 
    'train.ml' and utilizes numerical/categorical features from the 
    global context.

    Parameters
    ----------
    inputs : Agent4Input, optional
        Custom training configuration. If None, resolves defaults from pipeline sets.
    **kwargs : dict
        Additional parameters for the Agent 4 worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.agent4_predictor.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for agent4_predictor training
    - woe_features != None → optional subset of woe features

    Returns
    -------
    Agent4PredictorWorker
        The populated instance of Agent4PredictorWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.agent4_predictor
    - inputs    → ctx.workers.agent4_predictor.spec
    - result    → ctx.workers.agent4_predictor.result
    - state     → ctx.workers.agent4_predictor.state
    - report    → ctx.workers.agent4_predictor.report
    - stats     → ctx.workers.agent4_predictor.stats
    - fitter    → ctx.workers.agent4_predictor.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - predictions → ctx.datasets.predictions.train.agent4
    - features    → ctx.datasets.features.agent4.predicted
    - features    → ctx.datasets.features.agent4.selected
    - dropped     → ctx.datasets.features.agent4.dropped
    - stats       → ctx.datasets.stats.agents.agent4

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
    ctx.workers.agent4_predictor.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.agent4_predictor.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.flow.process('data')        # 1. process workers 0-3 → data load - data stability
    ctx.flow.process('encoding')    # 2. process workers 4-9 → train and apply encoding to train and test samples
    ctx.workers.agent4_predictor()  # 3. train agent4 on stabilized and encoded woe data (2)
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.agent4_predictor
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
    ctx.workers.agent4_predictor.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.agent4_predictor.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.agent4_predictor.fitter
    result_   = fitter.result
    features_ = fitter.features
    config_   = fitter.config
    params_   = fitter.params
    stats_    = fitter.stats
    model_    = fitter.model
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.agent4_predictor.fitters[0]
    prm = dict(
        alpha = 0.05,
        eps = 1e-3,
        tol = 1e-4,
        cv = 3,
        shadow_quantile = 0.9
    )
    agent4 = ctx.workers.agent4_predictor(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target = 'target2',
        val_size = 0.33,
        n_permutations = 3
    )
    agent4 = ctx.workers.agent4_predictor(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Training.agent4,
        choice  = PRM.TRAIN.gb_strategy.param,
        Model   = Agent4PredictorUI,
        Input   = Agent4Input,
        Result  = Agent4Result,
        offload = AIGFLOW.Training.agent4,
        title   = 'GBM OPTIMIZATION'
    )

    def _get_contract(self, is_default: bool = True):
        is_target  = self._processor.is_fitted(AIGFLOW.Encoding.woe_stabilizer)
        dst_target = 'target' if is_target else 'dist'
        return WorkerContract(

            defaults = Agent4Input(
                train_set    = self.sets.get('train.ml'),
                num_features = self.sets.get(f'features.num.stable.{dst_target}'),
                cat_features = self.sets.get(f'features.cat.stable.{dst_target}'),
            ),
            
            datasets = DatasetBinding(fitted = {
                'predictions.train.agent4'  : self._result.data,
                'features.agent4.predicted' : self._result.features,
                'features.agent4.selected'  : self._result.selected,
                'features.agent4.dropped'   : self._result.dropped,
                'stats.agents.agent4'       : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class TrainingPhase(WorkerPhaseBase):
    """
    AIG Flow Training Phase Orchestrator.
    
    Manages the transition from encoded data to trained agents. It ensures 
    that all agents are trained on the same synchronized 'ml' dataset produced 
    by the DataEncoder.
    """

    def __init__(self, ctx: ProjectContextProtocol):    
        super().__init__(ctx)
        self._register((
            Agent1PredictorWorker,
            Agent2PredictorWorker,
            Agent3PredictorWorker,
            Agent4PredictorWorker,
        ))
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬