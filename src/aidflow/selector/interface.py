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

from .predict               import  (
                                    AgentsPredictorInput,
                                    AgentsPredictorResult,
                                    AgentsPredictor
                                    )
from .compare               import  (
                                    AgentsComparatorInput, 
                                    AgentsComparatorResult,
                                    AgentsComparatorUI
                                    )
from .transition            import  (
                                    AgentsTransitionInput, 
                                    AgentsTransitionResult,
                                    AgentsTransitionUI
                                    )
from .select                import  (
                                    AgentSelectorEnvelopeCfg,
                                    AgentSelectorEnvelope,
                                    AgentSelectorInput,
                                    AgentSelectorResult,
                                    AgentSelectorUI
                                    )
from .select.agent          import  (
                                    AgentPredictorInput,
                                    AgentPredictorResultDual,
                                    AgentPredictorUI,
                                    AgentPredictorDashWaiter
                                    )


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SELECTION PHASE 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class AgentsPredictorWorker(WorkerBase[AgentsPredictorResult]):
    """
    Step 14: Ensemble Scoring (Candidate Generation).
    
    Generates probability predictions for all learning agents using a target 
    dataset (typically Out-of-Time samples). This step creates the raw evidence 
    base used by all subsequent evaluation workers in the Selection Phase.

    Parameters
    ----------
    inputs : AgentsPredictorInput, optional
        Custom scoring configuration. If None, resolves defaults from 'test.ml' state.
    **kwargs : dict
        Additional parameters passed to the Predictor worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.agents_predictor.spec` with main params:
    - inputs == None → pipeline data context
    - data   != None → custom data for agent predictions

    Returns
    -------
    AgentsPredictorWorker
        The populated instance of AgentsPredictorWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.agents_predictor
    - inputs    → ctx.workers.agents_predictor.spec
    - result    → ctx.workers.agents_predictor.result
    - state     → ctx.workers.agents_predictor.state
    - stats     → ctx.workers.agents_predictor.stats
    - fitter    → ctx.workers.agents_predictor.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - predictions → ctx.datasets.predictions.test.agent1
    - predictions → ctx.datasets.predictions.test.agent2
    - predictions → ctx.datasets.predictions.test.agent3
    - predictions → ctx.datasets.predictions.test.agent4
    - stats       → ctx.datasets.stats.agents.prediction

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
    ctx.workers.agents_predictor.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.agents_predictor.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.flow.process('data')        # 1. process workers 0-3 → data load - data stability
    ctx.flow.process('encoding')    # 2. process workers 4-9 → train and apply encoding to train and test samples
    ctx.flow.process('training')    # 3. process workers 10-13 → agents trainining on encoded train sample
    ctx.workers.agents_predictor()  # 4. test sample prediction by agents
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.agents_predictor
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
    ctx.workers.agents_predictor.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.agents_predictor.fitter
    params_   = fitter.params
    stats_    = fitter.stats
    ```    

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target = 'target2',
        preprocess = True
    )
    agents_predictor = ctx.workers.agents_predictor(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model   = AIGFLOW.Selection.predictor,
        Model   = AgentsPredictor,
        Input   = AgentsPredictorInput,
        Result  = AgentsPredictorResult,
        offload = AIGFLOW.Selection.predictor,
        title   = 'AGENTS PREDICTOR'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = AgentsPredictorInput(
                data = self.sets.get('test.ml')
            ),
            
            datasets = DatasetBinding(fitted = {
                'predictions.test.agent1' : self._result.agent1,
                'predictions.test.agent2' : self._result.agent2,
                'predictions.test.agent3' : self._result.agent3,
                'predictions.test.agent4' : self._result.agent4,
                'stats.agents.prediction' : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class AgentsComparatorWorker(WorkerBase[AgentsComparatorResult]):
    """
    Step 15: Performance Benchmarking (Objective Evaluation).
    
    Evaluates all agent predictions against a set of rigorous performance metrics. 
    It computes Scoring accuracy, Lift charts, and Calibration curves to determine 
    which agent provides the most reliable predictive power on unseen data.

    Parameters
    ----------
    inputs : AgentsComparatorInput, optional
        Custom evaluation configuration. If None, resolves from 'test.ml' state.
    **kwargs : dict
        Additional parameters passed to the Comparator worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.agents_comparator.spec` with main params:
    - inputs == None → pipeline data context
    - data   != None → custom data for agents_comparator assessment

    Returns
    -------
    AgentsComparatorWorker
        The populated instance of AgentsComparatorWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.agents_comparator
    - inputs    → ctx.workers.agents_comparator.spec
    - result    → ctx.workers.agents_comparator.result
    - state     → ctx.workers.agents_comparator.state
    - report    → ctx.workers.agents_comparator.report
    - stats     → ctx.workers.agents_comparator.stats
    - fitter    → ctx.workers.agents_comparator.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - stats     → ctx.datasets.stats.agents.comparison

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
    ctx.workers.agents_comparator.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.agents_comparator.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.flow.process('data')         # 1. process workers 0-3 → data load - data stability
    ctx.flow.process('encoding')     # 2. process workers 4-9 → train and apply encoding to train and test samples
    ctx.flow.process('training')     # 3. process workers 10-13 → agents trainining on encoded train sample
    ctx.workers.agents_comparator()  # 4. agents comparison on test sample
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.agents_comparator
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
    ctx.workers.agents_comparator.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.agents_comparator.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.agents_comparator.fitter
    params_   = fitter.params
    stats_    = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.agents_comparator.fitters[0]
    prm = dict(
        alpha = 0.05,
        proba_bins = 15
    )
    agents_comparator = ctx.workers.agents_comparator(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target = 'target2',
        preprocess = True
    )
    agents_comparator = ctx.workers.agents_comparator(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model  = AIGFLOW.Selection.comparator,
        Model  = AgentsComparatorUI,
        Input  = AgentsComparatorInput,
        Result = AgentsComparatorResult,
        title  = 'AGENTS COMPARATOR'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = AgentsComparatorInput(
                data = self.sets.get('test.ml')
            ),
            
            datasets = DatasetBinding(fitted = {
                'stats.agents.comparison' : self._processor.stats                
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class AgentsTransitionWorker(WorkerBase[AgentsTransitionResult]):
    """
    Step 16: Divergence Analysis (Stability Screening).
        
    Analyzes the 'agreement' between different agent strategies. By calculating 
    transition matrices, this step identifies where agents disagree on a sample's 
    risk level, highlighting potential instability or overfitting in specific 
    segments of the data.

    Parameters
    ----------
    inputs : AgentsTransitionInput, optional
        Custom transition configuration. If None, resolves from 'test.ml' state.
    **kwargs : dict
        Additional parameters passed to the Transition worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.agents_transition.spec` with main params:
    - inputs == None → pipeline data context
    - data   != None → custom data for agents_transition assessment

    Returns
    -------
    AgentsTransitionWorker
        The populated instance of AgentsTransitionWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.agents_transition
    - inputs    → ctx.workers.agents_transition.spec
    - result    → ctx.workers.agents_transition.result
    - state     → ctx.workers.agents_transition.state
    - report    → ctx.workers.agents_transition.report
    - stats     → ctx.workers.agents_transition.stats
    - fitter    → ctx.workers.agents_transition.fitter
    
    Registers
    ---------
    Registers data into the datasets context:
    - stats     → ctx.datasets.stats.agents.transition

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
    ctx.workers.agents_transition.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.agents_transition.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.flow.process('data')         # 1. process workers 0-3 → data load - data stability
    ctx.flow.process('encoding')     # 2. process workers 4-9 → train and apply encoding to train and test samples
    ctx.flow.process('training')     # 3. process workers 10-13 → agents trainining on encoded train sample
    ctx.workers.agents_transition()  # 4. agents transition on test sample
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.agents_transition
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
    ctx.workers.agents_transition.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.agents_transition.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.agents_transition.fitter
    params_   = fitter.params
    stats_    = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.agents_transition.fitters[0]
    prm = dict(
        alpha = 0.05,
        proba_bins = 15
    )
    agents_transition = ctx.workers.agents_transition(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target = 'target2',
        preprocess = True
    )
    agents_transition = ctx.workers.agents_transition(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model  = AIGFLOW.Selection.transition,
        Model  = AgentsTransitionUI,
        Input  = AgentsTransitionInput,
        Result = AgentsTransitionResult,
        title  = 'AGENTS TRANSITION'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = AgentsTransitionInput(
                data = self.sets.get('test.ml')
            ),
            
            datasets = DatasetBinding(fitted = {
                'stats.agents.transition' : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class AgentPredictorWorker(WorkerBase[AgentPredictorResultDual]):
    """
    Step 18: AIDflow Production Agent: The Deployment Artifact.
    
    This is the final, promoted agent destined for production inference. 
    Unlike the ensemble agents used during training, this class encapsulates 
    the entire 'winning' logic—integrating its specific fitting strategy, 
    associated preprocessing encoders, and feature mappings into a single, 
    deployable unit.
    
    Strategic Role:
    - Inference Encapsulation: Acts as the sole interface for production scoring, 
      removing all ensemble complexity to provide a lean, high-performance API.
    - End-to-End Lineage: Bridges the gap between raw input data and final 
      probability scores by internalizing the preprocessing pipeline.
    - Governance & Audit: Maintains an immutable record of selected features 
      and encoder states, ensuring that production predictions are perfectly 
      reproducible and compliant with model governance standards.

    Notes
    -----
    - AgentPredictorWorker is automatically executed during Step 17 (Champion Promotion)\
    based on user configuration.
    - Can be run independently to bypass Step 17 when feature tuning or overview analysis\
    is not required.
    
    Parameters
    ----------
    inputs : AgentPredictorInput, optional
        Custom selection configuration. If None, resolves from 'train.ml' and 'test.ml' state.
    **kwargs : dict
        Additional parameters passed to the Selector worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.champion_predictor.spec` with main params:
    - inputs    == None → pipeline data context
    - train_set != None → custom train data for champion_predictor training
    - test_set  != None → custom test data for champion_predictor test metrics
    - features  != None → custom feature subset used for training champion_predictor

    Returns
    -------
    AgentPredictorWorker
        The populated instance of AgentPredictorWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.champion_predictor
    - inputs    → ctx.workers.champion_predictor.spec
    - result    → ctx.workers.champion_predictor.result
    - state     → ctx.workers.champion_predictor.state
    - report    → ctx.workers.champion_predictor.report
    - stats     → ctx.workers.champion_predictor.stats
    - fitter    → ctx.workers.champion_predictor.fitter
    
    Registers
    ---------
    Registers data into the datasets context:

    - predictions → predictions.train.champion_agent
    - predictions → predictions.test.champion_agent
    - features    → features.champion_agent.predicted
    - features    → features.champion_agent.selected
    - dropped     → features.champion_agent.dropped
    - stats       → stats.agents.champion_agent
    - stats       → stats.agents.selection
        
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
    ctx.workers.champion_predictor.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.champion_predictor.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.flow.process('data')            # 1. process workers 0-3 → data load - data stability
    ctx.flow.process('encoding')        # 2. process workers 4-9 → train and apply encoding to train and test samples
    ctx.flow.process('training')        # 3. process workers 10-13 → agents trainining on encoded train sample
    ctx.workers.champion_predictor()    # 4. direct champion agent training - skipping selection overview and tuning
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.champion_predictor
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
    ctx.workers.champion_predictor.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.champion_predictor.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.champion_predictor.fitter
    result_   = fitter.result
    features_ = fitter.features
    params_   = fitter.params
    stats_    = fitter.stats
    model_    = fitter.model
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.champion_predictor.fitters[0]
    prm = dict(
        agent_choice = 1
    )
    champion_predictor = ctx.workers.champion_predictor(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target = 'target2',
        val_size = 0.3
    )
    champion_predictor = ctx.workers.champion_predictor(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model  = AIGFLOW.Selection.champion_agent,
        Model  = AgentPredictorUI,
        Input  = AgentPredictorInput,
        Result = AgentPredictorResultDual,
        title  = 'CHAMPION AGENT'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = AgentPredictorInput(
                train_set = self.sets.get('train.ml'),
                test_set  = self.sets.get('test.ml'),
            ),
            
            datasets = DatasetBinding(fitted = {
                'predictions.train.champion_agent'  : self._result.data,
                'predictions.test.champion_agent'   : self._result.test_data,
                'features.champion_agent.predicted' : self._result.features,
                'features.champion_agent.selected'  : self._result.selected,
                'features.champion_agent.dropped'   : self._result.dropped,
                'stats.agents.champion_agent'       : self._processor.stats
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class AgentSelectorWorker(WorkerBase[AgentSelectorResult]):
    """
    Step 17: Champion Promotion (The Executive Decision).
    
    The final gate of the pipeline. This method aggregates evidence from 
    the Prediction, Comparison, and Transition steps to officially promote 
    one agent as the 'Champion.' It locks the winning model's state and 
    generates a SHAP-based explainability report for audit purposes.

    Parameters
    ----------
    inputs : AgentSelectorInput, optional
        Custom selection configuration. If None, resolves from 'test.ml' state.
    **kwargs : dict
        Additional parameters passed to the Selector worker.

    Notes
    -----
    Inputs full schema can be obtained by `ctx.workers.agent_selector.spec` with main params:
    - inputs == None → pipeline data context
    - data   != None → custom data for agent_selector overview and choice

    Returns
    -------
    AgentSelectorWorker
        The populated instance of AgentSelectorWorker.
    
        
    Populates
    ---------
    Populates attributes into the worker context:
    - model     → ctx.workers.agent_selector
    - inputs    → ctx.workers.agent_selector.spec
    - result    → ctx.workers.agent_selector.result
    - state     → ctx.workers.agent_selector.state
    - report    → ctx.workers.agent_selector.report
    - stats     → ctx.workers.agent_selector.stats
    - fitter    → ctx.workers.agent_selector.fitter
    
    Registers
    ---------
    Registers data into the datasets context:

    - predictions → predictions.train.champion_agent
    - predictions → predictions.test.champion_agent
    - features    → features.champion_agent.predicted
    - features    → features.champion_agent.selected
    - dropped     → features.champion_agent.dropped
    - stats       → stats.agents.champion_agent
    - stats       → stats.agents.selection
        
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
    ctx.workers.agent_selector.spec
    ```

    Accessing model's fitter options and schema → parameters & types 
    (*shown in the display output*):

    ```python        
    ctx.workers.agent_selector.fitters
    ```   

    Executing workers with defaults → auto provided by the context:
    
    ```python        
    ctx.flow.process('data')        # 1. process workers 0-3 → data load - data stability
    ctx.flow.process('encoding')    # 2. process workers 4-9 → train and apply encoding to train and test samples
    ctx.flow.process('training')    # 3. process workers 10-13 → agents trainining on encoded train sample
    ctx.workers.agents_comparator() # 4. process agents comparator on test sample to obtain assessment for selection
    ctx.workers.agent_selector()    # 5. champion agent selection
    ```

    Once worker has been executed, thus its main model has been built,\
    we have access to all populated attributes.
    
    ```python   
    worker   = ctx.workers.agent_selector
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
    ctx.workers.agent_selector.report.show()
    ```

    ### Model → Fitter

    Once worker has been executed, we can obtain internal fitter state\
    and its parametrization (*shown in the display output*):

    ```python        
    ctx.workers.agent_selector.fitter
    ```

    The fitted fitter provides all populated attributes.

    ```python        
    fitter    = ctx.workers.agent_selector.fitter
    params_   = fitter.params
    stats_    = fitter.stats
    ```
    
    Direct choice and parametrization of internal fitter:

    ```python
    fx  = ctx.workers.agent_selector.fitters[0]
    prm = dict(
        alpha = 0.05,
        proba_bins = 15,
        agent_choice = 1
    )
    agent_selector = ctx.workers.agent_selector(fx_choice = fx, fx_params = prm)
    ```

    In the experimenting phase, when various parametrization settings might be under testing\
    (*note that internal fitter API (fx_config) can vary with respect to chosen fitter*):

    ```python
    cfg = dict(
        target = 'target2',
        preprocess = True
    )
    agent_selector = ctx.workers.agent_selector(fx_config = cfg)
    ```
    """
    config = WorkerConfig(
        model  = AIGFLOW.Selection.selector,
        Model  = AgentSelectorUI,
        Input  = AgentSelectorInput,
        Result = AgentSelectorResult,
        Target = AgentPredictorWorker,
        title  = 'AGENTS SELECTOR'
    )

    def _get_contract(self, is_default: bool = True):
        return WorkerContract(

            defaults = AgentSelectorEnvelope(
                inputs = AgentSelectorInput(
                    data = self.sets.get('test.ml'),
                    predictor = self.get_target_worker()
                ),
                config = AgentSelectorEnvelopeCfg(
                    train_ml = self.sets.get('train.ml'),
                )
            ),
            
            datasets = DatasetBinding(fitted = {
                'stats.agents.selection' : self._processor.stats               
            })
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class SelectionPhase(WorkerPhaseBase):
    """
    AIG Flow Selection Phase Orchestrator.
    
    The final decision-making stage of the ML pipeline. This phase coordinates 
    the transition from a collection of trained agents to a single, validated 
    Champion Agent. It orchestrates a rigorous four-step validation process—
    Prediction, Comparison, Transition, and Selection—ensuring that the chosen 
    model is objectively superior, stable across data folds, and fully explainable.    
    """

    def __init__(self, ctx: ProjectContextProtocol):    
        super().__init__(ctx)
        self._register((
            AgentsPredictorWorker,
            AgentsComparatorWorker,
            AgentsTransitionWorker,
            AgentSelectorWorker
        ))

    def report(self, verbose: bool = None) -> None:
        """
        Final Model Governance Report.
        
        Produces a comprehensive, human-readable summary of the champion agent's 
        identity, performance benchmarks, and feature importance. This serves as 
        the primary documentation for stakeholders to approve the model for 
        production deployment.

        Parameters
        ----------
        verbose : bool = None
            Optional verbosity override; None → flow global context verbosity level.

        Returns
        -------
        None
            Triggers the visualization/logging of the Champion Agent report.
        """        
        AgentPredictorDashWaiter(self.ctx.setup).show(verbose)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬