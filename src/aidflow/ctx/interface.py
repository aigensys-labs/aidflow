"""
Execution Context — Domain Pipeline Orchestration Layer.

This module provides the primary user-facing API for executing
the AIDflow pipeline. It wraps the underlying worker ecosystem
in a semantic, phase-based interface:

    ctx.flow.data.data_loader()
    ctx.flow.encoding.woe_encoder()
    ctx.flow.training.agent1_predictor()
    ctx.flow.selection.champion_predictor()

Pipeline Phases
---------------
The pipeline is organized into four sequential phases. Each
phase must be completed before the next can execute:
    - `data` → `encoding` → `training` → `selection`

Execution Modes
---------------
- **Full flow**: `ctx.flow.process('all')` - runs all phases in order.
- **Single phase**: `ctx.flow.process('training')` - runs one phase 
  (verifies all prior phases are complete).
- **Single worker**: `ctx.flow.training.agent2_predictor()` 
  - runs one specific step.

All execution runs within the context scope defined by
`ProjectContext` (setup parameters, project paths, state).
"""

from ..deps.types           import Literal

from ..core.structure       import AIGFLOW, REG
from ..core.foundation      import AigParam, AigStop
from ..core.base.worker     import WorkerBase
from ..core.protocol        import ProjectContextProtocol
from ..core.accessor        import Format

from ..data.interface       import DataPhase
from ..data.interface       import EncodingPhase   
from ..learner.interface    import TrainingPhase
from ..selector.interface   import SelectionPhase

from ..display.rgb          import RgbANSI
from ..display.verbosity    import Verbosity


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DOMAIN INTERFACE CONTEXT & PIPELINE ORCHESTRATION LAYER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ExecutionContext:
    """
    Execution Context — Pipeline Orchestration & Phase Execution.

    The primary API surface for executing the AIDflow pipeline.
    Provides phase-based access to all workers and a unified
    `process` method for sequential execution.
    
    Executes `workers` by domain-specific pipeline phase:
    - `data` → `encoding` → `training` → `selection`

    Notes
    -----
    - Automated execution of all pipeline steps or a selected phase via `ctx.flow.process()`
    - Execution runs within a defined context scope (`ctx.setup`), supplied by `ProjectContext`
    - Pipeline steps are sequence-dependent - e.g., the `encoding` phase cannot be executed prior to\
    the `data` phase

    Pipeline Execution Order
    ------------------------

    <b>1. data </b>
    <br />────────────────────

    0. data_loader
    1. fe_cleaner
    2. fe_stabilizer
    3. obs_stabilizer

    <b>2. encoding </b>
    <br />────────────────────
    
    4. num_encoder
    5. cat_encoder
    6. ixn_encoder
    7. woe_encoder
    8. woe_stabilizer
    9. data_encoder

    <b>3. training </b>
    <br />────────────────────
    
    10. agent1_predictor
    11. agent2_predictor
    12. agent3_predictor
    13. agent4_predictor

    <b>4. selection </b>
    <br />────────────────────
    
    14. agents_predictor
    15. agents_comparator
    16. agents_transition
    17. agent_selector
    18. champion_predictor

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

    Run the whole flow within the defined context:

    ```python
    ctx.flow.process('all', verbose = 0)
    ```

    Run selected flow phase within the defined context:
    
    ```python
    ctx.flow.process('data', verbose = 2)
    ```
        
    Run selected single worker within the defined context:

    ```python
    data_loader = ctx.flow.data.data_loader()
    ...
    ixn_encoder = ctx.flow.encoding.ixn_encoder()
    ...
    agent2_predictor = ctx.flow.training.agent2_predictor()
    ...
    agent_selector = ctx.flow.selection.agent_selector()
    ```
    """
        
    def __init__(self, ctx: ProjectContextProtocol):
        self.data       = DataPhase(ctx)
        self.encoding   = EncodingPhase(ctx)
        self.training   = TrainingPhase(ctx)
        self.selection  = SelectionPhase(ctx)

        self._processor = ExecutionProcessor(self, ctx.setup.verbose)
    
    def __repr__(self):
        return ExecutionRepr.repr()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_worker(self, model: str) -> WorkerBase:
        """
        Returns instance of a Worker from any phase.

        Parameters
        ----------
        model : str
            Model ID in standard provided by ctx.models.

        Returns
        -------
        WorkerBase
            Instance of a Worker under actual context state.
        """        
        return self._processor.get_worker(model)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def process(self, 
        phase: Literal['all', 'data', 'encoding', 'training', 'selection'] = 'all',
        verbose: bool | Verbosity | None = None
    ) -> None:
        """
        Executes workers for the chosen phase within the defined context scope.

        Parameters
        ----------
        phase : Literal['all', 'data', 'encoding', 'training', 'selection']
            Process workers by the chosen phase or run the whole flow by 'all'.

        verbose : bool | Verbosity | None = None
            Verbosity level of workers; None → context default value.
        """
        return self._processor.process(phase, verbose)

    @staticmethod
    def help() -> None:
        """Displays help description of the Execution Context."""
        ExecutionRepr.repr()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ExecutionProcessor:
    """
    Internal execution engine for phase orchestration.

    Resolves phase names to worker collections, enforces
    sequence dependency, and dispatches workers in order.
    """

    MAP = {
        'data'           : AIGFLOW.Data,
        'encoding'       : AIGFLOW.Encoding,
        'training'       : AIGFLOW.Training,
        'selection'      : AIGFLOW.Selection,
    }

    def __init__(self, 
        execution: ExecutionContext, 
        verbose: bool|int|Verbosity = None
    ):
        self._set_domain(execution)
        self._verbose = verbose

    def _set_domain(self, exe: ExecutionContext) -> None:
        self.domain = {
            w: exe.data for w in AIGFLOW.Data.values()
        } | {
            w: exe.encoding for w in AIGFLOW.Encoding.values()
        } | {
            w: exe.training for w in AIGFLOW.Training.values()
        } | {
            w: exe.selection for w in AIGFLOW.Selection.values()
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_worker(self, model:str) -> WorkerBase:
        """Returns instance of a Worker from any phase."""
        domain = self.domain[model]
        return getattr(domain, model, None)

    def process(self, 
        phase: Literal['all', 'data', 'encoding', 'training', 'selection'] = 'all',
        verbose: bool | Verbosity | None = None
    ) -> None:
        """Executes workers for the chosen phase within the defined context scope."""
        if phase != 'all':
            self._verify(phase)
            self._process(self.MAP[phase], verbose)
            return
        
        for workers in self.MAP.values():
            self._process(workers, verbose)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _excluded() -> list:
        return [AIGFLOW.Selection.champion_agent]
    
    def _process(self, workers: dict, verbose: bool | Verbosity) -> None:
        excluded = self._excluded()
        for w in workers.values():
            if w in excluded:
                continue
            worker = self.get_worker(w)
            verbose = AigParam.di(verbose, self._verbose)
            ExecutionRepr.header(worker, verbose)
            worker(verbose = verbose)

    def _verify(self, phase: str) -> None:
        m = list(self.MAP)
        if phase not in m:
            raise self._raise(f'Invalid phase choice, available options ► {', '.join(m)}')
        v = m.index(phase)
        for i in range(v):
            phase = self.MAP[m[i]]
            for model in phase.values():
                worker = self.get_worker(model)
                if not worker.is_fitted:
                    raise self._raise(
                'Pipeline steps are sequence-dependent → e.g., the "encoding" ' 
                'phase cannot be executed prior to the "data" phase'
                )

    @staticmethod
    def _raise(msg:str) -> None:
        AigStop.custom(RgbANSI.bold_red('AIG SAFETY')+f' ► {msg}', bold=False, red=False)                
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
                
class ExecutionRepr:
    """Display formatting for execution context representation."""

    @staticmethod
    def header(worker: WorkerBase, verbose: bool | Verbosity) -> None:
        """Print the phase header before worker execution."""
        if not verbose:
            return
        model = worker.config.model
        heads = {phase.value(): name for name, phase in ExecutionProcessor.MAP.items()}
        if model in heads:
            if model != AIGFLOW.Data.loader:
                print()
            Format.title(
                title = f'{heads[model]} phase'.upper(), 
                style = Format.StyleTitle(margin_t=0, margin_b=0)
            )

    @classmethod
    def repr(cls) -> str:
        """Generate the full pipeline execution overview."""
        flow = ['AIDFlow Pipeline Execution Order:']
        s = 0
        for i, (phase, steps) in enumerate(AIGFLOW.items(), start=1):
            if phase not in REG:
                continue
            flow.append( RgbANSI.bold(f'\n{i}. {phase.lower()}') )
            flow.append('─'*20)
            for step in steps.values():
                flow.append(f'{s}. {step}')
                s+=1
        return cls._desc(False) + '\n\n' + '\n'.join(flow)

    @classmethod
    def _desc(cls, show=True) -> str|None:
        h = (
            f'► Automated execution of all pipeline steps or a selected phase via {RgbANSI.bold(".process()")}'
            '\n► Execution runs under a defined context scope (ctx.setup) → provided by ProjectContext'
            '\n► Run individual workers via ctx.flow.{phase}.{worker_name} or ctx.workers.{worker_name}'
            f'\n► Use {RgbANSI.bold("ctx.flow.process(phase='all', verbose=False)")} to run the full pipeline, or run phases in sequence:'
            "\n  1. 'data' → 2. 'encoding' → 3. 'training' → 4. 'selection'"
            '\n► Pipeline steps are sequence-dependent → e.g., encoding phase cannot be executed prior to data phase'
        )
        if not show:
            return h
        print(h)    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬