"""
AIDflow Production Exporter.

Exports a fitted champion agent to production-ready deployment
artifacts. The exporter serializes the self-contained scoring
object (model + preprocessing + feature lineage) and generates
an orchestration layer for scheduled inference.

Export Targets
--------------
- **Pickle**: Full champion agent as a single `.pkl` file.
  Self-contained — holds the model, WOE encoder, feature
  cleaner, and all feature mappings internally.
- **Airflow DAG**: A ready-to-deploy Python DAG that
  orchestrates the scoring pipeline:
  load raw data → preprocess → score → push to warehouse.

The Exported Agent Contract
---------------------------
The exported pickle must expose:
- `preprocess(data)` — raw → ML-ready (cleaning + encoding).
- `transform(data)` — ML-ready → P(class=1) probabilities.
- `is_fitted` — boolean flag confirming scorable state.

This is the complete interface required by the Airflow DAG.
No other attributes are accessed at runtime.
"""

from ..deps.common      import  Path, pickle, textwrap
from ..deps.types       import  Any
from ..core.base        import  ModelUIBase
from ..core.base.worker import  WorkerBase
from ..core.logging     import  Logger
from ..core.factory     import  Dataclass
from ..storage.folder   import  Folder


logger = Logger(__name__)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PRODUCTION → EXPORT CONFIG
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class ExportConfig:
    """
    Production export configuration.

    Attributes
    ----------
    dag_id : str, default='aidflow_scoring'
        Airflow DAG identifier. Must be a valid Python identifier.
    schedule : str, default='0 2 * * *'
        Airflow cron expression. Default: nightly at 02:00.
    model_path : str, default='/models/champion.pkl'
        Path where the pickle will be available in the production
        environment (mounted volume, S3, HDFS).
    raw_data_loader : str, default='scoring.load_raw_data'
        Dotted path to the function that loads raw production data.
    result_sink : str, default='scoring.push_scores'
        Dotted path to the function that writes scores to the
        warehouse.
    retries : int, default=3
        Number of retries per task on failure.
    retry_delay_min : int, default=5
        Delay between retries (minutes). Exponential backoff
        is enabled.
    owner : str, default='data-science'
        DAG owner for Airflow alerting.
    tags : tuple[str, ...], default=('aidflow', 'production', 'scoring')
        Airflow DAG tags for UI filtering.
    """
    dag_id:          str  = "aidflow_scoring"
    schedule:        str  = "0 2 * * *"
    model_path:      str  = "/models/champion.pkl"
    raw_data_loader: str  = "scoring.load_raw_data"
    result_sink:     str  = "scoring.push_scores"
    retries:         int  = 3
    retry_delay_min: int  = 5
    owner:           str  = "data-science"
    tags: tuple[str, ...] = ("aidflow", "production", "scoring")
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PRODUCTION → EXPORTER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ProductionExporter:
    """
    Production deployment exporter.

    Serializes a fitted champion agent and generates the
    orchestration layer (Airflow DAG) for scheduled scoring.

    Parameters
    ----------
    agent : WorkerBase | ModelUIBase
        A fitted champion agent exposing `score(data)`, and `is_fitted`.
    config : ExportConfig | None, default=None
        Export configuration. Defaults to nightly schedule.

    Examples
    --------
    ```python
    from aidflow.production import ProductionExporter, ExportConfig
    config   = ExportConfig()
    exporter = ProductionExporter(agent, config)
    paths    = exporter.export_all('./deployment/')
    # paths['pickle']      → deployment/champion.pkl
    # paths['airflow_dag'] → deployment/dag.py
    ```        
    """
    _SIGNATURE = ('score', 'is_fitted')

    def __init__(self, 
        agent: WorkerBase | ModelUIBase, 
        config: ExportConfig | None = None
    ):
        self._verify(agent)
        self._set_agent(agent)
        self._config = config or ExportConfig()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def export_all(self, output_dir: Path = '') -> dict[str, Path]:
        """
        Export all production artifacts to the specified directory.

        Parameters
        ----------
        output_dir : Path
            Target directory. Created if it does not exist.

        Returns
        -------
        dict[str, Path]
            Mapping of artifact name → file path.
            Keys: `'pickle'`, `'airflow_dag'`.
        """
        path = self._mk_dir(output_dir)

        return {
            'pickle'      : self.export_pickle(path / 'champion.pkl'),
            'airflow_dag' : self.export_airflow_dag(path / 'dag.py'),
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def export_pickle(self, path: Path = '') -> Path:
        """
        Serialize the champion agent to a pickle file.

        The resulting file is self-contained — it holds the
        model, preprocessing chain, and feature lineage.
        No external AIDflow imports are required at load time.

        Parameters
        ----------
        path : Path
            Target file path.

        Returns
        -------
        Path
            The written file path.
        """
        path = self._mk_dir(path)

        with path.open('wb') as f:
            pickle.dump(self._agent, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info('Pickle ► %s (%.1f MB)', path, path.stat().st_size / 1_048_576)
        return path
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def export_airflow_dag(self, path: Path = '') -> Path:
        """
        Generate a production-ready Airflow DAG file.

        The DAG orchestrates:
        - load raw data 
        - preprocess and score by aidflow package
        - push results to warehouse

        Parameters
        ----------
        path : Path
            Target file path for the generated DAG.

        Returns
        -------
        Path
            The written file path.
        """
        path = self._mk_dir(path)
        path.write_text(self._render_dag(), encoding='utf-8')

        logger.info('Airflow DAG ► %s (dag_id=%s)', path, self._config.dag_id)
        return path
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _mk_dir(path: str | Path) -> Path:
        path, file = Folder.path_file(path)
        path = Folder.create(path)
        return path / file
        
    def _set_agent(self, agent: WorkerBase | ModelUIBase) -> None:
        """Sets ModelBase as the inference engine."""
        if isinstance(agent, WorkerBase):
            agent = agent.model
        self._agent = agent.model
            
    def _verify(self, agent: Any) -> None:
        """Validate that the agent is in a scorable state."""
        if not isinstance(agent, (WorkerBase, ModelUIBase)):
            raise ValueError(
                'AIG SAFETY ► Agent must be signed by WorkerBase or ModelUIBase.'
            )
        for m in self._SIGNATURE:
            if getattr(agent, m, None) is None:
                raise ValueError(
                    f'AIG SAFETY ► Agent must expose "{m}". '
                    f'Got: {type(agent).__name__}'
                )
        if not agent.is_fitted:
            raise ValueError('AIG SAFETY ► Agent is not fitted. Call fit() prior export.')
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _render_dag(self) -> str:
        """Render the Airflow DAG Python source code."""
        c = self._config
        tags = ', '.join(f'"{t}"' for t in c.tags)

        return textwrap.dedent(f'''
            """
            AIDflow Production Scoring DAG.
            DAG ID: {c.dag_id}
            Auto-generated — regenerate via ProductionExporter.export_airflow_dag()
            """

            import logging
            import pickle
            import importlib
            from datetime import datetime, timedelta

            from airflow import DAG
            from airflow.operators.python import PythonOperator

            logger = logging.getLogger(__name__)


            # ── CONFIG ───────────────────────────────────────────────────────

            MODEL_PATH      = "{c.model_path}"
            RAW_DATA_LOADER = "{c.raw_data_loader}"
            RESULT_SINK     = "{c.result_sink}"


            # ── TASK ─────────────────────────────────────────────────────────

            def _resolve(dotted_path: str):
                mod, _, fn = dotted_path.rpartition(".")
                return getattr(importlib.import_module(mod), fn)


            def scoring_pipeline(**kwargs) -> None:
                """Load model → preprocess → score → push."""
                with open(MODEL_PATH, "rb") as f:
                    agent = pickle.load(f)

                raw = _resolve(RAW_DATA_LOADER)()
                logger.info("Loaded %%s observations", len(raw))

                scores = agent.score(raw)

                _resolve(RESULT_SINK)(scores)
                logger.info("Pipeline complete.")


            # ── DAG ──────────────────────────────────────────────────────────

            default_args = {{
                "owner"                      : "{c.owner}",
                "depends_on_past"            : False,
                "retries"                    : {c.retries},
                "retry_delay"                : timedelta(minutes={c.retry_delay_min}),
                "retry_exponential_backoff"  : True,
            }}

            with DAG(
                dag_id       = "{c.dag_id}",
                schedule     = "{c.schedule}",
                start_date   = datetime(2026, 1, 1),
                catchup      = False,
                default_args = default_args,
                tags         = [{tags}],
            ) as dag:

                PythonOperator(
                    task_id         = "execute_scoring",
                    python_callable = scoring_pipeline,
                )

        ''').lstrip()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬