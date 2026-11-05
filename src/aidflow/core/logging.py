"""
AIDflow — Logging Infrastructure.

The library **emits** structured log records. The user's
application decides **where** they go (console, file, both,
or nowhere).

Design Principles
-----------------
- **Namespaced**: All loggers live under the `aidflow` tree
  (e.g., `aidflow.learner.agents.agent1_predictor`), enabling selective
  filtering by module.
- **Structured output**: Optional JSON formatting for machine
  ingestion (ELK, CloudWatch, Datadog, etc.).
- **Idempotent setup**: `Logger.setup()` is safe to call
  multiple times — it clears previous handlers before re-attaching.
"""

from ..deps.common import (
    logging,
    Logger as _Logger,
    threading,
    sys, 
    time,
    re,
    json,
    wraps,
    contextmanager
)
from ..deps.types import (
    Any,
    Literal,
    Callable
)
from .factory import (
    DictClass,
    DictDataclass
)
from ..utils.builtins import (
    Misc
)

PACKAGE = __name__.split(".")[0]


class LOG(DictClass):
    R = 'logging_logger'


class LogSignature:
    """
    Unique Log Run Signature Generator.

    Produces a cryptographically unique, time-ordered identifier
    for each logging session. Used as a correlation key across
    critical pipeline components (Project, Model, Dashboard)
    to link related log entries from different layers into a
    single traceable run.

    Format
    ------
    `{name}_{YYYYMMDD_HHMMSS_microseconds}_{8-char-uuid}`

    Examples
    --------
    >>> LogSignature.get()
    '20261007_194523_128473_a1b2c3d4'

    >>> LogSignature.get('project_run')
    'project_run_20261007_194523_128473_a1b2c3d4'

    Properties
    ----------
    - **Time-ordered**: Sortable by string comparison.
    - **Collision-safe**: UUID suffix prevents collisions even
      for runs initiated in the same microsecond.
    - **Human-readable**: Timestamp prefix allows visual
      identification without parsing.
    - **Component-scoped**: Optional name prefix associates
      the signature with a specific pipeline layer
      (e.g., `'model_...'`, `'dash_...'`).
    """

    @staticmethod
    def get(name: str=None) -> str:
        """
        Generate a unique log run signature.

        Combines a high-resolution timestamp (microsecond
        precision) with a random UUID suffix to produce a
        sortable, collision-safe identifier for the current
        logging session.

        Parameters
        ----------
        name : str | None, default=None
            Optional component prefix. Associates the
            signature with a specific pipeline layer
            (e.g., `'project'`, `'model'`, `'dashboard'`).
            If None, returns a bare signature without prefix.

        Returns
        -------
        str
            Unique signature string.

            With name:  `'model_20261007_194523_128473_a1b2c3d4'`
            Without:    `'20261007_194523_128473_a1b2c3d4'`

        Notes
        -----
        - Microsecond precision in the timestamp ensures
          monotonic ordering for runs within the same second.
        - The 8-character UUID suffix provides probabilistic
          uniqueness (1 in 4.3 billion collision chance) for
          scenarios where two signatures are generated in the
          same microsecond (e.g., parallel pipeline branches).
        - Safe for use in filenames, log headers, and
          cross-component correlation keys.

        Examples
        --------
        >>> LogSignature.get()
        '20261007_194523_128473_a1b2c3d4'

        >>> LogSignature.get('obs_stabilizer')
        'obs_stabilizer_20261007_194523_128473_a1b2c3d4'
        """        
        ts  = Misc.timestamp("%Y%m%d_%H%M%S_%f")
        uid = Misc.uuid(8)
        sig = f'{ts}_{uid}'
        if name:
            return f'{name}_{sig}'
        return sig
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class JsonFormatter(logging.Formatter):
    """
    Structured JSON log formatter.

    Converts each `LogRecord` into a single-line JSON object
    with standard fields (`ts`, `level`, `logger`, `msg`) 
    plus any custom `extra` fields and exception traces.

    Suitable for machine ingestion (ELK, CloudWatch, Datadog,
    or any JSON-line parser).
    """    
    _SKIP = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)))

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", record.created),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, val in vars(record).items():
            if key not in self._SKIP and val is not None:
                entry[key] = val
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Logger:
    """
    AIDflow logging API — setup, retrieval, and contextual logging.

    Provides three capabilities:
    1. **Setup**: One-call configuration of handlers, levels,
       and formatting (user entrypoint only).
    2. **Get**: Namespaced logger retrieval under the
       `aidflow` tree.
    3. **Context**: A context manager for action-scoped
       logging with automatic timing and error capture.

    Parameters
    ----------
    name : str | type
        A string name or model instance / class (used for the logger namespace).
    """
    PACKAGE = '_root_package'

    def __init__(self, name: str | type):
        self.name = name

    @wraps(_Logger.debug)
    def debug(self, msg: str, *args, **kwargs) -> None:
        logger = self.get_logger(self.name)
        logger.debug(self._format(msg), *args, **kwargs)

    @wraps(_Logger.info)
    def info(self, msg: str, *args, **kwargs) -> None:
        logger = self.get_logger(self.name)
        logger.info(self._format(msg), *args, **kwargs)

    @wraps(_Logger.warning)
    def warning(self, msg: str, *args, **kwargs) -> None:
        logger = self.get_logger(self.name)
        logger.warning(self._format(msg), *args, **kwargs)

    @wraps(_Logger.error)
    def error(self, msg: str, *args, **kwargs) -> None:
        logger = self.get_logger(self.name)
        logger.error(self._format(msg), *args, **kwargs)

    @wraps(_Logger.critical)
    def critical(self, msg: str, *args, **kwargs) -> None:
        logger = self.get_logger(self.name)
        logger.critical(self._format(msg), *args, **kwargs)

    @staticmethod
    def _format(msg: str) -> str:
        DASH = re.compile(r'^\s*─{2,}\s*$', re.MULTILINE)
        ANSI = re.compile(r'\x1b\[[0-9;]*m')
        msg  = DASH.sub('', str(msg)).replace('\n', '').strip()
        return ANSI.sub('', msg)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def setup(cls,
        level: int | str = logging.INFO,
        file: str = None,
        stream: bool = True,
        fmt: Literal["text", "json"] = "text",
        owner = 'user'
    ) -> None:
        """
        One-call logging configuration.

        Call this **once** in the user's application entrypoint.
        Do NOT call inside library code. Safe to call multiple
        times (idempotent — clears previous handlers before
        re-attaching).

        Parameters
        ----------
        level : int | str, default=logging.INFO
            Minimum log level. Accepts an integer (`20`)
            or a string (`"INFO"`, `"DEBUG"`, etc.).
        file : str | None, default=None
            Path to a log file. If None, no file handler is attached.
        stream : bool, default=True
            Attach a console (stderr) handler.
        fmt : "text" | "json", default="text"
            Output format:
            - `"text"`: Human-readable line format.
            - `"json"`: Structured JSON (one object per
              line) for machine ingestion.
        """
        root = logging.getLogger(PACKAGE)
        root.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper()))        

        for h in root.handlers[:]:
            if getattr(h, "_owner", None) == owner:
                root.removeHandler(h)   

        formatter = JsonFormatter() if fmt == "json" else logging.Formatter(
            "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        if stream:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(formatter)
            h._owner = owner
            root.addHandler(h)

        if file:
            h = logging.FileHandler(file, encoding="utf-8")
            h.setFormatter(formatter)
            h._owner = owner
            root.addHandler(h)

            if owner == cls.PACKAGE:
                h.stream.write(100*'═' + '\nAIDFLOW CONTEXT SET\n' + 100*'═' + '\n')
                h.stream.flush()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    @classmethod
    def get_logger(cls, name: str | type) -> logging.Logger:
        """
        Return a namespaced logger under the `aidflow` tree.

        Parameters
        ----------
        name : str | type
            Logger name. Accepts:
            - A string: used directly → e.g., use __name__ when in the module body.
            - A class: uses its module path → e.g., `MyClass` → `"aidflow.training.agent1"`.

        Returns
        -------
        logging.Logger
            A child logger under the `aidflow` package
            namespace.

        Examples
        --------
        >>> log = Logger.get_logger(MyAgent)
        >>> log.info("Fitted on 50k rows")
        """
        name = cls._get_name(name)
        return logging.getLogger(name)

    @staticmethod
    def _get_name(name: str | type = None) -> str:
        """Resolve a logger name from a string, class, or instance."""
        if isinstance(name, type):
            return name.__module__
        if not isinstance(name, str):
            return type(name).__module__
        return name        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def background_logger(sync: threading.Event, func: Callable) -> None:
        """Logging function func runs after `sync` event is ready."""
        from .execution import BackgroundProcess        
        BackgroundProcess(func, waiting_event=sync, daemon=False).run(sync)

    @contextmanager
    def model_logger(self, 
        fx_choice = None, 
        args: tuple | DictDataclass = None, 
        kwargs: dict = None
    ):
        """
        Context manager for action-scoped logging with timing.

        Wraps a model operation with:
        - Start log (INFO) with action details.
        - Success log (INFO) with elapsed duration.
        - Failure log (ERROR) with exception + elapsed duration.

        Parameters
        ----------
        fx_choice : RegGet.Choice | None
            The fit choice (strategy identifier + config).
            If None, the action is labeled `'update'`.
        args : tuple | None
            Positional arguments (for logging context).
        kwargs : dict | None
            Keyword arguments (for logging context).

        Raises
        ------
        Exception
            Any exception from the wrapped block is logged
            (with traceback) and re-raised.

        Examples
        --------
        >>> with Logger.model_logger(MyAgent, fx_choice, args, kwargs):
        ...     agent.fit(data)
        """        
        logger = self.get_logger(self.name)
        t0     = _ModelUtils.get_timing()
        a, prm = _ModelUtils.get_action_details(fx_choice, args, kwargs)
        try:
            logger.info(f"{a} started | {prm}")
            yield
            logger.info(f"{a} done ({_ModelUtils.get_timing(t0)})")

        except:
            logger.exception(f"{a} failed ({_ModelUtils.get_timing(t0)})")
            raise
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class _ModelUtils:
    """Model Processor logging info extensions."""

    @staticmethod
    def get_timing(t0:float=None) -> float | str:
        """
        Capture the current performance timestamp or compute elapsed duration.
        """
        t = time.perf_counter()
        if t0 is None:
            return t
        return f'{round(t - t0, 3)}s'

    @classmethod
    def get_action_details(cls,
        fx_choice, args: tuple | DictDataclass = None, kwargs: dict = None
    ) -> tuple[str, str]:
        """
        Resolve the action label and logging extra fields
        from the model processor signature.

        Parameters
        ----------
        fx_choice : RegGet.Choice | None
            The fit choice. If None, action is `'update'`.
        args : tuple | None
            Positional arguments (included in log extra).
        kwargs : dict | None
            Keyword arguments (included in log extra).

        Returns
        -------
        tuple[str, str]
            (action_label, extra_fields_for_logging)
        """
        from .registry import RegGet

        if fx_choice is None:
            return (
                'update', 
                f'args = {args}, kwargs = {kwargs}'
            )
        
        fx_choice: RegGet.Choice = fx_choice
        fx_params = cls._get_params(args, kwargs)

        return (
            'fit', 
            f'fx_choice = {fx_choice.name}, {fx_params}, fx_config = {fx_choice.config}'
        )

    @classmethod
    def _get_params(cls, inputs: DictDataclass, params: DictDataclass) -> str:
        """Model inputs and params formatting."""        
        d, p = {}, {}
        cls._add_dataclass(d, inputs)
        cls._add_dataclass(p, params, d)
        return f'inputs = {d}, fx_params = {p}'

    @classmethod
    def _add_dataclass(cls, d: dict, params: DictDataclass, inputs: dict = None) -> None:        
        inputs = inputs or {}
        for k,v in params.items():
            val = cls._get_value(v)
            if k in inputs or val in inputs.values():
                continue
            d[k] = val

    @staticmethod
    def _get_value(param: Any) -> Any:
        if isinstance(param, float):
            return float(param)
        if isinstance(param, int):
            return int(param)
        if isinstance(param, str):
            return str(param)
        if hasattr(param, 'shape'):
            return param.shape
        if hasattr(param, '__len__'):
            return len(param)
        return type(param).__name__
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬