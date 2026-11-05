"""
Animated Progress Indicators — Non-Blocking Visual Feedback.

This module provides non-blocking progress spinners for long-running
computations. It supports two rendering backends:

- **SpinnerHTML**: SVG-based spinner for Jupyter/IPython notebooks.
  Renders an animated spinning icon with optional elapsed-time
  display using IPython widgets.

- **SpinnerANSI**: Carriage-return spinner for plain terminals
  (CLI, scripts, non-notebook environments). Uses Unicode braille
  frames and ANSI escape codes.

Both backends implement the same `SpinnerBase` interface and
can be used interchangeably via the context-manager
(`with spinner.progress():`) or as a decorator.
"""

from ..deps.common import (
    sys, 
    time, 
    threading, 
    ABC, 
    abstractmethod, 
    contextmanager
)
from ..deps.display import  (
    wg, 
    display,
    clear_output
)
from ..deps.types import (
    Literal,
    Callable
)

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ANIMATED SPINNER - WORK IN PROGRESS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class SpinnerBase(ABC):
    """
    Abstract renderer interface for animated progress indicators.

    Defines the common lifecycle contract (`start` / `stop`)
    and provides two composable usage patterns:

    1. **Context manager**: `with spinner.progress(): ...`
       — wraps a block of code with animation.
    2. **Decorator**: `fx = spinner.with_progress()(fx)`
       — injects animation into a callable.

    Both patterns are exception-safe: the animation is
    guaranteed to stop (and cleanup to run) even if an
    exception is raised inside the wrapped block.

    Subclasses must implement `start` and `stop` with
    their environment-specific rendering (widgets, ANSI codes,
    etc.).
    """

    @abstractmethod
    def start(self) -> None:
        """Begin the visual indicator."""

    @abstractmethod
    def stop(self) -> None:
        """Stop and clean up. Optionally show a completion message."""

    @contextmanager
    def progress(self, verbose: bool = True):
        """
        Context manager for wrapping computations with a visual spinner.

        Usage::

            with spinner.progress():
                run_heavy_computation()

        Automatically handles the start/stop lifecycle, ensuring
        cleanup even if an exception occurs.

        Parameters
        ----------
        verbose : bool, default=True
            If True, start the animation. If False, the
            context is a no-op (for conditional suppression).
        """ 
        if verbose:
            self.start()
        try:
            yield
        finally:
            if verbose:
                self.stop()

    def with_progress(self, verbose: bool = True) -> Callable:
        """
        Return a decorator that injects progress animation into
        a target function.

        Parameters
        ----------
        verbose : bool, default=True
            If True, show the animation. If False, pass-through.

        Returns
        -------
        callable
            Decorator function.
        """
        def wrapper(fx, *args, **kwargs):
            if fx is None:
                return
            with self.progress(verbose=verbose):
                return fx(*args, **kwargs)
        return wrapper        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SpinnerHTML(SpinnerBase):
    """
    SVG-based animated spinner for Jupyter/IPython notebooks.

    Renders a spinning SVG icon with optional real-time elapsed
    timer via IPython widgets. The animation runs in a background
    daemon thread, keeping the main thread free for computation.

    Parameters
    ----------
    title : str, default='Process in progress'
        Display text shown alongside the spinner.
    show_timer : bool, default=False
        If True, display elapsed time (updated every 100 ms)
    show_done : bool, default=False
        If True, display a completion message (title + ✓) on
        stop. If False, clear the widget entirely.
    font_size : int, default=13
        Base font size in pixels (affects spinner size and
        text).
    title_first : bool, default=False
        If True, title appears left of the spinner;
        otherwise right.
        alongside the title.
    """
    def __init__(self, 
        title: str = 'Process in progress', 
        show_timer: bool = False, 
        show_done: bool = False,
        font_size: int = 13, 
        title_first: bool = False
    ):
        self._title       = title
        self._show_timer  = show_timer
        self._show_done   = show_done
        self._font_size   = font_size
        self._title_first = title_first

        self._output_widget = wg.Output()
        self._msg_widget    = wg.HTML()
        self._is_running    = False

    def _title_el(self, is_up: bool):
        """Resolve the title element position."""
        if not self._show_timer:
              t = wg.HTML(str(self._title))
        else: t = self._msg_widget

        if (
            (self._title_first and is_up) or
            (not self._title_first and not is_up)
        ):
            return t
        return wg.HTML()
    
    def _title_timer(self):
        """Background thread updating the elapsed-time display."""        
        start_time = time.time()

        while self._is_running:            
            elapsed = round(time.time() - start_time, 1)
            e_timer = f' | {elapsed}s'

            t = self._title + e_timer

            self._msg_widget.value = str(t)
            time.sleep(0.1)

    def _start_timer(self) -> None:
        if self._show_timer:
            threading.Thread(target=self._title_timer, daemon=True).start()

    def _build_body(self):
        """Assemble the spinner widget layout."""        
        font = wg.HTML(
            f'''<div style="font-size:{self._font_size}px; display:flex; 
            align-items:center; gap:10px; height:{self._font_size*2}px;">'''
        )
        svg  = wg.HTML(f'''
            <svg class="size-5 animate-spin" viewBox="0 0 48 48" 
             style="width: {self._font_size}px; height: {self._font_size}px; margin: 8px 4px 0 0;">
            <circle cx="24" cy="24" r="20" stroke="blue" stroke-width="5" fill="none" />
            <path fill="none" stroke="blue" stroke-width="4" stroke-linecap="round" d="M4 24a8 8 0 1 1 32 0" />
            </svg>
            <style>
                .animate-spin {{
                    animation: spin 1s linear infinite;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>            
            '''
        )
        return wg.HBox([font, self._title_el(True), svg, self._title_el(False)])

    def start(self):
        """
        Start the animation.

        Displays the output widget and renders the spinner
        layout. Launches the timer thread if enabled.
        """
        self._is_running = True
        self._start_timer()
        display(self._output_widget)
        with self._output_widget:
            display(self._build_body())

    def stop(self):
        """
        Stop the animation and clean up.

        Clears the widget output. If `show_done` is True,
        displays a completion message (title + ✓). If False,
        hides the widget entirely (leaves no blank space).
        """
        self._is_running = False
        with self._output_widget:
            clear_output(wait=True)
            if not self._show_done:
                self._output_widget.layout.display = 'none'  # clears potential leftover blank space
            else:
                html_done = f'<div style="font-size:{self._font_size}px;">{self._title} ✅</div>'
                display(wg.HTML(html_done))
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SpinnerANSI(SpinnerBase):
    """
    Carriage-return spinner for plain terminals (CLI).

    Uses Unicode braille characters and ANSI escape codes to
    render a smooth spinning indicator in the terminal. The
    animation runs in a background daemon thread.

    Thread-safe: all stdout writes are guarded by a lock to
    prevent frame interleaving with other output.

    Parameters
    ----------
    title : str, default='Process in progress'
        Display text shown alongside the spinner frames.
    show_timer : bool, default=False
        If True, display elapsed time alongside the title.
    show_done : bool, default=False
        If True, display a green checkmark (✓) on stop.
        If False, clear the line and leave a blank newline.
    """

    _FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    _CLEAR  = '\r\033[K'         # carriage-return + clear-to-EOL
    _HIDE   = '\033[?25l'        # hide cursor
    _SHOW   = '\033[?25h'        # show cursor
    _BOLD   = '\033[1m'
    _DIM    = '\033[2m'
    _RESET  = '\033[0m'
    _GREEN  = '\033[92m'

    def __init__(self, 
        title: str = 'Process in progress', 
        show_timer: bool = False, 
        show_done: bool = False,
        font_size:  int = None, 
        title_first: bool = None,
    ):
        self._title       = str(title)
        self._show_timer  = show_timer
        self._show_done   = show_done
        self._is_running  = False
        self._lock        = threading.Lock()

    def _frame_thread(self):
        """Background thread cycling through spinner frames."""
        start_time = time.time()
        i = 0
        while self._is_running:
            frame   = self._FRAMES[i % len(self._FRAMES)]
            elapsed = time.time() - start_time
            suffix  = f' | {self._DIM}{elapsed:.1f}s{self._RESET}' if self._show_timer else ''

            line = f'{self._BOLD}{frame}{self._RESET} {self._title}{suffix}'
            with self._lock:
                sys.stdout.write(f'{self._CLEAR}{line}')
                sys.stdout.flush()

            i += 1
            time.sleep(0.1)

    def start(self) -> None:
        """
        Start the animation.

        Hides the terminal cursor and launches the frame
        cycling thread.
        """        
        self._is_running = True
        sys.stdout.write(self._HIDE)
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._frame_thread, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop the animation and clean up.

        Joins the frame thread, clears the spinner line,
        optionally displays a completion message (green ✓),
        and restores the terminal cursor.
        """        
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        with self._lock:
            if self._show_done:
                msg = f'{self._GREEN}{self._title} ✓{self._RESET}'
            else:
                msg = ''
            sys.stdout.write(f'{self._CLEAR}{msg}')
            sys.stdout.write(self._SHOW)
            sys.stdout.flush()
        if not self._show_done:
            sys.stdout.write('')
            sys.stdout.flush()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# SPINNER HTML-ANSI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class SpinnerUI:
    """
    Factory for selecting the appropriate spinner backend.

    Resolves the correct spinner class (`SpinnerHTML` or
    `SpinnerANSI`) for the current execution environment.

    Parameters
    ----------
    UI : 'html' | 'ansi', default='html'
        Target environment. `'html'` for Jupyter/IPython,
        `'ansi'` for plain terminal/CLI.
    """

    @staticmethod
    def get_spinner(UI:Literal['html', 'ansi']='html'):
        """
        Return the spinner class for the specified environment.

        Parameters
        ----------
        UI : 'html' | 'ansi', default='html'
            Target rendering backend.

        Returns
        -------
        type[SpinnerBase]
            `SpinnerHTML` or `SpinnerANSI` class (not
            instance — instantiate with desired title/options).
        """        
        if UI == 'html':
            return SpinnerHTML
        else:
            return SpinnerANSI       
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬