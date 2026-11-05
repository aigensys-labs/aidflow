"""
Asynchronous execution engine providing non-blocking background processing 
and interactive UI synchronization.

This module implements a high-level concurrency wrapper designed to prevent 
the 'freezing' of user interfaces—particularly in Jupyter Notebooks and 
JupyterLab—during computationally intensive ML tasks. It manages the 
complex transition between threading for I/O-bound tasks and multiprocessing 
for CPU-bound training jobs.

Key operational capabilities:
1. Non-Blocking Execution: Decouples heavy ML workloads from the main execution 
   thread, allowing the pipeline to remain responsive while background 
   processes execute in parallel.

2. Intelligent Output Routing: Implements a sophisticated IPython kernel 
   header redirection system that anchors background output and animated 
   progress indicators to the original cell of origin, regardless of current 
   user focus.

3. Hybrid Concurrency Model: Dynamically switches between Python `threading` for 
   lightweight monitoring and `ProcessPoolExecutor` for heavy computation, 
   bypassing the GIL (Global Interpreter Lock) to maximize CPU utilization.

4. Lifecycle Synchronization: Uses threading Events and callbacks to manage 
   consequent pipeline steps, ensuring that dependent nodes only execute 
   once background prerequisites are successfully resolved.
   
5. Serialization Diagnostics: Includes a recursive `trace_pickle_error` engine 
   to debug complex object graphs, ensuring that all model states remain 
   serializable for cross-process communication and persistence.
"""

from ...deps.common     import  (
                                threading,
                                get_running_loop,
                                AbstractEventLoop,
                                ProcessPoolExecutor,
                                pickle
                                )
from ...deps.display    import  (
                                InteractiveShell,
                                IPythonKernel,
                                wg, 
                                get_ipython,
                                display, 
                                )
from ...deps.types      import  (
                                Any,
                                FuncType
                                )
from ..logging          import  Logger


logger = Logger(__name__)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# BACKGROUND PROCESSING BY NEW PROCESS / THREAD
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class BackgroundProcess:
    """
    Run task_fx process at the background to allow pipeline continuation while building UI.

    Process respects source of origin and display all callback output at the origin cell.    

    Parameters
    ----------
    callback: Function = None
        - callback function called once the main .run() fx finishes
        - callback fx design must accept the first arg as a result of the main function
           → callback(res)
        - callback optional second arg is provided when `call_args` is specified
           → callback(res, call_args)
        - `callback display output is shown always in the original cell where process was submitted`    

    call_args: tuple = None
        - Provides optinal second argument to the callback directly to avoid serialization issues.
        - callback(res, call_args), where the res is always provided as result from the main .run()

    new_process: bool = False
        - `new_process = False` → Uses a thread to watch a task that handles its own multiprocessing.
        - `new_process = True`  → Pushes the entire task into a separate process to avoid blocking Jupyter.

    daemon : bool = True
        If True, the thread is killed when the main thread exits.
        If False, the interpreter waits for the thread to complete
        before shutting down. Use False for threads doing final
        saves or critical cleanup.

    verbose: bool = False
        Shows animated waiting progress until main background process is finished.
    
    message: str = 'Task Background Processing'
        Custom message to show while waiting - active only with verbose=True
    
    show_timer: str = False
        Adds timer to the message while waiting - active only with verbose=True

    ipy_header : dict = None
        Cell id dict from ipy.kernel.get_parent() when widget can update dash with background processing.
    
    waiting_event: threading.Event = None
        threading.Event to synchronize pipeline's consequent steps.
    """

    class Event(threading.Event):
        """threading class implementing event objects.

        Events manage a flag that can be set to true with the set() method and reset
        to false with the clear() method. The wait() method blocks until the flag is
        true.  The flag is initially false.
        """        

    class _ResultError:
        def __init__(self, task_fx: FuncType, exception:str):
            self.fx = task_fx.__name__
            self.exception = exception

    def __init__(self, 
        callback: FuncType = None,
        call_args:     Any = None,
        new_process:  bool = False, 
        daemon:       bool = True,
        verbose:      bool = False, 
        message:       str = 'Task Background Processing',
        show_timer:   bool = False,
        ipy_header:   dict = None,
        waiting_event: threading.Event = None,
    ):
        self.callback    = callback
        self.call_args   = call_args
        self.new_process = new_process
        self.daemon      = daemon
        self.verbose     = verbose
        self.message     = message
        self.show_timer  = show_timer

        self.wait_event = waiting_event or self.Event()
        self._main_loop = self._get_main_thread()
        
        self._ipy_kernel        = self._get_ipython_kernel()
        self._ipy_orig_header   = ipy_header or self._get_ipython_header()
        self._output: wg.Output = wg.Output()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    @staticmethod
    def _get_ipython_kernel() -> IPythonKernel | None:
        """IPython kernel."""
        ipy: InteractiveShell = get_ipython()
        if ipy is None:
            return
        return ipy.kernel

    def _get_ipython_header(self) -> dict | None:
        """IPython Header=Cell ID in JN/JL."""
        if hasattr(self, '_ipy_kernel'):
            kernel = self._ipy_kernel 
        else: 
            kernel = self._get_ipython_kernel()
        if kernel is None: 
            return
        return kernel.get_parent()
    
    def _set_ipython_active_header(self, header:dict|None) -> None:
        """Assign requested cell header as active to ipy kernel."""
        if self._ipy_kernel is None: 
            return
        self._ipy_kernel.set_parent(self._ipy_kernel._parent_ident, header)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_main_thread() -> AbstractEventLoop | None:
        """Main running async loop / None for console."""
        try:    
            return get_running_loop()
        except: 
            None

    def _send_to_main_thread(self, callback:FuncType):
        """Pushes the execution to the Main Thread's next idle cycle."""
        if self._main_loop is not None:
            self._main_loop.call_soon_threadsafe(callback)
        else: 
            self._start_thread(callback)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def is_set(self) -> bool:
        """Return true if and only if the internal flag of *wait_event* (Event) is true."""
        return self.wait_event.is_set()
    
    def set_created(self) -> None:
        """Set the internal flag of *wait_event* (Event) to true.

        All threads waiting for it to become true are awakened. Threads
        that call wait() once the flag is true will not block at all.
        """        
        self.wait_event.set()

    def waiting_helper(self, *args, **kwargs) -> tuple[tuple, dict]:
        """Block until the internal flag of *wait_event* (Event) is true.
        Returns tuple of args, kwargs which were provided to the *run()* method.
        """        
        self.wait_event.wait()
        return args, kwargs
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _ui_waiting_start(self, message:str=None):
        """Show animated progress in the reserved display output."""
        from ..accessor import Spinner
        if not self.verbose:
            return
        self._al = Spinner(message or self.message, show_timer=self.show_timer)
        with self._output:            
            self._al.start()

    def _ui_waiting_stop(self):
        if not self.verbose:
            return
        with self._output:
            self._al.stop()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def run(self, task_fx, *args, **kwargs) -> None:
        """
        Run process at the background.
        When task_fx produces result, use `callback` to obtain it.

        When `task_fx is threading Event`, BP.run acts as a waiting helper:
        - idom → BackgroundProcess(my_callback, waiting_event = Event).run(Event)
        - new thread is created to wait for the Event by *waiting_helper()*
        - once the Event is ready, callback can consume args, kwargs provided to *run()*

        ```python
        def waiting_helper(self, *args, **kwargs) -> tuple[tuple, dict]:
            self.wait_event.wait()
            return args, kwargs        
        ```
        """
        if self._ipy_kernel is not None:
            display(self._output) # Reserve display spot in the notebook 

        self._ui_waiting_start()

        if task_fx is self.wait_event:
            self._run_new_thread(self.waiting_helper, *args, **kwargs)            
        elif not self.new_process:
            self._run_new_thread(task_fx, *args, **kwargs)    
        else: 
            self._run_new_process(task_fx, *args, **kwargs)    
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _logger(msg: str, e:Exception) -> None:
        logger.error(msg)
        logger.error(e)

    def _verify_result(self, result):
        if not isinstance(result, self._ResultError):
            return
        msg = f'Provided function `{result.fx}` exited with an error state. Please run it directly to debug.'
        self._logger(msg, result.exception)
        raise RuntimeError(msg + f'\nError:\n{result.exception}.')
        
    def _callback_exception(self, e:Exception):
        if self.call_args is None:
            a,b = '`one arg only`', ''
        else: 
            a = 'the first arg'
            b = '\n► Ensure that callback has the second argument for defined `call_args`.'
        m = (
             '► Callback function exited with an error state.\n'
            f'► Please verify that the callback has {a}'
            f'for the result from the main function provided to .run().{b}'            
        )
        if self._ipy_kernel is None:
            self._logger(m, e)
            return
        with self._output:
            self._logger(m, e)

    def _callback_pipe(self, result) -> None:
        
        def callback():
            if self.call_args is None:
                  self.callback(result)
            else: self.callback(result, self.call_args)

        def callback_no_ui():
            try:
                callback()
            except Exception as e:
                self._callback_exception(e)
        
        def callback_on_ui():
            # 1 Store current active notebook cell position
            ipy_current_header = self._get_ipython_header()
            try:
                # 2 Activate original cell where process started
                self._set_ipython_active_header(self._ipy_orig_header)
                with self._output:
                    try:
                        self._output.clear_output(wait=True)
                        callback()
                    except Exception as e:
                        self._callback_exception(e)
            finally:
                # 3 Restore orig. active notebook cell position
                self._set_ipython_active_header(ipy_current_header)
        
        if self._ipy_kernel is None:
              callback_no_ui()
        else: callback_on_ui()

    def _callback_handler(self, result) -> None:

        self._ui_waiting_stop()
        self._ui_waiting_start('Done √ Displaying results as soon as the Kernel is free')

        def callback():       
            self._ui_waiting_stop()   
            self._verify_result(result)
            if self.callback is None: 
                return                          
            self._callback_pipe(result)
        
        self._send_to_main_thread(callback)
        self.set_created()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _start_thread(self, fx:FuncType) -> None:
        threading.Thread(target=fx, daemon=self.daemon).start()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _run_new_thread(self, task_fx, *args, **kwargs) -> None:
        """Uses a thread to watch a task that handles its own multiprocessing."""
        
        def watcher():
            try:     
                result = task_fx(*args, **kwargs)
            except Exception as e:  
                result = self._ResultError(task_fx, e)
            finally: 
                self._callback_handler(result)

        self._start_thread(watcher)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _run_new_process(self, task_fx, *args, **kwargs) -> None:
        """Pushes the entire task into a separate process to avoid blocking Jupyter."""
        pool = ProcessPoolExecutor(max_workers=1)
        
        def watcher():
            try:
                future = pool.submit(task_fx, *args, **kwargs)
                result = future.result()
            except Exception as e:
                self._logger(f'Processing Exception', e)
                result = self._ResultError(task_fx, e)
            finally:
                self._callback_handler(result)
                try: 
                    pool.shutdown(wait=False)
                except: 
                    pass

        self._start_thread(watcher)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


def trace_pickle_error(obj, path="root", visited=None):
    """Dev helper to find source of pickle save issues."""
    if visited is None:
        visited = set()
    
    # Prevent infinite loops from circular references
    if id(obj) in visited:
        return
    visited.add(id(obj))

    try:
        pickle.dumps(obj)
        # If it succeeds, this branch is completely safe.
        print(f"🔍 {path}safe!")
        return 
    except Exception:
        # If it fails, we print the current path, then dig deeper.
        print(f"🔍 Digging into: {path} (Type: {type(obj).__name__})")
        
        # Check dictionaries
        if isinstance(obj, dict):
            for k, v in obj.items():
                trace_pickle_error(v, f"{path}['{k}']", visited)
                
        # Check standard objects / classes
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                trace_pickle_error(v, f"{path}.{k}", visited)
                
        # Check lists and tuples
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                trace_pickle_error(v, f"{path}[{i}]", visited)
        else:
            # We reached the bottom of the rabbit hole!
            print(f"\n🚨 CULPRIT FOUND! 🚨")
            print(f"Path: {path}")
            print(f"Type: {type(obj)}")
            print(f"Value: {repr(obj)[:150]}\n")