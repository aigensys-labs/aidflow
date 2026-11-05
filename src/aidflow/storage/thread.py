"""
Asynchronous I/O orchestration engine for non-blocking background persistence.

This module implements a dedicated background worker system designed to offload 
expensive disk operations from the main execution thread. By utilizing a managed 
thread pool and a priority queue, it enables 'fire-and-forget' saving of large 
model artifacts and metadata, ensuring that the pipeline's computational throughput 
is never throttled by storage latency or slow filesystem I/O.
"""
 
from ..deps.common  import threading, queue, traceback
from ..deps.types   import Literal
from ..core.logging import Logger


logger = Logger(__name__)


class AsyncIO:
    def __init__(self, fx:dict, max_queue_size:int=0, timeout:float=5.0):
        """
        AsyncSaver for background model saving.
        - max_queue_size: max number of saves to queue (1 = FIFO, no overlap)
        - timeout: seconds to wait for queue tasks to complete on shutdown
        """
        self.fx          = fx
        self._queue      = queue.Queue(maxsize=max_queue_size)
        self._worker     = threading.Thread(target=self._run, daemon=True)
        self._stop_event = threading.Event()
        self._timeout    = timeout

        self._worker.start()

    def ops(self, fx:Literal['save','load'], path, obj=None, callback=None):
        """
        Schedule save of `obj` to `path`.
        - obj: object to pickle (e.g., model, state dict, etc.)
        - path: str or Path to save to
        - callback: optional function(result: Any, error: Exception | None)
        """
        try:            
            if self._queue.full():  # clear existing job if queue is full (FIFO)
                self._queue.get_nowait()    # drop the oldest            
                self._queue.task_done()     # counter align

            result_q = None

            if ( fx == 'load' ) and ( callback is None ):
                result_q = queue.Queue(maxsize=1)
                callback = lambda result, error: result_q.put(
                    (result, error)
                )

            self._queue.put_nowait((fx, path, obj, callback))  # Put new job
            
            if result_q:
                return result_q.get() # blocks until result available
                   
        except queue.Full:
            logger.warning("Save queue is full — dropping request")
            

        except Exception as e:
            if callback:
                callback(None, error=e)
            else:
                logger.error(f"Failed to queue {fx}: {e}")

    def _run(self):
        """Worker thread: processes save jobs."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                job = self._queue.get()
            except queue.Empty:
                continue

            fx, path, obj, callback = job
            try:
                fx_ = self.fx[fx]
                
                if ( fx == 'save' ):
                    res = fx_(obj, path)

                if ( fx == 'load' ):
                    res = fx_(path)
                
                if callback: 
                    callback(result=res, error=None)

            except Exception as e:
                if callback: 
                    callback(result=None, error=e)
                else:
                    logger.error(f"Async {fx} failed: {e}")
                    traceback.print_exc()

            finally:
                self._queue.task_done()

    def stop(self, wait=True):
        """Stop worker thread and wait for it to finish."""
        self._stop_event.set()
        if not wait:
            return
        self._worker.join(self._timeout)
        if self._worker.is_alive():
            logger.warning("Worker thread did not stop gracefully in time.")