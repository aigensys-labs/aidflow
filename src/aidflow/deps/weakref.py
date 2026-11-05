"""
Weak-Reference Lazy Access Primitives.

Pure-Python utility providing zero-argument callables that resolve
attributes via weak references, eliminating closures and their
associated memory retention.

Capabilities:
    - **Memory-Safe Binding**: Stores only `weakref.ref` objects.
      If the referenced object is garbage collected, the callable
      returns a configurable default (typically `None`) rather than
      raising — safe for long-lived dashboard and repr bindings.
    - **No Closures**: All state is in `__slots__` instance
      attributes. No cell variables, no `__defaults__`, no
      accidental reference retention.
    - **Dot-Path Resolution**: Supports nested attribute paths
      (e.g., `'result.data.columns'`) with short-circuit
      evaluation on missing attributes.

Intended Use:
    - Binding a "live" reference to a pipeline worker's result
      without preventing the worker (or its parent context) from
      being garbage collected when the pipeline is disposed.
    - Constructing lazy dashboards and repr displays that evaluate
      at display time, reflecting the current state of the object
      graph.
      
Placement Rationale:
    This module resides in `deps` (rather than `utils`) because
    it is merged with native AIDflow types (`FuncType`, callable
    protocols) at the type-definition layer. It is a pure `getattr`
    wrapper with no external dependencies, making it safe to import
    at the lowest level of the dependency graph without creating
    circular imports.
"""

import weakref
from typing import Any, Callable


class WeakGetter:
    """
    Memory-safe lazy attribute resolution via weak references.

    Provides callable objects that dynamically evaluate an attribute
    path on one or two weakly-referenced targets. This eliminates the
    need for closures (which would create a hard reference and prevent
    garbage collection) while still providing "live" access to the
    current state of the object graph.

    All inner classes use `__slots__` to minimize per-instance
    memory overhead, as these objects may be created in large numbers
    (e.g., one per column in a dashboard binding).

    Classes:
        Attr: Unary lazy attribute accessor.
        PairAttr: Binary lazy attribute evaluator.

    Static Methods:
        getter: Resolve a dot-notation attribute path on an object.

    Example:
        >>> # Create a lazy reference to a worker's result:
        >>> live_result = WeakGetter.Attr(worker_instance, 'result.data')
        >>>
        >>> # Later (worker still alive): returns the current data
        >>> df = live_result()
        >>>
        >>> # After worker is garbage collected: returns None (safe)
        >>> del worker_instance
        >>> gc.collect()
        >>> live_result()  # → None (no exception)
    """

    class Attr:
        """
        Zero-argument callable that lazily resolves an attribute on a
        weakly-referenced object.

        When called, dereferences the weak reference and either:
        - Returns the object itself (if no attribute path is specified).
        - Calls the object (if it is callable and no attribute path).
        - Resolves a dot-notation attribute path on the object.

        If the referenced object has been garbage collected, returns
        the configured default (typically `None`) without raising.

        Memory Guarantees:
            - Stores only a `weakref.ref` (no hard reference).
            - Uses `__slots__` (no `__dict__` overhead).
            - No closures, no `__defaults__`, no cell variables.

        Parameters:
            obj (Any): The target object to weakly reference.
            attr (str): Optional dot-notation attribute path
                (e.g., `'result.data'`). If None, the object
                itself (or its call result) is returned.
            default (Any): Value returned when the object is
                garbage collected or the attribute is missing.
                Defaults to None.

        Example:
            >>> # Lazy access to a worker's fitted result:
            >>> getter = WeakGetter.Attr(worker, 'model.result.data')
            >>> df = getter()  # Resolves worker.model.result.data
            >>>
            >>> # Lazy access to the object itself:
            >>> live_obj = WeakGetter.Attr(worker)
            >>> obj = live_obj()  # Returns the worker instance
            >>>
            >>> # Lazy call (e.g., bound method):
            >>> live_fn = WeakGetter.Attr(lambda: compute())
            >>> result = live_fn()  # Calls the lambda
        """
        __slots__ = ('_ref', '_attr', '_default')

        def __init__(self, obj: Any, attr: str = None, default: Any = None):
            """
            Initialize the weak-reference attribute accessor.

            Parameters:
                obj (Any): The target object to reference weakly.
                    Must be an object that supports weak references
                    (most standard Python classes do).
                attr (str): Optional dot-notation path to resolve
                    on the object (e.g., `'a.b.c'`). If None,
                    the object itself is returned (or called if callable).
                default (Any): Fallback value when the object is
                    dead or the attribute is missing. Defaults to None.

            Raises:
                TypeError: If `obj` does not support weak references
                    (e.g., instances of classes with `__slots__`
                    that don't include `__weakref__`).
            """
            self._ref = weakref.ref(obj)
            self._attr = attr
            self._default = default

        def __call__(self) -> Any:
            """
            Resolve the lazily-referenced attribute.

            Dereferences the weak reference and applies the configured
            resolution strategy:
            - If the object is dead: returns `_default`.
            - If no attribute path is set: returns the object itself
              (or calls it if it is callable).
            - If an attribute path is set: resolves it via
              `WeakGetter.getter`.

            Returns:
                Any: The resolved attribute value, the object itself,
                    or the default if the object is no longer alive.
            """
            obj = self._ref()

            if obj is None:
                return self._default

            if self._attr is None:
                if callable(obj):
                    return obj()
                return obj   
            
            return WeakGetter.getter(obj, self._attr, self._default)

    class PairAttr:
        """
        Zero-argument callable that evaluates a binary function over
        dynamically resolved attributes of two weakly-referenced objects.

        Resolves one attribute on each of two objects and passes the
        results to a user-supplied evaluation function. This enables
        lazy, memory-safe computation over pairs of live objects
        (e.g., comparing two workers' results at display time).

        Memory Guarantees:
            - Stores two `weakref.ref` objects (no hard references).
            - The evaluation function is stored as a direct reference
              (assumed to be a stateless callable like `operator.sub`
              or a module-level function).
            - No closures, no `__defaults__`, no cell variables.

        Parameters:
            pair (tuple[object, object]): The two target objects
                to weakly reference.
            attrs (tuple[str, str]): Dot-notation attribute paths to
                resolve on each object respectively.
            eval_fx (Callable[[Any, Any], Any]): A binary function
                that receives the two resolved values and returns
                the final result (e.g., `lambda a, b: a - b`).
            default (Any): Fallback value returned when either
                object has been garbage collected or an attribute
                is missing. Defaults to None.

        Example:
            >>> # Lazily compute the difference between two workers' scores:
            >>> diff = WeakGetter.PairAttr(
            ...     pair=(worker_a, worker_b),
            ...     attrs=('result.score', 'result.score'),
            ...     eval_fx=lambda a, b: a - b,
            ...     default=0.0
            ... )
            >>> delta = diff()  # Resolves both and computes the difference
            >>>
            >>> # After either worker is GC'd: returns 0.0 (safe default)
        """
        __slots__ = ('_ref1', '_ref2', '_attr1', '_attr2', '_eval_fn', '_default')

        def __init__(
            self,
            pair:       tuple[object, object],
            attrs:      tuple[str, str],
            eval_fx:    Callable[[Any, Any], Any],
            default:    Any = None,
        ):
            """
            Initialize the binary weak-reference evaluator.

            Parameters:
                pair (tuple): Two target objects to reference weakly.
                attrs (tuple): Two attribute paths (one per object).
                eval_fx (Callable): Binary function to apply to
                    the two resolved values.
                default (Any): Fallback when either object is dead
                    or an attribute is missing. Defaults to None.

            Raises:
                TypeError: If either object in `pair` does not
                    support weak references.
            """
            self._ref1 = weakref.ref(pair[0])
            self._ref2 = weakref.ref(pair[1])
            self._attr1 = attrs[0]
            self._attr2 = attrs[1]
            self._eval_fn = eval_fx
            self._default = default

        def __call__(self) -> Any:
            """
            Resolve both attributes and evaluate the binary function.

            Dereferences both weak references. If either object has
            been garbage collected, returns the default immediately.
            Otherwise, resolves each attribute path and passes the
            results to the evaluation function.

            Returns:
                Any: The result of `eval_fx(v1, v2)`, or the
                    default if either object is no longer alive.
            """
            o1, o2 = self._ref1(), self._ref2()
            if o1 is None or o2 is None:
                return self._default

            v1 = WeakGetter.getter(o1, self._attr1, self._default)
            v2 = WeakGetter.getter(o2, self._attr2, self._default)
            
            return self._eval_fn(v1, v2)
    

    @classmethod
    def getter(cls, obj: object, attr: str, default: Any = None) -> Any:
        """
        Resolve a dot-notation attribute path on an object.

        Walks the attribute chain segment by segment (split on `'.'`).
        If any intermediate attribute is missing, returns the default
        value immediately (short-circuit) rather than raising
        `AttributeError`.

        This is the core resolution engine used by both `Attr` and
        `PairAttr`.

        Parameters:
            obj (object): The root object to start the attribute
                resolution from.
            attr (str): A dot-notation path (e.g., `'a.b.c'`).
                Each segment is resolved via `getattr`.
            default (Any): The fallback value returned if any
                attribute in the chain is missing. Defaults to None.

        Returns:
            Any: The final resolved value, or `default` if any
                attribute in the path does not exist.

        Example:
            >>> # Simple attribute:
            >>> WeakGetter.getter(obj, 'name')
            >>>
            >>> # Nested path:
            >>> WeakGetter.getter(obj, 'result.data.columns')
            >>>
            >>> # Missing attribute returns default:
            >>> WeakGetter.getter(obj, 'nonexistent.deep.path', default='N/A')
            # → 'N/A'
        """
        attrs = attr.split('.')        
        for a in attrs:
            obj = getattr(obj, a, default)
            if obj is default:
                return             
        return obj   