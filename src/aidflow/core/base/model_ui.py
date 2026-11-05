"""
Dynamic Execution Orchestrator and Model-WorkerUI Integration Layer.

This module implements the runtime bridge between AIDflow's backend ML models 
and their interactive Jupyter representations. It serves as the primary 
execution wrapper (`ModelUIRunner`), coordinating the transition from user-defined 
hyperparameters to actual model fitting and data transformation.

The core of this module is a 'Lifecycle Pipeline' that manages the sequential 
deployment of UI elements and execution hooks:
1. State Synchronization: Monitors dynamic data changes in the notebook, 
   triggering automatic UI refreshes when upstream data dependencies are updated.

2. Deterministic Execution Flow: Orchestrates an 8-stage lifecycle (defined 
   in `UIParams`) that sequences pre-execution guards, main fitting logic, 
   and post-fit result visualization.

3. Polymorphic Model Wrapping: Leverages `_ModelRunner` to wrap any `ModelBase` 
   implementation, enabling standardized 'fit' and 'transform' behavior across 
   diverse ML algorithms while maintaining a consistent user interface.

4. Reactive UI Feedback: Integrates real-time validation and progress tracking, 
   ensuring that the user is notified of mandatory data requirements before 
   heavy computation begins.
"""

from ...deps.common         import  (
                                    gc, 
                                    contextmanager
                                    )
from ...deps.data           import  pDF, pSe
from ...deps.display        import  (
                                    wg, 
                                    display, 
                                    clear_output
                                    )
from ...deps.plots          import  go
from ...deps.data_view      import  itables_show
from ...deps.types          import  (
                                    FuncType, 
                                    Callable,
                                    Any
                                    )

from ..parameters           import  Parameters
from .model                 import  (
                                    FIT,
                                    ModelUIRunnerBase,
                                    ModelBase, 
                                    ModelInput,
                                    ModelEnvelope,
                                    ModelResult
                                    )
from ..factory              import  (
                                    DictDataclass, 
                                    Dataclass,
                                    DictClass
                                    )
from ..foundation           import  (
                                    AigParam, 
                                    Dynamic
                                    )
from ..accessor             import  UI, Format
from ..display              import  (
                                    PanelConstructor,
                                    Confirm,
                                    Display
                                    )
from ..validation.data      import  DataValidation
from ..io.io                import  FITTED
from .fitted                import  FittedResult


class LBL(DictClass):
    UPDATE = '_is_with_custom_update_8'


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FIT → Standard fit-transform approach extended with interactive UI elements
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class UIParams:
    """
    Lifecycle configuration schema of .fit() method.

    This dataclass defines the 'choreography' of the UI, specifying which 
    functions are executed and displayed at each stage of the fit-transform 
    process (Stages 1-8). It allows developers to inject custom UI logic—such 
    as validation checks or result plots—without modifying the core execution engine.
    """    
    fit_transform_pipeline:      bool     = True
    fit_transform_inputs_link:   str      = None
    args_kwargs_to_verify_na:    dict     = None
    parent_data_check_update:    FuncType = None
    parent_has_update_FX_method: bool     = False
    # ─────────────────────────────────────────────────────
    gui_static_before_confirm_1: FuncType = None
    gui_static_after_confirm_2:  FuncType = None
    gui_dynamic_before_fit_3:    FuncType = None
    # ─────────────────────────────────────────────────────
    dynamic_to_args_kwargs_FX_4: FuncType = None
    # ─────────────────────────────────────────────────────
    gui_static_w_fitted_5:       FuncType = None
    gui_dynamic_w_fitted_6:      FuncType = None
    gui_static_w_fitted_7:       FuncType = None
    # ─────────────────────────────────────────────────────
    custom_update_fitted_8:      FuncType = None
    gui_dynamic_w_fitted_8:      FuncType = None
    # ─────────────────────────────────────────────────────
    custom_control_widget:       FuncType = None
    # ─────────────────────────────────────────────────────
    button_desc:    str  = 'CONFIRM'
    update_desc:    str  = 'CUSTOM UPDATE'
    is_button_last: bool = False
    output_wait:    bool = True
    show_data:      bool = None
    data_sample:    int  = None
    show_ui:        bool = True
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class _WgSession:

    def __init__(self, 
        registry: dict[int, dict[str, list]],
        snapshot: dict[int, dict[str, set|None]],
        ui_param: dict[int, dict[str, UIParams|None]],
        instance: "_ModelRunner | ModelUIRunner"
    ):
        self._registry = registry
        self._snap_reg = snapshot
        self._prm_reg  = ui_param
        self.model     = instance.model.MODEL                
        self._wg_id    = None

    def __len__(self):
        return sum(v for v in self._view().values())

    def __repr__(self):
        models = ", ".join(f"'{k}':{v}" for k, v in self._view().items())
        return f"WidgetLifecycle[{models}]"

    def _view(self) -> dict[str, int]:
        reg = self._registry.values()
        if not reg:
            return {}

        def u(x:tuple[dict]) -> dict:
            reg = {}
            for d in x:
                for model, val in d.items():
                    if model not in reg:
                        reg[model] = []
                    if isinstance(val, list):
                        reg[model] = reg[model] + val
                    elif isinstance(val, dict):
                        for v in val.values():
                            reg[model] = reg[model] + v            
            return reg

        def l(v:list) -> int:
            val = []
            for x in v:
                if x not in val:
                    val.append(x) 
            return len(val)

        return {model: l(val) for model, val in u(reg).items()}
    
    def _snapshot(self) -> None:
        _w = wg.widgets.widget
        self._set(self._snap_reg, set(_w._instances))

    def _acquire(self, ui_params: UIParams | None = None) -> None:
        _w = wg.widgets.widget
        snap = self._get(self._snap_reg)
        if snap is None:
            raise RuntimeError("call snapshot() first")

        new = set(_w._instances) - snap
        self._set(self._registry, [_w._instances[w] for w in new if w])

        if ui_params is None:
            return
        
        self._set(self._prm_reg,  ui_params)
    
    def _teardown_figure(self, w) -> None:
        """Clear Plotly callbacks; if any were registered."""
        if not (
            w is not None and 
            ( isinstance(w, go.FigureWidget) or
              getattr(w, "_model_module", None) == "plotlywidget" )
        ):
            return

        # 1. Clear Figure-level callbacks
        for fig_cb_attr in (
            "_on_relayout_callbacks", 
            "_on_restyle_callbacks", 
            "_on_edits_callbacks"
        ):
            cb = getattr(w, fig_cb_attr, None)
            if cb is not None and hasattr(cb, "clear"):
                cb.clear()

        # 2. Clear Trace-level callbacks
        if hasattr(w, "data"):
            for trace in w.data:
                for trace_cb_attr in (
                    "_on_click_callbacks",
                    "_on_hover_callbacks",
                    "_on_select_callbacks",
                    "_on_deselect_callbacks",
                ):
                    cb = getattr(trace, trace_cb_attr, None)
                    if cb is not None and hasattr(cb, "clear"):
                        cb.clear()

    def _teardown_widget(self, w) -> None:
        """
        Recursively tears down ipywidgets containers and purges Plotly FigureWidget
        trace and layout callback vectors to eliminate closure leaks.
        """
        if w is None:
            return

        # Clear standard ipywidgets button handlers
        if hasattr(w, "_click_handlers"):
            w._click_handlers.callbacks.clear()

        # Detach Traitlet observers
        if hasattr(w, "unobserve_all"):
            try:
                w.unobserve_all()
            except Exception:
                pass
        elif hasattr(w, "_observers"):
            w._observers.clear()

        # Clear go Figure
        self._teardown_figure(w)

        # Unmount frontend Comm channel and release from registry
        try:
            w.close()
        except Exception:
            pass

    def _release(self, ui_params: UIParams | None = None) -> None:
        """
        1.
        When ipywidgets renders UI components in a cell output, ipykernel 
        maintains a backend communication bridge (kernel.comm_manager.comms). 
        All previous comms are in the RAM even when cell is re-run.
        
        2.
        When bound methods / function hold self, the entire instance object 
        graph is still locked in the RAM → release links manually.
        """
        # 1
        for w in self._pop(self._registry, []):
            self._teardown_widget(w)

        # 2        
        p = self._pop(self._prm_reg, None)
        if isinstance(p, UIParams) and ( ui_params is not p ):            
            Dataclass.set_by_type(p, FuncType)
            
        gc.collect()

    def _cycle(self, reg_id:int, wg_id:int, ui_params: UIParams|None = None):
        self._set_ids(reg_id, wg_id)
        self._snapshot()
        try:
            yield
        finally:
            self._release(ui_params) # close widgets after yield makes UI update smooth
            self._acquire(ui_params)

    @contextmanager
    def figure(self, reg_id: int, wg_id: int = None):
        """Manages the go figure widget lifecycle for a flow MODEL step."""
        yield from self._cycle(reg_id = reg_id, wg_id = wg_id)

    @contextmanager
    def main(self, ui_params: UIParams = None, wg_id: int = None):
        """Manages the widget lifecycle for a flow MODEL step."""
        yield from self._cycle(reg_id = -1, wg_id = wg_id, ui_params = ui_params)
            

    def _set_ids(self, reg_id:int, wg_id:int|None) -> None:
        self._reg_id = reg_id
        self._wg_id  = wg_id

    def _set(self, reg: dict, val) -> None:
        if self._reg_id not in reg:
            reg[self._reg_id] = {}

        reg = reg[self._reg_id]

        if self._wg_id is None:
            reg[self.model] = val

        else:
            if reg.get(self.model) is None:
                reg[self.model] = {}
            reg[self.model][self._wg_id] = val

    def _get(self, reg: dict, default=None) -> list|set|UIParams|None:
        r:dict = reg.get(self._reg_id)
        if r is None:
            return default
        
        m = r.get(self.model, default)
        
        if ( self._wg_id is None ) or ( m is default ):
            return m
        
        return m.get(self._wg_id, default)

    def _pop(self, reg: dict, default) -> None:
        r:dict = reg.get(self._reg_id)
        if r is None:
            return default
        
        if self._wg_id is None:            
            return r.pop(self.model, default)
        
        m:dict = r.get(self.model)
        if m is None:
            return default
        return m.pop(self._wg_id, default)

class _WgLifecycle:
        
    def __init__(self):
        self._registry: dict[int, dict[str, list]] = {}
        self._snapshot: dict[int, dict[str, set|None]] = {}
        self._ui_param: dict[int, dict[str, UIParams|None]] = {}

    def __get__(self, instance, owner=None):
        return _WgSession(self._registry, self._snapshot, self._ui_param, instance)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class _ModelRunnerFx:
    """Dynamic wrapper around main (parent) fit()/transform()/update()."""

    def __init__(self,
        model: ModelBase, 
        params: UIParams, 
        choice: "ModelUIChoice", 
        verbose: bool,
        show_data: bool, 
        data_sample: int
    ):
        self.model       = model
        self.prm         = params
        self.choice      = choice
        self.verbose     = verbose
        self.show_data   = show_data
        self.data_sample = data_sample

        self.fit_sig  = AigParam.signature(self.model.fit).values()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _set_fx_choice(self, kwargs, fx_choice_prm='fx_choice'):
        """FX fit of the targeted fx → Connected with fx_choice param in parent fit fx."""
        if ( self.choice is not None ) and ( fx_choice_prm in self.fit_sig ):
            kwargs[fx_choice_prm] = self.choice.value

    def _set_verbose(self, kwargs, verbose_prm='verbose'):
        """Analyze available verbose params sources and set to kwargs if applicable."""
        verbose, self._is_gui = AigParam.verbose_std(kwargs, self.verbose, verbose_prm)
        if verbose_prm in self.fit_sig:
            kwargs[verbose_prm] = verbose
    
    def _set_mandatory_kwargs(self, kwargs) -> dict:
        """Set all mandatory kwargs for target fit fx."""
        self._set_fx_choice(kwargs)
        self._set_verbose(kwargs)
        return kwargs

    @property
    def is_gui(self):
        if not self.prm.show_ui:
            return False
        return self._is_gui
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _extend_transform_kw(self, kwargs:dict) -> dict:
        if not self.prm.fit_transform_inputs_link:
            return kwargs
        return kwargs | {
            FIT.TRANSFORM: self.prm.fit_transform_inputs_link
        }

    def _fit_transform(self, *args, **kwargs):
        if self.prm.fit_transform_pipeline:
            kw = self._extend_transform_kw(kwargs)
            return self.model.fit_transform(*args, **kw)
        return self.model.fit(*args, **kwargs).fitted
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _get_fx_params(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Extends fit() parametrization by Model-Fitter mapped params by OptionUI."""
        argsd, kwargsd = Dynamic.inject_params(self.prm.dynamic_to_args_kwargs_FX_4, args, kwargs)
        return Dynamic.params(argsd, kwargsd)

    def fit_in_console(self, args: tuple, kwargs: dict) -> ModelResult:
        """Console run dynamic wrapper around main (parent) fit()/transform()."""
        argsd, kwargsd = self._get_fx_params(args, kwargs)
        return self._fit_transform(*argsd, **kwargsd)
        
    def fit_in_ui(self, args:tuple, kwargs:dict, is_main_run:bool
    ) -> tuple[ModelResult, tuple, dict]:
        """UI run dynamic wrapper around main (parent) fit()/transform()/update()."""
        
        # (4) args, kwargs extension by dynamic params
        self._set_fx_choice(kwargs) # UI choice update
        argsd, kwargsd = self._get_fx_params(args, kwargs)
        
        if ( not is_main_run ) and ( self.prm.parent_has_update_FX_method ):                
            fitted = self.model.update_transform(*argsd, **kwargsd)            
        else: 
            fitted = self._fit_transform(*argsd, **kwargsd) 
            
        self._show_sample(fitted, self.prm)

        return fitted, argsd, kwargsd    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _show_sample(self, fitted: ModelBase | ModelResult, params: UIParams):     
        """Fit show sample design."""
        to_show = AigParam.di(params.show_data,   self.show_data)
        sample  = AigParam.di(params.data_sample, self.data_sample)
                
        if isinstance(fitted, DictDataclass):
            view = fitted.value()
        else: 
            view = fitted[0]

        if (
               (not to_show) 
            or (not sample) 
            or (not isinstance(view, (pDF,pSe)))
        ):
            return
        
        d = view.sample(sample, random_state=42)
        l = lambda: itables_show(d)
        s, f = '', Format.splitter(show=False)
        to_show = [s,f,s,l,s]
        Display.construct(to_show)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class _ModelRunner:
    """
    Wrapper around base Model class pipe: fit - transform - update:
    Adds UI choices and pass to a Worker's Model for training given the dynamic User Choices.
    """
    fitted    = FittedResult()
    _wg_cycle = _WgLifecycle()
    _d_ids    = None
    _is_init  = True


    def __init__(self, 
        model: ModelBase, 
        params: UIParams, 
        choice: "ModelUIChoice", 
        verbose: bool,
        show_data: bool, 
        data_sample: int,
        wg_id: int
    ):
        self.model       = model
        self.prm         = params
        self.choice      = choice
        self.verbose     = verbose
        self.show_data   = show_data
        self.data_sample = data_sample
        self.wg_id       = wg_id

        self.i = 0
        self._fx = _ModelRunnerFx(model, params, choice, verbose, show_data, data_sample)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _set_fitted(self, fitted: ModelResult):
        """Sets fitted result to the proxy decorator."""
        self.fitted = fitted

    def _set_ar_kw(self, ar:tuple, kw:dict) -> None:
        """Sets args-kwargs passed to the fit()."""        
        self.args = ar
        self.kwargs = self._fx._set_mandatory_kwargs(kw)

    def verify_mandatory_data(self) -> None:
        """
        Verify that input key dynamic data to fit fx are not empty.
        As widgets catches raised Exceptions, we must separate it from the update().
        """
        DataValidation.gui_dynamic_args(self.args, self.kwargs, self.prm.args_kwargs_to_verify_na) 
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_data_inputs(data: Any) -> list:
        if not isinstance(data, (list, tuple)):
            data = [data]
        data_  = []
        extend = lambda x: data_.extend(list(x.values()))
        for d in data:
            if isinstance(d, ModelEnvelope):
                extend(d.config)
                d = d.inputs
            if isinstance(d, (ModelInput, dict)):
                extend(d)
            else:
                data_.append(d)
        return data_

    @staticmethod
    def _get_id(d: Any) -> int:
        if not ( isinstance(d, pDF) and (FITTED.ID in d.attrs) ):
            return id(d)
        return d.attrs[FITTED.ID]
    
    def _is_parent_data_update(self, parent_data_fx: FuncType) -> bool:
        """
        Checks dynamic data update by user action of the previous step in JN.
        parent_data_fx must return current data retrieved at the time of the execution.
        """
        data  = Dynamic.data(parent_data_fx, *self.args, **self.kwargs) 
        data_ = self._flatten_data_inputs(data)
        
        if self._d_ids is None:
            self._d_ids = [id(None)]*len(data_)

        d_ids  = [self._get_id(d) for d in data_]
        update = any(d_id != self._d_ids[i] for i, d_id in enumerate(d_ids))

        if update: 
            self._d_ids = d_ids

        return update 

    def _is_cell_rerun(self) -> bool:
        """Ensures if User re-run JN cell the UI is reconstructed."""
        if self._is_init:
            self._is_init = False
            self._x = True 
            return True
        self._x = False
        return False

    def _is_dynamic_data_update(self) -> bool:
        """Dynamic Data Update Check ► Data update in the previous step."""

        update_cell = self._is_cell_rerun()

        if self.prm.parent_data_check_update is None:
            return update_cell

        # NOTE _is_parent_data_update() must run during init
        update_extras = [False]
        parent_data_fx = self.prm.parent_data_check_update
        
        # list structure: [0 → update data fx, 1+ → extra conditions as fx...]
        if isinstance(parent_data_fx, list):                                                    # ► Allows UI refresh for multiple events - e.g. (a) input data and (b) gui options changes
            parent_data_fx = self.prm.parent_data_check_update[0]                               # ► (a) Main input ◄ update check has to be provided at the first place
            update_extras  = [Dynamic.data(u) for u in self.prm.parent_data_check_update[1:]]   # ► (b) Extra update checks for parameters can be provided on following places
        
        update = self._is_parent_data_update(parent_data_fx)

        return any([update, *update_extras])    
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_outputs(self):
        outputs = [
            '1_2', 
            '3', 
            '4_fx', 
            '5', '6', '7', 
            '1_2_', 
            '8_e', '8_fx', '8'
        ]
        
        self._outputs = {}
        for o in outputs:
            self._outputs[o] = wg.Output()

    def _display_output(self, oid:str, *fx) -> None:
        if any([(f is not None) or (f is True) for f in fx]):
            display(self._outputs[oid])
    # ─────────────────────────────────────────────────────────────────────────────────────────

    def _set_main_wg(self, ui_pipeline: Callable):
        self._main_wg = Confirm(
            widget = self.prm.custom_control_widget, 
            output = self._outputs['4_fx'],
            desc   = self.prm.button_desc.upper(), 
            wait   = self.prm.output_wait
        ).get_button(ui_pipeline)

    def _gui_display_button(self, oid:str, at_top=True):
        gate = lambda x: not x if at_top else x
        if not gate(self.prm.is_button_last):
            return
        with self._outputs[oid]:
            if self._main_wg is not None:
                display(self._main_wg) 
    # ─────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _gui(fx, args, kwargs):  
        """Dynamic UI elements processor."""
        if fx is None:
            return  
        to_show = Dynamic.data(fx, *args, **kwargs)
        if to_show is not None:
            to_show = Dynamic.params(to_show)[0]                
            Display.construct(to_show)  

    def _gui_display_x(self, oid:str, fx, args, kwargs):
        if fx is None:
            return  
        with self._outputs[oid]:
            clear_output(wait=True)
            self._gui(fx, args, kwargs)

    def _gui_static_display_0(self, oid:str):
        with self._outputs[oid]:
            clear_output(wait=True)
            self._gui(self.prm.gui_static_before_confirm_1, self.args, self.kwargs)
            self._gui_display_button(oid, at_top=True)
            self._gui(self.prm.gui_static_after_confirm_2, self.args, self.kwargs)
                
    def _update_fitted(self, *args, **kwargs) -> None:
        """Parent FIT Update functionality propagated with dedicated `output_8`."""
        u = {}
        if self.prm.custom_update_fitted_8 is not None:
            with self._outputs['8_fx']:
                clear_output(wait=True)
                self._set_fitted(
                    self.prm.custom_update_fitted_8(*args, **kwargs)
                )
            u = {LBL.UPDATE: True}

        with self._wg_cycle.figure(8, self.wg_id):
            self._gui_display_x('8', self.prm.gui_dynamic_w_fitted_8, args, kwargs | u)    
    # ─────────────────────────────────────────────────────────────────────────────────────────

    def _run_fx(self, is_main_run:bool): 
        with self._outputs['4_fx']: 
            clear_output(wait=True)

            fitted, argsd, kwargsd = self._fx.fit_in_ui(self.args, self.kwargs, is_main_run)  
            self._set_fitted(fitted)

            return argsd, kwargsd    
        
    def _ui_pipeline(self) -> None:
        """
        UI Design ► Blocks respect data updates.

        **init run / input data update:**

        (1-2) UI static elements around CONFIRM button   
        (3) Dynamic UI elements before fit   
        ─────────────────────────────────────────────────────────────   
        (4) FX.fit → run FX & args, kwargs extension by dynamic params
        ─────────────────────────────────────────────────────────────   
        (5) UI static elements with fitted data   
        (6) Dynamic update of UI elements w fitted output   
        (7) UI static elements with fitted data
        ─────────────────────────────────────────────────────────────   
        (1-2-end) control button/widget at the UI bottom   
        ─────────────────────────────────────────────────────────────   
        (8) update fitted → FX.fit → in UI static element

        
        **Applying changes, clicking CONFIRM:**

        (3) Dynamic UI elements before fit   
        ─────────────────────────────────────────────────────────────   
        (4) FX.fit → args, kwargs extension by dynamic params   
        ─────────────────────────────────────────────────────────────   
        (6) Dynamic update of UI elements w fitted output   
        (8) UI dynamic elements with (updated) fitted data
        """
        ar_kw = self.args, self.kwargs

        if self._is_dynamic_data_update():
            self.verify_mandatory_data()

            with self._wg_cycle.figure(3, self.wg_id):
                self._gui_display_x('3', self.prm.gui_dynamic_before_fit_3, *ar_kw)
                ar_kw_d = self._run_fx(True)

            self._gui_display_x('5', self.prm.gui_static_w_fitted_5,  *ar_kw_d)

            with self._wg_cycle.figure(6, self.wg_id):
                self._gui_display_x('6', self.prm.gui_dynamic_w_fitted_6, *ar_kw_d)

            self._gui_display_x('7', self.prm.gui_static_w_fitted_7,  *ar_kw_d)
            self._gui_display_button('1_2_', at_top=False)
        
            self._update_fitted(*ar_kw_d)
        
        else:  
            self.verify_mandatory_data()

            with self._wg_cycle.figure(3, self.wg_id):
                self._gui_display_x('3', self.prm.gui_dynamic_before_fit_3, *ar_kw)
                ar_kw_d = self._run_fx(False)

            with self._wg_cycle.figure(6, self.wg_id):
                self._gui_display_x('6', self.prm.gui_dynamic_w_fitted_6, *ar_kw_d)

            with self._wg_cycle.figure(8, self.wg_id):
                self._gui_display_x('8', self.prm.gui_dynamic_w_fitted_8, *ar_kw_d)

                        
    def fit(self, *args, **kwargs) -> None:        
        """
        Fits parent fx based on User Choice in UI → Allows dynamic visual and data links across JN.
        """        

        self._set_ar_kw(args, kwargs)

        # ─────────────────────────────────────────────────────────────────────────────────────────
        # ► FIT Verbosity Fork ◄
        # ─────────────────────────────────────────────────────────────────────────────────────────
        # (A) Console
        if not self._fx.is_gui:
            self.verify_mandatory_data()
            self._set_fitted(
                self._fx.fit_in_console(self.args, self.kwargs)
            )
            return

        # ─────────────────────────────────────────────────────────────────────────────────────────                
        # ► (B) UI CONSTRUCT ◄
        # ─────────────────────────────────────────────────────────────────────────────────────────
        with self._wg_cycle.main(self.prm, self.wg_id):

            self._set_outputs()
            self._set_main_wg(self._ui_pipeline)
            self._gui_static_display_0('1_2')

            self._display_output('1_2',  not self.prm.is_button_last, self.prm.gui_static_before_confirm_1, self.prm.gui_static_after_confirm_2)
            self._display_output('3',    self.prm.gui_dynamic_before_fit_3)
            self._display_output('4_fx', self._run_fx)
            self._display_output('5',    self.prm.gui_static_w_fitted_5)
            self._display_output('6',    self.prm.gui_dynamic_w_fitted_6)
            self._display_output('7',    self.prm.gui_static_w_fitted_7)
            self._display_output('1_2_', self.prm.is_button_last)
            self._display_output('8_e',  self.prm.custom_update_fitted_8)
            self._display_output('8_fx', self.prm.custom_update_fitted_8)
            self._display_output('8',    self.prm.gui_dynamic_w_fitted_8)

            self._ui_pipeline()
            # ─────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────
            # ► Update Fitted ► Extra custom confirm
            # ─────────────────────────────────────────────────────────────────────────────────────────
            if self.prm.custom_update_fitted_8 is None:
                return        

            update_wg = Confirm(
                output = self._outputs['8_fx'],
                desc   = self.prm.update_desc.upper(),
                wait   = self.prm.output_wait, 
            ).get_button(self._update_fitted)

            with self._outputs['8_e']:
                display(update_wg)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ModelUIRunner(ModelUIRunnerBase):
    """
    Interactive wrapper around base Model class pipe: fit - transform - update:
    Adds UI choices and pass to a Worker's Model for training given the dynamic 
    User Choices.

    This class is the primary interface used by AIDflow Workers to expose their 
    models to the user. It manages the instantiation of `_ModelUIFit` and 
    coordinates the interaction between the model's logic and the interactive 
    parameter panels.
    """
    _runner: _ModelRunner = None
    choice: "ModelUIChoice" = None

    _wg_cycle = _WgLifecycle()

    @Dataclass.base
    class UIParams(UIParams):
        pass
        
    def __init__(self, 
        model:  ModelBase, 
        setup:  Parameters = None, 
        verbose:      bool = None,
        show_data:    bool = False, 
        data_sample:  int  = 100,
        wg_id:        int  = None
    ):
        self.model  = model
        self.setup  = setup or getattr(model, 'setup', None)

        if self.setup is None:
            raise TypeError("AIG SAFETY ► AIG Flow Setup must be provided to Worker's Model UI.")
        
        self.verbose = AigParam.di(verbose, self.setup.verbose)        

        self._set_data_view(show_data, data_sample)
        self.wg_id = wg_id
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @property
    def _fitted(self):
        if self._runner is not None:
            return self._runner.fitted    

    def _set_data_view(self, show_data: bool = False, data_sample:int = 100):
        self.show_data   = show_data
        self.data_sample = data_sample

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # UI init construct → Params and Choices to fit
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def panel(self, 
        choice_init:        str, 
        choice_options:     list, 
        choice_desc:        str = '', 
        choice_type:        str = 'radio', 
        title:              str = '', 
        show_data:          bool = False, 
        show_data_sample:   int = 100,
    ) -> None:
        """
        Synthesizes and displays the Model Selection interface.
        
        Based on the provided options, this method renders an interactive 
        choice panel (Radio or MultiChoice), allowing the user to specify 
        which fitter implementation should be used for the current execution.
        """
        with self._wg_cycle.main(wg_id = self.wg_id):
            self._set_data_view(show_data, show_data_sample)

            choice = ModelUIChoice(self.verbose, choice_options)
            self.choice = choice.create(choice_init, choice_desc, choice_type)                  

            if not choice.is_shown():
                return
            
            w = choice.get_widget()
            g = []
            if title: 
                desc = 'MODEL SELECTION'
                g.append(PanelConstructor._header(title.upper(), desc))
            if isinstance(w, MultiChoice):  
                g.append(w.hbox)
            else: 
                g.append(w)

            ui:wg.VBox = wg.VBox(g)
            ui.add_class(UI.get_css_id('panel'))
            display(ui)
            print()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

            
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # FIT wrapper around parent fit method
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def fit(self, *args, ui_params: UIParams = UIParams(), **kwargs) -> None:        
        """
        Fits parent fx based on User Choice in UI → Allows dynamic visual and data links across JN.
        
        This is the primary entry point for model training. It analyzes the 
        `ui_params` to determine if it should run in 'Headless Mode' (direct execution) 
        or 'Interactive Mode' (rendering a sequence of UI panels and waiting for 
        user confirmation).

        **params (self.UIParams) → Full parametrization space of .fit() → notes:**
        - FX_4          : Part where parent fitted = FIT.fit(*a,**k) | FIT.update(*a,**k) is executed
        - gui_x         : gui_ params accepts any function to execute as gui_fx(*a,**k) - function 
                          return (if any) is expected as a list [d1, d2,...] of gui elements to be shown
        - gui static    : UI part shown as a static element - cannot be updated once shown to user - 
                          only read its params.        
        - gui dynamic   : UI part refreshed / computed with every CONFIRM button click - i.e. function 
                          can use dynamic inputs.

        Params
        ------
        fit_transform_pipeline : bool = True
            True → fit() - transform() | False → fit() only |
            NOTE: True → self._fitted = transform() | False → self._fitted = model

        fit_transform_inputs_link : str = None
            Optional link for fit_transform() data - WorkerInput data which goes to 
            transform after fit; None → defaults to the first position in inputs / args
            - Using FIT.UPDATE value → transform() will use stored update data on model instance
            - Using any key present in inputs → transform() will use data by inputs[key]

        args_kwargs_to_verify_na : dict  = None
            dict provides indication of MANDATORY DATA THAT MUSTN'T BE EMPTY of parent fit fx

        parent_data_check_update : MethodType = None
            FX / [FX, conds] checks changes in dynamic input data by previous JN steps in a pipeline

        parent_has_update_FX_method : MethodType = False
            True → runs FX.update() instead of FX.fit() when no change in the input parent data

        gui_static_before_confirm_1 : MethodType = None
             (1) FX to be computed / shown → if shown then befor CONFIRM button

        gui_static_after_confirm_2 : MethodType = None
            (2) FX to be computed / shown → if shown then after CONFIRM button

        gui_dynamic_before_fit_3 : MethodType = None
            (3) FX to be computed / shown → if shown then after the (2) FX output

        dynamic_to_args_kwargs_FX_4 : MethodType = None
            (4) Addition to arguments computed/retrieved before fit

        gui_static_w_fitted_5 : MethodType = None
            (5) FX to be computed / shown → if shown then after the (4)

        gui_dynamic_w_fitted_6 : MethodType = None
            (6) FX to be computed / shown → if shown then after the (5)

        gui_static_w_fitted_7 : MethodType = None
            (7) FX to be computed / shown → if shown then after the (6)

        custom_update_fitted_8 : MethodType = None
            (8) Custom CONFIRM button → with self.fitted update by provided FX 
            → if shown then after the (7)

        gui_dynamic_w_fitted_8 : MethodType = None
            (8) FX to be computed / shown with updated fitted from (8) → if shown then after the (7)

        custom_control_widget : MethodType = None
            Custom control widget instead of default confirm button → has to be provided 
            as a function accepting only one arg: the main UI link function.
            For instance, lambda fx: self.slider.observe(fx, names='value')

        is_button_last : bool = True
            False → Confirm button is shown between (1) and (2) | True → Confirm 
            button is shown at the very end of the UI construct
        
        show_ui : bool = True
            True  → Standard behavior of ModelUIRunner.
            False → ModelUIRunner acts API interface without displaying any output.
        """        
        self._runner = _ModelRunner(
            model       = self.model, 
            params      = ui_params,
            choice      = self.choice, 
            verbose     = self.verbose,    
            show_data   = self.show_data, 
            data_sample = self.data_sample,
            wg_id       = self.wg_id  
        )
        self._runner.fit(*args, **kwargs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ModelUIChoice:
    """UI main dynamic choices."""
    _choice   = None
    _is_shown = False

    def __init__(self, verbose, options):
        self.verbose = verbose
        self.options = options

    def __getattr__(self, name:str):      
        if not self._is_shown:
            raise AttributeError(name)
        return getattr(self._choice, name)

    def __repr__(self):
        return repr(self._choice)

    def _repr_html_(self):
        if self._choice is None:
            return "<p>Not initialized</p>"
        if not self._is_shown:
            return f"<p>{self._choice}</p>"
        return self._choice._repr_html_()

    def is_shown(self) -> bool:
        return len(self.options) > 1 and self.verbose > 1
    
    def _create(self, 
        value:str, 
        description: str, 
        choice_type: str = 'radio'
    ) -> "wg.RadioButtons | MultiChoice":
        
        if choice_type == 'multi_choice':        
            return MultiChoice(value, self.options)
        
        if choice_type == 'radio':
            return wg.RadioButtons(
                    value       = value,
                    options     = self.options,
                    description = description,                    
                    layout      = {'margin':'0 0 0 7px'}, 
            )
    
    def create(self,
        init_value:  str, 
        description: str, 
        choice_type: str = 'radio'
    ):
        """Fitting ML choices for parent fx."""
        self._is_shown = self.is_shown()
        if not self._is_shown:
            self._choice = init_value
        else:
            self._choice = self._create(init_value, description, choice_type)
        return self

    def get_widget(self) -> "wg.RadioButtons | MultiChoice | str":
        """
        With respect to is_show state, it returns either:
        - False → chosen init value
        - True  → created underlying widget
        """
        return self._choice
    
    @property
    def value(self):
        if not self._is_shown:
            return self._choice
        return self._choice.value
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# Multi fx choice by Checkboxes
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class MultiChoice:
    
    def __init__(self, value: Any, options: list):
        self.options = options
        layout = wg.Layout(width='auto', margin='0 0 0 0')
        
        All = 'All'
        setup = [ True if ( value == All) or (value == o) else False for o in options + [All] ]
        
        self.checkboxes   = [wg.Checkbox(value=setup[i],  description=o, layout=layout) for i,o in enumerate(options)]
        self.all_checkbox =  wg.Checkbox(value=setup[-1], description=All, layout=layout)
        
        self.updating = False
        self.all_checkbox.observe(self.update_checkboxes, names='value')
        for checkbox in self.checkboxes:
            checkbox.observe(self.update_all_checkbox, names='value')

        self.display_checkboxes()

    def update_checkboxes(self, change):
        if not self.updating:
            self.updating = True
            if self.all_checkbox.value:
                # If "All" is checked, set all checkboxes to True
                for checkbox in self.checkboxes:
                    checkbox.value = True
            else:
                # If "All" is unchecked, set all checkboxes to False
                for checkbox in self.checkboxes:
                    checkbox.value = False
            self.updating = False

    def update_all_checkbox(self, change):
        if not self.updating:
            self.updating = True
            # Check if all or none checkboxes are checked
            if all(checkbox.value for checkbox in self.checkboxes) :
                self.all_checkbox.value = True
            else:
                # If any checkbox is unchecked, uncheck the "All" checkbox
                self.all_checkbox.value = False
            self.updating = False

    def display_checkboxes(self):
        self.checkboxes_with_all = [self.all_checkbox] + self.checkboxes
        self.hbox = wg.HBox(self.checkboxes_with_all, 
                            layout=wg.Layout(
                                display = 'flex', flex_flow = 'row', 
                                justify_content = 'flex-start',
                                align_items = 'center', padding = '0'
        ))        

    @property
    def value(self): return (
        [self.options[i] for i,c in enumerate(self.checkboxes_with_all[1:]) if c.value]
    )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬