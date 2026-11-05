"""
Unified Execution Trigger and Output Orchestration Utility.

This module provides a standardized mechanism for triggering pipeline 
actions via the GUI. Rather than using raw event handlers, it implements 
a consistent 'Confirm' pattern that couples an interactive button with a 
dedicated output stream.
"""

from ...deps.display    import wg, clear_output
from ...deps.types      import Callable
from ..accessor         import UI


class Confirm:
    """
    The standardized confirmation engine for AIDflow interactive panels.
    
    This class abstracts the complexity of binding functions to UI events, 
    handling the associated output buffers and visual state transitions.

    Parameters
    ----------
    widget (Callable, optional) : A dictionary containing a pre-existing 
        widget provided as binding function, e.g. lambda fx: slider.observe(fx)
    output (wg.Output, optional)  : An existing output container to reuse.
    desc (str)  : Label displayed on the button. Defaults to 'CONFIRM'.
    w/h/m (str) : CSS layout parameters for width, height, and margin.
    wait (bool) : If True, clears existing output before executing `fx`.
    """
    def __init__(self,
        widget: Callable = None,
        output: wg.Output = None,
        desc: str = 'CONFIRM', 
        w: str = 'auto', 
        h: str = '30px', 
        m: str = '7px 7px 12px 7px', 
        wait: bool = True,
    ):
        self.widget = widget
        self.output = output
        self.desc   = desc
        self.w      = w
        self.h      = h
        self.m      = m
        self.desc   = desc
        self.wait   = wait

    
    def get_button(self, fx: Callable) -> wg.Button | None:
        """
        Synthesizes a styled confirmation button linked to a specific execution function.

        This method creates a bound pair consisting of a trigger (the button) 
        and a destination (the output widget). It handles the internal wiring 
        required to ensure that when the button is clicked, the associated 
        function (`fx`) runs within the provided or created `wg.Output` context, 
        clearing previous logs if `wait=True`.

        Parameters
        ----------
        fx (Callable) : The function to be executed upon click.

        Returns
        -------
        wg.Button | None
        - wg.Button: The configured button instance.
        - None: When widget was provided without wg, linked to fx provided object.
        """        
        def run(b=None):
            with self.output:
                clear_output(wait = self.wait)
                fx()

        if self.output is None:          
            self.output = wg.Output()          
        layout = wg.Layout(width=self.w, height=self.h, margin=self.m)
        
        if self.widget is None:
            css = UI.get_css_id('button')
            b: wg.Button = wg.Button(
                description = self.desc, layout = layout, tooltip = ''
            )
            b.add_class(css)
            b.on_click(run)    
        else:
            b = None
            self.widget(run)        

        return b

    def get_output(self) -> wg.Output:
        """Returns create output under which the main function operates."""
        return self.output      