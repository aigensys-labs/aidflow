"""
Polymorphic Rendering Engine for Automated UI Construction.

This module implements the final output layer of AIDflow, responsible for 
dynamously resolving and rendering diverse data types into their optimal 
visual representation. By abstracting the display logic away from the core 
business logic, it ensures that the framework remains environment-agnostic—
seamlessly switching between Jupyter Notebooks, standalone Python scripts, 
and remote Linux/Hub environments.
"""

from ...deps.common     import  re
from ...deps.data       import  (
                                pDF, 
                                pSe, 
                                pd_Styler
                                )
from ...deps.display    import  (
                                tqdm, 
                                tqdm_notebook, 
                                display_html, 
                                display
                                )
from ...deps.plots      import  (
                                go, 
                                plt, 
                                plt_Figure, 
                                plt_Axes
                                )
from ...deps.types      import  FuncType

from ...utils.builtins  import  List
from ...utils.platform  import  Platform

from ..execution        import  KW


def _is_in_hub_lx():    
    return Platform.is_in_hub() or Platform.is_in_lx()


class Display:
    """
    Utility engine for type-aware UI rendering and progress orchestration.
    
    This class centralizes the logic for presenting data to the user, ensuring 
    that every object—from raw strings to complex ML plots—is displayed using 
    the most appropriate visual tool available in the current environment.
    """

    @staticmethod
    def is_html(s:str) -> bool:
        """Checks whether provided string is HTML kind."""
        html_pattern = re.compile(r'<[^>]+>')
        return bool(html_pattern.search(s))

    @classmethod
    def construct(cls, obj:list) -> None:
        """
        Dynamically renders a collection of objects based on their data types.
        
        This is the primary output gateway for AIDflow. It iterates through 
        the provided list and automatically resolves the correct display 
        mechanism:
        - HTML Strings → Rendered as raw HTML via `display_html`.
        - Pandas (DF/Series) → Rendered as formatted tables.
        - Plotly Figures → Invokes `.show()`.
        - Matplotlib (Figure/Axes) → Displays and closes the plot to save memory.
        - Callables → Executes the function for custom rendering logic.
        """
        for g in List.ensure(obj):
            if g is None: 
                continue
            if isinstance(g, str): 
                if  cls.is_html(g): display_html(g, raw=True)
                else: print(g)
            elif isinstance(g, (pDF, pSe, pd_Styler)):  
                display(g)        
            elif isinstance(g, (plt_Figure, plt_Axes)):      
                display(g)
                plt.close(g)
            elif isinstance(g, go.Figure):           
                g.show()
            elif isinstance(g, FuncType):
                g()
            else: 
                display(g)    
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def processing_progress(cls,
        to_process: list|tuple|dict, 
        kwargs:  dict = None, 
        verbose: bool = True,
        desc:    str  = 'Variables Processing', 
        desc_p:  str  = 'Processing Fold',
        min_th:  int  = 10
    ) -> tuple[list|tqdm_notebook, None|tqdm]:
        """
        Synthesizes environment-adaptive progress bars for single or multi-process tasks.
        
        This method detects the current execution platform to determine whether 
        to initialize a Jupyter-native `tqdm_notebook` (for interactive use) 
        or a standard `tqdm` bar (for parallel/background processes). It handles 
        the logic for 'verbose' thresholds to prevent UI clutter during small tasks.

        Returns:
            tuple: A pair containing the single-process progress object and 
                   the optional parallel-process progress object.
        """
        to_process = to_process if (to_process is not None) else []
        
        single_progress = to_process
        parallel_progress = None
        
        exe, pos = False, 0
        verbose = verbose and ( len(to_process) > min_th )

        if kwargs is not None:
            exe = kwargs.get(KW.EXE, False)
            pos = kwargs.get(KW.POS, 0)      

        if ( not exe ) and ( Platform.is_in_jupyter() ) and ( verbose ):
            single_progress = tqdm_notebook(to_process, desc = desc, leave = False)    
                
        # if ( exe ) and ( not _is_in_hub_lx() ):
        #     parallel_progress = tqdm(
        #         desc  = f"{desc_p} {pos}", 
        #         total = len(to_process), position=pos
        #     )
                
        return single_progress, parallel_progress
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def update_progress(progress:str|tqdm='progress', kwargs:dict=None) -> None:
        """
        Increments active multiprocessing progress bars via a centralized registry.
        
        Allows background processes to communicate their completion status 
        back to the main UI thread, ensuring that the user is kept informed of 
        the actual percentage of completion during heavy parallel executions.
        """
        if progress is None:
            return
        if isinstance(progress, str):
            progress:tqdm = kwargs.get(progress)
        if progress is not None: 
            progress.update(1)