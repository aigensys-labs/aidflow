"""
Dynamic Visualization Engine for Automated Grid-Plot Synthesis.

This module implements a high-level abstraction over Plotly, enabling the 
automatic generation of complex subplot grids through functional mapping. 
Instead of manual axis management, it allows users to define a single graph 
function (`graph_fx`) and apply it across a collection of variables, while 
the engine handles grid calculation, axis anchoring, and layout optimization.

Key operational capabilities:
1. Functional Plot Mapping: Implements a decoupled architecture where the 
   visual representation (graph/title/update functions) is separated from the 
   structural arrangement, allowing for rapid iteration of visual analytics.

2. Intelligent Grid Orchestration: Automatically computes optimal row and 
   column dimensions based on input volume, managing sub-axis anchors and 
   secondary Y-axes to maintain a clean, readable layout without overlap.

3. Performance Optimization: Integrates with the `Parallel` execution layer 
   to synthesize large grids of graphs concurrently, significantly reducing 
   rendering latency for high-dimensional datasets.

4. CSS-Driven Theme Integration: Features a seamless bridge to the AIGENSYS 
   design system via `adjust_to_css`, automatically mapping Plotly's layout 
   properties (colors, fonts, grids) to global CSS variables for a professional, 
   dark/light mode compatible experience.

5. Robust Axis Abstraction: Solves common Plotly subplot challenges by 
   internally managing axis anchors and domains, ensuring that shared axes 
   and secondary scales are applied consistently across all sub-plots.
"""

from ..deps.common      import  time
from ..deps.data        import  np
from ..deps.plots       import  go, make_subplots
from ..deps.types       import  FuncType, Any

from ..core.factory     import  Dataclass, DictClass
from ..core.accessor    import  Parallel, UI


@Dataclass.base
class GridParams:
    """Shared parameters of a Grid Plot."""
    X:                  list|tuple|dict
    graph_fx:           FuncType
    title_fx:           FuncType
    update_fx:          FuncType
    lX:                 int
    rows:               int
    cols:               int
    matrix:             int
    height:             int 
    width:              int
    secondary_y:        bool
    shared_yaxes:       bool
    desc_axes_by_first: bool
    horizontal_spacing: float
    vertical_spacing:   float
    grid_title:         str|dict
    grid_title_font:    dict
    annotations_font:   dict
    update_layout:      dict
    update_xaxes:       dict
    update_yaxes:       dict
    note:               str
    parallel:           bool

@Dataclass.base
class PlotAnchor:
    x: str
    y: str

@Dataclass.base
class PlotAxis:
    xaxis:  str
    yaxis:  str
    yaxis2: str

class PlotMainAxis(DictClass):
    """xaxis-yaxis-yaxis2"""
    xaxis:str  = 'xaxis'
    yaxis:str  = 'yaxis'
    yaxis2:str = 'yaxis2'


class Plot:
    """
    Orchestrator for the synthesis, layout, and styling of complex subplot grids.
    
    This class provides the logic required to transform a set of data variables 
    and a graphing function into a professional-grade visual matrix. It handles 
    the heavy lifting of Plotly's internal coordinate system, allowing developers 
    to focus on the analytics rather than the axes.
    """

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # FX → PLOTLY → AXES EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    def _get_anchors(fig:go.Figure, row:int, col:int) -> PlotAnchor:
        """Returns anchors of a graph's axis in a subplot grid x-y."""
        g = fig.get_subplot(row, col)
        x = g.yaxis['anchor']     # get primary x axis 
        y = g.xaxis['anchor']     # get primary y axis → secondary +1
        return PlotAnchor(x=x, y=y)
    
    @classmethod
    def get_sub_axes(cls, fig:go.Figure, row:int, col:int) -> PlotAxis:        
        """Returns actual names of graph's xaxis-yaxis-yaxis2."""
        p = lambda a: int(a[1:]) if a[1:] else 0
        
        a  = cls._get_anchors(fig, row, col)
        xi = p(a.x)
        yi = p(a.y)

        xaxis, yaxis = 'xaxis', 'yaxis'
        yi2 = yi+1
        if xi:
            xaxis = f'{xaxis}{xi}'
            yaxis = f'{yaxis}{yi}'
        else: 
            yi2+=1        
        yaxis2 = f'yaxis{yi2}'
        
        return PlotAxis(xaxis=xaxis, yaxis=yaxis, yaxis2=yaxis2)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _copy_axis(axis:go.Layout) -> dict:
        """Extract only explicitly set axis properties → (fast, generic, no defaults)."""
        if axis is None:
            return {}
        # return axis.to_plotly_json()  # ~slow full copy()
        props = axis._props
        if props is None:
            return {}
        return dict(axis._props)        # ~fast copy()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # FX → PLOTLY → CUSTOM SUBPLOTS CONSTRUCT
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _create_grid(prm:GridParams) -> go.Figure:
        hs = prm.horizontal_spacing / prm.cols 
        vs = prm.vertical_spacing / (max(2,prm.rows) - 1)
        return make_subplots(
            rows  = prm.rows, 
            cols  = prm.cols,
            shared_yaxes = prm.shared_yaxes,
            specs = [[{"secondary_y": prm.secondary_y}]*prm.cols]*prm.rows,
            subplot_titles = ['dummy title']*prm.matrix,    # forces plotly to calculate all annot. features
            vertical_spacing = vs,
            horizontal_spacing = hs     
        )         
    
    @staticmethod
    def _compute_grid_coor(lX:int, grid_cols:int) -> tuple[int, int, int]:
        cols = min(lX, grid_cols)
        rows = int(lX / cols)
        rows = max(1,rows)
        if cols*rows < lX: rows+=1 
        matrix = int(cols*rows)
        return rows, cols, matrix  

    @staticmethod
    def _max_grid_len(X:list, gmax:int) -> tuple[int, str]:
        lX = len(X)
        if lX <= gmax:                        
            return lX, ''
        return gmax, f' | Showing {gmax} variables of {lX}'        
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _prm_x_to_dict(prm:GridParams) -> dict:
        if isinstance(prm.X, dict):
            return prm.X
        return dict(zip(prm.X, prm.X))

    @staticmethod
    def _get_prm_x_key(i:int, prm:GridParams) -> str:
        if not isinstance(prm.X, dict):
            return prm.X[i]
        return list(prm.X)[i]
                    
    @classmethod
    def _get_prm_x_val(cls, i:int, prm:GridParams) -> str|Any:
        x = cls._get_prm_x_key(i, prm)
        if not isinstance(prm.X, dict):
            return x
        return prm.X[x]
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
        
    @classmethod
    def _stopwatch(cls, prm:GridParams) -> float:        
        start = time.perf_counter()
        prm.graph_fx( cls._get_prm_x_val(0, prm) )
        return time.perf_counter() - start

    @classmethod
    def _get_parallel_th(cls, prm:GridParams) -> float:
        return np.inf if not prm.parallel else 2 / cls._stopwatch(prm)        

    @classmethod
    def _create_graph(cls, prm:GridParams) -> dict[str, go.Figure]:
        fes = cls._prm_x_to_dict(prm)
        return {x: prm.graph_fx(x_) for x, x_ in fes.items()}
            
    @classmethod
    def _create_graphs(cls, prm:GridParams) -> dict[str, go.Figure]:
        th = cls._get_parallel_th(prm)
        g  = Parallel(th=th, verbose=False).process(Plot._create_graph)
        return g(prm)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _update_fx(x:str|Any, g:go.Figure, prm:GridParams) -> None:
        if prm.update_fx is None:
            return
        prm.update_fx(x, g)        
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def _update_grid(cls, fig:go.Figure, prm:GridParams) -> None:
        graph = cls._create_graphs(prm)
        coor  = [(r,c) for r in range(prm.rows) for c in range(prm.cols)]
        for r,c in coor:
            i = r*prm.cols + c
            if i == prm.lX: 
                break
            x = cls._get_prm_x_key(i, prm)
            g = graph[x]

            x_ = cls._get_prm_x_val(i, prm)
            cls._update_fx(x_, g, prm)
            
            cls._add_to_grid(fig, g, r+1, c+1, prm.desc_axes_by_first)

            t = prm.title_fx(x_)
            fig.layout.annotations[i].update(text = t)  # annot. dynamic update of generated  title

        for i in range(prm.lX, prm.matrix):
            fig.layout.annotations[i].update(text='')   # the last empty graph delete title        

    @classmethod
    def _add_to_grid(cls, 
        fig_subplots:go.Figure, graph:go.Figure, row:int, col:int, desc_axes_by_first=False
    ) -> None:       
        
        for trace in graph.data:
            secondary = ( trace.yaxis == 'y2' )
            fig_subplots.add_trace(trace, row=row, col=col, secondary_y=secondary)
        
        axes = cls.get_sub_axes(fig_subplots, row, col)            
        anch = cls._get_anchors(fig_subplots, row, col)

        for g_ax, s_ax in axes.items():
            is_yaxis2 = ( g_ax == PlotMainAxis.yaxis2 )
            if is_yaxis2 and ( not hasattr(graph.layout, g_ax) ): 
                break
            
            graph_axis = cls._copy_axis(graph.layout[g_ax])
            sub_f_axis = fig_subplots['layout'][s_ax]
            graph_axis['domain'] = sub_f_axis['domain']         # position in grid
            
            if ( g_ax == PlotMainAxis.xaxis ):
                graph_axis['anchor'] = anch.y
            else:
                graph_axis['anchor'] = anch.x
                if is_yaxis2:
                    graph_axis['overlaying'] = anch.y
                if desc_axes_by_first and ( col-1 != 0 ):          
                    graph_axis['title'] = {}                    # show yaxis desc only for the first in a row                
            
            sub_f_axis.update(graph_axis)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _grid_title(prm:GridParams) -> dict:
        if prm.grid_title is None:
            return {}        
        if isinstance(prm.grid_title, str):
            grid_title = {
                'text': f'<b>{str(prm.grid_title)}</b>' + prm.note,
                'y': 0.9985, 'x': 0.5, 'xanchor':'center', 'yanchor':'top',
                'font': prm.grid_title_font
            }          
        return {'title': grid_title}         

    @staticmethod
    def _grid_layout(prm:GridParams) -> dict:
        return dict(
            height = prm.height*prm.rows, width = prm.width*prm.cols,        
            annotations = [go.layout.Annotation({'font':prm.annotations_font})],
            showlegend  = False            
        )

    @classmethod
    def _update_layout(cls, fig:go.Figure, prm:GridParams):
        grid_title = cls._grid_title(prm)
        matrix_layout = cls._grid_layout(prm)
        update_layout = grid_title | matrix_layout | prm.update_layout

        fig.update_layout(**update_layout)  
        fig.update_xaxes(**prm.update_xaxes)
        fig.update_yaxes(**prm.update_yaxes)

    @staticmethod
    def _default_font() -> tuple[GridParams, dict[str, dict]]:
        schema = Dataclass.schema(GridParams)
        return schema, {
            schema.grid_title_font  : dict(family="Arial", size=15, color="black"),
            schema.annotations_font : dict(family="Arial", size=11, color="black"),
            schema.update_xaxes     : dict(tickfont = dict(family="Arial", size=10, color="black")),
            schema.update_yaxes     : dict(tickfont = dict(family="Arial", size=11, color="black"))
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def create_subplots(cls,
        features:   list|tuple|dict, 
        graph_fx:   FuncType, 
        title_fx:   FuncType, 
        update_fx:  FuncType = None,
        *,
        # ────────────────────────────────────
        grid_cols:              int  = 4,
        secondary_y:            bool = False,
        shared_yaxes:           bool = False,
        # ────────────────────────────────────
        height:                 int  = 300, 
        width:                  int  = 470,
        horizontal_spacing:    float = 0.2,
        vertical_spacing:      float = 0.3, 
        # ────────────────────────────────────
        grid_title:         str|dict = None,
        grid_title_font:        dict = None,
        annotations_font:       dict = None,  
        update_layout:          dict = None,
        update_xaxes:           dict = None,
        update_yaxes:           dict = None,
        # ────────────────────────────────────
        desc_axes_by_first_only:bool = False,
        gmax:                   int  = 400, 
        # ────────────────────────────────────
        parallel:               bool = False
    ) -> go.Figure|str:
        """
        Generates Plotly Subplots Grid for x in X by custom `graph_fx - title_fx - update_fx`.
        
        This is the primary API for creating visual matrices. It automatically 
        calculates grid coordinates and invokes the provided functions to populate 
        each cell.

        **fx args**: 
        - *graph_fx - title_fx* must accept only one argument.
        - *update_fx* must accept two args: update_fx(x, graph), where graph is created during loop by graph_fx

        **args by X types** behavior during graph gen loop:
        - `list | tuple` : input to *fx* is value from the list | tuple
            - fx must use outer scope data to construct graph if value is str type

        - `dict` : input to *fx* is dict's value during the iter loop
            - this allows to construct graph, title, update without outer scope data
            - NOTE that the dict key is not provided to the fx functions in this case (only value)

        **Default fonts**:
        - grid_title_font  = dict(family="Arial", size=15, color="black"),
        - annotations_font = dict(family="Arial", size=11, color="black"),
        - update_xaxes     = dict(tickfont = dict(family="Arial", size=10, color="black")),
        - update_yaxes     = dict(tickfont = dict(family="Arial", size=11, color="black"))
        """
        
        lX, note = cls._max_grid_len(features, gmax)
        if not lX:
            return

        rows, cols, matrix = cls._compute_grid_coor(lX, grid_cols)        
        schema, font = cls._default_font()

        prm = GridParams(
            X                   = features,
            graph_fx            = graph_fx,
            title_fx            = title_fx,
            update_fx           = update_fx,
            lX                  = lX,
            rows                = rows,
            cols                = cols,
            matrix              = matrix,
            secondary_y         = secondary_y,
            shared_yaxes        = shared_yaxes,
            desc_axes_by_first  = desc_axes_by_first_only,
            height              = height,
            width               = width,
            horizontal_spacing  = horizontal_spacing,
            vertical_spacing    = vertical_spacing,
            grid_title          = grid_title,
            grid_title_font     = grid_title_font  or font[schema.grid_title_font],
            annotations_font    = annotations_font or font[schema.annotations_font],
            update_layout       = update_layout    or {},
            update_xaxes        = update_xaxes     or font[schema.update_xaxes],
            update_yaxes        = update_yaxes     or font[schema.update_yaxes],
            note                = note,
            parallel            = parallel
        )

        fig = cls._create_grid(prm)
        cls._update_grid(fig, prm)
        cls._update_layout(fig, prm) 

        return fig
    # ─────────────────────────────────────────────────────────────────────────────────────────────────  

    @classmethod
    def adjust_to_css(cls, fig:go.Figure, add_class:bool=True) -> go.FigureWidget:
        """Format graph to align with main Panel CSS design."""

        fig.update_layout(
            paper_bgcolor = 'rgba(0,0,0,0)',
            plot_bgcolor  = 'rgba(0,0,0,0)',
            font = dict(
                color  = 'var(--text-main)', 
                family = '-apple-system, BlinkMacSystemFont, "Segoe UI"'
            ))

        def apply_pro_axis_theme(axis):

            # Base updates for all axes
            updates = dict(
                zerolinecolor = 'var(--border)',
                tickfont = dict(color='var(--text-dim)')
            )
            
            # Check if the grid was explicitly hidden
            is_hidden = (axis.showgrid is False) or (axis.gridcolor == 'rgba(0, 0, 0, 0)')
            
            # Only apply the CSS border variable if the grid is meant to be seen
            if not is_hidden:
                updates['gridcolor'] = 'var(--border)'
                
            # Apply the selective updates to the axis
            axis.update(updates)

        fig.for_each_xaxis(apply_pro_axis_theme)
        fig.for_each_yaxis(apply_pro_axis_theme)        

        if add_class:
            
            if not isinstance(fig, go.FigureWidget):
                fig:go.FigureWidget = go.FigureWidget(fig)

            for css in cls.get_css_classes():
                fig.add_class(css)            

        return fig
    
    @staticmethod
    def get_css_classes() -> tuple[str, str]:
        """Returns active Graph Panel CSS classes."""
        return UI.get_css_id('panel'), UI.get_css_id('graph')
# ─────────────────────────────────────────────────────────────────────────────────────────────────