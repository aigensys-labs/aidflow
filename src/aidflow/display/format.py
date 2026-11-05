"""
Visual Identity System and Polymorphic Styling Engine.

This module implements the centralized design language for all AIDflow
user interfaces. It abstracts low-level CSS and ANSI styling into a
set of high-level Style tokens, ensuring visual consistency across
diverse output environments (Jupyter HTML vs. Terminal ANSI).

Key Architectural Pillars
-------------------------
1. **Decoupled Design Tokens**: Specialized style classes
   (`StyleTitle`, `StyleStats`, `StyleSpliter`) define the
   visual hierarchy. Global theme changes propagate without
   modifying individual render calls.

2. **Environment Polymorphism**: Dual-backend rendering via
   `FormatHTML` and `FormatANSI`. Rich features (gradients,
   padding, flexbox) render in HTML; ANSI escape codes and
   character-based dividers provide equivalent structure in the
   terminal.

3. **Intelligent Text Synthesis**: The `sync_s` engine
   automatically adjusts grammatical endings (singular/plural,
   has/have) based on the quantity of data being reported.

4. **Layout Orchestration**: DataFrame width calculation and
   structured visual dividers (splitters) create clear spatial
   hierarchy in complex ML output logs.
"""

from ..deps.common      import  (
                                ABC, 
                                abstractmethod,
                                )
from ..deps.data        import  pDF, pd_Styler
from ..deps.display     import  HTML, display
from ..deps.types       import  (
                                Literal, 
                                FuncType
                                )
from ..core.factory     import  Dataclass
from ..core.logging     import  LOG, Logger
from .rgb               import  (
                                RgbANSI, 
                                ANSIToHTML,
                                HTMLToANSI
                                )

DIV = ['solid','double','dashed','gradient']


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TEXT FORMAT BASE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class Style:
    """
    Base design token — the full set of visual attributes for a
    rendered element.

    Variantion of default values by application 
    - Title / Stats / Splitter

    Encapsulates every possible CSS/ANSI styling property
    (font, color, gradient, border, spacing, alignment) in a
    single dataclass. Each Style instance defines the complete
    visual appearance of one element.

    Attributes
    ----------
    size : float | int, default=13
        Font size in pixels (HTML) or reference unit (ANSI).
    formatted : list[int] | int | None, default=None
        Column widths (px) for multi-column text layout.
    color : str, default='black'
        Primary text color.
    color2 : str | None, default=None
        Secondary color (gradient end, accent).
    background : str, default='none'
        Background color or gradient start.
    border_color : str, default='none'
        Divider/border color.
    gradient_kind : str, default='bottom'
        CSS gradient direction.
    font : str
        Font family stack.
    line_height : float, default=1.5
        Line height multiplier.
    display : str, default='block'
        CSS display mode (`'block'`, `'flex'`).
    padding : str, default='none'
        CSS padding shorthand.
    margin_t / margin_b / margin_r / margin_l : int, default=0
        CSS margins (top, bottom, right, left) in px.
    bold : bool, default=False
        Apply bold weight.
    center : bool, default=False
        Center-align text.
    to_html : bool, default=True
        Wrap output in `HTML()` (True) or return raw string.
    div_px : float | int, default=1.0
        Divider thickness in px (HTML) or characters (ANSI).
    div : str, default='solid'
        Divider style: `'solid'`, `'double'`, `'dashed'`,
        `'gradient'`, or a custom character (e.g., `'•'`).
    """    
    size:      float|int = 13
    formatted:  list|int = None
    color:           str = 'black'
    color2:          str = None
    background:      str = 'none' 
    border_color:    str = 'none'
    gradient_kind:   str = 'bottom'
    font:            str = "'Segoe UI', system-ui, sans-serif"
    line_height:   float = 1.5
    display:         str = 'block'
    padding:         str = 'none' 
    margin_t:  float|int = 0
    margin_b:  float|int = 0
    margin_r:  float|int = 0
    margin_l:  float|int = 0
    bold:           bool = False
    center:         bool = False 
    to_html:        bool = True
    div_px:    float|int = 1.0  
    div: Literal['solid','double','dashed','gradient'] = 'solid'

    def update(self, **kwargs) -> "Style":
        """
        Return a new Style with the specified attributes overridden.

        Parameters
        ----------
        **kwargs
            Attribute names and new values to override.

        Returns
        -------
        Style
            A new Style instance with the merged values.
        """        
        return Style( **(dict(self) | kwargs) )

@Dataclass.base
class StyleTitle(Style):
    color:           str = '#000000'
    color2:          str = "#C4C9DB"
    background:      str = "#E0E6EE"
    border_color:    str = "none"
    padding:         str = "8px 8px"  
    bold:           bool = True
    center:         bool = True

@Dataclass.base
class StyleStats(Style):
    color2:          str = "#E5E5E5"
    background:      str = "#F7F7F7"
    border_color:    str = "#E8E8E8"
    padding:         str = "7px 8px"  
    bold:           bool = True
    center:         bool = True
    margin_t:       int = 24
    margin_b:       int = 10
    div_px:    float|int = .5

@Dataclass.base
class StyleSplitter(Style):
    color:          str = "#A8AAB3"
    color2:         str = "#313983"
    margin_t:       int = 4
    margin_b:       int = 8
    margin_l:       int = 0
    margin_r:       int = 0
    padding:        str = 'none'
    div_px:   float|int = .5
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class FormatBase(ABC):    
    """
    Abstract Base Class defining the mandatory interface for all formatting backends.
    
    Ensures that any visual output (Title, Point, Splitter) has a 
    consistent API regardless of whether it is rendered in HTML or ANSI.
    """    
    @Dataclass.base
    class Style(Style):
        pass

    @Dataclass.base
    class StyleTitle(StyleTitle):
        pass

    @Dataclass.base
    class StyleStats(StyleStats):
        pass

    @Dataclass.base
    class StyleSplitter(StyleSplitter):
        pass

    @staticmethod
    def _sp(
        x: int | list | tuple | dict, 
        s: str = '', 
        h: tuple = ('has','have'), 
        *args
    ) -> tuple[str, str]:        
        """
        Resolve singular/plural grammatical forms based on quantity.
        - s: "s" or none 
        - h: has or have
        - args: any custom pairs as [singular, plural]
        """
        if not isinstance(x,int): 
            x = len(x)

        if x <= 1: 
            h_    = h[0]
            args_ = [a[0] for a in args]

        else: 
            s, h_ = 's', h[1]
            args_ = [a[1] for a in args]

        return s, h_, *args_
    
    @classmethod
    def sync_s(cls, message: str|FuncType, *args, **kwargs) -> str:
        """
        Resolve a deferred message template with correct
        singular/plural endings.

        If `message` is a string, it is returned as-is.
        If it is a callable, it is invoked with the resolved
        "s" endings computed from the provided quantities.

        Parameters
        ----------
        message : str | FuncType
            Static string or a callable that accepts
            `(s, ...)` ` keyword arguments.
        *args, **kwargs
            Quantities used to determine singular/plural.

        Returns
        -------
        str
            The resolved message string.
        """
        if not isinstance(message, FuncType):
            return message
        s = []
        for v in list(args) + list(kwargs.values()):
            if ( v is None ) or isinstance(v,bool):
                continue
            s.append(cls._sp(v)[0])
        return message(*s)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    @staticmethod
    def pdf_width(df: pDF | pd_Styler, padding: int = 1) -> int:
        """
        Calculate the total rendered width (in characters) of a
        DataFrame — used for aligning dividers and borders.

        Parameters
        ----------
        df : pDF | pd_Styler
            The DataFrame (or its Styler wrapper) to measure.
        padding : int, default=1
            Extra character padding per column.

        Returns
        -------
        int
            Total width in characters.
        """        
        col_widths = []
        cols = df.columns
        lc = len(cols)
        if lc:
            if isinstance(df, pd_Styler): 
                df = df.data
            df = df.reset_index()
            cols = df.columns
        for col in cols:
            d = df[col]
            if isinstance(d.iloc[0], float): 
                d = d.astype(str).str[:8]  # 8 is native pandas float display
            max_len = max(d.astype(str).apply(len).max(), len(str(col)))
            col_width = int(max_len + padding)
            col_widths.append(col_width)

        total_width = sum(col_widths)
        
        return total_width
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def div_print(cls, message:str) -> None:
        """
        Print a message flanked by horizontal dividers above
        and below (emphasized block).

        Parameters
        ----------
        message : str
            The text to display.
        """        
        cls.splitter()
        print(message)
        cls.splitter()

    @staticmethod
    def sync_ansi_html(message:str) -> str:
        """
        Convert embedded ANSI escape codes to HTML equivalents
        (base implementation: passthrough).

        Parameters
        ----------
        message : str
            The text to convert.

        Returns
        -------
        str
            Converted text.
        """        
        return message        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    @abstractmethod
    def display():
        ...
    
    @staticmethod
    @abstractmethod
    def point():
        ...

    @staticmethod
    @abstractmethod
    def splitter():
        ...

    @classmethod
    @abstractmethod
    def title():
        ...
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ReportLogger:
    """
    Logging bridge between display calls and the report log.

    Intercepts title generation and forwards the rendered text
    (ANSI codes stripped) to the structured report logger,
    maintaining a plain-text audit trail of all displayed
    headings.
    """

    @staticmethod
    def _logger(t:str, instance: type | None) -> None:
        """
        Forward a rendered title to the report logger (if
        a logger instance is available).

        Parameters
        ----------
        t : str
            The rendered title text.
        instance : type | None
            The logger target class (None = no logging).
        """        
        if not instance:
            return
        Logger(instance).info(HTMLToANSI.strip(t))

    @staticmethod
    def log(title: FuncType):
        """
        Decorator that logs the generated title to the report
        while passing it through unchanged.

        Parameters
        ----------
        title : FuncType
            The title-generating method to wrap.

        Returns
        -------
        FuncType
            Wrapped method that logs and returns the title.
        """        
        def wrapper(*args, **kwargs):
            l = kwargs.pop(LOG.R, None)
            t = title(*args, **kwargs)
            ReportLogger._logger(t, l)
            return t
        return wrapper     
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TEXT FORMAT HTML
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FormatHTML(FormatBase):
    """
    Rich-text rendering backend for Jupyter and Web environments.
    
    Translates `Style` tokens into inline CSS, supporting complex visual 
    effects like linear gradients, flexbox alignment, and custom fonts.
    """

    @staticmethod
    def sync_ansi_html(message:str) -> str:
        """Convert ANSI escape codes to HTML color spans."""
        return ANSIToHTML.convert(message)
    
    def _background(style:Style) -> str:
        """
        Generate the CSS background value — either a solid color
        or a linear gradient (if `color2` is set).        
        """
        if style.color2 is not None:
            return f'linear-gradient(to {style.gradient_kind}, {style.background}, {style.color2})'
        return style.background
    
    @classmethod
    def _html_text(cls, value:str, style:Style = Style()) -> HTML|str:
        """
        Render a text value as a styled HTML `<div>`.

        Applies all Style properties (font, color, background,
        border, spacing, alignment) as inline CSS.

        Parameters
        ----------
        value : str
            The text content.
        style : Style, default=Style()
            The visual attributes to apply.

        Returns
        -------
        HTML | str
            The rendered element (wrapped in `HTML()` if
            `style.to_html` is True, else raw string).
        """        
        weight = 'normal'
        align  = 'left' 
        if style.bold: weight = 600
        if style.center: align  = 'center'

        html = (f'''
        <div style="
            display:        {style.display};
            font-family:    {style.font}; 
            font-size:      {style.size}px; 
            font-weight:    {weight}; 
            line-height:    {style.line_height};
            color:          {style.color}; 
            background:     {cls._background(style)}; 
            border-top:     {style.div_px}px {style.div} {style.border_color};
            border-bottom:  {style.div_px}px {style.div} {style.border_color};
            text-align:     {align};
            padding:        {style.padding};
            margin-top:     {style.margin_t}px;
            margin-bottom:  {style.margin_b}px;
            margin-left:    {style.margin_l}px;
            margin-right:   {style.margin_r}px;
        ">
            {cls.sync_ansi_html(value)}
        </div>''')

        if style.to_html:
            return HTML(html)
        return html
    
    @staticmethod
    def _formatted(value:list[str]|tuple[str], formatted:list[int]|int) -> str:
        """
        Render a multi-column text layout using flexbox.

        Parameters
        ----------
        value : list[str] | tuple[str]
            Column contents.
        formatted : list[int] | int
            Fixed pixel widths for each column (except the last,
            which flexes to fill remaining space).

        Returns
        -------
        str
            HTML spans with flex layout.
        """        
        h = ''
        if isinstance(formatted, int):
            formatted = [formatted]
        for i,v in enumerate(value):
            if i == len(value)-1:
                h += f'<span style="flex: 1; word-break: break-all;">{v}</span>'
                break
            if i < len(formatted):
                h += f'<span style="flex: 0 0 {formatted[i]}px; overflow: hidden; text-overflow: ellipsis;">{v}</span>'
            else:  
                h += f'<span>{v}</span>'
        return h
    
    @classmethod
    def display(cls, *values, style:Style = Style(), **kwargs) -> None:  
        """
        Render one or more values using the HTML backend.

        Handles:
        - Plain strings: joined and rendered as one styled div.
        - HTML objects: displayed as-is.
        - Mixed: each value rendered individually.
        - Formatted lists: multi-column flexbox layout.

        Parameters
        ----------
        *values
            Content to display.
        style : Style, default=Style()
            Visual attributes.
        **kwargs
            Forwarded to `sync_s` for plural resolution.
        """              
        if all([isinstance(v,HTML) for v in values]):
            display(*values)
            return

        if all([isinstance(v,str) for v in values]):
            content = ' '.join(str(v) for v in values)
            display( cls._html_text(content, style) )
            return
        
        for v in values:
            v = cls.sync_s(v, **kwargs)
            if style.formatted is not None:
                if isinstance(v, (list,tuple)):
                    v = cls._formatted(v, style.formatted)
                    s = style.update(display='flex')
                    display( cls._html_text(v, s) )
                    continue
            if isinstance(v, str):
                display( cls._html_text(v, style) )
            else:
                display(v)

    @classmethod
    def point(cls, *values, style=Style(formatted=[10,15], div='•')) -> None:
        """
        Render bullet-point list items.

        Parameters
        ----------
        *values
            The bullet content.
        style : Style
            Bullet character and column widths.
        """        
        for v in values:
            cls.display(['', style.div, v], style=style)
    
    @staticmethod
    def splitter(
        style: StyleSplitter = StyleSplitter(),
        show: bool = True
    ) -> str|HTML|None:
        """
        Render a horizontal divider line.

        Supports solid, double, dashed, and gradient styles.

        Parameters
        ----------
        style : StyleSpliter, default=StyleSpliter()
            Divider appearance (color, thickness, style).
        show : bool, default=True
            If True, display immediately. If False, return the
            HTML string without displaying.

        Returns
        -------
        str | HTML | None
            The divider element (when `show=False`).
        """        
        margin_padding = (
            f"margin-top:  {style.margin_t}px; margin-bottom: {style.margin_b}px;"
            f"margin-left: {style.margin_l}px; margin-right:  {style.margin_r}px;"
            f"padding: {style.padding};"
        )
        if style.div == 'gradient':
            fs = (
                f'<hr style="background: linear-gradient(to {style.gradient_kind}, {style.color}, {style.color2});'
                f'height: {style.div_px}px; border: none; {margin_padding}">'
            )
        else: 
            fs = f'<hr style="border: {style.div_px}px {style.div} {style.color}; {margin_padding}">'
        if style.to_html: 
            fs = HTML(fs)
        if not show: 
            return fs
        display(fs) 

    @classmethod
    @ReportLogger.log
    def title(cls,
        title: str, 
        *args,
        style: StyleTitle = StyleTitle(),        
        show:  bool = True,
        **kwargs
    ) -> str|HTML:
        """
        Render a section title with decorative background bar.

        Parameters
        ----------
        title : str
            The title text (or a callable for dynamic resolution).
        *args
            Quantities for plural resolution.
        style : StyleTitle, default=StyleTitle()
            Title appearance.
        show : bool, default=True
            If True, display. If False, return the HTML.
        **kwargs
            Forwarded to `sync_s`.

        Returns
        -------
        str | HTML
            The rendered title.
        """        
        t = cls.sync_s(title, *args, **kwargs)        
        if not show:
            return cls._html_text(t, style) 
        cls.display(t, style=style)
        return t
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TEXT FORMAT ANSI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FormatANSI(FormatBase):
    """
    Terminal-optimized rendering backend for Console/SSH environments.
    
    Translates `Style` tokens into ANSI escape sequences, providing 
    bolding, centering, and character-based dividers (e.g., '─') to 
    maintain structural clarity in text-only interfaces.
    """
    px = 149
    
    @staticmethod
    def _print_center(t:str, style:Style=Style(), show:bool=True) -> str|None:
        """ANSI middle text → JN average text width is ~150"""
        l  = style.div_px
        tl = RgbANSI.ansi_text_len(t)
        fc = ' '*int(l/2-tl/2)
        fc = ' '*int((l-tl)/2) if len(fc) + tl > l else fc
        fc = fc + ( RgbANSI.bold(t) if style.bold else t )
        if not show:
            return fc
        print(fc)

    @classmethod
    def _align_px(cls, style:Style) -> int:
        """
        Resolve the divider width in characters.

        Uses the default terminal width (`px`) unless the style
        specifies a custom `div_px` ≥ 20.        
        """        
        return cls.px if not style.div_px or style.div_px < 20 else style.div_px

    @staticmethod
    def _align_div(style:Style) -> str | None:
        if style.div is None:
            return
        return '─' if style.div in DIV else str(style.div)

    @classmethod
    def display(cls, *values, style:Style=None, **kwargs) -> None:
        """
        Render values to the terminal.

        Parameters
        ----------
        *values
            Content to display.
        style : Style | None
            Style (used for sync_s resolution).
        **kwargs
            Forwarded to `sync_s`.
        """        
        val = [cls.sync_s(v, **kwargs) for v in values]        
        val = [v if isinstance(v, str) else ' '.join(v) for v in val]
        print(*val)
    
    @classmethod
    def point(cls, *values, style=Style(div='•')) -> None:
        """
        Render bullet-point list items in the terminal.

        Parameters
        ----------
        *values
            The bullet content.
        style : Style
            Bullet character.
        """        
        for v in values:
            cls.display(['', style.div, v], style=style)

    @classmethod
    def splitter(cls, 
        style: StyleSplitter = StyleSplitter(div='─', div_px=None), 
        show:  bool = True, 
        **kwargs
    ) -> str|None:
        """
        Render a horizontal divider line (character-based).

        Parameters
        ----------
        style : StyleSpliter
            Divider character and width.
        show : bool, default=True
            If True, print. If False, return the string.
        **kwargs
            Unused (interface compatibility).

        Returns
        -------
        str | None
            The divider line (when `show=False`).
        """          
        px  = cls._align_px(style)
        div = cls._align_div(style)
        if not div:
            return
        fs  = div*px
        if not show:
            return fs
        print(fs)

    @classmethod
    @ReportLogger.log
    def title(cls, 
        title: str, 
        *args, 
        style: StyleTitle = StyleTitle(div='─', div_px=None), 
        show:  bool = True,
        **kwargs
    ) -> str:    
        """
        Render a section title with character-based borders.

        Produces the format:

            ──────────────────────────────────
                  CENTERED BOLD TITLE
            ──────────────────────────────────

        Parameters
        ----------
        title : str
            The title text.
        *args
            Quantities for plural resolution.
        style : StyleTitle
            Title appearance (div char, width).
        show : bool, default=True
            If True, print. If False, return the string.
        **kwargs
            Forwarded to `sync_s`.

        Returns
        -------
        str
            The rendered title block.
        """  
        t   = cls.sync_s(title, *args, **kwargs)
        s_  = style.update(div = cls._align_div(style), div_px = cls._align_px(style))
        div = cls.splitter(s_, False)
        
        if style.center: t = cls._print_center(t, s_, False)
        if style.bold:   t = RgbANSI.bold(t)

        if not div:
            dt = t
        else:
            dt = div + '\n' + t + '\n' + div
        
        if not show:
            return dt
        
        print(dt)
        return dt
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TEXT FORMAT HTML-ANSI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class FormatUI(FormatBase):
    """
    Factory for selecting the active formatting backend.

    Provides `get_formatter` to dynamically return the
    appropriate backend class (`FormatHTML` or `FormatANSI`)
    based on the execution environment.
    """
    
    @staticmethod
    def get_formatter(UI: Literal['html', 'ansi']='html'):
        """
        Return the formatting backend class for the specified
        environment.

        Parameters
        ----------
        UI : 'html' | 'ansi', default='html'
            The rendering environment.

        Returns
        -------
        type[FormatHTML] | type[FormatANSI]
            The backend class (not an instance — methods are
            called as classmethods/staticmethods).
        """        
        if UI == 'html': return FormatHTML
        if UI == 'ansi': return FormatANSI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬