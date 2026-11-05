"""
Cross-Platform Visual Identity and Color Transformation Engine.

This module implements a universal coloring system designed to maintain 
visual consistency across diverse execution environments—including Jupyter 
Notebooks, standard terminals (CLI), and data visualization frameworks like 
Matplotlib and Seaborn. It abstracts the complexity of color representation 
by providing a single interface for RGB, HEX, and ANSI formatting.

Key architectural capabilities:
1. Polymorphic Formatting: Employs an abstract base class (`RgbBase`) to 
   dynamically generate style methods (e.g., `bold_red()`, `under_blue()`), 
   allowing the system to switch between HTML and ANSI renderers seamlessly 
   via the `RgbUI` gateway.
   
2. Mathematical Color Approximation: Implements a color-distance algorithm to 
   map high-precision RGB values to the closest available 256-color ANSI 
   sequence, ensuring professional branding even in restricted terminal environments.

3. Automated Palette Synthesis: Provides pre-defined, industry-standard palettes 
   (e.g., Tableau 10/100) and automatically scales them to [0, 1] ranges for 
   compatibility with scientific plotting libraries.

4. Bidirectional Format Conversion: Features a robust translation layer 
   (`ANSIToHTML`) that converts terminal-style escape codes into clean HTML 
   spans, enabling the reuse of CLI-formatted logs within web-based dashboards.
"""

from ..deps.common  import re, ABC, abstractmethod
from ..deps.display import HTML
from ..deps.types   import Literal


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ANSI ESCAPE CODES → GEN BASED ON RGB
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ANSI:    
    """
    Low-level handler for ANSI escape sequence generation and RGB mapping.
    
    Responsible for the mathematical derivation of the 256-color terminal 
    cube and providing the logic to transform RGB tuples into OS-agnostic 
    escape sequences.
    """
    
    @staticmethod
    def ansi_256_colors_as_rgb():
        """All extended ANSI colors as RGB"""
        
        # 16 basic colors
        colors = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
            (0, 128, 128), (192, 192, 192), (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
            (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)
        ]
        
        # 216 colors in a 6x6x6 cube
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    colors.append((r * 42, g * 42, b * 42))

        # 24 grayscale colors
        for i in range(24):
            gray_value = int(i * 10 + 8)
            colors.append((gray_value, gray_value, gray_value))

        return colors
    
    ansi_256_colors_as_rgb_ = ansi_256_colors_as_rgb()
    
    @staticmethod
    def ansi_256_to_escape_sequence(code, foreground=True):
        """All extended ANSI sequence colors based on logic in ansi_256_colors_as_rgb"""
        if 0 <= code <= 7:
            return f'\033[{30 + code}m'
        elif 8 <= code <= 15:  # High-intensity colors
            return f'\033[{90 + (code - 8)}m' if foreground else f'\033[{100 + (code - 8)}m'        
        elif 16 <= code <= 231:
            r = (code - 16) // 36 * 42
            g = ((code - 16) % 36) // 6 * 42
            b = ((code - 16) % 6) * 42
            return f'\033[38;2;{r};{g};{b}m' if foreground else f'\033[48;2;{r};{g};{b}m'
        elif 232 <= code <= 255:
            gray = (code - 232) * 10 + 8
            return f'\033[38;2;{gray};{gray};{gray}m' if foreground else f'\033[48;2;{gray};{gray};{gray}m'
        else:
            return '\033[30m'
    
    @staticmethod
    def _color_distance(c1:list, c2:list): 
        """Distance between 2 rgb colors to approximate ansi"""           
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ANSI | RGB | HEX → COLORS TRANSFORMATION FX
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Cfx:
    """
    High-performance color transformation utility suite.
    
    Provides a set of static tools for blending, filtering, and converting 
    between HEX, RGB, and ANSI formats, ensuring that colors remain visible 
    and aesthetically consistent across different backgrounds (e.g., Luminance Filtering).
    """
    
    @staticmethod
    def rgb_to_hex(rgb:list):
        """Transforms [r,g,b] color to #hex representation"""
        if all([c <=1 for c in rgb]): rgb = [int(c*255) for c in rgb]        
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    @staticmethod
    def hex_to_rgb(hex_color:str):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
    @classmethod    
    def hex_blend_colors(cls, hex_colors:list):
        """Blend multiple hex colors into one."""
        if not hex_colors:
            return "#000000"  
        rgb_colors = [cls.hex_to_rgb(hex_color) for hex_color in hex_colors]
        avg_rgb = tuple(
            sum(component) // len(component)
            for component in zip(*rgb_colors)
        )
        return cls.rgb_to_hex(avg_rgb) 
    
    @staticmethod
    def rgb_to_ansi(rgb:list):
        """Transforms rgb colors to closest ANSI color"""
        ansi_colors  = ANSI.ansi_256_colors_as_rgb_
        closest_ansi = min(range(len(ansi_colors)), key=lambda i: ANSI._color_distance(rgb, ansi_colors[i]))                
        return ANSI.ansi_256_to_escape_sequence(closest_ansi)   
        
    @classmethod
    def hex_to_ansi(cls, hex_color:str):
        """Transforms hex colors to closest ANSI color"""
        rgb_color = cls.hex_to_rgb(hex_color)
        return cls.rgb_to_ansi(rgb_color)    
    
    @classmethod
    def rgb_hex_ansi(cls, colors:list):
        """Generates rgb-hex-ansi color lists available across GUI"""
        colors_hex  = []
        colors_ansi = []
        for i, rgb in enumerate(colors):  
            colors_hex.append(  cls.rgb_to_hex(rgb)  )    
            colors_ansi.append( cls.rgb_to_ansi(rgb) )  
            colors[i] = [c / 255. for c in rgb]  # Scale the RGB values to the [0, 1] range, which is the format matplotlib and seaborn accepts             
        return colors, colors_hex, colors_ansi
         
    @staticmethod
    def ansi_text_len(t):
        """String length corrected by ansi seqeuences"""
        tl = len(str(t))
        for r in RgbANSI._all:
            if r in t: tl = tl - len(r)*t.count(r)        
        return tl  
    
    @staticmethod
    def rgb_filter_colors(colors:list, threshold=200):
        """Filter out colors that are close to white or light grey."""            
        def luminance(r, g, b):  return 0.299 * r + 0.587 * g + 0.114 * b                
        def is_visible(r, g, b): return luminance(r, g, b) < threshold                
        return [color for color in colors if is_visible(*color)]
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# RGB Themes
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Themes:

    # Tableau colors as RGB   
    # ─────────────────────────────────────────────────────────── 
    tableau20 = [
        ( 31, 119, 180), (174, 199, 232), (255, 127,  14), (255, 187, 120), ( 44, 160,  44), (152, 223, 138), (214,  39,  40),
        (255, 152, 150), (148, 103, 189), (197, 176, 213), (140,  86,  75), (196, 156, 148), (227, 119, 194), (247, 182, 210),
        (127, 127, 127), (199, 199, 199), (188, 189,  34), (219, 219, 141), ( 23, 190, 207), (158, 218, 229)
    ]    
    
    tableau100 = [
        (214,  39,  40), ( 44, 160,  44), ( 31, 119, 180), (255, 127,  14), (148, 103, 189), (140,  86,  75), (227, 119, 194), (127, 127, 127), (188, 189,  34), ( 23, 190, 207),
        (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148), (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
        (255, 255, 255), (197,  62,  39), (127, 127, 127), (255, 127,  31), (255, 127, 180), (255, 127, 227), (255, 127,   0), (255,  86, 140), ( 44, 160, 160), (214, 214, 214),
        (140, 214,  75), (214, 214,  39), (140, 214,  44), (214,  86,  75), ( 44,  86,  86), ( 75, 214, 214), ( 75,  86,  75), (214, 214, 119), (140, 214,  86), (214,  39, 140),
        (196, 103, 140), ( 75, 214,  75), ( 44,  86, 214), ( 75,  44, 140), ( 44, 214,  44), ( 44,  44,  44), (140,  86,  44), ( 86,  86, 214), (196,   0, 196), (214, 119, 214),
        (180,  39, 140), (180,  39, 180), ( 86,  39, 180), ( 75, 214, 139), (214, 214, 214), (127, 214, 214), (214,  86, 214), (214,  39, 119), (127, 214, 140), ( 44,  39, 180),
        (255, 255,   0), (196, 196, 196), (255, 127, 139), (255, 127, 197), (140, 127, 214), ( 44, 214,  75), (255, 127, 255), ( 86,  44,  86), ( 86, 127, 127), (199, 234, 229),
        (237, 204, 255), (174, 204, 235), (255, 217, 139), (199, 199, 199), (158, 158, 158), (237, 204, 204), (199, 199,   0), (196,   0, 148), (197, 176, 255), (127, 127,   0),
        (148, 103,   0), (140,  86, 255), (156, 148, 148), (174, 119, 119), (255, 204,   0), (180, 180,   0), (127, 127, 196), (180, 119, 180), ( 23, 190, 139), (227, 119, 119),
        ( 31, 127,  23), (255, 119, 214), (140,  86, 140), (152, 223, 197), (180, 119, 197), (255, 127, 255), (196, 103,  75), (196, 103, 103), (199, 199, 237), (148,  86,  75)
    ]        

    tableau100_keep_distinct_sorted = [
        (214,  39,  40), ( 44, 160,  44), ( 31, 119, 180), (255, 127,  14), ( 60, 240, 140), (140,  86,  75), (216, 233,  35), (188, 189,  34), (180,  39, 140), (255, 127, 227),
        ( 44,  86, 214), (174, 199, 232), ( 75,  86,  75), (255, 127,  14), (127, 127, 127), ( 44, 214,  44), (255, 152, 150), (196, 103, 103), (148, 103, 189), ( 23, 190, 207),
        (140, 214,  86), ( 86,  39, 180), (127, 127,   0), (255, 127, 180), (255, 127,   0), (255,  86, 140), ( 44, 160, 160), (140, 214,  75), (214, 214,  39), (140, 214,  44),
        (214,  86,  75), ( 44,  86,  86), ( 75, 214, 214), (214,  39, 140), (196, 103, 140), ( 75, 214,  75), ( 75,  44, 140), ( 44,  44,  44), (140,  86,  44), ( 86,  86, 214),
        (196,   0, 196), (214, 119, 214), (180,  39, 180), (127, 214, 214), (214,  86, 214), (214,  39, 119), (127, 214, 140), ( 44,  39, 180), (196, 196, 196), (255, 127, 139),
        (255, 127, 197), (140, 127, 214), ( 44, 214,  75), (174, 204, 235), (199, 199, 199), (158, 158, 158), (199, 199,   0), (196,   0, 148), (197, 176, 255), (148, 103,   0),
        (255, 127, 255), ( 86,  44,  86), ( 86, 127, 127), (255, 204,   0), (180, 180,   0), (127, 127, 196), (180, 119, 180), ( 23, 190, 139), (227, 119, 119), ( 31, 127,  23),
        (140,  86, 255), (156, 148, 148), (174, 119, 119), (255, 119, 214), (140,  86, 140), (152, 223, 197), (180, 119, 197), (196, 103,  75), (148,  86,  75)
    ]

    tableau100 = tableau100_keep_distinct_sorted
               
               
    # Filter to only clearly visible colors on white background
    # ───────────────────────────────────────────────────────────
    tableau100 = Cfx.rgb_filter_colors(tableau100)         
    
    
    # Hex | ansi | representation | rgb standardization
    # ───────────────────────────────────────────────────────────
    tableau20,  tableau20_hex,  tableau20_ansi  = Cfx.rgb_hex_ansi(tableau20)
    tableau100, tableau100_hex, tableau100_ansi = Cfx.rgb_hex_ansi(tableau100)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# RGB GEN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
def _set_name_doc(name: str, doc: str):
    """
    Decorator factory: assigns __name__, __doc__, and type hints.
    """
    def decorator(func):
        func.__name__ = name
        func.__doc__ = doc
        func.__annotations__ = {'text': str, 'return': str}
        return func
    return decorator

class RgbBase(ABC, Cfx, Themes):
    """
    Base class for text formatting (HTML, ANSI, ...)
    
    Construct to create set of methods such as:
    RGB_ANSI.red("Red text")
    RGB_ANSI.bold_red("Bold red text")
    
    RGB_HTML.red("Red text")
    RGB_HTML.bold_red("Bold red text")
    """

    COLORS = {
        'red':      'red',
        'green':    'green',
        'blue':     'blue',
        'purple':   'purple',
        'cyan':     'cyan',
        'yellow':   'orange',
        'grey':     'grey',
        'black':    'black'
    }

    BOLD  = 'bold'
    UNDER = 'under'
    STYLE = [BOLD, UNDER, BOLD+'_'+UNDER]

    @classmethod
    @abstractmethod
    def span(cls, text, color="black", bold=False, underline=False):
        """Abstract method to format text with color and style."""

    @classmethod
    def register_methods(cls):
        """Dynamically register color methods for this formatter."""
        
        for name, color in cls.COLORS.items():
            @_set_name_doc(name, f"Text in {color}")
            def method(text:str, color=color) -> str:
                return cls.span(text, color=color)
            setattr(cls, name, staticmethod(method))

        for name, color in cls.COLORS.items():
            @_set_name_doc(name, f"Text in bold {color}")
            def bold_method(text:str, color=color) -> str:
                return cls.span(text, color=color, bold=True)
            setattr(cls, "bold_" + name, staticmethod(bold_method))
    
        for name in cls.STYLE:
            bold  = ( cls.BOLD  in name)
            under = ( cls.UNDER in name)
            @_set_name_doc(name, f"Text in {name} black")
            def style_method(text:str, bold=bold, under=under) -> str:
                return cls.span(text, bold=bold, underline=under)
            setattr(cls, name, staticmethod(style_method))
    
    @staticmethod
    def convert_ansi_to_html(text:str) -> str:
        """Coverts ANSI formatted text to HTML formatting."""
        return ANSIToHTML.convert(text)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.register_methods()


    @staticmethod 
    @abstractmethod
    def red(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def green(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def blue(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def purple(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def cyan(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def yellow(text: str) -> str: ...
    
    @staticmethod 
    @abstractmethod
    def bold_red(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def bold_green(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def bold_blue(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def bold_purple(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def bold_cyan(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def bold_yellow(text: str) -> str: ...
    
    @staticmethod 
    @abstractmethod
    def bold_grey(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def bold(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def under(text: str) -> str: ...

    @staticmethod 
    @abstractmethod
    def bold_under(text: str) -> str: ... 
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def _adjust(color:str) -> str:
    if color == 'cyan': return '#259999'
    return color

class RgbHTML(RgbBase):
    """Coded HTML formatter for Jupyter Notebooks and Web-based GUIs."""

    @classmethod
    def span(cls, text, color="", bold=False, underline=False):
        color = f'color: {_adjust(color)}; ' if color else ''
        style = f'{color}font-weight: {"600" if bold else "normal"};'
        if underline:
            style += " text-decoration: underline;"
        return f'<span style="{style}">{text}</span>'
# ─────────────────────────────────────────────────────────────────────────────────────────────────

class RgbANSI(RgbBase):
    """SGR (Select Graphic Rendition) formatter for CLI and Terminal consoles."""

    COLORS = {
        'red':      '\033[31m',
        'green':    '\033[32m',
        'blue':     '\033[34m',
        'purple':   '\033[35m',
        'cyan':     '\033[36m',
        'yellow':   '\033[33m',
        'grey':     '\033[90m',
        'black':    '\033[30m'
    }
    
    _END  = '\033[0m'
    _BOLD = "\033[1m"
    _UNDER= "\033[4m"

    _all = list(COLORS.values()) + [_END, _BOLD, _UNDER]

    @classmethod
    def span(cls, text, color="", bold=False, underline=False):        
        code = ''
        if bold:      code += cls._BOLD
        if underline: code += cls._UNDER
        code+=color
        return code + str(text) + cls._END
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class ANSIToHTML:        
    """
    Translator for converting terminal-formatted text into HTML spans.
    
    Parses ANSI escape sequences using regex and maps them to corresponding 
    inline CSS styles, allowing CLI logs to be rendered natively in browser-based 
    interfaces without losing their semantic coloring.
    """

    style_map = {code: f'<span style="color:{_adjust(color)}">' for color, code in RgbANSI.COLORS.items()}    
    style_map = style_map | {
        RgbANSI._END:  '</span>', 
        RgbANSI._BOLD: '<span style="font-weight:bold">',
        RgbANSI._UNDER:'<span style="text-decoration:underline">'
    }
    ANSI_PATTERN = re.compile(r'(\x1b\[[\d;]*m)')

    @classmethod
    def convert(cls, text: str) -> str:
        """Coverts ANSI formatted text to HTML formatting."""
        parts = cls.ANSI_PATTERN.split(text)
        
        output = []
        open_tags = 0
        
        for part in parts:
            if part in cls.style_map:
                html_tag = cls.style_map[part]
                
                # End tag, close ALL open spans
                if html_tag == '</span>':
                    if open_tags > 0:
                        output.append('</span>' * open_tags)
                        open_tags = 0
                else:
                    # Start tag
                    output.append(html_tag)
                    open_tags += 1
            
            # ANSI code we don't recognize, ignore it (don't print raw code)
            elif cls.ANSI_PATTERN.match(part):
                continue
                
            # Regular text
            else:
                output.append(part)
        
        # Sanity check → close any tags left open at the very end of the string
        if open_tags > 0:
            output.append('</span>' * open_tags)
            
        return "".join(output).replace('\n', '<br>')

class HTMLToANSI:
    """Converts inline-style HTML spans to ANSI escape codes or plain text."""

    TOKEN = re.compile(r'<span style="([^"]*)">|</span>')

    @classmethod
    def _parse_style(cls, style: str) -> str:
        codes = []
        for prop in style.split(";"):
            if ":" not in prop:
                continue
            key, value = prop.split(":", 1)
            key, value = key.strip().lower(), value.strip().lower()

            if key == "color" and value in RgbANSI.COLORS:
                codes.append(RgbANSI.COLORS[value])

            elif key == "font-weight" and value in ("bold", "600", "700", "800", "900"):
                codes.append("\x1b[1m")

            elif key == "text-decoration" and "underline" in value:
                codes.append("\x1b[4m")

        return "".join(codes)

    @classmethod
    def convert(cls, text: str) -> str:
        """HTML spans → ANSI escape sequences."""
        output = []
        pos = 0
        for m in cls.TOKEN.finditer(text):
            if m.start() > pos:
                output.append(text[pos:m.start()])

            if m.group(0) == "</span>":
                output.append("\x1b[0m")

            else:
                output.append(cls._parse_style(m.group(1)))

            pos = m.end()

        if pos < len(text):
            output.append(text[pos:])

        return "".join(output)

    @classmethod
    def strip(cls, text: str | HTML) -> str:
        """HTML spans → plain text (for log files)."""
        if isinstance(text, HTML):
            text = text.data
        return re.sub(r"<[^>]+>", "", text).strip()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# RGB HTML-ANSI
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class RgbUI:
    """
    The central Factory for selecting the active environment formatter.
    
    Provides the `get_formatter` method to dynamically select the 
    appropriate backend (`html` or `ansi`) based on the current user environment.
    """

    @staticmethod
    def get_formatter(UI:Literal['html', 'ansi']='html'):
        if UI == 'html':
            return RgbHTML
        else:
            return RgbANSI           
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬