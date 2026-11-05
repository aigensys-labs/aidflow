"""
Visual Presentation Engine and Async Feedback System.

This module implements the final aesthetic layer of AIDflow, providing 
specialized utilities for high-fidelity data styling and real-time execution 
feedback. It ensures that the transition from raw computation to user 
presentation is seamless, professional, and visually intuitive.
"""

from ..deps.common      import  (
                                os, 
                                math, 
                                time,
                                threading,
                                contextmanager
                                )
from ..deps.data        import  pDF
from ..deps.data_view   import  JavascriptFunction
from ..deps.display     import  (
                                wg, 
                                HTML, 
                                display_html, 
                                display, 
                                clear_output
                                )
from ..utils.builtins   import  List


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ITABLES STYLING
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ItablesStyle:
    """
    JavaScript cell-formatting functions for itables.

    Provides reusable `createdCell` handlers that are injected
    into itables DataTable configurations. Each method returns
    a ``JavascriptFunction`` instance that can be passed to
    the itables `createdCell` option.
    """

    @staticmethod
    def bool_color(var: str, colors: tuple = ('#2BCF91', '#B50064')
    ) -> JavascriptFunction:
        """
        Color-format a boolean column: True → green, False → red.

        Parameters
        ----------
        var : str
            Column name to format.
        colors : tuple, default=('#2BCF91', '#B50064')
            (true_color, false_color) hex values.

        Returns
        -------
        JavascriptFunction
            itables ``createdCell`` handler.
        """
        return JavascriptFunction(f"""
        function (td, cellData, rowData, row, col) {{            
            const colName = this.api().column(col).header().textContent.trim();

            if (colName === "{var}") {{
                const isTrue = cellData === true || cellData === "True" || cellData === 1;
                const isFalse = cellData === false || cellData === "False" || cellData === 0;

                if (isTrue) {{
                    td.style.setProperty('color', "{colors[0]}", 'important');
                }} else if (isFalse) {{
                    td.style.setProperty('color', "{colors[1]}", 'important');
                }}
            }}
        }}
        """)
    
    @staticmethod
    def number_color(th:float=0, color:str='#B50064') -> None:
        """
        Conditionally color numeric values exceeding a threshold.

        Parameters
        ----------
        th : float, default=0
            Threshold — values strictly greater than this are
            colored.
        color : str, default='#B50064'
            Hex color to apply.

        Returns
        -------
        JavascriptFunction
            itables ``createdCell`` handler.
        """
        return JavascriptFunction(f"""
        function (td, cellData, rowData, row, col) {{
            const rawValue = this.api().cell(row, col).render('sort');
            
            if (typeof rawValue === 'number' && rawValue > {th}) {{
                td.style.setProperty('color', "{color}", 'important');
                td.style.setProperty('font-weight', '500', 'important');
            }};
        }}
        """)

    @staticmethod
    def replace_nan() -> JavascriptFunction:
        """
        Replace all NaN-like values (``None``, ``"NaN"``,
        ``"nan"``, JS ``NaN``) with empty strings.

        Returns
        -------
        JavascriptFunction
            itables ``createdCell`` handler.
        """
        return JavascriptFunction(f"""
        function (td, cellData, rowData, row, col) {{
            // Replace NaN by ''
            const isMissing = 
                    cellData === null || 
                    cellData === undefined || 
                    cellData === "NaN" || 
                    cellData === "nan" || 
                    (typeof cellData === 'number' && isNaN(cellData)) ||
                    (td.innerHTML && td.innerHTML.trim() === "NaN");

            if (isMissing) {{
                td.innerHTML = "";
                return;             
            }}
        }}
        """)

    @staticmethod
    def combine(*js_funcs) -> JavascriptFunction:
        """
        Combine multiple ``createdCell`` handlers into a single
        function (applied in order).

        Parameters
        ----------
        *js_funcs
            Individual ``JavascriptFunction`` instances to
            compose.

        Returns
        -------
        JavascriptFunction
            A single ``createdCell`` handler that invokes all
            provided handlers sequentially.
        """
        calls = []
        for fn in js_funcs:
            code = str(fn).strip()
            # IIFE with .call(this, ...) preserves DataTables `this.api()` context and scope
            calls.append(f"({code}).call(this, td, cellData, rowData, row, col);")

        js_body = "\n".join(calls)
        return JavascriptFunction(f"""
        function(td, cellData, rowData, row, col) {{
            {js_body}
        }}
        """)   
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
        


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DF COLOR HIGLIGHT
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class PdStyle:
    """
    Cell-level color mapping for pandas DataFrame styling.

    Provides utilities to apply per-cell CSS colors (text or
    background) based on a nested dictionary lookup — useful
    for highlighting specific rows/columns in diagnostic tables.
    """

    @staticmethod
    def df_color_cells(col, row, color_dict, color_bg) -> str:
        """
        Resolve the CSS style string for a specific cell.

        Parameters
        ----------
        col : hashable
            Column label.
        row : hashable
            Row label (index).
        color_dict : dict
            Nested mapping: ``{col: {row: 'color_hex'}}``.
        color_bg : str
            CSS property: ``'color'`` or ``'background-color'``.

        Returns
        -------
        str
            CSS declaration (e.g., ``'color: #B50064'``) or
            empty string if no style is defined.
        """ 
        color = color_dict.get(col, {}).get(row, '')             
        return f'{color_bg}: {color}' if color else ''
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def df_apply_styles(cls, df:pDF, color_dict:dict, color_bg:str='color') -> pDF:
        """
        Generate a full style matrix (same shape as ``df``) with
        CSS declarations from the color mapping.

        Parameters
        ----------
        df : pDF
            The DataFrame (defines the shape).
        color_dict : dict
            Nested mapping: ``{col: {row: 'color'}}``.
        color_bg : str, default='color'
            CSS property to apply.

        Returns
        -------
        pDF
            DataFrame of CSS strings (same index/columns as
            ``df``).
        """
        styles = pDF('', index=df.index, columns=df.columns)
        
        for col in df.columns:
            for row in df.index:
                styles.loc[row, col] = cls.df_color_cells(col, row, color_dict, color_bg)    
        
        return styles
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def df_apply_color_style(cls, df:pDF, color_dict:dict, color_bg='color'):
        """
        Apply a cell color mapping to a DataFrame and return the
        styled object (for notebook display).

        Parameters
        ----------
        df : pDF
            The target DataFrame.
        color_dict : dict
            Nested mapping: ``{column: {row: 'color'}}``.
        color_bg : str, default='color'
            CSS property: ``'color'`` or ``'background-color'``.

        Returns
        -------
        pandas.io.formats.style.Styler
            Styled DataFrame ready for display.
        """      
        df_ = df.copy()
        return df_.style.apply(lambda x: cls.df_apply_styles(df_, color_dict, color_bg), axis=None)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# Styling Tables in Dash
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class TableStyle:
    """
    HTML layout engine for multi-table presentations.

    Provides tools for arranging multiple DataFrames or HTML
    tables into cohesive grid layouts, preventing the vertical
    scroll fatigue typical of sequential Jupyter outputs.
    """

    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    # DISPLAY TABLES NEXT TO EACH OTHER
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    @staticmethod
    def combine_pdf_tables( 
        dfs:         list[pDF|str], 
        names:       list[str] = None, 
        bg_color:    str = '#EAEDF3', 
        header_bold: bool|list = True, 
        max_tables:  int = 6, 
        show:        bool = True
    ) -> str | None:    
        """
        Combine multiple DataFrames (or HTML strings) into a
        single inline HTML table layout.

        Parameters
        ----------
        dfs : list
            DataFrames or HTML strings to display side-by-side.
        names : list | None, default=None
            Header labels for each table.
        bg_color : str, default='#EAEDF3'
            Cell background color.
        header_bold : bool | list, default=True
            If True, all headers are bold. If a list, only the
            tables at those indices get bold headers.
        max_tables : int, default=6
            Maximum tables per row before wrapping.
        show : bool, default=True
            If True, render in notebook. If False, return the
            HTML string.

        Returns
        -------
        str | None
            HTML string (when ``show=False``) or None.
        """     
        count = 0
        names = names or []

        dfs = List.ensure(dfs)
        if not names: names = ['' for x in range(len(dfs))]
        names = List.ensure(names)

        b  = 'font-weight:bold;'
        n  = 'font-weight:normal;'

        th = [n]*len(dfs)
        
        # Determine which headers should be bold based on header_bold argument
        if isinstance(header_bold, list):
            for i in header_bold:
                if i not in range(len(dfs)):
                    raise ValueError('When header_bold is provided as a list, values represent tables position for which headers will be bold')
                th[i] = b
        elif header_bold:
            th = [b]*len(dfs)

        html_str = ''   # Initialize HTML string to build the table
        html_th  = ''   # Accumulator for <th> elements (table headers)
        html_td  = ''   # Accumulator for <td> elements (table data)

        for i, (df, name) in enumerate(zip(dfs, names)):
            df_html = df.to_html(index=True) if not isinstance(df, str) else df
            
            # columns in a row
            if count < (max_tables):
                if name: html_th += (''.join(f'<th style="text-align:center; font-size: 9pt; {th[i]}">{name}</th>'))
                html_td += (''.join(f'<td style="vertical-align:top; background-color:{bg_color}"> {df_html}</td>'))
                count += 1

            # new row
            else:
                html_str += f'<tr>{html_th}</tr><tr>{html_td}</tr>'
                if name: html_th = f'<th style="text-align:center; font-size: 9pt; {th[i]}">{name}</th>'
                html_td = f'<td style="vertical-align:top; background-color:{bg_color}"> {df_html}</td>'
                if name and name != names[-1]: count = 0

        if count != 0:
            html_str += f'<tr>{html_th}</tr><tr>{html_td}</tr>'
        
        html_str = f'<table>{html_str}</table>'
        html_str = html_str.replace('<table',f'<table style="display:inline"')
        
        if show: display_html(html_str, raw=True)
        else: return html_str
    # ─────────────────────────────────────────────────────────────────────────────────────────────            

        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # COMBINE MULTIPLE HTML TABLES NEXT TO EACH OTHER
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def combine_html_tables(
        tables:         list[str], 
        cols_per_row:   int = 2,
        gap_horizontal: int = 50, 
        gap_vertical:   int = 20, 
        align:          str = "left", 
        show:          bool = True
    ) -> str | None:
        """
        Arrange HTML tables in a flexbox grid layout.

        Parameters
        ----------
        tables : list[str]
            HTML table strings to arrange.
        cols_per_row : int, default=2
            Number of tables per row.
        gap_horizontal : int, default=50
            Horizontal gap in pixels.
        gap_vertical : int, default=20
            Vertical gap in pixels.
        align : str, default='left'
            Horizontal alignment: ``'left'``, ``'center'``,
            ``'right'``, ``'between'``, ``'around'``, ``'evenly'``.
        show : bool, default=True
            If True, render in notebook. If False, return HTML.

        Returns
        -------
        str | None
            HTML string (when ``show=False``) or None.
        """
        align_map = {
            "left": "flex-start",
            "center": "center",
            "right": "flex-end",
            "between": "space-between",
            "around": "space-around",
            "evenly": "space-evenly"
        }
        justify_value = align_map.get(align, "flex-start")  # Default to left alignment

        # Calculate number of rows needed
        num_tables = len(tables)
        num_rows = math.ceil(num_tables / cols_per_row)     # Round up to get full rows
        
        combined_html = f'<div style="display: flex; flex-direction: column; gap: {gap_vertical}px;">'  # Outer container (vertical)

        for row in range(num_rows):
            row_tables = tables[row * cols_per_row : (row + 1) * cols_per_row]  # Extract tables for this row
            row_html = f'<div style="display: flex; gap: {gap_horizontal}px; justify-content: {justify_value}; align-items: start;">'
            for table in row_tables:
                row_html += f'<div>{table.strip()}</div>'
            row_html += '</div>'
            combined_html += row_html  # Append row to main container

        combined_html += '</div>'
        
        if not show: 
            return combined_html
        display_html(combined_html, raw=True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────         

    @classmethod
    def combine_pdf_to_html(cls,
        pdf_tables: list|dict, 
        titles:     list|dict = None, 
        tables_per_block: int = 2,
        p_tables_per_row: int = 4,
        h_cols_per_row:   int = 3,
        h_gap_horizontal: int = 20,
        h_gap_vertical:   int = 20, 
        h_align:          str = "left",         
        show:            bool = True
    ) -> str | None:
        """
        Combine multiple DataFrames into a two-level HTML layout:
        inner blocks (tables side-by-side) arranged in an outer grid.

        Applies in sequence:
        >>> html = combine_pdf_tables(pdf_tables, titles, max_tables=p_tables_per_row, show=False)
        >>> return combine_html_tables(html, h_cols_per_row, h_gap_horizontal, h_gap_vertical, h_align, show=show)
        
        Parameters
        ----------
        pdf_tables : list | dict
            DataFrames to display.
        titles : list | dict | None, default=None
            Labels for each table.
        tables_per_block : int, default=2
            Number of tables per inner block.
        p_tables_per_row : int, default=4
            Max tables per row within each block.
        h_cols_per_row : int, default=3
            Number of blocks per row in the outer grid.
        h_gap_horizontal : int, default=20
            Horizontal gap between blocks (px).
        h_gap_vertical : int, default=20
            Vertical gap between block rows (px).
        h_align : str, default='left'
            Outer grid alignment.
        show : bool, default=True
            If True, render in notebook. If False, return HTML.

        Returns
        -------
        str | None
            Combined HTML string (when ``show=False``) or None.

        Raises
        ------
        TypeError
            If ``pdf_tables`` is a dict and ``titles`` is a
            list (or has mismatched keys).
        """
        titles = titles or []
        _is_d_dict = isinstance(pdf_tables, dict)
        _is_t_dict = isinstance(titles, dict)
        if ( 
            _is_d_dict
            and len(titles)
            and ( (not _is_t_dict) or [k for k in pdf_tables if k not in titles] )
        ):
            raise TypeError(
                'AIG SAFETY ► When both tables and titles are provided, ' 
                'they must be of the same type: either list or dict with same keys'
            )
        
        ld = len(pdf_tables)
        lt = len(titles)
        if (not _is_t_dict) and (ld != lt):
            titles = ['']*ld

        r = range(0, ld, tables_per_block)
        if _is_d_dict:
              s = lambda x: [x[k] for k in x]
        else: s = lambda x: [x[i:i+tables_per_block] for i in r]

        split_d = s(pdf_tables)
        split_t = s(titles)
        
        comb = []
        for d,t in zip(split_d, split_t):
            comb.append( cls.combine_pdf_tables(d, t, max_tables=p_tables_per_row, show=False ) )

        return cls.combine_html_tables(
            comb, 
            cols_per_row   = h_cols_per_row, 
            gap_horizontal = h_gap_horizontal,
            gap_vertical   = h_gap_vertical,
            align = h_align,
            show  = show 
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────         


    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    # DISPLAY CLICKABLE TAB
    # ─────────────────────────────────────────────────────────────────────────────────────────────            
    @staticmethod
    def clickable(data:dict|list|pDF, tab:str=None, show:bool=True, hide_columns:bool=True):
        """
        Render a table with clickable file-path links.

        Transforms specified column values into HTML hyperlinks
        (opening in a new tab) — useful for navigating to saved
        model artifacts or report files.

        Parameters
        ----------
        data : dict | list | pDF
            Data to display (wrapped in a DataFrame if not one).
        tab : str | None, default=None
            Column containing file paths to make clickable.
            If None, the last column is used.
        show : bool, default=True
            If True, render in notebook. If False, return the
            styled object.
        hide_columns : bool, default=True
            If True, hide the column header row.

        Returns
        -------
        Styler | None
            Styled table (when ``show=False``) or None.
        """
        def make_clickable(url):
            link = os.path.basename(url)
            return f'<a href="{url}" Target="_blank">{link}</a>'
        
        if not isinstance(data, pDF): 
            if isinstance(data, dict): data = [data]
            data = pDF(data)
        
        if tab is None:
            tab = data.columns[-1]

        dash = data.style.format({tab: make_clickable}).hide(axis=0)
        if hide_columns: dash = dash.hide(axis=1)
        
        if show: display(dash)
        else: return dash
    # ─────────────────────────────────────────────────────────────────────────────────────────────         
    

    # ─────────────────────────────────────────────────────────────────────────────────────────────         
    # HTML ENHACEMENT
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def format_html(elements: list) -> str:   
        """
        Wrap HTML elements in a standalone document with
        professional CSS styling (typography, spacing, table
        formatting, zebra striping, hover states).

        Parameters
        ----------
        elements : list
            HTML strings to include as sections.

        Returns
        -------
        str
            Complete HTML document string.
        """
        el = ''
        for e in elements:
            el += f'<div class="section">{e}</div>'

        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <style>
            body {{ 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
                padding: 20px; 
                background-color: #f8fafc; /* Soft slate background */
                color: #1e293b;
            }}            
            .container {{ 
                display: flex; 
                flex-direction: column; 
                gap: 30px; 
                max-width: 1450px; 
                margin: 0 auto; 
            }}
            .section {{ 
                background: #ffffff; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }}
            
            /*===============================================================
                                    TABLE STYLING
            ================================================================= */

            /* 2. Structural Resets*/
            td[style*="background-color:#EAEDF3"] {{
                background-color: transparent !important;
                padding: 10px !important; 
            }}
            
            table {{
                border-collapse: collapse;
                border-spacing: 0;
                margin: 0;
                padding: 0;
            }}

            table[id^="T_"] {{
                display: table !important;
                width: 100%;
                background: linear-gradient(145deg, #ffffff 0%, #f4f7fb 100%);
                border-radius: 8px; 
                overflow: hidden; 
                box-shadow: 0 5px 15px rgba(15, 23, 42, 0.05), 0 4px 6px rgba(15, 23, 42, 0.02);
                margin-bottom: 24px;
                border: 1px solid #e2e8f0;
            }}

            /* --- THE HEADERS --- */
            table[id^="T_"] thead th {{
                background-color: #E4EBF0;
                color: #475569;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.05em;
                padding: 12px 16px;
                border-bottom: 2px solid #e2e8f0; 
                text-align: left;
            }}

            /* --- THE DATA & THE INDEX --- */
            table[id^="T_"] tbody th,
            table[id^="T_"] tbody td {{
                padding: 8px 16px; 
                border-bottom: 1px solid #f1f5f9;
                font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
                font-size: 13px;
                font-weight: 400;
                text-align: left;
                color: #334155; 
                background-color: transparent; 
            }}

            /* --- ZEBRA STRIPING --- */
            table[id^="T_"] tbody tr:nth-child(even) th,
            table[id^="T_"] tbody tr:nth-child(even) td {{
                background-color: #F2F5F7; 
            }}

            /* Remove the bottom border from the very last row for a clean edge */
            table[id^="T_"] tbody tr:last-child th,
            table[id^="T_"] tbody tr:last-child td {{
                border-bottom: none;
            }}

            /* --- SUBTLE HOVER STATE --- */
            table[id^="T_"] tbody tr:hover th,
            table[id^="T_"] tbody tr:hover td {{
                background-color: #D8E1EB; 
                transition: background-color 0.15s ease;
            }}

            /* --- ALIGNMENT EXCEPTIONS --- */
            table[id^="T_"] tbody td.data {{
                text-align: right; 
            }}
            table[id^="T_"] tbody th.row_heading {{
                text-align: left; 
            }}
        </style>
        </head>
        <body>
            <div class="container">
                {el}
            </div>
        </body>
        </html>
        '''    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬