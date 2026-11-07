"""
Artifact Serialization and Multi-Format Export Engine.

This module provides a high-level interface for persisting pipeline results 
as permanent assets across various formats, including HTML, Excel. 
It streamlines the transition from interactive data exploration to formal 
reporting by automating the export of complex objects—such as Plotly figures 
and pandas DataFrames.
"""

from ..deps.common          import  Path
from ..deps.data            import  pd, pDF
from ..deps.plots           import  go
from ..deps.display         import  HTML
from ..deps.data_view       import  itables_html
from ..core.logging         import  Logger
from .file                  import  FileUtils


logger = Logger(__name__)


class Export: 
    """
    Centralized Utility for exporting pipeline artifacts as permanent assets.
    
    This class provides a high-level interface to transform in-memory data, 
    Plotly figures, and Matplotlib plots into shareable files (HTML, XLSX).     
    """

    @classmethod
    def plt_as_html(cls, fig:go.Figure, path:Path|str|None):
        """
        Export a Plotly figure as a standalone interactive HTML file.

        Utilizes the CDN-based JS injection to ensure the resulting file is 
        lightweight and portable while maintaining full interactivity (zoom/hover).

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure object to be serialized.
        path : Path | str, optional
            Destination path. If None, execution is skipped.

        Returns
        -------
        Path | str: The verified file path of the exported HTML asset.
        """        
        if path is None:
            return
        path = FileUtils.ensure_extension(path, 'html')                    
        try:
            fig.write_html(path, include_plotlyjs='cdn')
        except:
            cls._log(path, 'plotly figure as html')
        return path
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def itable_as_hmtl(cls, df:pDF, path:Path|str|None, sample:int=100, pageLength:int=100):
        """
        Export a DataFrame as an interactive HTML table via the `itables` library.

        This method handles large datasets by performing a random subsample 
        before rendering to avoid browser crashes (via `sample` param) and 
        configures pagination for better readability (via `pageLength`).

        Parameters
        ----------
        df : pDF
            The DataFrame to be rendered.
        path : Path | str, optional
            Destination path. If None, execution is skipped.
        sample : int, default=100
            Maximum number of rows to include in the interactive HTML table.
        pageLength : int, default=100
            The default number of rows displayed per page in the HTML UI.

        Returns
        -------
        None: Writes the HTML content directly to the destination file.
        """        
        if path is None:
            return
        sample  = df.sample(min(len(df), sample), random_state=42)
        html_df = itables_html(sample, pageLength=pageLength)
        path    = FileUtils.ensure_extension(path, 'html')
        try:
            with open(path, "w", encoding="utf-8") as file:
                file.write(html_df)
        except:                 
            cls._log(path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod    
    def html(cls, html:HTML|str, path:Path|str|None):
        """
        Save raw HTML strings or objects to a permanent file.

        Provides a safe wrapper for writing HTML content, ensuring that 
        custom `HTML` objects are correctly cast to strings and destination 
        paths are verified via `FileUtils`.

        Parameters
        ----------
        html : HTML | str
            The HTML content or object to be exported.
        path : Path | str, optional
            Destination path. If None, execution is skipped.

        Returns
        -------
        Path | str: The verified file path of the exported HTML asset.
        """        
        if path is None:
            return
        path = FileUtils.ensure_extension(path, 'html')
        try:
            if isinstance(html, HTML):
                html = html.data            
            with open(path, "w", encoding="utf-8") as file:
                file.write(html)
        except Exception as e: 
            cls._log(path)
        return path
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    def excel(cls, df:pDF, path:Path|str|None, name='Sheet1'):
        """
        Export a DataFrame to an Excel worksheet (.xlsx).

        Implements 'Surgical Update' logic: if the file already exists, it 
        opens the workbook in append mode and replaces only the specified 
        sheet ('replace' mode), preserving other existing sheets. It also 
        enforces a 31-character limit on sheet names to adhere to Excel constraints.

        Parameters
        ----------
        df : pDF
            The DataFrame to be exported.
        path : Path | str, optional
            Destination path. If None, execution is skipped.
        name : str, default='Sheet1'
            The name of the worksheet within the Excel file.

        Returns
        -------
        bool: True if the export was successful, False otherwise.
        """        
        if path is None:
            return
        verified = False
        try:
            path = FileUtils.ensure_extension(path, 'xlsx')                    
            name = FileUtils.obs_to_K(name)
            if len(name) > 31 : name = name[:31]
            
            mode, sheet = 'w', None
            if Path(path).is_file(): mode, sheet = 'a', 'replace'                    
            with pd.ExcelWriter(path, mode=mode, if_sheet_exists=sheet) as writer:  
                df.to_excel(writer, sheet_name=name)
            verified = True        
        except:
            cls._log(path, 'excel')
        return verified
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _log(path: str | Path, kind:str='html') -> None:
        msg = (                
            f'Could not save {kind} output to file - save path: {path}'
        )
        logger.error(msg)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬