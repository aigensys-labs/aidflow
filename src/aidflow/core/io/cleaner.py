"""
Project Folder Cleaner before a new run.
"""

from ..parameters   import  Parameters
from .io            import  FittedIO
from .temp          import  Temp


class ProjectIOCleaner:
    """Project Folder Cleaner before a new run."""

    def __init__(self, setup: Parameters):        
        self.setup = setup
        
    def clear(self):
        """Clears Project Temp Folders."""
        FittedIO(self.setup).clear()
        Temp(self.setup).clear()
        self._clear_report_html()

    def _clear_report_html(self):
        """
        Clears all HTML static files in report folders 
        when existing PROJECT with `save=1` is re-initialized.

        User intention is to mutate current outputs → safety measure.
        """
        if not self.setup.save:
            return
        for folder in self.setup.project_paths.reports.values():
            if not folder:
                continue
            for p in folder.glob("*.html"):
                if p.is_file():
                    p.unlink()