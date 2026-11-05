"""
Notebook environment discovery and UI state persistence utility.

This module implements a specialized session management system that enables 
AIDflow to remember user interface configurations across notebook restarts. 
By dynamically resolving the active notebook's filename through a multi-layered 
discovery process—supporting VS Code, JupyterLab, and standard Jupyter notebooks—
it creates a unique link between the interactive session and its persisted 
configuration state in the project output directory.
"""

from ...deps.common     import  (
                                os, 
                                request,
                                Path, 
                                json,
                                hashlib
                                )
from ...deps.display    import  (
                                get_connection_file, 
                                get_ipython, 
                                jupyter_runtime_dir
                                )
from ...storage.folder  import  Folder
from ...storage.file    import  FileIO, FileUtils

from ..structure        import  SID, STRUCTURE


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# Helper to store UI config for session restoration
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ConfigState:
    """
    Helper to create and erase UI config in PROJECT output folder.
    UI config are created only when HTML UI is configured.
    
    Naming convention and the only link between config and notebook
    is the notebook name - if user renames ntb, defaults will be shown.
    """

    UI = Folder.standard(STRUCTURE[SID.O]) / '.ui_config'
    FI = ['Attribute', 'Value']

    def __init__(self):
        name = NotebookName.get_name()
        self.name = FileUtils.ensure_extension(name, 'csv')

    def file(self):        
        Folder.create(self.UI)
        return Folder.standard(self.UI / self.name)      

    def save(self, config:dict) -> None:
        """Save config file under notebook name."""
        FileIO.save_dict_to_csv(config, self.file(), self.FI)

    def load(self) -> dict | None:
        """Loads stored config file, if exists, othrwise returns None."""
        try:    return FileIO.load_csv_as_dict(self.file(), self.FI)
        except: return None
    
    def clear(self):
        """Removes stored config file under notebook name if exists."""
        Path(self.file()).unlink()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class NotebookName:
    
    @staticmethod
    def _vs_code_env() -> Path | None:
        """VS Code Environment Check."""
        ip = get_ipython()
        if ip and '__vsc_ipynb_file__' in ip.user_ns:
            return Path(ip.user_ns['__vsc_ipynb_file__'])
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _jn_scan() -> Path | None:
        """Obtain name by Jupyter Notebook / JupyterLab Screening."""
        
        # Extract active Kernel ID
        conn_file = Path(get_connection_file()).stem
        kernel_id = conn_file.split('-', 1)[1]

        # Scan Jupyter runtime directory for active servers
        runtime_dir  = Path(jupyter_runtime_dir())
        server_files = list(runtime_dir.glob('nbserver-*.json')) + list(runtime_dir.glob('jpserver-*.json'))
        server_files.sort(key=os.path.getmtime, reverse=True)
        
        for file_path in server_files:
            try:
                with open(file_path, 'r') as f:
                    srv = json.load(f)
                
                # Ping server API for active sessions
                token = srv.get('token', os.environ.get("JUPYTERHUB_API_TOKEN", ""))
                url = f"{srv['url']}api/sessions{f'?token={token}' if token else ''}"
                
                with request.urlopen(url, timeout=1.0) as req:
                    sessions = json.load(req)

                # Match current Kernel ID to the session file path
                for s in sessions:
                    if s.get('kernel', {}).get('id') == kernel_id:
                        return Path(s['path'])
                    
            except Exception:
                continue # Skip stale server files gracefully

    @staticmethod
    def _jn_session() -> Path | None:
        """Obtain JPY_SESSION_NAME if present, otherwise None."""
        session = os.environ.get("JPY_SESSION_NAME")
        if session is None:
            return
        return Path(session)
    
    @staticmethod
    def _files() -> Path | None:
        """Fallback to extract notebook name in running root dir."""
        files = [p for p in Path('.').iterdir() if p.suffix == '.ipynb']
        if len(files) == 1:                            
            return files[0]
        print(
            'AIG SAFETY ► Cannot safely determine current session to load config'
            ' ► Defaults provided under "name_unknown".'
        )
        return Path('name_unknown')

    @classmethod
    def _jupyter_env(cls) -> Path | None:
        """Standard Jupyter Notebook / JupyterLab Check."""

        # Dynamic nb name
        name = cls._jn_scan()
        if name:
            return name
        
        # Static session name → doesn`t reflect renaming ntb during the session
        session = cls._jn_session()
        if session:
            return session
        
        # Fallback to find ipynb files
        return cls._files()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _hash_path(path: Path, length: int = 20) -> str:
        """Safely converts a file path into a short, persistent hash string."""
        normalized_path = path.resolve().as_posix()

        if os.name == "nt":
            normalized_path = normalized_path.lower()
        
        full_hash = hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()

        return full_hash[:length]    
    
    @classmethod
    def get_id(cls, path: Path) -> str:
        """Generates unique project ID."""
        return path.stem
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def get_path(cls) -> str | None:
        """Extracts the current notebook name."""        
        path = cls._vs_code_env()
        if path:
            return path        
        return cls._jupyter_env()

    @classmethod
    def get_name(cls) -> str:
        """Extracts the current notebook name."""        
        path = cls.get_path()
        return cls.get_id(path)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬