"""
Cross-platform filesystem abstraction and path validation utility.

This module provides a hardened wrapper around Python's Path library to ensure 
consistent folder management and file path normalization across different 
operating systems. It centralizes all directory creation logic and implements 
strict naming conventions, preventing runtime crashes caused by invalid characters, 
trailing spaces, or OS-reserved keywords (e.g., Windows CON/NUL devices).
"""

from ..deps.common  import platform, Path, re, inspect, wraps
from ..deps.types   import Callable
from ..core.logging import Logger


logger = Logger(__name__)


class Folder:
    """Utility class for standardized folder-file operations via Pathlib."""
    
    @staticmethod
    def standard(path: str|Path|None) -> Path:
        """
        Normalizes any path input into a pathlib.Path object to ensure 
        API consistency.
        """
        if path is None:
            return Path()
        if isinstance(path, Path):
            return path
        return Path(path)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, path: str|Path) -> Path:
        """
        Recursively creates the directory structure for a given path if it 
        does not already exist.
        """
        path = Folder.standard(path)
        
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def path_file(path: str|Path) -> tuple[Path|None, str]:
        """Splits a full path into its parent directory and final filename components."""
        path = Path(path)
        if path.parent == Path():
            file = path
            path = None
        else:
            file = path.name
            path = path.parent                
        
        return path, file
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def files_exist(folder_path: str|Path, files_to_check: list, verbose: bool = False) -> dict:
        """
        Verifies the presence of multiple files within a directory; returns a mapping 
        of file paths to boolean existence.
        """
        base_folder = Path(folder_path).resolve()
        
        if not base_folder.is_dir():
            raise ValueError(f"The provided path {base_folder} is not a valid directory.")

        presence = {}
        for file in files_to_check:
            file_path = Path(file)

            if not file_path.is_absolute():
                file_path = base_folder / file_path
            
            fe = file_path.exists()
            presence[str(file_path)] = fe
            
            if not fe: 
                m = f'{file_path.resolve()} → NOT FOUND!'
                logger.warning(m)
                if verbose:
                    print(m)

        return presence
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def is_full_path(cls, path: str|Path) -> bool:
        """
        Validates if the provided string is a complete file path 
        (contains both a directory and a filename).
        """
        path, file = cls.path_file(path)
        file = len( str(file).split('.') ) > 1
        if path and file: 
            return True
        return False
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def is_valid_path(cls, path: str|Path) -> bool:
        """Check if a normalized path exists on the current filesystem."""
        path = cls.standard(path)
        return path.exists()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def is_folder_name_valid(folder_name: str|Path, info:str = '', verbose:bool = False) -> bool:
        """
        Enforces OS-specific naming rules for folders. 
        Checks for forbidden characters (Windows/Unix), trailing dots, and reserved 
        system names (e.g., 'CON').
        """        
        if not folder_name:
            return False
        
        def pi(i):
            m = f'Invalid Folder Name ► Please remove {i} {info}'
            logger.warning(m)
            if verbose: 
                print(m)
            
        if platform.system() == 'Windows':
            
            if re.search(r'[<>:"/\\|?*]', folder_name):
                pi(r'invalid characters "\ / : * ? " < > |"')
                return False
            
            if folder_name.endswith(' '): 
                pi('ending trailing spaces')
                return False
            
            if folder_name.endswith('.'):
                pi('ending "." character')
                return False
            
            reserved_names = [
                'CON', 'PRN', 'AUX', 'NUL',
                'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
            ]
            if folder_name.upper() in reserved_names:
                pi(f'Windows reserved folder names ► {reserved_names}')
                return False
        
        # Unix/Linux/macOS
        else:          
            if '/' in folder_name:
                pi(r'invalid character "/"')
                return False
            
        return True
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def exists(cls, func: Callable):
        """
        Decorator that ensures the target directory of a function's file-argument 
        exists before the function is actually executed.
        """         
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
                        
            file_path = (
                bound_args.arguments.get("path")
                or bound_args.arguments.get("filepath") 
                or bound_args.arguments.get("dest")
                or bound_args.arguments.get("destination")
                or bound_args.arguments.get("source")
                or bound_args.arguments.get("output_file")
                or bound_args.arguments.get("where")
                or bound_args.arguments.get("base_dir")
            )
            
            if file_path is None:
                raise ValueError('Path to the file is a mandatory argument.')
            
            directory = cls.path_file(file_path)[0]
            if directory is not None:
                cls.create(directory)

            return func(*args, **kwargs)
        return wrapper    
    # ─────────────────────────────────────────────────────────────────────────────────────────────