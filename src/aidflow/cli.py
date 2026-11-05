"""
AIDflow CLI — Command-Line Interface.

Provides the `aidflow` executable with four subcommands:

    init      Scaffold a new project (assets + optional venv).
    run       Execute the pipeline with synthetic data.
    notebook  Launch the bundled Jupyter notebook.
    assets    List all bundled files.

All commands target a directory (default: current). Heavy imports
are deferred to command handlers to keep CLI startup fast.
"""

from .deps.common import (
    os,
    sys,
    Path,
)
from . import (
    UIGetter, 
    UIPaths
)

_PATHS = UIGetter.get_paths()
_GETTER = {
    UIPaths.NTB: UIGetter.get_notebook,
    UIPaths.CLI: UIGetter.get_script,
}    


class _Utils:

    @staticmethod
    def banner():
        """Print the AIDflow version banner."""
        print(r"""
        ┌─────────────────────────────────────────────┐
        │ AIDflow v3.0                                │
        │ Scorecard Development                       │
        │ Automated Interactive Development Framework │
        └─────────────────────────────────────────────┘
        """)

    @staticmethod
    def create_dir(args) -> Path:
        """Create opt. directory."""
        target = Path(args.dir) if args.dir else Path('.')
        target.mkdir(parents=True, exist_ok=True)
        print(f'  → scaffolding to {target.resolve()}')
        return target

    @staticmethod
    def copy_assets(target:Path, subset: tuple=(UIPaths.NTB, UIPaths.CLI)
    ) -> dict[str, Path] | Path:
        paths = {}
        for k in subset:
            paths[k] = _GETTER[k](target)

        for key, p in paths.items():
            print(f'    ✓ {key:<10} {p}')

        return paths[k] if len(subset) == 1 else paths

    @staticmethod
    def create_venv(args, target:Path) -> None:
        """Create optional venv."""
        if not args.venv:
            return 
        from .deps.cli import subprocess

        venv_dir = target / '.venv'
        print(f'\n  → creating venv at {venv_dir}')
        subprocess.call([sys.executable, '-m', 'venv', str(venv_dir)])

        pip = venv_dir / ('Scripts' if os.name == 'nt' else 'bin') / 'pip'
        print(f'  → installing AIDflow into venv')
        subprocess.call([str(pip), 'install', '-q', 'aidflow'])    


# ═══════════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════════
def cmd_init(args):
    """
    Scaffold a new AIDflow project.

    Copies bundled assets to the target directory. Optionally
    creates a virtual environment. Prints next-step instructions.
    """

    target = _Utils.create_dir(args)
    paths  = _Utils.copy_assets(target)

    _Utils.create_venv(args, target)

    # ── next steps ────────────────────────────────────────────────
    print(f'\n  Next steps:')
    if args.venv:
        act = r'.venv\Scripts\activate' if os.name == 'nt' else 'source .venv/bin/activate'
        print(f'    {act}')
    print(f'    Interactive : jupyter notebook {paths[UIPaths.NTB].name}')
    print(f'    Console     : python {paths[UIPaths.CLI].name}')
    print(f'    Or          : aidflow run --size {args.size}')
    print()


# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════
def cmd_run(args):
    """
    Execute the AIDflow pipeline with synthetic data.

    Scaffolds the console script into the target directory,
    then executes it — all outputs (data, models, reports)
    are written to that directory.
    """
    from .deps.cli import importlib

    target = _Utils.create_dir(args)
    path   = _Utils.copy_assets(target, (UIPaths.CLI,))

    spec = importlib.util.spec_from_file_location('aidflow_script', path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    os.chdir(target)

    print('► Executing demo script:')
    print(f'  → size  : {args.size}')
    print(f'  → phase : {args.phase}')
    print()

    mod.main(phase=args.phase, size=args.size)


# ═══════════════════════════════════════════════════════════════════
# NOTEBOOK
# ═══════════════════════════════════════════════════════════════════
def cmd_notebook(args):
    """
    Scaffold the notebook and launch Jupyter.

    Copies the bundled notebook to the target directory,
    then opens it in the Jupyter browser.
    """
    from .deps.cli import subprocess

    target = _Utils.create_dir(args)
    path   = _Utils.copy_assets(target, (UIPaths.NTB,))
    
    try:
        import notebook
    except ImportError:
        print('\n  ✗ notebook not installed.')
        print('    Install with:  pip install aidflow[notebook]')
        sys.exit(1)

    print(f'  → opening {path.name}')
    subprocess.call(['jupyter', 'notebook', str(path)])


# ═══════════════════════════════════════════════════════════════════
# ASSETS
# ═══════════════════════════════════════════════════════════════════
def cmd_assets(args):
    """
    List all bundled assets.

    Prints a table of asset name → source path for every
    static file included in the package.
    """
    print('  Bundled assets:')
    for key, p in _PATHS.items():
        print(f'    {key:<10} {p}')
    print()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
_EPILOG = """\
examples:
  aidflow init my_project/          Scaffold project + assets
  aidflow init my_project/ --venv   Scaffold + create virtualenv
  aidflow notebook ui_demo/         Scaffold notebook & open Jupyter
  aidflow run demo/                 Run pipeline (outputs → demo/)
  aidflow run demo/ --size large    Run with large synthetic data
  aidflow run demo/ --phase data    Run only the data phase
  aidflow assets                    List bundled files
"""


def main():
    """CLI entry point — parse arguments and dispatch.

    If no subcommand is provided, prints help and exits.
    """
    from .deps.cli import argparse
    _Utils.banner()

    parser = argparse.ArgumentParser(
        prog='aidflow',
        description='AIDflow — Automated Interactive Development Framework',
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='cmd')

    # ── init ───────────────────────────────────────────────────────
    p_init = sub.add_parser(
        'init',
        help='Scaffold project assets to a directory',
        description='Copy bundled assets (console script + notebook) to a target directory.',
    )
    p_init.add_argument(
        'dir',
        nargs='?',
        default='',
        help='Target directory (default: .)'
    )
    p_init.add_argument(
        '--venv',
        action='store_true',
        help='Create a virtualenv and install AIDflow'
    )
    p_init.add_argument(
        '--size',
        choices=['small', 'medium', 'large'],
        default='small',
        help='Default data size for quick-start hint (default: small)'
    )
    p_init.set_defaults(func=cmd_init)

    # ── run ────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        'run',
        help='Run the console pipeline',
        description='Execute the full or partial AIDflow pipeline with generated data.',
    )
    p_run.add_argument(
        'dir',
        nargs='?',
        default='',
        help='Target directory (default: .)'
    )      
    p_run.add_argument(
        '--size',
        choices=['small', 'medium', 'large'],
        default='small',
        help='Synthetic data size (default: small)'
    )
    p_run.add_argument(
        '--phase',
        choices=['all', 'data', 'encoding', 'training', 'selection'],
        default='all',
        help='Pipeline phase to run (default: all)'
    )
    p_run.set_defaults(func=cmd_run)

    # ── notebook ───────────────────────────────────────────────────
    p_nb = sub.add_parser(
        'notebook',
        help='Open the interactive notebook',
        description='Launch the bundled Jupyter notebook.',
    )
    p_nb.add_argument(
        'dir',
        nargs='?',
        default='',
        help='Target directory (default: .)'
    )        
    p_nb.set_defaults(func=cmd_notebook)

    # ── assets ─────────────────────────────────────────────────────
    p_assets = sub.add_parser(
        'assets',
        help='List bundled assets',
        description='Show all static files shipped with AIDflow.',
    )
    p_assets.set_defaults(func=cmd_assets)

    # ── dispatch ───────────────────────────────────────────────────
    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == '__main__':
    main()