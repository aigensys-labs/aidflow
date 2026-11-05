"""
Deterministic Project Structure and Artifact Management Engine.

This module defines the physical layout of an AIDflow project on the host 
file system. It transforms the high-level logical flow (defined in `AIGFLOW`) 
into a structured directory hierarchy, automating the creation and 
management of folders for models, metadata, and reports based on the 
project's unique signature.

Key architectural roles provided by this module:
1. Deterministic Hierarchy: Implements a rigid blueprint (`STRUCTURE`) that 
   ensures every project follows an identical organizational standard , enabling 
   seamless collaboration and standardized auditing across different ML experiments.

2. Automated Artifact Mapping: Dynamically maps every model, transformer, 
   and report to its corresponding OS path via `AigPaths`. This removes the 
   need for hardcoded paths in the codebase and allows the system to resolve 
   file locations based on the project's logical state.

3. State Persistence Infrastructure: Orchestrates the physical creation of 
   the output structure, ensuring that all necessary directories exist before 
   models are saved or reports are generated, thereby preventing runtime 
   I/O errors during long-running training cycles.

4. Project Path Resolution: Provides a centralized `ProjectPaths` object that 
   acts as the definitive map for the current project's filesystem, facilitating 
   rapid retrieval of fitted models and metadata without redundant OS lookups.
"""

from ..deps.struct      import dataclass
from ..deps.common      import Path
from ..storage.folder   import Folder
from .factory           import DictClass
from .foundation        import AigModel
from .interop           import Dict


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# STRUCTURE IDENTITY CONSTANTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class SID(DictClass):
    """Structure Access to the Main Components."""   
    D = 'DATA'
    O = 'OUTPUT' 
    P = 'PROJECT'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# FLOW DESIGN
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class AIGFLOW(DictClass):
    """Flow-steps design and access."""   
    
    class Data(DictClass):    
        loader:         str = 'data_loader'
        fe_cleaner:     str = 'fe_cleaner'
        fe_stabilizer:  str = 'fe_stabilizer'
        obs_stabilizer: str = 'obs_stabilizer'

    class Encoding(DictClass):    
        num_encoder:    str = 'num_encoder'
        cat_encoder:    str = 'cat_encoder'
        ixn_encoder:    str = 'ixn_encoder'
        woe_encoder:    str = 'woe_encoder'
        woe_stabilizer: str = 'woe_stabilizer'
        data_encoder:   str = 'data_encoder'

    class Training(DictClass):
        agent1:         str = 'agent1_predictor'
        agent2:         str = 'agent2_predictor'
        agent3:         str = 'agent3_predictor'
        agent4:         str = 'agent4_predictor'
    
    class Selection(DictClass):
        predictor:      str = 'agents_predictor'
        comparator:     str = 'agents_comparator'
        transition:     str = 'agents_transition'
        selector:       str = 'agent_selector'
        champion_agent: str = 'champion_predictor'

    class Schema(DictClass):
        features:       str = 'fe_schema'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

_schema = AIGFLOW.schema()
_model  = 'Model'
_meta   = 'Meta'
_report = 'Report'

REG = [_schema.Data, _schema.Encoding, _schema.Training, _schema.Selection]

PATHS = {
    _model  : REG + [_schema.Schema],
    _meta   : REG,
    _report : REG
}  

AIGMODELS = AigModel.structure(AIGFLOW, REG)
AIG_links = Dict.get_flattened_values(AIGFLOW)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PROJECT FOLDER STRUCTURE BLUEPRINT
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
ROOT  = 'root'
SETUP = 'setup'
LOG   = 'run.log'

# ROOT DIR
# ─────────────────────────────────────────────────────────────────────────────────────────────────
STRUCTURE = {
    SID.D: 'data', 
    SID.O: 'outputs',
    SID.P: {...}
}       

# PROJECT OUTPUT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────────────────────────
_model_structure = lambda root: {
    ROOT: 'models',
    'fx': {
        ROOT: root,
            _schema.Data:       'data',
            _schema.Encoding:   'encoding',
            _schema.Training:   'training',
            _schema.Selection:  'selection',
    }
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────

_report_structure = lambda : {
    ROOT: 'reports',

    _schema.Data: {
        ROOT: '1_data',
            AIGFLOW.Data.loader:         '1_load',
            AIGFLOW.Data.fe_cleaner:     '2_cleaning',       
            AIGFLOW.Data.fe_stabilizer:  '3_temporal_stability',
            AIGFLOW.Data.obs_stabilizer: '4_observation_stability',
    },
    _schema.Encoding: {
        ROOT: '2_encoding',        
            AIGFLOW.Encoding.num_encoder:    '1_num_ecoding',
            AIGFLOW.Encoding.cat_encoder:    '2_cat_encoding',
            AIGFLOW.Encoding.ixn_encoder:    '3_ixn_encoding',
            AIGFLOW.Encoding.woe_encoder:    '4_woe_encoding',
            AIGFLOW.Encoding.woe_stabilizer: '5_woe_stability',
    },
    _schema.Training: {
        ROOT: '3_training',
            AIGFLOW.Training.agent1: '1_agent',
            AIGFLOW.Training.agent2: '2_agent',
            AIGFLOW.Training.agent3: '3_agent',
            AIGFLOW.Training.agent4: '4_agent'
    },
    _schema.Selection: {        
        ROOT: '4_selection',
            AIGFLOW.Selection.comparator:       '1_comparison',
            AIGFLOW.Selection.transition:       '2_transition',
            AIGFLOW.Selection.selector:         '3_selection',
            AIGFLOW.Selection.champion_agent:   '4_champion_agent',
    }
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────

STRUCTURE[SID.P] = {
    
    ROOT: STRUCTURE[SID.O],

        _model:  _model_structure('components') | {_schema.Schema: 'schema'},
        _meta:   _model_structure('metadata'),
        _report: _report_structure(),
}   
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PROJECT's ACCESS TO MODELS AND OUTPUTS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@dataclass
class ProjectPaths:
    """
    AIG flow project object paths:
    - structure  (dict) → Complete folder structure of the AIG.
    - models     (dict) → All Model  Paths which AIG creates during the whole pipeline.
    - metas      (dict) → All Meta   Paths which AIG creates during the whole pipeline.
    - reports    (dict) → All Report Paths which AIG creates during the whole pipeline.
    - root       (Path) → Path of the Project root folder (2_outputs/{project}).
    - setup      (Path) → Path to the User Main Flow Params Setup.
    - log        (Path) → Path to the Project logging output during the pipeline run.
    """
    structure:  dict[str, Path]
    models:     dict[str, Path]
    reports:    dict[str, Path]
    metas:      dict[str, Path]
    root:       Path
    setup:      Path
    log:        Path
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class AigPaths:

    @classmethod
    def _get_nested_path(cls, structure:dict|str, name:str) -> dict|Path:
        """Search for the unique name in the structure nodes."""
        if not isinstance(structure, dict): # skip ROOT
            return
        if name in structure: # only 1 unique match possible
            return structure[name]
        for s in structure.values():
            path = cls._get_nested_path(s, name)
            if path is not None:  
                return path
       
    @classmethod
    def _get_structure(cls, structure:dict, stage:str) -> dict|Path:
        if not isinstance(structure, dict):
            raise TypeError('AIG SAFETY ► Structure nodes must be defined by a dict type.')
        return cls._get_nested_path(structure, stage)

    @classmethod
    def _get_path(cls, structure:dict|Path, name:str) -> Path:
        if isinstance(structure, Path):
            return structure                     
        return cls._get_nested_path(structure, name)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
                    
    @classmethod
    def get_project_paths(cls, structure:dict) -> ProjectPaths:
        """Obtain OS paths given AIG `STRUCTURE` design."""
        
        project     = structure[SID.P]        
        setup_path  = project[ROOT] / SETUP
        log_path    = project[ROOT] / LOG
        
        def get_design(phase:str) -> tuple[dict[str, dict], list[str]]:
            """Returns structure and stages for the given phase."""
            return project[phase], PATHS[phase]

        def get_paths(phase:str) -> dict:
            structure, stages = get_design(phase)
            paths = {}
            for stage in stages:
                s = cls._get_structure(structure, stage)
                for name in AIGFLOW[stage].values():
                    paths[name] = cls._get_path(s, name)
            return paths
        
        paths = {}
        for p in [_model, _meta, _report]:
            paths[p] = get_paths(p)

        return ProjectPaths(
            structure = structure,
            models    = paths[_model],
            metas     = paths[_meta],
            reports   = paths[_report],
            root      = project[ROOT],
            setup     = setup_path,
            log       = log_path
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def gen_structure(
        project: str, 
        structure: dict = None, 
        create: bool = False, 
        path: str = None,
        skip_create: list = None
    ) -> dict:
        """OS path structure gen given AIG `STRUCTURE` design."""
        
        if structure is None:
            structure = Dict.deep_copy(STRUCTURE)

        skip_create = skip_create or []

        def _create_path(structure:dict, k:str, path:str|Path):
            p = path
            if k != ROOT: 
                p = p / structure[k]
            structure[k] = Folder.standard(p)
            if create and k not in skip_create: 
                Folder.create(p)
            
        def _create_level(structure:dict, level:str, path = Path('')):
            sub_struct = structure[level]

            if not isinstance(sub_struct, dict):
                _create_path(structure, level, path)
                return
            
            p = path / sub_struct[ROOT]
            if level == SID.P: 
                p = p / project

            for k in sub_struct:
                _create_level(sub_struct, k, p)

        for k in structure:
            _create_level(structure, k, Folder.standard(path or ''))
        
        return structure
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PROJECT FOLDER STRUCTURE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class ProjectStructure:                
    """AIG flow structure to store/retrieve any trained object for a given `project`."""
    
    def __init__(self, 
        project: str, 
        create:  bool = False, 
        predict: bool = False, 
        path:    str  = None
    ) -> None:

        structure = Dict.deep_copy(STRUCTURE)
        
        if predict:
            structure = AigPaths.gen_structure(project, path = path)

        else:
            structure = AigPaths.gen_structure(project, structure, create, path, [SID.D])        
        
        self._project_paths = AigPaths.get_project_paths(structure) 

        self.STRUCTURE = structure

    @property    
    def project_paths(self):
        """
        AIG flow project object paths:
        - structure  (dict) → Complete folder structure of the AIG.
        - models     (dict) → All Model  Paths which AIG creates during the whole pipeline.
        - metas      (dict) → All Meta   Paths which AIG creates during the whole pipeline.
        - reports    (dict) → All Report Paths which AIG creates during the whole pipeline.
        - root       (Path) → Path of the Project root folder (2_outputs/{project}).
        - setup      (Path) → Path to the User Main Flow Params Setup.
        - log        (Path) → Path to the Project logging output during the pipeline run.   
        """        
        return self._project_paths 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬