"""
Centralized Parameter Schema and Global Configuration Engine.

This module implements the Single Source of Truth (SSOT) for all pipeline 
configurations within AIDflow. Instead of treating parameters as simple scalar 
values, this engine defines them as rich metadata objects that encapsulate 
definition, validation constraints, and UI rendering instructions.

Key architectural features:
1. Schema-Driven Design: Uses `ParamBase` to bind parameter names, data types, 
   and descriptive metadata (tips, labels) into a single object, ensuring 
   consistency across the CLI, GUI, and internal logic.

2. Automated Validation: Provides a standardized interface for the validation 
   engine (`utils/param.py`) to enforce constraints such as numeric ranges (min/max), 
   allowed categorical options, and type safety.

3. UI-Aware Definitions: Enables "Zero-Code" GUI generation via `get_panel_defaults`, 
   allowing the user interface to automatically render sliders, dropdowns, 
   and text fields based on the parameter's `kind` and constraints.

4. Domain-Specific Grouping: Organizes parameters into phase-specific 
   containers (e.g., `GEN` for Genesis Phase), providing a structured hierarchy 
   that is accessible throughout the flow via a global context manager.

By decoupling the value from its definition, AIDflow ensures that pipeline 
configurations are immutable in their schema but dynamic in their application.
"""

from ..deps.struct          import  dataclass

from ..storage.folder       import  Folder
from ..storage.file         import  BaseIO, FileIO   

from ..display.verbosity    import  Verbosity

from .structure             import  (
                                    SID, 
                                    STRUCTURE,
                                    ProjectStructure,
                                    AIGFLOW,
                                    AIG_links
                                    )
from .factory               import  (
                                    DictClass, 
                                    Dataclass
                                    )
from .foundation            import  AigParam
from .interop               import  Dict
from .validation.params     import  (
                                    Value, 
                                    ParamsValidator
                                    )
from .logging               import  (
                                    Logger,
                                    LogSignature
                                    )


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# AIG FLOW PARAMETRIZATION
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@Dataclass.base
class ParamBase:
    """
    The schema and factory for AIG pipeline parameter definitions.
    
    This class standardizes how parameters are defined, stored, and accessed throughout
    the system. A parameter is not just a value; it is a rich object containing:
    - **Metadata**: Name, description, and help tips for UI/CLI usage.
    - **Constraints**: Data types, min/max ranges, and allowed options.
    - **State**: The current value of the parameter.
    
    The class provides static factory methods to create valid parameter base structure
    and accessor methods to safely extract specific fields (e.g., `get_v_min`, `get_kind`).
    These accessors are the bridge between this schema and the validation engine in `utils/param.py`.
    """    
    _prm:       str
    _kind:      Value
    _val:       dict
    _name:      str
    _desc:      str
    _tip:       str
    _desc_h:    str = None

    @property
    def param(self) -> str:
        """Retrieves parameter name."""
        return self._prm

    @property
    def kind(self) -> Value:
        """Retrieves parameter type: float | int | str | list | dict | None."""
        return self._kind

    @property
    def value(self) -> Value:
        """Retrieves the default scalar value."""
        return self._val['value']

    @property
    def name(self) -> str:
        """Retrieves the default parameter description name."""
        return self._name

    @property
    def desc(self) -> str:
        """Retrieves the default parameter description."""
        return self._desc
    
    @property
    def desc_h(self) -> str | None:
        """Retrieves the description for horizontal outline if present."""
        return self._desc_h

    @property
    def tip(self) -> str:
        """Retrieves the quick tip/hint string."""
        return self._tip
    
    def get_panel_defaults(self, name:str=None) -> dict:
        """
        Retrieves the defaults for step-panel `[name, [min, max], kind]`.\n
        Whereas `kind` is the default kind[0] if param can be multi-kinds.\n
        When `name` is provided returned list is with custom name.
        """
        name = self.name if not name else name
        kind = self.kind if isinstance(self.kind, type) else self.kind[0]
        return { self.param: [name, self.get_v_min_max(), kind] }

    def get_values(self, val:Value=None) -> dict: 
        """
        Retrieves the full `value` container dictionary.
        
        If a specific `val` is provided, it merges this new value into the 
        existing constraints (min, max, options) without losing the schema.
        
        Parameters
        ----------
        val : Value, optional
            A new value to inject. If None, returns the default definition value.
            
        Returns
        -------
        dict
            The 'v' dictionary, optionally updated with a new 'value' key.
        """        
        if val is None: 
            return self._val
        return self._val | dict(value=val)
    
    def get_v_min_max(self) -> list[int|float|None]:
        """Retrieves the [min, max] constraint for numeric parameters (or None)."""
        return [self.get_v_min(), self.get_v_max()]

    def get_v_min(self) -> int|float|None:
        """Retrieves the minimum constraint for numeric parameters (or None)."""
        return self._val.get('min')
    
    def get_v_max(self) -> int|float|None: 
        """Retrieves the maximum constraint for numeric parameters (or None)."""
        return self._val.get('max')
    
    def get_v_step(self) -> int|float|None: 
        """Retrieves the maximum constraint for numeric parameters (or None)."""
        return self._val.get('step')

    def get_v_options(self) -> list|None:
        """Retrieves the list of allowed options for text/categorical parameters."""
        return self._val.get('options')
    
    def get_v_holder(self)  -> str|None:
        """Retrieves the placeholder text for text parameters."""
        return self._val.get('placeholder')
    
class ParamVal:
    """The schema and factory for AIG pipeline values definitions."""
    
    @staticmethod
    def num(*, val:int|float, mini:int|float, maxi:int|float, step:int|float) -> dict:
        """
        Factory to create the nested value dictionary for numeric parameters.
        
        Used for sliders, inputs, and ranges.
        
        Parameters
        ----------
        val : int | float
            The default or current value.
        mini : int | float
            The minimum allowed value.
        maxi : int | float
            The maximum allowed value.
        step : int | float
            The increment step (useful for UI sliders).
            
        Returns
        -------
        dict
            {'value': val, 'min': mini, 'max': maxi, 'step': step}
        """        
        return dict(
            value = val,
            min   = mini,
            max   = maxi,
            step  = step
        )

    @staticmethod
    def text(*, val:str='', placeholder:str='', options:list=None) -> dict:
        """
        Factory to create the nested value dictionary for text/selection parameters.
        
        Used for text inputs, dropdowns, or multi-select lists.
        
        Parameters
        ----------
        val : str
            The default or current string value.
        placeholder : str
            The placeholder text shown in empty input fields.
        options : list, optional
            A list of valid options for dropdown validation or selection.
            
        Returns
        -------
        dict
            {'value': val, 'placeholder': placeholder, 'options': options}
        """
        return dict(
            value       = val,
            placeholder = placeholder,
            options     = options,
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class GEN(DictClass):
    """Parameters Definitions of Project Genesis Phase."""

    project  = ParamBase(
        _prm  = 'project',
        _kind = str,
        _val  = ParamVal.text(placeholder='Enter project name...'),
        _name = 'Project Name',
        _desc = 'Project as root directory of all outputs',
        _tip  = ('Unique identifier for the project. Used as a root directory of model artifacts, and logging files to ensure traceability.<br>'
              f'Stored in ./{STRUCTURE[SID.O]}'+'/{PROJECT_NAME}/...')
    )
    data = ParamBase(
        _prm     = 'input_data',
        _kind    = str,
        _val     = ParamVal.text(placeholder='table / path'),
        _name    = 'Data Table',
        _desc    = 'File / Path / Connection | File Formats: .csv, .gz, .parquet, .feather',
        _desc_h  = 'File / Path / Connection',
        _tip     = ('Defines path and format of the input data.<br>'  
              f'1. <b>file</b>: file name (if in ./{STRUCTURE[SID.D]}/) or full path<br>'
              '2. <b>spark</b>: con. e.g. catalog.db.table_name<br>'
              '3. <b>db</b>: connection e.g. postgresql://user:pass@localhost:5432/mydb'
        )
    )
    source = ParamBase(
        _prm     = 'input_source',
        _kind    = str,
        _val     = ParamVal.text(val='file', options=['file', 'spark', 'db']),
        _name    = 'Data Source',
        _desc    = 'Source of the data table: file / spark / db',
        _desc_h  = 'Source of the data',
        _tip     = 'Defines the ingestion engine. Options:<br>'
                '1. <b>file</b>: File system (csv / parquet / ipc / gzip)<br>'
                '2. <b>spark</b>: Distributed Spark DataFrame<br>'
                '3. <b>db</b>: SQL Database connection'
    )
    cols = ParamBase(
        _prm  = 'input_cols',
        _kind = (list, dict, str, None),
        _val  = ParamVal.text(placeholder='e.g. AGE, INCOME (Empty for All)'),
        _name = 'Input Columns',
        _desc = 'Subset of data columns',
        _tip  = ('List of columns to load. Leave empty to load <i>ALL</i> available columns from the source.'
                '<br>Supports also syntax by dictionary: {"remove": "COL1", "COL2"} <br> {"keep": "COL1", "COL2"}')
    )
    flow = ParamBase(
        _prm  = 'flow_vars',
        _kind = (list, str, None),
        _val  = ParamVal.text(placeholder='e.g. CLIENT_ID, SESSION_ID'),
        _name = 'Flow Columns',
        _desc = 'IDs / tracking variables',
        _tip  = ('Columns to retain as non-feature metadata/IDs. These are excluded from feature engineering but preserved '
                'for model application (scoring) and traceability.')
    )
    target = ParamBase(
        _prm  = 'target',
        _kind = str,
        _val  = ParamVal.text(placeholder='e.g. TARGET'),
        _name = 'Target Variable',
        _desc = 'Supervised learning target',
        _tip  = 'The name of the target (label) column in the dataset. Specifies the dependent variable the model will learn to predict.'
    )
    time = ParamBase(
        _prm  = 'yy_mm',
        _kind = (str, None),
        _val  = ParamVal.text(placeholder='e.g. DATE_DECISION'),
        _name = 'Time Variable',
        _desc = 'Time variable for temporal analyses',
        _tip  = ('The column name representing the timestamp/decision date. Used for time-based train/test splits and PSI stability checks.<br>'
                 'If no time variable is specified, an artificial one is generated based on row order.')
    )
    time_f = ParamBase(
        _prm  = 'yy_mm_format',
        _kind = (str, None),
        _val  = ParamVal.text(options=['%Y-%m-%d', '%Y%m%d', '%Y-%m-%d %H:%M:%S', '%d%b%y']),
        _name = 'Time Format',
        _desc = 'Time variable date format',
        _desc_h = 'Time format',
        _tip  = ('The strftime format string used to parse the time column. Standard Python % directives apply. '
              'Common Examples:<br>"2023-01-15" -> "%Y-%m-%d"<br>"15Jan23" -> "%d%b%y"')
    )

class AIG(DictClass):
    time_var = ParamBase(
        _prm  = 'time_var',
        _kind = str,
        _val  = ParamVal.text(val='AIG_YY_MM'),
        _name = 'Standardized Time Variable',
        _desc = 'Standardized Time variable during temporal analyses',
        _desc_h = 'Standardized Time variable',
        _tip  = ('AIG Flow standardized time variable used during temporal analyses. This variable ensures consistent input '
                 f'format derived from {GEN.time.param} - {GEN.time_f.param}.')
    )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PRM(DictClass):
    """AIG Flow Parameters Definitions."""

    class LOAD(DictClass):
        """Parameters Definitions of Data Load Phase."""

        start = ParamBase(
            _prm  = 'import_start',
            _kind = float,
            _val  = ParamVal.num(val=0.0, mini=0.0, maxi=1.0, step=0.01),
            _name = 'Starting Point',
            _desc = 'Data import relative Starting Point',
            _tip  = 'Fraction of the dataset to start loading from. Useful if the dataset head contains stale/noisy data or if processing in chunks.'
        ) 
        end = ParamBase(
            _prm  = 'import_end',
            _kind = float,
            _val  = ParamVal.num(val=1.0, mini=0.0, maxi=1.0, step=0.01),
            _name = 'Ending Point',
            _desc = 'Data import relative Ending Point',
            _tip  = 'Fraction of the dataset to stop loading at. Allows testing with subsets before committing to full data ingestion.'
        ) 
        test = ParamBase(
            _prm  = 'train_test',
            _kind = float,
            _val  = ParamVal.num(val=0.3, mini=0.05, maxi=0.9, step=0.01),
            _name = 'Train/Test Ratio',
            _desc = 'Data split to Train and Test part Timewise - Test as the latest OOT sample',
            _tip  = 'Proportion of the loaded data reserved for the Test Set (held out). The remaining (1-ratio) is used for training/validation.'
        )     
        valid = ParamBase(
            _prm  = 'train_valid',
            _kind = float,
            _val  = ParamVal.num(val=0.33, mini=0.1, maxi=0.5, step=0.01),
            _name = 'Validation Ratio',
            _desc = 'The Train part split to Trainining and Validation part - Training Phase',
            _tip  = 'Proportion of the Training data further split into a Validation Set. Used for hyperparameter tuning and early stopping.'
        )
        sample = ParamBase( 
            _prm  = 'smp_train',
            _kind = (float, int),
            _val  = ParamVal.num(val=1.0, mini=0.01, maxi=1.0, step=0.01),
            _name = 'Train Sampling',
            _desc = 'Sub-sampling ratio for the Training set',
            _tip  = ('Sub-sampling ratio for the Training set to accelerate the learning during prototyping.'
                    '<br>• Value over 1 represents sample observation count.<br>• Value between (0, 1) represents fraction of the input dataset.')
        )
        features = ParamBase( 
            _prm  = 'smp_feature',
            _kind = (float, int),
            _val  = ParamVal.num(val=1.0, mini=0.1, maxi=1.0, step=0.01),
            _name = 'Variables Sampling',
            _desc = 'Sub-sampling ratio for the Input Variables',
            _tip  = ('Sub-sampling ratio of input variables to reduce dimensionality and accelerate the learning during prototyping.'
                    '<br>• Value over 1 represents sample feature count.<br>• Value between (0, 1) represents fraction of the features in the input dataset.')
        )         
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    class SCREEN(DictClass):
        """Parameters Definitions of Data Screening Phase."""

        miss = ParamBase(
            _prm  = 'max_miss',
            _kind = float,
            _val  = ParamVal.num(val=0.7, mini=0.0, maxi=1.0, step=0.01),
            _name = 'Max Missing',
            _desc = 'Filter variables with excessive missing rate',
            _tip  = 'Drop variables with missing values exceeding set percentage.'
        )     
        mode = ParamBase(
            _prm  = 'max_1val',
            _kind = float,
            _val  = ParamVal.num(val=0.9, mini=0.0, maxi=1.0, step=0.01),
            _name = 'Max Mode Share',
            _desc = 'Filter single dominant values',
            _tip  = 'Drop variables where > X% of values are identical (mode). High share indicates lack of variance (e.g., constant columns).'
        )         
        var = ParamBase(
            _prm  = 'max_var',
            _kind = float,
            _val  = ParamVal.num(val=0.9, mini=0.0, maxi=1.0, step=0.01),
            _name = 'Max Variability',
            _desc = 'Remove ID-like variables',
            _tip  = 'Threshold to detect quasi-IDs (variables with extremely high cardinality/variability). Variables exceeding this are removed to prevent overfitting.'
        )        
        corr = ParamBase(
            _prm  = 'max_corr',
            _kind = float,
            _val  = ParamVal.num(val=0.9, mini=0.0, maxi=1.0, step=0.01),
            _name = 'Max Correlation',
            _desc = 'Remove highly correlated pairs',
            _tip  = 'Threshold for pairwise correlation. Variables with absolute correlation > X are considered redundant (multicollinearity) and one is dropped.'
        )          
        cat = ParamBase(
            _prm  = 'cat_max',
            _kind = int,
            _val  = ParamVal.num(val=500, mini=2, maxi=2000, step=1),
            _name = 'Max Categories',
            _desc = 'Limit of unique categories in a categorical variable',
            _tip  = 'Maximum number of unique categories allowed in a categorical variable. Variables exceeding this are dropped or require special handling.'
        )       
        th_st = ParamBase(
            _prm  = 'th_stationarity',
            _kind = float,
            _val  = ParamVal.num(val=0.1, mini=0.0, maxi=1.0, step=0.01),
            _name = 'Stationarity TH',
            _desc = 'p-value threshold for the Dickey-Fuller test',
            _tip  = 'p-value threshold for the Dickey-Fuller test. Variables with p-value > X are considered non-stationary (unstable over time).'
        )         
        th_ol = ParamBase(
            _prm  = 'th_outliers',
            _kind = float,
            _val  = ParamVal.num(val=0.01, mini=0.0, maxi=0.5, step=0.01),
            _name = 'Outliers TH',
            _desc = 'Expected proportion of outliers (noise/errors) in the data',
            _tip  = 'Expected proportion of outliers (noise/errors) in the data. Outliers algorithm uses the set value as the data cutoff threshold.'
        )
        downcast = ParamBase(
            _prm  = 'float_downcast',
            _kind = (int, bool),
            _val  = ParamVal.num(val=0, mini=0, maxi=1, step=1),
            _name = 'Float Downcast',
            _desc = 'Downcast float64 variable to float32',
            _tip  = 'When enabled, all 64-bit float variables (NumPy & PyArrow) are cast to 32-bit float to reduce memory footprint.'
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    class ENCODE(DictClass):
        """Parameters Definitions of Data Transformation Phase."""
        
        gini = ParamBase(
            _prm  = 'gini_cutoff',
            _kind = float,
            _val  = ParamVal.num(val=0.02, mini=0.0, maxi=0.15, step=0.01),
            _name = 'Min Gini Index',
            _desc = 'Filter features with lower predictive power by Gini Cutoff',
            _tip  = 'Minimum Gini Index required for a feature to be retained. Features with lower predictive power are dropped.'
        )         
        mbs = ParamBase(
            _prm  = 'min_bin_share',
            _kind = (float, int),
            _val  = ParamVal.num(val=0.03, mini=0.0, maxi=1.0, step=0.01),
            _name = 'Min Bin Share',
            _desc = 'Minimum relative size share for a bin',
            _tip  = 'Minimum relative size (share) for a bin (cat + num). Bins smaller than this are merged with adjacent bins to ensure stability.'
        )              
        bins = ParamBase(
            _prm  = 'max_num_bins',
            _kind = int,
            _val  = ParamVal.num(val=10, mini=2, maxi=30, step=1),
            _name = 'NUM Max Bins',
            _desc = 'Maximum number of bins to explore during numeric feature discretization',
            _tip  = 'Maximum N bins to explore during ML numerical binning transformation. Binning exploration goes from 2 to N.'
        )           
        num_alpha = ParamBase(
            _prm  = 'num_alpha',
            _kind = float,
            _val  = ParamVal.num(val=0.05, mini=0.0, maxi=1.0, step=0.01),
            _name = 'NUM Bins alpha',
            _desc = 'p-value level (alpha) during numeric feature discretization',
            _tip  = ('Significance level (alpha) for statistical tests during binning optimization. Different bin versions are considered statistically '
                    'equal if feature performance Bin1 ~ Bin2.')
        )            
        ixn_strategy = ParamBase(
            _prm  = 'ixn_strategy',
            _kind = int,
            _val  = ParamVal.num(val=1, mini=1, maxi=2, step=1),
            _name = 'IXN Strategy',
            _desc = 'Strategy for generating interaction features 1: LGB | 2: XGB',
            _tip  = 'Strategy for generating interaction features: <br>1: LightGBM (LGB) based interactions <br>2: XGBoost (XGB) based interactions'
        )
        ixn_leaves = ParamBase(
            _prm  = 'ixn_max_leaves',
            _kind = int,
            _val  = ParamVal.num(val=10, mini=2, maxi=20, step=1),
            _name = 'IXN Tree Leaves',
            _desc = 'Maximum number of leaves per tree to generate interaction feature',
            _tip  = ('Maximum number of leaves per tree. Note that setting > 16 constructs high-order conjunctions. Binary dummy rules may become sparse, '
                    'increasing estimation variance and potentially degrading downstream Logit validation performance.')
        )        
        ixn_target = ParamBase(
            _prm  = 'ixn_min_target',
            _kind = float,
            _val  = ParamVal.num(val=0.10, mini=0.0, maxi=1.0, step=0.01),
            _name = 'IXN Min Target',
            _desc = 'Minimum target rate required in an interaction feature',
            _tip  = 'Minimum target rate (imbalance metric) required in an interaction feature. Interactions below this threshold are pruned as noise.'
        )         
        woe_strategy = ParamBase(
            _prm  = 'woe_strategy',
            _kind = int,
            _val  = ParamVal.num(val=1, mini=1, maxi=2, step=1),
            _name = 'WOE Strategy',
            _desc = 'Strategy for WOE binning 1: Gini Search | 2: Gini + t-test Search',
            _tip  = ('Minimum target rate (imbalance metric) required in an interaction variable. Transformation Strategy to WOE '
                    '<br>1: Gini search <br>2: Gini + t-test search')
        )              
        woe_alpha = ParamBase(
            _prm  = 'woe_alpha',
            _kind = float,
            _val  = ParamVal.num(val=0.05, mini=0.0, maxi=1.0, step=0.01),
            _name = 'WOE Pruning alpha',
            _desc = 'p-value level (alpha) for bin pruning and stability during WOE',
            _tip  = 'Significance level for bin pruning and stability checks during WOE transformation.'
        )
        woe_stability = ParamBase(
            _prm  = 'woe_stability',
            _kind = float,
            _val  = ParamVal.num(val=0.65, mini=0.0, maxi=1.0, step=0.01),
            _name = 'WOE Stability TH',
            _desc = 'Stability Index of WOE Predictive behavior',
            _tip  = 'Predictive Stability Index cutoff. Features with PSI < X are removed.'
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    class TRAIN(DictClass):
        """Parameters Definitions of Data Transformation / ML Training Phase."""
        
        gb_strategy = ParamBase(
            _prm  = 'gb_strategy',
            _kind = int,
            _val  = ParamVal.num(val=1, mini=1, maxi=2, step=1),
            _name = 'GBM Strategy',
            _desc = 'Strategy for GBM Tree Optimization 1: LGB | 2: XGB',
            _tip  = 'Model training by GBM Tree Strategy: <br>1: LightGBM (LGB) <br>2: XGBoost (XGB)'
        )  
        gb_leaves = ParamBase(
            _prm  = 'gb_max_leaves',
            _kind = int,
            _val  = ParamVal.num(val=31, mini=2, maxi=127, step=1),
            _name = 'Tree Max Leaves',
            _desc = 'Maximum number of leaves per tree',
            _tip  = 'Maximum number of leaves per tree. Controls model complexity/depth.'
        )
        gb_estimators = ParamBase(
            _prm  = 'gb_estimators',
            _kind = int,
            _val  = ParamVal.num(val=100, mini=2, maxi=1000, step=1),
            _name = 'Tree N Estimators',
            _desc = 'Total number of boosting stages (trees) to be added',
            _tip  = 'Total number of boosting stages (trees) to be added.'
        )
        gb_learning = ParamBase(
            _prm  = 'gb_learning_rate',
            _kind = (float, int),
            _val  = ParamVal.num(val=0.1, mini=0.01, maxi=0.3, step=0.01),
            _name = 'Tree Learning Rate',
            _desc = 'Step size shrinkage used in updating trees',
            _tip  = 'Step size shrinkage used in updating leaves. Lower values require more trees.'
        )
        gb_min_child = ParamBase(
            _prm  = 'gb_min_child_weight',
            _kind = (float, int),
            _val  = ParamVal.num(val=0.1, mini=0.0, maxi=10, step=0.01),
            _name = 'Tree Child Weight',
            _desc = 'Minimum sum of instance weights (Hessian) needed in a child node',
            _tip  = 'Minimum sum of instance weights (Hessian) needed in a child node. Prevents overfitting by limiting splits with low data support.'
        )
        shadow_quantile = ParamBase(
            _prm  = 'shadow_quantile',
            _kind = float,
            _val  = ParamVal.num(val=0.9, mini=0.0, maxi=1.0, step=0.01),
            _name = 'Shadow Quantile',
            _desc = 'Quantile of a noise floor produced by shadow features for pruning',
            _tip  = ('During Feature Pruning the shadow features are generated as permutated orig. features to provide a noise distribution. '
                    '<br>Shadow quantile value set to 1 means max value from the noise dst (stronger pruning); 0 min (weaker pruning).')
        )
        cv = ParamBase(
            _prm  = 'cross_val_folds',
            _kind = int,
            _val  = ParamVal.num(val=3, mini=2, maxi=10, step=1),
            _name = 'Cross Validation',
            _desc = 'Number of cross validation folds during optimization / shadow folds',
            _tip  = 'Number of cross-validation folds where feature significance to model residuals is assessed.'
        )          
        glm_strategy = ParamBase(
            _prm  = 'glm_strategy',
            _kind = int,
            _val  = ParamVal.num(val=1, mini=1, maxi=3, step=1),
            _name = 'GLM Strategy',
            _desc = 'Strategy for GLM Dense Optimization 1: LassoGLM | 2: ElasticGLM | 3: LogRegGLM',
            _tip  = 'Model training by GLM Dense Strategy: <br>1: LassoGLM <br>2: ElasticGLM <br>2: LogRegGLM'
        )  
        glm_alpha = ParamBase(
            _prm  = 'glm_alpha',
            _kind = float,
            _val  = ParamVal.num(val=0.05, mini=0.0, maxi=1.0, step=0.01),
            _name = 'GLM Sparse alpha',
            _desc = 'p-value level (alpha) during Sparse GLM feature selection',
            _tip  = 'Significance level during GLM Sparse training to add/remove <i>significant</i> feature given change in model residulas p-value.'
        )
        glm_eps = ParamBase(
            _prm  = 'glm_eps',
            _kind = float,
            _val  = ParamVal.num(val=1e-3, mini=1e-5, maxi=1e-2, step=1e-5),
            _name = 'GLM Epsilon (eps)',
            _desc = 'Length of the optimization regularization path',
            _tip  = ('Epsilon (eps) represents a relative search range. Defines the ratio between the strongest and weakest regularization strength. '
                    '<br>eps=1e-3 means that alpha_min / alpha_max = 1e-3')
        )
        glm_tol = ParamBase(
            _prm  = 'glm_tol',
            _kind = float,
            _val  = ParamVal.num(val=1e-4, mini=1e-6, maxi=1e-3, step=1e-5),
            _name = 'GLM Tolerance (tol)',
            _desc = 'The tolerance level to stop the optimization once achieved',
            _tip  = 'Tolerance (tol) represents optimization accuracy. Defines when the change in the loss function is small enough to stop computing a single model.'
        )      
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    class SELECT(DictClass):
        """Parameters Definitions of Agents Selection Phase."""
        
        select_alpha = ParamBase(
            _prm  = 'select_alpha',
            _kind = float,
            _val  = ParamVal.num(val=0.05, mini=0.0, maxi=1.0, step=0.01),
            _name = 'Selection alpha',
            _desc = 'p-value level (alpha) during Agents Selection',
            _tip  = 'Significance level during tuning phase for agents comparison and cross behavior checks.'
        )
        select_bins = ParamBase(
            _prm  = 'select_bins',
            _kind = int,
            _val  = ParamVal.num(val=10, mini=5, maxi=20, step=1),
            _name = 'Selection Bins',
            _desc = 'Number of bins for probability discretization',
            _tip  = 'Selection phase transform continous probability to distinct bins for comparison and transition matrix.'
        )
        select_choice = ParamBase(
            _prm  = 'select_choice',
            _kind = int,
            _val  = ParamVal.num(val=0, mini=0, maxi=4, step=1),
            _name = 'Selection Choice',
            _desc = 'Prior Agent selection as Production candidate | 0: auto select',
            _tip  = ('Selection of the final production candidate: <br> 0: Auto select the best one based on comparison scoring '
                    '<br> 1: Sparse Selector <br> 2: Dense Selector <br> 3: PLS Selector <br> 4: Tree Selector')
        )          
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    class OUTPUT(DictClass):
        """Parameters Definitions of Flow Output."""

        save = ParamBase(
            _prm  = 'save',
            _kind = (int, bool),
            _val  = ParamVal.num(val=1, mini=0, maxi=1, step=1),
            _name = 'Save Results',
            _desc = 'Persist models, metrics and reports to the project output',
            _tip  = 'Set to 1 (True) to persist models, metrics, reports and artifacts to the project output folder.'
        )
        logging = ParamBase(
            _prm  = 'logging',
            _kind = (int, bool),
            _val  = ParamVal.num(val=1, mini=0, maxi=1, step=1),
            _name = 'Save Logging',
            _desc = 'Persist logging to the project output',
            _tip  = 'Set to 1 (True) to persist logging to the project output folder. Note that logging is independent of verbosity level and save state.'
        )          
        offload = ParamBase(
            _prm  = 'offload',
            _kind = (int, bool),
            _val  = ParamVal.num(val=0, mini=0, maxi=1, step=1),
            _name = 'Offload Relay',
            _desc = 'Offloads relay data from memory to disk (Arrow-backed)',
            _tip  = ('Offloads relay-data structures to disk via the Arrow engine to reduce memory footprint. '
                    '<br>Enable when working with large feature spaces or limited RAM. Trade-off: minor I/O overhead '
                    'on access in exchange for significantly lower peak memory usage.')
        )        
        verbose = ParamBase(
            _prm  = 'verbose',
            _kind = (int, bool, Verbosity),
            _val  = ParamVal.num(val=2, mini=0, maxi=2, step=1),
            _name = 'Verbosity Level',
            _desc = 'Controls the amount of reporting output',
            _tip  = 'Controls the amount of output: <br>0: Silent (Production mode) <br>1: Info (Standard logs) <br>2: GUI Interactive Panels & Reports'
        )      
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

PRM_links: dict[str, ParamBase] = {prm.param: prm for prm in Dict.get_flattened_values(PRM)}

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@dataclass(kw_only=True)
class Parameters:
    """
    AIDflow parametrization design.

    Under the hood, `ProjectContext.params()` provides kwargs to Parameters for
    validation and application of default options. 

    The created Parameters instance is set to the context scope as `ctx.setup` and
    serves as the global default configuration for all workers.

    Each worker inherits required parameters from `ctx.setup`, but allows custom overrides 
    at instantiation: `worker(param1=1, param2=2)`.
    
    Parameters
    ----------

    **Genesis**

    project : str    
        Unique identifier for the project. 
        Used as a root directory of model artifacts, and logging files to ensure traceability.

    target : str
        The name of the target (label) column in the input dataset. Specifies the dependent variable
        the model will learn to predict.
        
    input_data : str | Path
        Defines path and format of the input dataset. 
        1. file: file name (if in data) or full path
        2. spark: connection e.g. catalog.db.table_name
        3. db: connection e.g. postgresql://user:pass@localhost:5432/mydb

        When local files are used:
        * Files in `data` folder can be loaded by their filename only, e.g. *test.csv*.
        * Files located out of the main project dir have to use OS path to the data file.

    input_source : str = "file"
        Defines the ingestion engine. Options:
        * file:  Local or network file system (csv / gz / parquet / feather)
        * spark: Distributed Spark DataFrame
        * db:    SQL Database connection
                
    input_cols : list = None
        Explicit list of columns to load. Leave empty to load ALL available columns from the source.
        Supports also syntax by dictionary: 
        * `{"remove": "COL1", "COL2"}` 
        * `{"keep": "COL1", "COL2"}` 

    flow_vars : list = None
        Columns to retain as non-feature metadata/IDs. These are excluded from feature engineering but\
        preserved for model application (scoring) and traceability.

    yy_mm : str = None
        The column name representing the timestamp/decision date. Used for time-based train/test
        splits and PSI stability checks.
        If no time variable is specified, an artificial one is generated based on row order.

    yy_mm_format : str = None
        The strftime format string used to parse the time column. Standard Python % directives apply.
        Common Examples:
        * "2023-01-15" -> "%Y-%m-%d"
        * "15Jan23" -> "%d%b%y"

    import_start : float = 0.0
        Fraction of the dataset to start loading from. Useful if the dataset head contains stale/noisy
        data or if processing in chunks.

    import_end : float = 1.0
        Fraction of the dataset to stop loading at. Allows testing with subsets before committing to
        full data ingestion.

    train_test : float = 0.3
        Proportion of the loaded data reserved for the Test Set (held out). The remaining (1-ratio) is
        used for training/validation.

    train_valid : float = 0.33
        Proportion of the Training data further split into a Validation Set. Used for hyperparameter
        tuning and early stopping.
    
    smp_train : int|float = 1.0
        Sub-sampling ratio for the Training set to accelerate the learning during prototyping.
        * Value over 1 represents sample observation count.
        * Value between (0, 1) represents fraction of the input dataset.

    smp_feature : int|float = 1.0
        Sub-sampling ratio of input variables to reduce dimensionality and accelerate the learning
        during prototyping.
        * Value over 1 represents sample feature count.
        * Value between (0, 1) represents fraction of the features in the input dataset.


    **Data Phase**
    
    max_miss : float = 0.7
        Drop variables with missing values exceeding set percentage.

    max_1val : float = 0.9
        Drop variables where > X% of values are identical (mode). High share indicates lack of variance\
        (e.g., constant columns).

    max_var : float = 0.9
        Threshold to detect quasi-IDs (variables with extremely high cardinality/variability).
        Variables exceeding this are removed to prevent overfitting.

    max_corr : float = 0.9
        Threshold for pairwise correlation. Variables with absolute correlation > X are considered redundant\
        (multicollinearity) and one is dropped.

    cat_max : int = 500
        Maximum number of unique categories allowed in a categorical variable. Variables exceeding\
        this are dropped or require special handling.

    th_stationarity : float = 0.1
        p-value threshold for the Dickey-Fuller test. Variables with p-value > X are considered
        non-stationary (unstable over time).

    th_outliers : float = 0.01
        Expected proportion of outliers (noise/errors) in the data. Outliers algorithm uses the set
        value as the data cutoff threshold.
    
    float_downcast : int | bool = 0
        Downcast float64 variable to float32. When enabled, all 64-bit float variables (NumPy & PyArrow) 
        are cast to 32-bit float to reduce memory footprint.

    
    **Encoding Phase**

    gini_cutoff : float = 0.02
        Minimum Gini Index required for a feature to be retained. Features with lower predictive
        power are dropped.

    min_bin_share : float = 0.03
        Minimum relative size (share) for a bin (cat + num). Bins smaller than this are merged
        with adjacent bins to ensure stability.

    max_num_bins : int = 10
        aximum N bins to explore during ML numerical binning transformation. Binning exploration
        goes from 2 to N.

    num_alpha : float = 0.05
        Significance level (alpha) for statistical tests during binning optimization.
        Different bin versions are considered statistically equal if feature performance Bin1 ~ Bin2.

    ixn_strategy : int = 1
        Strategy for generating interaction features: 
        * 1: LightGBM (LGB) based interactions
        * 2: XGBoost (XGB) based interactions

    ixn_max_leaves : int = 8
        Interaction GB-algo critical gen setup of interaction depth.
        Maximum number of leaves per tree. Controls model complexity/depth.

    ixn_min_target : float = 0.10
        Minimum target rate (imbalance metric) required in an interaction feature.
        Interactions below this threshold are pruned as noise.

    woe_strategy : int = 1
        Minimum target rate (imbalance metric) required in an interaction variable.
        Transformation Strategy to WOE 
        * 1: Gini search
        * 2: Gini + t-test search

    woe_alpha : float = 0.05
        Significance level for bin pruning and stability checks during WOE transformation.

    woe_stability : float = 0.65
        Predictive Stability Index cutoff. Features with PSI < X are removed.

        
    **Training Phase**

    gb_strategy : int = 1
        Model training by GBM Tree Strategy: 
        * 1: LightGBM (LGB) 
        * 2: XGBoost (XGB)

    gb_max_leaves : int = 31
        Maximum number of leaves per tree. Controls model complexity/depth.

    gb_estimators : int = 100
        Total number of boosting stages (trees) to be added.

    gb_learning_rate : float = 0.1
        Step size shrinkage used in updating leaves. Lower values require more trees.

    gb_min_child_weight : float = 0.1
        Minimum sum of instance weights (Hessian) needed in a child node.
        Prevents overfitting by limiting splits with low data support.

    shadow_quantile : float = 0.9
        During Feature Pruning the shadow features are generated as permutated orig.
        features to provide a noise distribution. Shadow quantile value set to 1 means
        max value from the noise dst (stronger pruning) 0 min (weaker pruning).

    cross_val_folds : int = 3
        Number of cross-validation folds for shadow assessment and where feature significance 
        to model residuals is assessed.

    glm_strategy : int = 1
        Strategy for GLM Dense Optimization 
        * 1: LassoGLM 
        * 2: ElasticGLM 
        * 3: LogRegGLM

    glm_alpha : float = 0.05
        Significance level during GLM Sparse training to add/remove *significant*
        feature given change in model residulas p-value.

    glm_eps : float = 1e-3
        Epsilon (eps) represents a relative search range. Defines the ratio between
        the strongest and weakest regularization strength.
        * eps=1e-3 means that alpha_min / alpha_max = 1e-3

    glm_tol : float = 1e-4
        Tolerance (tol) represents optimization accuracy. Defines when the change in
        the loss function is small enough to stop computing a single model.

    
    **Selection Phase**

    select_alpha : float = 0.05
        Significance level during tuning phase for agents comparison and cross behavior checks.

    select_bins : float = 10
        Selection phase transform continous probability to distinct bins for comparison and
        transition matrix.

    select_choice : int = 0
        Selection of the final production candidate: 
        * 0: Auto select the best one based on comparison scoring
        * 1: Sparse Selector <br> 2: Dense Selector
        * 3: PLS Selector
        * 4: Tree Selector
    
        
    **Output**    
    
    save : bool | int = 1
        Set to 1 (True) to persist models, metrics, reports and artifacts to the project output folder.

    logging : bool | int = 1
        Set to 1 (True) to persist logging during training to the project output folder "run.log"; 
        independent of verbosity level and save state.

    offload : bool | int = 0
        Offloads relay-data structures to disk via the Arrow engine to reduce memory footprint.
        Enable when working with large feature spaces or limited RAM. Trade-off: minor I/O overhead
        on access in exchange for significantly lower peak memory usage.

    verbose : int = 2
        Controls the amount of output: 
        * 0: Silent (Production mode) 
        * 1: Info (Standard logs) 
        * 2: GUI Interactive Panels & Reports    
    """

    project:                str    
    target:                 str
    input_data:             str

    yy_mm:                  str  = None
    yy_mm_format:           str  = None

    input_source:           str  =  GEN.source.value
    input_cols:             list =  GEN.cols.value
    flow_vars:              list =  GEN.flow.value

    save:               bool|int =  PRM.OUTPUT.save.value
    logging:                 int =  PRM.OUTPUT.logging.value
    offload:            bool|int =  PRM.OUTPUT.offload.value
    verbose:                 int =  PRM.OUTPUT.verbose.value
    
    import_start:           float = PRM.LOAD.start.value
    import_end:             float = PRM.LOAD.end.value

    train_test:             float = PRM.LOAD.test.value
    train_valid:            float = PRM.LOAD.valid.value
    
    smp_train:          int|float = PRM.LOAD.sample.value
    smp_feature:        int|float = PRM.LOAD.features.value
    
    max_miss:               float = PRM.SCREEN.miss.value
    max_1val:               float = PRM.SCREEN.mode.value
    max_var:                float = PRM.SCREEN.var.value
    max_corr:               float = PRM.SCREEN.corr.value
    cat_max:                int   = PRM.SCREEN.cat.value
    th_stationarity:        float = PRM.SCREEN.th_st.value
    th_outliers:            float = PRM.SCREEN.th_ol.value
    float_downcast:      bool|int = PRM.SCREEN.downcast.value
    
    gini_cutoff:            float = PRM.ENCODE.gini.value
    min_bin_share:          float = PRM.ENCODE.mbs.value
    max_num_bins:           int   = PRM.ENCODE.bins.value
    num_alpha:              float = PRM.ENCODE.num_alpha.value
    ixn_strategy:           int   = PRM.ENCODE.ixn_strategy.value
    ixn_max_leaves:         int   = PRM.ENCODE.ixn_leaves.value
    ixn_min_target:         float = PRM.ENCODE.ixn_target.value
    woe_strategy:           int   = PRM.ENCODE.woe_strategy.value
    woe_alpha:              float = PRM.ENCODE.woe_alpha.value
    woe_stability:          float = PRM.ENCODE.woe_stability.value

    gb_strategy:            int   = PRM.TRAIN.gb_strategy.value
    gb_max_leaves:          int   = PRM.TRAIN.gb_leaves.value
    gb_estimators:          int   = PRM.TRAIN.gb_estimators.value
    gb_learning_rate:       float = PRM.TRAIN.gb_learning.value
    gb_min_child_weight:    float = PRM.TRAIN.gb_min_child.value    
    shadow_quantile:        float = PRM.TRAIN.shadow_quantile.value
    cross_val_folds:        int   = PRM.TRAIN.cv.value
    glm_strategy:           int   = PRM.TRAIN.glm_strategy.value
    glm_alpha:              float = PRM.TRAIN.glm_alpha.value
    glm_eps:                float = PRM.TRAIN.glm_eps.value
    glm_tol:                float = PRM.TRAIN.glm_tol.value

    select_alpha:           float = PRM.SELECT.select_alpha.value
    select_bins:            float = PRM.SELECT.select_bins.value
    select_choice:          int   = PRM.SELECT.select_choice.value

    def __post_init__(self):
        Setup.safe_setup(self)        
        
        self._project_structures = ProjectStructure(self.project, create=self.save)
        
        self.time_var   = AIG.time_var.value
        flow_vars       = self.flow_vars or []
        flow_vars       = [self.time_var, self.target] + flow_vars
        self.flow_vars  = list(dict.fromkeys(flow_vars))
        self._signature = LogSignature.get(self.project)

    @property
    def project_signature(self) -> str:
        """AIG Flow Params - Project's Registry Signature."""         
        return self._signature

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
        return self._project_structures.project_paths

    @property
    def STRUCTURE(self) -> dict:
        """Project structure blueprint."""
        return self._project_structures.STRUCTURE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# AIG FLOW PARAMS VALIDATION
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class Setup(Parameters):
    """AIG flow parametrization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.save_setup()
        self.save_logging()

    def get_setup(self) -> Parameters: 
        return AigParam.make_instance(Parameters, self) 

    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # PROJECT PARAMETERS SAVE
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def save_setup(self):                                                    
        if not self.save:
            return
        
        # For User convenience, save setup in a csv form
        errm  = f'AIG User Setup ► Saving User setup to csv failed!'
        setup = AigParam.user_complete_setup(self, True)  # csv format setup                             
        FileIO.save_dict_to_csv(setup, self.project_paths.setup, err=errm)       
        
        # Project's initialized parameters
        BaseIO.save(self, self.project_paths.setup) # PROJECT complete setup

    def save_logging(self):
        if not self.logging:
            return
        Logger.setup(file = self.project_paths.log, stream = False, owner = Logger.PACKAGE)
        Logger(self).info('Project Signature: ' + self.project_signature)
        Logger(self).info('Parametrization: ' + repr(self))
    # ─────────────────────────────────────────────────────────────────────────────────────────────
                    
    
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    # PARAMETERS VALIDATION
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    def safe_setup(self):      

        verify = ParamsValidator(self)
        
        # ─────────────────────────────────────────────────────────────────────────────────────────
        name = 'GENESIS'
        # ─────────────────────────────────────────────────────────────────────────────────────────
        i = (
            f'• "{self.project}" is not valid folder name → Project name has to follow standard OS folder naming.'
            f'\n• Project name is used to create a folder under {STRUCTURE[SID.O]} where all pipeline outputs can be found.'
        )
        def ar(p):
            if p is GEN.project: 
                return dict(
                    args = not Folder.is_folder_name_valid(self.project, verbose=True),                
                    info = i
                )        

        def kw(p):
            if p is GEN.time_f: 
                return dict(values = False)  # do not force option values
            
        verify.auto_check(GEN, name, ar, kw)


        # ─────────────────────────────────────────────────────────────────────────────────────────
        name = 'LOAD'
        # ─────────────────────────────────────────────────────────────────────────────────────────              
        i  = (
            'NOTE  ► Starting_Point has to be smaller than Ending_Point by definition of relative position'
            '\nstart → Use a float value between 0 to "end" to indicate that imported data starting point starts from "start"'
            '\nend   → Use a float value between "start" to 1 to indicate that imported data ending point ends at "end"'
            '\nAIG   → Current setup will be ignored by AIG Flow and start will be set to 0 during the INPUT_DATA IMPORT'
        )
        def kw(p):
            if ( p is PRM.LOAD.sample ) or ( p is PRM.LOAD.features ): 
                return dict(mini = True, min_max = False)
        
            if p is PRM.LOAD.start:
                if self.import_start >= self.import_end:
                    print(i)
            
        verify.auto_check(PRM.LOAD, name, kw_fx = kw)
                                                  
                                            
        # ─────────────────────────────────────────────────────────────────────────────────────────
        name = 'SCREENING'
        # ─────────────────────────────────────────────────────────────────────────────────────────              
        verify.auto_check(PRM.SCREEN, name)


        # ─────────────────────────────────────────────────────────────────────────────────────────
        name = 'ENCODING'
        # ─────────────────────────────────────────────────────────────────────────────────────────
        def kw(p):
            if p is PRM.ENCODE.mbs:        return dict(mini = True, min_max = False)
            if p is PRM.ENCODE.ixn_target: return dict(min_incl = False)
            
        verify.auto_check(PRM.ENCODE, name, kw_fx = kw)

        
        # ─────────────────────────────────────────────────────────────────────────────────────────
        name = 'TRAINING'
        # ─────────────────────────────────────────────────────────────────────────────────────────
        verify.auto_check(PRM.TRAIN, name)

        
        # ─────────────────────────────────────────────────────────────────────────────────────────
        name = 'SELECTION'
        # ─────────────────────────────────────────────────────────────────────────────────────────
        verify.auto_check(PRM.SELECT, name)


        # ─────────────────────────────────────────────────────────────────────────────────────────
        name = 'OUTPUT'
        # ─────────────────────────────────────────────────────────────────────────────────────────
        verify.auto_check(PRM.OUTPUT, name)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬