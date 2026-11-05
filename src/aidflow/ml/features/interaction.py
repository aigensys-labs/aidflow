"""
Knowledge extraction engine for deriving transparent interaction 
rules from ensemble models.

This module implements a mining framework that decomposes complex non-linear 
relationships learned by tree ensembles (such as XGBoost and LightGBM) into 
sets of discrete, human-readable binary rules. By extracting these logical 
paths and refining them through Lasso-based regularization and Simpson 
similarity pruning, the library converts black-box predictive power into 
a structured set of transparent interaction features.
"""

from ...deps.common         import  (
                                    ABC,
                                    abstractmethod
                                    )
from ...deps.struct         import  dataclass
from ...deps.data           import  np, pd, pDF, pSe
from ...deps.ml.base        import  (
                                    BaseEstimator,
                                    TransformerMixin,
                                    )
from ...deps.ml.glm         import  LassoCV
from ...deps.ml.stats       import  norm
from ...deps.exceptions     import  NotFittedError
from ...deps.types          import  (
                                    NDArray,
                                    ArrayLike,
                                    )

from ...core.factory        import  Dataclass
from ...core.standard       import  VarStd
from ...core.interop        import  Pd, Arrow

from ..models               import  XGB, LGB


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TREE RULES STRUCTURE
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@dataclass(frozen=True, slots=True)
class RuleCondition:
    """Immutable representation of a single split."""

    feature:    str
    operator:   str
    threshold:  str|float
    incl_nan:   bool
    
    def __repr__(self) -> str:
        return f"{self.feature} {self.operator} {self.threshold}"

    def _res(self, mask:pSe, col:pSe) -> pSe:
        if self.incl_nan:
            return mask | col.isna()
        return mask
    
    def check(self, X: pDF) -> pSe:
        """Vectorized rule data condition."""        
        mask = None
        col  = X[self.feature]

        if "||" in str(self.threshold):
            cats = set(map(float, self.threshold.split('||')))            
            if self.operator == "==": return  col.isin(cats)
            if self.operator == "!=": return ~col.isin(cats)
        
        th = str(self.threshold)
        if   self.operator == "==": mask = col == th
        elif self.operator == "!=": mask = col != th
        
        if mask is not None:
            return self._res(mask, col)

        th = np.float64(self.threshold)        
        if   self.operator == "<":  mask = col <  th
        elif self.operator == "<=": mask = col <= th
        elif self.operator == ">":  mask = col >  th
        elif self.operator == ">=": mask = col >= th
        else:
            raise ValueError(f"Unknown operator: {self.operator}")  

        return self._res(mask, col)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@dataclass(frozen=True, slots=True)
class Rule:
    """Represents a specific path in a tree (Intersection of conditions)."""
    conditions: tuple[RuleCondition, ...]
    
    def __repr__(self) -> str:
        return " & ".join(map(str, self.conditions))

    def __len__(self) -> int:
        return len(self.conditions)    

    def apply(self, X:pDF|pSe) -> pSe:
        """Applies the rule to a dataframe. Returns a binary Series."""
        mask = pSe(True, index=X.index)
        for condition in self.conditions:
            mask &= condition.check(X)
            if not mask.any():  # early-exit if none remain
                return mask
        return Pd.create_cat_from_codes(mask, ['0', '1'])
    
    def get_features(self) -> list:
        return [x.feature for x in self.conditions]
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@Dataclass.base
class RuleStats:
    pop_share:   float
    target_rate: float
    target_sum:  int
    data_name:   str
    features:    list
    condition:   Rule

class RuleDict:
    """Returns dictionary of generated rules."""    
    
    @staticmethod
    def rule_name(i:int|str) -> str:
        return VarStd.fe_ixn(i)
    
    @classmethod
    def from_rules(cls, rules:list) -> dict[str, Rule]:
        return {cls.rule_name(i): r for i,r in enumerate(rules)}
    
    @staticmethod
    def add_stats(y:pSe, X_rules:pDF, rules:dict[str, Rule]) -> dict:        
        stats = {}
        r_ = X_rules.mean()
        s_ = X_rules.sum()
        y_ = TreeMiner._y_support(y, X_rules)
        l  = lambda x: x.round(4)

        for i, (n,r) in enumerate(rules.items()):            
            stats[n] = RuleStats(
                pop_share   = l( r_.iloc[i] ),
                target_rate = l( y_[i] / s_.iloc[i] ),
                target_sum  = int( y_[i] ),
                data_name   = n,
                features    = r.get_features(),
                condition   = r
            )
        return stats
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TREE ADAPTERS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class TreeParserABC(ABC):
  
    @abstractmethod
    def _is_leaf(node:dict) -> bool: pass

    @abstractmethod
    def _get_op(node:dict, is_left:bool) -> tuple[str,bool]: pass

    @abstractmethod
    def extract(self, model, feature_names:list[str]) -> list[Rule]: pass

class TreeParser(TreeParserABC):

    LEFT    = 'Yes'
    RIGHT   = 'No'
    FEATURE = 'Feature'
    SPLIT   = 'Split'

    def __init__(self):
        self.rules = []
    
    @staticmethod
    def _verify_root(node_map:dict, root_id:str, tree:str) -> None:
        if root_id in node_map:
            return
        raise ValueError(
            'AIG SAFETY ► Root-ID as `tree_index-S0` has not been found '
            f'► {tree} version mismatch in output of `.trees_to_dataframe()`.'
        )

    @staticmethod
    def _create_rules_dict(rules:list):
        """Deduplicate while preserving order."""
        return RuleDict.from_rules( list(dict.fromkeys(rules)) )
    
    def _extend_conds(self, node:dict, conds:list, is_left:bool) -> None:
        feature   = node[self.FEATURE]
        threshold = node[self.SPLIT]
        operator, incl_nan = self._get_op(node, is_left)
        return conds + [RuleCondition(feature, operator, threshold, incl_nan)]
    
    def _extend_rules(self, conds:list) -> None:
        if conds: 
            self.rules.append(Rule(tuple(conds)))

    def _build_conds(self, node_map:dict, node_id:str, conds:list) -> None:
        node = node_map[node_id]
      
        left_id  = node[self.LEFT]
        right_id = node[self.RIGHT]

        if left_id in node_map:
            ext_conds = self._extend_conds(node, conds, True)
            self._traverse_tree(node_map, left_id, ext_conds)
            
        if right_id in node_map:
            ext_conds = self._extend_conds(node, conds, False)
            self._traverse_tree(node_map, right_id, ext_conds)

    def _traverse_tree(self, node_map:dict, node_id:str, conds:list):
        node = node_map[node_id]
        
        if self._is_leaf(node):
            self._extend_rules(conds)
            return

        self._build_conds(node_map, node_id, conds)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class XGBParser(TreeParser):
    
    LEFT    = 'Yes'
    RIGHT   = 'No'
    FEATURE = 'Feature'
    SPLIT   = 'Split'
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _is_leaf(node:dict) -> bool:
        return node['Feature'] == 'Leaf'

    @staticmethod
    def _get_op(node:dict, is_left:bool) -> str:
        missing  = node.get('Missing')
        decision = node.get('Category', pd.NA)

        if not pd.isna(decision): 
            op_left = "=="
            op_right = "!="
        else:
            op_left = "<"   # Yes is <
            op_right = ">=" # No is >=    
                
        if is_left:
            operator = op_left
            incl_nan = ( missing == node.get('Yes') )
        else: 
            operator = op_right
            incl_nan = ( missing == node.get('No') )

        return operator, incl_nan   
    
    def extract(self, model:XGB) -> dict[str, Rule]:
        df = model.trees_to_dataframe()
        for tree_id, group in df.groupby('Tree'):
            
            node_map = group.set_index('ID').to_dict('index')
            root_id  = f"{tree_id}-0"

            self._verify_root(node_map, root_id, 'XGB')
            self._traverse_tree(node_map, root_id, [])
                
        return self._create_rules_dict(self.rules)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class LGBParser(TreeParser):
    
    LEFT    = 'left_child'
    RIGHT   = 'right_child'
    FEATURE = 'split_feature'
    SPLIT   = 'threshold'

    def __init__(self):
        super().__init__()
                
    @staticmethod
    def _is_leaf(node:dict) -> bool:
        return pd.isna( node['split_feature'] ) or ( node['split_feature'] is None )

    @staticmethod
    def _get_op(node:dict, is_left:bool) -> tuple[str, bool]:
        missing  = node.get('missing_direction', 'left')
        decision = node.get('decision_type', '<=')

        if decision == '==':
            op_left = "=="
            op_right = "!="
        else:
            op_left = "<="
            op_right = ">"             

        if is_left:
            operator = op_left
            incl_nan = ( missing == 'left' )
        else: 
            operator = op_right
            incl_nan = ( missing == 'right' )

        return operator, incl_nan        

    def extract(self, model:LGB) -> dict[str, Rule]:
        df = model.trees_to_dataframe()        
        for tree_id, group in df.groupby('tree_index'):
            
            node_map = group.set_index('node_index').to_dict('index')
            root_id  = f"{tree_id}-S0"

            self._verify_root(node_map, root_id, 'LGB')
            self._traverse_tree(node_map, root_id, [])
                
        return self._create_rules_dict(self.rules)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬



# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TREE MINER
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class TreeMiner(BaseEstimator, TransformerMixin):
    """
    Extract interaction rules (knowledge) from tree ensembles, filter & select them with Lasso.
    Focus: non-linear interactions -> return binary rule features.
    
    The primary objective is to identify the most significant non-linear drivers of 
    the target variable while eliminating redundancy and overfitting. The resulting 
    binary rule features can be integrated back into linear models or scorecard 
    frameworks, effectively bridging the gap between high-performance gradient boosting 
    machines and the strict interpretability requirements of industrial credit scoring.

    Parameters
    ----------
    tree_generator : XGB | LGB
        Fitted tree model (XGBoost or LightGBM) or None (will be fit).    
    
    min_rule_support : float = 0.01
        Minimum fraction of rows a rule must match.
    
    min_target_support : float = 0.15
        Minimum target presence (relative to the full sample) rule must match.
        For structural extraction set this value between <0.15, 0.20>.  
    
    lasso_penalty_length : float = 0.5
        Exponent p such that rule column scaled by len(rule)^(-p).
        Default 0.5 => 1/sqrt(len).
    
    overlap_threshold : float = 0.90
        Simpson similarity to prune near-duplicates.

    overlap_target_alpha : float = 0.05
        Simpson similarity `pardon` when target diff. is significant.

    max_correlation : float = 0.90
        Phi Coefficient measuring association between two binary rules.

    max_rules : int = 500
        Cap on rules returned (after selection).

    lasso_cv : int = 3
        Number of cv samples during lasso_cv.

    random_state : int = 42
        Random state for stochastic algos.

    verbose : bool = True
        Rules info during fit.
    """

    def __init__(
        self, 
        tree_generator:       XGB|LGB, 
        min_rule_support:       float = 0.01,
        min_target_support:     float = 0.15,
        lasso_penalty_length:   float = 0.5,    # Friedman's complexity penalty
        overlap_threshold:      float = 0.90,   # Collinearity filter
        overlap_target_alpha:   float = 0.05,
        max_correlation:        float = 0.90,
        max_rules:                int = 500,
        lasso_cv:                 int = 3,
        random_state:             int = 42,
        verbose:                 bool = True
    ):
        self.tree_generator         = tree_generator
        self.min_rule_support       = min_rule_support
        self.min_target_support     = min_target_support
        self.lasso_penalty_length   = lasso_penalty_length
        self.overlap_threshold      = overlap_threshold
        self.overlap_target_alpha   = overlap_target_alpha
        self.max_correlation        = max_correlation
        self.max_rules              = int(max_rules)
        self.lasso_cv               = lasso_cv
        self.random_state           = random_state
        self.verbose                = verbose
        
        self.rules: dict[str, Rule] = {}
        self.stats: dict[str, RuleStats] = {}
        self._is_fitted = False
        self.min_target = 300
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _info(self, step:str, rules:list):
        if self.verbose:
            print('►', step, '→ rules left:', len(rules))
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _verify_cat_input(self, rules_subset:list) -> tuple[list, list]:
        cat = self.tree_generator.get_categorical_feature()
        fes = [self.get_rule(x).features for x in rules_subset]
        fes = [x for fe in fes for x in fe]
        na  = [x for x in cat if x not in fes]
        return cat, na

    @Arrow.transition(contract = Arrow.Contract(safe_conversion=True))
    def _align_cat_input(self, X:pDF, rules_subset:list=None) -> pDF:
        """Dataset categorical features alignemnt with respect to provided Tree."""
        if rules_subset: # transform (inference) subset
            cat, na = self._verify_cat_input(rules_subset)
            if cat == na:
                return X
            X = X.assign(**dict(zip(na, ['']*len(na))))
        return self.tree_generator._align_X_transform_cat(X)
    
    def _get_parser(self):
        cond = lambda solver: (
            isinstance(self.tree_generator, (XGB, LGB)) and
            self.tree_generator._solver == solver
        )
        if cond('xgb'): return XGBParser()
        if cond('lgb'): return LGBParser()
        raise ValueError("Unsupported model type.")

    def _apply_rules(self, X:pDF, rules:dict[str, Rule]) -> pDF:
        """Applies rules to X and returns DataFrame (columns rule_0..rule_n)."""
        return pDF(
            {n: r.apply(X) for n,r in rules.items()},             
        ).set_index(X.index)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_target_support(self, y:pSe) -> None:
        """Target Support for Structural patterns extraction with respect to default rate."""
        s = y.sum()
        self.min_target = int( np.clip(self.min_target_support*s, 0.01*s, 0.5*s) )

    @staticmethod
    def _y_support(y:pSe, X_rules:pDF) -> ArrayLike:
        X_int  = X_rules.values
        return X_int.T @ y.values.astype(float)

    def _prune_support(self, y:pSe, X_rules:pDF, rules:list) -> tuple[pDF, list[Rule]]:
        """Low Target-Outliers-Ghost rules pruning."""
        if X_rules.empty:
            return X_rules, rules
                        
        r_support = X_rules.mean()
        y_support = self._y_support(y, X_rules)
        mask = ( 
            ( r_support > self.min_rule_support ) & # min share of the rule
            ( y_support > self.min_target )         # min target=1 within the rule
        )
        X_rules_  = X_rules.loc[:, mask]
        rules_    = [rule for i, rule in enumerate(rules) if mask.iloc[i]]
        
        self._info("Support Pruning", rules_)
        
        return X_rules_, rules_
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _simpson_matrix(self,  X_rules:pDF) -> tuple[NDArray, ArrayLike]:
        """
        |AuB| / min(|A|,|B|)
        Simpson is asymmetric in logic measuring containment of A-B.
        If Rule B is a subset of Rule A, the denominator becomes |B|. 
        Since |A∩B| then equals |B|, the result is 1.0, regardless of how much larger A is.
        """        
        X_int = X_rules.values
        intersections = X_int.T @ X_int
        supports = np.diag(intersections)

        s_row = supports.reshape(-1, 1)
        s_col = supports.reshape(1, -1)
        min_supports = np.minimum(s_row, s_col)        
        # union_supports = s_row + s_col - intersections # Jaccard matrix denominator

        with np.errstate(divide='ignore', invalid='ignore'):
            simpson_matrix = intersections / min_supports
            
        np.fill_diagonal(simpson_matrix, 0)

        return simpson_matrix, supports
            
    def _is_significant_diff(self, y:pSe, X_rules:pDF, supports:ArrayLike):
        """Helper to decide whether to prune rule with respect to target diff."""        
        z_crit = norm.ppf(1 - self.overlap_target_alpha / 2)                
        y_sum  = self._y_support(y, X_rules)
        rates  = y_sum / supports        

        def is_significant(idx:int, cand:int) -> bool:                
            p1, n1 = rates[idx],  supports[idx]
            p2, n2 = rates[cand], supports[cand]
            
            p_pool = (y_sum[idx] + y_sum[cand]) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            
            if se == 0:
                return False
            return ( abs(p1 - p2) / se ) > z_crit
        
        return is_significant
    
    def _prune_simpson(self, y:pSe, X_rules:pDF, rules:list[Rule]) -> tuple[pDF, list[Rule]]:
        """
        Simpson Collinearity Pruning - Crucial for LogReg/WOE stability.
        
        Best for: Sparse, hierarchical tree rules.
        Logic: If Intersection / Min_Size > threshold, one is a subset of the other.
        Action: Keep the one with higher Support (Parent) or Shorter length.
        """
        if X_rules.empty:
            return X_rules, rules

        # Matrix
        simpson_matrix, supports = self._simpson_matrix(X_rules)        
        is_significant = self._is_significant_diff(y, X_rules, supports)
        
        # Order by desc. pop share and asc. simplicity
        ordering = sorted(
            range(len(rules)), 
            key=lambda i: (-supports[i], len(rules[i]))
        )
        
        # Prune
        kept_indices = []
        dropped_indices = set()
        
        for idx in ordering:
            if idx in dropped_indices:
                continue
            
            kept_indices.append(idx)
            overlap_row = simpson_matrix[idx, :]            
            candidates  = np.where(overlap_row > self.overlap_threshold)[0]
            
            for cand in candidates:
                if cand in dropped_indices or cand == idx: 
                    continue                                    
                if not is_significant(idx, cand):
                    dropped_indices.add(cand)

        # Reconstruct
        rules_   = [rules[i] for i in kept_indices]
        X_rules_ = X_rules.iloc[:, kept_indices]
        
        self._info("Simpson Pruning", rules_)

        return X_rules_, rules_
    # ─────────────────────────────────────────────────────────────────────────────────────────────
       
    def _prune_phicorr(self, y:pSe, X_rules:pDF, rules:list[Rule]) -> tuple[pDF, list[Rule]]:
        """
        Rules are generated as binary variables, thus
        Pearson/Spearman correlation is equivalent to the phi coefficient.
        Phi Coefficient measures the association between two binary variables.
        """
        if X_rules.empty:
            return X_rules, rules
                
        corr = X_rules.corr()
        triu = np.triu(np.ones(corr.shape), k=1).astype(bool)
        corr = corr.where(triu).max().fillna(0)
        mask = corr < self.max_correlation

        X_rules_  = X_rules.loc[:, mask]
        rules_    = [rule for i, rule in enumerate(rules) if mask.iloc[i]]
        
        self._info("PhiCorr Pruning", rules_)

        return X_rules_, rules_
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _prune_lassocv(self, y:pSe, X_rules:pDF, rules:list) -> tuple[pDF, list[Rule]]:
        """
        Penalized Lasso Selection (RuleFit Logic)
        Weight = 1 / (Length ^ alpha). 
        Long rules get small weights -> Lasso shrinks them.
        """
        if X_rules.empty:
            return X_rules, rules
                
        rule_lengths = np.array([len(r) for r in rules])
        weights = 1.0 / (rule_lengths ** self.lasso_penalty_length)
        
        # Apply weights to X (Simulating penalty in Lasso)
        X_weighted = X_rules * weights
        
        lasso = LassoCV(cv = self.lasso_cv, random_state = self.random_state, n_jobs = -1)
        lasso.fit(X_weighted, y)
        
        # Select features with non-zero coefs
        coefs    = np.abs(lasso.coef_)
        idx_sort = np.argsort(coefs)[::-1][:self.max_rules]
        idx_coef = [i for i in idx_sort if coefs[i] > 1e-6]
        rules_   = [rules[i] for i in idx_coef]
        X_rules_ = X_rules.iloc[:, idx_coef]

        self._info("Lasso Selection", rules_)

        return X_rules_, rules_
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, X:pDF, y:pSe):
        """
        Extract rules, 
        filter by support/overlap, 
        scale by length penalty, 
        run Lasso -> select rules.
        """
        parser = self._get_parser()
        rules  = parser.extract(self.tree_generator)
        self._info("Tree Extraction", rules)
        
        if not rules:
            self._is_fitted = True
            return self
        
        X_ = self._align_cat_input(X)
        X_rules = self._apply_rules(X_, rules)
        
        self._set_target_support(y)
        X_rules = X_rules.astype(int)
        rules = list( rules.values() )
        
        X_rules, rules = self._prune_support(y, X_rules, rules)
        X_rules, rules = self._prune_simpson(y, X_rules, rules)
        X_rules, rules = self._prune_phicorr(y, X_rules, rules)
        X_rules, rules = self._prune_lassocv(y, X_rules, rules)

        self.rules = RuleDict.from_rules(rules)
        self.stats = RuleDict.add_stats(y, X_rules, self.rules)
        self._is_fitted = True
        
        return self

    def transform(self, X:pDF, rules_subset:list|tuple|dict=None) -> pDF:
        """Transform new data to learned rule-based vars interactions."""
        if not self._is_fitted:
            raise NotFittedError("Fit first.")
        if not self.rules:
            return pDF(index=X.index)
        X_ = self._align_cat_input(X, rules_subset)
        rules = self.rules
        if rules_subset is not None:
            rules = {n:r for n,r in self.rules.items() if n in rules_subset}
        if not rules:
            raise ValueError("Provided list of rules doesn't overlap with fitted rules.")
        return self._apply_rules(X_, rules)

    def get_rule(self, rule:str) -> RuleStats:
        """Provide definition and training stats of chosen rule."""
        if not self._is_fitted:
            raise NotFittedError("Fit first.")
        return self.stats[rule]

    def get_interaction(self, rule:str) -> RuleStats:
        """Provide definition and training stats of chosen rule."""
        return self.get_rule(rule)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬