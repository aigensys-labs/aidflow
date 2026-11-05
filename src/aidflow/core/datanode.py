"""
Dynamic data orchestration layer implementing a hierarchical state graph (DataNest).

This module manages the lifecycle and accessibility of all datasets, features, 
and metrics within AIDflow. By utilizing a recursive proxy pattern (`DataNode`), 
it enables lazy resolution of data dependencies, allowing users to define 
complex data paths that are only computed upon access.

It provides automated lineage tracking through a central registry, ensuring 
that every piece of data can be traced back to its originating model or worker.
"""

from ..deps.data        import  pDF, pSe
from ..deps.types       import  FuncType
from ..deps.weakref     import  WeakGetter
from ..utils.builtins   import  List
from .parameters        import  AIGFLOW
from .factory           import  MemoryCell
from .base              import  FIT


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA CONTEXT 
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬  
class DataNest:
    pass

class DataNode(DataNest):
    """Dynamic DataNode placeholder."""

    def __init__(self, root:str, is_nest:bool=False):
        self.root    = root
        self.is_nest = is_nest
        self._links  = {}

    def __getattr__(self, state:str) -> pDF|pSe|list|dict:
        """Dynamic resolution."""
        if state in self._links:
            getter = self._links[state]
            if isinstance(getter, DataNest):
                return getter                        
            return getter()        
        raise AttributeError(f'❌ {state} ► ✅ Available Created `states`:\n{self._repr()}')

    def __repr__(self) -> str:
        return self._repr()
    
    def _repr(self) -> str:
        return f"Entity: {self.root} ► DataNode(states={list(self._links)})"

    def set(self, state: str, getter: FuncType | DataNest) -> None:
        if not callable(getter) and not ( self.is_nest and isinstance(getter, DataNest) ):
            raise TypeError("getter must be callable")

        if self.is_nest:
            self._links[state] = getter
            return
        
        if state not in self._links:
            self._links[state] = MemoryCell(getter)
            return
        
        g: MemoryCell = self._links[state]
        g.value = getter

    def get(self, state:str) -> FuncType|DataNest:
        return self._links[state]
    
    def _set_is_nest(self, is_nest:bool) -> None:
        """
        Node Protection switcher:
         - is_nest = True  → node layers creation - adding nest
         - is_nest = False → node final edge to store data - must be callable
        """
        self.is_nest = is_nest

    def _has_nest(self, state:str, verify:bool=True) -> bool:
        getter  = self._links.get(state)        
        if verify:
            if isinstance(getter, FuncType):
                raise TypeError('AIG SAFETY ► Expected DataNest but the end node is present instead!')
        return isinstance(getter, DataNest)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DataSets:
    """Dynamic datasets placeholder."""

    def __init__(self):
        self.registry: dict[str, list] = {}
        self._links: dict[str, DataNest] = {}
        self._repr = SetsRepr(self.registry, self._links)

    def __getattr__(self, name:str) -> pDF|pSe|list|dict:
        """Dynamic resolution."""
        if name in self._links:
            getter = self.get(name)
            if isinstance(getter, DataNode):
                return getter
            if callable(getter):
                return getter()
            return getter
        raise AttributeError(f'❌ {name} ► ✅ Available `entities.states`:\n{self._repr()}')
    
    def __repr__(self) -> str:        
        return self._repr() 

    def _set_update_registry(self, model:str, links:dict|list|str) -> None:
        if isinstance(links, str):
            links = [links]
        elif not isinstance(links, list):
            links = list(links)
        if model not in self.registry:
            self.registry[model] = links
        else:
            reg = self.registry[model]            
            new = [x for x in links if x not in reg] 
            reg.extend(new)

    @staticmethod
    def _resolve(name:str) -> tuple[str, str|None]:
        return name.split('.')

    @staticmethod
    def _combine(root:str, state:list) -> str:
        return '.'.join( [root] + state )

    def _set_nest(self, root:str, state:list) -> DataNode:
        node:DataNode = self._links[root]
        node._set_is_nest(True)
        for i,s in enumerate(state[:-1], 1):
            name = self._combine(root, state[:i])
            if node._has_nest(s):
                node = node.get(s)
                node._set_is_nest(True)
            else:
                nest = DataNode(name, is_nest =True)
                node.set(s, nest)
                node = nest
        node._set_is_nest(False)
        return node
    
    def _get_node(self, root:str, state:list) -> DataNode:
        node:DataNode = self._links[root]
        for s in state[:-1]:
            node = node.get(s)
        return node
            
    def set(self, name:str, getter) -> None:
        """Registers dynamic link to worker output."""
        if not callable(getter):
            raise TypeError("getter must be callable for dynamic link")
        root, *state = self._resolve(name)
        if not state:
            self._links[root] = getter
            return
        if root not in self._links:
            self._links[root] = DataNode(root)
        node = self._set_nest(root, state)
        node.set(state[-1], getter)

    def get(self, name:str) -> FuncType|DataNode:
        """Obtain `dynamic link` to worker output."""
        root, *state = self._resolve(name)
        if not state:
            return self._links[root]
        node = self._get_node(root, state)
        return node.get(state[-1])

    def set_lambda(self, name:str, obj:object, attr:str, is_fitted:bool=True, model:str=None) -> None:
        """Creates dynamic link to worker `fitted` output and registers it."""
        getter = SetsUtils.getattr(obj, attr, is_fitted)
        if model:
            self._set_update_registry(model, name)
        self.set(name, getter)

    def set_links(self, model:str, obj:object, links:dict, is_fitted:bool=True) -> None:
        """Creates and registers dynamic links to worker `fitted` output."""
        self._set_update_registry(model, links)
        for name, attr in links.items():
            if attr == 'stats':
                self.set_lambda(name, obj, attr, False)
            else:
                self.set_lambda(name, obj, attr, is_fitted)
    
    def set_from_subset(self, 
        model: str, obj: object | list[object], links: dict[str, list], is_fitted: bool=True
    ) -> None:
        """
        Create subset list(s) given links config dict and registers it to the datanode.
        
        For instance:
         - obj = [self.fe_cleaner, self.fe_stabilizer]
         - links = { 'features.num.dropped.stable': ['vars_num', 'vars_dropped'] }
        
         1. Subset of vars_dropped in obj_1 (self.fe_stabilizer) given vars_num 
            (from obj_0 fe_cleaner) is created → [x for x in vars_dropped if x in vars_num]
         2. features.num.dropped.stable is added to dataset with value of created subset  
        """
        self._set_update_registry(model, links)
        links_subset = SetsUtils.create_subset_obj_links(obj, links, is_fitted)
        for name, getter in links_subset.items():
            self.set(name, getter)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SetsUtils:        

    @staticmethod
    def attr(attr:str|list[str], is_fitted:bool) -> str|list[str]:
        attrs = List.ensure(attr)
        res = []
        for a in attrs:
            res.append( f'{FIT.FITTED}.{a}' if is_fitted else a )
        return res[0] if len(attrs) == 1 else res
    
    @classmethod
    def getattr(cls, obj:object, attr:str, is_fitted:bool=True):
        return WeakGetter.Attr(obj, cls.attr(attr, is_fitted))
         
    @classmethod
    def create_subset_obj_links(cls,
        obj: object | list[object], links:dict[str, list], is_fitted:bool=True
    ) -> dict:
        """
        Create subset list(s) given links config dict.
        
        For instance:
         - obj = [self.fe_cleaner, self.fe_stabilizer]
         - links = { 'features.num.dropped.stable': ['vars_num', 'vars_dropped']  }
        
         1. Subset of vars_dropped in obj+1 (self.fe_stabilizer) 
            given vars_num (from obj → fe_cleaner) is created
         2. Proxy object is provided as `Proxy(vars_num = lambda: created_subset)`
         3. New links are provided as {'features.num.dropped.stable': 'vars_num'}
        """
        if not isinstance(obj, (list,tuple)):
            obj = [obj, obj]

        new_links = {}
        for name, attrs in links.items():
            new_links[name] = WeakGetter.PairAttr(
                pair    = obj, 
                attrs   = cls.attr(attrs, is_fitted), 
                eval_fx = List.subset
            )
        return new_links
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class SetsRepr:

    _adj = max( len(x) for x in AIGFLOW.Data.list() )#+ AIGFLOW.Selection.list() )
    _sets = {
        'input'     : ['input', 'dtypes'],
        'data'      : ['train', 'test'],
        'pred'      : ['predictions'],
        'features'  : ['all', 'num', 'cat', 'ixn', 'woe', 'ml'],
        'maps'      : ['encodings'],
        'stat'      : ['stats']
    }

    def __init__(self, registry:dict, links:dict):
        self._links   = links
        self.registry = registry
    
    def __call__(self) -> str:
        return self._repr_a() +'\n\n'+ self._repr_b()  

    @staticmethod
    def _title(t:str) -> str:
        d = '═'*70
        return ''.join([d, '\n', t, '\n' , d])
    
    def _repr_a(self) -> str:
        a = []
        e = [
            self._title('[a] Access:'),
            '\n► [Entity].[Scope].[State]'
            '\n→ [State]  optional given path complexity'
            '\n→ [Scope]  operates on 2 main axes - execution-driven vs. type-driven'
            '\n→ [Entity] train, test, dtypes, stats [Scope] is a flow step'
            '\n→ [Entity] predictions have the [Scope] train-test type'
            '\n→ [Entity] features & encodings have the [Scope] fe type'
        ]        
        e.append(f'\n\n{self._title("[b] Created Entities:")}')
        # j = lambda t, l: e.append( f'\n► {t} : {', '.join(l)}' ) if l else None
        j = lambda l: e.append(f'\n► {', '.join(l)}') if l else None
        for t, l in self._sets.items():
            l = [x for x in l if x in self._links]                  # elements
            g = t if (t not in l) and (t in self._links) else None  # or whole group
            l = l+[g] if g else l
            j(l)
            a.extend(l)
        m = [x for x in self._links if x not in a]
        j(m)
        return ''.join(e)

    def _repr_b(self) -> str:
        e = self._title('[c] Creation Step of Entities and their States:') + '\n'
        a = lambda m: m + ' '*(self._adj-len(m))
        return e + '\n'.join(
            f"► {i}. {a(m)} : { self._join_desc(d, a(m), i) }" 
            for i, (m,d) in enumerate(self.registry.items())
        )
    
    @classmethod
    def _join_desc(cls, desc:list, adj:str, i:int) -> None | str:
        if not desc:
            return
        d = cls._join(desc, adj)                        
        s = lambda x: f'\n{x*150}'
        p = [len(phase) for phase in AIGFLOW.values()]
        if i in [sum(p[:i])-1 for i in range(1, len(p))]: 
              d += s("═")
        else: d += s("-")
        return d

    @staticmethod
    def _join(desc:list, adj:str) -> str:
        a = ' '*(len(adj) + 8)
        r = lambda x: f'{', '.join(x)}'
        n = lambda x: f'\n{a}{r(x)}'

        l = [len(x) for x in desc]
        if sum(l) <= 120:
            return r(desc)
                
        d, c = '', 0
        for i in range(len(l)+1):
            if sum(l[c:i]) <= 110:
                continue
            m = desc[c:i-1]                
            j = n(m) if c else r(m)
            d += j
            c = i-1

        return d + n(desc[c:])
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬