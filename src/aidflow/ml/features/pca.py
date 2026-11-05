"""
Principal Component Analysis (PCA) wrapper for feature dimensionality reduction and 
explanation.

This module implements a streamlined PCA pipeline designed to reduce the complexity of 
high-dimensional datasets while preserving maximum variance. It encapsulates the entire 
lifecycle of a component analysis—including optional feature standardization, projection 
onto a reduced subspace, and automatic handling of component limits for visual analytics.
"""

from ...deps.data       import  np, pDF, pSe
from ...deps.ml.dim     import  (
                                PCA, 
                                StandardScaler, 
                                )
from ...deps.plots      import  go
from ...deps.display    import  display
from ...deps.types      import  Callable
from ...core.factory    import  (
                                DictClass, 
                                Dataclass
                                )
from ...core.accessor   import  Rgb
from ...display.plots   import  Plot


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# PCA ANALYSIS
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class PCV(DictClass):
    VAR: str = 'Explained Variance'
    PC:  Callable = lambda i: f'PC{i}'
    CON: Callable = lambda i: f'{i}. Biggest Contributor'

@Dataclass.base
class PCResult:
    data_transformed:   pDF
    var_explained:      pSe
    pca_explained:     dict
    npc:                int
    npc_even:           int
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PCAnalysis:
    """
    A wrapper class for Principal Component Analysis (PCA) tailored for the AIG pipeline.
    
    This class encapsulates the full lifecycle of a PCA analysis:
    1. **Preprocessing**: Optionally standardizes features using `StandardScaler`.
    2. **Fitting**: Reduces dimensionality to a specified number of components (NPC).
    3. **Transformation**: Projects new data onto the learned component space.
    4. **Explanation**: Identifies the most influential original features for each component.
    
    Key Features:
    - Automatic handling of `max_axis` to limit components for visualization (e.g., keeping 2D/3D).
    - Dynamic feature selection (defaults to all columns).
    - Built-in explanation method to rank features by loading magnitude.
    - Returns a structured `PCResult` object containing transformed data and analysis metrics.
    
    Attributes
    ----------
    npc : int
        The target number of principal components to extract.
    max_axis : int, optional
        Hard limit on the number of components (useful for forcing 2D/3D plots).
    scale : bool
        If True, standardizes data (mean=0, var=1) before fitting.
    top_k : int
        Number of top contributing features to display in the explanation report.
    """
    _scaler = None

    def __init__(self,
        npc:        int = None, 
        max_axis:   int = None, 
        scale:     bool = False,
        top_k:      int = 4
    ):
        self.NPC        = npc
        self.max_axis   = max_axis
        self.scale      = scale
        self.top_k      = top_k
# ─────────────────────────────────────────────────────────────────────────────────────────────────
    
    @property
    def model(self) -> PCA:
        if not hasattr(self, '_model'):
            raise ValueError('PCA model has not been fitted yet, please run .fit() prior.')
        return self._model
    
    def _fit_scaler(self, data:pDF) -> None:
        if not self.scale:
            return
        self._scaler = StandardScaler().fit(data) 

    def _scale(self, data:pDF) -> pDF:
        if self._scaler is None:
            return data
        return self._scaler.transform(data)   

    def _set_npc_fes(self, data:pDF, features:list = None) -> None:
        """Determines the final number of components (NPC) and the list of features to use."""        
        NPC = self.NPC
        if not features: 
            features = data.columns          
        if not NPC or (NPC > len(features)): 
            NPC = len(features)    
        if self.max_axis is not None: 
            NPC = np.minimum(self.max_axis, NPC)  
        self._npc = NPC
        self._fes = features

    def fit(self, data:pDF, features:list = None) -> None:
        """
        Fits the PCA model to the provided data.
        
        Steps:
        1. Validates and sets the number of components and features.
        2. Fits the scaler (if enabled).
        3. Scales the data.
        4. Fits the PCA model on the scaled data.
        
        Parameters
        ----------
        data : pDF
            The input dataframe containing the features.
        features : list, optional
            Specific column names to analyze. Defaults to all columns.
        """        
        self._set_npc_fes(data, features)
        self._fit_scaler(data)    
        self._model = PCA(self._npc).fit(self._scale(data))

    def transform(self, data:pDF) -> pDF:
        """
        Projects data onto the PCA components.
        
        Parameters
        ----------
        data : pDF
            The input dataframe to transform. Must have the same feature order as training.
            
        Returns
        -------
        pDF
            A dataframe containing the principal component scores. 
            Columns are named PC1, PC2, etc. (using PCV.PC helper).
        """        
        return pDF(
            self.model.transform(self._scale(data)), 
            columns = [PCV.PC(i) for i in range(self._npc)], 
            index = data.index
        )
# ─────────────────────────────────────────────────────────────────────────────────────────────────
    
    def explain(self) -> pDF:
        """
        Generates an explanation report for the PCA components.
        
        Returns a dataframe where:
        - Index: Principal Components (PC0, PC1, ...).
        - Column 'VAR': Explained variance ratio for each PC.
        - Columns 'CON1' to 'CONk': The top `top_k` features contributing to that PC,
          ranked by absolute loading magnitude.
        
        Returns
        -------
        pDF
            A styled-ready dataframe summarizing variance and top contributors.
        """ 
        components = self.model.components_
        npc, fes = components.shape
        safe_k = min(self.top_k, fes)
        
        pc_idx  = [PCV.PC(i)  for i in range(npc)]
        top_fes = [PCV.CON(i) for i in range(safe_k)]

        abs_loadings = np.abs(components)
        top_indices  = np.argsort(abs_loadings, axis=1)[:, ::-1][:, :safe_k]

        comps = {}
        for i in range(npc):
            comps[pc_idx[i]] = [self._fes[idx] for idx in top_indices[i]]        

        pca = pDF(
            {PCV.VAR: self.model.explained_variance_ratio_}, index = pc_idx
        )
        pca[top_fes] = pDF(comps, index = top_fes).T
        return pca
# ─────────────────────────────────────────────────────────────────────────────────────────────────

    def process(self, data:pDF, features:list = None) -> PCResult:   
        """
        End-to-end PCA pipeline execution.
        
        Performs fitting, transformation, and explanation, returning a structured 
        `PCResult` object containing all relevant outputs for downstream reporting.
        
        Parameters
        ----------
        data : pDF
            The input dataset.
        features : list, optional
            Specific columns to analyze.
            
        Returns
        -------
        PCResult
            An object containing:
            - data_transformed: The projected data (scores).
            - var_explained: Series of explained variance ratios.
            - pca_explained: Dict containing styled explanation table and title.
            - npc: Actual number of components extracted.
            - npc_even: The largest even number <= npc (for 2D plotting logic).
        """
        self.fit(data, features)

        transformed = self.transform(data)        
        explained = self.explain()
        
        formatted = {}
        formatted['data']  = explained.style.format({PCV.VAR: '{:,.1%}'})    
        formatted['title'] = (
            'PCA Decomposition → The Most Important Vars in Principal Components Axis'.upper()
        )
        return PCResult(
            data_transformed = transformed, 
            var_explained    = explained[PCV.VAR], 
            pca_explained    = formatted, 
            npc              = int(self._npc), 
            npc_even         = int(2 * (self._npc // 2)) # PC for Graphs → 2 PC per graph
        )  

    def graph(self, data:PCResult, hue:pSe|list=None, size:pSe|list=None, matrix:int=2) -> None:
        PCGraph(data, hue, size, matrix).show()
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class PCGraph:

    def __init__(self, data:PCResult, hue:pSe|list=None, size:pSe|list=None, matrix:int=2):
        self.sample = data
        self.hue    = hue
        self.size   = size
        self.matrix = matrix
        self.colors = self._colors()
    
    def _colors(self) -> tuple[str, str]:
        return [Rgb.tableau100_hex[9], Rgb.tableau100_hex[29]]
    
    def _pca_data(self):
        """PCA Variables Aggregation to PCA Features."""
        return PCAnalysis(max_axis=8, top_k=4).process(
            self.sample[self.num], self.num
        )

    def _marker(self) -> dict:
        if self.size is None or self.hue is None:
            return {}
        return dict(
            size  = self.size, 
            color = self.hue, 
            colorscale = self.colors, 
            opacity = 0.5
        )
            
    def _scatter(self, pca_data:pDF, gc:int) -> go.Figure:
        pa  = gc*2  
        x,y = PCV.PC(pa), PCV.PC(pa+1)
        fig = go.Figure(go.Scattergl(
            x = pca_data[x], 
            y = pca_data[y], 
            mode = 'markers',
            marker = self._marker()
        ))
        fig.layout.update(yaxis=dict(title = y), xaxis=dict(title = x))
        return fig
          
    def create(self):
        """Constructs PCA plots grid."""
        pca = self.sample

        to_show = pca.npc_even / 2                     # total pca axis for graphs
        to_show = int(min(to_show, self.matrix**2))

        def graph(gc:int):
            return self._scatter(pca.data_transformed, gc)
                
        def title(gc:int):
            return f'{1+gc}. View to Outliers through Merged PC Variables'

        grid_update = dict(margin= {'t': 40,'r':10,'l':10,'b':60})
        grid_axes = dict(gridwidth=0.1, gridcolor='rgba(255, 255, 255, 0.2)')

        fig = Plot.create_subplots(
                list(range(to_show)),
                graph_fx      = graph,
                title_fx      = title,     
                height        = 370,           
                width         = 490,           
                grid_cols     = self.matrix,                
                update_layout = grid_update,
                update_xaxes  = grid_axes,
                update_yaxes  = grid_axes,
                horizontal_spacing = 0.15,
                vertical_spacing   = 0.15,
                annotations_font   = dict(family="Arial", size=12, color="black")
        )
        return Plot.adjust_to_css(fig)

    def show(self):
        """Shows go fig of PCA plots grid."""
        display(self.create())
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬