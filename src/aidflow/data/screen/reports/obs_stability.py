"""
Standard Report Interface registered with the DashRegistry for a specified MODEL.

This interface serves as the bridge between the Model's raw output and the 
Dashboard visualization layer, ensuring consistent reporting schemas across 
different model versions.
"""

from ....deps.data              import  np, pDF, pSe
from ....deps.ml.stats          import  gaussian_kde
from ....deps.plots             import  go
from ....deps.display           import  display
from ....deps.types             import  NDArray

from ....core.parameters        import  AIGFLOW
from ....core.registry          import  DashRegister
from ....core.foundation        import  AigParam
from ....core.display           import  DashInterface                                                                                
from ....core.accessor          import  Rgb, Format

from ....storage.export         import  Export
from ....ml.features            import  PCV, PCAnalysis

from ....display.plots          import  Plot
from ....display.style          import  TableStyle
from ....display.rgb            import  RgbANSI

from ..fitters.obs_analyzer     import  LBL, IO
from ..obs_stabilizer           import  REG, Fitter


MODEL = AIGFLOW.Data.obs_stabilizer
DREG  = DashRegister(MODEL)


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# DATA → OBSERVATION STABILITY → GRAPHS & DASHBOARD
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
@DREG.register(REG.AIG)
class ObservationStabilizerDash(DashInterface):
    """Dash Fitter UI."""       

    def __init__(self, 
        fitter:     Fitter, 
        verbose:    int | bool = True,
    ):
        super().__init__(fitter, verbose)
        
        self._p = fitter.params
        self._f = fitter.stats
    # ─────────────────────────────────────────────────────────────────────────────────────────────    

    def graphs(self) -> dict|None:
        """Dashboard of Outliers by PCA axis and Distribution of IN/OUT."""
        if not self._is_gui:
            return [None]*2
        
        matrix = 2

        dg = DashGraphs(self._f.sample, self._p.num_features)
        fig, pca = dg.build(matrix)        
        display(fig)

        return pca

    def pca(self, pca:dict) -> pDF|None:
        """Outliers Graphs Axes Explanation via PCA."""
        if self._is_any(
            not self._is_gui,
            pca is None,
        ):  return
        Format.title(pca['title'], style=Format.StyleStats())  
        display(pca['data'])
    # ─────────────────────────────────────────────────────────────────────────────────────────────
                
    def overview(self):
        """Outliers Overview on Full Data."""             
        if not self._is_gui:
            return
        
        _IN, _OUT = IO.IN, IO.OUT
        _DST = 'DISTANCE'
        
        # 1 Outliers General Overview
        # ─────────────────────────────────────
        ld  = self._f.len_data
        ol  = self._f.sum_ol
        cnt = pSe([ld-ol, ol], index=[_IN,_OUT])
        dst = cnt / ld

        ou1 = pDF({'Count': cnt, 'Share': dst})
        ou1.index.name = LBL.VR
        
        ou1 = ou1.reset_index()
        ou1 = (ou1.style
                .format({'Count': '{:,.0f}', 'Share': '{:,.1%}'})
                .map(lambda x:'font-weight: bold', subset=[LBL.VR])
                .hide(axis='index')
        )           
        
        # 2 Outliers Statistic by NUM Features
        # ─────────────────────────────────────
        out_stat:pDF = self._f.compare
        if out_stat.get(_OUT) is None:
            out_stat[_DST] = 0
        else: 
            out_stat[_DST] = abs((out_stat[_IN] - out_stat[_OUT]) / out_stat[_IN])

        out_stat = out_stat.sort_values(by=_DST, ascending=False)        
        ou2 = out_stat.head(5).style.format({_IN: '{:,.2f}', _OUT: '{:,.2f}', _DST: '{:,.0%}'})
        
        Format.title('OUTLIERS HIGHLIGHTS', style=Format.StyleStats())  
        t1 = f'IMPACT TO TRAIN | TH {self._f.th:.1%}'
        t2 = 'TOP 5 DISTANCE OF NUM FEATURES'
        TableStyle.combine_pdf_tables([ou1, ou2], [t1, t2]) 
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def summary(self, _is_view=True):
        d = self._f.len_data - self._f.sum_ol
        f = lambda x: f"{x:,}"
        m = ' '.join([
            Rgb.bold('► Observation Stabilizer →'),
            Rgb.bold_green(f(d)), 'Stable Train Observations →', 
            Rgb.bold_red(f(self._f.sum_ol)), 'of',
            Rgb.bold(f(self._f.len_data)), 'Dropped as Outliers'
        ])
        self._final_summary(m=m, is_view=_is_view)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    @AigParam.with_setattr_runtime(verbose=None)    
    def build(self, save=None, custom_path=None, **kwargs):
        """Construct dashboard with option to save it to default project path or custom path."""
        self.save(save, custom_path)
        pca = self.graphs() 
        self.pca(pca)
        self.overview()
        self.summary(kwargs)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DashGraphs:

    def __init__(self, sample:pDF, num:list):        
        self.sample = sample
        self.num    = num
        self.hue    = sample[LBL.OL]
        self.size   = self._size(sample)
        self.colors = self._colors()
    
    def _size(self, sample:pDF) -> pSe:
        d:pSe = -sample[LBL.SC] # -1 to make anomalies positive
        min_size, max_size = 3, 25
        norm = (d - d.min()) / np.clip(d.max() - d.min(), 1e-6, np.inf)
        power_norm = np.power(norm, 3) # Cubing for high contrast
        return min_size + power_norm * (max_size - min_size)

    def _colors(self) -> tuple[str, str]:
        return [RgbANSI.tableau100_hex[9], RgbANSI.tableau100_hex[29]]
    
    def _pca_data(self):
        """PCA Features Aggregation to PCA Features."""
        return PCAnalysis(max_axis=8, top_k=4, scale=True).process(
            self.sample[self.num], self.num
        )

    def _hist_data(self) -> tuple[NDArray, NDArray, NDArray]:
        m = self.sample[LBL.OL] == 0
        i = self.sample[m][LBL.SC]
        o = self.sample[~m][LBL.SC]

        mn = min(i.min(), o.min()) - 0.1
        mx = max(i.max(), o.max()) + 0.1        
        x  = np.linspace(mn, mx, 200)        

        kde_0 = gaussian_kde(i)
        kde_1 = lambda x: None
        try: 
            kde_1 = gaussian_kde(o) if len(o) > 10 else kde_1
        except: 
            pass

        return x, kde_0(x), kde_1(x)

    def _hist(self) -> go.Figure:
        x, y0, y1 = self._hist_data()
        fig = go.Figure()

        for i, y in enumerate([y0, y1]):
            if y is None:
                break
            c = f'rgba{*Rgb.hex_to_rgb(self.colors[i]), 0.4}'
            fig.add_trace(go.Scatter(
                x = x, 
                y = y,
                mode = 'lines',
                name = 'OUT' if i else 'IN',
                line = dict(color=c, width=2),
                fill = 'tozeroy',
                fillcolor = c,
                opacity = 0.4,
            ))
        return fig

    def _scatter(self, pca_data:pDF, gc:int) -> go.Figure:
        pa  = gc*2  
        x,y = PCV.PC(pa), PCV.PC(pa+1)
        fig = go.Figure(go.Scattergl(
            x = pca_data[x], 
            y = pca_data[y], 
            hovertext = 'Score: ' + self.sample[LBL.SC].round(4).astype(str),
            mode = 'markers',
            marker = dict(
                size  = self.size, 
                color = self.hue, 
                colorscale = self.colors, 
                opacity = 0.5
            ),
        ))
        fig.layout.update(yaxis=dict(title = y), xaxis=dict(title = x))
        return fig
          
    def pca(self, matrix):
        """Outliers through PCA."""
        pca = self._pca_data()

        to_show = pca.npc_even / 2                     # total pca axis for graphs
        to_show = int(min(to_show, matrix**2))
        is_last = lambda g: g == to_show-1

        def graph(gc:int):
            if is_last(gc):
                return self._hist()
            return self._scatter(pca.data_transformed, gc)
                
        def title(gc:int):
            if is_last(gc):
                return "Density of Model Anomaly Scores by Verdict IN/OUT"
            return f'{1+gc}. View to Outliers through Merged PC Features'

        grid_update = dict(margin= {'t': 40,'r':10,'l':10,'b':60})
        grid_axes = dict(gridwidth=0.1, gridcolor='rgba(255, 255, 255, 0.2)')

        fig = Plot.create_subplots(
                list(range(to_show)),
                graph_fx      = graph,
                title_fx      = title,     
                height        = 370,           
                width         = 490,           
                grid_cols     = matrix,                
                update_layout = grid_update,
                update_xaxes  = grid_axes,
                update_yaxes  = grid_axes,
                horizontal_spacing = 0.15,
                vertical_spacing   = 0.15,
                annotations_font   = dict(family="Arial", size=12, color="black")
        )
        fig = Plot.adjust_to_css(fig)
        return fig, pca.pca_explained
    
    def build(self, matrix:int) -> tuple[go.Figure, dict[str, pDF]]:                
        return self.pca(matrix)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬