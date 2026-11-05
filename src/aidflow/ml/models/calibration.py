"""
Statistical model calibration engine for assessing predictive reliability.

This module implements a rigorous framework for evaluating model calibration by 
analyzing the alignment between expected probabilities E(P) and observed targets 
E(T). It provides high-performance tools for probability discretization via uniform 
and quantile binning, enabling the decomposition of model performance into discrete 
intervals to detect systematic overestimation or underestimation.
"""

from ...deps.data               import  np, pDF, pSe
from ...deps.ml.stats           import  norm
from ...deps.plots              import  go, make_subplots
from ...deps.types              import  (
                                        Literal,
                                        NDArray,
                                        )
from ...core.factory            import  DictClass
from ...core.interop            import  Pd
from ...display.plots           import  Plot
from ..tests                    import  TestSampleDiff


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# MODEL CALIBRATION STATISTICAL ASSESEMENT E(T) ~ E(P)
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    N  = lambda x: f'Model {x}'
    B  = 'BIN'
    PB = 'ProbaBIN'
    P  = 'Probability'
    T  = 'Target'
    W  = 'Weight'
    C  = 'Count'
    D  = 'PoolSTD'
    V  = 'SignificantΔ'
    A  = 'alpha_level'
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class CalibrationBinning:
    """
    Model probability discretization to bins.
    
    Parameters
    ----------
    n_bins : int 
        Number of bins to transform probability into.
    
    kind: Literal['uniform', 'quantile'] = 'uniform'
        Model probability discretization to:
         - uniform:  target equal spread in bins
         - quantile: equal pop. bins 
    """

    def __init__(self, 
        n_bins: int = 10, 
        kind: Literal['uniform', 'quantile'] = 'uniform',        
    ):
        self.n_bins = n_bins
        self.kind   = kind

    def _quantile(self, proba:pSe) -> tuple[pSe, dict]:
        return Pd.qcut(proba, self.n_bins, min_left = 0, max_right = 1, duplicates = 'drop')

    def _uniform(self, pred:pSe) -> pSe:
        bins = np.linspace(0, 1, self.n_bins + 1)
        return Pd.cut(pred, bins = bins, min_left = 0, max_right = 1, include_lowest = True)
    
    def _binning(self, a: int|str, pred: pSe) -> tuple[pSe, dict]:
        if self.kind == 'quantile':
            b = self._quantile(pred)
        else:
            b = self._uniform(pred)
        
        m = {b:i for i,b in enumerate(b.cat.categories)}
        b = b.cat.rename_categories(m)
        b.name = LBL.N(a)
        
        return b, m

    def process(self, proba: dict[int|str, pSe]) -> dict[int|str, tuple[pSe, dict]]:
        """
        Model probability discretization to pop. bins.

        Parameters
        ----------
        proba : dict[int|str, pSe]
            pd.Series dictionary of probability prediction by each model.
        
        Returns
        -------
        dict[int|str, tuple[pSe, dict]] : Probability series transformed to discrete bins series.
        """
        bins = {}
        for a, p in proba.items():
            bins[a] = self._binning(a, p)
        return bins   
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class CalibrationAssessment:
    """
    Standard model calibration as E(P) ~ E(T) given discrete bins.

    Assessment is based on evaluating model calibration by analyzing the alignment between expected 
    probabilities E(P) and observed targets E(T). It provides high-performance tools for probability 
    discretization via uniform and quantile binning, enabling the decomposition of model performance 
    into discrete intervals to detect systematic overestimation or underestimation.

    The difference E(P) ~ E(T) is evaluated by Welch's t-test given pooled standard deviation to 
    determine if drifts between predicted and observed means are statistically significant. 
    
    Parameters
    ----------
    alpha : float
        Statistical level of drift signficance between E(P) ~ E(T)

    n_bins : int 
        Number of bins to transform probability into.
    
    kind: Literal['uniform', 'quantile'] = 'uniform'
        Model probability discretization to:
         - uniform:  target equal spread in bins
         - quantile: equal pop. bins 
    """
        
    def __init__(self, 
        alpha: float = 0.05, 
        n_bins:  int = 10, 
        kind: Literal['uniform', 'quantile'] = 'uniform',
    ):
        self.alpha  = alpha
        self.n_bins = n_bins
        self.kind   = kind

    @staticmethod
    def _binning(target:pSe, pred:pSe, bins:pSe, mapping:dict) -> pDF:
        d = pDF({LBL.P: pred, LBL.T: target,}).groupby(bins).mean()  
        d[LBL.C]  = target.groupby(bins).count()
        d[LBL.W]  = d[LBL.C] / d[LBL.C].sum()
        d[LBL.PB] = {i:b for b,i in mapping.items()}
        Pd.move_column_inplace(d, LBL.PB, 0)
        d.index.name = LBL.B        
        return d

    @staticmethod
    def _pooled_std(d:pDF) -> pSe:
        """Pooled std of 2 samples."""
        m = d[LBL.P]
        t = d[LBL.T]
        return np.sqrt( 
            ( m*(1-m) / d[LBL.C] ) + 
            ( t*(1-t) / d[LBL.C] ) 
        ) 
    
    def _significance(self, d:pDF) -> pSe:
        """Welch test of 2-means stat. difference."""
        return TestSampleDiff(self.alpha).welch_mx(
                d[LBL.P], d[LBL.C], d[LBL.T], d[LBL.C]
            )
                
    def process(self, 
        target: pSe,
        proba:  dict[int|str, pSe] | pSe, 
    ) -> dict[int|str, pDF|float]:
        """
        Standard model calibration as E(P) ~ E(T) given discrete bins.

        Parameters
        ----------
        target : pSe
            Target variable data pd.Series.
        
        proba : dict[int|str, pSe] | pSe
            Series (1 model only) or dictionary of probability prediction by models to assess.
        
        Returns
        -------
        dict[int|str, pDF|float] : Calibration assessment by each model with alpha value for plots.
        """
        if isinstance(proba, pSe):
            proba = {0: proba}

        cb = CalibrationBinning(self.n_bins, self.kind)
        bins = cb.process(proba)

        cal = {}     
        for a, (b,m) in bins.items():
            p  = proba[a]
            d  = self._binning(target, p, b, m)
            d[LBL.D] = self._pooled_std(d)
            d[LBL.V] = self._significance(d)
            cal[a] = d

        cal[LBL.A] = self.alpha
        return cal    
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class CalibrationPlot:
    """
    Model calibration plots as E(P) ~ E(T) given discrete bins.

    Parameters
    ----------
    stats : dict[int|str, pDF]
        Output of ModelCalibration.process()

    dst_kind : Literal[1,2] = 2
        Distribution type:
        - 1 : Standard violin plot.          
        - 2 : Enchanced distribution plot.
    
    bar_kind : Literal[1,2] = 2
        Share type:
        - 1 : Standard bar plot.          
        - 2 : Enchanced bar plot.
        
    graph_titles : list | dict[int|str, str] = None
        Optional subtitles for each graph as list or as dict by keys in stats.
        If None → "Model {key}" is used, where key is stats keys of each model.
    """

    def __init__(self, 
        stats: dict[int|str, pDF], 
        dst_kind: Literal[1,2] = 2, 
        bar_kind: Literal[1,2] = 2,
        graph_titles: list | dict[int|str, str] = None
    ):        
        self.dst_kind = dst_kind
        self.bar_kind = bar_kind
        self.titles   = graph_titles                

        self._x = LBL.P
        self._y = LBL.T

        self._set_stats(stats)
        self._set_max_share()
        self._set_dst_data(dst_kind)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_stats(self, stats:dict) -> None:
        if not isinstance(stats, dict):
            raise ValueError('Populated ModelCalibration.process() required prior calling plots.')
        stats = {**stats}
        self._set_alpha(stats)
        
        self.stats = {}
        self._kmap = {}
        for i, (x, v) in enumerate(stats.items(), start=1):
            self.stats[i] = v
            self._kmap[i] = x

    def _set_alpha(self, stats:dict) -> None:        
        alpha = stats.pop(LBL.A, None)
        if alpha is None:
            raise ValueError(
            f'AIG SAFETY ► Input stats dictionary must contain {LBL.A} as assesement alpha value.'
            )
        self.alpha =  alpha

    def _title(self, i:int) -> str:
        x = self._kmap[i]
        if self.titles is not None:
            if isinstance(self.titles, list):
                return self.titles[i-1]
            return self.titles[x] 
        return LBL.N(x)

    def _get_calibration(self, i:int=None) -> dict[str, pDF] | pDF:
        cal = self.stats
        if i: return cal[i]
        return cal

    def _get_z_critial(self) -> float:
        a = self.alpha
        return norm.ppf(1 - a/2)
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_max_share(self) -> None:
        cal = self._get_calibration()
        x = 0
        for c in cal.values():
            x_ = c[LBL.C].max()
            x  = x_ if x_ > x else x
        self._max_share = x / c[LBL.C].sum() + 0.02
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _dst_critical(self, dst:NDArray, m_vector:NDArray, s_vector:NDArray) -> NDArray:
        """Clip DST with respect to chosen alpha value."""
        z_c = self._get_z_critial()
        return np.clip(dst, 
                np.clip(m_vector - z_c * s_vector, 0, 1), 
                np.clip(m_vector + z_c * s_vector, 0, 1)
        ) 
    
    def _create_violin_dst(self, n_points:int=200) -> dict[int, NDArray]:
        cal = self._get_calibration()
        dst = {}
        for a, c in cal.items():
            m = c[[self._y]].values
            s = c[[LBL.D]].values
            d = np.random.normal(
                loc   = m,  # y center → E(T)
                scale = s,  # pooled std
                size  = (len(m), n_points) 
            )
            dst[a] = self._dst_critical(d, m, s)            
        return dst
    
    def _create_scatter_dst(self, n_points:int=10) -> dict[int, list]:
        cal = self._get_calibration()
        z_c = self._get_z_critial()
        dst = {}
        for a, c in cal.items():
            ms_vector = c[[self._y, LBL.D]].values
            d = []
            for m, s in ms_vector:
                y_m = np.clip(m - z_c * s, 0, 1)
                y_x = np.clip(m + z_c * s, 0, 1)
                y_g = np.linspace(y_m, y_x, n_points)
                d.append( {'y': y_g, 'x': norm.pdf(y_g, loc=m, scale=s)} )
            dst[a] = d
        return dst
    
    def _set_dst_data(self, kind:int, n_points:int=200) -> None:
        if kind == 1:
            self._dst = self._create_violin_dst(n_points)
        else:
            self._dst = self._create_scatter_dst(np.clip(int(n_points/10), 15, 30))
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def _bins_share(self, fig:go.Figure, calibration:pDF, kind:int=2) -> None:
        x_ = calibration[self._x]
        y_ = calibration[LBL.C].sum()
        y_ = (calibration[LBL.C] / y_).values

        # Bar V1
        if kind == 1:
            fig.add_trace(go.Bar(
                x = x_,
                y = y_,
                width  = 0.1,
                marker = dict(
                    color = "rgba(180, 180, 180, 0.25)",
                ),
                hovertemplate = '<br>Share: %{y:.2%}<br>E(P): %{x:.2%}',
                zorder = 1
            ), secondary_y = True)
            return

        # Bar V2
        # → simple go.Bar doesn't respect bargap=0 in multiplot
        x_ = np.concatenate(([0.0], x_.values, [1.0]))
        y_ = np.concatenate(([y_[0]], y_, [y_[-1]]))

        fig.add_trace(go.Scatter(
            x = x_,
            y = y_,
            name        = f'Share',
            mode        = 'lines',
            line_shape  = 'hvh',
            fill        = 'tozeroy',
            fillcolor   = "rgba(180, 180, 180, 0.25)",
            line        = dict(color='rgba(180, 180, 180, 0.5)', width=1),
            hovertemplate = '<br>Share: %{y:.2%}<br>E(P): %{x:.2%}',
            zorder = 0
        ), secondary_y = True)

        fig.add_trace(go.Bar(
            x = x_,
            y = y_,
            width   = 0.08,
            name    = f'Share',
            marker  = dict(color="rgba(180, 180, 180, 0.5)"),
            opacity = 0.0,
            hovertemplate = '<br>Share: %{y:.2%}<br>E(P): %{x:.2%}',
            zorder = 1
        ), secondary_y = True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _bins_dst(self, fig:go.Figure, calibration:pDF, agent:int, kind:int=1) -> None:
        color = (lambda i: 
            f'rgba(191, 52,  52, 0.5)' if calibration[LBL.V].iloc[i] else
            f'rgba( 40, 80, 200, 0.1)'
        )              
        x_ = calibration[self._x]  
        d_ = self._dst[agent]
        
        width = 0.1

        if kind == 1:
            for i, dst in enumerate(d_):
                fig.add_trace(go.Violin(
                    y = dst,
                    x = [x_.iloc[i]]*len(dst),
                    points = False,
                    side        = 'positive',
                    spanmode    = 'hard',
                    scalemode   = 'width',
                    width       = width,
                    line_color  = color(i),
                    fillcolor   = color(i),
                    zorder      = 5
                ))
            return
                
        for i, dst in enumerate(d_):
            x_norm = (dst['x'] / dst['x'].max()) * (width / 2)
            x_base = x_.iloc[i]
            
            x_curve = x_base + x_norm
            y_curve = dst['y']
            
            x_path = np.concatenate((x_curve, [x_base, x_base]))
            y_path = np.concatenate((y_curve, [y_curve.max(), y_curve.min()]))

            fig.add_trace(go.Scatter(
                y = y_path,
                x = x_path,
                mode        = 'lines',
                line_shape  = 'linear',
                line=dict(
                    color     = color(i), 
                    width     = 1.5,
                    smoothing = 1.3
                ),
                fillcolor   = color(i),
                fill        = 'toself',
                name        = f"KDE",
                text        = f'E(T,P|min):  {y_curve.min():.2%} <br>E(T,P|max): {y_curve.max():.2%}',
                zorder      = 5
            ))

        fig.update_layout(xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))        
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _bins_cal(self, fig:go.Figure, calibration:pDF) -> None:
        x_ = calibration[self._x]
        y_ = calibration[self._y]

        fig.add_trace(go.Scatter(
            x = x_, 
            y = y_,
            name = f'Calibration',
            line = dict(
                color = "rgba(40, 80, 200, 1.0)",
                width = 2,
            ),
            hovertemplate = '<br>E(T): %{y:.2%}<br>E(P): %{x:.2%}',
            zorder = 4,
        ))

        l = len(y_)
        fig.add_trace(go.Scatter(
            x = np.linspace(0, 1, l),            
            y = np.linspace(0, 1, l),
            mode = 'lines',
            name = '45°',
            line = dict(
                color = "rgba(0, 0, 0, 0.5)",
                width = 1,
                dash = "dash"
            ),
            hovertemplate = '<br>E(T): %{y:.2%}<br>E(P): %{x:.2%}',
            zorder = 2
        ))        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def graph(self, a:int|str) -> go.Figure:
        """Creates calibration graph for model `a` in stats."""

        cal = self._get_calibration(a)
        fig = make_subplots(specs=[[{"secondary_y": True}]]) 

        self._bins_share(fig, cal,  kind = self.bar_kind)
        self._bins_dst(fig, cal, a, kind = self.dst_kind)
        self._bins_cal(fig, cal)


        font  = font=dict(family="Arial", size=12, color="black")
        title = lambda t: dict(text=t, font=font)
        grid  = lambda w=0.6: dict(
                gridwidth = w, 
                gridcolor = f'rgba(255, 255, 255, {w})',
        )

        ticks = np.linspace(0, 1, 9)
        tickt = dict(ticktext=['' if v == 0 else str(v) for v in ticks])

        b_ticks = np.linspace(0, self._max_share, 9)
        b_tickt = dict(ticktext=['' if v == 0 else str(round(v,3)) for v in b_ticks])
        
        axis  = lambda t: dict(title=title(t), range=[0, 1], tickvals=ticks) | grid()
        
        fig.update_layout(
            xaxis = axis('E(Probability)'),
            yaxis = axis('E(Target)' if a%2 else '') | tickt,
            yaxis2 = dict(
                title    = title('E(Share)' if not a%2 else ''),
                range    = [0, self._max_share],
                tickvals = b_ticks, 
                showgrid = False
            ) | b_tickt
        )
        return fig
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def create(self, show:bool=True) -> go.Figure | None:
        """Creates calibration graph for each model in stats."""
        cal = self._get_calibration()
        fig = Plot.create_subplots(
            list(cal), 
            self.graph, 
            lambda i: self._title(i),
            grid_cols = 2,
            secondary_y = True,
            height = 450,
            width  = 530,
            horizontal_spacing = 0.2,
            vertical_spacing   = 0.1,
            update_layout = {'margin': dict(t=20, b=20, l=0, r=0)}
        )
        if not show:
            return fig
        fig.show()                
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬