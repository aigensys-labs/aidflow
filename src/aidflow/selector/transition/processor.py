"""
Cross-Agent Transition Analysis — Agent Complementarity and Disagreement.

This module implements the cross-agent transition analysis stage,
providing a detailed view of how agents' predictions relate to each
other. It answers: "Where do agents agree? Where do they disagree?
And is that disagreement meaningful?"

Core Concept: Transition Matrix
-------------------------------
For each pair of agents (A_i, A_j), the transition matrix is a
cross-tabulation:

- **Rows** (Audited): Agent A_i's probability bins (from low to high
  predicted risk).
- **Columns** (Auditor): Agent A_j's probability bins.
- **Cell value**: E(T | A_i_bin, A_j_bin) — the observed target rate
  within that cross-bin cell.

Reading the matrix:
- **Diagonal** (A_i_bin ≈ A_j_bin): Where both agents agree on the
  risk level. The target rate here should be monotonic (low→high
  risk from top-left to bottom-right).
- **Off-diagonal**: Where agents disagree. If Agent A_i says "low
  risk" but Agent A_j says "high risk" (and vice versa), the
  significant cells in these positions reveal *who is right*.

Significance Testing
--------------------
Each cross-bin cell is tested: is the target rate in that cell
significantly different from the overall rate in the corresponding
row (A_i's bin)? If yes, Agent A_j is providing **additional
information** within that A_i bin — it's "auditing" A_i's prediction.

The summary aggregates:
- **Audited score** (per agent): How much "blind spot" this agent
  has — the mean significant divergence found by other agents
  within its bins.
- **Auditor score** (per agent): How much new information this
  agent reveals about other agents' bins.

Visualization
-------------
Two complementary views:
1. **Transition Matrix** (heatmap): The full cross-tabulation with
   color-coded significance and total rates.
2. **Target Evolution** (scatter + line): For each bin of the
   audited agent, shows how the target rate is decomposed across
   the auditor agent's bins — revealing whether the auditor adds
   signal within each of the audited agent's bins.

Interpretation Guide
--------------------
- Agent A is "stronger" than B if A's bins show more significant
  divergence within B's bins (A audits B) than vice versa.
- High off-diagonal significance = agents capture **different**
  information (complementary).
- Low off-diagonal significance = agents capture **similar**
  information (redundant).
- A well-calibrated agent has low audited score (few blind spots).
"""

from ...deps.data                   import  np, pd, pDF, pSe
from ...deps.plots                  import  go, make_subplots
from ...deps.types                  import  (
                                            NDArray, 
                                            )
from ...core.base.fitter            import  FitterModelBase
from ...core.factory                import  (
                                            Dataclass, 
                                            DictClass
                                            )
from ...core.foundation             import  AigParam
from ...core.interop                import  Pd
from ...ml.models.transition        import  (
                                            LBL as LBL_T,
                                            CrossTab,
                                            ModelsCrossTransition, 
                                            ModelCrossDivergence
                                            )

from ..base.processor               import  (
                                            AGENTS,
                                            FitterSelectionBase
                                            )
from ..predict.adapter              import  PredictorAdapter                                            


# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# TUNING → AGENTS TRANSITION
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
class LBL(DictClass):
    """Labels for agent pair identification and parsing."""
    N = lambda x: f'Agent {x}'
    X = lambda a1, a2: f'{a1} vs. {a2}'
    R = lambda a: [int(x) for x in a.split(' vs. ')]
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

@Dataclass.base
class TransitionParamsBase:
    """
    Transition analysis parameters.

    Attributes
    ----------
    alpha : float, default=0.05
        Statistical significance level for testing whether the
        target rate in a cross-bin cell is significantly different
        from the row total.
    proba_bins : int, default=10
        Number of quantile bins for probability discretization.
        Both agents are binned into the same number of equal-
        population bins for comparability.
    """    
    alpha:      float = 0.05
    proba_bins:   int = 10

@Dataclass.base(update_specs=dict(slots=False))
class TransitionParamsCfg:
    """
    Transition data source configuration.

    Attributes
    ----------
    predictors : PredictorAdapter
        Adapter holding all trained agents.
    target : str
        Target column name in the evaluation data.
    preprocess : bool | None, default=None
        If True, apply the full encoding pipeline to input data.
    """    
    predictors: PredictorAdapter
    target:     str
    preprocess: bool|None = None

@Dataclass.base
class TransitionParams(TransitionParamsBase, TransitionParamsCfg):
    """Woker params input design."""

@Dataclass.base
class TransitionInput:
    """
    Transition analysis input specification.

    Attributes
    ----------
    data : pDF
        OOT evaluation data (must contain the target column).
    agents : tuple | list | None
        Agents to include in the transition analysis.
        If None, all registered agents are used.
        Requires ≥ 2 agents (for at least one pair).
    """
    data:   pDF 
    agents: tuple|list|None = tuple(AGENTS)

@Dataclass.base
class TransitionResult:
    """
    Transition analysis output — the decision-relevant tables.

    Attributes
    ----------
    vectors : dict
        Per-pair transition vectors (mean rates, significance,
        counts) for plotting and inspection.
    summary : pDF
        Aggregated audited/auditor scores per agent.
    """    
    vectors: dict
    summary: pDF

@Dataclass.base
class TransitionStats:
    """
    Full transition diagnostic output.

    Attributes
    ----------
    crosstab : dict
        Per-pair cross-tabulation tables (CrossTab objects).
        Structure: {a1: {a2: CrossTab}}
    significant : dict
        Per-pair significance tables (which cells are significantly
        different from the row total).
        Structure: {a1: {a2: pDF}}
    vectors : dict
        Per-pair transition vectors for plotting.
        Structure: {'1 vs. 2': TransitionVectors, ...}
    summary : pDF
        Aggregated audited/auditor scores per agent.
    """    
    crosstab:    dict
    significant: dict
    vectors:     dict
    summary:     pDF

@Dataclass.base
class TransitionVectors:
    """
    Per-pair transition vectors (for plotting).

    Attributes
    ----------
    mean : pDF
        Cross-bin target rates (E(T | bin_i, bin_j)) with
        row/column totals.
    sig_mean : pDF
        Significance flags per cell (NaN = not significant,
        value = significant difference from row total).
    count : pDF
        Observation counts per cross-bin cell.
    """    
    mean:       pDF
    sig_mean:   pDF
    count:      pDF
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Fitter(FitterSelectionBase[TransitionInput, TransitionParams]):
    """
    Transition Analysis Worker — cross-agent crosstab computation.

    Executes the full transition analysis:
    1. Predict with all agents.
    2. Binarize into quantile bins.
    3. Compute pairwise cross-tabulations.
    4. Test significance of cross-bin differences.
    5. Assemble transition vectors and summary scores.

    Results are stored in `self.stats` (`TransitionStats`).

    Parameters
    ----------
    params : TransitionParams
        Transition analysis configuration.
    """    
    SelectionInput:  type = TransitionInput
    agents_condition: int = 2

    def __init__(self, params: TransitionParams):
        super().__init__(params)
        self.predictors = self.prm.predictors
    # ─────────────────────────────────────────────────────────────────────────────────────────────
        
    def predict(self, data:pDF) -> dict[int, pSe]:
        """
        Obtain probability predictions from all requested agents.

        Parameters
        ----------
        data : pDF
            Evaluation data.

        Returns
        -------
        dict[int, pSe]
            {agent_key: probability_series}
        """
        pred = {}
        for a in self.agents:
            pa = self.predictors.get_agent(a)
            pred[a] = pa.predict(data, preprocess = self.prm.preprocess).squeeze()
        return pred
    
    def _bins(self, a:int, pred:pSe) -> dict[int, pSe]:
        b = Pd.qcut(pred, self.prm.proba_bins, min_left=0, max_right=1, duplicates='drop')
        m = {q:i for i,q in enumerate(b.cat.categories)}
        b = b.cat.rename_categories(m)
        b.name = LBL.N(a)
        return b

    def binning(self, pred: dict) -> dict[int, pSe]:
        """
        Discretize all agents' probabilities into quantile bins.

        Parameters
        ----------
        pred : dict[int, pSe]
            Per-agent probability series.

        Returns
        -------
        dict[int, pSe]
            Per-agent binned series.
        """
        bins = {}
        for a in self.agents:
            bins[a] = self._bins(a, pred[a])
        return bins
    
    @staticmethod
    def transition_vectors( 
        x_tab: dict[int, dict[int, CrossTab]], 
        diff:  dict[int, dict[int, pDF]]
    ) -> dict[str, TransitionVectors]:
        """
        Assemble transition vectors for all agent pairs.

        For each pair (a0, a1), extracts:
        - **mean**: Cross-bin target rates with totals.
        - **sig_mean**: Significance flags (NaN = not significant).
        - **count**: Observation counts per cell.

        Parameters
        ----------
        x_tab : dict[int, dict[int, CrossTab]]
            Pairwise cross-tabulation tables.
        diff : dict[int, dict[int, pDF]]
            Pairwise significance tables.

        Returns
        -------
        dict[str, TransitionVectors]
            {'1 vs. 2': TransitionVectors, '1 vs. 3': ..., ...}
        """
        vectors = {}

        for a0, x in x_tab.items():
            for a1, y in x.items():

                c = y.count.copy()
                c.loc[LBL_T.S, :] = c.sum(axis=0)
                c.loc[:, LBL_T.S] = c.sum(axis=1)
                
                s = diff[a0][a1].copy()        
                s.loc[LBL_T.S, :] = np.nan
                s.loc[:, LBL_T.S] = np.nan
                
                vectors[LBL.X(a0, a1)] = TransitionVectors(
                    mean = y.total,
                    sig_mean = s,
                    count = c,
                )
        return vectors
    
    @staticmethod
    def signficant_summary(x_tab:dict, diff:dict) -> pDF:
        """
        Compute the aggregated audited/auditor scores per agent.

        For each agent:
        - **Audited** (weakness): Mean significant divergence found
          by other agents within its bins. Lower = fewer blind spots.
        - **Auditor** (strength): Mean information revealed about
          other agents' bins. Higher = more discriminative breadth.

        Parameters
        ----------
        x_tab : dict
            Pairwise cross-tabulations.
        diff : dict
            Pairwise significance tables.

        Returns
        -------
        pDF
            Per-agent audited and auditor scores.
        """
        div   = ModelCrossDivergence.process(diff, x_tab)
        div_s = ModelCrossDivergence.get_summary(div)        
        sig = []
        for d in div_s.values():
            audited = d[LBL_T.AE][LBL_T.S].mean()
            auditor = d[LBL_T.AO][LBL_T.S].mean()
            sig.append({LBL_T.AE: audited, LBL_T.AO: auditor})
        return pDF(sig, index = list(div_s))
    # ─────────────────────────────────────────────────────────────────────────────────────────────    
        
    def process(self, inputs: TransitionInput | pDF, **kwargs) -> None:
        """
        Execute the full cross-agent transition analysis.

        Steps:
        1. Predict with all agents.
        2. Binarize into quantile bins.
        3. Compute pairwise cross-tabulations (with bin alignment).
        4. Test significance of cross-bin differences.
        5. Assemble transition vectors.
        6. Compute audited/auditor summary.

        All results stored in `self.stats`.

        Parameters
        ----------
        inputs : TransitionInput | pDF
            Evaluation data or input specification.
        kwargs : dict
            Additional arguments (e.g., `agents=(1, 2)`).
        """
        inputs_ = self.get_inputs(inputs, kwargs)
        target  = inputs_.data[self.prm.target]

        pred = self.predict(inputs_.data)
        bins = self.binning(pred)

        cc = ModelsCrossTransition(self.prm.alpha, align_bins=True)
        x_tab = cc.process(target, bins)
        diff  = cc.get_sig_diff(x_tab)
        
        vectors = self.transition_vectors(x_tab, diff)
        summary = self.signficant_summary(x_tab, diff)

        self.stats = TransitionStats(
            crosstab    = x_tab,
            significant = diff,
            vectors     = vectors,
            summary     = summary
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class TransitionProcessor(FitterModelBase[Fitter]):
    """
    Cross-Agent Transition Analysis — Public API.

    Provides a detailed view of how agents' predictions relate to
    each other: where they agree, where they disagree, and whether
    that disagreement is statistically meaningful.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for cross-bin difference testing.
    proba_bins : int, default=10
        Number of quantile bins for probability discretization.
    predictors : PredictorAdapter
        Adapter holding trained agents.
    preprocess : bool | None, default=None
        If True, apply the encoding pipeline to input data.
    target : str
        Target column name.

    Output
    ------
    - **vectors**: Per-pair transition data (rates, significance,
      counts) — the raw material for plotting and inspection.
    - **summary**: Aggregated audited/auditor scores per agent.

    Visualization
    -------------
    Use `TransitionPlot` for interactive heatmaps and target
    evolution scatter plots.

    Examples
    --------
    >>> from aidflow.fitters import TransitionProcessor, TransitionPlot
    >>> tr = TransitionProcessor(
    ...     predictors=my_adapter,
    ...     target='is_churn',
    ...     proba_bins=10,
    ... )
    >>> result = tr.process(oot_data)
    >>> result.summary  # audited/auditor scores
    >>> plot = TransitionPlot(tr.fitter.stats)
    >>> plot.matrix()     # transition heatmap
    >>> plot.evolution()  # target evolution scatter
    >>> plot.dash()       # combined view
    """

    def __init__(self, *args, params: dict|TransitionParams = None, **kwargs):
        self._params = AigParam.param_ar_kw(TransitionParams, params, args, kwargs)
        self._fitter = Fitter(self._params)

    def process(self, inputs: TransitionInput | pDF, **kwargs) -> TransitionResult:
        """
        Execute the transition analysis and return the results.

        Parameters
        ----------
        inputs : TransitionInput | pDF
            OOT evaluation data (must contain the target column).
        kwargs : dict
            Additional arguments (e.g., `agents=(1, 2)`).

        Returns
        -------
        TransitionResult
            Transition vectors and summary scores.
        """
        self._fitter.process(inputs, **kwargs)
        stats: TransitionStats = self.stats
        return TransitionResult(
            vectors = stats.vectors,
            summary = stats.summary
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
    

class TransitionPlotBase:
    def __init__(self, stats: TransitionStats):
        if stats is None or not isinstance(stats, TransitionStats):
            raise ValueError('Populated TransitionStats instance required prior calling plots.')

        self.stats = stats
        self.agents_comb = self._get_agents_combination()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _get_vectors(self, a:str=None) -> TransitionVectors | dict[str, TransitionVectors]:
        v = self.stats.vectors
        if a is None:
            return v
        return v[a]
    
    def _get_crosstab(self, a:int=None) -> dict[int, CrossTab] | dict[int, dict[int, CrossTab]]:
        c = self.stats.crosstab
        if a is None:
            return c
        return c[a]
    
    def _get_significant(self, a:int=None) -> dict[int, pDF] | dict[int, dict[int, pDF]]:
        s = self.stats.significant
        if a is None:
            return s
        return s[a]
    
    def _get_agents_combination(self) -> list:
        return list(self._get_vectors())
    
    @staticmethod
    def _dropdown_layout(buttons:list, y_offset:float=1.14) -> dict:
        return dict(
            updatemenus = [dict(
                buttons    = buttons,
                direction  = "down",
                showactive = True,
                x = 0.0, 
                y = y_offset,
                xanchor = "left", 
                yanchor = "top"
            )]
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class TransitionMatrix(TransitionPlotBase):
    """
    Transition Matrix Heatmap Visualization.

    Renders the cross-agent transition matrix as an interactive
    heatmap with:
    - **Color coding**: Significance and diagonal position.
    - **Annotations**: Target rate, significance label, and count
      in each cell.
    - **Dropdown**: Switch between agent pairs.
    - **Hover**: Full details (cross rate, total rate, significance, count).

    Color Legend
    -----------
    - **Light Blue**: Not significant, off-diagonal (agents disagree,
      but not meaningfully).
    - **Light Red**: Significant, off-diagonal (agents disagree
      meaningfully — one is "auditing" the other).
    - **Deep Blue**: Not significant, on-diagonal (agents agree,
      no additional info).
    - **Deep Red**: Significant, on-diagonal (agents agree AND the
      cell is significantly different from the marginal — strong
      signal).
    - **Slate Grey**: Row/column totals.
    - **Dark Charcoal**: Grand total.
    """

    @staticmethod
    def _xy_totals_naming(agents:str, m:pDF) -> tuple[NDArray, NDArray]:
        a1, a2 = LBL.R(agents)
        n = lambda a: f'E(T|A{a})'

        x = np.array(m.columns.values)
        y = np.array(m.index.values)
        x[-1] = n(a1)
        y[-1] = n(a2)
        return x, y
        
    @staticmethod
    def _get_colorscale():
        return [   
            [0.0, "rgba(240, 240, 240, 1)"],   # 0: Light Grey (Empty)
            [0.142, "rgba(240, 240, 240, 1)"],
            
            [0.142, "rgba(215, 235, 255, 1)"], # 1: Light Blue (Not Sig, Off-Diag)
            [0.285, "rgba(215, 235, 255, 1)"],
            
            [0.285, "rgba(255, 200, 200, 1)"], # 2: Light Red (Sig, Off-Diag)
            [0.428, "rgba(255, 200, 200, 1)"],
            
            [0.428, "rgba(100, 165, 230, 1)"],  # 3: Deep Blue (Not Sig, On-Diag)
            [0.571, "rgba(100, 165, 230, 1)"],
            
            [0.571, "rgba(235, 90, 90, 1)"],   # 4: Deep Red (Sig, On-Diag)
            [0.714, "rgba(235, 90, 90, 1)"],
            
            [0.714, "rgba(200, 205, 210, 1)"], # 5: Slate Grey (Totals Row/Col)
            [0.857, "rgba(200, 205, 210, 1)"],
            
            [0.857, "rgba(70, 80, 90, 1)"],    # 6: Dark Charcoal (Grand Total Portfolio)
            [1.0,   "rgba(70, 80, 90, 1)"]
        ]
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _create_color_matrix(self) -> dict[str, NDArray]:
        """Creates Matrix coloring with respect to underlying values."""
        colors = {}
        for agents in self.agents_comb:
            v = self._get_vectors(agents)
            m = v.mean
            s = v.sig_mean
            
            l = len(m.index)
            color_matrix = np.zeros(m.shape)
            
            for i in range(l):
                for j in range(l):
                    m_val = m.iloc[i, j]
                    s_val = s.iloc[i, j]
                    
                    # Totals
                    if l-1 in [i, j]:
                        if (i == j):
                            color_matrix[i, j] = 6
                            continue
                        color_matrix[i, j] = 5
                        continue
                    
                    # nans    
                    if pd.isna(m_val):
                        color_matrix[i, j] = 0  
                        continue
                        
                    is_sig = not pd.isna(s_val)
                    is_diagonal = (i == j)
                    
                    if is_diagonal:
                        color_matrix[i, j] = 4 if is_sig else 3
                    else:
                        color_matrix[i, j] = 2 if is_sig else 1
            
            colors[agents] = color_matrix
        
        return colors

    def _create_annotations(self, font_size:int = 10) -> dict:
        """Creates all Matrix Annotations."""        

        def _is_total(is_total:bool) -> bool:
            return (is_total and row==col)
        
        def mean(m_val:float, is_total:bool=False):
            c = ('white' if _is_total(is_total) else 'black')
            return dict(
                x=col, y=row, 
                text = f"<b>{m_val:.1%}</b>", 
                showarrow = False,
                font = dict(size = font_size, color = c)
            )   
                 
        def count(c_val:int, is_total:bool=False):
            return dict(
                x=col, y=row, 
                text = f"n:{int(c_val)}",
                showarrow = False, 
                xref = "x", 
                yref = "y",
                xanchor = "right", 
                yanchor = "bottom", 
                xshift = 25, 
                yshift = -25, 
                font = dict(
                    size  = int(font_size/1.1), 
                    color = ("rgba(0,0,0,0.5)" if not _is_total(is_total) else 'white')
                )
            )
        
        def sig(s_val:float):
            is_sig = not pd.isna(s_val)
            return dict(
                x=col, y=row,
                text = f"<i>{'significant' if is_sig else 'not sig.'}</i>",
                showarrow = False, 
                xref = "x",
                yref = "y",
                xanchor = "right", 
                yanchor = "top", 
                xshift = 25, 
                yshift = 25, 
                font = dict(
                    size  = font_size/1.2, 
                    color = ("red" if is_sig else "rgba(0,0,0,0.4)"))
            )            
        
        annotations = {}
        for agents in self.agents_comb:
            v = self._get_vectors(agents)
            m = v.mean
            s = v.sig_mean            
            c = v.count
            
            anno = []
            for i, row in enumerate(m.index):
                for j, col in enumerate(m.columns):                
                    m_val = m.iloc[i, j]
                    s_val = s.iloc[i, j]
                    c_val = c.iloc[i, j]

                    # Totals
                    if LBL_T.S in [str(row), str(col)]:
                        anno.append(mean(m_val, True))
                        anno.append(count(c_val, True))
                        continue
                    
                    # nans
                    if pd.isna(m_val):
                        anno.append(dict(
                            x=col, y=row, 
                            text = "nan", 
                            showarrow = False,
                            font = dict(size=font_size, color="grey")
                        ))
                        continue

                    anno.append(mean(m_val))
                    anno.append(sig(s_val))
                    anno.append(count(c_val))
                    
            annotations[agents] = anno

        return annotations

    def _get_titles(self) -> dict:
        titles = {}
        for agents in self.agents_comb:
            m = self._get_vectors(agents).mean
            titles[agents] = {
                "x_title": f"{m.columns.name} → Auditor",
                "y_title": f"{m.index.name} → Audited"
            }
        return titles
    
    def _dropdown_menu(self, annotations:dict, titles:dict) -> list[dict]:
        l = len(self.agents_comb)
        dropdown_buttons = []
        for i, agents in enumerate(self.agents_comb):
            visibility_mask = [False] * l
            visibility_mask[i] = True # Turn on ONLY this specific trace index
            
            dropdown_buttons.append(dict(
                label  = agents,
                method = "update",
                args  = [
                    {
                        "visible": visibility_mask
                    },
                    {
                        "annotations": annotations[agents],
                        "xaxis.title.text": titles[agents]["x_title"],
                        "yaxis.title.text": titles[agents]["y_title"]
                    }
            ]))
        return dropdown_buttons
    
    def _layout(self, fig:go.Figure, annotations:dict, titles: dict, vis:int=0) -> None:
        default = self.agents_comb[vis]
        
        axis_cleaner = dict(
            showgrid = False,
            zeroline = False,
            showline = False,
            type = 'category'
        )        

        fig.update_layout(
            width  = 800, 
            height = 800,
            plot_bgcolor = 'rgb(240, 240, 240)', 

            xaxis = dict(
                title = titles[default]["x_title"], side = "top"
            ) | axis_cleaner,

            yaxis = dict(
                title = titles[default]["y_title"], autorange = "reversed"
            ) | axis_cleaner,

            margin = dict(l=0, r=0, t=0, b=0),
            
            annotations = annotations[default]
        )        
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _create(self, vis:int = 0) -> go.Figure:                
        color_scale  = self._get_colorscale()
        color_matrix = self._create_color_matrix()

        fig = go.Figure()

        for i, agents in enumerate(self.agents_comb):
            v = self._get_vectors(agents)
            m = v.mean
            s = v.sig_mean
            c = v.count
            t = np.column_stack([m.loc[:, LBL_T.S].values]*len(m))            

            fig.add_trace(go.Heatmap(
                x = m.columns,
                y = m.index,
                z = color_matrix[agents],
                colorscale = color_scale,
                zmin = 0, 
                zmax = len(color_scale)//2,
                showscale = False,
                xgap = 2, 
                ygap = 2,
                visible = (i == vis),
                customdata = np.stack((m.values, t, 1-pd.isna(s.values), c.values), axis=-1),
                hovertemplate = (
                    "Bin: Audited %{y} | Auditor %{x}<br>"
                    "E(T | Cross): %{customdata[0]:.1%}<br>"
                    "E(T | Total):   %{customdata[1]:.1%}<br>"
                    "Significant(Total - Cross): %{customdata[2]}<br>"
                    "Count: %{customdata[3]}<extra></extra>"
                )
            ))

        return fig
    
    def create(self, show:bool = True) -> go.Figure | None:
        """
        Create and display the transition matrix heatmap.

        Parameters
        ----------
        show : bool, default=True
            If True, display the plot. If False, return the Figure.

        Returns
        -------
        go.Figure | None
            The Plotly figure (if show=False).
        """        
        annotations  = self._create_annotations()
        titles       = self._get_titles()
        buttons      = self._dropdown_menu(annotations, titles)

        vis = 0
        fig = self._create(vis)
        self._layout(fig, annotations, titles, vis)

        fig.update_layout(
            **self._dropdown_layout(buttons)
        )

        if not show:
            return fig        
        fig.show()

    def _get_raw_components(self, vis:int=0) -> tuple[list, dict, dict]:        
        return (
            list( self._create(vis).data ),
            self._create_annotations(), 
            self._get_titles()
        )
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class TargetEvolutionPlot(TransitionPlotBase):
    """
    Target Evolution Scatter Visualization.

    For each bin of the audited agent (y-axis), shows how the
    target rate is distributed across the auditor agent's bins
    (x-position within the bin). The total target rate for each
    audited bin is shown as a line.

    Interpretation
    --------------
    - **Markers spread within a bin**: The auditor agent is
    decomposing the audited agent's bin into sub-segments with
    different target rates. Significant markers (red) indicate
    the auditor is finding meaningful structure the audited
    agent missed.
    - **Markers clustered at one point**: The auditor adds no
    information within that bin.
    - **Total line** (blue): The overall target rate for each
    audited bin — the "baseline" that the auditor is trying to beat.

    Parameters
    ----------
    stats : TransitionStats
        Populated transition stats.
    """    
    def __init__(self, stats: TransitionStats):
        super().__init__(stats)
        self._set_bins()
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _set_bins(self):        
        ct = self._get_crosstab()
        a1 = next(iter(ct))
        a2 = next(iter(ct[a1]))
        c:CrossTab = ct[a1][a2]
        self._bins = c.mean.index.to_list()

    def _get_bin_data(self, a1:int, a2:int) -> dict[int, dict[str, pSe]]:
        """
        Returns proba and count for each bin of Audited agent (row) given 
        spread by Auditor agent (cols).
        """
        t = self._get_crosstab(a1)[a2]
        s = self._get_significant(a1)[a2]
        m = t.mean
        c = t.count
        s_= t.total.loc[:, LBL_T.S]
        d = {}
        for b in self._bins:
            mask = c.loc[b,:] > 1
            d[b] = {
                'bins':  m.loc[b, mask].index.to_numpy(), # auditor bins
                'proba': m.loc[b, mask].values, 
                'count': c.loc[b, mask].values, 
                'sig': ~pd.isna(s.loc[b, mask].values),
                'total': np.array([s_.loc[b]]*sum(mask))
            }
        return d
    
    @staticmethod
    def _xaxis_title(agents:str) -> str:
        a1, a2 = LBL.R(agents)
        return f'{LBL.N(a1)} Proba Bins | Audited by {LBL.N(a2)}'
    
    def _get_titles(self) -> dict:
        titles = {}
        for agents in self.agents_comb:
            titles[agents] = self._xaxis_title(agents)
        return titles    
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _bins_dst(self, fig:go.Figure, a1:int, a2:int, is_visible:bool=True) -> None:
        color = lambda: [
            f'rgba(191, 52,  52, 0.5)' if x else
            f'rgba( 40, 80, 200, 0.1)' for x in dst['sig']
        ]
        size = lambda: (dst['count'] / dst['count'].sum()) * 100
        
        d = self._get_bin_data(a1, a2)

        for i, dst in d.items():
            y = dst['proba']
            fig.add_trace(go.Scatter(
                y = y,
                x = [i]*len(y),
                mode    = 'markers',
                marker  = dict(color = color(), size = size()),
                name    = f"E(T | Auditor)",
                zorder  = 5,
                visible = is_visible,
                customdata = np.column_stack(list(dst.values())),
                hovertemplate = (
                    "Bin: Audited %{x} | Auditor %{customdata[0]}<br>"
                    "E(T | Cross): %{customdata[1]:.1%}<br>"
                    "E(T | Total):   %{customdata[4]:.1%}<br>"
                    "Significant(Total - Cross): %{customdata[3]}<br>"
                    "Count: %{customdata[2]}<extra></extra>"
                )
            ))
    
    def _bins_total(self, fig:go.Figure, a1:int, a2:int, is_visible:bool=True) -> None:
        y_ = self._get_crosstab(a1)[a2]
        y_ = y_.total.loc[:, LBL_T.S]

        fig.add_trace(go.Scatter(
            x = self._bins, 
            y = y_.values,
            name = f'E(T | {LBL.N(a1)})',
            line = dict(
                color = "rgba(40, 80, 200, 1.0)",
                width = 2,
            ),
            zorder  = 4,
            visible = is_visible,
            hovertemplate = '<br>E(T | Bin %{x}): %{y:.1%}',
        ))
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _dropdown_menu(self) -> list[dict]:
        titles  = self._get_titles()        

        l = len(self.agents_comb)
        b = len(self._bins) + 1
        r = lambda x: int(x*b)

        dropdown_buttons = []

        for i, agents in enumerate(self.agents_comb):
            visibility_mask = [False] * r(l)
            visibility_mask[r(i):r(i+1)] = [True]*b # Turn on ONLY this specific trace index
            
            dropdown_buttons.append(dict(
                label  = agents,
                method = "update",
                args  = [
                    {"visible": visibility_mask},
                    {"xaxis.title.text": titles[agents]}
            ]))
        return dropdown_buttons

    def _layout(self, 
        fig: go.Figure, 
        vis: str|int, 
        is_subplot: bool = False, 
    ) -> None:
        
        font  = dict(family="Arial", size=11, color="black")
        title = lambda t: dict(text=t)#, font=font)
        grid  = lambda w=0.6: dict(
                gridwidth = w, 
                gridcolor = f'rgba(255, 255, 255, {w})',
        )

        ticks = np.linspace(0, 1, 11)
        axis  = lambda t: dict(title=title(t)) | grid()
        
        agents = self.agents_comb[vis] if isinstance(vis, int) else vis
        a  = lambda a: f'{a}axis' if not is_subplot else f'{a}axis2'
        xy = {
            a('x'): axis(self._xaxis_title(agents)) | dict(tickvals = self._bins),
            a('y'): axis('E(Target)') | dict(tickvals = ticks)
        }

        fig.update_layout(
            **xy,
            height = 550,
            width  = 1050,            
            margin = dict(l=0, r=0, t=0, b=0),
            showlegend = False
        )
    # ─────────────────────────────────────────────────────────────────────────────────────────────

    def _graph(self, agents:str, fig:go.Figure=None, is_visible:bool=True) -> go.Figure | None:
        fig_ = fig or go.Figure()

        a1, a2 = LBL.R(agents)
        args = (fig_, a1, a2, is_visible)
        
        self._bins_dst(*args)
        self._bins_total(*args)
        
        if fig is not None:
            return
        
        self._layout(fig_, agents)
        return fig_
    
    def _create(self, vis:int=0) -> go.Figure:
        fig = go.Figure()

        for i, agents in enumerate(self.agents_comb):
            self._graph(
                agents,
                fig = fig, 
                is_visible = (i==vis)
            )
        return fig
            
    def create(self, show:bool=True) -> go.Figure | None:
        """
        Create and display the target evolution plot.

        Parameters
        ----------
        show : bool, default=True
            If True, display. If False, return the Figure.

        Returns
        -------
        go.Figure | None
        """        
        vis = 0
        fig = self._create(vis)        
        self._layout(fig, vis)
        
        fig.update_layout(
            **self._dropdown_layout( self._dropdown_menu() )
        )

        if not show:
            return fig
        fig.show()
    
    def _get_raw_components(self, vis:int=0) -> list:
        return list( self._create(vis).data )        
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class TransitionPlot(TransitionPlotBase):
    """
    Combined Transition Visualization — Matrix + Target Evolution.

    Facilitates access to the three plot variants:
    - `matrix()`: Transition matrix heatmap.
    - `evolution()`: Target evolution scatter + line.
    - `dash()`: Combined view (matrix on top, evolution below)
      with a unified dropdown for agent pair switching.

    Parameters
    ----------
    stats : TransitionStats
        Populated transition stats (from `TransitionProcessor.process`).

    Examples
    --------
    >>> from aidflow.models import TransitionPlot
    >>> plot = TransitionPlot(tr.fitter.stats)
    >>> plot.matrix()     # heatmap  # doctest: +SKIP
    >>> plot.evolution()  # scatter  # doctest: +SKIP
    >>> plot.dash()       # combined  # doctest: +SKIP
    """
    def __init__(self, stats: TransitionStats):
        super().__init__(stats)

        self._matrix  = TransitionMatrix(stats)
        self._evolute = TargetEvolutionPlot(stats)

    def matrix(self, show:bool=True) -> go.Figure | None:
        """
        Display the transition matrix heatmap.

        Parameters
        ----------
        show : bool, default=True
            If True, display. If False, return the Figure.
        """        
        return self._matrix.create(show)

    def evolution(self, show:bool=True) -> go.Figure | None:
        """
        Display the target evolution scatter plot.

        Parameters
        ----------
        show : bool, default=True
            If True, display. If False, return the Figure.
        """        
        return self._evolute.create(show)
    
    def dash(self, show:bool=True) -> go.Figure | None:
        """
        Display the combined view (matrix + evolution in one figure).

        The matrix is on top, the evolution plot below, with a
        unified dropdown menu for switching between agent pairs.

        Parameters
        ----------
        show : bool, default=True
            If True, display. If False, return the Figure.

        Returns
        -------
        go.Figure | None
        """        
        vis = 0
        m_traces, m_anno, m_titles = self._matrix._get_raw_components(vis)
        e_traces = self._evolute._get_raw_components(vis)
        
        size_m  = self._size()
        rh = 450/size_m[0]
        
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.02, row_heights=[1-rh, rh])

        for trace in m_traces:
            fig.add_trace(trace, row=1, col=1)

        for trace in e_traces:
            fig.add_trace(trace, row=2, col=1)

        self._matrix._layout(fig, m_anno, m_titles, vis)
        self._evolute._layout(fig, vis, is_subplot = True)

        buttons = self._dropdown_menu(m_anno, m_titles)

        fig.update_layout(
            height = size_m[0],
            width  = size_m[1],
            margin = dict(t=0, b=0, l=size_m[2], r=0),
            **self._dropdown_layout(buttons, 1.06)                      
        )

        if not show:
            return fig
        fig.show()
    # ─────────────────────────────────────────────────────────────────────────────────────────────
    
    def _size(self) -> tuple[int, int, int]:
        l = len(self._evolute._bins)
        h  = 1100
        w  = 850
        h  = np.clip(h*l / 10.5, h, 1600)
        w  = np.clip(w*l / 11.5, w, 1050)
        m  = max(0, 1050 - w)
        return int(h), int(w), int(m)

    def _dropdown_menu(self, m_anno:dict, m_titles:dict) -> list[dict]:
        m_buttons = self._matrix._dropdown_menu(m_anno, m_titles)
        e_buttons = self._evolute._dropdown_menu()

        v = lambda x: x['args'][0]['visible']
        t = lambda x: x['args'][1]

        buttons = []

        for i, agents in enumerate(self.agents_comb):
            b = m_buttons[i]
            e = e_buttons[i]            
            
            buttons.append(dict(
                label  = agents,
                method = "update",
                args = [
                    {"visible": v(b) + v(e)},
                    {**t(b), "xaxis2.title.text": t(e)["xaxis.title.text"]}
                ]
            ))
        return buttons
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬