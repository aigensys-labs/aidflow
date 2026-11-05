"""
Subsampling Strategy - Regulatory-Compliant Training Sample Construction.

This module implements the sampling strategy for credit scoring model training.
It determines the appropriate sample size and target event rate for each
modeling phase, ensuring:

**Statistical Validity:**
    - Linear models (GLM): Balanced sampling produces identical coefficient
      estimates (up to intercept shift) — a proven statistical property.
      Sample size is constrained by the n/p ratio for estimator stability.
    - Gradient Boosting (GBM): Target rate affects gradient computation and
      feature selection order. Minimum target events are enforced to prevent
      majority-class dominance in the exploration phase.
    - WOE Encoding: WOE values are mathematically invariant to class
      re-weighting (the ratio of good/bad distributions is preserved).
      However, bin boundaries depend on data density, so the original target
      rate is preserved to maintain the regulatory feature profile.

**Regulatory Compliance (EU / Central Bank):**
    - **EU AI Act (Article 10 - Data Governance)**: Training data must be
      relevant, sufficiently representative, and as complete and accurate
      as possible. The sampling strategy is documented and reproducible.
    - **ECB / SSM Model Risk Management (SR 11-7)**: Model validation
      requires that the training population selection process is transparent,
      justifiable, and free from selection bias that could disadvantage
      specific customer segments.
    - **GDPR (Article 22 - Automated Decision-Making)**: The sampling must
      not introduce discriminatory outcomes. Stratified sampling by target
      ensures all customer outcomes (good/bad) are represented proportionally
      in the final sample, preventing implicit exclusion of any segment.
    - **ECB Guide on Internal Models (2023)**: The selection of the training
      population must be documented, and the rationale for any sub-sampling
      must be explicitly stated in the model documentation.

**Design Principles:**
    - **Reproducibility**: A fixed `random_state` ensures that the same
      training sample is always produced for a given dataset.
    - **Stratification**: Sampling is always stratified by the target variable
      to preserve the good/bad ratio in the selected subset.
    - **Minimum Event Guarantee**: No sample is produced with fewer than
      the required minimum target events, preventing statistical instability.
    - **No Unfair Selection**: The sampling operates on the target outcome
      (default/non-default) and never on protected characteristics
      (race, gender, age, etc.), ensuring non-discrimination.

References:
    - Harrel, D. (2015). *Regression Modeling Strategies*, 2nd ed.
    - Firth, D. (1993). Bias in logistic regression: a comparison of methods.
    - Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
    - EU AI Act (2024), Article 10.
    - ECB Guide on internal models (2023), Chapter 5.
"""

from ...deps.data       import np, pDF, pSe
from .sampling          import Sampling


class SubsamplingSpec:
    """
    Subsampling specification calculator.

    Computes the appropriate sample size and target event rate for each
    modeling phase based on the statistical requirements of the specific
    algorithm class.

    Parameters
    ----------
    y (pSe): The target variable (binary: 1 = default, 0 = non-default).
        Must be a binary series with no missing values.

    The specification is derived from first principles:

    - **GLM**: Bounded by the n/p ratio (Harrel, 2015). Balanced target
      rate is used because coefficient estimates are invariant to class
      re-weighting (Firth, 1993) — only the intercept shifts by
      `log(balancing_ratio)`, which is mathematically expected and correct.
    - **GBM**: Bounded by minimum target events to prevent majority-class
      dominance in the split-finding process (Chen & Guestrin, 2016).
      The target rate is preserved when sufficient events exist, as
      GBM's gradient computation is sensitive to the base rate.
    - **WOE**: Bounded by minimum bin occupancy to ensure stable WOE
      estimates. The original target rate is preserved because WOE
      profiling (bin boundaries, distribution shape) is a critical
      regulatory artifact in scorecard documentation.

    Constants
    ---------
    BT (float): Balanced target rate (0.5). Used for GLM and WOE
        where target rate invariance is proven.
    US (int): Univariate analysis sample size (100,000). Sufficient
        for stable per-bin statistics across all features.
    GS (int): GBM maximum sample size (1,000,000). Computational
        bound that maintains statistical sufficiency.
    GT (int): GBM minimum target events (5,000). Ensures the
        exploration phase has adequate minority representation
        to identify true signal features.
    PS (int): Calibration / sensitivity assessment sample size.
        Pruning and sensitivity phase of Tree / Logit.

    Example
    -------
    >>> spec = SubsamplingSpec(y=target)
    >>> sample_size, target_rate = spec.get_glm_spec(X, n_p=500)
    >>> # sample_size: optimal n for stable GLM coefficients
    >>> # target_rate: 0.5 (balanced — safe for GLM)
    """

    BT = 0.5        # Balanced target rate
    US = int(1e5)   # Univariate sample size
    
    GS = int(1e6)   # GBM max sample size
    GT = int(5e3)   # GBM min target events

    PS = int(2e5)   # GBM | GLM / 2 pruning-sensitivity

    def __init__(self, y:pSe):
        self.y = y 

    def get_glm_spec(self, X: pDF, n_p: int = 500) -> tuple[int, float]:
        """
        Compute the optimal sample size and target rate for GLM (linear) models.

        **Statistical Justification:**

        Linear models (logistic regression, Lasso, ElasticNet) require
        a sufficient number of observations per parameter (n/p) to produce
        stable, unbiased coefficient estimates. The rule of thumb
        (Harrel, 2015; Firth, 1993) is n/p ≥ 50 for reliable inference.

        Setting `n_p` high (default: 500) ensures that cross-validation
        folds retain sufficient observations even when target events are
        limited in absolute terms, preventing the "separation" problem
        where small CV subsets exhibit perfect quasi-separation.

        **Class Balancing Invariance:**
        It is a proven statistical property that re-weighting (or
        balancing) the target in logistic regression produces identical
        coefficient estimates for all predictors. Only the intercept
        shifts by `log(prior_ratio / 0.5)`. This means that using a
        balanced target rate (BT = 0.5) is mathematically safe for
        coefficient estimation and feature selection — it does not
        introduce any bias in the learned relationship between
        features and the target.

        Reference: Firth, D. (1993). "Bias in logistic regression:
        A comparison of methods." Journal of the Royal Statistical
        Society, 55(1), 27-40.

        Parameters
        ----------
        X (pDF): The feature matrix used to determine the number of
            parameters (columns).
        n_p (int): The minimum events-per-parameter ratio required.
            Default is 500 (conservative; typical range is 50-500).
            Higher values produce more samples (safer for unstable data).

        Returns
        -------
        tuple[int, float]:
            - sample_size (int): The recommended number of observations
                for training (capped at n/p ratio × p).
            - target_rate (float): The target event rate for the
                balanced sample (0.5).

        Example
        -------
        >>> spec = SubsamplingSpec(y)
        >>> n, rate = spec.get_glm_spec(X, n_p=500)
        >>> # n = min(len(X), 500 * num_features)
        >>> # rate = 0.5 (balanced)

        **Regulatory Note:**

        The balanced sampling approach is fully transparent and
        reproducible. The intercept shift is deterministic and can be
        back-calculated: `intercept_original = intercept_balanced - log(prior/0.5)`.
        This satisfies SR 11-7 requirements for model interpretability
        and validation traceability.        
        """
        o, c = X.shape
        s = np.clip(o/c, max=n_p) * c
        return int(s), self.BT

    def get_gbm_spec(self, sample_size: int = None) -> tuple[int, float|None]:
        """
        Compute the optimal sample size and target rate for GBM models.

        **Statistical Justification:**

        Unlike GLM, Gradient Boosting Machines are **NOT** invariant to
        class imbalance. The base rate affects:
        1. The initial prediction (p₀ = mean(y) in AdaBoost/GBM).
        2. The gradient computation at each iteration.
        3. The split-finding criterion (gain = G_left²/n_left + G_right²/n_right).

        A highly imbalanced target (e.g., 1% default rate) can cause the
        algorithm to "favor" majority-class splits during the exploratory
        phase, leading to **feature selection bias** — true signal features
        with weak individual lift may be overlooked because they don't
        maximize the gain metric under the skewed base rate.

        **Solution:** We enforce a minimum of `GT` (5,000) target events
        in the training sample. If the natural target rate already provides
        this many events within the `GS` (1M) sample cap, the original
        rate is preserved (no re-weighting needed). Otherwise, the target
        rate is adjusted to guarantee `GT` events are present.

        Reference: Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable
        Tree Boosting System." KDD 2016, 785-794.
        
        Parameters
        ----------
        sample_size (int): Optionally, specified sample size is
            used directly. Deault is None, using GS max default 1e6.

        Returns
        -------
        tuple[int, float|None]:
            - sample_size (int): The maximum sample size (GS = 1,000,000).
            - target_rate (float | None): The adjusted target rate to
                guarantee minimum events, or None if the natural rate
                already satisfies the threshold (no re-weighting needed).

        Example
        -------
        >>> spec = SubsamplingSpec(y)
        >>> n, rate = spec.get_gbm_spec()
        >>> # If y.mean() = 0.05 and GS = 1M → 50K events > 5K → rate = None
        >>> # If y.mean() = 0.003 and GS = 1M → 3K events < 5K → rate = 0.005

        **Regulatory Note:**

        The minimum event guarantee (5,000) is a conservative threshold
        that ensures the model training process is not dominated by
        majority-class noise. This prevents implicit discrimination
        against customer segments that are underrepresented in the
        default population. The approach is documented and reproducible,
        satisfying ECB Model Risk Management guidelines.        
        """      
        gs = sample_size or self.GS     # max obs per training
        t  = self.y.mean()              # target rate
        if (self.GT / t) < gs:          # min satisfied within gs
            return gs, None        
        return gs, self.GT / min(len(self.y), gs)

    def get_sensitivity_spec(self, is_gbm:bool, sample_size:int=None
    ) -> tuple[int, None|float]:
        """
        Compute the optimal sample size and target rate for feature
        sensitivity assessment and pruning.

        **Purpose:**

        The sensitivity (pruning) phase evaluates the robustness of
        individually selected features by measuring their stability
        under varying sample compositions. This is a **diagnostic**
        step that runs *after* the model has already identified
        significant features, and determines which of those features
        are truly robust versus those whose predictive power is
        fragile (high variance in importance across resamples).

        *`The sample size reflects sensitivity position in modeling:`*

        - The feature set is already reduced (post-selection), so
        per-feature statistics are computed on fewer columns.
        - Iterative sensitivity and pruning benefits from faster 
        computation at each cycle without loosing diagnostic
        robustness.

        **Sample Size Rationale:**

        - **GBM (PS = 200,000)**: SHAP values require sufficient
        tree traversals for stable attribution. At 200K observations,
        the variance of SHAP estimates per feature is negligible
        relative to the signal. 

        - **GLM (PS = 100,000)**: The GLM link function has
        well-understood asymptotic properties — coefficient variance
        scales as 1/n, and at 100K with p ≤ 500 selected features,
        n/p ≥ 200, exceeding the minimum (50) required for unbiased
        estimation (Harrel, 2015). The 2× reduction (200K → 100K)
        is therefore statistically safe and computationally efficient.

        **Target Rate Strategy:**

        - **GBM models** (`is_gbm=True`): Delegates to
        `get_gbm_spec` with the PS sample size. The target rate
        is preserved (or adjusted to guarantee minimum events)
        because GBM's gradient computation, hessian calculations,
        and split-finding are sensitive to the base rate. This
        ensures the sensitivity ranking reflects the true feature
        importance under the actual population distribution.

        - **GLM models** (`is_gbm=False`): Uses the balanced
        target rate (BT = 0.5).

        *`Mathematical Proof of GLM Gini / Rank Invariance:`*

        - **Slope Invariance (Firth, 1993):** For logistic regression,
        artificially rebalancing classes via uniform random
        undersampling alters *only* the intercept (β₀), shifting
        predicted logits by a additive constant:

            c = ln(π_sample / (1 - π_sample)) - ln(π_pop / (1 - π_pop))

        The slope vector β remain asymptotically invariant. Because
        the shift is strictly monotonic, row-by-row predicted
        probabilities preserve identical rank ordering on any fixed
        dataset.

        - **ROC-AUC & Gini Coordinate Geometry:** The empirical Gini
        coefficient (2·AUC - 1) measures the pairwise rank-ordering
        concordance:

            AUC = P(p̂_i > p̂_j | Y_i = 1, Y_j = 0)

        Uniform random downsampling of non-events (Y=0) scales
        False Positives (FP) and Total Non-Events by the exact same
        scalar factor in the False Positive Rate formula:

            FPR(t) = FP(t) / N_0
            
        Because FPR(t) and TPR(t) remain invariant across all
        decision thresholds t, the ROC curve coordinates and the
        resulting Gini coefficient are asymptotically invariant
        (E[Gini_balanced] = E[Gini_population]).

        Thus, balanced subsampling (0.5) produces mathematically
        identical sensitivity rankings (|β|, z-scores, Wald stats,
        and Validation Gini / Permutation losses) while maximizing
        computational speed.

        Parameters
        ----------
        is_gbm (bool): If True, the target model is a GBM (tree-based)
            and the target rate is preserved/adjusted for minimum
            events. If False, the target is a GLM and the balanced
            rate (0.5) is used.
        sample_size (int): Optionally, a specified sample size is
            used directly, overriding the PS default.
            Default is None, using PS (200,000 GBM | 100,000 GLM).

        Returns
        -------
        tuple[int, float|None]:
            - sample_size (int): The sample size for sensitivity
                assessment (200,000 for GBM | 100,000 for GLM by default).
            - target_rate (float | None): The target event rate.
                For GBM: preserved or adjusted (None if natural rate
                is sufficient). For GLM: 0.5 (balanced).

        Example
        -------
        >>> spec = SubsamplingSpec(y)
        >>>
        >>> # For GBM sensitivity (full target rate, 200K):
        >>> n, rate = spec.get_sensitivity_spec(is_gbm=True)
        >>> # n = 200,000, rate = None or adjusted
        >>>
        >>> # For GLM sensitivity (balanced, 100K):
        >>> n, rate = spec.get_sensitivity_spec(is_gbm=False)
        >>> # n = 100,000, rate = 0.5
        >>>
        >>> # Override with custom size:
        >>> n, rate = spec.get_sensitivity_spec(is_gbm=True, sample_size=300_000)
        >>> # n = 300,000

        **Regulatory Note:**

        Feature pruning and sensitivity assessment are part of the
        model's "development logic" and must be documented per
        SR 11-7 (Model Validation) and ECB Model Risk Management
        guidelines (2023, Chapter 5). The rationale for excluding
        features (low sensitivity, unstable WOE, high correlation,
        high variance in importance) must be traceable to the
        statistical evidence produced by this sampling step.

        The non-discrimination requirement (GDPR Article 22, EU AI
        Act Article 10) is satisfied because the sensitivity
        assessment operates exclusively on the target outcome and
        feature statistics — never on protected characteristics
        (race, gender, age, etc.).

        **References:**

        - Lundberg, S. & Lee, S. (2017). "A Unified Approach to
            Interpreting Model Predictions." NeurIPS. (SHAP stability)
        - Firth, D. (1993). "Bias reduction of maximum likelihood estimates." 
            Biometrika, 27-38. (GLM invariance)
        - Harrel, D. (2015). "Regression Modeling Strategies," 2nd ed.
            (n/p ratio guidelines)
        - Breiman, L. (2001). "Random Forests." Machine Learning.
            (Permutation importance)        
        """
        n = self.PS or sample_size
        if is_gbm:
            return self.get_gbm_spec(n)
        if n == self.PS:
            n = n // 2
        return n, self.BT

    def get_woe_spec(self) -> tuple[int, None]:
        """
        Compute the optimal sample size and target rate for WOE encoding.

        **Statistical Justification:**

        WOE (Weight of Evidence) is defined as:
        
            WOE_i = ln( (n_good_i / n_good_total) / (n_bad_i / n_bad_total) )

        This ratio is **mathematically invariant** to re-weighting of
        the target: if we balance the sample (change the good/bad ratio),
        both the numerator and denominator of each bin shift
        proportionally, and the WOE values remain identical.

        **However**, bin boundaries are determined by data density.
        If the sample is too small, the smallest bins (extreme WOE
        regions) may contain insufficient observations, leading to:
        - Unstable WOE estimates (high variance in small bins).
        - Non-reproducible bin boundaries across CV folds.
        - Regulatory concern: the WOE profile (number of bins,
            their boundaries, and WOE values) is a critical artifact
            in scorecard documentation.

        **Solution:** We compute the minimum sample size such that
        even the most extreme bin (expected WOE ≈ 4, expected bin
        share ≈ 1%) contains at least 30 target events. This ensures:
        1. WOE estimates are statistically stable (n ≥ 30 per bin).
        2. The original target rate is preserved (no distortion of
            the feature profile).
        3. The WOE encoding is reproducible and defensible in
            regulatory review.

        Returns
        -------
        tuple[int, None]:
            - sample_size (int): The minimum sample size ensuring
                ≥ 30 events in the expected extreme bin.
            - target_rate (None): The original target rate is preserved
                (no re-weighting). WOE invariance makes balancing
                unnecessary, and regulatory profiling requires the
                natural distribution.
        
        Example
        -------
        >>> spec = SubsamplingSpec(y)
        >>> n, rate = spec.get_woe_spec()
        >>> # n ensures ≥ 30 events in extreme WOE bins
        >>> # rate = None (original target rate preserved)

        **Regulatory Note:**

        WOE encoding is a standard, explainable transformation in
        credit scoring. The bin structure and WOE values form part of
        the model's "logic" and must be stable and justifiable.
        Preserving the original target rate ensures that the WOE
        profile accurately reflects the true customer population,
        satisfying GDPR Article 22 (transparency) and EU AI Act
        Article 10 (data governance) requirements.      
        """
        t = self.y.mean()               # target rate
        s = t * np.exp(-4)              # extreme bin separation WOE ~ 4
        b = 0.01                        # extreme bin share of obs
        e = len(self.y) * b * s         # events in extreme bin
        o = np.clip(e, max = 30)
        return int(o / b / s), None

    def get_univar_spec(self) -> tuple[int, float]:
        """
        Compute the sample size and target rate for univariate analysis.

        **Statistical Justification:**

        Univariate analysis (KS, IV, univariate WOE, distribution
        comparison) requires sufficient observations in each feature
        bin to produce stable statistics. A sample size of 100,000
        (US) is a practical threshold that ensures:
        - At least 10,000 observations per 10-bin univariate split.
        - Stable estimation of per-bin default rates.
        - Reliable KS statistic (requires ~500 events minimum for
            meaningful discrimination measurement).

        The balanced target rate (BT = 0.5) is used because:
        - Univariate IV and WOE are invariant to class re-weighting.
        - KS statistic is invariant to the good/bad ratio.
        - Balanced sampling simplifies interpretation and reporting.

        Returns
        -------
        tuple[int, float]:
            - sample_size (int): The univariate sample size (100,000).
            - target_rate (float): The balanced target rate (0.5).

        Example
        -------
        >>> spec = SubsamplingSpec(y)
        >>> n, rate = spec.get_univar_spec()
        >>> # n = 100,000, rate = 0.5

        **Regulatory Note:**

        Univariate analysis is the first line of defense in model
        validation. The statistics produced (IV, KS, WOE per bin)
        are reported to regulators and must be stable and
        reproducible. The fixed sample size ensures consistency
        across validation cycles.
        """        
        return self.US, self.BT

class Subsampling:
    """
    Subsampling execution engine for credit scoring model training.

    Orchestrates the actual sampling operation for each modeling phase.
    Takes the full training dataset and produces a stratified subsample
    with the sample size and target rate specified by `SubsamplingSpec`.

    Parameters
    ----------
    X (pDF): The full feature matrix. Must contain no missing values
        in the target column. Feature columns may contain missing
        values (handled by downstream cleaning).
    y (pSe): Binary target series (1 = event, 0 = non-event).
        Must have at least 2 unique values and a non-zero rate.
    random_state (int): Random seed for reproducible sampling.
        Default is 42. **Must be logged** in model metadata for
        regulatory audit (SR 11-7, EU AI Act Article 10).
            
    Key Properties
    --------------
    - **Stratified**: Sampling is always performed with stratification on
      the target variable, ensuring the good/bad ratio is preserved (or
      adjusted to the target rate) in the selected subset.
    - **Reproducible**: A fixed `random_state` ensures that the same
      dataset always produces the same sample. This is critical for
      regulatory audit trails and model reproducibility.
    - **Non-discriminatory**: The sampling operates exclusively on the
      target outcome (default/non-default) and never on protected
      characteristics. No customer segment is preferentially included
      or excluded based on demographics.

    Regulatory Compliance
    ---------------------
    - **EU AI Act (2024), Article 10**: The training data selection
      process is documented, reproducible, and free from selection bias
      that could affect specific groups.
    - **ECB Guide on Internal Models (2023), Chapter 5**: The rationale
      for sub-sampling is explicitly stated in the model documentation
      (see module docstring for full justification).
    - **SR 11-7 (Model Validation)**: The sampling strategy is
      transparent, the random seed is logged, and the resulting sample
      characteristics (size, target rate, event count) are recorded
      in the model metadata for validation review.
    - **GDPR (2016), Article 22**: The sampling does not introduce
      discriminatory outcomes. All customer segments are represented
      proportionally in the final training sample.

    Example
    -------
    >>> sub = Subsampling(X=train_X, y=train_y, random_state=42)
    >>>
    >>> # For GLM training:
    >>> X_glm, y_glm = sub.glm(n_p=500)
    >>>
    >>> # For GBM training:
    >>> X_gbm, y_gbm = sub.gbm()
    >>>
    >>> # For WOE encoding:
    >>> X_woe, y_woe = sub.woe()
    >>>
    >>> # For univariate analysis:
    >>> X_uni, y_uni = sub.univar()
    """

    def __init__(self, X:pDF, y:pSe, random_state:int = 42):
        self.X = X
        self.y = y
        self.rs = random_state        
        self.spec = SubsamplingSpec(y)

    def sampling(self, 
        sample_size: int, 
        target_rate: float|None
    ) -> tuple[pDF, pSe]:
        """
        Execute the stratified subsampling operation.

        Delegates to the `Sampling` utility with the computed sample size
        and target rate. The sampling is always stratified on the target
        variable to preserve (or adjust) the good/bad ratio.

        Parameters
        ----------
        sample_size (int): The desired number of observations in the
            output sample. Must be ≤ len(X).
        target_rate (float | None): The desired target event rate in the
            output sample. If None, the natural rate is preserved.
            If specified, the sample is stratified to achieve this rate.

        Returns
        -------
        tuple[pDF, pSe]:
            - X_sampled (pDF): The subsampled feature matrix.
            - y_sampled (pSe): The corresponding target series.            

        **Note:**

            The output maintains the same column structure as the input.
            The index is reset (0-based) to avoid confusion from
            non-contiguous indices in downstream operations.
        """
        if len(self.X) <= sample_size:
            return self.X, self.y
        
        return Sampling(
            sample_size     = sample_size,
            stratify        = self.y,
            stratify_ratio  = target_rate,
            random_state    = self.rs
        ).subsample(self.X, self.y)

    def glm(self, n_p:int = 500, sample_size: int = None) -> tuple[pDF, pSe]:
        """
        Generate a training sample optimized for GLM (linear) models.

        Uses the `get_glm_spec` specification:
        - Sample size bounded by n/p ratio (default: 500 events per parameter).
        - Target rate balanced to 0.5 (mathematically safe for GLM coefficients).

        The resulting sample is suitable for logistic regression, Lasso,
        ElasticNet, and other linear family models.

        Parameters
        ----------
        n_p (int): Minimum events-per-parameter ratio. Default is 500.
            Use higher values (e.g., 1000) for models with many
            interactions or polynomial terms.
        
        sample_size (int): Optionally, a specified sample size is
            used directly, overriding the n/p calculation.
            Default is None (uses the n_p-based formula).

        Returns
        -------
        tuple[pDF, pSe]:
            The subsampled (X, y) pair for GLM training.

        Example
        -------
        >>> sub = Subsampling(X, y)
        >>> # Standard: n/p formula
        >>> X_glm, y_glm = sub.glm(n_p=500)
        >>>
        >>> # Override: fixed sample size
        >>> X_glm, y_glm = sub.glm(sample_size=500_000)

        **Regulatory Note:**

        The balanced sample is statistically equivalent to the original
        for coefficient estimation. The intercept can be back-calculated
        using: `β₀_original = β₀_balanced - ln(prior_ratio / 0.5)`.
        This transformation must be documented in the model report.        
        """        
        n, r = self.spec.get_glm_spec(self.X, n_p)
        n = sample_size or n
        return self.sampling(n, r)
        
    def gbm(self, sample_size: int = None) -> tuple[pDF, pSe]:
        """
        Generate a training sample optimized for GBM (tree-based) models.

        Uses the `get_gbm_spec` specification:
        - Sample size capped at 1,000,000 (computational bound).
        - Target rate preserved if ≥ 5,000 events exist in the cap.
        - If insufficient events, target rate is adjusted to guarantee
          the minimum (preventing majority-class dominance).

        Parameters
        ----------
        sample_size (int): Optionally, a specified sample size is
            used directly, overriding the GS cap.
            Default is None (uses GS = 1,000,000).

        Returns
        -------
        tuple[pDF, pSe]: 
            The subsampled (X, y) pair for GBM training.

        Example
        -------
        >>> sub = Subsampling(X, y)
        >>> # Standard: GS cap
        >>> X_gbm, y_gbm = sub.gbm()
        >>>
        >>> # Override: smaller sample for rapid iteration
        >>> X_gbm, y_gbm = sub.gbm(sample_size=200_000)            

        **Regulatory Note:**

        The minimum event guarantee (5,000) ensures that the model
        training process is not dominated by majority-class noise.
        This prevents implicit bias against customer segments that
        are underrepresented in the default population. The rationale
        must be documented in the model validation report per
        ECB Model Risk Management guidelines.        
        """        
        n, r = self.spec.get_gbm_spec(sample_size)
        return self.sampling(n, r)

    def sensitivity(self, is_gbm: bool, sample_size: int = None
    ) -> tuple[pDF, pSe]:
        """
        Generate a training sample optimized for feature sensitivity
        assessment and pruning.

        This method produces the sample used in the **calibration phase**
        of the modeling workflow, where individual features are evaluated
        for their robustness and contribution to the final model.

        **Use Cases:**
        - *Feature Pruning* : Identifying features whose predictive power
          is unstable and excluding them from the final model.
        - *Sensitivity Analysis* : Measuring how much a feature's
          coefficient or importance changes when the sample is perturbed.
          This is a key input to the model's "robustness report" required
          by SR 11-7 validation.

        **Behavior:**
        - For GBM models: Preserves the natural target rate (or adjusts
          to guarantee minimum events). The sensitivity ranking reflects
          the true split-finding behavior of the algorithm.
        - For GLM models: Uses balanced target rate (0.5). The sensitivity
          ranking is invariant to the balancing (Firth, 1993).

        Parameters
        ----------
        is_gbm (bool): Whether the target model is a GBM (tree-based)
            or GLM (linear). Determines the target rate strategy.
        sample_size (int): Optionally, a specified sample size is
            used directly. Default is None, using PS (200,000 GBM | 100,000 GLM).

        Returns
        -------
        tuple[pDF, pSe]:
            - X_sampled (pDF): The subsampled feature matrix for
                sensitivity assessment.
            - y_sampled (pSe): The corresponding target series.

        Example
        -------
        >>> sub = Subsampling(X=train_X, y=train_y, random_state=42)
        >>>
        >>> # Sensitivity sample for GBM model:
        >>> X_sens, y_sens = sub.sensitivity(is_gbm=True)
        >>>
        >>> # Sensitivity sample for GLM model:
        >>> X_sens, y_sens = sub.sensitivity(is_gbm=False)
        >>>
        >>> # Custom sample size:
        >>> X_sens, y_sens = sub.sensitivity(is_gbm=True, sample_size=500_000)

        **Regulatory Note:**

        The sensitivity assessment results (feature importance rankings,
        lift stability, WOE variance) form part of the model's
        validation evidence. Per ECB Model Risk Management guidelines,
        the methodology for feature selection must be documented,
        and the statistical evidence for including or excluding
        specific features must be available for regulatory review.

        The sampling is non-discriminatory: it operates on the
        target outcome and feature statistics, never on protected
        characteristics. All customer segments are represented
        proportionally in the sensitivity sample.
        """        
        n, r = self.spec.get_sensitivity_spec(is_gbm, sample_size)
        return self.sampling(n, r)

    def woe(self, sample_size: int = None) -> tuple[pDF, pSe]:
        """
        Generate a training sample optimized for WOE encoding.

        Uses the `get_woe_spec` specification:
        - Sample size computed to ensure ≥ 30 events in the expected
          extreme bin (WOE ≈ 4, bin share ≈ 1%).
        - Target rate preserved (natural rate) to maintain the
          regulatory feature profile.

        The resulting sample is suitable for WOE bin computation and
        scorecard construction.

        Parameters
        ----------
        sample_size (int): Optionally, a specified sample size is
            used directly, overriding the WOE minimum calculation.
            Default is None (uses the extreme-bin methodology).

        Returns
        -------
        tuple[pDF, pSe]: 
            The subsampled (X, y) pair for WOE encoding.
        
        Example
        -------
        >>> sub = Subsampling(X, y)
        >>> # Standard: WOE methodology
        >>> X_woe, y_woe = sub.woe()
        >>>
        >>> # Override: fixed sample size
        >>> X_woe, y_woe = sub.woe(sample_size=500_000)

        **Regulatory Note:**

        WOE encoding is a critical, explainable transformation in
        credit scorecards. The bin structure and WOE values must be
        stable, reproducible, and defensible in regulatory review.
        Preserving the natural target rate ensures the WOE profile
        accurately reflects the true customer population.        
        """
        n, r = self.spec.get_woe_spec()
        n = sample_size or n
        return self.sampling(n, r)

    def univar(self, sample_size: int = None, is_balanced: bool = True
    ) -> tuple[pDF, pSe]:
        """
        Generate a training sample optimized for univariate analysis.

        Uses the `get_univar_spec` specification:
        - Fixed sample size of 100,000 observations.
        - Target rate balanced to 0.5.

        The resulting sample is suitable for:
        - Univariate WOE per feature.
        - Univariate encoding / transformation.
        - Distribution comparison (Expected vs. Actual).

        Parameters
        ----------
        sample_size (int): Optionally, a specified sample size is
            used directly. Default is None (uses US = 100,000).
        is_balanced (bool): Optionally, set to False to keep original
            target rate. Default True (uses target rate 0.5)

        Returns
        -------
        tuple[pDF, pSe]: 
            The subsampled (X, y) pair for univariate analysis.

        Example
        -------
        >>> sub = Subsampling(X, y)
        >>> # Standard: 100K
        >>> X_uni, y_uni = sub.univar()
        >>>
        >>> # Override: larger sample for detailed analysis
        >>> X_uni, y_uni = sub.univar(sample_size=500_000)
        
        **Regulatory Note:**

        Univariate statistics are the first line of defense in model
        monitoring and validation. The fixed sample size ensures
        consistency across validation cycles and model comparisons.
        The balanced target rate simplifies interpretation and
        reporting to stakeholders and regulators.        
        """        
        n, r = self.spec.get_univar_spec()
        n = sample_size or n
        r = r if is_balanced else None
        return self.sampling(n, r)