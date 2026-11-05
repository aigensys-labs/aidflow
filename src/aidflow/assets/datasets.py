"""
Synthetic test datasets for pipeline validation.

Generates reproducible binary classification datasets with
mixed feature types (numeric + categorical) to exercise the
full AIDflow encoding pipeline — from feature cleaning
through WOE encoding and agent training.

All data is generated locally (no network, no licensing
concerns). Fixed seeds ensure identical data across runs
and machines.

Presets
-------
- **small**:   10k rows,  60 features. Quick smoke test.
- **medium**: 100k rows, 150 features. Standard pipeline run.
- **large**: 1000k rows, 400 features. Heavy-duty stress test.

Usage
-----
>>> from aidflow.assets import DataGetter
>>> data = DataGetter.get('medium')
>>> data.shape  # (100000, 152)  ← 150 features + target + time
>>> data.target.value_counts()
"""

from ..deps.common   import Path
from ..deps.data     import np, pd, pDF
from ..deps.ml.stats import ArmaProcess


_PRESETS = {
    'small':  dict(
        n_samples       = 10_000,
        n_numeric       = 30,
        n_categorical   = 30,
        n_informative   = 8,
        n_redundant     = 8,
        n_repeated      = 4,
        n_categories    = 4,
        p_target        = 0.30,
        seed            = 42,
    ),
    'medium': dict(
        n_samples       = 100_000,
        n_numeric       = 75,
        n_categorical   = 75,
        n_informative   = 20,
        n_redundant     = 20,
        n_repeated      = 10,
        n_categories    = 5,
        p_target        = 0.25,
        seed            = 42,
    ),
    'large':  dict(
        n_samples       = 1_000_000,
        n_numeric       = 200,
        n_categorical   = 200,
        n_informative   = 50,
        n_redundant     = 40,
        n_repeated      = 20,
        n_categories    = 6,
        p_target        = 0.20,
        seed            = 42,
    ),
}

_CAT_LABELS = {
    1: ['A'],
    2: ['A', 'B'],
    3: ['A', 'B', 'C'],
    4: ['A', 'B', 'C', 'D'],
    5: ['A', 'B', 'C', 'D', 'E'],
    6: ['A', 'B', 'C', 'D', 'E', 'F'],
}


class DataGetter:
    """
    Synthetic dataset generator for pipeline testing.

    Produces reproducible binary classification data with
    mixed feature types. All data is generated in-memory —
    no downloads, no external dependencies beyond sklearn.

    Attributes
    ----------
    Presets are defined in `_PRESETS`. Each preset specifies:
    sample count, feature counts (numeric/categorical),
    informative/redundant split, category cardinality,
    and base target rate.
    """

    @classmethod
    def get(cls, preset: str = 'medium', target: str = 'target') -> pDF:
        """
        Generate a synthetic dataset from a named preset.

        Parameters
        ----------
        preset : str, default='medium'
            Dataset size: `'small'`, `'medium'`, or `'large'`.
        target : str, default='target'
            Column name for the binary target.

        Returns
        -------
        pDF
            DataFrame with numeric features (`num_0` … `num_N`),
            categorical features (`cat_0` … `cat_M`),
            a time column (`time`), and the binary target.

        Raises
        ------
        ValueError
            If the preset name is not recognized.
        """
        if preset not in _PRESETS:
            raise ValueError(
                f'AIG SAFETY ► Unknown preset "{preset}". '
                f'Available: {list(_PRESETS.keys())}'
            )
        cfg = _PRESETS[preset]
        return cls._generate(cfg, target)

    @classmethod
    def presets(cls) -> dict[str, dict]:
        """
        Return the available preset configurations.

        Returns
        -------
        dict[str, dict]
            Mapping of preset name → configuration dict.
        """
        return dict(_PRESETS)

    @classmethod
    def create_data(cls, 
        size: str = 'small', 
        path: str|Path = 'data/test.csv'
    ) -> str:
        """
        Generate synthetic demo data and save to disk.

        Parameters
        ----------
        size : 'small' | 'medium' | 'large', default='small'
            Dataset scale.
        path : str | Path, default='data/test.csv'
            Disk save path.

        Returns
        -------
        str
            File path to the generated CSV.
        """    	
        demo_data = DataGetter.get(size)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        demo_data.to_csv(path, index=False)
        return str(path)
    
    @staticmethod
    def generate_data(
        n_samples: int = 10000,
        n_features: int = 50,
        p_positive: float = 0.15,
        n_informative: int | None = None,   # features carrying true signal
        n_redundant: int | None = None,     # linear combos of informative
        n_repeated: int | None = None,      # exact duplicates of informative
        corr_strength: float = 0.3,
        linear_weight: float = 1.0,
        frac_smooth: float = 0.25,
        smooth_amp: tuple[float, float] = (0.2, 0.4),
        frac_ixn: float = 0.15,
        ixn_amp: tuple[float, float] = (0.05, 0.25),
        feature_noise: float = 0.20,
        label_noise: float = 0.01,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic classification data with realistic feature structure.

        Feature composition (mirrors make_classification + realistic imperfections):
        - informative:  Carry true signal (linear + smooth + interactions)
        - redundant: Linear combinations of informative features
        - repeated: Exact duplicates of informative features
        - noise: Pure Gaussian, no relationship to target
  
        Parameters
        ----------
        n_samples : int
            Number of samples.
        n_features : int
            Total number of features.
        p_positive : float
            Target positive base rate before label noise (e.g., 0.05 for credit).
        n_informative : int | None
            Number of features carrying true signal.
            Default: max(5, p // 10).
        n_redundant : int | None
            Number of features that are linear combos of informative ones.
            Default: max(3, p // 20).
        n_repeated : int | None
            Number of features that are exact copies of informative ones.
            Default: max(2, p // 25).
        corr_strength : float
            Background correlation applied to noise features.
        linear_weight : float
            Std-dev of true coefficients on informative features.
        frac_smooth : float
            Fraction of *informative* features getting a tanh term.
        smooth_amp : tuple[float, float]
            Amplitude range for smooth terms.
        frac_ixn : float
            Fraction of *informative* feature-pairs getting an interaction.
        ixn_amp : tuple[float, float]
            Amplitude range for interaction terms.
        feature_noise : float
            Gaussian noise added to all features (EIV).
        label_noise : float
            Fraction of labels randomly flipped.
        seed : int
            Random seed.

        Returns
        -------
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,)

        Examples
        --------
        >>> X, y = make_realistic(n=5000, p=50, n_informative=8, n_redundant=5)
        >>> # Only 8 features truly matter; 5 are derivatives; 37 are noise
        """
        rng = np.random.default_rng(seed)

        # ── Resolve defaults
        if n_informative is None:
            n_informative = max(5, n_features // 10)
        if n_redundant is None:
            n_redundant = max(3, n_features // 20)
        if n_repeated is None:
            n_repeated = max(2, n_features // 25)

        n_noise = n_features - n_informative - n_redundant - n_repeated
        assert n_noise >= 0, (
            f"p={n_features} too small for informative={n_informative}, "
            f"redundant={n_redundant}, repeated={n_repeated}"
        )

        # ── 1. Generate informative features (correlated with each other)
        A_inf = rng.normal(size=(n_informative, n_informative)) * corr_strength
        X_inf = rng.normal(size=(n_samples, n_informative)) @ A_inf.T
        X_inf = (X_inf - X_inf.mean(0)) / (X_inf.std(0) + 1e-8)

        # ── 2. Redundant = linear combos of informative
        if n_redundant > 0:
            B = rng.normal(size=(n_informative, n_redundant)) * 0.5
            X_red = X_inf @ B + rng.normal(size=(n_samples, n_redundant)) * 0.1
        else:
            X_red = np.empty((n_samples, 0))

        # ── 3. Repeated = exact copies
        if n_repeated > 0:
            idx = rng.choice(n_informative, size=n_repeated, replace=False)
            X_rep = X_inf[:, idx]
        else:
            X_rep = np.empty((n_samples, 0))

        # ── 4. Noise features (independent, no signal)
        if n_noise > 0:
            X_noise = rng.normal(size=(n_samples, n_noise))
        else:
            X_noise = np.empty((n_samples, 0))

        # ── 5. Assemble full feature matrix
        X = np.hstack([X_inf, X_red, X_rep, X_noise])

        # Permute column order so informative aren't all first
        perm = rng.permutation(n_features)
        X = X[:, perm]

        # EIV noise on all features
        X_obs = X + rng.normal(size=(n_samples, n_features)) * feature_noise

        # ── 6. True logit (ONLY from informative features)
        # Scale linear_weight so total signal stays constant regardless of n_informative
        # Target: total logit std ≈ 3.5 (gives Gini ~0.65 with linear_weight=1.0)
        _TARGET_LOGIT_STD = 3.5
        effective_weight = _TARGET_LOGIT_STD / np.sqrt(n_informative) * linear_weight

        # Map: which columns are the informative ones after permutation?
        # (We build logit BEFORE permuting for clarity)
        w = rng.normal(size=n_informative) * effective_weight
        logit = X_inf @ w

        # Smooth non-linear on informative features
        n_smooth = max(1, int(n_informative * frac_smooth))
        for j in rng.choice(n_informative, size=n_smooth, replace=False):
            logit += rng.uniform(*smooth_amp) * np.tanh(X_inf[:, j])

        # Interactions among informative features
        n_ixn = max(1, int(n_informative * frac_ixn))
        for _ in range(n_ixn):
            j, k = rng.choice(n_informative, size=2, replace=False)
            logit += rng.uniform(*ixn_amp) * X_inf[:, j] * X_inf[:, k]

        # ── 7. Calibrate base rate
        target_logit = np.log(p_positive / (1 - p_positive))
        logit += target_logit - logit.mean()

        # ── 8. Labels
        p_pos = 1.0 / (1.0 + np.exp(-logit))
        threshold = np.quantile(p_pos, 1 - p_positive)
        y = (p_pos > threshold).astype(np.int8)

        flip = rng.uniform(size=n_samples) < label_noise
        y[flip] = 1 - y[flip]

        return X_obs, y

    @classmethod
    def _generate(cls, cfg: dict, target: str) -> pDF:
        """Generate the full dataset from a preset config."""
        fes  = cfg['n_numeric'] + cfg['n_categorical']
        seed = cfg['seed']

        X, y = cls.generate_data(
            n_samples       = cfg['n_samples'],
            n_features      = fes,
            n_informative   = cfg['n_informative'],
            n_redundant     = cfg['n_redundant'],
            n_repeated      = cfg['n_repeated'],            
            p_positive      = cfg['p_target'],
            linear_weight   = 1.5,
            label_noise     = 0.03,
            feature_noise   = 0.25,
            seed            = seed,
        )

        d = pDF(X, columns =
            [f'num_{i}' for i in range(cfg['n_numeric'])] +
            [f'cat_{i}' for i in range(cfg['n_categorical'])]
        )
        d[target] = y.astype(int)

        d = cls._add_time(d, seed)
        cls._add_categoricals(d, cfg)
        cls._add_non_stationary(d, seed)

        ids  = ['time', target]
        cols = [x for x in d.columns if x not in ids]

        return d[ids + cols]

    @staticmethod
    def _add_categoricals(d: pDF, cfg: dict) -> None:
        """Convert the categorical portion of columns to binned string labels."""
        cat_cols = [c for c in d.columns if c.startswith('cat_')]
        n_cat    = cfg['n_categories']
        labels   = _CAT_LABELS[n_cat]

        for c in cat_cols:
            d[c] = pd.qcut(d[c], q=n_cat, labels=labels, duplicates='drop').astype(str)

    @staticmethod
    def _add_non_stationary(d: pDF, seed: int) -> None:
        """Add non-stationary variables."""
        n = len(d)
        rng = np.random.default_rng(seed)

        for i in range(len(d.columns)//10):
            phi   = rng.uniform(0.3, 0.8)      # AR(1) coef
            theta = rng.uniform(-0.5, 0.5)     # MA(1) coef

            ar   = [1.0, -phi]
            ma   = [1.0, theta]
            arma = ArmaProcess(ar, ma)
            stationary = arma.generate_sample(nsample=n)

            d[f'ns_{i}'] = np.cumsum(stationary) # ARIMA d=1

    @staticmethod
    def _add_time(d: pDF, seed: int) -> pDF:
        """Add a synthetic time column (daily granularity, 2 years)."""
        rng  = np.random.default_rng(seed)
        days = rng.integers(0, 730, size=len(d))
        base = pd.Timestamp('2023-01-01')
        d['time'] = base + pd.to_timedelta(days, unit='D')
        return d.sort_values('time').reset_index(drop=True)