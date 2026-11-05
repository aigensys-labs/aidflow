Getting AIDflow into your environment is straightforward. We support all major Python 3.12+ distributions.


## Requirements

- **Python 3.12+**
- **OS:** Linux, macOS, Windows
- **Memory:** 8 GB+ (recommended: 16 GB for `--size large` runs)


## Basic Installation

Install the stable release via pip:

```bash
# Core (CLI pipeline)
pip install aidflow

# + Interactive notebook UI
pip install aidflow[notebook]

# + Ray (multiprocess acceleration, Python < 3.14)
pip install aidflow[ray]

# All optional
pip install aidflow[all]
```


## What's Inside

- **Core Compute:** PyArrow, NumPy, Pandas, SciPy, Scikit-Learn, statsmodels
- **Model Backends:** Logistic Regression, Lasso, ElasticNet, PLS, XGBoost, LightGBM
- **Optimization:** SHAP, Optuna, Permutation Importance, Shadow Features
- **Execution:** Ray (optional) | multiprocessing
- **Output:** HTML reports | ANSI terminal | Jupyter interactive


## Verifying Installation
To ensure AIDflow is correctly installed and your environment is ready for the ProjectContext orchestration, run:

```python
import aidflow
print(aidflow.__version__)
```

## Customizing Your Environment
For corporate environments with specific security constraints (e.g., internal PyPI mirrors), you can install via your config file:

```bash
pip install -r requirements.txt --index-url http://your-internal-mirror/simple
```