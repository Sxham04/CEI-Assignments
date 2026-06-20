# Week 1 — Machine Learning Foundations

Covers the core mathematical and programming building blocks required for Machine Learning.

---

## Topics Covered

- **Python Fundamentals** — Control flow, data structures, exception handling, lambda functions
- **NumPy** — Array creation, reshaping, slicing, boolean indexing, matrix/vector operations
- **Pandas** — Series vs DataFrame operations, `loc`/`iloc`, groupby aggregation, missing data imputation
- **Linear Algebra** — L2 norm, matrix transformations, eigendecomposition, SVD rank-1 approximation
- **Statistics** — Descriptive stats, hypothesis testing (t-test), Pearson correlation, regression error metrics (MAE/MSE/RMSE/R²), KS test, Augmented Dickey-Fuller
- **Probability** — Joint/conditional probability, Normal/Binomial/Poisson distributions, Bayes' theorem, Central Limit Theorem verification

---

## File

| File | Description |
|------|-------------|
| `week1_soham.ipynb` | Solutions, verification tests, and visualizations |

---

## Key Implementations

**Population Stability Index (PSI)**  
Custom PSI function built from scratch to measure data distribution shift over time.

```python
def compute_psi(expected, actual, bins=10):
    combined = np.concatenate([expected, actual])
    bin_edges = np.linspace(min(combined), max(combined), bins + 1)
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    eps = 1e-10
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
```

**Regression Metrics from Scratch**  
Implemented MAE, MSE, RMSE, R², and Adjusted R² without external dependencies.

---

## Setup

```bash
pip install numpy pandas matplotlib scipy statsmodels
jupyter notebook week1_soham.ipynb
```
