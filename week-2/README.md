# Week 2 — End-to-End ML Pipeline on Tesla Delivery Data

**Intern:** Soham Sharma
**LMS ID:** CT_CSI_DS_1031
**Dataset:** Tesla Deliveries Dataset (2015–2025)

---

## Objective

Build a complete ML pipeline on real-world sales/price data covering preprocessing,
EDA, feature engineering, regression modeling, hyperparameter tuning, and time series forecasting.

---

## Topics Covered

| # | Topic |
|---|-------|
| 1 | Data Cleaning |
| 2 | Exploratory Data Analysis (EDA) |
| 3 | Time Series Components & Stationarity |
| 4 | Feature Engineering |
| 5 | Encoding & Chronological Split |
| 6 | Feature Scaling |
| 7 | Regression Models & Evaluation |
| 8 | Bias-Variance Tradeoff |
| 9 | Cross Validation |
| 10 | Hyperparameter Tuning (GridSearchCV) |
| 11 | sklearn Pipeline |
| 12 | Time Series Forecasting (SARIMAX) |

---

## Key Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~106 | ~151 | 0.9983 |
| Ridge (L2) | ~106 | ~151 | 0.9983 |
| Lasso (L1) | ~106 | ~151 | 0.9983 |

**Best Ridge Alpha (GridSearchCV):** 0.01
**SARIMAX Forecast:** 6-month ahead forecast on monthly aggregated deliveries

---

## Note

The high R² is a characteristic of the dataset — monthly Tesla delivery data is smooth
and strongly autocorrelated, so lag features are near-perfect predictors by construction.
Train and test curves track identically across all alpha values, confirming the model
generalizes well and is not overfitting.

---

## Stack

`pandas` `numpy` `scikit-learn` `statsmodels` `matplotlib` `seaborn`
