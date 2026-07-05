## Week 3 — Country Clustering for Strategic Aid Allocation

**Intern:** Soham Sharma  
**LMS ID:** CT_CSI_DS_1031  
**Dataset:** HELP International Country-Data (167 countries)  

### Objective

Segment countries by socio-economic and health factors to identify which are in the direst need of aid, using clustering validated through classification.

### Topics Covered

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Outlier Treatment
4. Feature Scaling
5. PCA
6. K-Means Clustering
7. Hierarchical Clustering
8. DBSCAN
9. Cluster Profiling & Ranking
10. Ensemble Classification (Random Forest, XGBoost)
11. Feature Importance Analysis

### Key Results

| Cluster | Label | Avg Child Mort | Avg Income | Avg GDPP |
|---------|-------|-----------------|------------|----------|
| 0 | Developing | 22.86 | 12317.53 | 6278.85 |
| 1 | Developed | 5.24 | 38711.08 | 37745.68 |
| 2 | Under-developed | 94.54 | 3312.73 | 1633.60 |

- **Optimal K:** 3 (elbow + silhouette)
- **PCA Explained Variance (2 components):** 66.7% + 15.1% = 81.8%
- **DBSCAN:** 1 cluster, 12 noise points — underperformed vs K-Means; noise points skewed toward higher-income outliers
- **Classification Accuracy:** Random Forest 96%, XGBoost 98%
- **Countries flagged in direst need:** 45
- **Top 3 priority countries:** Haiti, Sierra Leone, Chad

### Note

Random Forest and XGBoost were trained on K-Means cluster labels, not external ground truth, to validate cluster separability and surface the strongest drivers of segmentation.

### Stack

`pandas` `numpy` `scikit-learn` `xgboost` `scipy` `matplotlib` `seaborn`
