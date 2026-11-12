# Companion Explanation: AcademiQ_Feature_Selector.pkl

## 1. Academic Understanding: Univariate Hypothesis Testing & Feature Filtering

### What is SelectKBest?
`SelectKBest` is a supervised univariate filter method that evaluates each candidate feature individually against the target variable (`Exam_Score`). Unlike wrapper or embedded methods, filter methods evaluate statistical relationships without training a predictive model, offering high computational efficiency and complete independence from algorithmic bias.

### How `f_regression` Evaluates Features
The scoring function `f_regression` tests the linear correlation between each feature and the target:
- **Null Hypothesis ($H_0$):** The true population correlation between feature $X_j$ and target $y$ is zero ($\rho = 0$).
- **Test Statistic ($F$):** Evaluates the variance explained by the linear regression divided by unexplained residual variance.
- **Interpretation:** Features with high $F$-scores and statistically significant $p$-values ($p < 0.05$) exhibit strong linear association with examination performance.

### Limitations: Predictive Association vs. Causality
A high $F$-score confirms that a feature is a strong statistical marker of exam performance within this observational cohort. It does **not** prove that intervening on that feature in isolation will produce a guaranteed, deterministic change in student marks.

---

## 2. Implementation in AcademiQ

### Stored Object & Parameters
`AcademiQ_Feature_Selector.pkl` is a serialized scikit-learn `SelectKBest` object saved using `joblib`:
- **Trained In:** `notebooks/AcademiQ_04_Feature_Analysis.ipynb`
- **Fitted On:** The standardized 43-column design matrix from `AcademiQ_Train_Cleaned.csv` ($5,285$ rows) against `y_train`.
- **Key Parameters Stored:**
  - `selector.scores_`: Array of 43 univariate $F$-scores.
  - `selector.pvalues_`: Array of corresponding $p$-values.
  - `selector.get_support()`: Boolean mask selecting the top 10 ranked features.
- **Top 2 Dominant Drivers:**
  - `num__Attendance`: $F = 2666.48, p \approx 0.0$
  - `num__Hours_Studied`: $F = 1244.44, p = 5.56 \times 10^{-245}$

---

## 3. Significance in the Complete Project & Viva Defense

### Workflow Position
- **Position in Pipeline:** Stage 04 primary analytical artifact.
- **Consumed By:** Evaluated in `AcademiQ_04_Feature_Analysis.ipynb` and synthesized in `AcademiQ_05_Final_Integration.ipynb`.
- **Role in What-If Analysis:** Identifies `Attendance` and `Hours_Studied` as the primary leverage points for scenario exploration in `whatif.py`.

### Viva Defense Insight
- **Question:** *"Why did you use SelectKBest instead of multivariate feature importance?"*
- **Defense Answer:** In AcademiQ, we use both! `SelectKBest` provides pure univariate statistical screening to isolate individual feature associations, while Multiple Linear Regression provides multivariate partial regression coefficients to capture combined effects. Both methods identify attendance and study hours as the dominant factors.
