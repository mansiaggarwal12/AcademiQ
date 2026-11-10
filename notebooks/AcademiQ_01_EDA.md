# Companion Explanation: AcademiQ_01_EDA.ipynb

## A. Academic Understanding: Exploratory Data Analysis (EDA)
- **What is EDA?** EDA is an analytical approach pioneered by John Tukey to systematically inspect, summarize, and visualize data before formal modeling. It tests statistical assumptions, identifies underlying distribution geometry, discovers bivariate associations, and uncovers data collection errors.
- **Parametric Assumptions & Target Normality:** Classical Ordinary Least Squares (OLS) regression assumes that error residuals follow a normal distribution $\epsilon \sim \mathcal{N}(0, \sigma^2)$. When the raw target variable $y$ is already approximately symmetric and Gaussian, linear models naturally produce well-behaved residuals without requiring non-linear transformations (such as logarithmic or Box-Cox transforms).
- **Correlation vs. Causation:** Pearson's correlation coefficient $r = rac{	ext{Cov}(X, Y)}{\sigma_X \sigma_Y}$ measures the linear association between two continuous variables. A high correlation (e.g., between attendance and score) proves predictive co-variation; it does not scientifically prove that attendance alone causes higher scores, as confounding variables (motivation, baseline knowledge) may influence both.

## B. Implementation in AcademiQ
- **Role in the Project:** First analytical stage of AcademiQ. Conducts initial audits on the raw dataset (`Student Performance Factors.csv`, 6,607 rows $\times$ 20 columns).
- **Major Cell Stages & Operations:**
  1. **Dimensionality & Missingness Audit:** Identified 19 predictor features (6 continuous, 13 categorical) and 1 target (`Exam_Score`). Detected missing values across 4 columns (`Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home`, and minor continuous entries).
  2. **Target Distribution Diagnostics:** Evaluated `Exam_Score`. Verified mean of $67.24$, median of $67.0$, standard deviation of $3.91$, skewness of $0.084$, and kurtosis of $-0.021$. These moments confirmed a virtually symmetrical bell curve.
  3. **Anomaly & Boundary Detection:** Flagged a single record where `Exam_Score = 101`. Because academic examinations operate on a fixed $[0, 100]$ grading scale, 101 represents an impossible recording artifact.
  4. **Bivariate Correlation Analysis:** Generated Pearson correlation heatmaps across numerical attributes. Discovered that `Attendance` ($r pprox 0.58$) and `Hours_Studied` ($r pprox 0.44$) are the primary linear predictors of academic performance.
- **Outputs Produced:** `docs/AcademiQ_EDA_Summary.csv` and exploratory distribution plots.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 01 — Foundation of the entire empirical pipeline.
- **Subsequent Consumers:** Directly informs every preprocessing decision in `AcademiQ_02_Preprocessing.ipynb`:
  - Confirmed that `Exam_Score` does not need log-scaling.
  - Mandated the capping of the 101 outlier to 100.
  - Highlighted which 4 columns require dedicated imputation strategies.
- **What Would Be Missing Without It:** Without EDA, modeling would be conducted blindly. The 101-point outlier would exert leverage on regression weights, unhandled NaNs would crash scikit-learn estimators, and practitioners would lack empirical evidence regarding which features dominate performance.\n