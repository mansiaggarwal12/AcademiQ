# Companion Explanation: AcademiQ_06_PCA_Feature_Analysis.ipynb

## A. Academic Understanding: Feature Selection vs. Dimensionality Reduction
- **The Curse of Dimensionality:** As feature dimensions grow, sample density decreases exponentially, distance metrics become equidistant, and multicollinearity inflates model coefficient variances.
- **Feature Selection (Filter Method - SelectKBest):** Selects a subset of existing features based on statistical significance against the target ($y$). Preserves original feature names, meanings, and physical units, ensuring maximum human interpretability.
- **Dimensionality Reduction (Unsupervised - PCA):** Performs eigenvalue decomposition on the sample covariance matrix (or SVD on $\mathbf{X}$), projecting features onto orthogonal axes that maximize variance. Eliminates multicollinearity completely and compresses data, but creates synthetic mathematical combinations (`PC1`, `PC2`) that lose direct physical meaning.
- **$F$-Regression Mechanics:** Tests the null hypothesis $H_0: ho = 0$ between a regressor and the target: $F = rac{r^2}{1 - r^2}(n - 2) \sim \mathcal{F}(1, n - 2)$.

## B. Implementation in AcademiQ
- **Role in the Project:** Stage 06 — Identifies critical educational policy levers and compresses the feature space.
- **Technique 1: SelectKBest Feature Ranking:**
  - Evaluated all 43 dummy-expanded features using `f_regression` against continuous `Exam_Score`.
  - Discovered that **`Attendance` ($F = 2666.48, p = 0.0$)** and **`Hours_Studied` ($F = 1244.44, p = 5.56 	imes 10^{-245}$)** are the overwhelming primary drivers of student performance.
  - Extracted the top 10 features (`num__Attendance`, `num__Hours_Studied`, `num__Previous_Scores`, `num__Tutoring_Sessions`, resource/involvement indicators).
- **Technique 2: Principal Component Analysis (PCA):**
  - Configured PCA with a 95% cumulative variance threshold (`n_components=0.95`).
  - Compressed the 43 preprocessed features into **24 orthogonal principal components**, capturing **$96.42\%$** of total dataset variance ($44.2\%$ dimension reduction with only $3.58\%$ information loss).
- **Outputs Produced:** `models/AcademiQ_Feature_Selector.pkl`, `models/AcademiQ_PCA_Pipeline.pkl`, PCA transformed datasets, and summary CSVs in `results/feature_analysis/`.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 06 — Explanatory and feature compression module.
- **Preceding Dependencies:** Consumes preprocessed feature matrices from `AcademiQ_02_Preprocessing.ipynb`.
- **Subsequent Consumers:** Feature rankings and PCA compression metrics are integrated into `AcademiQ_07_Final_Integration.ipynb`.
- **Viva Defense Insight:** Why does AcademiQ use both? SelectKBest answers the **policy question** (Which specific habits should a school encourage? Attendance and Study Hours). PCA answers the **engineering question** (Can we compress the dataset and eliminate collinearity without losing information? Yes, 24 components capture $96.42\%$ variance).\n