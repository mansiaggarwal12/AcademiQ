# Companion Explanation: AcademiQ_Feature_Selector.pkl

## A. Academic Understanding: Univariate Hypothesis Testing & Feature Filtering
- **What is SelectKBest?** A supervised filter method that ranks candidate features by evaluating univariate statistical significance against the target variable.
- **F-Regression Hypothesis Test:** Tests $H_0: \rho = 0$. Features with the largest $F$-scores (smallest $p$-values) explain the highest proportion of target variance in isolation.

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized scikit-learn `SelectKBest(score_func=f_regression, k=10)` object.
- **Key Parameters Stored:** Fitted $F$-scores and $p$-values across all 43 dummy-expanded columns, plus the boolean mask extracting the top 10 features.
- **Top Identified Drivers:**
  - `Attendance`: $F = 2666.48, p = 0.0$
  - `Hours_Studied`: $F = 1244.44, p = 5.56 \times 10^{-245}$

## C. Significance in the Complete Project
- **Workflow Position:** Stage 06 feature analysis artifact.
- **Trained By:** `AcademiQ_06_PCA_Feature_Analysis.ipynb`.
- **Consumed By:** Re-verified in `AcademiQ_07_Final_Integration.ipynb`.
- **Why It Matters:** Provides institutional stakeholders with actionable, named behavioral policy levers (encourage attendance and study hours) without abstract mathematical obfuscation.\n