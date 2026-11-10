# Companion Explanation: AcademiQ_Polynomial_Regression.pkl

## A. Academic Understanding: Non-Linear Basis Expansion & Overfitting
- **What is Polynomial Regression?** A Generalized Linear Model (GLM) where original features are mapped into non-linear polynomial powers and pairwise interaction terms: $\phi_2(\mathbf{x}) = [1, x_1, \dots, x_p, x_1^2, x_1 x_2, \dots, x_p^2]^T$. The hypothesis remains linear with respect to parameters $\mathbf{w}$, allowing closed-form OLS optimization.
- **Combinatorial Feature Explosion:** For $p=24$ preprocessed features and degree $d=2$, the feature space expands to $M = \binom{p+d}{d} \approx 324$ terms.
- **The Bias-Variance Tradeoff:** While expanding model capacity reduces training bias, estimating hundreds of interaction parameters from finite samples increases estimator variance, leading to slight test set degradation.

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized scikit-learn `Pipeline` containing `ColumnTransformer`, `PolynomialFeatures(degree=2, include_bias=False)`, and `LinearRegression()`.
- **Expected Input:** A Pandas DataFrame containing all 19 raw student features.
- **Verified Benchmark Performance (Test $n = 1,322$):**
  - $R^2 = 0.7684$
  - $\text{MAE} = 0.4660$ points
  - $\text{RMSE} = 1.8093$ points
- **Artifact Role:** Loaded via `joblib.load()`; never retrained during evaluation.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 03 artifact; non-linear regression benchmark.
- **Trained By:** `AcademiQ_03_Regression.ipynb`.
- **Consumed By:** `AcademiQ_07_Final_Integration.ipynb`.
- **Viva Defense Insight:** Test MAE worsened slightly by $0.0162$ points compared to Multiple Linear Regression ($0.4660$ vs $0.4498$). This provides empirical proof of Occam's Razor: the extra interaction terms modeled sample noise rather than genuine academic grading mechanics.\n