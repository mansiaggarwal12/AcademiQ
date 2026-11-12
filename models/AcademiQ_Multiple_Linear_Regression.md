# Companion Explanation: AcademiQ_Multiple_Linear_Regression.pkl

## A. Academic Understanding: Multivariable Linear Regression & Best Linear Unbiased Estimation
- **What is Multiple Linear Regression (MLR)?** A parametric model estimating a linear combination across multiple input features: $\hat{y} = w_0 + \sum_{j=1}^p w_j x_j = \tilde{\mathbf{X}} \boldsymbol{\beta}$.
- **Gauss-Markov Theorem & BLUE:** Under the classical assumptions of linearity, strict exogeneity ($\mathbb{E}[\epsilon|\mathbf{X}] = 0$), no perfect multicollinearity, and homoscedastic, uncorrelated errors, the OLS estimator $\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ is BLUE (Best Linear Unbiased Estimator).
- **Singular Value Decomposition (SVD):** In scikit-learn, OLS is computed via SVD of $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$, avoiding explicit matrix inversion and handling dummy variable collinearity gracefully.
- **Parametric Interpretability:** With standardized features, each weight $w_j$ represents the expected change in exam score per standard deviation change in feature $j$, holding all other factors constant (*ceteris paribus*).

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized, production-ready scikit-learn `Pipeline` object containing:
  1. `preprocessor`: A `ColumnTransformer` with `StandardScaler` for the 6 numerical features and `OneHotEncoder(handle_unknown='ignore')` for the 13 categorical features.
  2. `regressor`: A trained `LinearRegression(fit_intercept=True)` estimator.
- **Expected Input:** A Pandas DataFrame containing all 19 raw student attributes.
- **Champion Benchmark Performance (Test $n = 1,322$):**
  - **$R^2 = 0.7699$** ($77.0\%$ total variance explained)
  - **$\text{MAE} = 0.4498$ points** (less than half a point average error)
  - **$\text{RMSE} = 1.8033$ points**
- **Artifact Role:** Deployed in `predict.py` for live score prediction, in `whatif.py` for counterfactual scenario exploration, and re-verified in Notebook 05.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 03 artifact; the **champion continuous forecasting model** of AcademiQ.
- **Trained By:** `AcademiQ_03_Regression.ipynb`.
- **Consumed By:** `predict.py`, `whatif.py`, and `AcademiQ_05_Final_Integration.ipynb`.
- **Why Did It Beat Polynomial Regression?** Expanding to degree-2 polynomial features created 300+ terms that overfitted training noise. MLR generalized better because the underlying grading mechanics are predominantly linear and additive across attendance, study hours, and prior scores.\n