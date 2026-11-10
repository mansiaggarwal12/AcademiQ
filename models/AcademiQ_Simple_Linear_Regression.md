# Companion Explanation: AcademiQ_Simple_Linear_Regression.pkl

## A. Academic Understanding: Univariate OLS Regression
- **What is Simple Linear Regression?** A parametric supervised learning algorithm that models a linear relationship between a single independent predictor $x$ and a continuous dependent response $y$: $\hat{y} = w_0 + w_1 x$.
- **Closed-Form Ordinary Least Squares (OLS):** Parameters are calculated analytically by minimizing the Residual Sum of Squares (RSS):
  $$w_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}, \quad w_0 = \bar{y} - w_1 \bar{x}$$
- **Omitted Variable Bias:** In real-world educational contexts, exam performance is multi-factorial. By ignoring attendance, resources, and baseline ability, a univariate model suffers from omitted variable bias, restricting its explanatory capacity.

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized scikit-learn `Pipeline` object consisting of:
  1. `preprocessor`: A `ColumnTransformer` isolating `Hours_Studied`.
  2. `regressor`: A trained `LinearRegression(fit_intercept=True)` estimator.
- **Expected Input:** A Pandas DataFrame containing `Hours_Studied`.
- **Inference Mechanics:** Passes hours through the pipeline, evaluates $w_0 + w_1 x$, and outputs the continuous predicted score.
- **Verified Benchmark Performance (Test $n = 1,322$):**
  - $R^2 = 0.2320$ ($23.2\%$ score variance explained)
  - $\text{MAE} = 2.4476$ points
  - $\text{RMSE} = 3.2948$ points
- **Artifact Role:** Loaded from disk via `joblib.load()`; never retrained during inference.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 03 artifact; minimal complexity reference baseline.
- **Trained By:** `AcademiQ_03_Regression.ipynb`.
- **Consumed By:** Loaded and independently verified in `AcademiQ_07_Final_Integration.ipynb`.
- **Why It Is Essential:** In empirical ML, complex models must justify their complexity against a simple baseline. This model proves quantitatively that while study hours correlate positively with performance, they explain less than a quarter of the variance, making multivariable modeling mandatory.\n