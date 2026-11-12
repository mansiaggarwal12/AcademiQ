# Companion Explanation: AcademiQ_03_Regression.ipynb

## A. Academic Understanding: Continuous Score Estimation & Model Complexity
- **The Continuous Regression Task:** Regression estimates a continuous mapping $f: \mathbb{R}^p 	o \mathbb{R}$ predicting the exact numerical value of $y$ (`Exam_Score`). It answers: "What percentage grade is a student expected to achieve?"
- **The Bias-Variance Tradeoff & Occam's Razor:**
  - *Simple Linear Regression (High Bias / Underfitting):* Modeling score purely as a function of study hours ignores attendance, resources, and prior foundation, resulting in severe omitted variable bias.
  - *Polynomial Regression (High Variance / Risk of Overfitting):* Expanding 24 features to degree 2 creates $pprox 324$ quadratic and interaction terms. If the true data generation mechanism is linear, these extra degrees of freedom capture idiosyncratic noise rather than genuine signal.
  - *Occam's Razor:* When a simpler parametric model explains the data as well as or better than a complex model, the simpler model must be chosen because it generalizes more robustly and offers complete transparency.
- **Evaluation Metrics:**
  - $	ext{MAE} = rac{1}{n}\sum |y_i - \hat{y}_i|$: Average linear magnitude of error in actual grade points.
  - $	ext{RMSE} = \sqrt{rac{1}{n}\sum (y_i - \hat{y}_i)^2}$: Penalizes large error outliers quadratically.
  - $R^2 = 1 - rac{	ext{SS}_{	ext{res}}}{	ext{SS}_{	ext{tot}}}$: Proportion of target variance explained by predictors.

## B. Implementation in AcademiQ
- **Role in the Project:** Stage 03 — Develops and validates models for forecasting final examination scores.
- **Models Benchmarked on Test Set ($n = 1,322$):**
  1. **Simple Linear Regression:** Univariate baseline on `Hours_Studied` ($R^2 = 0.2320$, $	ext{MAE} = 2.4476$ pts, $	ext{RMSE} = 3.2948$ pts).
  2. **Multiple Linear Regression (Champion):** Full multivariable model across all 19 features (**$R^2 = 0.7699$**, **$	ext{MAE} = 0.4498$ pts**, **$	ext{RMSE} = 1.8033$ pts**).
  3. **Polynomial Regression (Degree 2):** Quadratic expansion ($R^2 = 0.7684$, $	ext{MAE} = 0.4660$ pts, $	ext{RMSE} = 1.8093$ pts).
- **Validation Protocol:** Executed 5-fold cross-validation on training data (MLR mean $R^2 = 0.7237, 	ext{MAE} = 0.5067$). Conducted residual scatter and Q-Q normality diagnostics.
- **Outputs Produced:** Saved model artifacts in `models/` (`AcademiQ_Multiple_Linear_Regression.pkl`, etc.) and metric CSVs in `results/regression/`.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 03 — Primary quantitative forecasting module.
- **Preceding Dependencies:** Consumes cleaned train and test partitions from `AcademiQ_02_Preprocessing.ipynb`.
- **Subsequent Consumers:** The champion MLR model is deployed in `predict.py`, explored counterfactually in `whatif.py`, and consolidated into the final scorecard in `AcademiQ_05_Final_Integration.ipynb`.
- **What Breaks Without It:** AcademiQ would lose its continuous quantitative forecasting ability, forcing institutions to rely solely on coarse category buckets rather than exact grade predictions.
- **Viva Defense Insight:** Why did MLR beat Polynomial Regression? The true relationship between student factors and exam scores is largely linear and additive. Expanding to 324 polynomial parameters increased test MAE from $0.4498$ to $0.4660$ because the model overfitted subtle training sample noise.\n