# Companion Explanation: predict.py

## 1. Academic Understanding: Inference Pipelines, Model Deployment & Post-Hoc Discretization

### Model Training vs. Operational Inference
- **Training Phase:** Computationally intensive estimation of model parameters ($\mathbf{w}, b$) by minimizing mean squared error over historical training data ($5,285$ observations).
- **Inference Phase:** Operational deployment where an unseen single student instance $\mathbf{x}_{\text{new}}$ is passed through the frozen learned weights to obtain an immediate prediction $\hat{y}$.

### Threshold-Based Discretization vs. Redundant Classification
Rather than training a separate classification algorithm (which can introduce conflicting decision boundaries and independent classification errors), AcademiQ assigns students into educational tiers using empirical score cutoffs established from training score distribution tertiles:
- **Developing ($\le 66.0$):** Identifies students who may benefit from academic support or targeted tutoring.
- **Proficient ($66.0 < \text{Score} \le 69.0$):** Students performing at expected baseline grade standards.
- **Advanced ($> 69.0$):** Students demonstrating strong academic mastery and examination readiness.

Deriving categories directly from the continuous regression score guarantees complete mathematical harmony between score estimates and tier labels.

### Essential Distinction: Prediction vs. Causation
A prediction answers: *"What examination score is statistically expected for a student with this specific profile?"* It does **not** prove that forcing an individual student to study more hours will automatically cause a deterministic grade increase. Predictive correlation measures conditional expectation, not causal mechanism.

---

## 2. Implementation in AcademiQ

### Role and Execution Flow
`predict.py` is an interactive command-line interface that allows an educator, student, or counselor to input 19 raw student factors and receive an instant `Exam_Score` prediction and assigned academic tier:
1. **Model Ingestion:** Loads `models/AcademiQ_Multiple_Linear_Regression.pkl` using `joblib.load()` (**strictly zero retraining**).
2. **Interactive Input & Validation:** Prompts the user through 19 features with strict bounds checking. For categorical attributes, displays numbered menus (`[1] Low, [2] Medium, [3] High`) and accepts case-insensitive text or integer selections.
3. **DataFrame Assembly:** Formats user input into a single-row Pandas DataFrame matching the exact 19 column names and ordering expected by the pipeline.
4. **Direct Pipeline Prediction:** Passes the raw DataFrame directly to `pipeline.predict()`. The pipeline's internal `ColumnTransformer` handles `StandardScaler` normalization and `OneHotEncoder` dummy variable alignment automatically without training-serving skew.
5. **Threshold Categorization:** Passes the predicted continuous score to `get_performance_tier()` to map it to `Developing`, `Proficient`, or `Advanced`.
6. **Unclipped Display & Disclaimer:** Emits the raw continuous score (e.g., $74.13$ points) along with an academic disclaimer that predictions are statistical guidance, not certified grades.

### How to Execute
```powershell
python predict.py
```

---

## 3. Significance in the Complete Project & Viva Defense

### Workflow Position
- **Position in Pipeline:** Operational inference utility.
- **Preceding Dependencies:** Depends directly on `models/AcademiQ_Multiple_Linear_Regression.pkl` produced in Notebook 03.
- **Complementary Utility:** Works alongside `whatif.py` to provide both baseline estimation and counterfactual scenario exploration.

### Viva Defense Insight
- **Question:** *"Why did you not use a separate classification model to predict whether a student is Developing, Proficient, or Advanced?"*
- **Defense Answer:** Multiple Linear Regression estimates the continuous grade with an average error of only $0.45$ points. If we trained a separate classifier, edge cases near the cutoffs (e.g. a predicted score of $66.1$ classified as Developing) could contradict the regression score. Deriving the category directly from the continuous prediction via threshold logic guarantees $100\%$ consistency across the entire project.
