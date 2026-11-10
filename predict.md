# Companion Explanation: predict.py

## A. Academic Understanding: Inference Pipelines & Model Deployment
- **Model Training vs. Inference:** Training is the computationally intensive process of estimating parameters ($\mathbf{w}, b$) by minimizing a loss function over historical data. Inference is the operational deployment phase: passing an unseen single instance $\mathbf{x}_{	ext{new}}$ through the frozen, learned parameters to obtain an immediate prediction $\hat{y}$.
- **Pipeline Architecture in Production:** In real-world machine learning, raw input data is rarely numeric and standardized. If preprocessing (imputation, scaling, one-hot encoding) is done via standalone scripts separate from model inference, discrepancies ("training-serving skew") occur. A scikit-learn `Pipeline` encapsulates feature transformations and the final estimator into an atomic object, ensuring that operational inference mirrors training preprocessing identically.
- **Prediction vs. Explanation vs. Causation:** A prediction answers "What score is expected for a student with these characteristics?" It does *not* prove that forcing a student to study 5 more hours will cause an exact increase in their exam score. Predictive correlation estimates conditional expectations, not causal mechanisms.

## B. Implementation in AcademiQ
- **Role in the Project:** A lightweight, interactive command-line interface that allows an educator or student to input 19 raw factors and obtain an instant final `Exam_Score` prediction using the pre-trained champion Multiple Linear Regression model.
- **Execution Flow:**
  1. **Model Ingestion:** Loads `models/AcademiQ_Multiple_Linear_Regression.pkl` using `joblib.load()` (**strictly zero retraining**).
  2. **Interactive Input & Validation:** Prompts the user through 19 features with bounds checking (e.g., `Attendance` validated between $0-100\%$, `Sleep_Hours` between $0-24$). For categorical attributes, displays numbered menus (`[1] Low, [2] Medium, [3] High`) and accepts case-insensitive text or integer selections.
  3. **DataFrame Assembly:** Formats user input into a single-row Pandas DataFrame matching the exact 19 column names and ordering expected by the pipeline.
  4. **Direct Pipeline Prediction:** Passes the raw DataFrame directly to `pipeline.predict()`. The pipeline's internal `ColumnTransformer` handles `StandardScaler` normalization and `OneHotEncoder` dummy variable alignment automatically.
  5. **Unclipped Display:** Emits the raw continuous score (e.g., $64.53$ points) along with an academic disclaimer that predictions are statistical guidance, not certified grades.
- **Key Code Snippet:**
  ```python
  pipeline = joblib.load(MODEL_PATH)
  input_df = pd.DataFrame([student_data], columns=FEATURE_COLUMNS)
  raw_prediction = pipeline.predict(input_df)[0]
  print(f"Predicted Exam_Score: {raw_prediction:.2f} points")
  ```
- **How to Execute:**
  ```powershell
  python predict.py
  ```

## C. Significance in the Complete Project
- **Workflow Position:** The end-user application layer of the AcademiQ project.
- **Preceding Dependencies:** Depends directly on `models/AcademiQ_Multiple_Linear_Regression.pkl` produced in Notebook 03.
- **What Breaks Without It:** Without this utility, evaluating a new student requires opening Jupyter Notebooks, writing Python code, and manually manipulating DataFrames. `predict.py` transforms AcademiQ from an academic research notebook into an accessible software tool.
- **Viva Defense Insight:** Notice that `predict.py` does not manually scale or encode inputs in Python code. All transformations occur inside the saved `Pipeline` object loaded from disk. This eliminates the risk of training-serving skew and guarantees that production inputs are preprocessed with the exact training parameters ($\mu, \sigma$, categories) established in Notebook 02.\n