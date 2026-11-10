# Companion Explanation: AcademiQ_logistic_regression.pkl

## A. Academic Understanding: Multinomial Logistic Regression (Softmax)
- **What is Multinomial Logistic Regression?** A parametric discriminative classification algorithm that models class-conditional probabilities for $K > 2$ discrete categories using the Softmax link function:
  $$P(Y = k | \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
- **Categorical Cross-Entropy Loss:** Parameters are estimated via Maximum Likelihood Estimation (MLE) using quasi-Newton optimization (L-BFGS) with $\ell_2$ weight regularization:
  $$\mathcal{L}(\mathbf{W}) = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K y_{ik} \log P(Y_i = k | \mathbf{x}_i) + \frac{1}{2C} \sum_{k=1}^K \|\mathbf{w}_k\|_2^2$$
- **Linear Decision Boundaries:** The decision boundary separating any two classes $j$ and $k$ is where $P(Y=j) = P(Y=k) \iff (\mathbf{w}_j - \mathbf{w}_k)^T \mathbf{x} + (b_j - b_k) = 0$, forming a strictly linear hyperplane in feature space.

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized scikit-learn `Pipeline` containing `ColumnTransformer` and `LogisticRegression(max_iter=2000, random_state=42)`.
- **Expected Input:** A Pandas DataFrame containing all 19 raw student features.
- **Output:** Predicted performance category string (`Developing`, `Proficient`, `Advanced`).
- **Champion Benchmark Performance (Test $n = 1,322$):**
  - **Accuracy = 95.08%** (`0.950832`)
  - **Macro Precision = 95.15%** (`0.951490`)
  - **Macro Recall = 94.76%** (`0.947592`)
  - **Macro F1 Score = 0.9492** (`0.949201`)
- **Artifact Role:** The **champion classification model** of the project; loaded directly without retraining.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 04 artifact; early-warning student triage engine.
- **Trained By:** `AcademiQ_04_Classification.ipynb`.
- **Consumed By:** `AcademiQ_07_Final_Integration.ipynb`.
- **Why Did It Outperform Complex Models?** The performance categories were formed by cutting continuous exam scores into tertiles. Because continuous exam score is a linear function of student features, the true boundaries separating Developing, Proficient, and Advanced are linear hyperplanes. Logistic Regression fits these optimal linear boundaries directly, while non-linear tree models suffered from jagged step approximations.\n