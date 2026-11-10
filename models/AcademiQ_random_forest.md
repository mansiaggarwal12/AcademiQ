# Companion Explanation: AcademiQ_random_forest.pkl

## A. Academic Understanding: Ensemble Bagging & Random Subspaces
- **What is a Random Forest?** An ensemble of $B$ decorrelated decision trees trained using Bootstrap Aggregating (Bagging) and the Random Subspace Method.
- **Variance Reduction Formula:** If individual trees have variance $\sigma^2$ and pairwise correlation $\rho$, the ensemble average has variance $\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$. As $B \to \infty$, variance drops to $\rho \sigma^2$.
- **Random Subspace Selection:** Restricting each split to a random subset of $m = \lfloor \sqrt{p} \rfloor = 4$ features prevents dominant predictors (like Attendance) from dominating every tree, driving down tree correlation $\rho$.

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized scikit-learn `Pipeline` containing `ColumnTransformer` and `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Expected Input:** A Pandas DataFrame containing all 19 raw student features.
- **Output:** Predicted performance category.
- **Verified Benchmark Performance (Test $n = 1,322$):**
  - **Accuracy = 81.39%** (`0.813918`)
  - **Macro F1 = 0.8071** (`0.807116`)
  - Rank 3 of 6 classifiers.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 04 artifact; non-linear ensemble benchmark.
- **Trained By:** `AcademiQ_04_Classification.ipynb`.
- **Consumed By:** `AcademiQ_07_Final_Integration.ipynb`.
- **Viva Defense Insight:** Random Forest beat the single Decision Tree by **11.5%** ($81.39\%$ vs $69.89\%$) through ensemble averaging. However, because decision trees partition space via axis-aligned rectangular boxes, approximating a smooth diagonal linear decision boundary resulted in a staircase approximation error compared to Logistic Regression.\n