# Companion Explanation: AcademiQ_decision_tree.pkl

## A. Academic Understanding: Recursive Partitioning & Gini Impurity
- **What is a Decision Tree?** A non-parametric supervised model that recursively partitions feature space into class-homogeneous leaf regions using binary axis-aligned orthogonal cuts ($x_j \le t$).
- **Gini Impurity Metric:** Evaluates node impurity: $G = 1 - \sum_{k=1}^K p_k^2$. The algorithm greedily selects the feature and threshold that maximize impurity reduction $\Delta G$.
- **High Variance & Instability:** Unpruned decision trees grow deep leaves that memorize idiosyncratic sample noise, leading to high generalization variance.

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized scikit-learn `Pipeline` containing `ColumnTransformer` and `DecisionTreeClassifier(random_state=42)`.
- **Expected Input:** A Pandas DataFrame containing all 19 raw student features.
- **Output:** Predicted performance category.
- **Verified Benchmark Performance (Test $n = 1,322$):**
  - Accuracy = $69.89\%$ (`0.698941`)
  - Macro F1 = $0.6876$ (`0.687589`)
  - Rank 4 of 6 classifiers.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 04 artifact; rule-based baseline.
- **Trained By:** `AcademiQ_04_Classification.ipynb`.
- **Consumed By:** `AcademiQ_07_Final_Integration.ipynb`.
- **Viva Defense Insight:** Serves as the transparent single-tree baseline. The jump from Decision Tree ($69.89\%$) to Random Forest ($81.39\%$) quantitatively proves the value of ensemble variance reduction on this dataset.\n