# Companion Explanation: AcademiQ_k_nearest_neighbors.pkl

## A. Academic Understanding: Instance-Based Learning & High-Dimensional Pitfalls
- **What is K-Nearest Neighbors?** A non-parametric, lazy-learning algorithm that stores training instances and classifies queries via majority vote among the $K$ closest spatial neighbors: $\hat{y} = \arg\max_k \sum_{i \in \mathcal{N}_K(\mathbf{x})} \mathbb{I}(y_i = k)$.
- **The Curse of Dimensionality:** In 24 dimensions, feature space volume expands exponentially, data points become sparse, and Euclidean distance loses contrast (ratio of distance to nearest neighbor vs farthest neighbor approaches 1).
- **Isotropic Distance Distortion:** Unweighted Euclidean distance treats a 1-unit shift in Attendance identically to a 1-unit shift in School Type, allowing weak categorical noise to corrupt local neighborhoods.

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized scikit-learn `Pipeline` containing `ColumnTransformer` and `KNeighborsClassifier(n_neighbors=5, metric='minkowski')`.
- **Expected Input:** A Pandas DataFrame containing all 19 raw student features.
- **Verified Benchmark Performance (Test $n = 1,322$):**
  - Accuracy = $52.12\%$ (`0.521180`)
  - Macro F1 = $0.4934$ (`0.493416`)
  - Rank 6 of 6 classifiers (lowest performing).

## C. Significance in the Complete Project
- **Workflow Position:** Stage 04 artifact; instance-based baseline.
- **Trained By:** `AcademiQ_04_Classification.ipynb`.
- **Consumed By:** `AcademiQ_07_Final_Integration.ipynb`.
- **Viva Defense Insight:** Why did KNN fail? 24 dimensions diluted Euclidean distances, and equal feature weighting allowed uninformative features to overwhelm dominant signals (`Attendance` and `Hours_Studied`).\n