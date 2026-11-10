# Companion Explanation: AcademiQ_Clustering_Pipeline.pkl

## A. Academic Understanding: Unsupervised End-to-End Inference
- **Encapsulating Clustering Pipelines:** Standardizing features before clustering is mandatory because Euclidean distance is scale-sensitive. Encapsulating `ColumnTransformer` and `KMeans` into a single `Pipeline` guarantees that raw student records are scaled consistently before distance computation.
- **Inference in Clustering:** Unseen records are assigned to clusters via nearest-centroid calculation: $\arg\min_k \|\mathbf{x}_{\text{new}} - \boldsymbol{\mu}_k\|_2$.

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized scikit-learn `Pipeline` containing:
  1. `preprocessor`: `ColumnTransformer` (StandardScaler for numericals + OneHotEncoder for categoricals).
  2. `clusterer`: `KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)`.
- **Expected Input:** A Pandas DataFrame containing 19 raw student attributes (`Exam_Score` excluded).
- **Output:** Cluster assignment integer (`0` or `1`).

## C. Significance in the Complete Project
- **Workflow Position:** Stage 05 artifact; deployment clustering pipeline.
- **Trained By:** `AcademiQ_05_Clustering.ipynb`.
- **Consumed By:** Re-verified in `AcademiQ_07_Final_Integration.ipynb`.
- **What Breaks Without It:** Assigning new students to discovered behavioral cohorts would require manually scaling features in external code, risking training-serving skew.\n