# Companion Explanation: AcademiQ_KMeans_Model.pkl

## A. Academic Understanding: Centroid Representation & Vector Quantization
- **What is Stored in a Fitted KMeans Object?** The primary mathematical payload is the cluster centroid matrix $\mathbf{M} = [\boldsymbol{\mu}_0, \boldsymbol{\mu}_1]^T \in \mathbb{R}^{2 \times 24}$, representing the geometric center of each discovered cohort in standardized feature space.
- **Inertia Minimization:** Centroids represent the local minimum of Within-Cluster Sum of Squares (Inertia = $62,579.68$).

## B. Implementation in AcademiQ
- **Stored Object Type:** A standalone scikit-learn `KMeans` estimator object with $K=2$, `init='k-means++'`, and `random_state=42`.
- **Expected Input:** A 24-dimensional preprocessed numeric feature array.
- **Output:** Cluster index (`0` or `1`).

## C. Significance in the Complete Project
- **Workflow Position:** Stage 05 core unsupervised artifact.
- **Trained By:** `AcademiQ_05_Clustering.ipynb`.
- **Consumed By:** Loaded in Notebook 05 and Notebook 07 for cluster profiling.
- **Viva Defense Insight:** Silhouette score is $0.0580$. Centroids capture central tendencies of two continuous cohorts, but do not represent isolated natural boundaries.\n