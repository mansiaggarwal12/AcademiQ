# Companion Explanation: AcademiQ_PCA_Model.pkl

## A. Academic Understanding: Truncated Eigenvector Basis
- **What is Stored in a Fitted PCA Object?**
  1. The $43 \times 24$ right singular vector projection matrix $\mathbf{V}_{24}$ (eigenvectors / loadings).
  2. Singular values $s_k$ corresponding to eigenvalues $\lambda_k = \frac{s_k^2}{n-1}$.
  3. Mean centering vectors $\boldsymbol{\mu}$.

## B. Implementation in AcademiQ
- **Stored Object Type:** A standalone scikit-learn `PCA` object with 24 retained components.
- **Variance Retention:** Retains **96.42%** of total multivariate variance, achieving a $44.2\%$ dimension reduction with only $3.58\%$ information loss.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 06 core compression artifact.
- **Trained By:** `AcademiQ_06_PCA_Feature_Analysis.ipynb`.
- **Consumed By:** Loaded in Notebook 06 and Notebook 07.
- **Viva Defense Insight:** Eliminates multicollinearity completely because principal components are mutually orthogonal ($\mathbf{Z}^T \mathbf{Z}$ is diagonal).\n