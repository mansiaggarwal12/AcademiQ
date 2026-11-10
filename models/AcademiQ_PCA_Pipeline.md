# Companion Explanation: AcademiQ_PCA_Pipeline.pkl

## A. Academic Understanding: End-to-End Latent Space Projections
- **Why Pipeline PCA?** PCA is sensitive to feature scale; unscaled features distort principal axes. Binding preprocessing and PCA into an atomic pipeline guarantees that new student records are standardized before projection into the 24-dimensional latent space.

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized scikit-learn `Pipeline` joining `ColumnTransformer` (with full 43-column dummy expansion) and `PCA(n_components=0.95, random_state=42)`.
- **Expected Input:** A Pandas DataFrame with 19 raw student features.
- **Output:** 24-dimensional orthogonal component vector (`PC1` through `PC24`).
- **Variance Captured:** Exactly **96.42%** cumulative explained variance.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 06 artifact; deployment-ready dimensionality reduction pipeline.
- **Trained By:** `AcademiQ_06_PCA_Feature_Analysis.ipynb`.
- **Consumed By:** Verified in `AcademiQ_07_Final_Integration.ipynb`.\n