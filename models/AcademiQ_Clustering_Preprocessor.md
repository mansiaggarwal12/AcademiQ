# Companion Explanation: AcademiQ_Clustering_Preprocessor.pkl

## A. Academic Understanding: Unsupervised Preprocessing Architecture
- **Unsupervised Transformation Isolation:** Unlike supervised pipelines that preprocess features alongside a target, unsupervised preprocessors must operate strictly on feature subsets without target awareness to preserve exploratory integrity.

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized scikit-learn `ColumnTransformer` fitted exclusively on the 19 input features of `X_train`.
- **Transformations Applied:** Median imputation and $z$-score scaling on the 6 continuous variables; mode imputation and one-hot encoding on the 13 categorical variables.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 05 supporting artifact.
- **Trained By:** `AcademiQ_05_Clustering.ipynb`.
- **Role:** Allowed the clustering notebook to transform feature matrices independently to evaluate candidate cluster counts $K \in [2, 8]$.\n