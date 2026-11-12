# Companion Explanation: AcademiQ_Train_Cleaned.csv

## A. Academic Understanding: Partitioned Training Data
- **Role of the Training Set:** The subset of historical observations ($pprox 80\%$) used by supervised learning algorithms to estimate parameters (weights, decision tree splits, class-conditional probabilities) through empirical risk minimization.
- **Domain Cleaning vs. Sample Deletion:** When an anomalous value violates a physical boundary (such as `Exam_Score = 101`), deleting the row discards valid survey information. Applying domain capping ($\min(y, 100)$) remediates the recording error while preserving sample integrity.
- **Data Schema:** Retains raw feature scales and categorical string labels so that preprocessing can be encapsulated inside scikit-learn pipelines.

## B. Implementation in AcademiQ
- **Dimensions:** Exactly **5,285 rows** $\times$ 20 columns ($80\%$ of 6,607).
- **Cleaning Applied:** Capped the single `Exam_Score = 101` entry to `100`.
- **Features Included:** All 19 original predictor columns + cleaned `Exam_Score`.
- **Partitioning Method:** Created via `train_test_split(test_size=0.20, random_state=42, shuffle=True)` in `AcademiQ_02_Preprocessing.ipynb`.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 02 output; primary training substrate for all supervised pipelines.
- **Downstream Consumers:** Loaded directly by `AcademiQ_03_Regression.ipynb` and `AcademiQ_04_Feature_Analysis.ipynb` to fit estimators and feature selectors.
- **What Breaks Without It:** If training and test sets are not persisted to disk as frozen files, re-running notebooks with different random states would train on differing subsets of students, invalidating reproducibility across notebooks.\n