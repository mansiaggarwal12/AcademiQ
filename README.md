# AcademiQ: Applied Machine Learning Framework for Academic Performance Prediction

AcademiQ is an integrated, end-to-end Machine Learning research pipeline designed to analyze, predict, and profile student examination performance using multi-dimensional behavioural and demographic factors.

## Dataset Overview

The investigation analyzes the **Student Performance Factors** benchmark dataset obtained from **Kaggle**:
- **Total Observations:** 6,607 student records
- **Total Columns:** 20 (19 predictor features + 1 target variable)
- **Target Variable:** `Exam_Score` (continuous grade on a 0–100 scale)
- **Predictor Features:** Academic habits (e.g., `Hours_Studied`, `Attendance`, `Tutoring_Sessions`, `Previous_Scores`), lifestyle factors (e.g., `Sleep_Hours`, `Physical_Activity`), and demographic/institutional attributes (e.g., `Parental_Involvement`, `Access_to_Resources`, `Teacher_Quality`, `School_Type`).
- **Data Quality Handling:** Missing categorical values in three features (`Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home`) are explicitly imputed as `'Missing'`, and one recorded target outlier (`Exam_Score = 101`) is adjusted to 100 before dataset partitioning to ensure data integrity.

## Three Machine Learning Perspectives

Rather than introducing an artificial or overly complex algorithm, AcademiQ's central contribution is the complementary, multi-paradigm evaluation of the same student cohort across three distinct analytical lenses:

1. **Regression (Continuous Score Estimation):**
   Estimates a student's continuous exam score, providing fine-grained performance predictions and identifying the quantitative impact of study hours and attendance rates.
2. **Classification (Performance Tier Assignment):**
   Assigns students into actionable academic categories (`Developing`, `Proficient`, or `Advanced`) derived from empirical score tertiles, enabling targeted educational interventions.
3. **Clustering (Exploratory Student Profiling):**
   Explores latent behavioral patterns and lifestyle similarities across students without using examination labels.

> **Methodological Note on Clustering:**
> The optimal clustering model ($K=2$) achieves a silhouette score of approximately **0.0580**. This modest score reflects overlapping, continuous multivariate distributions rather than distinctly separated, isolated student archetypes. Clustering is therefore utilized for exploratory profiling along a continuum, rather than discrete natural classification.

## Quick Start & Setup

The project is designed to run in a clean, self-contained Python virtual environment. It has been successfully tested on **Python 3.13.4** (Windows 64-bit).

### 1. Environment Setup (Windows PowerShell)

```powershell
python -m venv venv
.env\Scriptsctivate
pip install -r requirements.txt
```

### 2. Execution Instructions

Execute the notebooks sequentially from fresh kernels in the documented order:

1. `notebooks/AcademiQ_01_EDA.ipynb` — Exploratory Data Analysis & Anomaly Detection
2. `notebooks/AcademiQ_02_Preprocessing.ipynb` — Leakage-Safe Data Preprocessing & Splitting
3. `notebooks/AcademiQ_03_Regression.ipynb` — Continuous Score Prediction Modeling
4. `notebooks/AcademiQ_04_Classification.ipynb` — Performance Category Tier Classification
5. `notebooks/AcademiQ_05_Clustering.ipynb` — Unsupervised Student Profiling & Silhouette Analysis
6. `notebooks/AcademiQ_06_PCA_Feature_Analysis.ipynb` — Feature Selection & Dimensionality Reduction
7. `notebooks/AcademiQ_07_Final_Integration.ipynb` — Multi-Paradigm Synthesis & Comparison Visualizations

> **Note:** Each notebook automatically detects the project root directory regardless of whether it is launched from the project root or from inside `notebooks/`. Notebook 07 completes the pipeline by generating the master summary table and comparative visualization plots in `results/final/`.

## Pipeline Architecture

The investigation executes across seven sequential stages:

### 1. Exploratory Data Analysis (`notebooks/AcademiQ_01_EDA.ipynb`)
- Bivariate screening, distribution analysis, and anomaly detection.
- Missing value audit in categorical features (`Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home`).
- Correlation analysis showing `Attendance` (r = 0.58) and `Hours_Studied` (r = 0.45) as principal correlates.

### 2. Preprocessing & Leakage Prevention (`notebooks/AcademiQ_02_Preprocessing.ipynb`)
- Outlier boundary enforcement prior to partitioning.
- Missing category imputation with explicit 'Missing' category.
- Leakage-free 80:20 Train-Test split (`random_state=42`).
- StandardScaler on numerical features, OneHotEncoder on categorical attributes.

### 3. Regression Modeling (`notebooks/AcademiQ_03_Regression.ipynb`)
- Architectures: Simple Linear Regression, Multiple Linear Regression, Polynomial Regression (degree 2).
- Validation: 5-fold cross-validation inside pipeline structures.
- Primary Metrics: MAE, MSE, RMSE, R2.

### 4. Classification Modeling (`notebooks/AcademiQ_04_Classification.ipynb`)
- Discretization: Target tertiles derived strictly from training data quantiles.
- Classifiers: Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes, Support Vector Machine (RBF).
- Metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, Confusion Matrix, 5-fold CV.

### 5. Exploratory Clustering (`notebooks/AcademiQ_05_Clustering.ipynb`)
- Algorithm: K-Means clustering on 43 preprocessed features.
- Optimal K Selection: Elbow curve and Silhouette analysis.
- Methodological Finding: Silhouette score (0.0580) reflects overlapping continuous distributions; clustering serves exploratory profiling rather than distinct classification.

### 6. Feature Selection & PCA (`notebooks/AcademiQ_06_PCA_Feature_Analysis.ipynb`)
- SelectKBest: Univariate F-test identifying Attendance and Hours Studied as dominant features.
- PCA: 24 principal components capture >95% (96.42%) cumulative explained variance.

## Benchmark Results

### Regression Performance
- **Best Model:** Multiple Linear Regression
- **Test MAE:** 0.4498
- **Test RMSE:** 1.8033
- **Test R2:** 0.7699

### Classification Performance
- **Best Model:** Logistic Regression
- **Test Accuracy:** 0.9508
- **Test Macro F1:** 0.9492

### Clustering Profile
- **Optimal K:** 2
- **Silhouette Score:** 0.0580
- **Interpretation:** Continuous multivariate distribution with exploratory student grouping.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Structure

```
PROJECT 2.0/
|-- notebooks/
|   |-- AcademiQ_01_EDA.ipynb                   # Stage 1: Exploratory Data Analysis
|   |-- AcademiQ_02_Preprocessing.ipynb         # Stage 2: Leakage-Safe Data Preprocessing
|   |-- AcademiQ_03_Regression.ipynb            # Stage 3: Score Prediction Regression
|   |-- AcademiQ_04_Classification.ipynb        # Stage 4: Performance Tier Classification
|   |-- AcademiQ_05_Clustering.ipynb            # Stage 5: Unsupervised Student Profiling
|   |-- AcademiQ_06_PCA_Feature_Analysis.ipynb  # Stage 6: Feature Selection & PCA
|   `-- AcademiQ_07_Final_Integration.ipynb     # Stage 7: Synthesis & Packaging
|-- data/
|   |-- raw/
|   |   `-- Student Performance Factors.csv     # Raw Dataset
|   `-- processed/                              # Cleaned, Preprocessed, Clustered, and PCA Datasets
|       |-- AcademiQ_Train_Cleaned.csv
|       |-- AcademiQ_Test_Cleaned.csv
|       |-- AcademiQ_Train_Preprocessed.csv
|       |-- AcademiQ_Test_Preprocessed.csv
|       |-- AcademiQ_Train_Clustered.csv
|       |-- AcademiQ_Test_Clustered.csv
|       |-- AcademiQ_Train_PCA.csv
|       `-- AcademiQ_Test_PCA.csv
|-- models/                                     # Serialized Scikit-learn Pipelines & Estimators
|   |-- AcademiQ_Simple_Linear_Regression.pkl
|   |-- AcademiQ_Multiple_Linear_Regression.pkl
|   |-- AcademiQ_Polynomial_Regression.pkl
|   |-- AcademiQ_logistic_regression.pkl
|   |-- AcademiQ_k_nearest_neighbors.pkl
|   |-- AcademiQ_decision_tree.pkl
|   |-- AcademiQ_random_forest.pkl
|   |-- AcademiQ_naive_bayes.pkl
|   |-- AcademiQ_support_vector_machine.pkl
|   |-- AcademiQ_KMeans_Model.pkl
|   |-- AcademiQ_Clustering_Preprocessor.pkl
|   |-- AcademiQ_Clustering_Pipeline.pkl
|   |-- AcademiQ_PCA_Preprocessor.pkl
|   |-- AcademiQ_Feature_Selector.pkl
|   |-- AcademiQ_PCA_Model.pkl
|   `-- AcademiQ_PCA_Pipeline.pkl
|-- results/
|   |-- regression/                             # Regression evaluation tables (Test & CV)
|   |-- classification/                         # Classification evaluation tables (Test, CV & Thresholds)
|   |-- clustering/                             # Clustering evaluations and profile tables
|   |-- feature_analysis/                       # Feature importance scores and PCA summaries
|   `-- final/                                  # Master summary table and comparative visualization figures
|-- docs/
|   `-- AcademiQ_EDA_Summary.csv                # Dataset profiling summary metrics
|-- requirements.txt                            # Verified environment dependencies
`-- README.md                                   # Comprehensive project documentation
```

## Limitations

- Statistical associations do not imply direct causation.
- Performance category boundaries are derived from the empirical sample distribution.
- Clustering profiles describe general tendencies along a continuum rather than isolated student archetypes.
