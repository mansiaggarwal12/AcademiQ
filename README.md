# AcademiQ: Student Academic Performance Prediction and Analysis

AcademiQ is a focused, academically grounded Machine Learning project designed to **predict students' final examination scores** and **analyse the student-related factors associated with academic performance**.

The project prioritizes clarity, scientific explainability, and rigorous evaluation over arbitrary algorithmic complexity. Rather than maintaining multiple disconnected ML branches, AcademiQ centers on **interpretable regression modeling**, **statistical factor analysis**, and **interactive scenario exploration**.

---

## Central Research Objectives

1. **Continuous Score Prediction:** Model final examination marks (`Exam_Score` on a 0–100 scale) as a function of 19 behavioral, lifestyle, and demographic attributes using regression algorithms.
2. **Student Factor Analysis:** Empirically rank and interpret which student-related factors share the strongest statistical associations with exam outcomes using univariate hypothesis testing (`SelectKBest` with $F$-regression) and multivariate regression coefficients.
3. **Threshold-Based Performance Categorization:** Derive transparent educational performance categories (`Developing`, `Proficient`, `Advanced`) directly from predicted continuous scores without introducing a separate, potentially contradictory classification model.
4. **Counterfactual What-If Analysis:** Allow educators, counselors, and students to explore how hypothetical adjustments in key habits (e.g., Attendance or Study Hours) shift predicted grades under *ceteris paribus* conditions, supported by domain-grounded input validation and non-causal disclaimers.

---

## Dataset Overview

The investigation analyzes the **Student Performance Factors** benchmark dataset obtained from **Kaggle**:
- **Total Observations:** 6,607 student records ($5,285$ training, $1,322$ holdout test)
- **Total Columns:** 20 (19 predictor features + 1 target variable)
- **Target Variable:** `Exam_Score` (continuous grade on a 0–100 scale; training mean: $67.21$, median: $67.0$)
- **Predictor Features:** 
  - *Academic Habits:* `Hours_Studied`, `Attendance`, `Tutoring_Sessions`, `Previous_Scores`
  - *Lifestyle & Health:* `Sleep_Hours`, `Physical_Activity`, `Extracurricular_Activities`
  - *Contextual & Demographic:* `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Learning_Disabilities`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`, `Internet_Access`
- **Data Quality Handling:** Missing categorical values in three features (`Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home`) are explicitly imputed with the most frequent training category, and one recorded target outlier (`Exam_Score = 101`) is adjusted to 100 before dataset partitioning.

---

## Quick Start & Setup

The project runs in a clean, self-contained Python virtual environment. It has been successfully tested on **Python 3.13.4** (Windows 64-bit).

### 1. Environment Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Sequential Notebook Execution

Execute the notebooks sequentially from fresh kernels in the documented order:

1. `notebooks/AcademiQ_01_EDA.ipynb` — Exploratory Data Analysis & Anomaly Detection
2. `notebooks/AcademiQ_02_Preprocessing.ipynb` — Leakage-Safe Data Preprocessing & 80:20 Partitioning
3. `notebooks/AcademiQ_03_Regression.ipynb` — Continuous Score Prediction Modeling (SLR, MLR, Polynomial)
4. `notebooks/AcademiQ_04_Feature_Analysis.ipynb` — Student Factor Analysis (SelectKBest & MLR Coefficients)
5. `notebooks/AcademiQ_05_Final_Integration.ipynb` — Synthesis, Benchmark Plots, Inference & What-If Demo

> **Note:** Each notebook automatically detects the project root directory regardless of whether it is launched from the project root or from inside `notebooks/`.

### 3. Interactive Command-Line Utilities

AcademiQ includes two standalone terminal applications:

#### A. Interactive Student Score Prediction (`predict.py`)
Accepts student inputs, passes them to the saved MLR pipeline, predicts the numerical score, and assigns a performance tier:
```powershell
python predict.py
```

#### B. What-If Academic Scenario Analysis (`whatif.py`)
Allows users to alter a single student attribute (e.g., Attendance from $65\%$ to $85\%$) while holding all other 18 factors constant, displaying the predicted shift and educational tier impact:
```powershell
python whatif.py
```

---

## Pipeline Architecture

```
[ Raw Kaggle Dataset: 6,607 records ]
                 │
                 ▼
 Stage 1: Exploratory Data Analysis (AcademiQ_01_EDA)
   - Audited distributions, detected missingness, flagged 101-point anomaly.
   - Identified Attendance (r = 0.58) and Hours_Studied (r = 0.44) as primary correlates.
                 │
                 ▼
 Stage 2: Leakage-Safe Preprocessing (AcademiQ_02_Preprocessing)
   - Enforced 0-100 score bounds; 80:20 disjoint split (5,285 train / 1,322 test).
   - Median/mode imputation; StandardScaler and OneHotEncoder fitted strictly on training data.
                 │
                 ▼
 Stage 3: Regression Modeling & Benchmark (AcademiQ_03_Regression)
   - Simple Linear Regression (Baseline): R² = 0.2320, MAE = 2.45 pts (Underfitting)
   - Multiple Linear Regression (CHAMPION): R² = 0.7699, MAE = 0.45 pts, RMSE = 1.80 pts
   - Polynomial Regression (Degree 2): R² = 0.7684, MAE = 0.47 pts (Capacity Plateau)
                 │
                 ▼
 Stage 4: Student Factor Analysis (AcademiQ_04_Feature_Analysis)
   - Univariate SelectKBest with f_regression identifies dominant correlates:
     1. Attendance (F = 2666.48)
     2. Hours_Studied (F = 1244.44)
     3. Previous_Scores (F = 162.90)
     4. Tutoring_Sessions (F = 131.86)
   - Examined multivariate MLR standardized coefficients to understand directional weightings.
                 │
                 ▼
 Stage 5: Final Synthesis & Decision Support (AcademiQ_05_Final_Integration)
   - Master model scorecard and comparative visual figures exported.
   - Live pipeline inference demonstrated on sample student.
   - Threshold-based tier categorization: Developing (≤ 66), Proficient (66–69), Advanced (> 69).
   - Counterfactual what-if scenario evaluation with non-causal disclaimers.
```

---

## Benchmark Results (Test Set, $n = 1,322$)

| Regression Model | Features Used | Test $R^2$ | Test MAE | Test RMSE | Selection Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Simple Linear Regression** | `Hours_Studied` Only | `0.2320` | `2.4476` | `3.2948` | Baseline (Severe Underfitting) |
| **Multiple Linear Regression** | All 19 Student Factors | **`0.7699`** | **`0.4498`** | **`1.8033`** | **CHAMPION MODEL (Optimal Parity & Interpretability)** |
| **Polynomial Regression (Degree 2)** | All 19 Factors + Degree-2 Poly | `0.7684` | `0.4660` | `1.8093` | Evaluated (Capacity Plateau / Overfitting Risk) |

### Why Multiple Linear Regression Was Selected
1. **Superior Generalization:** MLR achieves the highest $R^2$ ($0.7699$) and the lowest test MAE ($0.4498$ points, an $81.6\%$ error reduction over Simple Linear Regression).
2. **Occam's Razor & Parsimony:** Degree-2 Polynomial Regression introduces over 500 interaction parameters but fails to improve $R^2$ ($0.7684$) while slightly worsening test MAE ($0.4660$). MLR achieves optimal fit without parameter bloat.
3. **Complete Interpretability:** Linear regression coefficients directly quantify the contribution of each student factor in standardized units, making it fully explainable to educators, parents, and academic evaluators.

---

## Threshold-Based Performance Categorization

Rather than training a separate classification model (which risks conflicting predictions near decision boundaries), AcademiQ categorizes students using transparent threshold logic derived from empirical training score tertiles:

| Performance Tier | Score Boundary | Academic Interpretation |
| :--- | :--- | :--- |
| **Developing** | $\text{Exam\_Score} \le 66.0$ | Identifies students who may benefit from academic support or targeted interventions. |
| **Proficient** | $66.0 < \text{Exam\_Score} \le 69.0$ | Students performing at expected baseline academic standards. |
| **Advanced** | $\text{Exam\_Score} > 69.0$ | Students demonstrating strong academic mastery and high examination readiness. |

---

## Simplified Project Structure

```
PROJECT 2.0/
├── notebooks/
│   ├── AcademiQ_01_EDA.ipynb                   # Stage 1: Exploratory Data Analysis
│   ├── AcademiQ_01_EDA.md                      # Companion documentation
│   ├── AcademiQ_02_Preprocessing.ipynb         # Stage 2: Leakage-Safe Preprocessing
│   ├── AcademiQ_02_Preprocessing.md            # Companion documentation
│   ├── AcademiQ_03_Regression.ipynb            # Stage 3: Score Prediction Modeling
│   ├── AcademiQ_03_Regression.md               # Companion documentation
│   ├── AcademiQ_04_Feature_Analysis.ipynb      # Stage 4: Student Factor Analysis
│   ├── AcademiQ_04_Feature_Analysis.md         # Companion documentation
│   ├── AcademiQ_05_Final_Integration.ipynb     # Stage 5: Synthesis & Integration
│   └── AcademiQ_05_Final_Integration.md        # Companion documentation
├── data/
│   ├── raw/
│   │   ├── Student Performance Factors.csv     # Raw Kaggle Dataset (6,607 rows)
│   │   └── Student Performance Factors.md      # Companion documentation
│   └── processed/
│       ├── AcademiQ_Train_Cleaned.csv          # Cleaned training data (5,285 rows)
│       ├── AcademiQ_Train_Cleaned.md           # Companion documentation
│       ├── AcademiQ_Test_Cleaned.csv           # Cleaned holdout test data (1,322 rows)
│       ├── AcademiQ_Test_Cleaned.md            # Companion documentation
│       ├── AcademiQ_Train_Preprocessed.csv     # Scaled/encoded training matrix
│       ├── AcademiQ_Train_Preprocessed.md      # Companion documentation
│       ├── AcademiQ_Test_Preprocessed.csv      # Scaled/encoded test matrix
│       └── AcademiQ_Test_Preprocessed.md       # Companion documentation
├── models/
│   ├── AcademiQ_Simple_Linear_Regression.pkl   # Baseline univariate regression model
│   ├── AcademiQ_Simple_Linear_Regression.md    # Companion documentation
│   ├── AcademiQ_Multiple_Linear_Regression.pkl # Champion MLR pipeline (preprocessor + model)
│   ├── AcademiQ_Multiple_Linear_Regression.md  # Companion documentation
│   ├── AcademiQ_Polynomial_Regression.pkl      # Degree-2 polynomial regression model
│   ├── AcademiQ_Polynomial_Regression.md       # Companion documentation
│   ├── AcademiQ_Feature_Selector.pkl           # Fitted SelectKBest feature selector
│   └── AcademiQ_Feature_Selector.md            # Companion documentation
├── results/
│   ├── regression/
│   │   ├── AcademiQ_Regression_Test_Results.csv # Test evaluation metrics (MAE, RMSE, R²)
│   │   ├── AcademiQ_Regression_Test_Results.md  # Companion documentation
│   │   ├── AcademiQ_Regression_CV_Results.csv   # 5-fold cross-validation metrics
│   │   └── AcademiQ_Regression_CV_Results.md    # Companion documentation
│   ├── feature_analysis/
│   │   ├── AcademiQ_Feature_Selection_Results.csv # Univariate ANOVA F-scores & p-values
│   │   ├── AcademiQ_Feature_Selection_Results.md  # Companion documentation
│   │   ├── AcademiQ_Selected_Features.csv         # Top 10 ranked student features
│   │   └── AcademiQ_Selected_Features.md          # Companion documentation
│   └── final/
│       ├── AcademiQ_Final_Model_Summary.csv    # Consolidated master model scorecard
│       ├── AcademiQ_Final_Model_Summary.md     # Companion documentation
│       ├── AcademiQ_Regression_Comparison.png  # Multi-panel benchmark visualization figure
│       └── AcademiQ_Regression_Comparison.md   # Companion documentation
├── docs/
│   ├── AcademiQ_EDA_Summary.csv                # EDA summary profiling metrics
│   └── AcademiQ_EDA_Summary.md                 # Companion documentation
├── predict.py                                  # CLI Student Score Prediction Utility
├── predict.md                                  # Companion documentation
├── whatif.py                                   # CLI What-If Academic Scenario Analysis Utility
├── whatif.md                                   # Companion documentation
├── requirements.txt                            # Pinned Python package dependencies
├── requirements.md                             # Companion documentation
├── README.md                                   # Project documentation
└── .gitignore                                  # Git exclusion configuration
```

---

## Methodological Limitations & Ethical Considerations

1. **Prediction is Not Causation:** Statistical associations (such as high $F$-scores for attendance and study hours) quantify predictive correlation within observational historical data. They do **not** prove that manipulating a single variable in isolation will automatically produce an identical grade gain in a real-world classroom.
2. **Confounding Factors:** Unmeasured variables (e.g., student intrinsic motivation, family stability, individual learning pace, teacher effectiveness) can influence both student habits and exam scores simultaneously.
3. **Exploratory Guidance Only:** Predictions and what-if scenario outputs are intended as analytical counseling guidance to help students and mentors identify areas of academic focus, never as deterministic or punitive assessments.
