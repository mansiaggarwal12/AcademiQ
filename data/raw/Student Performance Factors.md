# Companion Explanation: Student Performance Factors.csv

## A. Academic Understanding: Raw Observational Data in Education
- **What is Raw Data?** Raw data is the primary observational record collected directly from surveys, school databases, or instrumentation prior to any cleaning, filtering, imputation, or transformation.
- **Tabular Data Representation:** Each row represents a single student observation; each column represents an attribute or variable. In educational data mining, tabular data combines quantitative behavioral metrics (hours, percentages) with qualitative socio-environmental indicators.
- **Immutability Principle:** In scientific research, raw data must be treated as strictly read-only and immutable. Cleaning transformations must never overwrite raw source files directly; they must output versioned, processed artifacts.

## B. Implementation in AcademiQ
- **Dataset Dimensions:** 6,607 student observations $\times$ 20 columns.
- **Target Variable:** `Exam_Score` (continuous grade percentage, raw range $[55, 101]$).
- **Features Breakdown (19 Attributes):**
  - *Numerical (6):* `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`.
  - *Categorical (13):* `Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities`, `Motivation_Level`, `Internet_Access`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Learning_Disabilities`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`.
- **Known Real-World Flaws:**
  - Contains missing values across 4 columns (`Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home`, minor numericals).
  - Contains exactly one physical outlier record (`Exam_Score = 101`).

## C. Significance in the Complete Project
- **Workflow Position:** Root input of the entire AcademiQ project.
- **Downstream Consumers:** Ingested exclusively by `AcademiQ_01_EDA.ipynb` (for statistical inspection) and `AcademiQ_02_Preprocessing.ipynb` (for cleaning).
- **What Breaks Without It:** Without this raw dataset, the entire project has no foundation to analyze, train, or validate upon.\n