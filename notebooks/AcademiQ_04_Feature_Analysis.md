# Companion Explanation: AcademiQ_04_Feature_Analysis.ipynb

## 1. Academic Understanding: Univariate Feature Selection and Statistical Factor Analysis

### What Feature Selection Means Academically
In supervised machine learning, **feature selection** is the process of identifying a subset of the most relevant features to include in an analytical investigation. Rather than transforming or projecting the feature space into abstract coordinates (as in PCA), feature selection retains the original, interpretable identity of each student attribute.

### How `SelectKBest` and `f_regression` Work
1. **Univariate Linear Hypothesis Testing:** The `f_regression` scoring function tests the individual linear relationship between each input predictor $X_j$ and the continuous target $y$ (`Exam_Score`). It computes the Pearson correlation $r(X_j, y)$ and converts it into an ANOVA $F$-statistic:
   $$F = \frac{r^2}{1 - r^2} \cdot (n - 2)$$
   where $n$ is the number of training observations ($5,285$).
2. **Ranking by Explanatory Strength:** A larger $F$-score (and corresponding $p$-value approaching $0$) indicates that the variable shares a strong linear association with the examination score when examined individually.
3. **Difference from Multivariate Regression:** Univariate scoring evaluates each feature in isolation, whereas multivariate regression (MLR) estimates the partial effect of each feature while controlling for all other variables simultaneously. Comparing both yields a complete picture of student factor relationships.

### Important Limitations: Predictive Association vs. Causal Proof
- **Correlation is Not Causation:** A high $F$-score for `Attendance` or `Hours_Studied` demonstrates strong predictive alignment within this observational cohort. It **does not prove** that artificially raising attendance will automatically cause a deterministic score increase in a real-world classroom.
- **Confounding Variables:** Unobserved student traits (e.g., intrinsic motivation, personal interest, prior foundation) may simultaneously drive both high attendance and high marks. Statistical models estimate associations, not intervention guarantees.

---

## 2. Implementation in AcademiQ

### Analytical Workflow
`AcademiQ_04_Feature_Analysis.ipynb` executes the following steps:
1. **Data Loading:** Loads `AcademiQ_Train_Cleaned.csv` and `AcademiQ_Test_Cleaned.csv` ($5,285$ train, $1,322$ test).
2. **Preprocessing Pipeline:** Applies `StandardScaler` to numerical inputs and `OneHotEncoder(handle_unknown="ignore")` to categorical features, generating a standardized 43-column design matrix.
3. **`SelectKBest` Scoring:** Fits `SelectKBest(score_func=f_regression, k=10)` strictly on the training partition to avoid data leakage.
4. **Empirical Top 10 Features:**
   - **`Attendance`:** $F = 2666.48$ ($p \approx 0.0$) — Primary linear correlate.
   - **`Hours_Studied`:** $F = 1244.44$ ($p \approx 0.0$) — Second strongest individual predictor.
   - **`Previous_Scores`:** $F = 162.90$ ($p = 9.16 \times 10^{-37}$).
   - **`Tutoring_Sessions`:** $F = 131.86$ ($p = 3.65 \times 10^{-30}$).
   - **`Access_to_Resources (High)`:** $F = 119.58$.
   - **`Parental_Involvement (High)`:** $F = 107.67$.
   - **`Access_to_Resources (Low)`:** $F = 97.06$.
   - **`Parental_Involvement (Low)`:** $F = 63.02$.
   - **`Parental_Education_Level (Postgraduate)`:** $F = 54.58$.
   - **`Learning_Disabilities (Yes)`:** $F = 44.74$.
5. **Artifact Outputs:**
   - Results: `results/feature_analysis/AcademiQ_Feature_Selection_Results.csv`
   - Top Features: `results/feature_analysis/AcademiQ_Selected_Features.csv`
   - Fitted Estimator: `models/AcademiQ_Feature_Selector.pkl`

---

## 3. Significance in the Complete Project & Viva Defense

### Role in the Simplified AcademiQ Pipeline
In our simplified, student-focused framework, this notebook directly answers the core research question: **"Which student habits and environmental factors are most strongly associated with academic performance?"**
- It confirms that while `Hours_Studied` is vital, `Attendance` is an even stronger correlate, and contextual factors (`Previous_Scores`, `Parental_Involvement`, `Access_to_Resources`) add meaningful predictive value.
- This finding motivates our **What-if Scenario Analysis**, allowing students and educators to explore how adjusting key numerical factors (`Hours_Studied`, `Attendance`) shifts model predictions.

### Sample Viva Defense Q&A
- **Q: Why did you choose SelectKBest with f_regression instead of PCA?**
  - **A:** PCA transforms features into abstract orthogonal components (linear combinations of all 43 features) that cannot be directly interpreted as actionable student habits. In contrast, `SelectKBest` preserves feature identity, allowing us to report exactly which student-related factors (e.g., attendance and study hours) have the strongest statistical association with exam outcomes.
- **Q: Does an F-score of 2666.48 mean that improving attendance by 10% will raise exam scores by a specific amount?**
  - **A:** No. An F-score indicates statistical significance and strength of linear correlation within observational data, not a causal guarantee. While our MLR model predicts an increase based on historical patterns, real-world educational interventions involve complex unmeasured confounders like student engagement and teaching quality.

