# Companion Explanation: AcademiQ_04_Classification.ipynb

## A. Academic Understanding: Discrete Performance Triage & Model Families
- **Regression vs. Classification:** While regression predicts exact numerical marks, classification assigns instances to discrete qualitative buckets. In institutional management, administrators cannot track thousands of continuous numbers; they require operational triage to allocate limited tutoring and counseling resources to students who need them most.
- **Tertile Discretization:** Splitting a continuous distribution into three equal-frequency quantiles (tertiles) creates naturally balanced classes, avoiding artificial class-imbalance penalties during training.
- **Algorithm Families Evaluated:**
  - *Logistic Regression (Linear / Discriminative):* Models log-odds using the Softmax function; draws linear decision hyperplanes.
  - *Support Vector Machine (Kernel / Maximum Margin):* Finds the optimal separating hyperplane maximizing geometric margin in Hilbert space via an RBF kernel.
  - *Decision Trees & Random Forests (Non-Parametric / Axis-Aligned):* Partition space using recursive orthogonal step splits; Random Forest reduces tree variance via bootstrap aggregating (bagging) and random subspace feature sampling.
  - *Gaussian Naive Bayes (Generative / Probabilistic):* Inverts likelihoods using Bayes' Theorem assuming class-conditional feature independence.
  - *K-Nearest Neighbors (Instance-Based / Lazy Learning):* Predicts majority vote among the $K$ closest spatial neighbors in Euclidean space.

## B. Implementation in AcademiQ
- **Role in the Project:** Stage 04 — Evaluates and deploys multi-class early-warning triage models.
- **Target Category Formulation (Frozen on Training Tertiles):**
  - **Developing:** $	ext{Exam Score} \le 66.0$ (Bottom 33.3%, urgent intervention needed).
  - **Proficient:** $66.0 < 	ext{Exam Score} \le 69.0$ (Middle 33.3%, on-grade pacing).
  - **Advanced:** $	ext{Exam Score} > 69.0$ (Top 33.3%, honors enrichment).
- **Verified Benchmark Test Results ($n = 1,322$):**
  1. **Logistic Regression (Champion):** **Accuracy = 95.08%**, **Macro F1 = 0.9492**, Precision = $95.15\%$.
  2. **Support Vector Machine (RBF):** Accuracy = $86.16\%$, Macro F1 = $0.8565$.
  3. **Random Forest (100 trees):** Accuracy = $81.39\%$, Macro F1 = $0.8071$.
  4. **Decision Tree (CART):** Accuracy = $69.89\%$, Macro F1 = $0.6876$.
  5. **Gaussian Naive Bayes:** Accuracy = $63.39\%$, Macro F1 = $0.6175$.
  6. **K-Nearest Neighbors ($K=5$):** Accuracy = $52.12\%$, Macro F1 = $0.4934$.
- **Outputs Produced:** 6 serialized model pipelines in `models/` and metric CSVs in `results/classification/`.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 04 — Early-warning qualitative diagnostic module.
- **Preceding Dependencies:** Consumes cleaned train and test partitions from `AcademiQ_02_Preprocessing.ipynb`.
- **Subsequent Consumers:** Champion Logistic Regression model is integrated into the final evaluation in `AcademiQ_07_Final_Integration.ipynb`.
- **Why Did Linear Win Over Non-Linear?** Because the continuous `Exam_Score` is a linear combination of features, the category boundaries dividing Developing, Proficient, and Advanced are linear hyperplanes. Logistic Regression directly fits these linear hyperplanes, whereas tree models had to approximate them with jagged rectangular cuts and SVM had excess boundary curvature.\n