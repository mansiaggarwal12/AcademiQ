# Companion Explanation: AcademiQ_naive_bayes.pkl

## A. Academic Understanding: Generative Bayesian Modeling
- **What is Gaussian Naive Bayes?** A generative parametric classifier that applies Bayes' Theorem: $P(Y = k | \mathbf{x}) \propto P(Y = k) \prod_{j=1}^p P(x_j | Y = k)$. Continuous features are modeled via normal probability densities $\mathcal{N}(\mu_{jk}, \sigma_{jk}^2)$.
- **The "Naive" Independence Assumption:** Assumes all features are conditionally independent given the class label: $\text{Cov}(x_a, x_b | Y) = 0$.
- **Failure on Correlated Features:** When features correlate (e.g., Attendance, Study Hours, Tutoring), Naive Bayes double-counts redundant evidence, inflating posterior probabilities toward 0 or 1 and distorting rankings.

## B. Implementation in AcademiQ
- **Stored Object Type:** A serialized scikit-learn `Pipeline` containing `ColumnTransformer` and `GaussianNB()`.
- **Expected Input:** A Pandas DataFrame containing all 19 raw student features.
- **Verified Benchmark Performance (Test $n = 1,322$):**
  - Accuracy = $63.39\%$ (`0.633888`)
  - Macro F1 = $0.6175$ (`0.617524`)
  - Rank 5 of 6 classifiers.

## C. Significance in the Complete Project
- **Workflow Position:** Stage 04 artifact; generative probabilistic baseline.
- **Trained By:** `AcademiQ_04_Classification.ipynb`.
- **Consumed By:** `AcademiQ_07_Final_Integration.ipynb`.
- **Viva Defense Insight:** Highlights the cost of the naive independence assumption. Because real student behaviors are deeply interconnected, Naive Bayes lagged behind discriminative Logistic Regression ($95.08\%$) by over $31\%$.\n