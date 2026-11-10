# Companion Explanation: requirements.txt

## A. Academic Understanding: Dependency Management & Reproducibility
- **What is a Dependency File?** In software engineering and empirical machine learning, a dependency manifest records the exact external libraries and version constraints required to execute a project's codebase.
- **The Core Problem it Solves:** The "Works on my machine" dilemma. As open-source ML libraries (such as scikit-learn or pandas) evolve, default parameters, underlying solvers, and class APIs frequently change. Without pinned dependencies, running the same notebook across different environments can cause execution crashes, numerical drift in gradient optimization, or unpickling failures.
- **Reproducibility in Machine Learning:** True scientific reproducibility requires two pillars: deterministic algorithmic execution (`random_state`) and an identical computational environment. `requirements.txt` establishes the second pillar.

## B. Implementation in AcademiQ
- **Role in the Project:** Pinpoints the minimal, clean set of libraries needed to execute all 7 notebooks, load saved `.pkl` model artifacts, and run the `predict.py` CLI utility.
- **Key Packages Specified:**
  - `pandas`: Tabular data manipulation, DataFrame slicing, and CSV serialization.
  - `numpy`: Multi-dimensional array operations and linear algebra routines.
  - `scikit-learn`: Estimators, preprocessing transformers (`StandardScaler`, `OneHotEncoder`), pipeline abstractions, and cross-validation harnesses.
  - `matplotlib` & `seaborn`: Statistical visualizations (correlation heatmaps, box plots, grouped performance charts).
  - `joblib`: High-performance object serialization with internal memory-mapping for NumPy arrays.
- **Execution Command:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  pip install -r requirements.txt
  ```
- **Tested Environment:** Python 3.13.4 on Windows x86_64.

## C. Significance in the Complete Project
- **Workflow Position:** Entry point for any external evaluator, researcher, or team member setting up AcademiQ from scratch.
- **Dependencies:** Standard Python Package Index (PyPI).
- **Subsequent Consumers:** All 7 notebooks in `notebooks/` and `predict.py` depend entirely on these pinned packages.
- **What Breaks Without It:** Attempting to run the pipeline without these dependencies causes `ModuleNotFoundError`. Using mismatched major scikit-learn versions can break unpickling of the 16 serialized model artifacts in `models/`.
- **Viva Defense Insight:** Standard library modules (`math`, `os`, `sys`, `pathlib`) are intentionally omitted because they are distributed with Python core. Notebook execution environments (`jupyter`, `ipykernel`) belong to the local developer harness and are kept distinct from core project runtime dependencies.\n