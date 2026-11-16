"""
AcademiQ - Streamlit Frontend
Multi-page academic performance prediction and analysis dashboard.
Run with: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH   = PROJECT_ROOT / "models" / "AcademiQ_Multiple_Linear_Regression.pkl"

# --- Feature Config ---
FEATURE_COLUMNS = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
    "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender",
]
CATEGORICAL_OPTIONS = {
    "Parental_Involvement":    ["Low", "Medium", "High"],
    "Access_to_Resources":     ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["No", "Yes"],
    "Motivation_Level":        ["Low", "Medium", "High"],
    "Internet_Access":         ["No", "Yes"],
    "Family_Income":           ["Low", "Medium", "High"],
    "Teacher_Quality":         ["Low", "Medium", "High"],
    "School_Type":             ["Public", "Private"],
    "Peer_Influence":          ["Negative", "Neutral", "Positive"],
    "Learning_Disabilities":   ["No", "Yes"],
    "Parental_Education_Level":["High School", "College", "Postgraduate"],
    "Distance_from_Home":      ["Near", "Moderate", "Far"],
    "Gender":                  ["Male", "Female"],
}

# --- Page Config ---
st.set_page_config(
    page_title="AcademiQ",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# --- Helper: Tier Logic ---
def get_tier(score: float):
    if score <= 66.0:
        return "Developing", "#E74C3C", "Student may benefit from targeted academic support."
    elif score <= 69.0:
        return "Proficient", "#F39C12", "Student is performing at expected baseline standards."
    else:
        return "Advanced", "#27AE60", "Student demonstrates strong academic mastery."

# --- Sidebar Navigation ---
st.sidebar.markdown("# 🎓 AcademiQ")
st.sidebar.markdown("*Student Performance Prediction*")
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Data & EDA", "🤖 Model Results", "🏆 Feature Analysis",
     "🎯 Predict Score", "🔄 What-If Analysis"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption("Champion Model: Multiple Linear Regression\nTest R² = 0.7699 · MAE = 0.45 pts")

# =============================================================================
# PAGE 1: HOME
# =============================================================================
if page == "🏠 Home":
    st.title("🎓 AcademiQ")
    st.subheader("Student Academic Performance Prediction & Analysis")
    st.markdown(
        "AcademiQ is a Machine Learning project that **predicts students' final exam scores** (0-100) "
        "and analyses which student factors most associate with academic performance."
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📚 Total Students",  "6,607")
    col2.metric("🔢 Features",        "19")
    col3.metric("🏋️ Training Set",    "5,285")
    col4.metric("🧪 Test Set",        "1,322")

    st.divider()
    st.subheader("Pipeline Overview")
    st.markdown("""
| Stage | Notebook | Description |
|---|---|---|
| 1 | AcademiQ_01_EDA | Exploratory Data Analysis & Anomaly Detection |
| 2 | AcademiQ_02_Preprocessing | Leakage-Safe Preprocessing & 80:20 Split |
| 3 | AcademiQ_03_Regression | SLR · MLR · Polynomial Regression Modeling |
| 4 | AcademiQ_04_Feature_Analysis | SelectKBest + MLR Coefficient Analysis |
| 5 | AcademiQ_05_Final_Integration | Synthesis, Benchmarks & Inference |
""")
    st.divider()
    st.subheader("Performance Tiers")
    tc1, tc2, tc3 = st.columns(3)
    tc1.error("**Developing** — Score ≤ 66\nMay benefit from academic support")
    tc2.warning("**Proficient** — 66 < Score ≤ 69\nMeeting baseline standards")
    tc3.success("**Advanced** — Score > 69\nDemonstrating strong mastery")

# =============================================================================
# PAGE 2: DATA & EDA
# =============================================================================
elif page == "📊 Data & EDA":
    st.title("📊 Data Exploration & EDA")

    @st.cache_data
    def load_raw():
        for p in [
            PROJECT_ROOT / "data" / "raw" / "Student Performance Factors.csv",
            PROJECT_ROOT / "notebooks" / "Student Performance Factors.csv",
        ]:
            if p.exists():
                return pd.read_csv(p)
        raise FileNotFoundError("Raw dataset not found")

    try:
        df = load_raw()
        st.success(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.divider()
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("Mean Score",   f"{df['Exam_Score'].mean():.2f}")
        col_stat2.metric("Median Score", f"{df['Exam_Score'].median():.2f}")
        col_stat3.metric("Std Dev",      f"{df['Exam_Score'].std():.2f}")
        col_stat4.metric("Score Range",  f"{df['Exam_Score'].min():.0f} - {df['Exam_Score'].max():.0f}")

        st.divider()
        st.subheader("Exam Score Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df["Exam_Score"], bins=30, color="#4C72B0", edgecolor="white", linewidth=0.6)
        ax.axvline(df["Exam_Score"].mean(),   color="#DD4444", linestyle="--", linewidth=1.8,
                   label=f"Mean = {df['Exam_Score'].mean():.2f}")
        ax.axvline(df["Exam_Score"].median(), color="#228B22", linestyle=":",  linewidth=1.8,
                   label=f"Median = {df['Exam_Score'].median():.2f}")
        ax.set_xlabel("Exam Score (0-100)", fontsize=12)
        ax.set_ylabel("Number of Students", fontsize=12)
        ax.set_title("Distribution of Exam Scores", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("Correlation Heatmap (Numerical Features)")
        try:
            import seaborn as sns
            num_cols = df.select_dtypes(include="number").columns.tolist()
            corr = df[num_cols].corr()
            fig2, ax2 = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                        linewidths=0.5, linecolor="white", ax=ax2, annot_kws={"size": 10})
            ax2.set_title("Correlation Heatmap - Numerical Features", fontsize=13, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
        except ImportError:
            st.info("Install seaborn (pip install seaborn) to view the correlation heatmap.")

        st.divider()
        st.subheader("Missing Values Summary")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Feature", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        st.dataframe(missing[missing["Missing Count"] > 0], use_container_width=True, hide_index=True)
        if missing["Missing Count"].sum() == 0:
            st.success("No missing values found in the raw dataset after cleaning.")

    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")

# =============================================================================
# PAGE 3: MODEL RESULTS
# =============================================================================
elif page == "🤖 Model Results":
    st.title("🤖 Model Comparison & Results")

    st.subheader("Benchmark Performance (Test Set, n = 1,322)")
    bench = pd.DataFrame({
        "Model": ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression (Deg-2)"],
        "Features": ["Hours_Studied Only", "All 19 Factors", "All 19 + Interactions"],
        "Test R2": [0.2320, 0.7699, 0.7684],
        "Test MAE (pts)": [2.4476, 0.4498, 0.4660],
        "Test RMSE (pts)": [3.2948, 1.8033, 1.8093],
        "Status": ["Baseline (Underfitting)", "CHAMPION MODEL", "Capacity Plateau"],
    })
    st.dataframe(bench, use_container_width=True, hide_index=True)

    st.divider()
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Champion R²",   "0.7699", "+0.5379 vs Baseline")
    mc2.metric("Champion MAE",  "0.45 pts", "-2.00 pts vs Baseline")
    mc3.metric("Champion RMSE", "1.80 pts", "-1.49 pts vs Baseline")

    st.divider()
    st.subheader("5-Fold Cross-Validation Results")
    cv_path = PROJECT_ROOT / "results" / "regression" / "AcademiQ_Regression_CV_Results.csv"
    if cv_path.exists():
        cv_df = pd.read_csv(cv_path)
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        st.info("Low variance across CV folds confirms the model generalises well and is not overfitted.")
    else:
        st.warning("CV results file not found. Run notebook 03 first.")

    st.divider()
    st.subheader("Actual vs Predicted — MLR (Test Set)")
    img_avp = PROJECT_ROOT / "results" / "final" / "AcademiQ_Actual_vs_Predicted.png"
    if img_avp.exists():
        st.image(str(img_avp), use_column_width=True)
    else:
        st.info("Run the updated AcademiQ_03_Regression notebook to generate this chart.")

    st.divider()
    st.subheader("Why MLR Was Selected")
    st.markdown("""
1. **Superior Generalisation**: Highest R² (0.7699) and lowest MAE (0.45 pts) on the held-out test set.
2. **Occam's Razor**: Polynomial Regression adds 500+ interaction parameters but fails to improve R² (0.7684) while slightly worsening MAE.
3. **Full Interpretability**: Coefficients directly quantify each student factor's contribution — explainable to educators, parents, and students.
""")

# =============================================================================
# PAGE 4: FEATURE ANALYSIS
# =============================================================================
elif page == "🏆 Feature Analysis":
    st.title("🏆 Student Factor Analysis")

    feat_path = PROJECT_ROOT / "results" / "feature_analysis" / "AcademiQ_Feature_Selection_Results.csv"
    if feat_path.exists():
        feat_df = pd.read_csv(feat_path)
        st.subheader("SelectKBest F-Scores (Univariate ANOVA)")
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Feature Importance Bar Chart")

        # Try to plot from CSV
        try:
            fc = feat_df.columns[0]
            sc = feat_df.columns[1] if len(feat_df.columns) > 1 else None
            if sc:
                top = feat_df.sort_values(by=sc, ascending=False).head(15)
                colors = ["#4C72B0" if i < 4 else "#7FB3D3" for i in range(len(top))]
                fig, ax = plt.subplots(figsize=(11, 6))
                bars = ax.barh(top[fc].astype(str), top[sc].astype(float), color=colors,
                               edgecolor="white", linewidth=0.5)
                ax.invert_yaxis()
                ax.set_xlabel("ANOVA F-Score", fontsize=12)
                ax.set_title("Top Student Factors by Predictive Strength\n(SelectKBest with F-Regression)",
                             fontsize=13, fontweight="bold")
                max_val = top[sc].astype(float).max()
                for bar, val in zip(bars, top[sc].astype(float)):
                    ax.text(bar.get_width() + max_val * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}", va="center", fontsize=9)
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor="#4C72B0", label="Top 4 (Dominant Predictors)"),
                    Patch(facecolor="#7FB3D3", label="Secondary Predictors"),
                ]
                ax.legend(handles=legend_elements, fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        except Exception as e:
            img_fi = PROJECT_ROOT / "results" / "final" / "AcademiQ_Feature_Importance.png"
            if img_fi.exists():
                st.image(str(img_fi), use_column_width=True)
            else:
                st.error(f"Could not plot feature importance: {e}")

        st.divider()
        st.subheader("Key Findings")
        st.markdown("""
| Rank | Factor | F-Score | Interpretation |
|---|---|---|---|
| 1 | Attendance | 2666.48 | Strongest single predictor of exam outcome |
| 2 | Hours_Studied | 1244.44 | Structured study time is second-highest impact |
| 3 | Previous_Scores | 162.90 | Past performance signals future readiness |
| 4 | Tutoring_Sessions | 131.86 | Supplemental instruction adds measurable value |
""")
        st.warning("These are correlational associations, not causal proof. Unmeasured confounders may influence both habits and scores simultaneously.")

    else:
        st.error("Feature results file not found. Run notebook 04 first.")

# =============================================================================
# PAGE 5: PREDICT SCORE
# =============================================================================
elif page == "🎯 Predict Score":
    st.title("🎯 Predict Student Exam Score")
    st.markdown("Enter a student's details to predict their exam score using the champion MLR model.")

    try:
        pipeline = load_model()

        with st.form("predict_form"):
            st.subheader("Academic Habits")
            c1, c2 = st.columns(2)
            hours      = c1.slider("Hours Studied per week", 1, 43, 20)
            attendance = c2.slider("Attendance (%)", 60, 100, 75)
            prev       = c1.slider("Previous Exam Score", 50, 100, 75)
            tutoring   = c2.slider("Tutoring Sessions per month", 0, 8, 1)

            st.subheader("Lifestyle & Health")
            l1, l2 = st.columns(2)
            sleep    = l1.slider("Sleep Hours per night", 4, 10, 7)
            physical = l2.slider("Physical Activity (hrs/week)", 0, 6, 3)
            extra    = l1.selectbox("Extracurricular Activities", CATEGORICAL_OPTIONS["Extracurricular_Activities"])

            st.subheader("Contextual & Demographic")
            d1, d2 = st.columns(2)
            parental_inv   = d1.selectbox("Parental Involvement",    CATEGORICAL_OPTIONS["Parental_Involvement"])
            access         = d2.selectbox("Access to Resources",     CATEGORICAL_OPTIONS["Access_to_Resources"])
            motivation     = d1.selectbox("Motivation Level",        CATEGORICAL_OPTIONS["Motivation_Level"])
            internet       = d2.selectbox("Internet Access",         CATEGORICAL_OPTIONS["Internet_Access"])
            income         = d1.selectbox("Family Income",           CATEGORICAL_OPTIONS["Family_Income"])
            teacher_q      = d2.selectbox("Teacher Quality",         CATEGORICAL_OPTIONS["Teacher_Quality"])
            school_type    = d1.selectbox("School Type",             CATEGORICAL_OPTIONS["School_Type"])
            peer           = d2.selectbox("Peer Influence",          CATEGORICAL_OPTIONS["Peer_Influence"])
            disability     = d1.selectbox("Learning Disabilities",   CATEGORICAL_OPTIONS["Learning_Disabilities"])
            parent_edu     = d2.selectbox("Parental Education Level",CATEGORICAL_OPTIONS["Parental_Education_Level"])
            distance       = d1.selectbox("Distance from Home",      CATEGORICAL_OPTIONS["Distance_from_Home"])
            gender         = d2.selectbox("Gender",                  CATEGORICAL_OPTIONS["Gender"])

            submitted = st.form_submit_button("Predict Exam Score", use_container_width=True)

        if submitted:
            input_data = {
                "Hours_Studied": hours, "Attendance": attendance,
                "Parental_Involvement": parental_inv, "Access_to_Resources": access,
                "Extracurricular_Activities": extra, "Sleep_Hours": sleep,
                "Previous_Scores": prev, "Motivation_Level": motivation,
                "Internet_Access": internet, "Tutoring_Sessions": tutoring,
                "Family_Income": income, "Teacher_Quality": teacher_q,
                "School_Type": school_type, "Peer_Influence": peer,
                "Physical_Activity": physical, "Learning_Disabilities": disability,
                "Parental_Education_Level": parent_edu, "Distance_from_Home": distance,
                "Gender": gender,
            }
            input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
            score = float(pipeline.predict(input_df)[0])
            tier_label, tier_color, tier_desc = get_tier(score)

            st.divider()
            st.subheader("Prediction Result")
            r1, r2 = st.columns(2)
            r1.metric("Predicted Exam Score", f"{score:.2f} / 100")
            r2.metric("Performance Tier", tier_label)

            # Score bar gauge
            fig, ax = plt.subplots(figsize=(9, 1.8))
            ax.barh(["Score"], [100], color="#EEEEEE", height=0.4)
            bar_color = "#E74C3C" if score <= 66 else ("#F39C12" if score <= 69 else "#27AE60")
            ax.barh(["Score"], [min(score, 100)], color=bar_color, height=0.4)
            ax.axvline(66, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.7, label="Developing (66)")
            ax.axvline(69, color="#F39C12", linestyle="--", linewidth=1.2, alpha=0.7, label="Proficient (69)")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score (0 - 100)", fontsize=10)
            ax.set_title(f"Predicted Score: {score:.2f} — {tier_label}", fontsize=11, fontweight="bold")
            ax.text(min(score, 97) + 0.5, 0, f"{score:.1f}", va="center", fontsize=10,
                    fontweight="bold", color=bar_color)
            ax.legend(fontsize=9, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            if tier_label == "Advanced":
                st.success(f"**{tier_label}**: {tier_desc}")
            elif tier_label == "Proficient":
                st.warning(f"**{tier_label}**: {tier_desc}")
            else:
                st.error(f"**{tier_label}**: {tier_desc}")

            st.caption("This is a statistical model estimate for academic counselling purposes only, not an official grade.")

    except FileNotFoundError:
        st.error("Model not found at models/AcademiQ_Multiple_Linear_Regression.pkl. Run the training notebooks first.")

# =============================================================================
# PAGE 6: WHAT-IF ANALYSIS
# =============================================================================
elif page == "🔄 What-If Analysis":
    st.title("🔄 What-If Scenario Analysis")
    st.markdown(
        "Explore how changing **one student factor** while holding all others constant "
        "shifts the predicted exam score (ceteris paribus)."
    )

    try:
        pipeline = load_model()

        st.subheader("Step 1 — Set Baseline Student Profile")
        with st.expander("Configure Baseline Student", expanded=True):
            b1, b2 = st.columns(2)
            b_hours    = b1.slider("Hours Studied / week",    1, 43, 20, key="b_hours")
            b_attend   = b2.slider("Attendance (%)",          60, 100, 75, key="b_attend")
            b_prev     = b1.slider("Previous Score",          50, 100, 75, key="b_prev")
            b_tutor    = b2.slider("Tutoring Sessions/month", 0, 8, 1,     key="b_tutor")
            b_sleep    = b1.slider("Sleep Hours/night",       4, 10, 7,    key="b_sleep")
            b_physical = b2.slider("Physical Activity hrs/week", 0, 6, 3, key="b_physical")
            b_par_inv  = b1.selectbox("Parental Involvement",    CATEGORICAL_OPTIONS["Parental_Involvement"],   index=1, key="b_par_inv")
            b_access   = b2.selectbox("Access to Resources",     CATEGORICAL_OPTIONS["Access_to_Resources"],    index=1, key="b_access")
            b_motiv    = b1.selectbox("Motivation Level",        CATEGORICAL_OPTIONS["Motivation_Level"],        index=1, key="b_motiv")
            b_internet = b2.selectbox("Internet Access",         CATEGORICAL_OPTIONS["Internet_Access"],         index=1, key="b_internet")
            b_income   = b1.selectbox("Family Income",           CATEGORICAL_OPTIONS["Family_Income"],           index=1, key="b_income")
            b_teacher  = b2.selectbox("Teacher Quality",         CATEGORICAL_OPTIONS["Teacher_Quality"],         index=1, key="b_teacher")
            b_school   = b1.selectbox("School Type",             CATEGORICAL_OPTIONS["School_Type"],             index=0, key="b_school")
            b_peer     = b2.selectbox("Peer Influence",          CATEGORICAL_OPTIONS["Peer_Influence"],          index=1, key="b_peer")
            b_extra    = b1.selectbox("Extracurricular",         CATEGORICAL_OPTIONS["Extracurricular_Activities"], index=1, key="b_extra")
            b_disab    = b2.selectbox("Learning Disabilities",   CATEGORICAL_OPTIONS["Learning_Disabilities"],   index=0, key="b_disab")
            b_par_edu  = b1.selectbox("Parental Education",      CATEGORICAL_OPTIONS["Parental_Education_Level"],index=0, key="b_par_edu")
            b_dist     = b2.selectbox("Distance from Home",      CATEGORICAL_OPTIONS["Distance_from_Home"],      index=0, key="b_dist")
            b_gender   = b1.selectbox("Gender",                  CATEGORICAL_OPTIONS["Gender"],                  index=1, key="b_gender")

        baseline = {
            "Hours_Studied": b_hours, "Attendance": b_attend,
            "Parental_Involvement": b_par_inv, "Access_to_Resources": b_access,
            "Extracurricular_Activities": b_extra, "Sleep_Hours": b_sleep,
            "Previous_Scores": b_prev, "Motivation_Level": b_motiv,
            "Internet_Access": b_internet, "Tutoring_Sessions": b_tutor,
            "Family_Income": b_income, "Teacher_Quality": b_teacher,
            "School_Type": b_school, "Peer_Influence": b_peer,
            "Physical_Activity": b_physical, "Learning_Disabilities": b_disab,
            "Parental_Education_Level": b_par_edu, "Distance_from_Home": b_dist,
            "Gender": b_gender,
        }
        base_df = pd.DataFrame([baseline], columns=FEATURE_COLUMNS)
        base_score = float(pipeline.predict(base_df)[0])
        base_tier, _, _ = get_tier(base_score)
        st.metric("Baseline Predicted Score", f"{base_score:.2f} / 100", help=f"Tier: {base_tier}")

        st.divider()
        st.subheader("Step 2 — Choose a Factor to Change")

        NUMERICAL_FEATURES = {
            "Attendance (%)":             ("Attendance",        60,  100, False),
            "Hours Studied/week":          ("Hours_Studied",     1,   43,  False),
            "Previous Score":              ("Previous_Scores",   50,  100, False),
            "Tutoring Sessions/month":     ("Tutoring_Sessions", 0,   8,   True),
            "Sleep Hours/night":           ("Sleep_Hours",       4,   10,  False),
            "Physical Activity hrs/week":  ("Physical_Activity", 0,   6,   False),
        }
        CATEGORICAL_FEATURES_MAP = {
            "Parental Involvement":    "Parental_Involvement",
            "Access to Resources":     "Access_to_Resources",
            "Motivation Level":        "Motivation_Level",
            "Internet Access":         "Internet_Access",
            "Family Income":           "Family_Income",
            "Teacher Quality":         "Teacher_Quality",
            "School Type":             "School_Type",
            "Peer Influence":          "Peer_Influence",
            "Extracurricular":         "Extracurricular_Activities",
            "Learning Disabilities":   "Learning_Disabilities",
            "Parental Education":      "Parental_Education_Level",
            "Distance from Home":      "Distance_from_Home",
            "Gender":                  "Gender",
        }

        all_opts = list(NUMERICAL_FEATURES.keys()) + list(CATEGORICAL_FEATURES_MAP.keys())
        selected_label = st.selectbox("Select factor to modify", all_opts)
        scenario = baseline.copy()

        if selected_label in NUMERICAL_FEATURES:
            feat_key, mn, mx, is_int = NUMERICAL_FEATURES[selected_label]
            current_val = baseline[feat_key]
            step = 1 if is_int else 1
            new_val = st.slider(
                f"New value for {selected_label} (current: {current_val})",
                mn, mx, int(current_val) if is_int else current_val, step=step,
            )
            scenario[feat_key] = new_val
        else:
            feat_key = CATEGORICAL_FEATURES_MAP[selected_label]
            options  = CATEGORICAL_OPTIONS[feat_key]
            current_val = baseline[feat_key]
            curr_idx = options.index(current_val) if current_val in options else 0
            new_val = st.selectbox(
                f"New value for {selected_label} (current: {current_val})", options, index=curr_idx
            )
            scenario[feat_key] = new_val

        scen_df    = pd.DataFrame([scenario], columns=FEATURE_COLUMNS)
        scen_score = float(pipeline.predict(scen_df)[0])
        scen_tier, _, _ = get_tier(scen_score)
        delta = scen_score - base_score
        sign  = "+" if delta >= 0 else ""

        st.divider()
        st.subheader("Scenario Result")
        res1, res2, res3 = st.columns(3)
        res1.metric("Baseline Score", f"{base_score:.2f}", help=base_tier)
        res2.metric("Scenario Score", f"{scen_score:.2f}", delta=f"{sign}{delta:.2f} pts")
        tier_change_label = f"{base_tier} → {scen_tier}" if base_tier != scen_tier else f"Unchanged ({base_tier})"
        res3.metric("Tier", tier_change_label)

        # Comparison bar chart
        fig, ax = plt.subplots(figsize=(7, 3))
        vals = [base_score, scen_score]
        bar_colors = []
        for s in vals:
            bar_colors.append("#E74C3C" if s <= 66 else ("#F39C12" if s <= 69 else "#27AE60"))
        bars = ax.bar(["Baseline", "Scenario"], vals, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.4)
        ax.axhline(66, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.5, label="Developing threshold (66)")
        ax.axhline(69, color="#F39C12", linestyle="--", linewidth=1, alpha=0.5, label="Proficient threshold (69)")
        y_min = max(0,   min(vals) - 5)
        y_max = min(100, max(vals) + 5)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Predicted Exam Score", fontsize=11)
        ax.set_title(f"What-If Impact: {selected_label} changed to {new_val}", fontsize=12, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.warning(
            "**Non-Causal Disclaimer**: The predicted shift reflects a statistical association in observational data. "
            "It does NOT guarantee that changing this factor in real life will produce an identical improvement. "
            "Unmeasured confounders (personal motivation, school environment) also influence real-world performance."
        )

    except FileNotFoundError:
        st.error("Model not found. Run the training notebooks first.")
