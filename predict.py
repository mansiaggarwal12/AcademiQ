"""
AcademiQ - Student Exam Score Prediction Utility
=================================================
This command-line utility loads the trained AcademiQ Multiple Linear Regression
pipeline and predicts a student's final Exam_Score based on 19 input features.

Usage:
    python predict.py
"""

import sys
import warnings
from pathlib import Path

# Suppress version mismatch warnings during unpickling for clean user output
warnings.filterwarnings("ignore")

import joblib
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
# Path to the trained Multiple Linear Regression pipeline artifact
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "AcademiQ_Multiple_Linear_Regression.pkl"

# Exact 19 feature names expected by the trained pipeline in exact column order
FEATURE_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]

# Valid choices for all 13 categorical features (matching the original dataset)
CATEGORICAL_OPTIONS = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["No", "Yes"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Internet_Access": ["No", "Yes"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "School_Type": ["Public", "Private"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Learning_Disabilities": ["No", "Yes"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
    "Gender": ["Male", "Female"],
}


# -----------------------------------------------------------------------------
# Helper Functions for User Input & Validation
# -----------------------------------------------------------------------------
def load_trained_pipeline(model_path: Path):
    """
    Load the trained Multiple Linear Regression pipeline from disk.
    Does NOT retrain or modify the model.
    """
    if not model_path.exists():
        print(f"\n[ERROR] Model artifact not found at:\n  {model_path}")
        print("Please ensure the model file exists before running predict.py.")
        sys.exit(1)

    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as exc:
        print(f"\n[ERROR] Failed to load model artifact: {exc}")
        sys.exit(1)


def prompt_numerical(
    feature_name: str,
    prompt_label: str,
    min_val: float,
    max_val: float,
    is_integer: bool = False,
):
    """
    Prompt the user for a numerical feature, validate bounds and type.
    Allows user to type 'q' or 'exit' to quit.
    """
    type_desc = "integer" if is_integer else "number"
    prompt_str = f"  {prompt_label} (Range: {min_val} to {max_val}): "

    while True:
        try:
            raw_input = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting prediction utility.")
            sys.exit(0)

        # Check for exit command
        if raw_input.lower() in ["q", "quit", "exit"]:
            print("\nExiting prediction utility. Goodbye!")
            sys.exit(0)

        if not raw_input:
            print(f"    [!] Input cannot be empty. Please enter a valid {type_desc}.")
            continue

        try:
            val = int(raw_input) if is_integer else float(raw_input)
        except ValueError:
            print(f"    [!] Invalid input '{raw_input}'. Please enter a valid {type_desc}.")
            continue

        if val < min_val or val > max_val:
            print(f"    [!] Value out of bounds ({val}). Must be between {min_val} and {max_val}.")
            continue

        return val


def prompt_categorical(feature_name: str, prompt_label: str, options: list):
    """
    Prompt the user for a categorical feature with clear numbered choices.
    Accepts either the option number (e.g., 1, 2) or the exact name case-insensitively.
    Allows user to type 'q' or 'exit' to quit.
    """
    # Create numbered representation for clarity
    options_str = ", ".join(f"[{i + 1}] {opt}" for i, opt in enumerate(options))
    prompt_str = f"  {prompt_label} ({options_str}): "

    # Case-insensitive mapping for direct text input
    lookup = {opt.lower(): opt for opt in options}
    # Number mapping for 1-based index
    lookup_num = {str(i + 1): opt for i, opt in enumerate(options)}

    while True:
        try:
            raw_input = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting prediction utility.")
            sys.exit(0)

        # Check for exit command
        if raw_input.lower() in ["q", "quit", "exit"]:
            print("\nExiting prediction utility. Goodbye!")
            sys.exit(0)

        if not raw_input:
            print("    [!] Input cannot be empty. Please select one of the options.")
            continue

        # Check number selection first (e.g. '1', '2')
        if raw_input in lookup_num:
            return lookup_num[raw_input]

        # Check name selection (e.g. 'low', 'High')
        raw_lower = raw_input.lower()
        if raw_lower in lookup:
            return lookup[raw_lower]

        print(f"    [!] Invalid choice '{raw_input}'. Please enter a valid option number (1-{len(options)}) or name.")


def collect_student_features() -> dict:
    """
    Interactively collect all 19 original student features from the user.
    """
    print("\n" + "=" * 65)
    print("  STUDENT INFORMATION INPUT")
    print("  (Type 'q' or 'exit' at any prompt to quit)")
    print("=" * 65)

    student_data = {}

    # 1. Hours_Studied (dataset range: 1 to 44, realistic bounds: 0 to 100)
    print("\n[1/19] Academic Habits:")
    student_data["Hours_Studied"] = prompt_numerical(
        feature_name="Hours_Studied",
        prompt_label="Hours Studied per week",
        min_val=0,
        max_val=100,
        is_integer=False,
    )

    # 2. Attendance (dataset range: 60 to 100%, realistic bounds: 0 to 100)
    student_data["Attendance"] = prompt_numerical(
        feature_name="Attendance",
        prompt_label="Class Attendance percentage",
        min_val=0,
        max_val=100,
        is_integer=False,
    )

    # 3. Parental_Involvement
    print("\n[2/19] Family & Home Environment:")
    student_data["Parental_Involvement"] = prompt_categorical(
        feature_name="Parental_Involvement",
        prompt_label="Parental Involvement",
        options=CATEGORICAL_OPTIONS["Parental_Involvement"],
    )

    # 4. Access_to_Resources
    student_data["Access_to_Resources"] = prompt_categorical(
        feature_name="Access_to_Resources",
        prompt_label="Access to Resources",
        options=CATEGORICAL_OPTIONS["Access_to_Resources"],
    )

    # 5. Extracurricular_Activities
    print("\n[3/19] Activities & Routine:")
    student_data["Extracurricular_Activities"] = prompt_categorical(
        feature_name="Extracurricular_Activities",
        prompt_label="Extracurricular Activities",
        options=CATEGORICAL_OPTIONS["Extracurricular_Activities"],
    )

    # 6. Sleep_Hours (dataset range: 4 to 10, realistic bounds: 0 to 24)
    student_data["Sleep_Hours"] = prompt_numerical(
        feature_name="Sleep_Hours",
        prompt_label="Sleep Hours per night",
        min_val=0,
        max_val=24,
        is_integer=False,
    )

    # 7. Previous_Scores (dataset range: 50 to 100, realistic bounds: 0 to 100)
    print("\n[4/19] Prior Academic Performance:")
    student_data["Previous_Scores"] = prompt_numerical(
        feature_name="Previous_Scores",
        prompt_label="Previous Exam Score",
        min_val=0,
        max_val=100,
        is_integer=False,
    )

    # 8. Motivation_Level
    student_data["Motivation_Level"] = prompt_categorical(
        feature_name="Motivation_Level",
        prompt_label="Motivation Level",
        options=CATEGORICAL_OPTIONS["Motivation_Level"],
    )

    # 9. Internet_Access
    print("\n[5/19] Learning Environment & Resources:")
    student_data["Internet_Access"] = prompt_categorical(
        feature_name="Internet_Access",
        prompt_label="Internet Access at home",
        options=CATEGORICAL_OPTIONS["Internet_Access"],
    )

    # 10. Tutoring_Sessions (dataset range: 0 to 8, realistic bounds: 0 to 30)
    student_data["Tutoring_Sessions"] = prompt_numerical(
        feature_name="Tutoring_Sessions",
        prompt_label="Tutoring Sessions per month",
        min_val=0,
        max_val=30,
        is_integer=True,
    )

    # 11. Family_Income
    student_data["Family_Income"] = prompt_categorical(
        feature_name="Family_Income",
        prompt_label="Family Income Level",
        options=CATEGORICAL_OPTIONS["Family_Income"],
    )

    # 12. Teacher_Quality
    print("\n[6/19] School & Social Factors:")
    student_data["Teacher_Quality"] = prompt_categorical(
        feature_name="Teacher_Quality",
        prompt_label="Teacher Quality",
        options=CATEGORICAL_OPTIONS["Teacher_Quality"],
    )

    # 13. School_Type
    student_data["School_Type"] = prompt_categorical(
        feature_name="School_Type",
        prompt_label="School Type",
        options=CATEGORICAL_OPTIONS["School_Type"],
    )

    # 14. Peer_Influence
    student_data["Peer_Influence"] = prompt_categorical(
        feature_name="Peer_Influence",
        prompt_label="Peer Influence",
        options=CATEGORICAL_OPTIONS["Peer_Influence"],
    )

    # 15. Physical_Activity (dataset range: 0 to 6, realistic bounds: 0 to 40)
    student_data["Physical_Activity"] = prompt_numerical(
        feature_name="Physical_Activity",
        prompt_label="Physical Activity (hours/week)",
        min_val=0,
        max_val=40,
        is_integer=False,
    )

    # 16. Learning_Disabilities
    print("\n[7/19] Student Background:")
    student_data["Learning_Disabilities"] = prompt_categorical(
        feature_name="Learning_Disabilities",
        prompt_label="Learning Disabilities",
        options=CATEGORICAL_OPTIONS["Learning_Disabilities"],
    )

    # 17. Parental_Education_Level
    student_data["Parental_Education_Level"] = prompt_categorical(
        feature_name="Parental_Education_Level",
        prompt_label="Parental Education Level",
        options=CATEGORICAL_OPTIONS["Parental_Education_Level"],
    )

    # 18. Distance_from_Home
    student_data["Distance_from_Home"] = prompt_categorical(
        feature_name="Distance_from_Home",
        prompt_label="Distance from Home to school",
        options=CATEGORICAL_OPTIONS["Distance_from_Home"],
    )

    # 19. Gender
    student_data["Gender"] = prompt_categorical(
        feature_name="Gender",
        prompt_label="Gender",
        options=CATEGORICAL_OPTIONS["Gender"],
    )

    return student_data


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("   AcademiQ: Student Examination Score Prediction Utility")
    print("=" * 65)
    print("This utility uses AcademiQ's trained Multiple Linear Regression")
    print("pipeline to estimate a student's final examination score.")
    print("Model Artifact: models/AcademiQ_Multiple_Linear_Regression.pkl")
    print("=" * 65)

    # 1. Load the pre-trained pipeline (no retraining)
    pipeline = load_trained_pipeline(MODEL_PATH)

    # 2. Prediction loop
    while True:
        # Collect inputs
        student_data = collect_student_features()

        # 3. Construct DataFrame with exact column names and order
        input_df = pd.DataFrame([student_data], columns=FEATURE_COLUMNS)

        # 4. Pass DataFrame directly to pipeline (pipeline performs scaling and encoding)
        raw_prediction = pipeline.predict(input_df)[0]

        # 5. Display prediction results clearly (without artificial 0-100 clipping)
        print("\n" + "#" * 65)
        print("                   PREDICTION RESULT")
        print("#" * 65)
        print(f"\n  Predicted Exam_Score : {raw_prediction:.2f} points (out of 100)")
        print(f"  Exact Raw Prediction : {raw_prediction:.4f}")
        print("\n" + "-" * 65)
        print("  NOTE & DISCLAIMER:")
        print("  This value is a statistical model prediction generated by")
        print("  Multiple Linear Regression (trained on student performance data).")
        print("  It is an analytical estimate to guide academic counseling, not")
        print("  a guaranteed or official examination grade.")
        print("#" * 65 + "\n")

        # Prompt for another prediction or exit
        try:
            again = input("Would you like to predict for another student? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if again not in ["y", "yes"]:
            print("\nThank you for using AcademiQ. Goodbye!")
            break


if __name__ == "__main__":
    main()

