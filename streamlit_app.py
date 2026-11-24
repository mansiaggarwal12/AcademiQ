"""
AcademiQ - Streamlit Cloud Entrypoint
Redirects execution to app.py for default Streamlit Cloud deployments.
"""
import runpy
from pathlib import Path

app_path = Path(__file__).resolve().parent / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
