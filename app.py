"""Employee Attrition Prediction — Streamlit dashboard.

Run: streamlit run app.py
The app expects the trusted sklearn Pipeline saved by the training notebook.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "Best model (cross-validation F1)": "employee_attrition_best_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Support Vector Machine": "svm_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
}


st.set_page_config(page_title="PeoplePulse | HR Analytics", page_icon="📊", layout="wide")
st.markdown("""<style>
    .stApp {background:#0b1020; color:#e2e8f0;}
    .block-container {padding-top: 1.4rem; padding-bottom:3rem; max-width: 1350px;}
    [data-testid="stSidebar"] {background:#111a2e;}
    [data-testid="stMetric"] {background:#17233b; border:1px solid #2a3c5b;
        padding:18px; border-radius:15px;}
    [data-testid="stMetric"] * {color:#f8fafc !important;}
    [data-testid="stMetricLabel"] * {color:#afc3dd !important;}
    [data-testid="stForm"] {border:1px solid #293951; border-radius:17px; padding:22px;}
    .stTabs [data-baseweb="tab-list"] {gap:12px;}
    .stTabs [data-baseweb="tab"] {background:#17233b; border-radius:10px; padding:10px 17px;}
    h1, h2, h3 {letter-spacing:-.025em;}
    .eyebrow {color:#38bdf8; font-size:.83rem; font-weight:700; letter-spacing:.12em;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(model_bytes):
    # Pickle/joblib files can execute arbitrary code. Only load your own trusted export.
    return joblib.load(BytesIO(model_bytes))


@st.cache_resource(show_spinner=False)
def load_local_model(filename, modified_ns):
    # Only explicitly allowlisted model names within this local project are selectable.
    return joblib.load(PROJECT_DIR / filename)


@st.cache_data(show_spinner=False)
def load_data(csv_bytes):
    return pd.read_csv(BytesIO(csv_bytes))


def expected_columns(model, data):
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "named_steps"):
        preprocessing = model.named_steps.get("preprocessing")
        names = getattr(preprocessing, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return [c for c in data.columns if c not in
            {"Attrition", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"}]


def probability(model, row):
    if not hasattr(model, "predict_proba"):
        raise ValueError("The model must support predict_proba().")
    classes = list(getattr(model, "classes_", [0, 1]))
    positive = classes.index(1) if 1 in classes else classes.index("Yes")
    return float(model.predict_proba(row)[0, positive])


def input_widget(column, series, key):
    non_null = series.dropna()
    if non_null.empty:
        return st.text_input(column, value="", key=key)
    if pd.api.types.is_numeric_dtype(series):
        typical = float(non_null.median())
        low, high = float(non_null.min()), float(non_null.max())
        is_int = pd.api.types.is_integer_dtype(series)
        if is_int:
            return st.number_input(column, value=int(round(typical)), step=1, key=key,
                                   help=f"Observed dataset range: {int(low)} to {int(high)}")
        return st.number_input(column, value=typical, key=key,
                               help=f"Observed dataset range: {low:g} to {high:g}")
    choices = sorted(non_null.astype(str).unique().tolist())
    return st.selectbox(column, choices, key=key)


st.markdown('<div class="eyebrow">WORKFORCE INTELLIGENCE · ML ANALYTICS</div>', unsafe_allow_html=True)
st.title("📊 PeoplePulse")
st.caption("Interactive employee attrition analytics and model comparison · Academic prototype")

with st.sidebar:
    st.header("⚙️ Control center")
    local_data = PROJECT_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    st.markdown("### 1. Employee data")
    data_upload = st.file_uploader("Upload employee dataset (.csv)", type="csv",
                                   help="An uploaded dataset takes priority over the local sample CSV.")
    if data_upload is not None:
        st.success(f"Using uploaded dataset: {data_upload.name}")
    elif local_data.is_file():
        st.caption("Using the local IBM sample dataset until you upload a CSV.")
    st.markdown("### Model selector")
    available = {name: filename for name, filename in MODEL_FILES.items()
                 if (PROJECT_DIR / filename).is_file()}
    st.markdown("### 2. Prediction model")
    model_upload = st.file_uploader("Upload a trusted trained model (.pkl)",
                                    type=["pkl", "joblib"],
                                    help="The uploaded model takes priority over the local algorithm selector.")
    model_choice = st.selectbox("Or select a local trained algorithm", list(available)) if available else None
    if not available:
        st.caption("Run `python train_model.py` to enable the five-algorithm dropdown.")
    if model_upload is not None:
        st.success(f"Using uploaded model: {model_upload.name}")
    st.divider()
    st.caption("Only use models you trained or trust. Serialized model files can execute code when loaded.")
    st.caption("Prediction history is session-only; download it before closing the app.")

if "history" not in st.session_state:
    st.session_state.history = []

if data_upload is None and not local_data.is_file():
    st.warning("Start by uploading your IBM HR Analytics CSV in the sidebar.")
    st.stop()

try:
    df = load_data(data_upload.getvalue() if data_upload is not None else local_data.read_bytes())
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if "Attrition" not in df.columns:
    st.error("Expected an 'Attrition' column (Yes/No) in the IBM employee dataset.")
    st.stop()

model = None
if model_upload is not None:
    try:
        model = load_model(model_upload.getvalue())
    except Exception as exc:
        st.error(f"Could not load uploaded model: {exc}")
elif model_choice is not None:
    selected_path = PROJECT_DIR / available[model_choice]
    try:
        model = load_local_model(selected_path.name, selected_path.stat().st_mtime_ns)
        st.success(f"Active model: {model_choice}", icon="✅")
    except Exception as exc:
        st.error(f"Could not load {selected_path.name}: {exc}")

overview, prediction_tab, performance_tab, history_tab = st.tabs([
    "📊 Overview", "🎯 Predict", "🏆 Compare models", "🕘 History"
])

with overview:
    st.subheader("Workforce snapshot")
    st.caption("Understand the patterns in the uploaded IBM HR Analytics dataset.")
    total = len(df)
    left = int(df["Attrition"].astype(str).str.lower().eq("yes").sum())
    retained = total - left
    a, b, c, d = st.columns(4)
    a.metric("Employees", f"{total:,}")
    b.metric("Left", f"{left:,}")
    c.metric("Stayed", f"{retained:,}")
    d.metric("Attrition rate", f"{left / total:.1%}" if total else "N/A")
    left_col, right_col = st.columns(2)
    with left_col:
        counts = df["Attrition"].value_counts().reset_index()
        counts.columns = ["Attrition", "Employees"]
        st.plotly_chart(px.pie(counts, names="Attrition", values="Employees", hole=.58,
                               title="Attrition distribution", color="Attrition",
                               color_discrete_map={"Yes": "#f97373", "No": "#3b82f6"}),
                        width="stretch")
    with right_col:
        if "OverTime" in df:
            ct = df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Employees")
            st.plotly_chart(px.bar(ct, x="OverTime", y="Employees", color="Attrition",
                                   barmode="group", title="Overtime and attrition"),
                            width="stretch")
    charts = st.columns(2)
    with charts[0]:
        if "JobSatisfaction" in df:
            rate = df.assign(Left=df.Attrition.eq("Yes").astype(int)).groupby(
                "JobSatisfaction", as_index=False)["Left"].mean()
            st.plotly_chart(px.bar(rate, x="JobSatisfaction", y="Left",
                                   title="Attrition rate by job satisfaction",
                                   labels={"Left": "Attrition rate"}), width="stretch")
    with charts[1]:
        if "MonthlyIncome" in df:
            st.plotly_chart(px.box(df, x="Attrition", y="MonthlyIncome", color="Attrition",
                                   title="Monthly income by attrition"), width="stretch")
    with st.expander("Explore uploaded dataset"):
        st.dataframe(df.head(100), width="stretch")

with prediction_tab:
    st.subheader("Employee attrition assessment")
    if model is None:
        st.info("Run `python train_model.py` to create the model files, or upload your trusted .pkl model in the sidebar.")
    else:
        columns = expected_columns(model, df)
        missing = [c for c in columns if c not in df.columns]
        if missing:
            st.error(f"Dataset does not contain model features: {', '.join(missing)}")
        else:
            st.caption("Change the employee attributes below and compare estimates from different algorithms. All inputs are needed because the model was trained on them.")
            threshold = st.slider("Classification threshold", min_value=0.10, max_value=0.90,
                                  value=0.50, step=0.05, format="%.2f",
                                  help="At lower thresholds, more employees may be flagged; this can increase false positives.")
            with st.form("prediction_form"):
                entries = {}
                priority = ["Age", "JobRole", "Department", "MonthlyIncome", "OverTime",
                            "JobSatisfaction", "WorkLifeBalance", "YearsAtCompany",
                            "YearsSinceLastPromotion", "DistanceFromHome"]
                quick_cols = [c for c in priority if c in columns]
                extra_cols = [c for c in columns if c not in quick_cols]
                st.markdown("#### Main employee details")
                grid = st.columns(2)
                for i, column in enumerate(quick_cols):
                    with grid[i % 2]:
                        entries[column] = input_widget(column, df[column], f"input_{column}")
                with st.expander(f"Additional employee attributes ({len(extra_cols)})", expanded=False):
                    extra_grid = st.columns(3)
                    for i, column in enumerate(extra_cols):
                        with extra_grid[i % 3]:
                            entries[column] = input_widget(column, df[column], f"input_{column}")
                single_button, compare_button = st.columns(2)
                with single_button:
                    submitted = st.form_submit_button("Predict with selected model", type="primary", width="stretch")
                with compare_button:
                    compare_submitted = st.form_submit_button("Compare all five models", width="stretch")
            if submitted:
                row = pd.DataFrame([entries], columns=columns)
                try:
                    score = probability(model, row)
                    label = "Flagged for review" if score >= threshold else "Below review threshold"
                    left_score, right_score = st.columns([1, 2])
                    with left_score:
                        st.metric("Estimated attrition probability", f"{score:.1%}")
                    with right_score:
                        st.write("**Classification:**", label)
                        st.progress(min(100, max(0, int(score * 100))))
                        st.caption(f"Selected threshold: {threshold:.0%} · Model: {model_choice or 'Uploaded model'}")
                    st.caption("This probability is an experimental model estimate, not a calibrated or validated employee-level decision.")
                    st.session_state.history.insert(0, {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Prediction": label,
                        "Threshold": threshold,
                        "Attrition probability": round(score, 4),
                        "Model": type(model.named_steps["classifier"]).__name__
                        if hasattr(model, "named_steps") and "classifier" in model.named_steps
                        else type(model).__name__,
                    })
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
            if compare_submitted:
                row = pd.DataFrame([entries], columns=columns)
                comparison_rows = []
                failures = []
                for name, filename in MODEL_FILES.items():
                    if name.startswith("Best model"):
                        continue  # The best model is already one of the five algorithms.
                    path = PROJECT_DIR / filename
                    if not path.is_file():
                        failures.append(f"{name}: model file not found ({filename})")
                        continue
                    try:
                        candidate = load_local_model(filename, path.stat().st_mtime_ns)
                        candidate_columns = expected_columns(candidate, df)
                        missing_inputs = [c for c in candidate_columns if c not in row.columns]
                        if missing_inputs:
                            raise ValueError("Missing required inputs: " + ", ".join(missing_inputs))
                        candidate_row = row[candidate_columns]
                        score = probability(candidate, candidate_row)
                        comparison_rows.append({
                            "Model": name,
                            "Attrition probability": score,
                            "Prediction": "Flagged for review" if score >= threshold else "Below review threshold",
                        })
                    except Exception as exc:
                        failures.append(f"{name}: {exc}")
                if comparison_rows:
                    st.session_state.employee_comparison = comparison_rows
                    st.session_state.comparison_threshold = threshold
                    st.session_state.comparison_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for result in comparison_rows:
                        st.session_state.history.insert(0, {
                            "Timestamp": st.session_state.comparison_timestamp,
                            "Prediction": result["Prediction"],
                            "Threshold": threshold,
                            "Attrition probability": round(result["Attrition probability"], 4),
                            "Model": result["Model"],
                        })
                else:
                    st.session_state.pop("employee_comparison", None)
                if failures:
                    st.warning("Some models could not be compared:\n\n" + "\n\n".join(f"- {item}" for item in failures))
            if st.session_state.get("employee_comparison"):
                st.markdown("### Side-by-side employee predictions")
                st.caption(f"Last comparison: {st.session_state.comparison_timestamp} · Threshold: {st.session_state.comparison_threshold:.0%}. All models were given identical employee inputs.")
                employee_comparison = pd.DataFrame(st.session_state.employee_comparison)
                formatted = employee_comparison.copy()
                formatted["Attrition probability"] = formatted["Attrition probability"].map(lambda x: f"{x:.2%}")
                st.dataframe(formatted, hide_index=True, width="stretch")
                chart_data = employee_comparison.copy()
                chart_data["Probability (%)"] = chart_data["Attrition probability"] * 100
                chart = px.bar(chart_data, x="Model", y="Probability (%)", color="Prediction",
                               text="Probability (%)", range_y=[0, 100],
                               title="Attrition probability by model",
                               color_discrete_map={"Flagged for review": "#f97373", "Below review threshold": "#38bdf8"})
                chart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                chart.add_hline(y=st.session_state.comparison_threshold * 100,
                                line_dash="dash", line_color="#fbbf24",
                                annotation_text="Review threshold")
                st.plotly_chart(chart, width="stretch")
                st.caption("Different models can give different probabilities, and those probabilities may not be calibrated. This comparison is for academic exploration, not employment decisions.")

with performance_tab:
    st.subheader("Compare five classification algorithms")
    st.caption("Cross-validation helps select a model; test metrics show performance on held-out data.")
    local_metrics = PROJECT_DIR / "model_comparison.csv"
    metrics_file = None
    if local_metrics.is_file():
        st.success("Loaded model_comparison.csv from your project folder")
    else:
        metrics_file = st.file_uploader("Upload model_comparison.csv", type="csv", key="metrics_upload")
    if local_metrics.is_file() or metrics_file is not None:
        try:
            comparison = pd.read_csv(local_metrics if local_metrics.is_file() else metrics_file)
            if "Model" not in comparison:
                st.error("The comparison CSV needs a Model column.")
            else:
                rank_col = "CV_F1" if "CV_F1" in comparison else ("CV_ROC_AUC" if "CV_ROC_AUC" in comparison else None)
                if rank_col:
                    winner = comparison.sort_values(rank_col, ascending=False).iloc[0]
                    st.metric("Top model by cross-validation", str(winner["Model"]),
                              f"{rank_col}: {float(winner[rank_col]):.3f}")
                st.dataframe(comparison.round(4), width="stretch", hide_index=True)
                metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"] if c in comparison]
                if metric_cols:
                    selected_metrics = st.multiselect("Metrics to compare", metric_cols,
                                                      default=[m for m in ["Recall", "F1 Score", "ROC-AUC"] if m in metric_cols])
                    if not selected_metrics:
                        st.info("Select at least one metric above to draw the chart.")
                    else:
                        long = comparison.melt(id_vars="Model", value_vars=selected_metrics,
                                               var_name="Metric", value_name="Score")
                        st.plotly_chart(px.bar(long, x="Model", y="Score", color="Metric",
                                               barmode="group", title="Classifier performance",
                                               range_y=[0, 1]), width="stretch")
        except Exception as exc:
            st.error(f"Could not load comparison CSV: {exc}")
    else:
        st.info("Run the five-model training script. It generates model_comparison.csv automatically.")
    st.caption("Accuracy alone can be misleading here: only 237 of 1,470 employees in the IBM sample left.")

with history_tab:
    st.subheader("Prediction history")
    if not st.session_state.history:
        st.info("No predictions made in this session yet.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, width="stretch", hide_index=True)
        st.download_button("Download history CSV", history_df.to_csv(index=False),
                           "prediction_history.csv", mime="text/csv")
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()

st.divider()
st.caption("Educational prototype only. Predictions are associations, not causes. Review fairness, privacy, calibration and consent before any organizational use.")
