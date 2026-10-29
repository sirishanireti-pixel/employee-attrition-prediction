# PeoplePulse — Employee Attrition Dashboard

A Streamlit interface for the IBM HR Analytics Employee Attrition machine-learning project.

## Start

1. From your training notebook download `employee_attrition_model.pkl` (a full sklearn Pipeline).
2. Download the IBM HR Analytics CSV used in training.
3. Download `model_comparison.csv` from the notebook with:

   ```python
   results_df.to_csv("model_comparison.csv", index=False)
   from google.colab import files
   files.download("model_comparison.csv")
   ```

4. Install Python 3.10+ and run:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

5. Upload the model and dataset in the sidebar, then optionally upload the comparison CSV under Model performance.

## Features

- Workforce KPI dashboard and interactive attrition charts
- Employee prediction form covering **all** features expected by the trained Pipeline
- Estimated attrition probability and classification using 0.5 as the demonstration threshold
- Model comparison table and chart from honest notebook-generated evaluation results
- Session-only prediction history with CSV export and clear button

## Important notes

- Model files serialized with joblib/pickle can execute code when loaded. Only upload your own trusted file.
- Your saved training Pipeline must contain the fitted `preprocessing` and `classifier` steps shown in the notebook, or at least accept the raw input DataFrame with the same column names.
- The model should be exported from a notebook using compatible Python, scikit-learn and dependency versions.
- Model evaluation charts require the `results_df` exported from your notebook; do not evaluate on the original full training dataset and call those test scores.
- This app stores history only in Streamlit session state. It does not permanently retain employee inputs or history.
- The IBM HR dataset is synthetic. Predictions do not prove why employees leave and should not be used for individual employment decisions.
