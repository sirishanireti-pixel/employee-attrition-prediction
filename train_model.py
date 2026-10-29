import pandas as pd
import numpy as np
import joblib
import sklearn

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# STEP 1: Load dataset
df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

print("Dataset loaded:", df.shape)

# STEP 2: Remove unnecessary columns
df = df.drop(columns=[
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
])

# STEP 3: Features and target
X = df.drop("Attrition", axis=1)

y = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})

# STEP 4: Identify data types
num_cols = X.select_dtypes(
    include=np.number
).columns

cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns

# STEP 5: Preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# STEP 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# STEP 7: Five ML algorithms
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        random_state=42
    )
}

results = []
trained_models = {}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# STEP 8: Train and evaluate
for name, algorithm in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", algorithm)
    ])

    # Cross-validation on training data
    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "f1": "f1",
            "roc_auc": "roc_auc"
        },
        n_jobs=-1
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "CV_F1": cv_scores["test_f1"].mean(),
        "CV_ROC_AUC": cv_scores["test_roc_auc"].mean(),
        "Accuracy": accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions
        ),
        "F1 Score": f1_score(
            y_test, predictions
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        )
    })

    trained_models[name] = pipeline

    # Save individual model
    filename = name.lower().replace(
        " ", "_"
    ) + "_model.pkl"

    joblib.dump(pipeline, filename)

    print(name, "completed!")

# STEP 9: Compare models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV_F1",
    ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# STEP 10: Select best model
best_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_name]

print("\nBest Model:", best_name)

# STEP 11: Save best model
joblib.dump(
    best_model,
    "employee_attrition_best_model.pkl"
)

# STEP 12: Export results
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nAll five models trained successfully!")
print("Best model saved!")
print("Model comparison saved!")
print("scikit-learn version:", sklearn.__version__)