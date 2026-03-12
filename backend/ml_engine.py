"""
ML Engine — training, evaluation, preprocessing, auto-tuning, MLflow tracking,
regression support, and dimensionality reduction (PCA / t-SNE).
Uses scikit-learn, XGBoost, Optuna, and MLflow.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ── Classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ── Regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# ── Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, roc_curve,
    precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ── Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ── MLflow
try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
    # Configure local tracking directory
    mlflow.set_tracking_uri(os.path.join(os.path.dirname(__file__), "mlruns"))
except ImportError:
    HAS_MLFLOW = False

from store import store, TrainedModel, Experiment, TrainingJob


# ─── Classification algorithms ───────────────────────────────────────────────
CLASSIFICATION_ALGOS = {
    "svm", "random_forest", "knn", "logistic_regression",
    "neural_network", "gradient_boosting", "decision_tree", "naive_bayes",
}

# ─── Regression algorithms ───────────────────────────────────────────────────
REGRESSION_ALGOS = {
    "linear_regression", "svr", "rf_regression",
}


# ─── Algorithm registry ──────────────────────────────────────────────────────

def _build_estimator(algorithm: str, hyperparams: dict) -> Any:
    """Instantiate a scikit-learn estimator from algorithm id + hyperparams."""
    hp = dict(hyperparams)  # shallow copy

    # ── Classification ────────────────────────────────────────────────────────
    if algorithm == "svm":
        hp.setdefault("probability", True)  # needed for roc_auc
        return SVC(**hp)

    if algorithm == "random_forest":
        hp.pop("max_features", None) if hp.get("max_features") == "auto" else None
        return RandomForestClassifier(**hp)

    if algorithm == "knn":
        return KNeighborsClassifier(**hp)

    if algorithm == "logistic_regression":
        return LogisticRegression(**hp)

    if algorithm == "neural_network":
        hidden = hp.pop("hidden_layers", "128,64,32")
        if isinstance(hidden, str):
            hidden = tuple(int(x.strip()) for x in hidden.split(","))
        lr = hp.pop("learning_rate", 0.001)
        epochs = hp.pop("epochs", 50)
        batch = hp.pop("batch_size", 32)
        act = hp.pop("activation", "relu")
        optim = hp.pop("optimizer", "adam")
        dropout = hp.pop("dropout", 0.2)  # MLP doesn't have dropout but we keep param
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            learning_rate_init=lr,
            max_iter=epochs,
            batch_size=batch,
            activation=act,
            solver=optim if optim in ("adam", "sgd", "lbfgs") else "adam",
        )

    if algorithm == "gradient_boosting":
        if HAS_XGB:
            hp.setdefault("use_label_encoder", False)
            hp.setdefault("eval_metric", "logloss")
            return XGBClassifier(**hp)
        return GradientBoostingClassifier(
            n_estimators=hp.get("n_estimators", 200),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", 6),
            subsample=hp.get("subsample", 0.8),
        )

    if algorithm == "decision_tree":
        return DecisionTreeClassifier(**hp)

    if algorithm == "naive_bayes":
        return GaussianNB(var_smoothing=hp.get("var_smoothing", 1e-9))

    # ── Regression ────────────────────────────────────────────────────────────
    if algorithm == "linear_regression":
        return LinearRegression(**hp)

    if algorithm == "svr":
        hp.setdefault("kernel", "rbf")
        return SVR(**hp)

    if algorithm == "rf_regression":
        hp.pop("max_features", None) if hp.get("max_features") == "auto" else None
        return RandomForestRegressor(**hp)

    raise ValueError(f"Unknown algorithm: {algorithm}")


def _is_regression(algorithm: str) -> bool:
    """Return True if the algorithm is a regression algorithm."""
    return algorithm in REGRESSION_ALGOS


# ─── Preprocessing helpers ────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame, operations: list[str]) -> pd.DataFrame:
    """Apply a sequence of cleaning operations to a DataFrame."""
    df = df.copy()
    for op in operations:
        if op == "drop_na":
            df = df.dropna()
        elif op == "fill_mean":
            num_cols = df.select_dtypes(include="number").columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif op == "fill_median":
            num_cols = df.select_dtypes(include="number").columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        elif op == "fill_mode":
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else df[col])
        elif op == "drop_duplicates":
            df = df.drop_duplicates()
        elif op == "normalize":
            num_cols = df.select_dtypes(include="number").columns
            scaler = MinMaxScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
        elif op == "standardize":
            num_cols = df.select_dtypes(include="number").columns
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
        elif op == "encode_label":
            cat_cols = df.select_dtypes(include="object").columns
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        elif op == "encode_onehot":
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            if cat_cols:
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.reset_index(drop=True)
    return df


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    selected_columns: list[str] | None = None,
    selected_classes: list[Any] | None = None,
    test_size: float = 0.2,
    max_rows: int = 50_000,
    regression: bool = False,
):
    """Prepare X_train, X_test, y_train, y_test from a DataFrame."""
    if selected_columns:
        cols = [c for c in selected_columns if c in df.columns and c != target_column]
    else:
        cols = [c for c in df.columns if c != target_column]

    df_work = df[cols + [target_column]].copy()

    # Filter classes if specified (classification only)
    if selected_classes and not regression:
        df_work = df_work[df_work[target_column].isin(selected_classes)]

    # Sample large datasets to avoid memory issues
    if len(df_work) > max_rows:
        df_work = df_work.sample(n=max_rows, random_state=42).reset_index(drop=True)

    # Encode target
    le = None
    if regression:
        y = df_work[target_column].astype(float).values
    else:
        le = LabelEncoder()
        y = le.fit_transform(df_work[target_column].astype(str))

    X = df_work[cols]

    # Encode categorical columns — use label encoding for high-cardinality,
    # one-hot only for low-cardinality (<=10 unique values) to avoid memory explosion
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        n_unique = X[col].nunique()
        if n_unique <= 10:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        else:
            col_le = LabelEncoder()
            X[col] = col_le.fit_transform(X[col].astype(str))

    # Fill remaining NaN
    X = X.fillna(0)

    # Scale numeric
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratify only for classification when every class has at least 2 members
    stratify_arg = None
    if not regression:
        from collections import Counter
        class_counts = Counter(y)
        can_stratify = all(c >= 2 for c in class_counts.values()) and len(class_counts) > 1
        stratify_arg = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=stratify_arg
    )
    return X_train, X_test, y_train, y_test, le, list(X.columns)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute all metrics, confusion matrix, ROC and PR curve data."""
    y_pred = model.predict(X_test)

    # Probabilities (if available)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)

    n_classes = len(set(y_test))
    is_binary = n_classes == 2

    metrics = {
        "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
        "precision": float(round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)),
        "f1_score": float(round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)),
    }

    # ROC AUC (binary or multi-class)
    if y_proba is not None:
        try:
            if is_binary:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = float(round(roc_auc_score(y_test, proba_pos), 4))
            else:
                metrics["roc_auc"] = float(round(
                    roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"), 4
                ))
        except Exception:
            metrics["roc_auc"] = 0.0
        try:
            if is_binary:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["log_loss"] = float(round(log_loss(y_test, proba_pos), 4))
            else:
                metrics["log_loss"] = float(round(log_loss(y_test, y_proba), 4))
        except Exception:
            metrics["log_loss"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
        metrics["log_loss"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # ROC curve data (binary)
    roc_data = []
    if is_binary and y_proba is not None:
        proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        fpr, tpr, _ = roc_curve(y_test, proba_pos)
        roc_data = [{"fpr": float(round(f, 4)), "tpr": float(round(t, 4))} for f, t in zip(fpr, tpr)]

    # PR curve data (binary)
    pr_data = []
    if is_binary and y_proba is not None:
        proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, proba_pos)
        pr_data = [
            {"recall": float(round(r, 4)), "precision": float(round(p, 4))}
            for r, p in zip(rec_arr, prec_arr)
        ]

    return {
        "metrics": metrics,
        "confusion": cm,
        "roc": roc_data,
        "pr": pr_data,
    }


def evaluate_model_regression(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute regression metrics (MAE, MSE, RMSE, R²) and residual data."""
    y_pred = model.predict(X_test)

    mae = float(round(mean_absolute_error(y_test, y_pred), 4))
    mse = float(round(mean_squared_error(y_test, y_pred), 4))
    rmse = float(round(np.sqrt(mse), 4))
    r2 = float(round(r2_score(y_test, y_pred), 4))

    # Residual scatter data (sample max 200 points for frontend)
    indices = np.random.choice(len(y_test), size=min(200, len(y_test)), replace=False)
    residuals = [
        {"actual": float(round(y_test[i], 4)), "predicted": float(round(y_pred[i], 4))}
        for i in indices
    ]

    return {
        "metrics": {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2},
        "confusion": [],
        "roc": [],
        "pr": [],
        "residuals": residuals,
    }


# ─── Training pipeline ───────────────────────────────────────────────────────

def train_model(
    job_id: str,
    algorithm: str,
    hyperparams: dict,
    dataset_id: str,
    target_column: str,
    selected_columns: list[str] | None = None,
    selected_classes: list[Any] | None = None,
    from_scratch: bool = True,
) -> dict:
    """Train a single model. Updates job status in the store as it progresses.
    Logs everything to MLflow when available."""

    regression = _is_regression(algorithm)

    try:
        # Step 0 — Validation
        store.update_job(job_id, status="running", progress=5.0, step=0)
        ds = store.get_dataset(dataset_id)
        if ds is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        df = pd.DataFrame(ds.rows)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        # Step 1 — Preprocessing
        store.update_job(job_id, progress=15.0, step=1)
        X_train, X_test, y_train, y_test, label_enc, feature_names = prepare_data(
            df, target_column, selected_columns, selected_classes, regression=regression,
        )

        # Step 2 — Training (with MLflow tracking)
        store.update_job(job_id, progress=30.0, step=2)
        start = time.time()
        estimator = _build_estimator(algorithm, hyperparams)

        # ── MLflow experiment ─────────────────────────────────────────────────
        mlflow_run_id = None
        if HAS_MLFLOW:
            experiment_name = "ML_Platform_Experiments"
            mlflow.set_experiment(experiment_name)
            run = mlflow.start_run(run_name=f"{_algo_display_name(algorithm)}_{job_id}")
            mlflow_run_id = run.info.run_id

            # Log parameters
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("dataset_id", dataset_id)
            mlflow.log_param("target_column", target_column)
            mlflow.log_param("problem_type", "regression" if regression else "classification")
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            for k, v in hyperparams.items():
                mlflow.log_param(f"hp_{k}", v)

        estimator.fit(X_train, y_train)
        duration = round(time.time() - start, 2)

        # Step 3 — Evaluation
        store.update_job(job_id, progress=80.0, step=3)
        if regression:
            eval_result = evaluate_model_regression(estimator, X_test, y_test)
        else:
            eval_result = evaluate_model(estimator, X_test, y_test)

        # Log metrics to MLflow
        if HAS_MLFLOW and mlflow_run_id:
            for k, v in eval_result["metrics"].items():
                mlflow.log_metric(k, v)
            mlflow.log_metric("duration_seconds", duration)

        # Step 4 — Save model
        store.update_job(job_id, progress=95.0, step=4)
        model_id = str(uuid.uuid4())[:8]
        version_count = len(store.get_model_versions(algorithm))
        version = f"v1.{version_count}"
        artifact_path = f"models/{algorithm}_{model_id}.joblib"

        os.makedirs("models", exist_ok=True)
        joblib.dump(estimator, artifact_path)

        # Log model artifact to MLflow
        if HAS_MLFLOW and mlflow_run_id:
            mlflow.sklearn.log_model(estimator, artifact_path=f"model_{algorithm}")
            mlflow.log_artifact(artifact_path)
            mlflow.end_run()

        trained_model = TrainedModel(
            id=model_id,
            algorithm=algorithm,
            version=version,
            hyperparams=hyperparams,
            metrics=eval_result["metrics"],
            confusion=eval_result["confusion"],
            roc=eval_result["roc"],
            pr=eval_result["pr"],
            trained_at=datetime.now(timezone.utc).isoformat(),
            duration=duration,
            dataset_id=dataset_id,
            artifact_path=artifact_path,
            problem_type="regression" if regression else "classification",
            mlflow_run_id=mlflow_run_id,
            residuals=eval_result.get("residuals", []),
        )
        store.save_model(trained_model)

        experiment = Experiment(
            id=model_id,
            algorithm=algorithm,
            name=_algo_display_name(algorithm),
            version=version,
            dataset_version="v1.0",
            metrics=eval_result["metrics"],
            trained_at=trained_model.trained_at,
            duration=duration,
            mlflow_run_id=mlflow_run_id,
            problem_type="regression" if regression else "classification",
        )
        store.add_experiment(experiment)

        result = {
            "experimentId": model_id,
            "algorithm": algorithm,
            "metrics": eval_result["metrics"],
            "confusion": eval_result["confusion"],
            "roc": eval_result["roc"],
            "pr": eval_result["pr"],
            "residuals": eval_result.get("residuals", []),
            "trainedAt": trained_model.trained_at,
            "duration": duration,
            "problemType": "regression" if regression else "classification",
            "mlflowRunId": mlflow_run_id,
        }

        store.update_job(job_id, status="completed", progress=100.0, step=4, result=result)
        return result

    except Exception as e:
        if HAS_MLFLOW:
            try:
                mlflow.end_run(status="FAILED")
            except Exception:
                pass
        store.update_job(job_id, status="failed", error=str(e))
        raise


# ─── Auto-tuning ──────────────────────────────────────────────────────────────

PARAM_GRIDS = {
    "svm": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf", "poly"], "gamma": ["scale", "auto"]},
    "random_forest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20], "min_samples_split": [2, 5, 10]},
    "knn": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"], "metric": ["minkowski", "euclidean"]},
    "logistic_regression": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs", "liblinear"]},
    "neural_network": {"hidden_layer_sizes": [(64,), (128, 64), (128, 64, 32)], "learning_rate_init": [0.001, 0.01], "max_iter": [100, 200]},
    "gradient_boosting": {"n_estimators": [100, 200], "max_depth": [3, 6, 10], "learning_rate": [0.01, 0.1, 0.2]},
    "decision_tree": {"max_depth": [5, 10, 20], "criterion": ["gini", "entropy"], "min_samples_split": [2, 5, 10]},
    "naive_bayes": {"var_smoothing": [1e-9, 1e-7, 1e-5]},
    # Regression
    "linear_regression": {},  # no tunable hyperparams for basic LinearRegression
    "svr": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf", "poly"], "gamma": ["scale", "auto"]},
    "rf_regression": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20], "min_samples_split": [2, 5, 10]},
}


def auto_tune(
    algorithm: str,
    method: str,
    dataset_id: str,
    target_column: str,
    selected_columns: list[str] | None = None,
    selected_classes: list[Any] | None = None,
) -> dict:
    """Run GridSearch, RandomSearch, or Optuna to find best hyperparams."""

    ds = store.get_dataset(dataset_id)
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    regression = _is_regression(algorithm)
    df = pd.DataFrame(ds.rows)
    X_train, X_test, y_train, y_test, _, _ = prepare_data(
        df, target_column, selected_columns, selected_classes, regression=regression,
    )

    scoring = "neg_mean_squared_error" if regression else "accuracy"

    if method == "optuna" and HAS_OPTUNA:
        return _optuna_tune(algorithm, X_train, y_train, X_test, y_test, regression=regression)

    # Build a base estimator without hyperparams (defaults)
    base = _build_estimator(algorithm, {})
    param_grid = PARAM_GRIDS.get(algorithm, {})

    if not param_grid:
        return {"algorithm": algorithm, "method": method, "best_params": {}, "best_score": 0.0}

    # Special handling for SVM — need probability=True
    if algorithm == "svm":
        base.set_params(probability=True)

    if method == "grid_search":
        search = GridSearchCV(base, param_grid, cv=3, scoring=scoring, n_jobs=-1)
    else:  # random_search
        search = RandomizedSearchCV(base, param_grid, n_iter=10, cv=3, scoring=scoring, n_jobs=-1, random_state=42)

    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score = float(round(abs(search.best_score_), 4))

    return {
        "algorithm": algorithm,
        "method": method,
        "best_params": _serialize_params(best_params),
        "best_score": best_score,
    }


def _optuna_tune(
    algorithm: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int = 30,
    regression: bool = False,
) -> dict:
    """Bayesian optimization with Optuna."""

    if regression:
        def objective(trial: optuna.Trial) -> float:
            hp = _suggest_hyperparams(trial, algorithm)
            model = _build_estimator(algorithm, hp)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return -mean_squared_error(y_test, y_pred)  # maximize = less MSE

        study = optuna.create_study(direction="maximize")
    else:
        def objective(trial: optuna.Trial) -> float:
            hp = _suggest_hyperparams(trial, algorithm)
            model = _build_estimator(algorithm, hp)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)

        study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "algorithm": algorithm,
        "method": "optuna",
        "best_params": _serialize_params(study.best_params),
        "best_score": float(round(abs(study.best_value), 4)),
    }


def _suggest_hyperparams(trial: "optuna.Trial", algorithm: str) -> dict:
    """Suggest hyperparams for Optuna based on algorithm."""
    if algorithm == "svm":
        return {
            "C": trial.suggest_float("C", 0.01, 100, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "probability": True,
        }
    if algorithm == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
    if algorithm == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    if algorithm == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.01, 100, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            "max_iter": trial.suggest_int("max_iter", 100, 1000, step=100),
        }
    if algorithm == "neural_network":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = tuple(trial.suggest_int(f"units_{i}", 32, 256, step=32) for i in range(n_layers))
        return {
            "hidden_layers": ",".join(str(x) for x in layers),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
            "epochs": trial.suggest_int("epochs", 50, 300, step=50),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        }
    if algorithm == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    if algorithm == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
    if algorithm == "naive_bayes":
        return {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-3, log=True),
        }
    # ── Regression ────────────────────────────────────────────────────────────
    if algorithm == "linear_regression":
        return {}  # no tunable hyperparams

    if algorithm == "svr":
        return {
            "C": trial.suggest_float("C", 0.01, 100, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

    if algorithm == "rf_regression":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

    return {}


# ─── Dimensionality reduction (PCA / t-SNE) ──────────────────────────────────

def compute_dimension_reduction(
    dataset_id: str,
    target_column: str,
    method: str = "pca",
    n_components: int = 2,
    selected_columns: list[str] | None = None,
    perplexity: int = 30,
    max_rows: int = 5000,
) -> dict:
    """Compute PCA or t-SNE 2D projection for visualization."""
    ds = store.get_dataset(dataset_id)
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    df = pd.DataFrame(ds.rows)

    if selected_columns:
        cols = [c for c in selected_columns if c in df.columns and c != target_column]
    else:
        cols = [c for c in df.columns if c != target_column]

    X = df[cols].copy()
    y = df[target_column].copy()

    # Encode categoricals
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.fillna(0)

    # Sample if too large (t-SNE is slow on big data)
    if len(X) > max_rows:
        idx = np.random.choice(len(X), size=max_rows, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "pca":
        reducer = PCA(n_components=n_components)
        coords = reducer.fit_transform(X_scaled)
        explained_variance = [float(round(v, 4)) for v in reducer.explained_variance_ratio_]
    elif method == "tsne":
        perp = min(perplexity, len(X_scaled) - 1)
        reducer = TSNE(n_components=n_components, perplexity=perp, random_state=42, n_iter=1000)
        coords = reducer.fit_transform(X_scaled)
        explained_variance = []
    else:
        raise ValueError(f"Unknown method: {method}")

    points = [
        {"x": float(round(coords[i, 0], 4)), "y": float(round(coords[i, 1], 4)), "label": str(y.iloc[i])}
        for i in range(len(coords))
    ]

    return {
        "method": method,
        "points": points,
        "explained_variance": explained_variance,
        "n_samples": len(points),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

_ALGO_NAMES = {
    "svm": "Support Vector Machine (SVM)",
    "random_forest": "Random Forest",
    "knn": "K-Nearest Neighbors (KNN)",
    "logistic_regression": "Régression Logistique",
    "neural_network": "Réseau de Neurones (MLP)",
    "gradient_boosting": "Gradient Boosting (XGBoost)",
    "decision_tree": "Arbre de Décision",
    "naive_bayes": "Naive Bayes",
    "linear_regression": "Régression Linéaire",
    "svr": "Support Vector Regression (SVR)",
    "rf_regression": "Random Forest Régression",
}


def _algo_display_name(algo_id: str) -> str:
    return _ALGO_NAMES.get(algo_id, algo_id)


def _serialize_params(params: dict) -> dict:
    """Ensure all param values are JSON-serializable."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (tuple, list)):
            out[k] = str(v)
        else:
            out[k] = v
    return out
