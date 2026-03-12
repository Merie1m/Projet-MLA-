"""
FastAPI backend for ML Platform.
Run with: uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import io
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ml_engine import auto_tune, clean_dataframe, evaluate_model, train_model, compute_dimension_reduction
from store import Dataset, TrainingJob, store

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="ML Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

# ─── Pydantic schemas ────────────────────────────────────────────────────────


class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    models: list[dict]  # [{algorithm, hyperparams, fromScratch}]
    selected_columns: list[str] | None = None
    selected_classes: list | None = None


class CleanRequest(BaseModel):
    datasetId: str
    operations: list[str]  # ["drop_na", "fill_mean", …]


class AutoTuneRequest(BaseModel):
    algorithm: str
    method: str  # "grid_search" | "random_search" | "optuna"
    dataset_id: str
    target_column: str
    selected_columns: list[str] | None = None
    selected_classes: list | None = None


class RollbackRequest(BaseModel):
    version: str


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV or Excel file and parse it."""
    try:
        contents = await file.read()
        filename = file.filename or "upload.csv"

        if filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))

        # Build column stats
        columns = []
        for col in df.columns:
            vals = df[col].dropna()
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            columns.append({
                "name": col,
                "type": "number" if is_numeric else "string",
                "missing": int(df[col].isna().sum()),
                "unique": int(df[col].nunique()),
                "total": len(df),
            })

        ds_id = str(uuid.uuid4())[:8]
        rows = df.where(df.notna(), None).to_dict(orient="records")

        ds = Dataset(
            id=ds_id,
            name=filename,
            columns=columns,
            rows=rows,
            total_rows=len(df),
            total_cols=len(df.columns),
        )
        store.save_dataset(ds)

        return {
            "id": ds_id,
            "name": filename,
            "columns": columns,
            "preview": rows[:100],
            "totalRows": len(df),
            "totalCols": len(df.columns),
            "size": len(contents),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dataset/{dataset_id}/preview")
def preview_dataset(dataset_id: str):
    ds = store.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "id": ds.id,
        "name": ds.name,
        "columns": ds.columns,
        "preview": ds.rows[:100],
        "totalRows": ds.total_rows,
        "totalCols": ds.total_cols,
    }


@app.post("/api/dataset/clean")
def clean_dataset(req: CleanRequest):
    ds = store.get_dataset(req.datasetId)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.DataFrame(ds.rows)
    df_cleaned = clean_dataframe(df, req.operations)

    rows = df_cleaned.where(df_cleaned.notna(), None).to_dict(orient="records")

    # Rebuild column stats
    columns = []
    for col in df_cleaned.columns:
        vals = df_cleaned[col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(df_cleaned[col])
        columns.append({
            "name": col,
            "type": "number" if is_numeric else "string",
            "missing": int(df_cleaned[col].isna().sum()),
            "unique": int(df_cleaned[col].nunique()),
            "total": len(df_cleaned),
        })

    ds.rows = rows
    ds.columns = columns
    ds.total_rows = len(df_cleaned)
    ds.total_cols = len(df_cleaned.columns)
    store.save_dataset(ds)

    return {
        "id": ds.id,
        "name": ds.name,
        "columns": columns,
        "preview": rows[:100],
        "totalRows": len(df_cleaned),
        "totalCols": len(df_cleaned.columns),
        "removed": len(df) - len(df_cleaned),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/models")
def list_models():
    """List available models and their latest versions."""
    return store.list_models()


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    m = store.get_model(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "id": m.id,
        "algorithm": m.algorithm,
        "version": m.version,
        "hyperparams": m.hyperparams,
        "metrics": m.metrics,
        "confusion": m.confusion,
        "roc": m.roc,
        "pr": m.pr,
        "trainedAt": m.trained_at,
        "duration": m.duration,
        "datasetId": m.dataset_id,
    }


@app.get("/api/models/{algorithm}/versions")
def get_model_versions(algorithm: str):
    versions = store.get_model_versions(algorithm)
    if not versions:
        raise HTTPException(status_code=404, detail="No versions found for this algorithm")
    return versions


@app.post("/api/models/{model_id}/rollback")
def rollback_model(model_id: str, req: RollbackRequest):
    """Rollback to a specific version (re-set as latest)."""
    # Find the version
    all_models = list(store.models.values())
    target = None
    for m in all_models:
        if m.version == req.version and (m.id == model_id or m.algorithm == model_id):
            target = m
            break
    if not target:
        raise HTTPException(status_code=404, detail=f"Version {req.version} not found")

    # Re-save as latest
    store.models[target.id] = target
    return {"detail": f"Rolled back to {req.version}", "model_id": target.id}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/api/train")
def start_training(req: TrainRequest):
    """Launch training for one or more models in background threads."""
    job_ids = []

    for model_cfg in req.models:
        job_id = str(uuid.uuid4())[:8]
        algorithm = model_cfg.get("algorithm", "")
        hyperparams = model_cfg.get("hyperparams", {})
        from_scratch = model_cfg.get("fromScratch", True)

        job = TrainingJob(job_id=job_id, algorithm=algorithm, status="pending")
        store.create_job(job)

        # Launch in background thread
        executor.submit(
            train_model,
            job_id=job_id,
            algorithm=algorithm,
            hyperparams=hyperparams,
            dataset_id=req.dataset_id,
            target_column=req.target_column,
            selected_columns=req.selected_columns,
            selected_classes=req.selected_classes,
            from_scratch=from_scratch,
        )

        job_ids.append({"jobId": job_id, "algorithm": algorithm})

    return {"jobs": job_ids}


@app.get("/api/train/status/{job_id}")
def get_training_status(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "jobId": job.job_id,
        "algorithm": job.algorithm,
        "status": job.status,
        "progress": job.progress,
        "step": job.step,
        "result": job.result,
        "error": job.error,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/results")
def list_results(algorithm: str | None = None):
    """Return all completed training results."""
    results = []
    for m in store.models.values():
        if algorithm and m.algorithm != algorithm:
            continue
        results.append({
            "experimentId": m.id,
            "algorithm": m.algorithm,
            "metrics": m.metrics,
            "confusion": m.confusion,
            "roc": m.roc,
            "pr": m.pr,
            "residuals": m.residuals,
            "trainedAt": m.trained_at,
            "duration": m.duration,
            "problemType": m.problem_type,
            "mlflowRunId": m.mlflow_run_id,
        })
    return results


@app.get("/api/results/{experiment_id}")
def get_experiment_result(experiment_id: str):
    m = store.get_model(experiment_id)
    if not m:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {
        "experimentId": m.id,
        "algorithm": m.algorithm,
        "metrics": m.metrics,
        "confusion": m.confusion,
        "roc": m.roc,
        "pr": m.pr,
        "residuals": m.residuals,
        "trainedAt": m.trained_at,
        "duration": m.duration,
        "problemType": m.problem_type,
        "mlflowRunId": m.mlflow_run_id,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-TUNING ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/api/autotune")
def run_autotune(req: AutoTuneRequest):
    """Run GridSearch / RandomSearch / Optuna auto-tuning."""
    try:
        result = auto_tune(
            algorithm=req.algorithm,
            method=req.method,
            dataset_id=req.dataset_id,
            target_column=req.target_column,
            selected_columns=req.selected_columns,
            selected_classes=req.selected_classes,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS (MLOps)
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/experiments")
def list_experiments():
    return store.list_experiments()


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/export/model/{model_id}")
def export_model(model_id: str, format: str = "joblib"):
    m = store.get_model(model_id)
    if not m or not m.artifact_path:
        raise HTTPException(status_code=404, detail="Model artifact not found")

    if not os.path.exists(m.artifact_path):
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    media_type = "application/octet-stream"
    filename = f"{m.algorithm}_{m.version}.{format}"

    return FileResponse(
        path=m.artifact_path,
        media_type=media_type,
        filename=filename,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION REDUCTION (PCA / t-SNE)
# ═══════════════════════════════════════════════════════════════════════════════


class DimReductionRequest(BaseModel):
    dataset_id: str
    target_column: str
    method: str = "pca"  # "pca" or "tsne"
    n_components: int = 2
    selected_columns: list[str] | None = None
    perplexity: int = 30


@app.post("/api/dimension-reduction")
def dimension_reduction(req: DimReductionRequest):
    """Compute PCA or t-SNE 2D projection."""
    try:
        result = compute_dimension_reduction(
            dataset_id=req.dataset_id,
            target_column=req.target_column,
            method=req.method,
            n_components=req.n_components,
            selected_columns=req.selected_columns,
            perplexity=req.perplexity,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# MLFLOW INFO
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/mlflow/status")
def mlflow_status():
    """Return MLflow tracking status."""
    try:
        from ml_engine import HAS_MLFLOW
        info = {"enabled": HAS_MLFLOW, "tracking_uri": None, "experiments": []}
        if HAS_MLFLOW:
            import mlflow
            info["tracking_uri"] = str(mlflow.get_tracking_uri())
            try:
                client = mlflow.MlflowClient()
                for exp in client.search_experiments():
                    info["experiments"].append({
                        "id": exp.experiment_id,
                        "name": exp.name,
                        "lifecycle_stage": exp.lifecycle_stage,
                    })
            except Exception:
                pass
        return info
    except Exception as e:
        return {"enabled": False, "error": str(e)}


@app.get("/api/mlflow/runs")
def mlflow_runs(experiment_name: str = "ML_Platform_Experiments", max_results: int = 50):
    """List MLflow runs for comparison."""
    try:
        from ml_engine import HAS_MLFLOW
        if not HAS_MLFLOW:
            return {"runs": []}
        import mlflow
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return {"runs": []}
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"],
        )
        result = []
        for run in runs:
            result.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": dict(run.data.params),
                "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
            })
        return {"runs": result}
    except Exception as e:
        return {"runs": [], "error": str(e)}


# ─── Health check ─────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok", "datasets": len(store.datasets), "models": len(store.models)}


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
