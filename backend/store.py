"""
In-memory store for datasets, experiments, trained models, and job tracking.
In production, replace with a real database (PostgreSQL, MongoDB, etc.).
"""

from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Any

# ── Thread-safe singleton store ──────────────────────────────────────────────

_lock = threading.Lock()


@dataclass
class Dataset:
    id: str
    name: str
    columns: list[dict]          # [{name, type, missing, unique, total}]
    rows: list[dict]
    total_rows: int
    total_cols: int


@dataclass
class TrainedModel:
    id: str
    algorithm: str
    version: str
    hyperparams: dict
    metrics: dict
    confusion: list[list[int]]
    roc: list[dict]
    pr: list[dict]
    trained_at: str
    duration: float
    dataset_id: str
    artifact_path: str | None = None   # path to saved .joblib file
    problem_type: str = "classification"  # "classification" or "regression"
    mlflow_run_id: str | None = None
    residuals: list[dict] = field(default_factory=list)


@dataclass
class Experiment:
    id: str
    algorithm: str
    name: str
    version: str
    dataset_version: str
    metrics: dict
    trained_at: str
    duration: float
    status: str = "completed"
    mlflow_run_id: str | None = None
    problem_type: str = "classification"


@dataclass
class TrainingJob:
    job_id: str
    algorithm: str
    status: str = "pending"       # pending | running | completed | failed
    progress: float = 0.0
    step: int = 0
    result: dict | None = None
    error: str | None = None


class Store:
    """Simple thread-safe in-memory store."""

    def __init__(self) -> None:
        self.datasets: dict[str, Dataset] = {}
        self.models: dict[str, TrainedModel] = {}
        self.experiments: list[Experiment] = []
        self.jobs: dict[str, TrainingJob] = {}
        self._model_versions: dict[str, list[TrainedModel]] = {}   # algo → [versions]

    # ── Datasets ──────────────────────────────────────────────────────────────

    def save_dataset(self, ds: Dataset) -> None:
        with _lock:
            self.datasets[ds.id] = ds

    def get_dataset(self, ds_id: str) -> Dataset | None:
        return self.datasets.get(ds_id)

    def list_datasets(self) -> list[Dataset]:
        return list(self.datasets.values())

    # ── Models ────────────────────────────────────────────────────────────────

    def save_model(self, model: TrainedModel) -> None:
        with _lock:
            self.models[model.id] = model
            self._model_versions.setdefault(model.algorithm, []).append(model)

    def get_model(self, model_id: str) -> TrainedModel | None:
        return self.models.get(model_id)

    def list_models(self) -> list[dict]:
        """Return summary for each unique algorithm (latest version)."""
        summary = {}
        for m in self.models.values():
            if m.algorithm not in summary or m.trained_at > summary[m.algorithm]["trained_at"]:
                summary[m.algorithm] = {
                    "id": m.id,
                    "algorithm": m.algorithm,
                    "version": m.version,
                    "metrics": m.metrics,
                    "trained_at": m.trained_at,
                    "dataset_id": m.dataset_id,
                }
        return list(summary.values())

    def get_model_versions(self, algorithm: str) -> list[dict]:
        versions = self._model_versions.get(algorithm, [])
        return [
            {"id": v.id, "version": v.version, "metrics": v.metrics, "trained_at": v.trained_at}
            for v in versions
        ]

    # ── Experiments ───────────────────────────────────────────────────────────

    def add_experiment(self, exp: Experiment) -> None:
        with _lock:
            self.experiments.append(exp)

    def list_experiments(self) -> list[dict]:
        return [
            {
                "id": e.id,
                "algorithm": e.algorithm,
                "name": e.name,
                "version": e.version,
                "datasetVersion": e.dataset_version,
                "metrics": e.metrics,
                "trainedAt": e.trained_at,
                "duration": e.duration,
                "status": e.status,
                "mlflowRunId": e.mlflow_run_id,
                "problemType": e.problem_type,
            }
            for e in self.experiments
        ]

    # ── Training jobs ─────────────────────────────────────────────────────────

    def create_job(self, job: TrainingJob) -> None:
        with _lock:
            self.jobs[job.job_id] = job

    def get_job(self, job_id: str) -> TrainingJob | None:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        with _lock:
            job = self.jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)


# Global singleton
store = Store()
