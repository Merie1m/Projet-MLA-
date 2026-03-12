/* ─── API Service Layer ───────────────────────────────────────────────────────
   All communication with the Python backend (Flask / FastAPI) goes through
   this module. Base URL is proxied via Vite → http://localhost:8000/api
   ──────────────────────────────────────────────────────────────────────────── */

const BASE = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Erreur serveur');
  }
  return res.json();
}

// ─── Models ──────────────────────────────────────────────────────────────────

/** GET /api/models → liste modèles disponibles et versions */
export function fetchModels() {
  return request('/models');
}

/** GET /api/models/:id */
export function fetchModel(id) {
  return request(`/models/${id}`);
}

// ─── Training ────────────────────────────────────────────────────────────────

/** POST /api/train — envoie dataset + hyperparams */
export function startTraining(payload) {
  return request('/train', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

/** GET /api/train/status/:jobId */
export function getTrainingStatus(jobId) {
  return request(`/train/status/${jobId}`);
}

// ─── Results ─────────────────────────────────────────────────────────────────

/** GET /api/results — récupère métriques et visualisations */
export function fetchResults(params = {}) {
  const qs = new URLSearchParams(params).toString();
  return request(`/results${qs ? `?${qs}` : ''}`);
}

/** GET /api/results/:experimentId */
export function fetchExperimentResult(experimentId) {
  return request(`/results/${experimentId}`);
}

// ─── Dataset ─────────────────────────────────────────────────────────────────

/** POST /api/dataset/upload — upload CSV / Excel */
export function uploadDataset(file) {
  const formData = new FormData();
  formData.append('file', file);
  return fetch(`${BASE}/dataset/upload`, { method: 'POST', body: formData }).then((r) => {
    if (!r.ok) throw new Error('Upload échoué');
    return r.json();
  });
}

/** POST /api/dataset/clean — nettoyage rapide */
export function cleanDataset(datasetId, options) {
  return request('/dataset/clean', {
    method: 'POST',
    body: JSON.stringify({ datasetId, ...options }),
  });
}

/** GET /api/dataset/:id/preview */
export function previewDataset(id) {
  return request(`/dataset/${id}/preview`);
}

// ─── Auto-tuning ─────────────────────────────────────────────────────────────

/** POST /api/autotune — GridSearch / RandomSearch / Optuna */
export function startAutoTune(payload) {
  return request('/autotune', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

// ─── MLOps ───────────────────────────────────────────────────────────────────

/** GET /api/experiments */
export function fetchExperiments() {
  return request('/experiments');
}

/** POST /api/models/:id/rollback */
export function rollbackModel(modelId, version) {
  return request(`/models/${modelId}/rollback`, {
    method: 'POST',
    body: JSON.stringify({ version }),
  });
}

/** GET /api/models/:id/versions */
export function fetchModelVersions(modelId) {
  return request(`/models/${modelId}/versions`);
}

// ─── Export ──────────────────────────────────────────────────────────────────

/** GET /api/export/model/:id — télécharger pickle / joblib */
export function exportModel(modelId, format = 'joblib') {
  return fetch(`${BASE}/export/model/${modelId}?format=${format}`);
}

// ─── Dimension Reduction ─────────────────────────────────────────────────────

/** POST /api/dimension-reduction — PCA or t-SNE */
export function computeDimensionReduction(payload) {
  return request('/dimension-reduction', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

// ─── MLflow ──────────────────────────────────────────────────────────────────

/** GET /api/mlflow/status — MLflow tracking status */
export function fetchMlflowStatus() {
  return request('/mlflow/status');
}

/** GET /api/mlflow/runs — list MLflow runs */
export function fetchMlflowRuns(experimentName = 'ML_Platform_Experiments') {
  return request(`/mlflow/runs?experiment_name=${encodeURIComponent(experimentName)}`);
}
