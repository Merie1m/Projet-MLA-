/* ─── Zustand Global Store ────────────────────────────────────────────────── */
import { create } from 'zustand';

const useStore = create((set, get) => ({
  // ── Dataset ────────────────────────────────────────────────────────────────
  dataset: null,          // { id, name, columns, rows, preview }
  datasetLoading: false,
  setDataset: (dataset) => set({ dataset }),
  setDatasetLoading: (v) => set({ datasetLoading: v }),

  // ── Selected columns / classes ─────────────────────────────────────────────
  selectedColumns: [],
  selectedClasses: [],
  targetColumn: '',
  problemType: 'classification', // 'classification' | 'regression'
  setSelectedColumns: (cols) => set({ selectedColumns: cols }),
  setSelectedClasses: (cls) => set({ selectedClasses: cls }),
  setTargetColumn: (col) => set({ targetColumn: col }),
  setProblemType: (t) => set({ problemType: t }),

  // ── Models ─────────────────────────────────────────────────────────────────
  availableModels: [],
  selectedModels: [],     // array of { algorithm, hyperparams, fromScratch }
  setAvailableModels: (m) => set({ availableModels: m }),
  toggleModelSelection: (model) => {
    const sel = get().selectedModels;
    const exists = sel.find((s) => s.algorithm === model.algorithm);
    set({ selectedModels: exists ? sel.filter((s) => s.algorithm !== model.algorithm) : [...sel, model] });
  },
  updateModelHyperparams: (algorithm, hyperparams) => {
    set({
      selectedModels: get().selectedModels.map((m) =>
        m.algorithm === algorithm ? { ...m, hyperparams: { ...m.hyperparams, ...hyperparams } } : m
      ),
    });
  },
  clearSelectedModels: () => set({ selectedModels: [] }),

  // ── Saved configs ──────────────────────────────────────────────────────────
  savedConfigs: JSON.parse(localStorage.getItem('ml_saved_configs') || '[]'),
  saveConfig: (config) => {
    const configs = [...get().savedConfigs, { ...config, id: Date.now(), savedAt: new Date().toISOString() }];
    localStorage.setItem('ml_saved_configs', JSON.stringify(configs));
    set({ savedConfigs: configs });
  },
  deleteConfig: (id) => {
    const configs = get().savedConfigs.filter((c) => c.id !== id);
    localStorage.setItem('ml_saved_configs', JSON.stringify(configs));
    set({ savedConfigs: configs });
  },
  loadConfig: (config) => {
    set({ selectedModels: config.models || [] });
  },

  // ── Training ───────────────────────────────────────────────────────────────
  trainingJobs: [],       // [{ jobId, status, progress, algorithm }]
  trainingResults: [],    // [{ experimentId, algorithm, metrics, confusion, roc, pr }]
  addTrainingJob: (job) => set({ trainingJobs: [...get().trainingJobs, job] }),
  updateTrainingJob: (jobId, updates) => {
    set({
      trainingJobs: get().trainingJobs.map((j) => (j.jobId === jobId ? { ...j, ...updates } : j)),
    });
  },
  addTrainingResult: (result) => set({ trainingResults: [...get().trainingResults, result] }),
  clearResults: () => set({ trainingResults: [] }),

  // ── Analysis Results ───────────────────────────────────────────────────────
  biasVarianceData: [],
  stabilityData: { results: [], statistics: {} },
  setBiasVarianceData: (data) => set({ biasVarianceData: data }),
  setStabilityData: (data) => set({ stabilityData: data }),
  clearAnalysisData: () => set({ biasVarianceData: [], stabilityData: { results: [], statistics: {} } }),

  // ── MLOps ──────────────────────────────────────────────────────────────────
  experiments: [],
  setExperiments: (e) => set({ experiments: e }),

  // ── Auto-tune ──────────────────────────────────────────────────────────────
  autoTuneMethod: 'grid',   // 'grid' | 'random' | 'optuna'
  setAutoTuneMethod: (m) => set({ autoTuneMethod: m }),

  // ── Notifications ──────────────────────────────────────────────────────────
  notifications: [],
  addNotification: (n) => set({ notifications: [n, ...get().notifications].slice(0, 50) }),
  clearNotifications: () => set({ notifications: [] }),
}));

export default useStore;
