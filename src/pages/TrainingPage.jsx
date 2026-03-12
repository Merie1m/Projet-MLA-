import React, { useState, useEffect, useRef } from 'react';
import {
  Card, CardContent, Typography, Box, Button, LinearProgress, Chip,
  Alert, Stepper, Step, StepLabel, CircularProgress, Divider, Tooltip,
} from '@mui/material';
import {
  PlayArrow, Stop, RocketLaunch, CheckCircle, Error as ErrorIcon,
  Timer, Memory, Dataset,
} from '@mui/icons-material';
import toast from 'react-hot-toast';
import useStore from '../store/useStore';
import HelpTooltip from '../components/HelpTooltip';
import { ML_ALGORITHMS, REGRESSION_ALGO_IDS } from '../constants';
import { startTraining, getTrainingStatus } from '../services/api';

const STEPS = ['Validation des données', 'Pré-traitement', 'Entraînement', 'Évaluation', 'Terminé'];

export default function TrainingPage() {
  const {
    dataset, selectedModels, addTrainingJob, updateTrainingJob, trainingJobs,
    addTrainingResult, addNotification,
  } = useStore();
  const [running, setRunning] = useState(false);
  const [jobStates, setJobStates] = useState({});  // { algorithm: { progress, step } }
  const cancelRefs = useRef({});

  const algoMap = {};
  ML_ALGORITHMS.forEach((a) => { algoMap[a.id] = a; });

  const canTrain = dataset && selectedModels.length > 0 && !running;
  const pollingRefs = useRef({});

  // ── Poll job status from the backend ───────────────────────────────────────
  const pollJob = (jobId, algorithm) => {
    const interval = setInterval(async () => {
      try {
        const status = await getTrainingStatus(jobId);
        setJobStates((prev) => ({ ...prev, [algorithm]: { progress: status.progress, step: status.step } }));

        if (status.status === 'completed') {
          clearInterval(interval);
          delete pollingRefs.current[algorithm];
          if (status.result) {
            addTrainingResult(status.result);
            const isRegression = REGRESSION_ALGO_IDS.includes(algorithm);
            const metricMsg = isRegression
              ? `R²: ${status.result.metrics.r2?.toFixed(4)}`
              : `accuracy: ${(status.result.metrics.accuracy * 100).toFixed(1)}%`;
            addNotification({
              id: Date.now(),
              type: 'success',
              message: `${algoMap[algorithm]?.name || algorithm} — Entraînement terminé (${metricMsg})`,
              timestamp: new Date().toISOString(),
            });
            toast.success(`${algoMap[algorithm]?.name} terminé — ${metricMsg}`);
          }
          // Check if all jobs done
          if (Object.keys(pollingRefs.current).length === 0) setRunning(false);
        } else if (status.status === 'failed') {
          clearInterval(interval);
          delete pollingRefs.current[algorithm];
          toast.error(`${algoMap[algorithm]?.name || algorithm} — Échec : ${status.error || 'Erreur inconnue'}`);
          addNotification({
            id: Date.now(), type: 'error',
            message: `${algoMap[algorithm]?.name || algorithm} — Échec`, timestamp: new Date().toISOString(),
          });
          if (Object.keys(pollingRefs.current).length === 0) setRunning(false);
        }
      } catch {
        // Network error — keep polling
      }
    }, 1000);
    pollingRefs.current[algorithm] = interval;
  };

  // ── Launch training via real API ───────────────────────────────────────────
  const handleTrain = async () => {
    if (!canTrain) return;
    setRunning(true);
    const states = {};
    selectedModels.forEach((m) => { states[m.algorithm] = { progress: 0, step: 0 }; });
    setJobStates(states);

    try {
      const targetCol = useStore.getState().targetColumn
        || dataset.columns?.find(c => c.type === 'string')?.name
        || dataset.columns?.[dataset.columns.length - 1]?.name
        || 'target';

      const payload = {
        dataset_id: dataset.id,
        target_column: targetCol,
        models: selectedModels,
        selected_columns: useStore.getState().selectedColumns,
        selected_classes: useStore.getState().selectedClasses?.length ? useStore.getState().selectedClasses : null,
      };

      const resp = await startTraining(payload);
      // Start polling each job
      (resp.jobs || []).forEach((job) => {
        pollJob(job.jobId, job.algorithm);
      });
    } catch (err) {
      toast.error('Erreur au lancement : ' + err.message);
      setRunning(false);
    }
  };

  // ── Stop training ──────────────────────────────────────────────────────────
  const handleStop = () => {
    Object.values(pollingRefs.current).forEach((interval) => clearInterval(interval));
    pollingRefs.current = {};
    setRunning(false);
    toast('Entraînement arrêté', { icon: '⏹️' });
  };

  return (
    <Box className="space-y-6">
      {/* ── Header ────────────────────────────────────────────── */}
      <Box className="flex flex-wrap items-center justify-between gap-4">
        <Box>
          <Typography variant="h5">Entraînement</Typography>
          <Typography variant="body2" color="text.secondary">
            Lancez l'entraînement de vos modèles sélectionnés sur le dataset chargé.
          </Typography>
        </Box>
        <Box className="flex gap-2">
          {running ? (
            <Button variant="contained" color="error" startIcon={<Stop />} onClick={handleStop}>
              Arrêter
            </Button>
          ) : (
            <Button variant="contained" startIcon={<RocketLaunch />} onClick={handleTrain} disabled={!canTrain}>
              Lancer l'entraînement
            </Button>
          )}
        </Box>
      </Box>

      {/* ── Warnings ──────────────────────────────────────────── */}
      {!dataset && (
        <Alert severity="warning" icon={<Dataset />}>
          Aucun dataset chargé. Allez dans <strong>Données</strong> pour uploader un fichier.
        </Alert>
      )}
      {dataset && selectedModels.length === 0 && (
        <Alert severity="info">
          Aucun modèle sélectionné. Allez dans <strong>Modèles</strong> pour choisir des algorithmes.
        </Alert>
      )}

      {/* ── Training summary ──────────────────────────────────── */}
      {selectedModels.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>Résumé de l'entraînement</Typography>
            <Box className="grid gap-4 grid-cols-1 sm:grid-cols-3 mb-4">
              <Box className="flex items-center gap-3 p-3 rounded-lg bg-blue-50">
                <Dataset color="primary" />
                <Box>
                  <Typography variant="caption" color="text.secondary">Dataset</Typography>
                  <Typography variant="subtitle2">{dataset?.name || '—'}</Typography>
                </Box>
              </Box>
              <Box className="flex items-center gap-3 p-3 rounded-lg bg-purple-50">
                <Memory sx={{ color: '#7c3aed' }} />
                <Box>
                  <Typography variant="caption" color="text.secondary">Modèles</Typography>
                  <Typography variant="subtitle2">{selectedModels.length} sélectionné(s)</Typography>
                </Box>
              </Box>
              <Box className="flex items-center gap-3 p-3 rounded-lg bg-green-50">
                <Timer sx={{ color: '#16a34a' }} />
                <Box>
                  <Typography variant="caption" color="text.secondary">Statut</Typography>
                  <Typography variant="subtitle2">{running ? 'En cours…' : 'Prêt'}</Typography>
                </Box>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* ── Per-model progress ────────────────────────────────── */}
      {selectedModels.map((model) => {
        const algo = algoMap[model.algorithm];
        const state = jobStates[model.algorithm] || { progress: 0, step: 0 };
        const done = state.progress >= 100;
        return (
          <Card key={model.algorithm} sx={{ border: done ? '2px solid #16a34a' : running ? '2px solid #2563eb' : undefined }}>
            <CardContent>
              <Box className="flex items-center justify-between mb-3">
                <Box className="flex items-center gap-2">
                  {done ? <CheckCircle color="success" /> : running ? <CircularProgress size={20} /> : null}
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {algo?.name || model.algorithm}
                  </Typography>
                  <Chip label={model.fromScratch ? 'Depuis zéro' : 'Pré-entraîné'} size="small" variant="outlined" />
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {Math.round(state.progress)}%
                </Typography>
              </Box>

              <LinearProgress
                variant="determinate"
                value={state.progress}
                sx={{ height: 8, borderRadius: 4, mb: 2,
                  '& .MuiLinearProgress-bar': { borderRadius: 4, bgcolor: done ? '#16a34a' : '#2563eb' } }}
              />

              <Stepper activeStep={state.step} alternativeLabel>
                {STEPS.map((label) => (
                  <Step key={label}>
                    <StepLabel>{label}</StepLabel>
                  </Step>
                ))}
              </Stepper>

              {/* hyperparams summary */}
              <Box className="mt-3 flex flex-wrap gap-1">
                {Object.entries(model.hyperparams || {}).map(([k, v]) => (
                  <Chip key={k} label={`${k}: ${v}`} size="small" variant="outlined" />
                ))}
              </Box>
            </CardContent>
          </Card>
        );
      })}
    </Box>
  );
}
