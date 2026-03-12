import React, { useState, useMemo } from 'react';
import {
  Card, CardContent, Typography, Box, Button, Chip, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow, Paper, IconButton,
  Dialog, DialogTitle, DialogContent, DialogActions, Alert,
  Divider, TextField, InputAdornment,
} from '@mui/material';
import {
  History, Restore, Compare, Download, Search, Science,
  CalendarMonth, CheckCircle, Speed, Delete, Visibility,
} from '@mui/icons-material';
import toast from 'react-hot-toast';
import useStore from '../store/useStore';
import { ML_ALGORITHMS, METRIC_LABELS, METRIC_COLORS, REGRESSION_METRIC_LABELS } from '../constants';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell,
} from 'recharts';

export default function MLOpsPage() {
  const { trainingResults, experiments, setExperiments } = useStore();
  const [search, setSearch] = useState('');
  const [detailDialog, setDetailDialog] = useState(null);
  const [compareMode, setCompareMode] = useState(false);
  const [compareSelection, setCompareSelection] = useState([]);

  const algoMap = {};
  ML_ALGORITHMS.forEach((a) => { algoMap[a.id] = a; });

  // Build experiment history from training results
  const allExperiments = useMemo(() => {
    return trainingResults.map((r, i) => ({
      id: r.experimentId,
      algorithm: r.algorithm,
      name: algoMap[r.algorithm]?.name || r.algorithm,
      metrics: r.metrics,
      trainedAt: r.trainedAt,
      duration: r.duration,
      version: `v1.${i}`,
      datasetVersion: 'v1.0',
      status: 'completed',
      mlflowRunId: r.mlflowRunId || null,
      problemType: r.problemType || 'classification',
    }));
  }, [trainingResults]);

  const filtered = allExperiments.filter((e) =>
    e.name.toLowerCase().includes(search.toLowerCase()) ||
    e.algorithm.toLowerCase().includes(search.toLowerCase())
  );

  const toggleCompare = (id) => {
    setCompareSelection((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const comparedExps = allExperiments.filter((e) => compareSelection.includes(e.id));

  const comparisonChartData = Object.keys(METRIC_LABELS).filter((k) => k !== 'log_loss').map((key) => {
    const point = { metric: METRIC_LABELS[key] };
    comparedExps.forEach((e) => {
      point[e.name] = Math.round((e.metrics[key] || 0) * 10000) / 10000;
    });
    return point;
  });

  const handleRollback = (exp) => {
    toast.success(`Rollback vers ${exp.name} ${exp.version}`);
  };

  const handleExportModel = (exp) => {
    toast.success(`Export du modèle ${exp.name} (joblib)`);
  };

  if (allExperiments.length === 0) {
    return (
      <Box className="flex flex-col items-center justify-center py-20 text-center">
        <History sx={{ fontSize: 64, color: '#cbd5e1', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">Aucune expérimentation enregistrée</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Lancez un entraînement pour voir l'historique ici. Intégration MLflow / DVC disponible.
        </Typography>
      </Box>
    );
  }

  return (
    <Box className="space-y-6">
      {/* Header */}
      <Box className="flex flex-wrap items-center justify-between gap-4">
        <Box>
          <Typography variant="h5">MLOps & Historique</Typography>
          <Typography variant="body2" color="text.secondary">
            {allExperiments.length} expérimentation(s) — Suivi des versions et rollback
          </Typography>
        </Box>
        <Box className="flex gap-2">
          <Button
            variant={compareMode ? 'contained' : 'outlined'}
            startIcon={<Compare />}
            color={compareMode ? 'secondary' : 'primary'}
            onClick={() => { setCompareMode(!compareMode); setCompareSelection([]); }}
          >
            {compareMode ? 'Quitter comparaison' : 'Comparer'}
          </Button>
        </Box>
      </Box>

      {/* Search */}
      <TextField
        size="small" fullWidth placeholder="Rechercher une expérimentation…"
        value={search} onChange={(e) => setSearch(e.target.value)}
        InputProps={{ startAdornment: <InputAdornment position="start"><Search /></InputAdornment> }}
      />

      {compareMode && compareSelection.length >= 2 && (
        <Alert severity="info" icon={<Compare />}>
          {compareSelection.length} expérimentations sélectionnées pour comparaison
        </Alert>
      )}

      {/* ── Comparison chart ──────────────────────────────────── */}
      {compareMode && comparedExps.length >= 2 && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>Comparaison des expérimentations</Typography>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={comparisonChartData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="metric" angle={-15} textAnchor="end" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                {comparedExps.map((e, i) => (
                  <Bar key={e.id} dataKey={e.name} fill={METRIC_COLORS[i % METRIC_COLORS.length]} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* ── Experiments Table ─────────────────────────────────── */}
      <Card>
        <CardContent>
          <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
            <Table>
              <TableHead>
                <TableRow>
                  {compareMode && <TableCell padding="checkbox" sx={{ bgcolor: '#f1f5f9' }} />}
                  <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Modèle</TableCell>
                  <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Version</TableCell>
                  <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Type</TableCell>
                  <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Dataset</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Score principal</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>F1 / R²</TableCell>
                  <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>MLflow Run</TableCell>
                  <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Date</TableCell>
                  <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Durée</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filtered.map((exp) => (
                  <TableRow
                    key={exp.id}
                    hover
                    selected={compareSelection.includes(exp.id)}
                    onClick={() => compareMode && toggleCompare(exp.id)}
                    sx={{ cursor: compareMode ? 'pointer' : 'default' }}
                  >
                    {compareMode && (
                      <TableCell padding="checkbox">
                        <input type="checkbox" checked={compareSelection.includes(exp.id)} readOnly />
                      </TableCell>
                    )}
                    <TableCell>
                      <Box className="flex items-center gap-2">
                        <Science fontSize="small" color="primary" />
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>{exp.name}</Typography>
                      </Box>
                    </TableCell>
                    <TableCell><Chip label={exp.version} size="small" variant="outlined" color="primary" /></TableCell>
                    <TableCell>
                      <Chip
                        label={exp.problemType === 'regression' ? 'Régression' : 'Classification'}
                        size="small"
                        variant="outlined"
                        color={exp.problemType === 'regression' ? 'warning' : 'info'}
                      />
                    </TableCell>
                    <TableCell><Chip label={exp.datasetVersion} size="small" variant="outlined" /></TableCell>
                    <TableCell align="center">
                      {exp.problemType === 'regression' ? (
                        <Chip
                          label={`R² ${exp.metrics.r2?.toFixed(4) ?? '—'}`}
                          size="small"
                          color={exp.metrics.r2 > 0.9 ? 'success' : exp.metrics.r2 > 0.7 ? 'primary' : 'warning'}
                        />
                      ) : (
                        <Chip
                          label={(exp.metrics.accuracy * 100).toFixed(1) + '%'}
                          size="small"
                          color={exp.metrics.accuracy > 0.9 ? 'success' : exp.metrics.accuracy > 0.8 ? 'primary' : 'warning'}
                        />
                      )}
                    </TableCell>
                    <TableCell align="center">
                      {exp.problemType === 'regression'
                        ? (exp.metrics.mae?.toFixed(4) ?? '—')
                        : (exp.metrics.f1_score?.toFixed(4) ?? '—')}
                    </TableCell>
                    <TableCell>
                      {exp.mlflowRunId ? (
                        <Chip label={exp.mlflowRunId.slice(0, 8)} size="small" variant="outlined" color="secondary" />
                      ) : (
                        <Typography variant="caption" color="text.secondary">—</Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption">
                        {new Date(exp.trainedAt).toLocaleString('fr-FR')}
                      </Typography>
                    </TableCell>
                    <TableCell>{exp.duration}s</TableCell>
                    <TableCell align="center">
                      <Box className="flex gap-0.5 justify-center">
                        <IconButton size="small" onClick={(e) => { e.stopPropagation(); setDetailDialog(exp); }}>
                          <Visibility fontSize="small" />
                        </IconButton>
                        <IconButton size="small" color="secondary" onClick={(e) => { e.stopPropagation(); handleRollback(exp); }}>
                          <Restore fontSize="small" />
                        </IconButton>
                        <IconButton size="small" color="primary" onClick={(e) => { e.stopPropagation(); handleExportModel(exp); }}>
                          <Download fontSize="small" />
                        </IconButton>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* ── Detail Dialog ─────────────────────────────────────── */}
      <Dialog open={!!detailDialog} onClose={() => setDetailDialog(null)} maxWidth="sm" fullWidth>
        <DialogTitle>Détails de l'expérimentation</DialogTitle>
        <DialogContent>
          {detailDialog && (
            <Box className="space-y-3 mt-2">
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{detailDialog.name}</Typography>
              <Chip label={detailDialog.version} color="primary" size="small" />
              <Chip label={`Dataset ${detailDialog.datasetVersion}`} variant="outlined" size="small" sx={{ ml: 1 }} />
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" sx={{ mb: 1 }}>Métriques</Typography>
              <Box className="grid grid-cols-2 gap-2">
                {Object.entries(detailDialog.metrics).map(([k, v]) => (
                  <Box key={k} className="flex justify-between p-2 rounded-lg bg-slate-50">
                    <Typography variant="body2" color="text.secondary">
                      {METRIC_LABELS[k] || REGRESSION_METRIC_LABELS[k] || k}
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{v.toFixed(4)}</Typography>
                  </Box>
                ))}
              </Box>
              <Divider sx={{ my: 2 }} />
              <Typography variant="caption" color="text.secondary">
                Entraîné le {new Date(detailDialog.trainedAt).toLocaleString('fr-FR')} — Durée: {detailDialog.duration}s
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { handleExportModel(detailDialog); setDetailDialog(null); }} startIcon={<Download />}>
            Exporter (joblib)
          </Button>
          <Button onClick={() => setDetailDialog(null)}>Fermer</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
