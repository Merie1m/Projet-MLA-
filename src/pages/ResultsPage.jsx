import React, { useState, useRef } from 'react';
import {
  Card, CardContent, Typography, Box, Button, Chip, ToggleButton,
  ToggleButtonGroup, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Paper, FormControl, InputLabel, Select, MenuItem,
  Tabs, Tab, Tooltip as MuiTooltip, IconButton, Dialog, DialogTitle,
  DialogContent,
} from '@mui/material';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Cell, ScatterChart, Scatter,
  AreaChart, Area,
} from 'recharts';
import { Download, CompareArrows, ZoomIn, TableChart } from '@mui/icons-material';
import { toPng } from 'html-to-image';
import { saveAs } from 'file-saver';
import Papa from 'papaparse';
import toast from 'react-hot-toast';
import useStore from '../store/useStore';
import { ML_ALGORITHMS, METRIC_LABELS, METRIC_COLORS, REGRESSION_METRIC_LABELS, REGRESSION_ALGO_IDS } from '../constants';

/* tab panels */
function TabPanel({ children, value, index }) {
  return value === index ? <Box sx={{ pt: 3 }}>{children}</Box> : null;
}

export default function ResultsPage() {
  const { trainingResults, clearResults } = useStore();
  const [tab, setTab] = useState(0);
  const [selectedMetric, setSelectedMetric] = useState('accuracy');
  const [confusionModel, setConfusionModel] = useState('');
  const [zoomChart, setZoomChart] = useState(null);
  const chartRef = useRef(null);

  const algoMap = {};
  ML_ALGORITHMS.forEach((a) => { algoMap[a.id] = a; });

  // Separate classification and regression results
  const classifResults = trainingResults.filter((r) => !REGRESSION_ALGO_IDS.includes(r.algorithm));
  const regressionResults = trainingResults.filter((r) => REGRESSION_ALGO_IDS.includes(r.algorithm));
  const hasRegression = regressionResults.length > 0;
  const hasClassification = classifResults.length > 0;

  // Determine active metric labels based on results
  const activeMetricLabels = hasClassification ? METRIC_LABELS : REGRESSION_METRIC_LABELS;

  // Derived data
  const comparisonData = trainingResults.map((r) => ({
    name: algoMap[r.algorithm]?.name || r.algorithm,
    ...r.metrics,
  }));

  const radarMetrics = hasClassification
    ? Object.keys(METRIC_LABELS).filter((k) => k !== 'log_loss')
    : Object.keys(REGRESSION_METRIC_LABELS);

  const radarData = radarMetrics.map((key) => {
    const point = { metric: (METRIC_LABELS[key] || REGRESSION_METRIC_LABELS[key] || key) };
    trainingResults.forEach((r) => {
      point[r.algorithm] = Math.round((r.metrics[key] || 0) * 100) / 100;
    });
    return point;
  });

  const activeConfusion = trainingResults.find((r) => r.algorithm === confusionModel) || classifResults[0];

  // ── Export functions ───────────────────────────────────────────────────────
  const exportPNG = async () => {
    if (!chartRef.current) return;
    try {
      const dataUrl = await toPng(chartRef.current, { quality: 0.95, backgroundColor: '#fff' });
      saveAs(dataUrl, 'ml_results.png');
      toast.success('Graphique exporté en PNG');
    } catch { toast.error('Erreur lors de l\'export'); }
  };

  const exportCSV = () => {
    const data = trainingResults.map((r) => ({
      model: algoMap[r.algorithm]?.name || r.algorithm,
      ...r.metrics,
      duration_s: r.duration,
      trained_at: r.trainedAt,
      problem_type: r.problemType || 'classification',
      mlflow_run_id: r.mlflowRunId || '',
    }));
    const csv = Papa.unparse(data);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, 'ml_results.csv');
    toast.success('Résultats exportés en CSV');
  };

  if (trainingResults.length === 0) {
    return (
      <Box className="flex flex-col items-center justify-center py-20 text-center">
        <BarChart sx={{ fontSize: 64, color: '#cbd5e1', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">Aucun résultat disponible</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Lancez un entraînement depuis la page "Entraînement" pour voir les résultats ici.
        </Typography>
      </Box>
    );
  }

  // Build tabs dynamically
  const tabs = [
    { label: 'Comparaison', icon: <CompareArrows /> },
  ];
  if (hasClassification) {
    tabs.push({ label: 'Matrice de confusion' });
    tabs.push({ label: 'Courbes ROC' });
    tabs.push({ label: 'Courbes PR' });
  }
  if (hasRegression) {
    tabs.push({ label: 'Résidus (Régression)' });
  }
  tabs.push({ label: 'Radar' });
  tabs.push({ label: 'Tableau', icon: <TableChart /> });

  return (
    <Box className="space-y-6">
      {/* ── Header ────────────────────────────────────────────── */}
      <Box className="flex flex-wrap items-center justify-between gap-4">
        <Box>
          <Typography variant="h5">Résultats & Visualisation</Typography>
          <Typography variant="body2" color="text.secondary">
            {trainingResults.length} modèle(s) évalué(s)
            {hasRegression && hasClassification && ' (Classification + Régression)'}
            {hasRegression && !hasClassification && ' (Régression)'}
          </Typography>
        </Box>
        <Box className="flex gap-2">
          <Button variant="outlined" startIcon={<Download />} onClick={exportPNG}>PNG</Button>
          <Button variant="outlined" startIcon={<Download />} onClick={exportCSV}>CSV</Button>
          <Button variant="text" color="error" onClick={clearResults}>Effacer</Button>
        </Box>
      </Box>

      {/* ── Tabs ──────────────────────────────────────────────── */}
      <Card>
        <CardContent>
          <Tabs value={tab} onChange={(_, v) => setTab(v)} variant="scrollable" scrollButtons="auto">
            {tabs.map((t, i) => (
              <Tab key={i} label={t.label} icon={t.icon} iconPosition="start" />
            ))}
          </Tabs>
        </CardContent>
      </Card>

      <Box ref={chartRef}>
        {/* ── Tab 0: Bar comparison ───────────────────────────── */}
        <TabPanel value={tab} index={0}>
          <Card>
            <CardContent>
              <Box className="flex items-center gap-4 mb-4">
                <FormControl size="small" sx={{ minWidth: 160 }}>
                  <InputLabel>Métrique</InputLabel>
                  <Select value={selectedMetric} label="Métrique" onChange={(e) => setSelectedMetric(e.target.value)}>
                    {Object.entries(METRIC_LABELS).map(([k, v]) => (
                      <MenuItem key={k} value={k}>{v}</MenuItem>
                    ))}
                    {hasRegression && Object.entries(REGRESSION_METRIC_LABELS).map(([k, v]) => (
                      <MenuItem key={k} value={k}>{v}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={comparisonData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" angle={-20} textAnchor="end" tick={{ fontSize: 12 }} />
                  <YAxis domain={selectedMetric === 'log_loss' || selectedMetric === 'mse' || selectedMetric === 'mae' || selectedMetric === 'rmse' ? [0, 'auto'] : [0, 'auto']} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey={selectedMetric} fill="#2563eb" radius={[6, 6, 0, 0]}>
                    {comparisonData.map((_, i) => (
                      <Cell key={i} fill={METRIC_COLORS[i % METRIC_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabPanel>

        {/* Dynamic tab indexing */}
        {(() => {
          let tabIdx = 1;

          const panels = [];

          if (hasClassification) {
            // Confusion Matrix
            panels.push(
              <TabPanel key="confusion" value={tab} index={tabIdx++}>
                <Card>
                  <CardContent>
                    {classifResults.length > 1 && (
                      <FormControl size="small" sx={{ minWidth: 200, mb: 3 }}>
                        <InputLabel>Modèle</InputLabel>
                        <Select value={confusionModel || classifResults[0]?.algorithm} label="Modèle"
                          onChange={(e) => setConfusionModel(e.target.value)}>
                          {classifResults.map((r) => (
                            <MenuItem key={r.algorithm} value={r.algorithm}>
                              {algoMap[r.algorithm]?.name || r.algorithm}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    )}
                    {activeConfusion?.confusion && activeConfusion.confusion.length > 0 && (
                      <Box className="flex justify-center">
                        <Box>
                          <Typography variant="subtitle2" align="center" sx={{ mb: 1 }}>
                            {algoMap[activeConfusion.algorithm]?.name}
                          </Typography>
                          <Table size="small" sx={{ width: 'auto' }}>
                            <TableHead>
                              <TableRow>
                                <TableCell />
                                {activeConfusion.confusion[0]?.map((_, ci) => (
                                  <TableCell key={ci} align="center" sx={{ fontWeight: 700 }}>Classe {ci}</TableCell>
                                ))}
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {activeConfusion.confusion.map((row, ri) => (
                                <TableRow key={ri}>
                                  <TableCell sx={{ fontWeight: 700 }}>Classe {ri}</TableCell>
                                  {row.map((val, ci) => (
                                    <TableCell key={ci} align="center" sx={{
                                      bgcolor: ri === ci ? '#dcfce7' : '#fef2f2',
                                      fontWeight: 700, fontSize: 18,
                                    }}>
                                      {val}
                                    </TableCell>
                                  ))}
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </Box>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </TabPanel>
            );

            // ROC curves
            panels.push(
              <TabPanel key="roc" value={tab} index={tabIdx++}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>Courbes ROC</Typography>
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: 'FPR', position: 'bottom' }} tick={{ fontSize: 12 }} />
                        <YAxis domain={[0, 1]} label={{ value: 'TPR', angle: -90, position: 'insideLeft' }} tick={{ fontSize: 12 }} />
                        <Tooltip />
                        <Legend />
                        <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" stroke="#cbd5e1" strokeDasharray="5 5" name="Random" dot={false} />
                        {classifResults.map((r, i) => (
                          <Line
                            key={r.algorithm}
                            data={r.roc}
                            dataKey="tpr"
                            stroke={METRIC_COLORS[i % METRIC_COLORS.length]}
                            name={`${algoMap[r.algorithm]?.name || r.algorithm} (AUC: ${r.metrics.roc_auc?.toFixed(3)})`}
                            dot={false}
                            strokeWidth={2}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </TabPanel>
            );

            // PR curves
            panels.push(
              <TabPanel key="pr" value={tab} index={tabIdx++}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>Courbes Precision-Recall</Typography>
                    <ResponsiveContainer width="100%" height={400}>
                      <AreaChart margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="recall" type="number" domain={[0, 1]} label={{ value: 'Recall', position: 'bottom' }} tick={{ fontSize: 12 }} />
                        <YAxis domain={[0, 1]} label={{ value: 'Precision', angle: -90, position: 'insideLeft' }} tick={{ fontSize: 12 }} />
                        <Tooltip />
                        <Legend />
                        {classifResults.map((r, i) => (
                          <Area
                            key={r.algorithm}
                            data={r.pr}
                            dataKey="precision"
                            stroke={METRIC_COLORS[i % METRIC_COLORS.length]}
                            fill={METRIC_COLORS[i % METRIC_COLORS.length]}
                            fillOpacity={0.1}
                            name={algoMap[r.algorithm]?.name || r.algorithm}
                            dot={false}
                            strokeWidth={2}
                          />
                        ))}
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </TabPanel>
            );
          }

          if (hasRegression) {
            // Residuals scatter
            panels.push(
              <TabPanel key="residuals" value={tab} index={tabIdx++}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>Actual vs Predicted (Régression)</Typography>
                    {regressionResults.map((r, i) => (
                      <Box key={r.algorithm} sx={{ mb: 4 }}>
                        <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                          {algoMap[r.algorithm]?.name || r.algorithm} — R²: {r.metrics.r2?.toFixed(4)}
                        </Typography>
                        <ResponsiveContainer width="100%" height={350}>
                          <ScatterChart margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                            <XAxis dataKey="actual" type="number" name="Actual" label={{ value: 'Valeur réelle', position: 'bottom' }} tick={{ fontSize: 12 }} />
                            <YAxis dataKey="predicted" type="number" name="Predicted" label={{ value: 'Prédiction', angle: -90, position: 'insideLeft' }} tick={{ fontSize: 12 }} />
                            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                            <Scatter
                              data={r.residuals || []}
                              fill={METRIC_COLORS[i % METRIC_COLORS.length]}
                              fillOpacity={0.6}
                            />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </Box>
                    ))}
                  </CardContent>
                </Card>
              </TabPanel>
            );
          }

          // Radar
          panels.push(
            <TabPanel key="radar" value={tab} index={tabIdx++}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle2" sx={{ mb: 2 }}>Comparaison Radar</Typography>
                  <ResponsiveContainer width="100%" height={450}>
                    <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                      <PolarGrid stroke="#e2e8f0" />
                      <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
                      <PolarRadiusAxis tick={{ fontSize: 10 }} />
                      <Legend />
                      {trainingResults.map((r, i) => (
                        <Radar
                          key={r.algorithm}
                          name={algoMap[r.algorithm]?.name || r.algorithm}
                          dataKey={r.algorithm}
                          stroke={METRIC_COLORS[i % METRIC_COLORS.length]}
                          fill={METRIC_COLORS[i % METRIC_COLORS.length]}
                          fillOpacity={0.15}
                          strokeWidth={2}
                        />
                      ))}
                    </RadarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabPanel>
          );

          // Table
          const allMetricLabels = { ...METRIC_LABELS, ...REGRESSION_METRIC_LABELS };
          const visibleMetrics = [...new Set([
            ...trainingResults.flatMap((r) => Object.keys(r.metrics)),
          ])];

          panels.push(
            <TabPanel key="table" value={tab} index={tabIdx++}>
              <Card>
                <CardContent>
                  <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Modèle</TableCell>
                          {visibleMetrics.map((k) => (
                            <TableCell key={k} align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>
                              {allMetricLabels[k] || k}
                            </TableCell>
                          ))}
                          <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Durée (s)</TableCell>
                          <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>MLflow Run</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {trainingResults.map((r) => (
                          <TableRow key={r.algorithm} hover>
                            <TableCell sx={{ fontWeight: 600 }}>
                              {algoMap[r.algorithm]?.name || r.algorithm}
                            </TableCell>
                            {visibleMetrics.map((k) => (
                              <TableCell key={k} align="center">
                                {r.metrics[k] !== undefined ? r.metrics[k].toFixed(4) : '—'}
                              </TableCell>
                            ))}
                            <TableCell align="center">{r.duration}s</TableCell>
                            <TableCell align="center">
                              {r.mlflowRunId ? (
                                <Chip label={r.mlflowRunId.slice(0, 8)} size="small" variant="outlined" color="secondary" />
                              ) : '—'}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </TabPanel>
          );

          return panels;
        })()}
      </Box>
    </Box>
  );
}
