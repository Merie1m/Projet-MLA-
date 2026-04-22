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
  const { 
    trainingResults, clearResults, 
    biasVarianceData, setBiasVarianceData,
    stabilityData, setStabilityData,
  } = useStore();
  const [tab, setTab] = useState(0);
  const [selectedMetric, setSelectedMetric] = useState('accuracy');
  const [confusionModel, setConfusionModel] = useState('');
  const [zoomChart, setZoomChart] = useState(null);
  const [biasVarianceLoading, setBiasVarianceLoading] = useState(false);
  const [biasVarianceAlgorithm, setBiasVarianceAlgorithm] = useState('');
  const [stabilityLoading, setStabilityLoading] = useState(false);
  const [stabilityAlgorithm, setStabilityAlgorithm] = useState('');
  const chartRef = useRef(null);

  const algoMap = {};
  ML_ALGORITHMS.forEach((a) => { algoMap[a.id] = a; });

  // Separate classification and regression results
  const classifResults = trainingResults.filter((r) => !REGRESSION_ALGO_IDS.includes(r.algorithm));
  const regressionResults = trainingResults.filter((r) => REGRESSION_ALGO_IDS.includes(r.algorithm));
  const hasRegression = regressionResults.length > 0;
  const hasClassification = classifResults.length > 0;

  // Pre-filter classifiers for bias/variance and stability analysis
  const biasVarianceClassifiers = classifResults.filter((r) => ['random_forest', 'decision_tree'].includes(r.algorithm));
  const stabilityClassifiers = classifResults.filter((r) => ['random_forest', 'decision_tree'].includes(r.algorithm));

  // Initialize algorithms on first render when classifResults becomes available
  React.useEffect(() => {
    if (!biasVarianceAlgorithm && classifResults.length > 0) {
      const rfOrDt = classifResults.find((r) => ['random_forest', 'decision_tree'].includes(r.algorithm));
      if (rfOrDt) setBiasVarianceAlgorithm(rfOrDt.algorithm);
    }
  }, [classifResults.length]);

  React.useEffect(() => {
    if (!stabilityAlgorithm && classifResults.length > 0) {
      const rfOrDt = classifResults.find((r) => ['random_forest', 'decision_tree'].includes(r.algorithm));
      if (rfOrDt) setStabilityAlgorithm(rfOrDt.algorithm);
    }
  }, [classifResults.length]);

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

  const fetchBiasVarianceAnalysis = async (algorithm) => {
    // Récupérer les données depuis le premier résultat d'entraînement
    const firstResult = trainingResults[0];
    if (!firstResult) {
      toast.error('Aucun résultat d\'entraînement disponible');
      return;
    }
    
    const datasetId = firstResult.datasetId;
    const targetCol = firstResult.targetColumn;
    
    if (!datasetId || !targetCol) {
      toast.error('Dataset ou colonne cible non disponible');
      console.error('Dataset ID:', datasetId, 'Target Column:', targetCol);
      return;
    }
    
    setBiasVarianceLoading(true);
    try {
      const params = new URLSearchParams({
        dataset_id: datasetId,
        target_column: targetCol,
        algorithm: algorithm,
      });
      const response = await fetch(`http://localhost:8000/api/bias-variance-analysis?${params}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      // Stocker les résultats complets dans le store pour persistance
      const resultsData = {
        algorithm: data.algorithm,
        results: data.results || [],
        mlflow_run_id: data.mlflow_run_id,
      };
      setBiasVarianceData(resultsData);
      toast.success('Analyse Bias/Variance complétée');
    } catch (error) {
      toast.error(`Erreur: ${error.message}`);
      console.error('Bias/Variance error:', error);
    } finally {
      setBiasVarianceLoading(false);
    }
  };

  const fetchStabilityAnalysis = async (algorithm) => {
    // Récupérer les données depuis le premier résultat d'entraînement
    const firstResult = trainingResults[0];
    if (!firstResult) {
      toast.error('Aucun résultat d\'entraînement disponible');
      return;
    }
    
    const datasetId = firstResult.datasetId;
    const targetCol = firstResult.targetColumn;
    
    if (!datasetId || !targetCol) {
      toast.error('Dataset ou colonne cible non disponible');
      console.error('Dataset ID:', datasetId, 'Target Column:', targetCol);
      return;
    }
    
    setStabilityLoading(true);
    try {
      const params = new URLSearchParams({
        dataset_id: datasetId,
        target_column: targetCol,
        algorithm: algorithm,
      });
      const response = await fetch(`http://localhost:8000/api/stability-analysis?${params}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      // Stocker les résultats complets dans le store pour persistance
      const stabilityResultsData = {
        algorithm: data.algorithm,
        results: data.results || [],
        statistics: data.statistics || {},
        mlflow_run_id: data.mlflow_run_id,
      };
      setStabilityData(stabilityResultsData);
      toast.success('Analyse Stabilité complétée');
    } catch (error) {
      toast.error(`Erreur: ${error.message}`);
      console.error('Stability error:', error);
    } finally {
      setStabilityLoading(false);
    }
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
    tabs.push({ label: 'Importance des Features' });
    tabs.push({ label: 'Analyse Erreurs' });
    tabs.push({ label: 'Analyse Bias/Variance' });
    tabs.push({ label: 'Analyse Stabilité' });
    
    // Only add RF vs DT comparison if both algorithms are trained
    if (classifResults.some((r) => r.algorithm === 'random_forest') && classifResults.some((r) => r.algorithm === 'decision_tree')) {
      tabs.push({ label: 'RF vs Decision Tree' });
    }
  }
  if (hasRegression) {
    tabs.push({ label: 'Résidus (Régression)' });
    tabs.push({ label: 'Importance des Features' });
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

            // ROC curves with per-class display and zoom
            panels.push(
              <TabPanel key="roc" value={tab} index={tabIdx++}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>Courbes ROC (par classe avec zoom supérieur-gauche)</Typography>
                    <ResponsiveContainer width="100%" height={500}>
                      <LineChart margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis 
                          dataKey="fpr" 
                          type="number" 
                          domain={[0, 0.3]} 
                          label={{ value: 'FPR (False Positive Rate)', position: 'bottom' }} 
                          tick={{ fontSize: 12 }} 
                        />
                        <YAxis 
                          domain={[0.7, 1.05]} 
                          label={{ value: 'TPR (True Positive Rate)', angle: -90, position: 'insideLeft' }} 
                          tick={{ fontSize: 12 }} 
                        />
                        <Tooltip />
                        <Legend wrapperStyle={{ paddingTop: '20px' }} />
                        {/* Diagonal reference line (random classifier) */}
                        <Line 
                          data={[{ fpr: 0, tpr: 0.7 }, { fpr: 0.3, tpr: 1.0 }]} 
                          dataKey="tpr" 
                          stroke="#cbd5e1" 
                          strokeDasharray="5 5" 
                          name="Random (AUC=0.5)" 
                          dot={false} 
                          strokeWidth={2}
                        />
                        {classifResults.map((r, algoIdx) => {
                          const rocData = r.roc;
                          // Handle both old format (array) and new format (object with class keys)
                          const isNewFormat = typeof rocData === 'object' && !Array.isArray(rocData);
                          
                          if (isNewFormat) {
                            // New format: per-class curves
                            const classes = Object.keys(rocData).sort();
                            return classes.map((classKey, classIdx) => (
                              <Line
                                key={`${r.algorithm}-${classKey}`}
                                data={rocData[classKey]}
                                dataKey="tpr"
                                stroke={METRIC_COLORS[(algoIdx * 3 + classIdx) % METRIC_COLORS.length]}
                                name={`${algoMap[r.algorithm]?.name || r.algorithm} — ${classKey}`}
                                dot={{ r: 4, fill: METRIC_COLORS[(algoIdx * 3 + classIdx) % METRIC_COLORS.length] }}
                                strokeWidth={3}
                                isAnimationActive={false}
                              />
                            ));
                          } else {
                            // Old format: single curve (backward compatibility)
                            return (
                              <Line
                                key={r.algorithm}
                                data={rocData}
                                dataKey="tpr"
                                stroke={METRIC_COLORS[algoIdx % METRIC_COLORS.length]}
                                name={`${algoMap[r.algorithm]?.name || r.algorithm} (AUC: ${r.metrics.roc_auc?.toFixed(3)})`}
                                dot={{ r: 4, fill: METRIC_COLORS[algoIdx % METRIC_COLORS.length] }}
                                strokeWidth={3}
                                isAnimationActive={false}
                              />
                            );
                          }
                        })}
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </TabPanel>
            );

            // PR curves with per-class display and zoom
            panels.push(
              <TabPanel key="pr" value={tab} index={tabIdx++}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>Courbes Precision-Recall (par classe avec zoom supérieur-droit)</Typography>
                    <ResponsiveContainer width="100%" height={500}>
                      <LineChart margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis 
                          dataKey="recall" 
                          type="number" 
                          domain={[0.7, 1.05]} 
                          label={{ value: 'Recall', position: 'bottom' }} 
                          tick={{ fontSize: 12 }} 
                        />
                        <YAxis 
                          domain={[0.7, 1.05]} 
                          label={{ value: 'Precision', angle: -90, position: 'insideLeft' }} 
                          tick={{ fontSize: 12 }} 
                        />
                        <Tooltip />
                        <Legend wrapperStyle={{ paddingTop: '20px' }} />
                        {classifResults.map((r, algoIdx) => {
                          const prData = r.pr;
                          // Handle both old format (array) and new format (object with class keys)
                          const isNewFormat = typeof prData === 'object' && !Array.isArray(prData);
                          
                          if (isNewFormat) {
                            // New format: per-class curves
                            const classes = Object.keys(prData).sort();
                            return classes.map((classKey, classIdx) => (
                              <Line
                                key={`${r.algorithm}-${classKey}`}
                                data={prData[classKey]}
                                dataKey="precision"
                                stroke={METRIC_COLORS[(algoIdx * 3 + classIdx) % METRIC_COLORS.length]}
                                name={`${algoMap[r.algorithm]?.name || r.algorithm} — ${classKey}`}
                                dot={{ r: 4, fill: METRIC_COLORS[(algoIdx * 3 + classIdx) % METRIC_COLORS.length] }}
                                strokeWidth={3}
                                isAnimationActive={false}
                              />
                            ));
                          } else {
                            // Old format: single curve (backward compatibility)
                            return (
                              <Line
                                key={r.algorithm}
                                data={prData}
                                dataKey="precision"
                                stroke={METRIC_COLORS[algoIdx % METRIC_COLORS.length]}
                                name={algoMap[r.algorithm]?.name || r.algorithm}
                                dot={{ r: 4, fill: METRIC_COLORS[algoIdx % METRIC_COLORS.length] }}
                                strokeWidth={3}
                                isAnimationActive={false}
                              />
                            );
                          }
                        })}
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </TabPanel>
            );
          }

          // Feature Importances (for classification or regression models)
          const hasFeatureImportances = trainingResults.some((r) => r.featureImportances && r.featureImportances.length > 0);
          if (hasFeatureImportances) {
            panels.push(
              <TabPanel key="feature-importances" value={tab} index={tabIdx++}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>Importance des Features</Typography>
                    {trainingResults.map((r, algoIdx) => {
                      const features = r.featureImportances || [];
                      if (features.length === 0) return null;
                      
                      return (
                        <Box key={r.algorithm} sx={{ mb: 4 }}>
                          <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                            {algoMap[r.algorithm]?.name || r.algorithm}
                          </Typography>
                          <ResponsiveContainer width="100%" height={features.length * 30 + 40}>
                            <BarChart
                              data={features}
                              layout="vertical"
                              margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                              <XAxis type="number" tick={{ fontSize: 12 }} />
                              <YAxis dataKey="name" type="category" width={145} tick={{ fontSize: 12 }} />
                              <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                              <Bar dataKey="importance" fill={METRIC_COLORS[algoIdx % METRIC_COLORS.length]} radius={[0, 6, 6, 0]} />
                            </BarChart>
                          </ResponsiveContainer>
                        </Box>
                      );
                    })}
                  </CardContent>
                </Card>
              </TabPanel>
            );
          }

          // Error Analysis (classification only)
          if (hasClassification) {
            panels.push(
              <TabPanel key="error-analysis" value={tab} index={tabIdx++}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>Analyse des Erreurs</Typography>
                    {classifResults.map((r, algoIdx) => {
                      const errors = r.errorAnalysis || [];
                      if (errors.length === 0) {
                        return (
                          <Box key={r.algorithm} sx={{ mb: 2, p: 2, bgcolor: '#ecfdf5', borderRadius: 1 }}>
                            <Typography variant="body2" sx={{ fontWeight: 600, color: '#059669' }}>
                              ✓ {algoMap[r.algorithm]?.name || r.algorithm} — Aucune erreur
                            </Typography>
                          </Box>
                        );
                      }
                      
                      return (
                        <Box key={r.algorithm} sx={{ mb: 4 }}>
                          <Typography variant="body2" sx={{ fontWeight: 600, mb: 2 }}>
                            {algoMap[r.algorithm]?.name || r.algorithm} — {errors.length} erreur(s) détectée(s)
                          </Typography>
                          {errors.map((err, errIdx) => (
                            <Box key={errIdx} sx={{ p: 2, mb: 1, border: '1px solid #e2e8f0', borderRadius: 1, bgcolor: '#fef2f2' }}>
                              <Box sx={{ display: 'flex', gap: 4, mb: 1 }}>
                                <Box>
                                  <Typography variant="caption" color="text.secondary">Classe réelle</Typography>
                                  <Chip label={err.actual_class} size="small" variant="outlined" />
                                </Box>
                                <Box>
                                  <Typography variant="caption" color="text.secondary">Classe prédite</Typography>
                                  <Chip label={err.predicted_class} size="small" color="error" variant="filled" />
                                </Box>
                                <Box>
                                  <Typography variant="caption" color="text.secondary">Confiance</Typography>
                                  <Chip label={`${(err.confidence * 100).toFixed(1)}%`} size="small" />
                                </Box>
                              </Box>
                              <Typography variant="caption" sx={{ display: 'block', mt: 1, fontFamily: 'monospace', color: '#666' }}>
                                Valeurs features: {JSON.stringify(err.features).slice(0, 80)}...
                              </Typography>
                            </Box>
                          ))}
                        </Box>
                      );
                    })}
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

          // Bias/Variance Analysis (classification only)
          if (hasClassification) {
            panels.push(
              <TabPanel key="bias-variance" value={tab} index={tabIdx++}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>Analyse Bias/Variance</Typography>
                    <Box sx={{ mb: 2 }}>
                      <FormControl size="small" sx={{ minWidth: 160 }}>
                        <InputLabel>Algorithme</InputLabel>
                        <Select
                          value={biasVarianceAlgorithm}
                          label="Algorithme"
                          onChange={(e) => setBiasVarianceAlgorithm(e.target.value)}
                        >
                          {biasVarianceClassifiers.map((r) => (
                            <MenuItem key={r.algorithm} value={r.algorithm}>
                              {algoMap[r.algorithm]?.name || r.algorithm}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <Button
                        variant="contained"
                        sx={{ ml: 2 }}
                        onClick={() => fetchBiasVarianceAnalysis(biasVarianceAlgorithm)}
                        disabled={biasVarianceLoading || !biasVarianceAlgorithm}
                      >
                        {biasVarianceLoading ? 'Analyse en cours...' : 'Lancer l\'analyse'}
                      </Button>
                    </Box>
                    {biasVarianceData && biasVarianceData.results && biasVarianceData.results.length > 0 && (
                      <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>n_estimators</TableCell>
                              <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>max_depth</TableCell>
                              <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Train Acc</TableCell>
                              <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Test Acc</TableCell>
                              <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Bias</TableCell>
                              <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Variance</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {biasVarianceData.results.map((row, idx) => (
                              <TableRow key={idx} hover>
                                <TableCell>{row.n_estimators || '—'}</TableCell>
                                <TableCell>{row.max_depth || '∞'}</TableCell>
                                <TableCell align="center">{row.train_score?.toFixed(4)}</TableCell>
                                <TableCell align="center">{row.test_score?.toFixed(4)}</TableCell>
                                <TableCell align="center" sx={{ color: row.bias > 0.1 ? '#dc2626' : '#059669' }}>
                                  {row.bias?.toFixed(4)}
                                </TableCell>
                                <TableCell align="center" sx={{ color: row.variance > 0.15 ? '#dc2626' : '#059669' }}>
                                  {row.variance?.toFixed(4)}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    )}
                  </CardContent>
                </Card>
              </TabPanel>
            );
          }

          // Stability Analysis (classification only)
          if (hasClassification) {
            panels.push(
              <TabPanel key="stability" value={tab} index={tabIdx++}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>Analyse Stabilité</Typography>
                    <Box sx={{ mb: 2 }}>
                      <FormControl size="small" sx={{ minWidth: 160 }}>
                        <InputLabel>Algorithme</InputLabel>
                        <Select
                          value={stabilityAlgorithm}
                          label="Algorithme"
                          onChange={(e) => setStabilityAlgorithm(e.target.value)}
                        >
                          {stabilityClassifiers.map((r) => (
                            <MenuItem key={r.algorithm} value={r.algorithm}>
                              {algoMap[r.algorithm]?.name || r.algorithm}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <Button
                        variant="contained"
                        sx={{ ml: 2 }}
                        onClick={() => fetchStabilityAnalysis(stabilityAlgorithm)}
                        disabled={stabilityLoading || !stabilityAlgorithm}
                      >
                        {stabilityLoading ? 'Analyse en cours...' : 'Lancer l\'analyse'}
                      </Button>
                    </Box>
                    {stabilityData && stabilityData.results.length > 0 && (
                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 600, mb: 2 }}>Résultats par Random State</Typography>
                        <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2, mb: 3 }}>
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Random State</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Train Score</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Test Score</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Accuracy</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Precision</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Recall</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>F1</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {stabilityData.results.map((row, idx) => (
                                <TableRow key={idx} hover>
                                  <TableCell>{row.random_state}</TableCell>
                                  <TableCell align="center">{row.train_score?.toFixed(4)}</TableCell>
                                  <TableCell align="center">{row.test_score?.toFixed(4)}</TableCell>
                                  <TableCell align="center">{row.accuracy?.toFixed(4)}</TableCell>
                                  <TableCell align="center">{row.precision?.toFixed(4)}</TableCell>
                                  <TableCell align="center">{row.recall?.toFixed(4)}</TableCell>
                                  <TableCell align="center">{row.f1?.toFixed(4)}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>

                        <Typography variant="body2" sx={{ fontWeight: 600, mb: 2 }}>Statistiques</Typography>
                        <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Métrique</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Moyenne</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Std Dev</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Min</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Max</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {Object.entries(stabilityData.statistics || {}).map(([metric, stats]) => (
                                <TableRow key={metric} hover>
                                  <TableCell sx={{ fontWeight: 600 }}>{metric}</TableCell>
                                  <TableCell align="center">{stats.mean?.toFixed(4)}</TableCell>
                                  <TableCell align="center" sx={{ color: stats.std > 0.05 ? '#dc2626' : '#059669' }}>
                                    {stats.std?.toFixed(4)}
                                  </TableCell>
                                  <TableCell align="center">{stats.min?.toFixed(4)}</TableCell>
                                  <TableCell align="center">{stats.max?.toFixed(4)}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </TabPanel>
            );
          }

          // RF vs Decision Tree Comparison
          if (hasClassification && classifResults.some((r) => r.algorithm === 'random_forest') && classifResults.some((r) => r.algorithm === 'decision_tree')) {
            const rfResult = classifResults.find((r) => r.algorithm === 'random_forest');
            const dtResult = classifResults.find((r) => r.algorithm === 'decision_tree');
            
            const comparisonMetrics = [
              { name: 'Accuracy', rf: rfResult?.metrics.accuracy, dt: dtResult?.metrics.accuracy },
              { name: 'Precision', rf: rfResult?.metrics.precision, dt: dtResult?.metrics.precision },
              { name: 'Recall', rf: rfResult?.metrics.recall, dt: dtResult?.metrics.recall },
              { name: 'F1 Score', rf: rfResult?.metrics.f1_score, dt: dtResult?.metrics.f1_score },
              { name: 'ROC AUC', rf: rfResult?.metrics.roc_auc, dt: dtResult?.metrics.roc_auc },
            ];
            
            panels.push(
              <TabPanel key="rf-dt-comparison" value={tab} index={tabIdx++}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" sx={{ mb: 3 }}>Comparaison Random Forest vs Decision Tree</Typography>
                    
                    {/* Metrics Comparison Chart */}
                    <Box sx={{ mb: 4 }}>
                      <Typography variant="body2" sx={{ fontWeight: 600, mb: 2 }}>Comparaison des Métriques</Typography>
                      <ResponsiveContainer width="100%" height={350}>
                        <BarChart data={comparisonMetrics} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                          <XAxis dataKey="name" angle={-20} textAnchor="end" tick={{ fontSize: 12 }} />
                          <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                          <Tooltip formatter={(value) => value?.toFixed(4)} />
                          <Legend />
                          <Bar dataKey="rf" fill="#2563eb" name="Random Forest" radius={[6, 6, 0, 0]} />
                          <Bar dataKey="dt" fill="#f97316" name="Decision Tree" radius={[6, 6, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>

                    {/* Detailed Metrics Table */}
                    <Box sx={{ mb: 4 }}>
                      <Typography variant="body2" sx={{ fontWeight: 600, mb: 2 }}>Métriques Détaillées</Typography>
                      <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Métrique</TableCell>
                              <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Random Forest</TableCell>
                              <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Decision Tree</TableCell>
                              <TableCell align="center" sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>Différence</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {comparisonMetrics.map((row) => {
                              const diff = (row.rf || 0) - (row.dt || 0);
                              const color = diff > 0 ? '#059669' : diff < 0 ? '#dc2626' : '#666';
                              return (
                                <TableRow key={row.name} hover>
                                  <TableCell sx={{ fontWeight: 600 }}>{row.name}</TableCell>
                                  <TableCell align="center">{row.rf?.toFixed(4)}</TableCell>
                                  <TableCell align="center">{row.dt?.toFixed(4)}</TableCell>
                                  <TableCell align="center" sx={{ color, fontWeight: 600 }}>
                                    {diff > 0 ? '+' : ''}{diff.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Box>

                    {/* Additional Characteristics */}
                    <Box sx={{ mb: 4 }}>
                      <Typography variant="body2" sx={{ fontWeight: 600, mb: 2 }}>Caractéristiques</Typography>
                      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2, color: '#2563eb' }}>
                              Random Forest
                            </Typography>
                            <Box sx={{ fontSize: 14, lineHeight: 1.8 }}>
                              <Typography variant="body2">✓ Meilleure généralisation</Typography>
                              <Typography variant="body2">✓ Plus robuste au surapprentissage</Typography>
                              <Typography variant="body2">✓ Gère bien les données déséquilibrées</Typography>
                              <Typography variant="body2">✗ Moins interprétable</Typography>
                              <Typography variant="body2">✗ Temps d'entraînement plus long</Typography>
                              <Typography variant="body2" sx={{ mt: 1 }}>Temps: {rfResult?.duration}s</Typography>
                            </Box>
                          </CardContent>
                        </Card>

                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2, color: '#f97316' }}>
                              Decision Tree
                            </Typography>
                            <Box sx={{ fontSize: 14, lineHeight: 1.8 }}>
                              <Typography variant="body2">✓ Très interprétable</Typography>
                              <Typography variant="body2">✓ Entraînement rapide</Typography>
                              <Typography variant="body2">✓ Pas de normalisation requise</Typography>
                              <Typography variant="body2">✗ Prone au surapprentissage</Typography>
                              <Typography variant="body2">✗ Moins stable</Typography>
                              <Typography variant="body2" sx={{ mt: 1 }}>Temps: {dtResult?.duration}s</Typography>
                            </Box>
                          </CardContent>
                        </Card>
                      </Box>
                    </Box>

                    {/* Recommendation */}
                    <Card sx={{ bgcolor: rfResult?.metrics.f1_score > dtResult?.metrics.f1_score ? '#ecfdf5' : '#fef2f2', border: '2px solid #059669' }}>
                      <CardContent>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                          💡 Recommandation
                        </Typography>
                        {rfResult?.metrics.f1_score > dtResult?.metrics.f1_score ? (
                          <Typography variant="body2">
                            <strong>Random Forest</strong> est recommandé pour ce dataset. Il offre une meilleure performance globale ({(rfResult?.metrics.f1_score * 100)?.toFixed(1)}% F1 vs {(dtResult?.metrics.f1_score * 100)?.toFixed(1)}% pour Decision Tree) et une meilleure généralisation.
                          </Typography>
                        ) : (
                          <Typography variant="body2">
                            <strong>Decision Tree</strong> est recommandé pour ce dataset. Il offre une meilleure performance ({(dtResult?.metrics.f1_score * 100)?.toFixed(1)}% F1) avec un entraînement beaucoup plus rapide et une interprétabilité supérieure.
                          </Typography>
                        )}
                      </CardContent>
                    </Card>
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
