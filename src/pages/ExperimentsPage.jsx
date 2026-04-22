import React, { useState, useEffect } from 'react';
import {
  Card, CardContent, CardHeader, Typography, Box, CircularProgress,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Paper, Chip, Grid, LinearProgress,
} from '@mui/material';
import {
  TrendingUp, CheckCircle, Timer, Storage,
} from '@mui/icons-material';
import toast from 'react-hot-toast';

const METRIC_COLORS = {
  'mean_bias': '#FF6B6B',
  'std_bias': '#FFA07A',
  'mean_variance': '#4ECDC4',
  'std_variance': '#45B7D1',
  'mean_test_score': '#95E1D3',
  'best_test_score': '#38ADA9',
  'accuracy_mean': '#06D6A0',
  'accuracy_std': '#118B7F',
  'precision_mean': '#00B4D8',
  'recall_mean': '#0077B6',
  'f1_mean': '#03045E',
};

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedExp, setSelectedExp] = useState(null);

  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/mlflow/experiments');
      if (!response.ok) throw new Error('Failed to fetch experiments');
      const data = await response.json();
      setExperiments(data.experiments || []);
      if (data.experiments?.length > 0) {
        setSelectedExp(data.experiments[0]);
      }
    } catch (error) {
      toast.error(`Erreur: ${error.message}`);
      console.error('Error fetching experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="600px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 'bold' }}>
        📊 Expérimentations MLflow
      </Typography>

      {experiments.length === 0 ? (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 5 }}>
            <Storage sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              Aucune expérimentation enregistrée
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Lancez des analyses (Bias/Variance, Stabilité) pour voir les résultats ici
            </Typography>
          </CardContent>
        </Card>
      ) : (
        <Grid container spacing={3}>
          {/* Liste des expériences */}
          <Grid item xs={12} md={4}>
            <Card sx={{ maxHeight: 'calc(100vh - 300px)', overflow: 'auto' }}>
              <CardHeader
                title="Expériences"
                titleTypographyProps={{ variant: 'h6' }}
              />
              <CardContent sx={{ p: 0 }}>
                {experiments.map((exp) => (
                  <Box
                    key={exp.experiment_id}
                    onClick={() => setSelectedExp(exp)}
                    sx={{
                      p: 2,
                      borderLeft: selectedExp?.experiment_id === exp.experiment_id ? '4px solid #1976d2' : '4px solid transparent',
                      bgcolor: selectedExp?.experiment_id === exp.experiment_id ? '#f5f5f5' : 'transparent',
                      cursor: 'pointer',
                      '&:hover': { bgcolor: '#fafafa' },
                    }}
                  >
                    <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                      {exp.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {exp.runs?.length || 0} runs
                    </Typography>
                  </Box>
                ))}
              </CardContent>
            </Card>
          </Grid>

          {/* Détails de l'expérience sélectionnée */}
          <Grid item xs={12} md={8}>
            {selectedExp && (
              <>
                <Card sx={{ mb: 3 }}>
                  <CardHeader
                    title={selectedExp.name}
                    titleTypographyProps={{ variant: 'h6' }}
                    subheader={`${selectedExp.runs?.length || 0} runs`}
                  />
                </Card>

                {selectedExp.runs && selectedExp.runs.length > 0 ? (
                  <TableContainer component={Paper}>
                    <Table>
                      <TableHead sx={{ bgcolor: '#f5f5f5' }}>
                        <TableRow>
                          <TableCell sx={{ fontWeight: 'bold' }}>Run Name</TableCell>
                          <TableCell sx={{ fontWeight: 'bold' }}>Status</TableCell>
                          <TableCell sx={{ fontWeight: 'bold' }}>Métriques Clés</TableCell>
                          <TableCell sx={{ fontWeight: 'bold' }}>Date</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {selectedExp.runs.map((run) => (
                          <TableRow key={run.run_id} hover>
                            <TableCell>
                              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                                {run.tags?.['mlflow.runName'] || run.run_id?.slice(0, 8)}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip
                                icon={<CheckCircle />}
                                label={run.status || 'ACTIVE'}
                                size="small"
                                color={run.status === 'FINISHED' ? 'success' : 'default'}
                              />
                            </TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                {Object.entries(run.metrics || {}).slice(0, 3).map(([key, val]) => (
                                  <Chip
                                    key={key}
                                    label={`${key}: ${typeof val === 'number' ? val.toFixed(4) : val}`}
                                    size="small"
                                    variant="outlined"
                                    sx={{
                                      fontSize: '0.7rem',
                                      borderColor: METRIC_COLORS[key] || '#ccc',
                                      color: METRIC_COLORS[key] || '#666',
                                    }}
                                  />
                                ))}
                              </Box>
                            </TableCell>
                            <TableCell sx={{ fontSize: '0.85rem' }}>
                              {run.start_time
                                ? new Date(run.start_time / 1000).toLocaleString('fr-FR')
                                : '-'
                              }
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Card>
                    <CardContent sx={{ textAlign: 'center', py: 4 }}>
                      <Typography color="text.secondary">
                        Aucun run dans cette expérience
                      </Typography>
                    </CardContent>
                  </Card>
                )}
              </>
            )}
          </Grid>
        </Grid>
      )}
    </Box>
  );
}
