import React from 'react';
import { Typography, Box, Button, Card, CardContent, Grid, Chip } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import {
  Storage, Psychology, ModelTraining, Assessment, History, ArrowForward,
  TrendingUp, Dataset, Speed,
} from '@mui/icons-material';
import useStore from '../store/useStore';
import StatCard from '../components/StatCard';
import { ML_ALGORITHMS, REGRESSION_ALGO_IDS } from '../constants';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import { METRIC_COLORS } from '../constants';

export default function DashboardPage() {
  const navigate = useNavigate();
  const { dataset, selectedModels, trainingResults, notifications } = useStore();

  const algoMap = {};
  ML_ALGORITHMS.forEach((a) => { algoMap[a.id] = a; });

  const quickActions = [
    { label: 'Charger un dataset', icon: <Storage />, path: '/dataset', color: '#2563eb' },
    { label: 'Choisir des modèles', icon: <Psychology />, path: '/models', color: '#7c3aed' },
    { label: 'Lancer l\'entraînement', icon: <ModelTraining />, path: '/training', color: '#16a34a' },
    { label: 'Voir les résultats', icon: <Assessment />, path: '/results', color: '#ea580c' },
    { label: 'Historique MLOps', icon: <History />, path: '/mlops', color: '#0891b2' },
  ];

  const bestModel = trainingResults.length > 0
    ? trainingResults.reduce((best, r) => {
        const isReg = REGRESSION_ALGO_IDS.includes(r.algorithm);
        const bestIsReg = REGRESSION_ALGO_IDS.includes(best.algorithm);
        const rScore = isReg ? (r.metrics.r2 || 0) : (r.metrics.accuracy || 0);
        const bestScore = bestIsReg ? (best.metrics.r2 || 0) : (best.metrics.accuracy || 0);
        return rScore > bestScore ? r : best;
      })
    : null;

  const bestModelIsReg = bestModel && REGRESSION_ALGO_IDS.includes(bestModel.algorithm);

  const recentChartData = trainingResults.slice(-6).map((r) => ({
    name: (algoMap[r.algorithm]?.name || r.algorithm).split(' ')[0],
    score: REGRESSION_ALGO_IDS.includes(r.algorithm)
      ? Math.round((r.metrics.r2 || 0) * 100) / 100
      : Math.round((r.metrics.accuracy || 0) * 100) / 100,
  }));

  return (
    <Box className="space-y-6">
      {/* ── Welcome ───────────────────────────────────────────── */}
      <Box>
        <Typography variant="h4" sx={{ mb: 0.5 }}>Bienvenue sur ML Platform</Typography>
        <Typography variant="body1" color="text.secondary">
          Plateforme d'entraînement, d'évaluation et de comparaison de modèles de Machine Learning.
        </Typography>
      </Box>

      {/* ── Stats ─────────────────────────────────────────────── */}
      <Box className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<Dataset />}
          title="Dataset"
          value={dataset ? dataset.name : '—'}
          subtitle={dataset ? `${dataset.totalRows?.toLocaleString()} lignes` : 'Aucun chargé'}
          color="#2563eb"
        />
        <StatCard
          icon={<Psychology />}
          title="Modèles sélectionnés"
          value={selectedModels.length}
          subtitle={selectedModels.map((m) => algoMap[m.algorithm]?.name?.split(' ')[0]).join(', ') || 'Aucun'}
          color="#7c3aed"
        />
        <StatCard
          icon={<Speed />}
          title="Expérimentations"
          value={trainingResults.length}
          subtitle="Résultats disponibles"
          color="#16a34a"
        />
        <StatCard
          icon={<TrendingUp />}
          title="Meilleur modèle"
          value={bestModel ? (bestModelIsReg ? `R² ${bestModel.metrics.r2?.toFixed(3)}` : `${(bestModel.metrics.accuracy * 100).toFixed(1)}%`) : '—'}
          subtitle={bestModel ? (algoMap[bestModel.algorithm]?.name || bestModel.algorithm) : 'Pas encore entraîné'}
          color="#ea580c"
        />
      </Box>

      {/* ── Quick actions ─────────────────────────────────────── */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 3 }}>Actions rapides</Typography>
          <Box className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {quickActions.map((action) => (
              <Button
                key={action.path}
                variant="outlined"
                fullWidth
                onClick={() => navigate(action.path)}
                sx={{
                  py: 2.5, flexDirection: 'column', gap: 1, borderColor: '#e2e8f0',
                  '&:hover': { borderColor: action.color, bgcolor: `${action.color}08` },
                }}
              >
                <Box sx={{ color: action.color }}>{action.icon}</Box>
                <Typography variant="caption" sx={{ fontWeight: 600, color: '#475569' }}>
                  {action.label}
                </Typography>
              </Button>
            ))}
          </Box>
        </CardContent>
      </Card>

      {/* ── Recent results chart ──────────────────────────────── */}
      {recentChartData.length > 0 && (
        <Card>
          <CardContent>
            <Box className="flex items-center justify-between mb-3">
              <Typography variant="h6">Derniers résultats</Typography>
              <Button size="small" endIcon={<ArrowForward />} onClick={() => navigate('/results')}>
                Voir tout
              </Button>
            </Box>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={recentChartData} margin={{ top: 10, right: 20, left: 20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                <ReTooltip />
                <Bar dataKey="score" radius={[6, 6, 0, 0]}>
                  {recentChartData.map((_, i) => (
                    <Cell key={i} fill={METRIC_COLORS[i % METRIC_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* ── Recent notifications ──────────────────────────────── */}
      {notifications.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>Notifications récentes</Typography>
            <Box className="space-y-2">
              {notifications.slice(0, 5).map((n) => (
                <Box key={n.id} className="flex items-center gap-3 p-2 rounded-lg bg-slate-50">
                  <Chip
                    label={n.type}
                    size="small"
                    color={n.type === 'success' ? 'success' : n.type === 'error' ? 'error' : 'primary'}
                  />
                  <Typography variant="body2">{n.message}</Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto', whiteSpace: 'nowrap' }}>
                    {new Date(n.timestamp).toLocaleTimeString('fr-FR')}
                  </Typography>
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
