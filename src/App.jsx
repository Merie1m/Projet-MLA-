import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import DashboardPage from './pages/DashboardPage';
import DatasetPage from './pages/DatasetPage';
import ModelsPage from './pages/ModelsPage';
import TrainingPage from './pages/TrainingPage';
import ResultsPage from './pages/ResultsPage';
import MLOpsPage from './pages/MLOpsPage';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<DashboardPage />} />
        <Route path="dataset" element={<DatasetPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="training" element={<TrainingPage />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="mlops" element={<MLOpsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
