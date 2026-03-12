import React, { useState, useCallback, useMemo } from 'react';
import {
  Card, CardContent, Typography, Box, Button, Chip, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow, Paper, Checkbox,
  FormControlLabel, IconButton, Alert, LinearProgress, Tooltip, Dialog,
  DialogTitle, DialogContent, DialogActions, MenuItem, Select, FormControl, InputLabel,
  ToggleButtonGroup, ToggleButton, CircularProgress,
} from '@mui/material';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import {
  CloudUpload, DeleteSweep, CleaningServices, Visibility, Download,
  CheckCircle, Warning, FilterList, TableChart, BubbleChart,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import toast from 'react-hot-toast';
import useStore from '../store/useStore';
import HelpTooltip from '../components/HelpTooltip';
import { CLEANING_OPERATIONS, DIM_REDUCTION_METHODS } from '../constants';
import { uploadDataset, computeDimensionReduction } from '../services/api';

export default function DatasetPage() {
  const { dataset, setDataset, setDatasetLoading, datasetLoading,
    selectedColumns, setSelectedColumns, selectedClasses, setSelectedClasses,
    targetColumn: storeTargetColumn, setTargetColumn: setStoreTargetColumn } = useStore();
  const [preview, setPreview] = useState(null);
  const [cleanDialogOpen, setCleanDialogOpen] = useState(false);
  const [targetColumn, setTargetColumnLocal] = useState(storeTargetColumn || '');
  const [dimMethod, setDimMethod] = useState('pca');
  const [dimData, setDimData] = useState(null);
  const [dimLoading, setDimLoading] = useState(false);

  const setTargetColumn = (col) => {
    setTargetColumnLocal(col);
    setStoreTargetColumn(col);
  };

  // ── File drop handler ──────────────────────────────────────────────────────
  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;
    setDatasetLoading(true);

    // 1) Parse locally for instant preview
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: (results) => {
        const columns = results.meta.fields || [];
        const rows = results.data;
        const stats = columns.map((col) => {
          const vals = rows.map((r) => r[col]).filter((v) => v !== null && v !== undefined && v !== '');
          const missing = rows.length - vals.length;
          const numeric = vals.every((v) => typeof v === 'number');
          return { name: col, type: numeric ? 'number' : 'string', missing, unique: new Set(vals).size, total: rows.length };
        });

        const localDs = {
          id: Date.now().toString(),
          name: file.name,
          size: file.size,
          columns: stats,
          rows,
          preview: rows.slice(0, 100),
          totalRows: rows.length,
          totalCols: columns.length,
        };
        setDataset(localDs);
        setSelectedColumns(columns);
        setPreview(localDs.preview);
        setDatasetLoading(false);
        toast.success(`Dataset "${file.name}" chargé : ${rows.length} lignes × ${columns.length} colonnes`);

        // 2) Also upload to backend so it can be used for training
        uploadDataset(file)
          .then((resp) => {
            // Update the dataset id with the backend id
            setDataset({ ...localDs, id: resp.id });
            toast.success('Dataset synchronisé avec le backend');
          })
          .catch(() => {
            // Backend might not be running — that's ok, local data still works
            console.warn('Backend upload failed — using local data only');
          });
      },
      error: (err) => {
        setDatasetLoading(false);
        toast.error('Erreur de lecture : ' + err.message);
      },
    });
  }, [setDataset, setDatasetLoading, setSelectedColumns]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'text/csv': ['.csv'], 'application/vnd.ms-excel': ['.xls', '.xlsx'] }, maxFiles: 1,
  });

  // ── Column toggle ──────────────────────────────────────────────────────────
  const toggleColumn = (col) => {
    setSelectedColumns(
      selectedColumns.includes(col)
        ? selectedColumns.filter((c) => c !== col)
        : [...selectedColumns, col]
    );
  };

  // ── Quick clean ────────────────────────────────────────────────────────────
  const handleClean = (opId) => {
    if (!dataset) return;
    let rows = [...dataset.rows];
    switch (opId) {
      case 'drop_na':
        rows = rows.filter((r) => Object.values(r).every((v) => v !== null && v !== undefined && v !== ''));
        break;
      case 'fill_mean': {
        const numCols = dataset.columns.filter((c) => c.type === 'number').map((c) => c.name);
        const means = {};
        numCols.forEach((col) => {
          const vals = rows.map((r) => r[col]).filter((v) => typeof v === 'number');
          means[col] = vals.reduce((a, b) => a + b, 0) / vals.length;
        });
        rows = rows.map((r) => {
          const nr = { ...r };
          numCols.forEach((col) => { if (nr[col] === null || nr[col] === undefined || nr[col] === '') nr[col] = Math.round(means[col] * 100) / 100; });
          return nr;
        });
        break;
      }
      case 'fill_median': {
        const numCols2 = dataset.columns.filter((c) => c.type === 'number').map((c) => c.name);
        const medians = {};
        numCols2.forEach((col) => {
          const vals = rows.map((r) => r[col]).filter((v) => typeof v === 'number').sort((a, b) => a - b);
          const mid = Math.floor(vals.length / 2);
          medians[col] = vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
        });
        rows = rows.map((r) => {
          const nr = { ...r };
          numCols2.forEach((col) => { if (nr[col] === null || nr[col] === undefined || nr[col] === '') nr[col] = medians[col]; });
          return nr;
        });
        break;
      }
      case 'drop_duplicates': {
        const seen = new Set();
        rows = rows.filter((r) => {
          const key = JSON.stringify(r);
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
        break;
      }
      default:
        toast('Opération à implémenter côté backend');
        return;
    }
    const newDs = { ...dataset, rows, preview: rows.slice(0, 100), totalRows: rows.length };
    const removedCount = dataset.rows.length - rows.length;
    setDataset(newDs);
    setPreview(newDs.preview);
    toast.success(`Nettoyage terminé — ${removedCount >= 0 ? removedCount + ' lignes supprimées' : 'valeurs remplies'}`);
    setCleanDialogOpen(false);
  };

  // ── Unique classes for target column ───────────────────────────────────────
  const uniqueClasses = useMemo(() => {
    if (!dataset || !targetColumn) return [];
    return [...new Set(dataset.rows.map((r) => r[targetColumn]))].filter(Boolean).sort();
  }, [dataset, targetColumn]);

  // ── Export as CSV ──────────────────────────────────────────────────────────
  const exportCSV = () => {
    if (!dataset) return;
    const csv = Papa.unparse(dataset.rows.map((r) => {
      const filtered = {};
      selectedColumns.forEach((c) => { filtered[c] = r[c]; });
      return filtered;
    }));
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `${dataset.name}_filtered.csv`; a.click();
    URL.revokeObjectURL(url);
    toast.success('CSV exporté');
  };

  return (
    <Box className="space-y-6">
      {/* ── Upload Zone ───────────────────────────────────────────── */}
      <Card>
        <CardContent>
          <Box className="flex items-center gap-2 mb-4">
            <Typography variant="h6">Charger un Dataset</Typography>
            <HelpTooltip text="Glissez-déposez un fichier CSV ou cliquez pour sélectionner. Les fichiers sont parsés localement." />
          </Box>
          <Box
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all
              ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'}`}
          >
            <input {...getInputProps()} />
            <CloudUpload sx={{ fontSize: 48, color: isDragActive ? '#2563eb' : '#94a3b8', mb: 1 }} />
            <Typography variant="body1" color="text.secondary">
              {isDragActive ? 'Déposez le fichier ici…' : 'Glissez-déposez un CSV ou cliquez pour parcourir'}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Formats : .csv, .xls, .xlsx — Max 50 Mo
            </Typography>
          </Box>
          {datasetLoading && <LinearProgress sx={{ mt: 2, borderRadius: 2 }} />}
        </CardContent>
      </Card>

      {dataset && (
        <>
          {/* ── Dataset Info ─────────────────────────────────────── */}
          <Card>
            <CardContent>
              <Box className="flex flex-wrap items-center justify-between gap-4 mb-4">
                <Box>
                  <Typography variant="h6">{dataset.name}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {dataset.totalRows.toLocaleString()} lignes × {dataset.totalCols} colonnes
                    — {(dataset.size / 1024).toFixed(1)} Ko
                  </Typography>
                </Box>
                <Box className="flex gap-2">
                  <Button variant="outlined" startIcon={<CleaningServices />} onClick={() => setCleanDialogOpen(true)}>
                    Nettoyer
                  </Button>
                  <Button variant="outlined" startIcon={<Download />} onClick={exportCSV}>
                    Exporter CSV
                  </Button>
                </Box>
              </Box>

              {/* column stats */}
              <Typography variant="subtitle2" sx={{ mb: 1 }}>Colonnes</Typography>
              <Box className="flex flex-wrap gap-2 mb-4">
                {dataset.columns.map((col) => (
                  <Tooltip key={col.name} title={`Type: ${col.type} | Manquantes: ${col.missing} | Uniques: ${col.unique}`}>
                    <Chip
                      label={col.name}
                      variant={selectedColumns.includes(col.name) ? 'filled' : 'outlined'}
                      color={selectedColumns.includes(col.name) ? 'primary' : 'default'}
                      onClick={() => toggleColumn(col.name)}
                      icon={col.missing > 0 ? <Warning fontSize="small" color="warning" /> : <CheckCircle fontSize="small" color="success" />}
                    />
                  </Tooltip>
                ))}
              </Box>

              {/* target column / class filter */}
              <Box className="flex flex-wrap items-end gap-4 mb-4">
                <FormControl size="small" sx={{ minWidth: 200 }}>
                  <InputLabel>Colonne cible (label)</InputLabel>
                  <Select value={targetColumn} label="Colonne cible (label)" onChange={(e) => setTargetColumn(e.target.value)}>
                    {dataset.columns.map((c) => <MenuItem key={c.name} value={c.name}>{c.name}</MenuItem>)}
                  </Select>
                </FormControl>
                {uniqueClasses.length > 0 && (
                  <Box className="flex flex-wrap gap-1">
                    {uniqueClasses.map((cls) => (
                      <Chip
                        key={cls}
                        label={String(cls)}
                        size="small"
                        variant={selectedClasses.length === 0 || selectedClasses.includes(cls) ? 'filled' : 'outlined'}
                        color="secondary"
                        onClick={() => {
                          if (selectedClasses.includes(cls)) setSelectedClasses(selectedClasses.filter((c) => c !== cls));
                          else setSelectedClasses([...selectedClasses, cls]);
                        }}
                      />
                    ))}
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>

          {/* ── Data Preview Table ───────────────────────────────── */}
          <Card>
            <CardContent>
              <Box className="flex items-center gap-2 mb-3">
                <TableChart color="primary" />
                <Typography variant="h6">Prévisualisation</Typography>
                <Typography variant="caption" color="text.secondary">(100 premières lignes)</Typography>
              </Box>
              <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 480, borderRadius: 2 }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 700, bgcolor: '#f1f5f9' }}>#</TableCell>
                      {selectedColumns.map((col) => (
                        <TableCell key={col} sx={{ fontWeight: 700, bgcolor: '#f1f5f9', whiteSpace: 'nowrap' }}>
                          {col}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {(preview || []).slice(0, 100).map((row, i) => (
                      <TableRow key={i} hover>
                        <TableCell sx={{ color: '#94a3b8' }}>{i + 1}</TableCell>
                        {selectedColumns.map((col) => (
                          <TableCell key={col} sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {row[col] !== null && row[col] !== undefined ? String(row[col]) : <em style={{ color: '#ef4444' }}>NaN</em>}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>

          {/* ── Dimension Reduction (PCA / t-SNE) ─────────────── */}
          {targetColumn && (
            <Card>
              <CardContent>
                <Box className="flex items-center gap-2 mb-3">
                  <BubbleChart color="secondary" />
                  <Typography variant="h6">Réduction de dimension</Typography>
                </Box>
                <Box className="flex flex-wrap items-center gap-3 mb-4">
                  <ToggleButtonGroup
                    value={dimMethod}
                    exclusive
                    onChange={(_, v) => v && setDimMethod(v)}
                    size="small"
                  >
                    {DIM_REDUCTION_METHODS.map((m) => (
                      <ToggleButton key={m.id} value={m.id}>
                        <Tooltip title={m.description}><span>{m.name}</span></Tooltip>
                      </ToggleButton>
                    ))}
                  </ToggleButtonGroup>
                  <Button
                    variant="contained"
                    size="small"
                    disabled={dimLoading}
                    onClick={async () => {
                      setDimLoading(true);
                      try {
                        const result = await computeDimensionReduction({
                          dataset_id: dataset.id,
                          target_column: targetColumn,
                          method: dimMethod,
                          selected_columns: selectedColumns.filter((c) => c !== targetColumn),
                        });
                        setDimData(result);
                        toast.success(`${dimMethod.toUpperCase()} calculé — ${result.n_samples} points`);
                      } catch (e) {
                        toast.error('Erreur: ' + e.message);
                      }
                      setDimLoading(false);
                    }}
                  >
                    {dimLoading ? <CircularProgress size={20} /> : 'Calculer'}
                  </Button>
                </Box>

                {dimData && dimData.points && (
                  <>
                    {dimData.explained_variance?.length > 0 && (
                      <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                        Variance expliquée : {dimData.explained_variance.map((v) => `${(v * 100).toFixed(1)}%`).join(' + ')}
                      </Typography>
                    )}
                    <ResponsiveContainer width="100%" height={400}>
                      <ScatterChart margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="x" type="number" name="Comp 1" tick={{ fontSize: 11 }} />
                        <YAxis dataKey="y" type="number" name="Comp 2" tick={{ fontSize: 11 }} />
                        <ReTooltip cursor={{ strokeDasharray: '3 3' }} content={({ payload }) => {
                          if (!payload || !payload.length) return null;
                          const d = payload[0].payload;
                          return (
                            <Paper sx={{ p: 1 }}>
                              <Typography variant="caption">Label: {d.label}</Typography><br />
                              <Typography variant="caption">x: {d.x}, y: {d.y}</Typography>
                            </Paper>
                          );
                        }} />
                        {(() => {
                          const labels = [...new Set(dimData.points.map((p) => p.label))];
                          const colors = ['#2563eb', '#dc2626', '#16a34a', '#ea580c', '#7c3aed', '#0891b2', '#ca8a04', '#be185d', '#4f46e5', '#059669'];
                          return labels.map((label, i) => (
                            <Scatter
                              key={label}
                              name={String(label)}
                              data={dimData.points.filter((p) => p.label === label)}
                              fill={colors[i % colors.length]}
                              fillOpacity={0.7}
                            />
                          ));
                        })()}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </>
      )}

      {/* ── Cleaning Dialog ───────────────────────────────────────── */}
      <Dialog open={cleanDialogOpen} onClose={() => setCleanDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Nettoyage rapide des données</DialogTitle>
        <DialogContent>
          <Box className="space-y-2 mt-2">
            {CLEANING_OPERATIONS.map((op) => (
              <Button
                key={op.id}
                variant="outlined"
                fullWidth
                sx={{ justifyContent: 'flex-start', textAlign: 'left', py: 1.5 }}
                onClick={() => handleClean(op.id)}
              >
                {op.label}
              </Button>
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCleanDialogOpen(false)}>Fermer</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
