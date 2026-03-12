import React, { useState } from 'react';
import {
  Card, CardContent, Typography, Box, Button, Chip, Checkbox,
  FormControlLabel, Accordion, AccordionSummary, AccordionDetails,
  Dialog, DialogTitle, DialogContent, DialogActions, TextField,
  ToggleButton, ToggleButtonGroup, Alert, Tooltip, IconButton, Divider,
} from '@mui/material';
import {
  ExpandMore, Psychology, Tune, Save, FolderOpen, Delete,
  Compare, AutoFixHigh, Info, CheckCircle,
} from '@mui/icons-material';
import toast from 'react-hot-toast';
import useStore from '../store/useStore';
import HelpTooltip from '../components/HelpTooltip';
import HyperparamForm from '../components/HyperparamForm';
import { ML_ALGORITHMS, TUNING_STRATEGIES } from '../constants';

export default function ModelsPage() {
  const {
    selectedModels, toggleModelSelection, updateModelHyperparams, clearSelectedModels,
    savedConfigs, saveConfig, deleteConfig, loadConfig, autoTuneMethod, setAutoTuneMethod,
  } = useStore();
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [configName, setConfigName] = useState('');
  const [loadDialogOpen, setLoadDialogOpen] = useState(false);

  const isSelected = (id) => selectedModels.some((m) => m.algorithm === id);
  const getModelConfig = (id) => selectedModels.find((m) => m.algorithm === id);

  const handleToggle = (algo) => {
    const defaults = {};
    algo.hyperparams.forEach((p) => { defaults[p.key] = p.default; });
    toggleModelSelection({ algorithm: algo.id, hyperparams: defaults, fromScratch: true });
  };

  const handleSaveConfig = () => {
    if (!configName.trim()) return;
    saveConfig({ name: configName, models: selectedModels, tuneMethod: autoTuneMethod });
    toast.success(`Configuration "${configName}" sauvegardée`);
    setConfigName('');
    setSaveDialogOpen(false);
  };

  return (
    <Box className="space-y-6">
      {/* ── Header ────────────────────────────────────────────── */}
      <Box className="flex flex-wrap items-center justify-between gap-4">
        <Box>
          <Typography variant="h5">Sélection des Modèles</Typography>
          <Typography variant="body2" color="text.secondary">
            Choisissez un ou plusieurs algorithmes ML pour entraînement ou comparaison.
          </Typography>
        </Box>
        <Box className="flex gap-2">
          <Button variant="outlined" startIcon={<FolderOpen />} onClick={() => setLoadDialogOpen(true)}>
            Charger config
          </Button>
          <Button variant="outlined" startIcon={<Save />} onClick={() => setSaveDialogOpen(true)} disabled={selectedModels.length === 0}>
            Sauvegarder config
          </Button>
          {selectedModels.length > 0 && (
            <Button variant="text" color="error" onClick={clearSelectedModels}>
              Tout désélectionner
            </Button>
          )}
        </Box>
      </Box>

      {selectedModels.length > 0 && (
        <Alert severity="info" icon={<Compare />}>
          <strong>{selectedModels.length} modèle(s)</strong> sélectionné(s) pour comparaison côte à côte.
          Allez dans "Entraînement" pour lancer.
        </Alert>
      )}

      {/* ── Algorithm cards ───────────────────────────────────── */}
      <Box className="grid gap-4 grid-cols-1 lg:grid-cols-2">
        {ML_ALGORITHMS.map((algo) => {
          const selected = isSelected(algo.id);
          const config = getModelConfig(algo.id);
          return (
            <Card
              key={algo.id}
              sx={{
                border: selected ? '2px solid #2563eb' : '2px solid transparent',
                transition: 'all 0.2s',
                '&:hover': { boxShadow: 4 },
              }}
            >
              <CardContent>
                {/* header */}
                <Box className="flex items-start justify-between mb-2">
                  <Box className="flex items-center gap-2">
                    <Checkbox
                      checked={selected}
                      onChange={() => handleToggle(algo)}
                      color="primary"
                    />
                    <Box>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                        {algo.name}
                      </Typography>
                      <Chip label={algo.category} size="small" variant="outlined" color="primary" />
                    </Box>
                  </Box>
                  {selected && <CheckCircle color="primary" />}
                </Box>

                {/* description */}
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2, ml: 5 }}>
                  {algo.description}
                </Typography>

                {/* hyperparameters (only when selected) */}
                {selected && (
                  <Accordion defaultExpanded sx={{ boxShadow: 'none', border: '1px solid #e2e8f0', borderRadius: '8px !important', '&:before': { display: 'none' } }}>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box className="flex items-center gap-2">
                        <Tune fontSize="small" />
                        <Typography variant="subtitle2">Hyperparamètres</Typography>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <HyperparamForm
                        params={algo.hyperparams}
                        values={config?.hyperparams || {}}
                        onChange={(vals) => updateModelHyperparams(algo.id, vals)}
                      />
                      <Box className="mt-3">
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={config?.fromScratch ?? true}
                              onChange={(e) => {
                                const sel = selectedModels.map((m) =>
                                  m.algorithm === algo.id ? { ...m, fromScratch: e.target.checked } : m
                                );
                                useStore.setState({ selectedModels: sel });
                              }}
                            />
                          }
                          label={
                            <Box className="flex items-center gap-1">
                              <Typography variant="body2">Entraîner à partir de zéro</Typography>
                              <HelpTooltip text="Décochez pour charger un modèle pré-entraîné existant" />
                            </Box>
                          }
                        />
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                )}
              </CardContent>
            </Card>
          );
        })}
      </Box>

      {/* ── Auto-tune section ─────────────────────────────────── */}
      <Card>
        <CardContent>
          <Box className="flex items-center gap-2 mb-3">
            <AutoFixHigh color="secondary" />
            <Typography variant="h6">Tuning Automatique</Typography>
            <HelpTooltip text="Lance une recherche automatique des meilleurs hyperparamètres sur le/les modèle(s) sélectionné(s)." />
          </Box>
          <Box className="flex flex-wrap gap-4">
            {TUNING_STRATEGIES.map((s) => (
              <Card
                key={s.id}
                variant="outlined"
                sx={{
                  flex: '1 1 220px',
                  cursor: 'pointer',
                  border: autoTuneMethod === s.id ? '2px solid #7c3aed' : undefined,
                  bgcolor: autoTuneMethod === s.id ? '#f5f3ff' : undefined,
                }}
                onClick={() => setAutoTuneMethod(s.id)}
              >
                <CardContent>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{s.name}</Typography>
                  <Typography variant="caption" color="text.secondary">{s.description}</Typography>
                </CardContent>
              </Card>
            ))}
          </Box>
        </CardContent>
      </Card>

      {/* ── Save Config Dialog ────────────────────────────────── */}
      <Dialog open={saveDialogOpen} onClose={() => setSaveDialogOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Sauvegarder la configuration</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus fullWidth label="Nom de la configuration" sx={{ mt: 1 }}
            value={configName} onChange={(e) => setConfigName(e.target.value)}
            helperText={`${selectedModels.length} modèle(s) sélectionné(s)`}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)}>Annuler</Button>
          <Button variant="contained" onClick={handleSaveConfig}>Sauvegarder</Button>
        </DialogActions>
      </Dialog>

      {/* ── Load Config Dialog ────────────────────────────────── */}
      <Dialog open={loadDialogOpen} onClose={() => setLoadDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Configurations sauvegardées</DialogTitle>
        <DialogContent>
          {savedConfigs.length === 0 ? (
            <Typography color="text.secondary" sx={{ py: 2 }}>Aucune configuration sauvegardée.</Typography>
          ) : (
            <Box className="space-y-2 mt-2">
              {savedConfigs.map((cfg) => (
                <Box key={cfg.id} className="flex items-center justify-between p-3 rounded-lg border hover:bg-slate-50">
                  <Box>
                    <Typography variant="subtitle2">{cfg.name}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {cfg.models?.length || 0} modèle(s) — {new Date(cfg.savedAt).toLocaleDateString('fr-FR')}
                    </Typography>
                  </Box>
                  <Box className="flex gap-1">
                    <Button size="small" onClick={() => { loadConfig(cfg); setLoadDialogOpen(false); toast.success('Configuration chargée'); }}>
                      Charger
                    </Button>
                    <IconButton size="small" color="error" onClick={() => deleteConfig(cfg.id)}>
                      <Delete fontSize="small" />
                    </IconButton>
                  </Box>
                </Box>
              ))}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setLoadDialogOpen(false)}>Fermer</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
