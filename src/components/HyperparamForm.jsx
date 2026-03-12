import React from 'react';
import {
  Card, CardContent, Typography, Box, TextField, Select, MenuItem,
  FormControl, InputLabel, Slider, Switch, FormControlLabel,
} from '@mui/material';
import HelpTooltip from './HelpTooltip';

/**
 * Renders a form for editing hyperparameters of a given algorithm.
 */
export default function HyperparamForm({ params, values, onChange }) {
  const handleChange = (key, val) => onChange({ ...values, [key]: val });

  return (
    <Box className="grid gap-4 grid-cols-1 sm:grid-cols-2">
      {params.map((p) => {
        const val = values[p.key] ?? p.default;

        if (p.type === 'select') {
          return (
            <FormControl key={p.key} size="small" fullWidth>
              <InputLabel>{p.label}</InputLabel>
              <Select value={val} label={p.label} onChange={(e) => handleChange(p.key, e.target.value)}>
                {p.options.map((opt) => (
                  <MenuItem key={opt} value={opt}>{opt}</MenuItem>
                ))}
              </Select>
            </FormControl>
          );
        }

        if (p.type === 'number') {
          return (
            <TextField
              key={p.key}
              label={p.label}
              type="number"
              size="small"
              fullWidth
              value={val}
              inputProps={{ min: p.min, max: p.max, step: p.step }}
              onChange={(e) => handleChange(p.key, parseFloat(e.target.value))}
            />
          );
        }

        // text
        return (
          <TextField
            key={p.key}
            label={p.label}
            size="small"
            fullWidth
            value={val}
            onChange={(e) => handleChange(p.key, e.target.value)}
          />
        );
      })}
    </Box>
  );
}
