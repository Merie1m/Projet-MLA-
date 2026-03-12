import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

export default function StatCard({ icon, title, value, subtitle, color = '#2563eb' }) {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent className="flex items-start gap-4">
        <Box
          className="flex items-center justify-center rounded-xl"
          sx={{ bgcolor: `${color}15`, color, width: 48, height: 48, minWidth: 48 }}
        >
          {icon}
        </Box>
        <Box>
          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
            {title}
          </Typography>
          <Typography variant="h5" sx={{ fontWeight: 700, mt: 0.5 }}>
            {value}
          </Typography>
          {subtitle && (
            <Typography variant="caption" color="text.secondary">
              {subtitle}
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}
