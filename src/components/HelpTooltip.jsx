import React from 'react';
import { Tooltip, IconButton } from '@mui/material';
import { HelpOutline } from '@mui/icons-material';

/**
 * Small help icon with a tooltip — used throughout the UI to guide users.
 */
export default function HelpTooltip({ text, placement = 'top' }) {
  return (
    <Tooltip title={text} placement={placement} arrow>
      <IconButton size="small" sx={{ color: '#94a3b8', ml: 0.5 }}>
        <HelpOutline fontSize="small" />
      </IconButton>
    </Tooltip>
  );
}
