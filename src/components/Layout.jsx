import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar, Toolbar, Typography, Drawer, List, ListItemButton, ListItemIcon,
  ListItemText, Box, IconButton, Tooltip, Divider, useMediaQuery, Badge,
} from '@mui/material';
import {
  Dashboard, Storage, Psychology, ModelTraining, Assessment, History,
  Menu as MenuIcon, ChevronLeft, Notifications, HelpOutline, DarkMode,
} from '@mui/icons-material';
import useStore from '../store/useStore';

const DRAWER_WIDTH = 260;
const NAV = [
  { label: 'Dashboard', icon: <Dashboard />, path: '/' },
  { label: 'Données', icon: <Storage />, path: '/dataset' },
  { label: 'Modèles', icon: <Psychology />, path: '/models' },
  { label: 'Entraînement', icon: <ModelTraining />, path: '/training' },
  { label: 'Résultats', icon: <Assessment />, path: '/results' },
  { label: 'MLOps', icon: <History />, path: '/mlops' },
];

export default function Layout() {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const isMobile = useMediaQuery('(max-width:900px)');
  const [open, setOpen] = useState(!isMobile);
  const notifications = useStore((s) => s.notifications);

  return (
    <Box className="flex min-h-screen bg-slate-50">
      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      <Drawer
        variant={isMobile ? 'temporary' : 'persistent'}
        open={open}
        onClose={() => setOpen(false)}
        sx={{
          width: open ? DRAWER_WIDTH : 0,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
            background: 'linear-gradient(180deg, #1e293b 0%, #0f172a 100%)',
            color: '#e2e8f0',
            borderRight: 'none',
          },
        }}
      >
        <Box className="flex items-center justify-between px-4 py-4">
          <Typography variant="h6" className="font-bold text-white tracking-tight">
            🧠 ML Platform
          </Typography>
          <IconButton onClick={() => setOpen(false)} sx={{ color: '#94a3b8' }}>
            <ChevronLeft />
          </IconButton>
        </Box>
        <Divider sx={{ borderColor: '#334155' }} />
        <List sx={{ px: 1, mt: 1 }}>
          {NAV.map((item) => {
            const active = pathname === item.path;
            return (
              <ListItemButton
                key={item.path}
                onClick={() => { navigate(item.path); if (isMobile) setOpen(false); }}
                sx={{
                  borderRadius: 2, mb: 0.5, mx: 0.5,
                  backgroundColor: active ? 'rgba(59,130,246,0.2)' : 'transparent',
                  '&:hover': { backgroundColor: active ? 'rgba(59,130,246,0.25)' : 'rgba(255,255,255,0.05)' },
                }}
              >
                <ListItemIcon sx={{ color: active ? '#60a5fa' : '#94a3b8', minWidth: 40 }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.label}
                  primaryTypographyProps={{
                    fontWeight: active ? 600 : 400,
                    color: active ? '#f1f5f9' : '#cbd5e1',
                    fontSize: 14,
                  }}
                />
              </ListItemButton>
            );
          })}
        </List>
        <Box sx={{ flexGrow: 1 }} />
        <Box className="px-4 pb-4">
          <Typography variant="caption" sx={{ color: '#64748b' }}>
            ML Platform v1.0
          </Typography>
        </Box>
      </Drawer>

      {/* ── Main Content ────────────────────────────────────────────── */}
      <Box className="flex-1 flex flex-col" sx={{ ml: open && !isMobile ? `${DRAWER_WIDTH}px` : 0, transition: 'margin 0.3s' }}>
        <AppBar
          position="sticky"
          elevation={0}
          sx={{
            backgroundColor: '#ffffff',
            borderBottom: '1px solid #e2e8f0',
            color: '#1e293b',
          }}
        >
          <Toolbar className="flex justify-between">
            <Box className="flex items-center gap-2">
              {!open && (
                <IconButton onClick={() => setOpen(true)}>
                  <MenuIcon />
                </IconButton>
              )}
              <Typography variant="h6" sx={{ fontWeight: 600, fontSize: 18 }}>
                {NAV.find((n) => n.path === pathname)?.label || 'Dashboard'}
              </Typography>
            </Box>
            <Box className="flex items-center gap-1">
              <Tooltip title="Aide & tutoriels">
                <IconButton><HelpOutline /></IconButton>
              </Tooltip>
              <Tooltip title="Notifications">
                <IconButton>
                  <Badge badgeContent={notifications.length} color="error" max={9}>
                    <Notifications />
                  </Badge>
                </IconButton>
              </Tooltip>
              <Tooltip title="Mode sombre (bientôt)">
                <IconButton><DarkMode /></IconButton>
              </Tooltip>
            </Box>
          </Toolbar>
        </AppBar>

        <Box component="main" className="flex-1 p-4 md:p-6 lg:p-8">
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
}
