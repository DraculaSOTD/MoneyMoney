import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { logout } from '../../features/auth/authSlice';
import { adminAuthService } from '../../services/adminAuthApi';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Avatar,
  Menu,
  MenuItem,
  Chip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  ChevronLeft as ChevronLeftIcon,
  Dashboard as DashboardIcon,
  CloudDownload as CloudDownloadIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  AccountCircle,
  Logout as LogoutIcon,
  VerifiedUser,
  Email,
} from '@mui/icons-material';

const drawerWidth = 240;

interface NavItem {
  title: string;
  path: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  { title: 'Dashboard', path: '/dashboard', icon: <DashboardIcon /> },
  { title: 'Data Collection', path: '/data-collection', icon: <CloudDownloadIcon /> },
  { title: 'Data Management', path: '/data', icon: <StorageIcon /> },
  { title: 'ML Models', path: '/models', icon: <MemoryIcon /> },
];

export const MainLayout: React.FC = () => {
  const [open, setOpen] = useState(true);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch = useAppDispatch();
  const { status } = useAppSelector((state) => state.system);
  const { user } = useAppSelector((state) => state.auth);

  const handleDrawerToggle = () => {
    setOpen(!open);
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    handleMenuClose();

    try {
      // Call backend logout API
      await adminAuthService.logout();
    } catch (error) {
      console.error('Logout API call failed:', error);
      // Continue with local logout anyway
    }

    // Clear local state and redirect
    dispatch(logout());
    navigate('/login');
  };

  // Get user initials for avatar
  const getInitials = () => {
    if (!user?.username) return 'A';
    return user.username.substring(0, 2).toUpperCase();
  };

  const getStatusColor = () => {
    switch (status) {
      case 'RUNNING':
        return '#00E676';
      case 'STOPPED':
        return '#FFC107';
      case 'ERROR':
        return '#FF5252';
      default:
        return '#757575';
    }
  };

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: `calc(100% - ${open ? drawerWidth : 0}px)`,
          ml: `${open ? drawerWidth : 0}px`,
          transition: 'width 0.2s, margin 0.2s',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            onClick={handleDrawerToggle}
            edge="start"
            sx={{ mr: 2 }}
          >
            {open ? <ChevronLeftIcon /> : <MenuIcon />}
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
            Trading Dashboard - Admin Panel
            <Chip
              label="ADMIN"
              color="primary"
              size="small"
              sx={{ height: 20, fontSize: '0.7rem', fontWeight: 700 }}
            />
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box
                sx={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  backgroundColor: getStatusColor(),
                }}
              />
              <Typography variant="body2">System: {status}</Typography>
            </Box>
            {user?.is_superuser && (
              <Chip
                label="Superuser"
                size="small"
                color="primary"
                icon={<VerifiedUser />}
                sx={{ height: 24 }}
              />
            )}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2">{user?.username || 'Admin'}</Typography>
              <IconButton onClick={handleMenuOpen} color="inherit" size="small">
                <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                  {getInitials()}
                </Avatar>
              </IconButton>
            </Box>
          </Box>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            PaperProps={{
              sx: { minWidth: 250 },
            }}
          >
            <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: 'divider' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                {user?.username || 'Admin'}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                <Email fontSize="small" sx={{ fontSize: 14, color: 'text.secondary' }} />
                <Typography variant="caption" color="text.secondary">
                  {user?.email || 'admin@local'}
                </Typography>
              </Box>
              {user?.is_superuser && (
                <Chip
                  label="Superuser"
                  size="small"
                  color="primary"
                  icon={<VerifiedUser />}
                  sx={{ mt: 1, height: 20, fontSize: '0.7rem' }}
                />
              )}
            </Box>
            <MenuItem onClick={handleLogout} sx={{ mt: 1 }}>
              <ListItemIcon>
                <LogoutIcon fontSize="small" />
              </ListItemIcon>
              Logout
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>
      <Drawer
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
          },
        }}
        variant="persistent"
        anchor="left"
        open={open}
      >
        <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <Avatar sx={{ bgcolor: 'primary.main' }}>ML</Avatar>
          <Typography variant="h6">ML Trading</Typography>
        </Box>
        <Divider />
        <List>
          {navItems.map((item) => (
            <ListItem key={item.path} disablePadding>
              <ListItemButton
                selected={location.pathname === item.path}
                onClick={() => navigate(item.path)}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.title} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Drawer>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          bgcolor: 'background.default',
          p: 3,
          width: `calc(100% - ${open ? drawerWidth : 0}px)`,
          transition: 'width 0.2s',
          mt: 8,
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
};