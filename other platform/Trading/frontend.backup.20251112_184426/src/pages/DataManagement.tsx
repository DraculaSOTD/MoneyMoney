import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  Tab,
  Tabs,
  Card,
  CardContent,
  CardActions,
  LinearProgress,
  Tooltip,
  FormControlLabel,
  Switch,
} from '@mui/material';
import {
  Add,
  Edit,
  Delete,
  PlayArrow,
  Stop,
  Refresh,
  Settings,
  Analytics,
  TrendingUp,
  Warning,
  CheckCircle,
  Error as ErrorIcon,
  Storage,
  Speed,
  ModelTraining,
} from '@mui/icons-material';
import { ProfileSelector } from '../components';
import {
  useGetProfilesQuery,
  useCreateProfileMutation,
  useUpdateProfileMutation,
  useDeleteProfileMutation,
  useGetProfileModelsQuery,
  useGetLatestMetricsQuery,
  TradingProfile,
} from '../store/slices/profileSlice';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`data-tabpanel-${index}`}
      aria-labelledby={`data-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export const DataManagement: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingProfile, setEditingProfile] = useState<TradingProfile | null>(null);
  const [selectedProfileId, setSelectedProfileId] = useState<number | null>(null);
  
  const { data: profiles, isLoading: profilesLoading } = useGetProfilesQuery({});
  const [createProfile] = useCreateProfileMutation();
  const [updateProfile] = useUpdateProfileMutation();
  const [deleteProfile] = useDeleteProfileMutation();
  
  const { data: models } = useGetProfileModelsQuery(
    { profileId: selectedProfileId! },
    { skip: !selectedProfileId }
  );
  
  const { data: latestMetrics } = useGetLatestMetricsQuery(
    selectedProfileId!,
    { skip: !selectedProfileId }
  );

  const [formData, setFormData] = useState({
    symbol: '',
    name: '',
    profile_type: 'crypto' as const,
    exchange: 'binance',
    description: '',
    base_currency: '',
    quote_currency: '',
    data_source: 'binance',
    timeframe: '1h',
    lookback_days: 365,
    min_trade_size: 0.001,
    max_trade_size: 1,
    max_position_size: 10,
    trading_fee: 0.001,
    max_drawdown_limit: 0.2,
    position_risk_limit: 0.02,
    daily_loss_limit: 0.05,
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleOpenDialog = (profile?: TradingProfile) => {
    if (profile) {
      setEditingProfile(profile);
      setFormData({
        symbol: profile.symbol,
        name: profile.name,
        profile_type: profile.profile_type,
        exchange: profile.exchange,
        description: profile.description || '',
        base_currency: profile.base_currency,
        quote_currency: profile.quote_currency,
        data_source: profile.data_source,
        timeframe: profile.timeframe,
        lookback_days: profile.lookback_days,
        min_trade_size: profile.min_trade_size,
        max_trade_size: profile.max_trade_size,
        max_position_size: profile.max_position_size,
        trading_fee: profile.trading_fee,
        max_drawdown_limit: profile.max_drawdown_limit,
        position_risk_limit: profile.position_risk_limit,
        daily_loss_limit: profile.daily_loss_limit,
      });
    } else {
      setEditingProfile(null);
      setFormData({
        symbol: '',
        name: '',
        profile_type: 'crypto',
        exchange: 'binance',
        description: '',
        base_currency: '',
        quote_currency: '',
        data_source: 'binance',
        timeframe: '1h',
        lookback_days: 365,
        min_trade_size: 0.001,
        max_trade_size: 1,
        max_position_size: 10,
        trading_fee: 0.001,
        max_drawdown_limit: 0.2,
        position_risk_limit: 0.02,
        daily_loss_limit: 0.05,
      });
    }
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingProfile(null);
  };

  const handleSaveProfile = async () => {
    try {
      if (editingProfile) {
        await updateProfile({
          id: editingProfile.id,
          updates: formData,
        }).unwrap();
      } else {
        await createProfile(formData).unwrap();
      }
      handleCloseDialog();
    } catch (error) {
      console.error('Failed to save profile:', error);
    }
  };

  const handleDeleteProfile = async (id: number) => {
    if (window.confirm('Are you sure you want to delete this profile?')) {
      try {
        await deleteProfile(id).unwrap();
      } catch (error) {
        console.error('Failed to delete profile:', error);
      }
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'deployed':
        return <CheckCircle color="success" />;
      case 'trained':
        return <CheckCircle color="info" />;
      case 'training':
        return <LinearProgress sx={{ width: 20 }} />;
      case 'failed':
        return <ErrorIcon color="error" />;
      default:
        return <Warning color="warning" />;
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Data Management</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => handleOpenDialog()}
        >
          Create Profile
        </Button>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ width: '100%' }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={tabValue} onChange={handleTabChange}>
                <Tab label="Profiles" icon={<Storage />} iconPosition="start" />
                <Tab label="Configuration" icon={<Settings />} iconPosition="start" />
                <Tab label="Metrics" icon={<Analytics />} iconPosition="start" />
              </Tabs>
            </Box>

            <TabPanel value={tabValue} index={0}>
              {profilesLoading ? (
                <LinearProgress />
              ) : (
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Name</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Exchange</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Performance</TableCell>
                        <TableCell>Models</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {profiles?.map((profile) => (
                        <TableRow key={profile.id}>
                          <TableCell>
                            <Typography variant="subtitle2">{profile.symbol}</Typography>
                          </TableCell>
                          <TableCell>{profile.name}</TableCell>
                          <TableCell>
                            <Chip
                              label={profile.profile_type}
                              size="small"
                              color="primary"
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>{profile.exchange}</TableCell>
                          <TableCell>
                            <Box display="flex" alignItems="center" gap={1}>
                              {profile.is_active ? (
                                <Chip label="Active" color="success" size="small" />
                              ) : (
                                <Chip label="Inactive" color="default" size="small" />
                              )}
                            </Box>
                          </TableCell>
                          <TableCell>
                            <Box>
                              <Typography variant="caption" display="block">
                                Win Rate: {(profile.win_rate * 100).toFixed(1)}%
                              </Typography>
                              <Typography
                                variant="caption"
                                display="block"
                                color={profile.total_pnl >= 0 ? 'success.main' : 'error.main'}
                              >
                                P&L: ${profile.total_pnl.toFixed(2)}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell>
                            <Box display="flex" gap={0.5}>
                              <Chip
                                label={`${profile.active_models}`}
                                size="small"
                                icon={<ModelTraining />}
                                variant="outlined"
                                color="info"
                              />
                              {profile.deployed_models > 0 && (
                                <Chip
                                  label={`${profile.deployed_models}`}
                                  size="small"
                                  icon={<PlayArrow />}
                                  variant="outlined"
                                  color="success"
                                />
                              )}
                            </Box>
                          </TableCell>
                          <TableCell>
                            <Box display="flex" gap={1}>
                              <Tooltip title="Edit Profile">
                                <IconButton
                                  size="small"
                                  onClick={() => handleOpenDialog(profile)}
                                >
                                  <Edit />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="View Details">
                                <IconButton
                                  size="small"
                                  onClick={() => setSelectedProfileId(profile.id)}
                                >
                                  <Analytics />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Delete Profile">
                                <IconButton
                                  size="small"
                                  color="error"
                                  onClick={() => handleDeleteProfile(profile.id)}
                                >
                                  <Delete />
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <ProfileSelector
                    fullWidth
                    onProfileChange={setSelectedProfileId}
                    showMetrics
                  />
                </Grid>
                
                {selectedProfileId && (
                  <>
                    <Grid item xs={12} md={6}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Data Configuration
                          </Typography>
                          <Box display="grid" gap={2}>
                            <TextField
                              label="Data Source"
                              value={
                                profiles?.find(p => p.id === selectedProfileId)?.data_source || ''
                              }
                              disabled
                              fullWidth
                            />
                            <TextField
                              label="Timeframe"
                              value={
                                profiles?.find(p => p.id === selectedProfileId)?.timeframe || ''
                              }
                              disabled
                              fullWidth
                            />
                            <TextField
                              label="Lookback Days"
                              type="number"
                              value={
                                profiles?.find(p => p.id === selectedProfileId)?.lookback_days || 0
                              }
                              disabled
                              fullWidth
                            />
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>

                    <Grid item xs={12} md={6}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Risk Management
                          </Typography>
                          <Box display="grid" gap={2}>
                            <TextField
                              label="Max Drawdown Limit"
                              value={
                                profiles?.find(p => p.id === selectedProfileId)?.max_drawdown_limit || 0
                              }
                              disabled
                              fullWidth
                              InputProps={{
                                endAdornment: '%',
                              }}
                            />
                            <TextField
                              label="Position Risk Limit"
                              value={
                                profiles?.find(p => p.id === selectedProfileId)?.position_risk_limit || 0
                              }
                              disabled
                              fullWidth
                              InputProps={{
                                endAdornment: '%',
                              }}
                            />
                            <TextField
                              label="Daily Loss Limit"
                              value={
                                profiles?.find(p => p.id === selectedProfileId)?.daily_loss_limit || 0
                              }
                              disabled
                              fullWidth
                              InputProps={{
                                endAdornment: '%',
                              }}
                            />
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  </>
                )}
              </Grid>
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              {selectedProfileId && latestMetrics && (
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="text.secondary" gutterBottom>
                          Current Price
                        </Typography>
                        <Typography variant="h5">
                          ${latestMetrics.current_price.toFixed(2)}
                        </Typography>
                        <Box display="flex" gap={1} mt={1}>
                          <Chip
                            label={`24h: ${latestMetrics.price_change_24h.toFixed(2)}%`}
                            size="small"
                            color={latestMetrics.price_change_24h >= 0 ? 'success' : 'error'}
                          />
                          <Chip
                            label={`7d: ${latestMetrics.price_change_7d.toFixed(2)}%`}
                            size="small"
                            variant="outlined"
                          />
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="text.secondary" gutterBottom>
                          Technical Indicators
                        </Typography>
                        <Box display="grid" gap={1}>
                          <Box display="flex" justifyContent="space-between">
                            <Typography variant="body2">RSI</Typography>
                            <Typography variant="body2">{latestMetrics.rsi.toFixed(2)}</Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between">
                            <Typography variant="body2">MACD</Typography>
                            <Typography variant="body2">{latestMetrics.macd.toFixed(2)}</Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between">
                            <Typography variant="body2">SMA 20</Typography>
                            <Typography variant="body2">${latestMetrics.sma_20.toFixed(2)}</Typography>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="text.secondary" gutterBottom>
                          Volume
                        </Typography>
                        <Typography variant="h5">
                          ${(latestMetrics.volume_24h / 1e6).toFixed(2)}M
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          24h Volume
                        </Typography>
                        <Box mt={1}>
                          <Typography variant="body2">
                            Change: {latestMetrics.volume_change_24h.toFixed(2)}%
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              )}
            </TabPanel>
          </Paper>
        </Grid>
      </Grid>

      {/* Create/Edit Profile Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingProfile ? 'Edit Profile' : 'Create New Profile'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Symbol"
                value={formData.symbol}
                onChange={(e) => setFormData({ ...formData, symbol: e.target.value })}
                disabled={!!editingProfile}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Profile Type</InputLabel>
                <Select
                  value={formData.profile_type}
                  label="Profile Type"
                  onChange={(e) =>
                    setFormData({ ...formData, profile_type: e.target.value as any })
                  }
                >
                  <MenuItem value="crypto">Cryptocurrency</MenuItem>
                  <MenuItem value="stock">Stock</MenuItem>
                  <MenuItem value="forex">Forex</MenuItem>
                  <MenuItem value="commodity">Commodity</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Exchange"
                value={formData.exchange}
                onChange={(e) => setFormData({ ...formData, exchange: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Base Currency"
                value={formData.base_currency}
                onChange={(e) => setFormData({ ...formData, base_currency: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Quote Currency"
                value={formData.quote_currency}
                onChange={(e) => setFormData({ ...formData, quote_currency: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={2}
                label="Description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Trading Configuration
              </Typography>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Min Trade Size"
                value={formData.min_trade_size}
                onChange={(e) =>
                  setFormData({ ...formData, min_trade_size: parseFloat(e.target.value) })
                }
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Max Trade Size"
                value={formData.max_trade_size}
                onChange={(e) =>
                  setFormData({ ...formData, max_trade_size: parseFloat(e.target.value) })
                }
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Max Position Size"
                value={formData.max_position_size}
                onChange={(e) =>
                  setFormData({ ...formData, max_position_size: parseFloat(e.target.value) })
                }
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Risk Management
              </Typography>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Max Drawdown Limit (%)"
                value={formData.max_drawdown_limit * 100}
                onChange={(e) =>
                  setFormData({ ...formData, max_drawdown_limit: parseFloat(e.target.value) / 100 })
                }
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Position Risk Limit (%)"
                value={formData.position_risk_limit * 100}
                onChange={(e) =>
                  setFormData({ ...formData, position_risk_limit: parseFloat(e.target.value) / 100 })
                }
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Daily Loss Limit (%)"
                value={formData.daily_loss_limit * 100}
                onChange={(e) =>
                  setFormData({ ...formData, daily_loss_limit: parseFloat(e.target.value) / 100 })
                }
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleSaveProfile} variant="contained">
            {editingProfile ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};