import React, { useEffect, useState } from 'react';
import {
  Paper,
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  ShowChart,
  CurrencyBitcoin,
  Speed,
  Analytics,
  Warning,
  Refresh,
  Timeline,
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import { ProfileSelector } from '../components';
import {
  useGetProfilesQuery,
  useGetLatestMetricsQuery,
  useGetMetricsHistoryQuery,
  useGetProfilePredictionsQuery,
  useGetProfileModelsQuery,
} from '../store/slices/profileSlice';
import { useGetSystemStatusQuery } from '../services/systemApi';
import { useGetPerformanceQuery } from '../services/monitoringApi';
import { useGetPositionsQuery } from '../services/tradingApi';
import { createChart, ColorType, LineData } from 'lightweight-charts';
import { format } from 'date-fns';

export const Dashboard: React.FC = () => {
  const selectedProfileId = useSelector((state: RootState) => state.profile.selectedProfileId);
  const [chartData, setChartData] = useState<LineData[]>([]);
  
  // System-wide queries
  const { data: systemStatus } = useGetSystemStatusQuery(undefined, {
    pollingInterval: 5000,
  });
  const { data: performance } = useGetPerformanceQuery({ period: '24h' }, {
    pollingInterval: 30000,
  });
  const { data: positions } = useGetPositionsQuery('OPEN', {
    pollingInterval: 10000,
  });
  
  // Profile-specific queries
  const { data: profiles } = useGetProfilesQuery({ is_active: true });
  const { data: latestMetrics } = useGetLatestMetricsQuery(
    selectedProfileId!,
    { skip: !selectedProfileId, pollingInterval: 10000 }
  );
  const { data: metricsHistory } = useGetMetricsHistoryQuery(
    { profileId: selectedProfileId!, hours: 24 },
    { skip: !selectedProfileId }
  );
  const { data: predictions } = useGetProfilePredictionsQuery(
    { profileId: selectedProfileId!, hours: 6 },
    { skip: !selectedProfileId }
  );
  const { data: models } = useGetProfileModelsQuery(
    { profileId: selectedProfileId!, is_deployed: true },
    { skip: !selectedProfileId }
  );

  const selectedProfile = profiles?.find(p => p.id === selectedProfileId);

  // Profile-specific metrics
  const profileMetrics = selectedProfile ? [
    {
      title: 'Profile P&L',
      value: `$${selectedProfile.total_pnl.toFixed(2)}`,
      icon: <AccountBalance />,
      color: selectedProfile.total_pnl >= 0 ? 'success.main' : 'error.main',
      subtitle: selectedProfile.symbol,
    },
    {
      title: 'Win Rate',
      value: `${(selectedProfile.win_rate * 100).toFixed(1)}%`,
      icon: <ShowChart />,
      color: 'info.main',
      subtitle: `${selectedProfile.total_trades} trades`,
    },
    {
      title: 'Sharpe Ratio',
      value: selectedProfile.sharpe_ratio.toFixed(2),
      icon: <Analytics />,
      color: selectedProfile.sharpe_ratio > 1.5 ? 'success.main' : 'warning.main',
      subtitle: 'Risk-adjusted returns',
    },
    {
      title: 'Current Price',
      value: `$${latestMetrics?.current_price.toFixed(4) || '0.00'}`,
      icon: <CurrencyBitcoin />,
      color: 'primary.main',
      subtitle: latestMetrics ? `${latestMetrics.price_change_24h > 0 ? '+' : ''}${latestMetrics.price_change_24h.toFixed(2)}%` : 'Loading...',
    },
  ] : [];

  // System-wide metrics
  const systemMetrics = [
    {
      title: 'Total System P&L',
      value: `$${performance?.total_pnl?.toFixed(2) || '0.00'}`,
      icon: <AccountBalance />,
      color: (performance?.total_pnl || 0) >= 0 ? 'success.main' : 'error.main',
      subtitle: 'All profiles combined',
    },
    {
      title: 'Active Positions',
      value: positions?.length || 0,
      icon: <TrendingUp />,
      color: 'warning.main',
      subtitle: 'Across all profiles',
    },
    {
      title: 'System Status',
      value: systemStatus?.status || 'UNKNOWN',
      icon: <Speed />,
      color: systemStatus?.status === 'RUNNING' ? 'success.main' : 'warning.main',
      subtitle: 'Trading engine',
    },
    {
      title: 'Active Profiles',
      value: profiles?.filter(p => p.is_active).length || 0,
      icon: <Timeline />,
      color: 'primary.main',
      subtitle: `${profiles?.length || 0} total`,
    },
  ];

  useEffect(() => {
    if (metricsHistory && metricsHistory.length > 0) {
      const data: LineData[] = metricsHistory.map(m => ({
        time: new Date(m.timestamp).getTime() / 1000 as any,
        value: m.current_price,
      })).reverse();
      setChartData(data);
    }
  }, [metricsHistory]);

  useEffect(() => {
    if (chartData.length > 0) {
      const chartContainer = document.getElementById('price-chart');
      if (chartContainer) {
        chartContainer.innerHTML = '';
        const chart = createChart(chartContainer, {
          layout: {
            background: { type: ColorType.Solid, color: 'transparent' },
            textColor: '#d1d4dc',
          },
          grid: {
            vertLines: { color: '#2B2B43' },
            horzLines: { color: '#363C4E' },
          },
          width: chartContainer.clientWidth,
          height: 350,
        });

        const lineSeries = chart.addLineSeries({
          color: '#2962FF',
          lineWidth: 2,
        });

        lineSeries.setData(chartData);
        chart.timeScale().fitContent();

        const handleResize = () => {
          chart.applyOptions({ width: chartContainer.clientWidth });
        };
        
        window.addEventListener('resize', handleResize);
        
        return () => {
          window.removeEventListener('resize', handleResize);
          chart.remove();
        };
      }
    }
  }, [chartData]);

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Dashboard</Typography>
        <Box display="flex" gap={2}>
          <ProfileSelector showMetrics />
          <IconButton onClick={() => window.location.reload()}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>
      
      {!selectedProfileId && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Select a trading profile to view profile-specific metrics
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Profile Metrics */}
        {selectedProfileId && (
          <>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Profile Metrics - {selectedProfile?.name}
              </Typography>
            </Grid>
            {profileMetrics.map((metric, index) => (
              <Grid item xs={12} sm={6} md={3} key={`profile-${index}`}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Box
                        sx={{
                          p: 1,
                          borderRadius: 1,
                          backgroundColor: `${metric.color}20`,
                          color: metric.color,
                          mr: 2,
                        }}
                      >
                        {metric.icon}
                      </Box>
                      <Box>
                        <Typography color="text.secondary" variant="body2">
                          {metric.title}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {metric.subtitle}
                        </Typography>
                      </Box>
                    </Box>
                    <Typography variant="h5" component="div" color={metric.color}>
                      {metric.value}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </>
        )}

        {/* System Metrics */}
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
            System Overview
          </Typography>
        </Grid>
        {systemMetrics.map((metric, index) => (
          <Grid item xs={12} sm={6} md={3} key={`system-${index}`}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Box
                    sx={{
                      p: 1,
                      borderRadius: 1,
                      backgroundColor: `${metric.color}20`,
                      color: metric.color,
                      mr: 2,
                    }}
                  >
                    {metric.icon}
                  </Box>
                  <Box>
                    <Typography color="text.secondary" variant="body2">
                      {metric.title}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {metric.subtitle}
                    </Typography>
                  </Box>
                </Box>
                <Typography variant="h5" component="div">
                  {metric.value}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}

        {/* Price Chart */}
        {selectedProfileId && (
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Price History - {selectedProfile?.symbol}
              </Typography>
              <Box id="price-chart" sx={{ height: 350 }} />
              {metricsHistory && metricsHistory.length === 0 && (
                <Box sx={{ 
                  height: 350, 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  color: 'text.secondary'
                }}>
                  No price data available
                </Box>
              )}
            </Paper>
          </Grid>
        )}

        {/* Model Status & Predictions */}
        {selectedProfileId && (
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2, height: selectedProfileId ? 400 : 'auto' }}>
              <Typography variant="h6" gutterBottom>
                Active Models & Predictions
              </Typography>
              
              {models && models.length > 0 ? (
                <Box>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Deployed Models ({models.length})
                  </Typography>
                  {models.slice(0, 3).map((model) => (
                    <Box key={model.id} sx={{ mb: 2, p: 1, bgcolor: 'background.default', borderRadius: 1 }}>
                      <Typography variant="body2" fontWeight="bold">
                        {model.model_name}
                      </Typography>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="caption" color="text.secondary">
                          Type: {model.model_type}
                        </Typography>
                        <Typography variant="caption">
                          Accuracy: {model.test_accuracy ? `${(model.test_accuracy * 100).toFixed(1)}%` : 'N/A'}
                        </Typography>
                      </Box>
                    </Box>
                  ))}
                </Box>
              ) : (
                <Typography color="text.secondary" variant="body2">
                  No deployed models
                </Typography>
              )}

              {predictions && predictions.length > 0 && (
                <Box mt={3}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Latest Predictions
                  </Typography>
                  {predictions.slice(0, 2).map((pred, idx) => (
                    <Box key={idx} sx={{ mb: 2, p: 1, bgcolor: 'background.default', borderRadius: 1 }}>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Box>
                          <Typography variant="body2">
                            Target: ${pred.price_prediction?.toFixed(4) || 'N/A'}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {pred.prediction_horizon} horizon
                          </Typography>
                        </Box>
                        <Chip
                          label={pred.signal || 'HOLD'}
                          size="small"
                          color={
                            pred.signal === 'BUY' ? 'success' :
                            pred.signal === 'SELL' ? 'error' : 'default'
                          }
                        />
                      </Box>
                      {pred.confidence && (
                        <LinearProgress
                          variant="determinate"
                          value={pred.confidence * 100}
                          sx={{ mt: 1 }}
                        />
                      )}
                    </Box>
                  ))}
                </Box>
              )}
            </Paper>
          </Grid>
        )}

        {/* All Profiles Summary */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Trading Profiles Summary
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Profile</TableCell>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell align="right">Current Price</TableCell>
                    <TableCell align="right">24h Change</TableCell>
                    <TableCell align="right">Total P&L</TableCell>
                    <TableCell align="right">Win Rate</TableCell>
                    <TableCell align="right">Models</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {profiles?.slice(0, 10).map((profile) => (
                    <TableRow key={profile.id}>
                      <TableCell>{profile.name}</TableCell>
                      <TableCell>
                        <Chip label={profile.symbol} size="small" />
                      </TableCell>
                      <TableCell>{profile.profile_type}</TableCell>
                      <TableCell align="right">
                        ${profile.current_price?.toFixed(4) || 'N/A'}
                      </TableCell>
                      <TableCell align="right">
                        <Typography
                          variant="body2"
                          color={
                            profile.price_change_24h && profile.price_change_24h > 0
                              ? 'success.main'
                              : 'error.main'
                          }
                        >
                          {profile.price_change_24h ? `${profile.price_change_24h.toFixed(2)}%` : 'N/A'}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography
                          variant="body2"
                          color={profile.total_pnl >= 0 ? 'success.main' : 'error.main'}
                        >
                          ${profile.total_pnl.toFixed(2)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        {(profile.win_rate * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell align="right">
                        <Box display="flex" gap={0.5} justifyContent="flex-end">
                          <Chip
                            label={profile.active_models}
                            size="small"
                            color="info"
                            variant="outlined"
                          />
                          {profile.deployed_models > 0 && (
                            <Chip
                              label={profile.deployed_models}
                              size="small"
                              color="success"
                            />
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        {/* Recent Trades */}
        {positions && positions.length > 0 && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Active Positions
              </Typography>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Side</TableCell>
                      <TableCell align="right">Quantity</TableCell>
                      <TableCell align="right">Entry Price</TableCell>
                      <TableCell align="right">Current Price</TableCell>
                      <TableCell align="right">P&L</TableCell>
                      <TableCell align="right">P&L %</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {positions.slice(0, 5).map((position) => (
                      <TableRow key={position.id}>
                        <TableCell>{position.symbol}</TableCell>
                        <TableCell>
                          <Chip
                            label={position.side}
                            size="small"
                            color={position.side === 'BUY' ? 'success' : 'error'}
                          />
                        </TableCell>
                        <TableCell align="right">{position.quantity}</TableCell>
                        <TableCell align="right">${position.entry_price.toFixed(4)}</TableCell>
                        <TableCell align="right">
                          ${position.current_price?.toFixed(4) || '-'}
                        </TableCell>
                        <TableCell align="right">
                          <Typography
                            variant="body2"
                            color={(position.unrealized_pnl || 0) >= 0 ? 'success.main' : 'error.main'}
                          >
                            ${position.unrealized_pnl?.toFixed(2) || '0.00'}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography
                            variant="body2"
                            color={(position.unrealized_pnl || 0) >= 0 ? 'success.main' : 'error.main'}
                          >
                            {position.unrealized_pnl && position.entry_price
                              ? `${((position.unrealized_pnl / (position.quantity * position.entry_price)) * 100).toFixed(2)}%`
                              : '0.00%'}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};