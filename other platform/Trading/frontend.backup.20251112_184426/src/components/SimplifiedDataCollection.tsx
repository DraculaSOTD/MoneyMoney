import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  LinearProgress,
  Alert,
  Chip,
  Grid,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  CloudDownload,
  CheckCircle,
  Error as ErrorIcon,
  Storage,
  Transform,
  Assessment,
} from '@mui/icons-material';
import axios from 'axios';
import { useDataCollectionWebSocket } from '../hooks/useWebSocket';

interface DataCollectionJob {
  job_id: string;
  symbol: string;
  status: string;
  progress: number;
  current_stage?: string;
  total_records: number;
  error_message?: string;
  started_at: string;
  completed_at?: string;
}

interface AvailableSymbol {
  symbol: string;
  name: string;
  has_data: boolean;
  models_trained: boolean;
  data_updated_at?: string;
  total_data_points: number;
}

const POPULAR_SYMBOLS = [
  { symbol: 'BTCUSDT', name: 'Bitcoin' },
  { symbol: 'ETHUSDT', name: 'Ethereum' },
  { symbol: 'BNBUSDT', name: 'Binance Coin' },
  { symbol: 'ADAUSDT', name: 'Cardano' },
  { symbol: 'SOLUSDT', name: 'Solana' },
  { symbol: 'XRPUSDT', name: 'Ripple' },
  { symbol: 'DOTUSDT', name: 'Polkadot' },
  { symbol: 'DOGEUSDT', name: 'Dogecoin' },
];

export const SimplifiedDataCollection: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [customSymbol, setCustomSymbol] = useState('');
  const [daysBack, setDaysBack] = useState(30);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [currentJob, setCurrentJob] = useState<DataCollectionJob | null>(null);
  const [availableSymbols, setAvailableSymbols] = useState<AvailableSymbol[]>([]);

  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001';
  const token = localStorage.getItem('adminToken');

  // WebSocket connection for real-time progress updates
  const { status: wsStatus, progress: wsProgress, lastMessage } = useDataCollectionWebSocket(token);

  // Fetch available symbols on mount
  useEffect(() => {
    fetchAvailableSymbols();
  }, []);

  // Handle WebSocket messages for current job
  useEffect(() => {
    if (!lastMessage || !currentJob) return;

    const { type, job_id, progress, status, stage, error: wsError, total_data_points } = lastMessage;

    // Only update if message is for current job
    if (job_id && job_id === currentJob.job_id) {
      setCurrentJob(prev => prev ? {
        ...prev,
        progress: progress || prev.progress,
        status: status || prev.status,
        current_stage: stage || prev.current_stage,
        error_message: wsError,
      } : null);

      // Handle completion
      if (type === 'data_collection_completed') {
        setSuccess(`Data collection completed! ${total_data_points || 0} records collected.`);
        fetchAvailableSymbols();
      }

      // Handle failure
      if (type === 'data_collection_failed') {
        setError(wsError || 'Data collection failed');
      }
    }
  }, [lastMessage, currentJob]);

  const fetchAvailableSymbols = async () => {
    try {
      const token = localStorage.getItem('adminToken');
      const response = await axios.get(`${apiUrl}/admin/data/available`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setAvailableSymbols(response.data);
    } catch (err) {
      console.error('Error fetching available symbols:', err);
    }
  };


  const handleStartCollection = async () => {
    const symbol = selectedSymbol || customSymbol.toUpperCase();

    if (!symbol) {
      setError('Please select or enter a symbol');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);
    setCurrentJob(null);

    try {
      const token = localStorage.getItem('adminToken');
      const response = await axios.post(
        `${apiUrl}/admin/data/collect/${symbol}`,
        null,
        {
          params: { days_back: daysBack },
          headers: { Authorization: `Bearer ${token}` },
        }
      );

      setCurrentJob({
        job_id: response.data.job_id,
        symbol: response.data.symbol,
        status: 'pending',
        progress: 0,
        total_records: 0,
        started_at: new Date().toISOString(),
      });

      setSuccess(`Data collection started for ${symbol}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start data collection');
    } finally {
      setLoading(false);
    }
  };

  const getProgressStage = (progress: number) => {
    if (progress < 33) return { stage: 'Fetching', icon: <CloudDownload />, color: 'primary' };
    if (progress < 66) return { stage: 'Preprocessing', icon: <Transform />, color: 'secondary' };
    if (progress < 100) return { stage: 'Storing', icon: <Storage />, color: 'warning' };
    return { stage: 'Completed', icon: <CheckCircle />, color: 'success' };
  };

  const currentStage = currentJob ? getProgressStage(currentJob.progress) : null;

  return (
    <Box>
      <Typography variant="h4" gutterBottom className="gradient-text">
        Data Collection Workflow
      </Typography>
      <Typography variant="body1" color="textSecondary" paragraph>
        Simple workflow: Select Coin → Get Data → Progress Bar → Store
      </Typography>

      {/* Main Collection Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Step 1: Select Cryptocurrency
          </Typography>

          <Grid container spacing={3}>
            {/* Quick Select */}
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Popular Symbols</InputLabel>
                <Select
                  value={selectedSymbol}
                  onChange={(e) => {
                    setSelectedSymbol(e.target.value);
                    setCustomSymbol('');
                  }}
                  label="Popular Symbols"
                >
                  <MenuItem value="">
                    <em>Select a symbol</em>
                  </MenuItem>
                  {POPULAR_SYMBOLS.map((s) => (
                    <MenuItem key={s.symbol} value={s.symbol}>
                      {s.name} ({s.symbol})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* Custom Symbol */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Or Enter Custom Symbol"
                placeholder="e.g., MATICUSDT"
                value={customSymbol}
                onChange={(e) => {
                  setCustomSymbol(e.target.value);
                  setSelectedSymbol('');
                }}
                helperText="Symbol must be available on Binance"
              />
            </Grid>

            {/* Days Back */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Days of Historical Data"
                value={daysBack}
                onChange={(e) => setDaysBack(parseInt(e.target.value) || 30)}
                inputProps={{ min: 1, max: 365 }}
                helperText="1-365 days (default: 30)"
              />
            </Grid>

            {/* Action Button */}
            <Grid item xs={12} md={6} sx={{ display: 'flex', alignItems: 'center' }}>
              <Button
                fullWidth
                variant="contained"
                color="primary"
                size="large"
                onClick={handleStartCollection}
                disabled={loading || (currentJob?.status !== 'completed' && currentJob?.status !== 'failed' && !!currentJob)}
                startIcon={loading ? <CircularProgress size={20} /> : <CloudDownload />}
              >
                {loading ? 'Starting...' : 'Start Data Collection'}
              </Button>
            </Grid>
          </Grid>

          {/* Error/Success Messages */}
          {error && (
            <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          )}
          {success && !currentJob && (
            <Alert severity="success" sx={{ mt: 2 }} onClose={() => setSuccess(null)}>
              {success}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Progress Card */}
      {currentJob && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">
                Step 2: Data Collection Progress
              </Typography>
              <Chip
                label={currentJob.status.toUpperCase()}
                color={
                  currentJob.status === 'completed' ? 'success' :
                  currentJob.status === 'failed' ? 'error' :
                  'primary'
                }
                icon={currentJob.status === 'completed' ? <CheckCircle /> : currentJob.status === 'failed' ? <ErrorIcon /> : undefined}
              />
            </Box>

            <Typography variant="body2" color="textSecondary" gutterBottom>
              Symbol: <strong>{currentJob.symbol}</strong>
            </Typography>

            {/* Progress Bar */}
            <Box sx={{ mt: 3, mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="textSecondary">
                  {currentStage?.stage} ({currentJob.progress}%)
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {currentJob.total_records > 0 ? `${currentJob.total_records} records` : 'Processing...'}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={currentJob.progress}
                sx={{ height: 10, borderRadius: 1 }}
              />
            </Box>

            {/* Stage Indicators */}
            <Grid container spacing={2} sx={{ mt: 2 }}>
              <Grid item xs={4}>
                <Box sx={{ textAlign: 'center' }}>
                  <CloudDownload
                    sx={{
                      fontSize: 40,
                      color: currentJob.progress >= 0 ? 'primary.main' : 'text.disabled',
                    }}
                  />
                  <Typography variant="caption" display="block">
                    Fetching (0-33%)
                  </Typography>
                  {currentJob.progress > 0 && currentJob.progress <= 33 && (
                    <CircularProgress size={20} sx={{ mt: 1 }} />
                  )}
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box sx={{ textAlign: 'center' }}>
                  <Transform
                    sx={{
                      fontSize: 40,
                      color: currentJob.progress >= 33 ? 'secondary.main' : 'text.disabled',
                    }}
                  />
                  <Typography variant="caption" display="block">
                    Preprocessing (34-66%)
                  </Typography>
                  {currentJob.progress > 33 && currentJob.progress <= 66 && (
                    <CircularProgress size={20} sx={{ mt: 1 }} />
                  )}
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box sx={{ textAlign: 'center' }}>
                  <Storage
                    sx={{
                      fontSize: 40,
                      color: currentJob.progress >= 66 ? 'warning.main' : 'text.disabled',
                    }}
                  />
                  <Typography variant="caption" display="block">
                    Storing (67-100%)
                  </Typography>
                  {currentJob.progress > 66 && currentJob.progress < 100 && (
                    <CircularProgress size={20} sx={{ mt: 1 }} />
                  )}
                </Box>
              </Grid>
            </Grid>

            {/* Completion Message */}
            {currentJob.status === 'completed' && (
              <Alert severity="success" sx={{ mt: 3 }}>
                <strong>Data collection completed successfully!</strong>
                <br />
                Collected {currentJob.total_records} one-minute interval records.
                <br />
                Data stored with 1m intervals (enforced). Ready for model training.
              </Alert>
            )}

            {/* Error Message */}
            {currentJob.status === 'failed' && currentJob.error_message && (
              <Alert severity="error" sx={{ mt: 3 }}>
                <strong>Data collection failed:</strong>
                <br />
                {currentJob.error_message}
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      {/* Available Symbols List */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Assessment />
            Available Symbols with Data
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Symbols that have data collected and are ready for model training
          </Typography>

          {availableSymbols.length === 0 ? (
            <Alert severity="info">
              No symbols with data yet. Start collecting data for a symbol above!
            </Alert>
          ) : (
            <List>
              {availableSymbols.map((symbol, index) => (
                <React.Fragment key={symbol.symbol}>
                  {index > 0 && <Divider />}
                  <ListItem>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body1" fontWeight="600">
                            {symbol.symbol}
                          </Typography>
                          {symbol.models_trained && (
                            <Chip label="Models Trained" color="success" size="small" />
                          )}
                          {!symbol.models_trained && (
                            <Chip label="Ready for Training" color="primary" size="small" />
                          )}
                        </Box>
                      }
                      secondary={
                        <Box sx={{ mt: 0.5 }}>
                          <Typography variant="caption" display="block">
                            Data Points: {symbol.total_data_points.toLocaleString()} (1-minute intervals)
                          </Typography>
                          {symbol.data_updated_at && (
                            <Typography variant="caption" display="block">
                              Updated: {new Date(symbol.data_updated_at).toLocaleString()}
                            </Typography>
                          )}
                        </Box>
                      }
                    />
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};
