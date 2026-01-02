import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Stepper,
  Step,
  StepLabel,
  Button,
  Paper,
  LinearProgress,
  Alert,
  Grid,
  Card,
  CardContent,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  CircularProgress,
} from '@mui/material';
import {
  CloudDownload,
  Settings,
  Storage,
  CheckCircle,
  Error as ErrorIcon,
  ModelTraining,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { ProfileSelector } from '../components';
import { useGetProfilesQuery } from '../store/slices/profileSlice';
import { websocketService } from '../services/websocket';
import axios from 'axios';

const steps = ['Select Symbol', 'Configure Collection', 'Collect Data', 'Complete'];

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

interface CollectionProgress {
  jobId: string;
  status: 'pending' | 'fetching' | 'preprocessing' | 'storing' | 'completed' | 'failed';
  progress: number;
  currentStage: string;
  totalRecords: number;
  errorMessage?: string;
}

export const DataCollection: React.FC = () => {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [selectedProfileId, setSelectedProfileId] = useState<number | null>(null);
  const [daysBack, setDaysBack] = useState(30);
  const [exchange, setExchange] = useState('binance');
  const [isCollecting, setIsCollecting] = useState(false);
  const [collectionProgress, setCollectionProgress] = useState<CollectionProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { data: profiles } = useGetProfilesQuery({});

  const selectedProfile = profiles?.find(p => p.id === selectedProfileId);

  useEffect(() => {
    // Connect WebSocket
    websocketService.connect();

    // Subscribe to data collection events
    const handleProgress = (data: any) => {
      setCollectionProgress({
        jobId: data.job_id,
        status: data.status,
        progress: data.progress,
        currentStage: data.current_stage,
        totalRecords: data.total_records || 0,
      });
    };

    const handleCompleted = (data: any) => {
      setCollectionProgress({
        jobId: data.job_id,
        status: 'completed',
        progress: 100,
        currentStage: 'completed',
        totalRecords: data.total_records,
      });
      setIsCollecting(false);
      setActiveStep(3);
    };

    const handleFailed = (data: any) => {
      setCollectionProgress({
        jobId: data.job_id,
        status: 'failed',
        progress: 0,
        currentStage: 'failed',
        totalRecords: 0,
        errorMessage: data.error,
      });
      setError(data.error);
      setIsCollecting(false);
    };

    websocketService.on('dataCollectionProgress', handleProgress);
    websocketService.on('dataCollectionCompleted', handleCompleted);
    websocketService.on('dataCollectionFailed', handleFailed);

    return () => {
      websocketService.off('dataCollectionProgress', handleProgress);
      websocketService.off('dataCollectionCompleted', handleCompleted);
      websocketService.off('dataCollectionFailed', handleFailed);
    };
  }, []);

  const handleNext = () => {
    if (activeStep === 0 && !selectedProfileId) {
      setError('Please select a symbol first');
      return;
    }
    setError(null);
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleStartCollection = async () => {
    if (!selectedProfile) return;

    setIsCollecting(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/admin/collect-data`, {
        symbol: selectedProfile.symbol,
        days_back: daysBack,
        exchange: exchange,
      });

      setCollectionProgress({
        jobId: response.data.job_id,
        status: 'pending',
        progress: 0,
        currentStage: 'initializing',
        totalRecords: 0,
      });

      setActiveStep(2);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start data collection');
      setIsCollecting(false);
    }
  };

  const handleTrainModels = () => {
    navigate('/models', { state: { selectedProfileId, autoTrain: true } });
  };

  const getStageIcon = (stage: string) => {
    switch (stage) {
      case 'fetching':
        return '=';
      case 'preprocessing':
        return '�';
      case 'storing':
        return '=�';
      case 'completed':
        return '';
      case 'failed':
        return 'L';
      default:
        return '�';
    }
  };

  const getProgressColor = (progress: number) => {
    if (progress < 33) return 'info';
    if (progress < 66) return 'warning';
    return 'success';
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Select Trading Symbol
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Choose the symbol you want to collect historical data for. Data will be fetched at 1-minute intervals.
            </Typography>

            <ProfileSelector
              fullWidth
              onProfileChange={setSelectedProfileId}
              showMetrics
            />

            {selectedProfile && (
              <Card sx={{ mt: 3 }}>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Selected Symbol Information
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Symbol
                      </Typography>
                      <Typography variant="h6">{selectedProfile.symbol}</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Type
                      </Typography>
                      <Chip label={selectedProfile.profile_type} color="primary" size="small" />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Exchange
                      </Typography>
                      <Typography>{selectedProfile.exchange}</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Current Data
                      </Typography>
                      <Chip
                        label={selectedProfile.has_data ? 'Has Data' : 'No Data'}
                        color={selectedProfile.has_data ? 'success' : 'default'}
                        size="small"
                      />
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            )}
          </Box>
        );

      case 1:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Configure Data Collection
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Set the parameters for data collection. All data is collected at 1-minute intervals.
            </Typography>

            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      Lookback Period
                    </Typography>
                    <Box sx={{ px: 2, pt: 2 }}>
                      <Slider
                        value={daysBack}
                        onChange={(_, value) => setDaysBack(value as number)}
                        marks={[
                          { value: 7, label: '7d' },
                          { value: 30, label: '30d' },
                          { value: 90, label: '90d' },
                          { value: 365, label: '1y' },
                        ]}
                        step={null}
                        min={7}
                        max={365}
                        valueLabelDisplay="on"
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                      Selected: {daysBack} days (~{(daysBack * 1440).toLocaleString()} 1-minute candles)
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <FormControl fullWidth>
                      <InputLabel>Exchange</InputLabel>
                      <Select
                        value={exchange}
                        label="Exchange"
                        onChange={(e) => setExchange(e.target.value)}
                      >
                        <MenuItem value="binance">Binance</MenuItem>
                        <MenuItem value="binance_us">Binance US</MenuItem>
                        <MenuItem value="coinbase">Coinbase Pro</MenuItem>
                      </Select>
                    </FormControl>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      Data interval: 1 minute (fixed for accurate model training)
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Alert severity="info">
                  <Typography variant="body2">
                    <strong>Note:</strong> Data collection will fetch historical 1-minute candles and store them in the database.
                    Timeframe aggregations (5m, 1h, 1D, etc.) will be generated on-demand during training and analysis.
                  </Typography>
                </Alert>
              </Grid>
            </Grid>
          </Box>
        );

      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Collecting Data
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Please wait while we fetch and process historical data for {selectedProfile?.symbol}
            </Typography>

            {collectionProgress && (
              <Box>
                <Box sx={{ mb: 4 }}>
                  <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                    <Typography variant="h3" className="progress-percentage">
                      {collectionProgress.progress}%
                    </Typography>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="h4">{getStageIcon(collectionProgress.currentStage)}</Typography>
                      <Typography variant="body1" color="text.secondary">
                        {collectionProgress.currentStage.charAt(0).toUpperCase() + collectionProgress.currentStage.slice(1)}
                      </Typography>
                    </Box>
                  </Box>

                  <LinearProgress
                    variant="determinate"
                    value={collectionProgress.progress}
                    sx={{ height: 10, borderRadius: 5 }}
                    color={getProgressColor(collectionProgress.progress)}
                  />
                </Box>

                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Card className={collectionProgress.progress >= 10 ? 'stage-indicator active' : 'stage-indicator'}>
                      <CardContent>
                        <Box display="flex" alignItems="center" gap={2}>
                          <CloudDownload fontSize="large" color={collectionProgress.progress >= 10 ? 'primary' : 'disabled'} />
                          <Box>
                            <Typography variant="subtitle1">Fetching</Typography>
                            <Typography variant="caption" color="text.secondary">
                              Downloading from exchange
                            </Typography>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Card className={collectionProgress.progress >= 40 ? 'stage-indicator active' : 'stage-indicator'}>
                      <CardContent>
                        <Box display="flex" alignItems="center" gap={2}>
                          <Settings fontSize="large" color={collectionProgress.progress >= 40 ? 'warning' : 'disabled'} />
                          <Box>
                            <Typography variant="subtitle1">Preprocessing</Typography>
                            <Typography variant="caption" color="text.secondary">
                              Cleaning and validating
                            </Typography>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Card className={collectionProgress.progress >= 70 ? 'stage-indicator active' : 'stage-indicator'}>
                      <CardContent>
                        <Box display="flex" alignItems="center" gap={2}>
                          <Storage fontSize="large" color={collectionProgress.progress >= 70 ? 'success' : 'disabled'} />
                          <Box>
                            <Typography variant="subtitle1">Storing</Typography>
                            <Typography variant="caption" color="text.secondary">
                              Saving to database
                            </Typography>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>

                {collectionProgress.totalRecords > 0 && (
                  <Alert severity="info" sx={{ mt: 3 }}>
                    <Typography variant="body2">
                      <strong>Progress:</strong> {collectionProgress.totalRecords.toLocaleString()} records processed
                    </Typography>
                  </Alert>
                )}
              </Box>
            )}

            {isCollecting && !collectionProgress && (
              <Box display="flex" flexDirection="column" alignItems="center" py={4}>
                <CircularProgress size={60} />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  Initializing data collection...
                </Typography>
              </Box>
            )}
          </Box>
        );

      case 3:
        return (
          <Box textAlign="center">
            <CheckCircle color="success" sx={{ fontSize: 80, mb: 2 }} />
            <Typography variant="h5" gutterBottom>
              Data Collection Complete!
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              Successfully collected {collectionProgress?.totalRecords.toLocaleString()} records for {selectedProfile?.symbol}
            </Typography>

            <Card sx={{ mt: 4, textAlign: 'left' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Next Steps
                </Typography>
                <Box display="flex" flexDirection="column" gap={2}>
                  <Box display="flex" alignItems="center" gap={2}>
                    <ModelTraining color="primary" />
                    <Typography variant="body2">
                      Train machine learning models using the collected data
                    </Typography>
                  </Box>
                  <Button
                    variant="contained"
                    startIcon={<ModelTraining />}
                    onClick={handleTrainModels}
                    fullWidth
                  >
                    Train Models Now
                  </Button>
                </Box>
              </CardContent>
            </Card>

            <Button
              variant="outlined"
              onClick={() => {
                setActiveStep(0);
                setSelectedProfileId(null);
                setCollectionProgress(null);
                setError(null);
              }}
              sx={{ mt: 3 }}
            >
              Collect More Data
            </Button>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Data Collection Workflow
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Collect historical market data for training machine learning models
      </Typography>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Paper sx={{ p: 4, mt: 3 }}>
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        <Box sx={{ minHeight: 400 }}>
          {renderStepContent(activeStep)}
        </Box>

        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
          <Button
            disabled={activeStep === 0 || isCollecting}
            onClick={handleBack}
          >
            Back
          </Button>
          <Box>
            {activeStep === 1 && (
              <Button
                variant="contained"
                onClick={handleStartCollection}
                disabled={!selectedProfileId || isCollecting}
                startIcon={<CloudDownload />}
              >
                Start Collection
              </Button>
            )}
            {activeStep === 0 && (
              <Button
                variant="contained"
                onClick={handleNext}
                disabled={!selectedProfileId}
              >
                Next
              </Button>
            )}
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};
