import React, { useState, useEffect } from 'react';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Button,
  Paper,
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
  Alert,
  Chip,
  Grid,
  Card,
  CardContent,
  Checkbox,
  FormControlLabel,
  FormGroup,
  CircularProgress,
  IconButton,
  Collapse,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Snackbar,
} from '@mui/material';
import {
  CloudDownload,
  Transform,
  ModelTraining,
  CheckCircle,
  Error as ErrorIcon,
  ExpandMore,
  ExpandLess,
  Storage,
  Speed,
  Functions,
  Timeline,
  WifiOff,
  Wifi,
} from '@mui/icons-material';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store';
import {
  setDataFetchJob,
  setPreprocessJob,
  setModelCatalog,
  setFetchedData,
  setProcessedData,
  setDataFetchConfig,
  setPreprocessConfig,
  clearPipelineData,
  type DataPipelineJob,
} from '../features/mlModels/mlModelsSlice';
import mlPipelineApi from '../services/mlPipelineApi';
import { useTrainingWebSocket } from '../hooks/useTrainingWebSocket';

const INTERVALS = [
  { value: '1m', label: '1 Minute' },
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '30m', label: '30 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
  { value: '1d', label: '1 Day' },
];

const FEATURES = [
  { value: 'close', label: 'Close Price', category: 'price' },
  { value: 'volume', label: 'Volume', category: 'volume' },
  { value: 'high', label: 'High', category: 'price' },
  { value: 'low', label: 'Low', category: 'price' },
  { value: 'open', label: 'Open', category: 'price' },
  { value: 'rsi', label: 'RSI', category: 'technical' },
  { value: 'macd', label: 'MACD', category: 'technical' },
  { value: 'macd_signal', label: 'MACD Signal', category: 'technical' },
  { value: 'sma_20', label: 'SMA 20', category: 'technical' },
  { value: 'sma_50', label: 'SMA 50', category: 'technical' },
  { value: 'ema_12', label: 'EMA 12', category: 'technical' },
  { value: 'ema_26', label: 'EMA 26', category: 'technical' },
  { value: 'bollinger_upper', label: 'Bollinger Upper', category: 'technical' },
  { value: 'bollinger_lower', label: 'Bollinger Lower', category: 'technical' },
  { value: 'bollinger_middle', label: 'Bollinger Middle', category: 'technical' },
  { value: 'atr', label: 'ATR', category: 'volatility' },
  { value: 'obv', label: 'On Balance Volume', category: 'volume' },
  { value: 'vwap', label: 'VWAP', category: 'volume' },
  { value: 'adx', label: 'ADX', category: 'trend' },
  { value: 'cci', label: 'CCI', category: 'momentum' },
];

interface DataPipelineProps {
  profileId: number;
  onComplete?: (processedDataId: string) => void;
}

export const DataPipeline: React.FC<DataPipelineProps> = ({ profileId, onComplete }) => {
  const dispatch = useDispatch();
  const {
    dataFetchJob,
    preprocessJob,
    modelCatalog,
    fetchedData,
    processedData,
    pipelineConfig,
  } = useSelector((state: RootState) => state.mlModels);

  const [activeStep, setActiveStep] = useState(0);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    price: true,
    technical: true,
    volume: false,
    volatility: false,
    trend: false,
    momentum: false,
  });
  const [showConnectionStatus, setShowConnectionStatus] = useState(false);

  // Use WebSocket for real-time updates
  const { isConnected, connectionError, subscribeToJob } = useTrainingWebSocket({
    onJobUpdate: (jobId, update) => {
      // Handle any custom job updates here if needed
      console.log('Job update received:', jobId, update);
    },
  });

  // Data fetch form
  const [fetchForm, setFetchForm] = useState({
    symbol: 'BTCUSDT',
    interval: '1h',
    days_back: 30,
  });

  // Preprocess form
  const [preprocessForm, setPreprocessForm] = useState({
    scaling_method: 'standard' as 'standard' | 'minmax' | 'robust',
    handle_missing: 'forward_fill' as 'forward_fill' | 'interpolate' | 'drop',
    features: ['close', 'volume', 'rsi', 'macd', 'sma_20', 'sma_50'],
  });

  // Load model catalog on mount
  useEffect(() => {
    loadModelCatalog();
  }, []);

  const loadModelCatalog = async () => {
    try {
      const catalog = await mlPipelineApi.getModelCatalog();
      dispatch(setModelCatalog(catalog));
    } catch (error) {
      console.error('Failed to load model catalog:', error);
    }
  };

  const handleFetchData = async () => {
    try {
      dispatch(setDataFetchConfig(fetchForm));
      
      const job = await mlPipelineApi.fetchData({
        ...fetchForm,
        profile_id: profileId,
      });
      
      dispatch(setDataFetchJob(job));
      
      // Subscribe to job updates via WebSocket
      subscribeToJob(job.job_id);
      
      // WebSocket will handle updates, no need for polling
    } catch (error) {
      console.error('Data fetch failed:', error);
    }
  };

  // Monitor job completion
  useEffect(() => {
    if (dataFetchJob?.status === 'completed' && dataFetchJob.result) {
      dispatch(setFetchedData({
        data_id: dataFetchJob.result.data_id!,
        rows: dataFetchJob.result.rows!,
        start_date: dataFetchJob.result.start_date!,
        end_date: dataFetchJob.result.end_date!,
        columns: dataFetchJob.result.columns!,
      }));
      setActiveStep(1);
    }
  }, [dataFetchJob, dispatch]);

  const handlePreprocessData = async () => {
    if (!fetchedData) return;

    try {
      dispatch(setPreprocessConfig(preprocessForm));
      
      const job = await mlPipelineApi.preprocessData({
        ...preprocessForm,
        profile_id: profileId,
        data_id: fetchedData.data_id,
      });
      
      dispatch(setPreprocessJob(job));
      
      // Subscribe to job updates via WebSocket
      subscribeToJob(job.job_id);
      
      // WebSocket will handle updates, no need for polling
    } catch (error) {
      console.error('Preprocessing failed:', error);
    }
  };

  // Monitor preprocessing job completion
  useEffect(() => {
    if (preprocessJob?.status === 'completed' && preprocessJob.result) {
      dispatch(setProcessedData({
        processed_id: preprocessJob.result.processed_id!,
        rows: preprocessJob.result.rows!,
        features: preprocessJob.result.features!,
        scaling_method: preprocessForm.scaling_method,
      }));
      setActiveStep(2);
      
      if (onComplete) {
        onComplete(preprocessJob.result.processed_id!);
      }
    }
  }, [preprocessJob, preprocessForm.scaling_method, dispatch, onComplete]);

  const toggleFeature = (feature: string) => {
    setPreprocessForm(prev => ({
      ...prev,
      features: prev.features.includes(feature)
        ? prev.features.filter(f => f !== feature)
        : [...prev.features, feature],
    }));
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  const getFeaturesByCategory = (category: string) => {
    return FEATURES.filter(f => f.category === category);
  };

  const renderJobProgress = (job: DataPipelineJob | undefined, title: string) => {
    if (!job) return null;

    return (
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2} mb={2}>
            {job.status === 'running' && <CircularProgress size={20} />}
            {job.status === 'completed' && <CheckCircle color="success" />}
            {job.status === 'failed' && <ErrorIcon color="error" />}
            <Typography variant="subtitle1">{title}</Typography>
            <Chip
              label={job.status}
              size="small"
              color={
                job.status === 'completed' ? 'success' :
                job.status === 'failed' ? 'error' :
                job.status === 'running' ? 'warning' : 'default'
              }
            />
          </Box>
          
          {job.status === 'running' && (
            <>
              <LinearProgress
                variant="determinate"
                value={job.progress * 100}
                sx={{ mb: 1 }}
              />
              <Typography variant="body2" color="text.secondary">
                {job.current_step}: {job.message}
              </Typography>
            </>
          )}
          
          {job.status === 'completed' && job.result && (
            <Box>
              <Typography variant="body2" color="success.main">
                {job.message}
              </Typography>
              {job.result.rows && (
                <Typography variant="caption" color="text.secondary">
                  {job.result.rows.toLocaleString()} rows processed
                </Typography>
              )}
            </Box>
          )}
          
          {job.status === 'failed' && (
            <Alert severity="error" sx={{ mt: 1 }}>
              {job.error || 'Job failed'}
            </Alert>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <Box>
      {/* WebSocket Connection Status */}
      <Box display="flex" alignItems="center" gap={1} mb={2}>
        <Chip
          icon={isConnected ? <Wifi /> : <WifiOff />}
          label={isConnected ? "Real-time Updates Connected" : "Real-time Updates Disconnected"}
          color={isConnected ? "success" : "error"}
          size="small"
          onClick={() => setShowConnectionStatus(true)}
        />
      </Box>

      <Stepper activeStep={activeStep} orientation="vertical">
        <Step>
          <StepLabel
            optional={fetchedData && (
              <Typography variant="caption">
                {fetchedData.rows.toLocaleString()} rows fetched
              </Typography>
            )}
          >
            Fetch Data
          </StepLabel>
          <StepContent>
            <Box sx={{ mb: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Symbol"
                    value={fetchForm.symbol}
                    onChange={(e) => setFetchForm({ ...fetchForm, symbol: e.target.value })}
                    placeholder="BTCUSDT"
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>Interval</InputLabel>
                    <Select
                      value={fetchForm.interval}
                      label="Interval"
                      onChange={(e) => setFetchForm({ ...fetchForm, interval: e.target.value })}
                    >
                      {INTERVALS.map((interval) => (
                        <MenuItem key={interval.value} value={interval.value}>
                          {interval.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Days Back"
                    value={fetchForm.days_back}
                    onChange={(e) => setFetchForm({ 
                      ...fetchForm, 
                      days_back: parseInt(e.target.value) || 30 
                    })}
                    InputProps={{
                      inputProps: { min: 1, max: 365 }
                    }}
                  />
                </Grid>
              </Grid>
              
              {renderJobProgress(dataFetchJob, 'Data Fetching Progress')}
              
              <Box sx={{ mt: 2 }}>
                <Button
                  variant="contained"
                  onClick={handleFetchData}
                  disabled={dataFetchJob?.status === 'running'}
                  startIcon={<CloudDownload />}
                >
                  Fetch Data
                </Button>
                {fetchedData && (
                  <Button
                    sx={{ ml: 1 }}
                    onClick={() => setActiveStep(1)}
                  >
                    Next
                  </Button>
                )}
              </Box>
            </Box>
          </StepContent>
        </Step>

        <Step>
          <StepLabel
            optional={processedData && (
              <Typography variant="caption">
                {processedData.features.length} features selected
              </Typography>
            )}
          >
            Preprocess Data
          </StepLabel>
          <StepContent>
            <Box sx={{ mb: 2 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Scaling Method</InputLabel>
                    <Select
                      value={preprocessForm.scaling_method}
                      label="Scaling Method"
                      onChange={(e) => setPreprocessForm({ 
                        ...preprocessForm, 
                        scaling_method: e.target.value as any 
                      })}
                    >
                      <MenuItem value="standard">Standard Scaler</MenuItem>
                      <MenuItem value="minmax">MinMax Scaler</MenuItem>
                      <MenuItem value="robust">Robust Scaler</MenuItem>
                    </Select>
                  </FormControl>
                  
                  <FormControl fullWidth>
                    <InputLabel>Handle Missing Values</InputLabel>
                    <Select
                      value={preprocessForm.handle_missing}
                      label="Handle Missing Values"
                      onChange={(e) => setPreprocessForm({ 
                        ...preprocessForm, 
                        handle_missing: e.target.value as any 
                      })}
                    >
                      <MenuItem value="forward_fill">Forward Fill</MenuItem>
                      <MenuItem value="interpolate">Interpolate</MenuItem>
                      <MenuItem value="drop">Drop</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Select Features ({preprocessForm.features.length} selected)
                  </Typography>
                  
                  <Paper variant="outlined" sx={{ maxHeight: 300, overflow: 'auto' }}>
                    <List dense>
                      {Object.entries({
                        price: 'Price Data',
                        technical: 'Technical Indicators',
                        volume: 'Volume Indicators',
                        volatility: 'Volatility',
                        trend: 'Trend',
                        momentum: 'Momentum',
                      }).map(([category, label]) => (
                        <React.Fragment key={category}>
                          <ListItem
                            button
                            onClick={() => toggleSection(category)}
                          >
                            <ListItemIcon>
                              {category === 'price' && <Timeline />}
                              {category === 'technical' && <Functions />}
                              {category === 'volume' && <Storage />}
                              {category === 'volatility' && <Speed />}
                              {category === 'trend' && <Timeline />}
                              {category === 'momentum' && <Speed />}
                            </ListItemIcon>
                            <ListItemText primary={label} />
                            {expandedSections[category] ? <ExpandLess /> : <ExpandMore />}
                          </ListItem>
                          
                          <Collapse in={expandedSections[category]} timeout="auto" unmountOnExit>
                            <List component="div" disablePadding>
                              {getFeaturesByCategory(category).map((feature) => (
                                <ListItem key={feature.value} sx={{ pl: 4 }}>
                                  <FormControlLabel
                                    control={
                                      <Checkbox
                                        checked={preprocessForm.features.includes(feature.value)}
                                        onChange={() => toggleFeature(feature.value)}
                                        size="small"
                                      />
                                    }
                                    label={feature.label}
                                  />
                                </ListItem>
                              ))}
                            </List>
                          </Collapse>
                          <Divider />
                        </React.Fragment>
                      ))}
                    </List>
                  </Paper>
                </Grid>
              </Grid>
              
              {renderJobProgress(preprocessJob, 'Preprocessing Progress')}
              
              <Box sx={{ mt: 2 }}>
                <Button
                  onClick={() => setActiveStep(0)}
                >
                  Back
                </Button>
                <Button
                  variant="contained"
                  onClick={handlePreprocessData}
                  disabled={!fetchedData || preprocessJob?.status === 'running'}
                  startIcon={<Transform />}
                  sx={{ ml: 1 }}
                >
                  Preprocess Data
                </Button>
                {processedData && (
                  <Button
                    sx={{ ml: 1 }}
                    onClick={() => setActiveStep(2)}
                  >
                    Next
                  </Button>
                )}
              </Box>
            </Box>
          </StepContent>
        </Step>

        <Step>
          <StepLabel>Select & Train Model</StepLabel>
          <StepContent>
            <Box sx={{ mb: 2 }}>
              {processedData ? (
                <Alert severity="success" sx={{ mb: 2 }}>
                  Data pipeline completed! Your processed data is ready for model training.
                  <br />
                  <Typography variant="caption">
                    Processed Data ID: {processedData.processed_id}
                  </Typography>
                </Alert>
              ) : (
                <Alert severity="info">
                  Complete the data preprocessing step first.
                </Alert>
              )}
              
              <Box sx={{ mt: 2 }}>
                <Button onClick={() => setActiveStep(1)}>
                  Back
                </Button>
                {processedData && (
                  <Button
                    variant="contained"
                    startIcon={<ModelTraining />}
                    sx={{ ml: 1 }}
                    onClick={() => {
                      if (onComplete) {
                        onComplete(processedData.processed_id);
                      }
                    }}
                  >
                    Continue to Model Selection
                  </Button>
                )}
              </Box>
            </Box>
          </StepContent>
        </Step>
      </Stepper>
      
      {activeStep === 3 && (
        <Paper sx={{ p: 3, mt: 2 }}>
          <Typography variant="h6" gutterBottom>
            Pipeline Complete!
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Your data has been successfully fetched and preprocessed. 
            You can now use it to train machine learning models.
          </Typography>
          <Button
            variant="outlined"
            sx={{ mt: 2 }}
            onClick={() => {
              dispatch(clearPipelineData());
              setActiveStep(0);
            }}
          >
            Start New Pipeline
          </Button>
        </Paper>
      )}
      
      {/* Connection Status Snackbar */}
      <Snackbar
        open={showConnectionStatus}
        autoHideDuration={3000}
        onClose={() => setShowConnectionStatus(false)}
        message={
          isConnected 
            ? "WebSocket connected. You'll receive real-time updates for your jobs."
            : connectionError || "WebSocket disconnected. Updates may be delayed."
        }
      />
    </Box>
  );
};