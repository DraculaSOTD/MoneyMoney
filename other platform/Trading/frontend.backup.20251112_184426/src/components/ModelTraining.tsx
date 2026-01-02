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
  Grid,
  Chip,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Paper,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  CheckCircle,
  Error as ErrorIcon,
  Schedule,
  Memory,
  Speed,
  TrendingUp,
  School,
  Refresh,
  Info,
  Close,
} from '@mui/icons-material';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store';
import {
  updateTrainingJob,
  type ModelCatalog,
  type DataPipelineJob,
} from '../features/mlModels/mlModelsSlice';
import mlPipelineApi from '../services/mlPipelineApi';
import { useTrainingWebSocket } from '../hooks/useTrainingWebSocket';

interface ModelTrainingProps {
  profileId: number;
  modelId: number;
  processedDataId?: string;
  onComplete?: (modelId: string) => void;
}

export const ModelTraining: React.FC<ModelTrainingProps> = ({
  profileId,
  modelId,
  processedDataId,
  onComplete,
}) => {
  const dispatch = useDispatch();
  const { modelCatalog, processedData } = useSelector((state: RootState) => state.mlModels);
  const [trainingJob, setTrainingJob] = useState<DataPipelineJob | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);

  // Training configuration
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
    validation_split: 0.2,
  });

  // Use WebSocket for real-time training updates
  const { isConnected, subscribeToJob } = useTrainingWebSocket({
    onJobUpdate: (jobId, update) => {
      if (trainingJob && jobId === trainingJob.job_id) {
        setTrainingJob(update);
        
        // Add log messages
        if (update.message && !trainingLogs.includes(update.message)) {
          setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${update.message}`]);
        }
        
        // Handle completion
        if (update.status === 'completed' && update.result?.model_id && onComplete) {
          onComplete(update.result.model_id);
        }
      }
    },
  });

  const selectedModel = modelCatalog.find(m => m.model_type === 'lstm'); // TODO: Get actual model type

  const handleStartTraining = async () => {
    if (!processedDataId && !processedData) {
      alert('No processed data available. Please complete the data pipeline first.');
      return;
    }

    try {
      const dataId = processedDataId || processedData?.processed_id;
      if (!dataId) return;

      const job = await mlPipelineApi.startTraining({
        profile_id: profileId,
        model_id: modelId,
        data_id: dataId,
        ...trainingConfig,
      });

      setTrainingJob(job);
      setTrainingLogs([`[${new Date().toLocaleTimeString()}] Training started`]);
      
      // Subscribe to real-time updates
      subscribeToJob(job.job_id);
    } catch (error) {
      console.error('Failed to start training:', error);
      setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Failed to start training: ${error}`]);
    }
  };

  const handleStopTraining = async () => {
    if (!trainingJob) return;

    try {
      await mlPipelineApi.cancelJob(trainingJob.job_id);
      setTrainingLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Training cancelled`]);
    } catch (error) {
      console.error('Failed to stop training:', error);
    }
  };

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'running': return <Schedule color="warning" />;
      case 'completed': return <CheckCircle color="success" />;
      case 'failed': return <ErrorIcon color="error" />;
      default: return <Info />;
    }
  };

  const getProgressVariant = (status?: string) => {
    return status === 'running' ? 'indeterminate' : 'determinate';
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Training Configuration */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Configuration
              </Typography>
              
              <Stack spacing={2}>
                <TextField
                  fullWidth
                  type="number"
                  label="Epochs"
                  value={trainingConfig.epochs}
                  onChange={(e) => setTrainingConfig({
                    ...trainingConfig,
                    epochs: parseInt(e.target.value) || 1
                  })}
                  InputProps={{
                    inputProps: { min: 1, max: 1000 }
                  }}
                />
                
                <TextField
                  fullWidth
                  type="number"
                  label="Batch Size"
                  value={trainingConfig.batch_size}
                  onChange={(e) => setTrainingConfig({
                    ...trainingConfig,
                    batch_size: parseInt(e.target.value) || 1
                  })}
                  InputProps={{
                    inputProps: { min: 1, max: 512 }
                  }}
                />
                
                <TextField
                  fullWidth
                  type="number"
                  label="Learning Rate"
                  value={trainingConfig.learning_rate}
                  onChange={(e) => setTrainingConfig({
                    ...trainingConfig,
                    learning_rate: parseFloat(e.target.value) || 0.0001
                  })}
                  InputProps={{
                    inputProps: { min: 0.0001, max: 0.1, step: 0.0001 }
                  }}
                />
                
                <TextField
                  fullWidth
                  type="number"
                  label="Validation Split"
                  value={trainingConfig.validation_split}
                  onChange={(e) => setTrainingConfig({
                    ...trainingConfig,
                    validation_split: parseFloat(e.target.value) || 0.1
                  })}
                  InputProps={{
                    inputProps: { min: 0.1, max: 0.5, step: 0.05 }
                  }}
                />
                
                <Button
                  variant="contained"
                  fullWidth
                  startIcon={trainingJob?.status === 'running' ? <Stop /> : <PlayArrow />}
                  onClick={trainingJob?.status === 'running' ? handleStopTraining : handleStartTraining}
                  disabled={!isConnected || trainingJob?.status === 'completed'}
                  color={trainingJob?.status === 'running' ? 'error' : 'primary'}
                >
                  {trainingJob?.status === 'running' ? 'Stop Training' : 'Start Training'}
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Training Progress */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                <Typography variant="h6">Training Progress</Typography>
                {trainingJob && (
                  <Chip
                    icon={getStatusIcon(trainingJob.status)}
                    label={trainingJob.status}
                    size="small"
                    color={
                      trainingJob.status === 'completed' ? 'success' :
                      trainingJob.status === 'failed' ? 'error' :
                      trainingJob.status === 'running' ? 'warning' : 'default'
                    }
                  />
                )}
              </Box>

              {trainingJob ? (
                <Stack spacing={2}>
                  <Box>
                    <Box display="flex" justifyContent="space-between" mb={1}>
                      <Typography variant="body2" color="text.secondary">
                        {trainingJob.current_step}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {(trainingJob.progress * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant={trainingJob.status === 'running' ? 'determinate' : 'determinate'}
                      value={trainingJob.progress * 100}
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Box>

                  <Typography variant="body2">
                    {trainingJob.message}
                  </Typography>

                  {trainingJob.status === 'completed' && trainingJob.result && (
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Training Results
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Box display="flex" alignItems="center" gap={1}>
                            <TrendingUp color="primary" />
                            <Box>
                              <Typography variant="caption" color="text.secondary">
                                Accuracy
                              </Typography>
                              <Typography variant="body2">
                                {(trainingJob.result.accuracy * 100).toFixed(2)}%
                              </Typography>
                            </Box>
                          </Box>
                        </Grid>
                        <Grid item xs={6}>
                          <Box display="flex" alignItems="center" gap={1}>
                            <Speed color="success" />
                            <Box>
                              <Typography variant="caption" color="text.secondary">
                                Sharpe Ratio
                              </Typography>
                              <Typography variant="body2">
                                {trainingJob.result.sharpe_ratio?.toFixed(3)}
                              </Typography>
                            </Box>
                          </Box>
                        </Grid>
                        <Grid item xs={6}>
                          <Box display="flex" alignItems="center" gap={1}>
                            <School color="info" />
                            <Box>
                              <Typography variant="caption" color="text.secondary">
                                Final Loss
                              </Typography>
                              <Typography variant="body2">
                                {trainingJob.result.final_loss?.toFixed(4)}
                              </Typography>
                            </Box>
                          </Box>
                        </Grid>
                        <Grid item xs={6}>
                          <Box display="flex" alignItems="center" gap={1}>
                            <Memory color="warning" />
                            <Box>
                              <Typography variant="caption" color="text.secondary">
                                Model ID
                              </Typography>
                              <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                                {trainingJob.result.model_id?.substring(0, 8)}...
                              </Typography>
                            </Box>
                          </Box>
                        </Grid>
                      </Grid>
                    </Paper>
                  )}

                  <Button
                    variant="text"
                    size="small"
                    onClick={() => setShowDetails(true)}
                    disabled={trainingLogs.length === 0}
                  >
                    View Training Logs ({trainingLogs.length})
                  </Button>
                </Stack>
              ) : (
                <Box textAlign="center" py={4}>
                  <Typography variant="body2" color="text.secondary">
                    Configure parameters and start training to see progress
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Training Logs Dialog */}
      <Dialog
        open={showDetails}
        onClose={() => setShowDetails(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Typography variant="h6">Training Logs</Typography>
            <IconButton onClick={() => setShowDetails(false)} size="small">
              <Close />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              bgcolor: 'background.default',
              maxHeight: 400,
              overflow: 'auto',
              fontFamily: 'monospace',
              fontSize: '0.875rem',
            }}
          >
            {trainingLogs.map((log, index) => (
              <Box key={index} mb={0.5}>
                {log}
              </Box>
            ))}
          </Paper>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTrainingLogs([])}>Clear</Button>
          <Button onClick={() => setShowDetails(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};