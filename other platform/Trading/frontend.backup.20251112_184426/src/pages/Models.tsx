import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  LinearProgress,
  Card,
  CardContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Tooltip,
  Tab,
  Tabs,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Collapse,
  CircularProgress,
  Stack,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Delete,
  Refresh,
  Add,
  ModelTraining,
  TrendingUp,
  Schedule,
  CheckCircle,
  Error as ErrorIcon,
  Warning,
  ExpandMore,
  ExpandLess,
  Rocket,
  Code,
  Analytics,
  History,
  CloudUpload,
  CloudDownload,
  DataUsage,
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store';
import { ProfileSelector } from '../components';
import { DataPipeline } from '../components/DataPipeline';
import { setModelCatalog } from '../features/mlModels/mlModelsSlice';
import mlPipelineApi from '../services/mlPipelineApi';
import {
  useGetProfileModelsQuery,
  useGetTrainingHistoryQuery,
  useCreateProfileModelMutation,
  useDeployModelMutation,
  useUndeployModelMutation,
  ProfileModel,
  TrainingHistory,
} from '../store/slices/profileSlice';
import { formatDistanceToNow } from 'date-fns';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ py: 2 }}>{children}</Box>}
    </div>
  );
}

export const Models: React.FC = () => {
  const dispatch = useDispatch();
  const selectedProfileId = useSelector((state: RootState) => state.profile.selectedProfileId);
  const { modelCatalog, processedData } = useSelector((state: RootState) => state.mlModels);
  const [tabValue, setTabValue] = useState(0);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ProfileModel | null>(null);
  const [expandedHistory, setExpandedHistory] = useState<number | null>(null);
  
  // Model creation form - enhanced with catalog
  const [modelForm, setModelForm] = useState({
    model_name: '',
    model_type: 'lstm',
    model_version: '1.0.0',
    parameters: {
      epochs: 100,
      batch_size: 32,
      learning_rate: 0.001,
      hidden_size: 128,
      num_layers: 2,
    },
    features: [
      'close', 'volume', 'rsi', 'macd', 'sma_20', 'sma_50', 
      'bollinger_upper', 'bollinger_lower', 'atr', 'obv'
    ],
    processed_data_id: '',
  });
  
  // API hooks
  const { data: models, isLoading: modelsLoading, refetch: refetchModels } = useGetProfileModelsQuery(
    { profileId: selectedProfileId! },
    { skip: !selectedProfileId }
  );
  
  const { data: trainingHistory, isLoading: historyLoading } = useGetTrainingHistoryQuery(
    { profileId: selectedProfileId!, limit: 50 },
    { skip: !selectedProfileId }
  );
  
  const [createModel] = useCreateProfileModelMutation();
  const [deployModel] = useDeployModelMutation();
  const [undeployModel] = useUndeployModelMutation();
  
  // Load model catalog on mount
  useEffect(() => {
    loadModelCatalog();
  }, []);
  
  // Update form when processed data is available
  useEffect(() => {
    if (processedData) {
      setModelForm(prev => ({
        ...prev,
        processed_data_id: processedData.processed_id,
        features: processedData.features,
      }));
    }
  }, [processedData]);
  
  const loadModelCatalog = async () => {
    try {
      const catalog = await mlPipelineApi.getModelCatalog();
      dispatch(setModelCatalog(catalog));
    } catch (error) {
      console.error('Failed to load model catalog:', error);
    }
  };
  
  const handleCreateModel = async () => {
    if (!selectedProfileId) return;
    
    try {
      await createModel({
        profileId: selectedProfileId,
        model: modelForm,
      }).unwrap();
      
      setCreateDialogOpen(false);
      refetchModels();
    } catch (error) {
      console.error('Failed to create model:', error);
    }
  };
  
  const handleModelTypeChange = (modelType: string) => {
    const catalogModel = modelCatalog.find(m => m.model_type === modelType);
    if (catalogModel) {
      setModelForm({
        ...modelForm,
        model_type: modelType,
        parameters: { ...catalogModel.default_parameters },
      });
    }
  };
  
  const handleDeployModel = async (model: ProfileModel) => {
    try {
      await deployModel({
        profileId: model.profile_id,
        modelId: model.id,
      }).unwrap();
      refetchModels();
    } catch (error) {
      console.error('Failed to deploy model:', error);
    }
  };
  
  const handleUndeployModel = async (model: ProfileModel) => {
    try {
      await undeployModel({
        profileId: model.profile_id,
        modelId: model.id,
      }).unwrap();
      refetchModels();
    } catch (error) {
      console.error('Failed to undeploy model:', error);
    }
  };
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'success';
      case 'trained': return 'info';
      case 'training': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'deployed': return <Rocket fontSize="small" />;
      case 'trained': return <CheckCircle fontSize="small" />;
      case 'training': return <CircularProgress size={16} />;
      case 'failed': return <ErrorIcon fontSize="small" />;
      default: return <Warning fontSize="small" />;
    }
  };
  
  const formatPerformanceMetric = (value: number | null | undefined, suffix = '') => {
    if (value === null || value === undefined) return 'N/A';
    return `${(value * 100).toFixed(2)}${suffix}`;
  };

  if (!selectedProfileId) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          ML Models
        </Typography>
        <Alert severity="info">
          Please select a trading profile to view and manage models
        </Alert>
        <Box mt={2}>
          <ProfileSelector fullWidth showMetrics />
        </Box>
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">ML Models</Typography>
        <Box display="flex" gap={2}>
          <ProfileSelector showMetrics />
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setCreateDialogOpen(true)}
          >
            Create Model
          </Button>
          <IconButton onClick={() => refetchModels()}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      <Paper>
        <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)}>
          <Tab label="Data Pipeline" icon={<DataUsage />} iconPosition="start" />
          <Tab label="Model Registry" icon={<ModelTraining />} iconPosition="start" />
          <Tab label="Training History" icon={<History />} iconPosition="start" />
          <Tab label="Performance Analytics" icon={<Analytics />} iconPosition="start" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          {selectedProfileId ? (
            <DataPipeline 
              profileId={selectedProfileId}
              onComplete={(processedDataId) => {
                setTabValue(1);
                setCreateDialogOpen(true);
              }}
            />
          ) : (
            <Alert severity="info">
              Please select a trading profile to use the data pipeline.
            </Alert>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          {modelsLoading ? (
            <LinearProgress />
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Model Name</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Version</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Performance</TableCell>
                    <TableCell>Last Trained</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {models?.map((model) => (
                    <TableRow key={model.id}>
                      <TableCell>
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography variant="subtitle2">{model.model_name}</Typography>
                          {model.is_primary && (
                            <Chip label="Primary" size="small" color="primary" />
                          )}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip label={model.model_type.toUpperCase()} size="small" />
                      </TableCell>
                      <TableCell>{model.model_version}</TableCell>
                      <TableCell>
                        <Chip
                          icon={getStatusIcon(model.status)}
                          label={model.status}
                          size="small"
                          color={getStatusColor(model.status)}
                        />
                      </TableCell>
                      <TableCell>
                        <Stack spacing={0.5}>
                          <Typography variant="caption">
                            Accuracy: {formatPerformanceMetric(model.test_accuracy, '%')}
                          </Typography>
                          <Typography variant="caption">
                            Sharpe: {model.test_sharpe?.toFixed(2) || 'N/A'}
                          </Typography>
                        </Stack>
                      </TableCell>
                      <TableCell>
                        {model.last_trained ? (
                          <Tooltip title={new Date(model.last_trained).toLocaleString()}>
                            <Typography variant="caption">
                              {formatDistanceToNow(new Date(model.last_trained), { addSuffix: true })}
                            </Typography>
                          </Tooltip>
                        ) : (
                          'Never'
                        )}
                      </TableCell>
                      <TableCell>
                        <Box display="flex" gap={0.5}>
                          {model.status === 'trained' && !model.is_deployed && (
                            <Tooltip title="Deploy Model">
                              <IconButton
                                size="small"
                                color="primary"
                                onClick={() => handleDeployModel(model)}
                              >
                                <CloudUpload />
                              </IconButton>
                            </Tooltip>
                          )}
                          {model.is_deployed && (
                            <Tooltip title="Undeploy Model">
                              <IconButton
                                size="small"
                                color="warning"
                                onClick={() => handleUndeployModel(model)}
                              >
                                <CloudDownload />
                              </IconButton>
                            </Tooltip>
                          )}
                          <Tooltip title="Train Model">
                            <IconButton
                              size="small"
                              color="success"
                              disabled={model.status === 'training'}
                            >
                              <PlayArrow />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="View Details">
                            <IconButton
                              size="small"
                              onClick={() => setSelectedModel(model)}
                            >
                              <Code />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete Model">
                            <IconButton
                              size="small"
                              color="error"
                              disabled={model.is_deployed}
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
              {models?.length === 0 && (
                <Box py={4}>
                  <Typography align="center" color="text.secondary">
                    No models created yet. Click "Create Model" to get started.
                  </Typography>
                </Box>
              )}
            </TableContainer>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          {historyLoading ? (
            <LinearProgress />
          ) : (
            <List>
              {trainingHistory?.map((history) => (
                <React.Fragment key={history.id}>
                  <ListItem
                    button
                    onClick={() => setExpandedHistory(
                      expandedHistory === history.id ? null : history.id
                    )}
                  >
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography variant="subtitle1">
                            Run {history.run_id.slice(0, 8)}
                          </Typography>
                          <Chip
                            label={history.status}
                            size="small"
                            color={
                              history.status === 'completed' ? 'success' :
                              history.status === 'failed' ? 'error' :
                              history.status === 'running' ? 'warning' : 'default'
                            }
                          />
                        </Box>
                      }
                      secondary={
                        <Box display="flex" gap={2}>
                          <Typography variant="caption">
                            Started: {new Date(history.started_at).toLocaleString()}
                          </Typography>
                          {history.duration && (
                            <Typography variant="caption">
                              Duration: {(history.duration / 60).toFixed(1)} min
                            </Typography>
                          )}
                          <Typography variant="caption">
                            Epochs: {history.epochs_trained}
                          </Typography>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <IconButton edge="end">
                        {expandedHistory === history.id ? <ExpandLess /> : <ExpandMore />}
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                  <Collapse in={expandedHistory === history.id} timeout="auto" unmountOnExit>
                    <Box px={4} pb={2}>
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={6}>
                          <Card variant="outlined">
                            <CardContent>
                              <Typography variant="subtitle2" gutterBottom>
                                Training Metrics
                              </Typography>
                              <Stack spacing={1}>
                                <Box display="flex" justifyContent="space-between">
                                  <Typography variant="body2">Final Train Loss</Typography>
                                  <Typography variant="body2">
                                    {history.final_train_loss?.toFixed(4) || 'N/A'}
                                  </Typography>
                                </Box>
                                <Box display="flex" justifyContent="space-between">
                                  <Typography variant="body2">Final Val Loss</Typography>
                                  <Typography variant="body2">
                                    {history.final_val_loss?.toFixed(4) || 'N/A'}
                                  </Typography>
                                </Box>
                                <Box display="flex" justifyContent="space-between">
                                  <Typography variant="body2">Best Epoch</Typography>
                                  <Typography variant="body2">
                                    {history.best_epoch || 'N/A'}
                                  </Typography>
                                </Box>
                                <Box display="flex" justifyContent="space-between">
                                  <Typography variant="body2">Test Accuracy</Typography>
                                  <Typography variant="body2">
                                    {formatPerformanceMetric(history.test_accuracy, '%')}
                                  </Typography>
                                </Box>
                              </Stack>
                            </CardContent>
                          </Card>
                        </Grid>
                        <Grid item xs={12} md={6}>
                          <Card variant="outlined">
                            <CardContent>
                              <Typography variant="subtitle2" gutterBottom>
                                Backtest Performance
                              </Typography>
                              <Stack spacing={1}>
                                <Box display="flex" justifyContent="space-between">
                                  <Typography variant="body2">Sharpe Ratio</Typography>
                                  <Typography variant="body2">
                                    {history.backtest_sharpe?.toFixed(2) || 'N/A'}
                                  </Typography>
                                </Box>
                                <Box display="flex" justifyContent="space-between">
                                  <Typography variant="body2">Total Returns</Typography>
                                  <Typography 
                                    variant="body2"
                                    color={history.backtest_returns && history.backtest_returns > 0 ? 'success.main' : 'error.main'}
                                  >
                                    {formatPerformanceMetric(history.backtest_returns, '%')}
                                  </Typography>
                                </Box>
                                <Box display="flex" justifyContent="space-between">
                                  <Typography variant="body2">Max Drawdown</Typography>
                                  <Typography variant="body2" color="error.main">
                                    {formatPerformanceMetric(history.backtest_max_drawdown, '%')}
                                  </Typography>
                                </Box>
                                <Box display="flex" justifyContent="space-between">
                                  <Typography variant="body2">Win Rate</Typography>
                                  <Typography variant="body2">
                                    {formatPerformanceMetric(history.backtest_win_rate, '%')}
                                  </Typography>
                                </Box>
                              </Stack>
                            </CardContent>
                          </Card>
                        </Grid>
                      </Grid>
                    </Box>
                  </Collapse>
                </React.Fragment>
              ))}
            </List>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <TrendingUp color="primary" />
                    <Typography variant="h6">Best Performing Model</Typography>
                  </Box>
                  {models && models.length > 0 ? (
                    <>
                      <Typography variant="subtitle1">
                        {models.reduce((best, model) => 
                          (model.test_sharpe || 0) > (best.test_sharpe || 0) ? model : best
                        ).model_name}
                      </Typography>
                      <Typography variant="h4" color="primary">
                        {models.reduce((best, model) => 
                          (model.test_sharpe || 0) > (best.test_sharpe || 0) ? model : best
                        ).test_sharpe?.toFixed(2) || 'N/A'}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Sharpe Ratio
                      </Typography>
                    </>
                  ) : (
                    <Typography color="text.secondary">No models available</Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <Schedule color="warning" />
                    <Typography variant="h6">Training Time</Typography>
                  </Box>
                  {trainingHistory && trainingHistory.length > 0 ? (
                    <>
                      <Typography variant="h4" color="warning.main">
                        {(trainingHistory.reduce((sum, h) => 
                          sum + (h.duration || 0), 0
                        ) / 3600).toFixed(1)}h
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Total training time
                      </Typography>
                    </>
                  ) : (
                    <Typography color="text.secondary">No training history</Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <CheckCircle color="success" />
                    <Typography variant="h6">Success Rate</Typography>
                  </Box>
                  {trainingHistory && trainingHistory.length > 0 ? (
                    <>
                      <Typography variant="h4" color="success.main">
                        {(
                          (trainingHistory.filter(h => h.status === 'completed').length / 
                           trainingHistory.length) * 100
                        ).toFixed(0)}%
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Training completion rate
                      </Typography>
                    </>
                  ) : (
                    <Typography color="text.secondary">No training history</Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>

      {/* Enhanced Create Model Dialog with Catalog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create New Model</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label="Model Name"
              value={modelForm.model_name}
              onChange={(e) => setModelForm({ ...modelForm, model_name: e.target.value })}
            />
            
            <FormControl fullWidth>
              <InputLabel>Model Type</InputLabel>
              <Select
                value={modelForm.model_type}
                label="Model Type"
                onChange={(e) => handleModelTypeChange(e.target.value)}
              >
                {modelCatalog.map((model) => (
                  <MenuItem key={model.model_type} value={model.model_type}>
                    <Box>
                      <Typography>{model.display_name}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {model.description}
                      </Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <TextField
              fullWidth
              label="Model Version"
              value={modelForm.model_version}
              onChange={(e) => setModelForm({ ...modelForm, model_version: e.target.value })}
            />
            
            {processedData && (
              <Alert severity="info">
                Using preprocessed data with {processedData.features.length} features
              </Alert>
            )}
            
            <Typography variant="subtitle2">Hyperparameters</Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Epochs"
                  value={modelForm.parameters.epochs}
                  onChange={(e) => setModelForm({
                    ...modelForm,
                    parameters: { ...modelForm.parameters, epochs: parseInt(e.target.value) }
                  })}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Batch Size"
                  value={modelForm.parameters.batch_size}
                  onChange={(e) => setModelForm({
                    ...modelForm,
                    parameters: { ...modelForm.parameters, batch_size: parseInt(e.target.value) }
                  })}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Learning Rate"
                  value={modelForm.parameters.learning_rate}
                  onChange={(e) => setModelForm({
                    ...modelForm,
                    parameters: { ...modelForm.parameters, learning_rate: parseFloat(e.target.value) }
                  })}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Hidden Size"
                  value={modelForm.parameters.hidden_size}
                  onChange={(e) => setModelForm({
                    ...modelForm,
                    parameters: { ...modelForm.parameters, hidden_size: parseInt(e.target.value) }
                  })}
                />
              </Grid>
            </Grid>
            
            {!processedData && (
              <Alert severity="warning">
                No preprocessed data available. Please complete the data pipeline first.
              </Alert>
            )}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleCreateModel} 
            variant="contained"
            disabled={!modelForm.model_name || !processedData}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};