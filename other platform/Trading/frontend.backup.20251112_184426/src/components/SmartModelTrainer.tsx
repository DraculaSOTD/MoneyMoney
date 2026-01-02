import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Alert,
  Chip,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Checkbox,
  Divider,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  Psychology,
  CheckCircle,
  Error as ErrorIcon,
  Pending,
  Speed,
  TrendingUp,
  Functions,
  BubbleChart,
  Timeline,
} from '@mui/icons-material';
import axios from 'axios';

interface AvailableSymbol {
  symbol: string;
  name: string;
  has_data: boolean;
  models_trained: boolean;
  data_updated_at?: string;
  total_data_points: number;
}

interface ModelInfo {
  id: string;
  name: string;
  displayName: string;
  category: 'statistical' | 'deep_learning' | 'ensemble';
  description: string;
  icon: React.ReactNode;
  recommended: boolean;
}

interface TrainingJob {
  job_id: string;
  symbol: string;
  model_name: string;
  status: string;
  progress: number;
  accuracy?: number;
  started_at: string;
  completed_at?: string;
}

const AVAILABLE_MODELS: ModelInfo[] = [
  {
    id: 'ARIMA',
    name: 'ARIMA',
    displayName: 'ARIMA (Time Series)',
    category: 'statistical',
    description: 'AutoRegressive Integrated Moving Average - Classical time series forecasting',
    icon: <Timeline />,
    recommended: true,
  },
  {
    id: 'GARCH',
    name: 'GARCH',
    displayName: 'GARCH (Volatility)',
    category: 'statistical',
    description: 'Generalized Autoregressive Conditional Heteroskedasticity - Volatility modeling',
    icon: <BubbleChart />,
    recommended: true,
  },
  {
    id: 'GRU_Attention',
    name: 'GRU_Attention',
    displayName: 'GRU with Attention',
    category: 'deep_learning',
    description: 'Gated Recurrent Unit with multi-head attention mechanism',
    icon: <Psychology />,
    recommended: true,
  },
  {
    id: 'CNN_Pattern',
    name: 'CNN_Pattern',
    displayName: 'CNN Pattern Recognition',
    category: 'deep_learning',
    description: 'Convolutional Neural Network for pattern recognition in price charts',
    icon: <Functions />,
    recommended: true,
  },
  {
    id: 'Ensemble',
    name: 'Ensemble',
    displayName: 'Ensemble (All Models)',
    category: 'ensemble',
    description: 'Meta-learning ensemble combining all model predictions',
    icon: <Speed />,
    recommended: false,
  },
];

export const SmartModelTrainer: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [availableSymbols, setAvailableSymbols] = useState<AvailableSymbol[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [autoSelect, setAutoSelect] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);

  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001';

  useEffect(() => {
    fetchAvailableSymbols();
  }, []);

  useEffect(() => {
    if (selectedSymbol) {
      fetchTrainingStatus(selectedSymbol);
    }
  }, [selectedSymbol]);

  // Auto-select recommended models when toggled on
  useEffect(() => {
    if (autoSelect) {
      setSelectedModels(AVAILABLE_MODELS.filter(m => m.recommended).map(m => m.id));
    }
  }, [autoSelect]);

  const fetchAvailableSymbols = async () => {
    try {
      const token = localStorage.getItem('adminToken');
      const response = await axios.get(`${apiUrl}/admin/data/available`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setAvailableSymbols(response.data.filter((s: AvailableSymbol) => s.has_data));
    } catch (err) {
      console.error('Error fetching symbols:', err);
    }
  };

  const fetchTrainingStatus = async (symbol: string) => {
    try {
      const token = localStorage.getItem('adminToken');
      const response = await axios.get(`${apiUrl}/admin/models/status/${symbol}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setTrainingJobs(response.data);
    } catch (err) {
      console.error('Error fetching training status:', err);
    }
  };

  const handleModelToggle = (modelId: string) => {
    setSelectedModels(prev =>
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
    setAutoSelect(false); // Disable auto-select when manually choosing
  };

  const handleStartTraining = async () => {
    if (!selectedSymbol) {
      setError('Please select a symbol');
      return;
    }

    if (selectedModels.length === 0 && !autoSelect) {
      setError('Please select at least one model');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const token = localStorage.getItem('adminToken');
      const response = await axios.post(
        `${apiUrl}/admin/models/train/${selectedSymbol}`,
        {
          symbol: selectedSymbol,
          models: autoSelect ? null : selectedModels, // null triggers auto-selection
        },
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );

      setSuccess(
        `Training started for ${response.data.models.length} model(s): ${response.data.models.join(', ')}`
      );

      // Refresh training status
      setTimeout(() => fetchTrainingStatus(selectedSymbol), 1000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start training');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'training':
      case 'running':
        return 'primary';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle />;
      case 'failed':
        return <ErrorIcon />;
      case 'training':
      case 'running':
        return <CircularProgress size={20} />;
      default:
        return <Pending />;
    }
  };

  const recommendedModels = AVAILABLE_MODELS.filter(m => m.recommended);
  const optionalModels = AVAILABLE_MODELS.filter(m => !m.recommended);

  return (
    <Box>
      <Typography variant="h4" gutterBottom className="gradient-text">
        Smart Model Training
      </Typography>
      <Typography variant="body1" color="textSecondary" paragraph>
        Automatically select and train the best models for your data
      </Typography>

      {/* Symbol Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Step 1: Select Symbol for Training
          </Typography>

          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Symbol with Available Data</InputLabel>
            <Select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              label="Symbol with Available Data"
            >
              <MenuItem value="">
                <em>Select a symbol</em>
              </MenuItem>
              {availableSymbols.map((symbol) => (
                <MenuItem key={symbol.symbol} value={symbol.symbol}>
                  {symbol.symbol} - {symbol.total_data_points.toLocaleString()} data points
                  {symbol.models_trained && ' (Already Trained)'}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {availableSymbols.length === 0 && (
            <Alert severity="info">
              No symbols with data available. Please collect data first from the Data Collection page.
            </Alert>
          )}

          {selectedSymbol && (
            <Alert severity="success" sx={{ mt: 2 }}>
              Selected: <strong>{selectedSymbol}</strong>
              <br />
              Data points: {availableSymbols.find(s => s.symbol === selectedSymbol)?.total_data_points.toLocaleString()}
              <br />
              Interval: <strong>1 minute (enforced)</strong>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Model Selection */}
      {selectedSymbol && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Step 2: Select Models to Train
              </Typography>
              <Button
                variant={autoSelect ? 'contained' : 'outlined'}
                size="small"
                onClick={() => setAutoSelect(!autoSelect)}
              >
                {autoSelect ? 'Auto-Select (Recommended)' : 'Manual Selection'}
              </Button>
            </Box>

            {autoSelect && (
              <Alert severity="info" sx={{ mb: 2 }}>
                <strong>Smart Selection Active</strong>
                <br />
                The system will automatically train the recommended models based on your data type (OHLCV 1-minute data).
                <br />
                Models: ARIMA, GARCH, GRU with Attention, CNN Pattern Recognition
              </Alert>
            )}

            {/* Recommended Models */}
            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2, color: 'primary.main' }}>
              Recommended Models (for OHLCV Data)
            </Typography>
            <List>
              {recommendedModels.map((model) => (
                <ListItem
                  key={model.id}
                  dense
                  button={!autoSelect as any}
                  onClick={() => !autoSelect && handleModelToggle(model.id)}
                  disabled={autoSelect}
                >
                  <ListItemIcon>
                    <Checkbox
                      edge="start"
                      checked={autoSelect || selectedModels.includes(model.id)}
                      disabled={autoSelect}
                    />
                  </ListItemIcon>
                  <ListItemIcon>{model.icon}</ListItemIcon>
                  <ListItemText
                    primary={model.displayName}
                    secondary={model.description}
                  />
                  <Chip label={model.category} size="small" />
                </ListItem>
              ))}
            </List>

            {/* Optional Models */}
            {!autoSelect && (
              <>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle2" gutterBottom sx={{ color: 'text.secondary' }}>
                  Optional Models
                </Typography>
                <List>
                  {optionalModels.map((model) => (
                    <ListItem
                      key={model.id}
                      dense
                      button
                      onClick={() => handleModelToggle(model.id)}
                    >
                      <ListItemIcon>
                        <Checkbox
                          edge="start"
                          checked={selectedModels.includes(model.id)}
                        />
                      </ListItemIcon>
                      <ListItemIcon>{model.icon}</ListItemIcon>
                      <ListItemText
                        primary={model.displayName}
                        secondary={model.description}
                      />
                      <Chip label={model.category} size="small" />
                    </ListItem>
                  ))}
                </List>
              </>
            )}

            {/* Start Training Button */}
            <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
              <Button
                fullWidth
                variant="contained"
                color="primary"
                size="large"
                onClick={handleStartTraining}
                disabled={loading || (!autoSelect && selectedModels.length === 0)}
                startIcon={loading ? <CircularProgress size={20} /> : <TrendingUp />}
              >
                {loading ? 'Starting Training...' : `Start Training ${autoSelect ? '(4 Models)' : `(${selectedModels.length} Model${selectedModels.length !== 1 ? 's' : ''})`}`}
              </Button>
            </Box>

            {/* Messages */}
            {error && (
              <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError(null)}>
                {error}
              </Alert>
            )}
            {success && (
              <Alert severity="success" sx={{ mt: 2 }} onClose={() => setSuccess(null)}>
                {success}
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      {/* Training Status */}
      {selectedSymbol && trainingJobs.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Step 3: Training Status
            </Typography>

            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Model</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Progress</TableCell>
                    <TableCell>Accuracy</TableCell>
                    <TableCell>Started</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {trainingJobs.map((job) => (
                    <TableRow key={job.job_id}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {AVAILABLE_MODELS.find(m => m.id === job.model_name)?.icon}
                          <Typography variant="body2">
                            {job.model_name}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={job.status}
                          color={getStatusColor(job.status) as any}
                          size="small"
                          icon={getStatusIcon(job.status)}
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ width: 100 }}>
                          <LinearProgress
                            variant="determinate"
                            value={job.progress}
                            color={getStatusColor(job.status) as any}
                          />
                          <Typography variant="caption" color="textSecondary">
                            {job.progress}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        {job.accuracy ? (
                          <Chip
                            label={`${(job.accuracy * 100).toFixed(1)}%`}
                            color={job.accuracy > 0.7 ? 'success' : 'warning'}
                            size="small"
                          />
                        ) : (
                          '-'
                        )}
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" color="textSecondary">
                          {new Date(job.started_at).toLocaleString()}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            {trainingJobs.some(j => j.status === 'completed') && (
              <Alert severity="success" sx={{ mt: 2 }}>
                <strong>Training completed for some models!</strong>
                <br />
                The symbol is now ready to appear in the MoneyMoney frontend for users.
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  );
};
