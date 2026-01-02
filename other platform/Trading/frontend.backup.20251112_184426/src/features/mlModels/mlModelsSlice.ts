import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

export interface MLModel {
  id: string;
  name: string;
  type: 'LSTM' | 'GRU' | 'CNN-LSTM' | 'Transformer' | 'XGBoost' | 'RandomForest' | 'PPO' | 'DRQN';
  version: string;
  status: 'training' | 'ready' | 'deployed' | 'failed';
  metrics?: {
    sharpe_ratio: number;
    accuracy?: number;
    mse?: number;
    win_rate?: number;
    max_drawdown?: number;
  };
  created_at: string;
  updated_at: string;
  config?: Record<string, any>;
}

export interface TrainingJob {
  id: string;
  model_type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  start_time: string;
  end_time?: string;
  logs: string[];
  hyperparameters?: Record<string, any>;
}

export interface DataPipelineJob {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_step: string;
  message: string;
  result?: {
    data_id?: string;
    processed_id?: string;
    rows?: number;
    columns?: string[];
    features?: string[];
  };
  error?: string;
}

export interface ModelCatalog {
  model_type: string;
  display_name: string;
  description: string;
  category: 'deep_learning' | 'statistical' | 'reinforcement' | 'ensemble';
  supported_features: string[];
  default_parameters: Record<string, any>;
}

export interface DataFetchConfig {
  symbol: string;
  interval: string;
  days_back: number;
}

export interface PreprocessConfig {
  scaling_method: 'standard' | 'minmax' | 'robust';
  handle_missing: 'forward_fill' | 'interpolate' | 'drop';
  features: string[];
}

interface MLModelsState {
  models: MLModel[];
  trainingJobs: TrainingJob[];
  selectedModel?: MLModel;
  // Data pipeline state
  dataFetchJob?: DataPipelineJob;
  preprocessJob?: DataPipelineJob;
  modelCatalog: ModelCatalog[];
  fetchedData?: {
    data_id: string;
    rows: number;
    start_date: string;
    end_date: string;
    columns: string[];
  };
  processedData?: {
    processed_id: string;
    rows: number;
    features: string[];
    scaling_method: string;
  };
  pipelineConfig: {
    dataFetch?: DataFetchConfig;
    preprocess?: PreprocessConfig;
  };
}

const initialState: MLModelsState = {
  models: [],
  trainingJobs: [],
  modelCatalog: [],
  pipelineConfig: {},
};

const mlModelsSlice = createSlice({
  name: 'mlModels',
  initialState,
  reducers: {
    setModels: (state, action: PayloadAction<MLModel[]>) => {
      state.models = action.payload;
    },
    updateModel: (state, action: PayloadAction<MLModel>) => {
      const index = state.models.findIndex(m => m.id === action.payload.id);
      if (index !== -1) {
        state.models[index] = action.payload;
      } else {
        state.models.push(action.payload);
      }
    },
    setTrainingJobs: (state, action: PayloadAction<TrainingJob[]>) => {
      state.trainingJobs = action.payload;
    },
    updateTrainingJob: (state, action: PayloadAction<TrainingJob>) => {
      const index = state.trainingJobs.findIndex(j => j.id === action.payload.id);
      if (index !== -1) {
        state.trainingJobs[index] = action.payload;
      } else {
        state.trainingJobs.push(action.payload);
      }
    },
    selectModel: (state, action: PayloadAction<MLModel | undefined>) => {
      state.selectedModel = action.payload;
    },
    // Data pipeline actions
    setDataFetchJob: (state, action: PayloadAction<DataPipelineJob>) => {
      state.dataFetchJob = action.payload;
    },
    setPreprocessJob: (state, action: PayloadAction<DataPipelineJob>) => {
      state.preprocessJob = action.payload;
    },
    setModelCatalog: (state, action: PayloadAction<ModelCatalog[]>) => {
      state.modelCatalog = action.payload;
    },
    setFetchedData: (state, action: PayloadAction<typeof initialState.fetchedData>) => {
      state.fetchedData = action.payload;
    },
    setProcessedData: (state, action: PayloadAction<typeof initialState.processedData>) => {
      state.processedData = action.payload;
    },
    setDataFetchConfig: (state, action: PayloadAction<DataFetchConfig>) => {
      state.pipelineConfig.dataFetch = action.payload;
    },
    setPreprocessConfig: (state, action: PayloadAction<PreprocessConfig>) => {
      state.pipelineConfig.preprocess = action.payload;
    },
    clearPipelineData: (state) => {
      state.dataFetchJob = undefined;
      state.preprocessJob = undefined;
      state.fetchedData = undefined;
      state.processedData = undefined;
    },
  },
});

export const { 
  setModels, 
  updateModel, 
  setTrainingJobs, 
  updateTrainingJob, 
  selectModel,
  setDataFetchJob,
  setPreprocessJob,
  setModelCatalog,
  setFetchedData,
  setProcessedData,
  setDataFetchConfig,
  setPreprocessConfig,
  clearPipelineData
} = mlModelsSlice.actions;

export default mlModelsSlice.reducer;