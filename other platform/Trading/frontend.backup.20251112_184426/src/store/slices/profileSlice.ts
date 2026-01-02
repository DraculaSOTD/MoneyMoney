import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export interface TradingProfile {
  id: number;
  symbol: string;
  name: string;
  profile_type: 'crypto' | 'stock' | 'forex' | 'commodity';
  exchange: string;
  description: string | null;
  base_currency: string;
  quote_currency: string;
  created_at: string;
  updated_at: string;
  is_active: boolean;
  
  // Data Configuration
  data_source: string;
  timeframe: string;
  lookback_days: number;
  
  // Trading Configuration
  min_trade_size: number;
  max_trade_size: number;
  max_position_size: number;
  trading_fee: number;
  
  // Risk Management
  max_drawdown_limit: number;
  position_risk_limit: number;
  daily_loss_limit: number;
  
  // Performance Metrics
  total_trades: number;
  win_rate: number;
  avg_profit: number;
  total_pnl: number;
  sharpe_ratio: number;
  max_drawdown: number;
  last_trade_date: string | null;
  
  // Live data
  current_price?: number;
  price_change_24h?: number;
  volume_24h?: number;
  
  // Model counts
  active_models: number;
  deployed_models: number;
}

export interface ProfileModel {
  id: number;
  profile_id: number;
  model_name: string;
  model_type: string;
  model_version: string;
  status: 'untrained' | 'training' | 'trained' | 'failed' | 'deployed';
  parameters: Record<string, any>;
  features: string[];
  preprocessing_config: Record<string, any>;
  last_trained: string | null;
  training_duration: number | null;
  training_samples: number | null;
  validation_accuracy: number | null;
  validation_loss: number | null;
  test_accuracy: number | null;
  test_sharpe: number | null;
  is_primary: boolean;
  is_deployed: boolean;
  deployed_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface TrainingHistory {
  id: number;
  profile_id: number;
  model_id: number;
  run_id: string;
  started_at: string;
  completed_at: string | null;
  duration: number | null;
  status: string;
  parameters: Record<string, any>;
  dataset_config: Record<string, any>;
  hardware_info: Record<string, any>;
  epochs_trained: number;
  final_train_loss: number | null;
  final_val_loss: number | null;
  best_epoch: number | null;
  best_val_loss: number | null;
  train_accuracy: number | null;
  val_accuracy: number | null;
  test_accuracy: number | null;
  backtest_sharpe: number | null;
  backtest_returns: number | null;
  backtest_max_drawdown: number | null;
  backtest_win_rate: number | null;
}

export interface ProfilePrediction {
  id: number;
  profile_id: number;
  model_id: number;
  timestamp: string;
  prediction_horizon: string;
  price_prediction: number | null;
  direction_prediction: string | null;
  confidence: number | null;
  predicted_high: number | null;
  predicted_low: number | null;
  predicted_volatility: number | null;
  signal: string | null;
  signal_strength: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  actual_price: number | null;
  prediction_error: number | null;
}

export interface ProfileMetrics {
  id: number;
  profile_id: number;
  timestamp: string;
  current_price: number;
  price_change_24h: number;
  price_change_7d: number;
  price_change_30d: number;
  volume_24h: number;
  volume_change_24h: number;
  avg_volume_7d: number;
  rsi: number;
  macd: number;
  macd_signal: number;
  bollinger_upper: number;
  bollinger_lower: number;
  sma_20: number;
  sma_50: number;
  ema_12: number;
  ema_26: number;
  market_cap: number | null;
  circulating_supply: number | null;
  sentiment_score: number | null;
  social_volume: number | null;
  news_mentions: number | null;
  custom_metrics: Record<string, any>;
}

interface ProfileState {
  selectedProfileId: number | null;
  profiles: TradingProfile[];
  isLoading: boolean;
  error: string | null;
}

const initialState: ProfileState = {
  selectedProfileId: null,
  profiles: [],
  isLoading: false,
  error: null,
};

// RTK Query API
export const profileApi = createApi({
  reducerPath: 'profileApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api',
    prepareHeaders: (headers) => {
      const token = localStorage.getItem('authToken');
      if (token) {
        headers.set('Authorization', `Bearer ${token}`);
      }
      return headers;
    },
  }),
  tagTypes: ['Profile', 'Model', 'Training', 'Prediction', 'Metrics'],
  endpoints: (builder) => ({
    // Profile endpoints
    getProfiles: builder.query<TradingProfile[], { profile_type?: string; is_active?: boolean }>({
      query: (params) => ({
        url: '/profiles',
        params,
      }),
      providesTags: ['Profile'],
    }),
    
    getProfile: builder.query<TradingProfile, number>({
      query: (id) => `/profiles/${id}`,
      providesTags: (_result, _error, id) => [{ type: 'Profile', id }],
    }),
    
    createProfile: builder.mutation<TradingProfile, Partial<TradingProfile>>({
      query: (profile) => ({
        url: '/profiles',
        method: 'POST',
        body: profile,
      }),
      invalidatesTags: ['Profile'],
    }),
    
    updateProfile: builder.mutation<TradingProfile, { id: number; updates: Partial<TradingProfile> }>({
      query: ({ id, updates }) => ({
        url: `/profiles/${id}`,
        method: 'PUT',
        body: updates,
      }),
      invalidatesTags: (_result, _error, { id }) => [{ type: 'Profile', id }, 'Profile'],
    }),
    
    deleteProfile: builder.mutation<void, number>({
      query: (id) => ({
        url: `/profiles/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['Profile'],
    }),
    
    // Model endpoints
    getProfileModels: builder.query<ProfileModel[], { profileId: number; status?: string; is_deployed?: boolean }>({
      query: ({ profileId, ...params }) => ({
        url: `/profiles/${profileId}/models`,
        params,
      }),
      providesTags: (_result, _error, { profileId }) => [{ type: 'Model', id: profileId }],
    }),
    
    createProfileModel: builder.mutation<ProfileModel, { profileId: number; model: Partial<ProfileModel> }>({
      query: ({ profileId, model }) => ({
        url: `/profiles/${profileId}/models`,
        method: 'POST',
        body: model,
      }),
      invalidatesTags: (_result, _error, { profileId }) => [{ type: 'Model', id: profileId }],
    }),
    
    deployModel: builder.mutation<void, { profileId: number; modelId: number }>({
      query: ({ profileId, modelId }) => ({
        url: `/profiles/${profileId}/models/${modelId}/deploy`,
        method: 'PUT',
      }),
      invalidatesTags: (_result, _error, { profileId }) => [
        { type: 'Model', id: profileId },
        { type: 'Profile', id: profileId }
      ],
    }),
    
    undeployModel: builder.mutation<void, { profileId: number; modelId: number }>({
      query: ({ profileId, modelId }) => ({
        url: `/profiles/${profileId}/models/${modelId}/undeploy`,
        method: 'PUT',
      }),
      invalidatesTags: (_result, _error, { profileId }) => [
        { type: 'Model', id: profileId },
        { type: 'Profile', id: profileId }
      ],
    }),
    
    // Training history
    getTrainingHistory: builder.query<TrainingHistory[], { profileId: number; modelId?: number; limit?: number }>({
      query: ({ profileId, ...params }) => ({
        url: `/profiles/${profileId}/training-history`,
        params,
      }),
      providesTags: (_result, _error, { profileId }) => [{ type: 'Training', id: profileId }],
    }),
    
    // Predictions
    getProfilePredictions: builder.query<ProfilePrediction[], { profileId: number; hours?: number; modelId?: number }>({
      query: ({ profileId, hours = 24, ...params }) => ({
        url: `/profiles/${profileId}/predictions`,
        params: { hours, ...params },
      }),
      providesTags: (_result, _error, { profileId }) => [{ type: 'Prediction', id: profileId }],
    }),
    
    // Metrics
    getLatestMetrics: builder.query<ProfileMetrics, number>({
      query: (profileId) => `/profiles/${profileId}/metrics/latest`,
      providesTags: (_result, _error, profileId) => [{ type: 'Metrics', id: profileId }],
    }),
    
    getMetricsHistory: builder.query<ProfileMetrics[], { profileId: number; hours?: number }>({
      query: ({ profileId, hours = 24 }) => ({
        url: `/profiles/${profileId}/metrics/history`,
        params: { hours },
      }),
      providesTags: (_result, _error, { profileId }) => [{ type: 'Metrics', id: profileId }],
    }),
  }),
});

// Export hooks
export const {
  useGetProfilesQuery,
  useGetProfileQuery,
  useCreateProfileMutation,
  useUpdateProfileMutation,
  useDeleteProfileMutation,
  useGetProfileModelsQuery,
  useCreateProfileModelMutation,
  useDeployModelMutation,
  useUndeployModelMutation,
  useGetTrainingHistoryQuery,
  useGetProfilePredictionsQuery,
  useGetLatestMetricsQuery,
  useGetMetricsHistoryQuery,
} = profileApi;

// Regular Redux slice for UI state
const profileSlice = createSlice({
  name: 'profile',
  initialState,
  reducers: {
    selectProfile: (state, action: PayloadAction<number | null>) => {
      state.selectedProfileId = action.payload;
    },
    setProfiles: (state, action: PayloadAction<TradingProfile[]>) => {
      state.profiles = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    updateProfileInState: (state, action: PayloadAction<TradingProfile>) => {
      const index = state.profiles.findIndex(p => p.id === action.payload.id);
      if (index !== -1) {
        state.profiles[index] = action.payload;
      }
    },
    removeProfileFromState: (state, action: PayloadAction<number>) => {
      state.profiles = state.profiles.filter(p => p.id !== action.payload);
      if (state.selectedProfileId === action.payload) {
        state.selectedProfileId = null;
      }
    },
  },
});

export const { 
  selectProfile, 
  setProfiles, 
  setLoading, 
  setError,
  updateProfileInState,
  removeProfileFromState 
} = profileSlice.actions;

export default profileSlice.reducer;