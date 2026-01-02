import { apiSlice } from './api';
import { SystemStatus } from '../features/system/systemSlice';

export interface SystemConfig {
  max_position_size: number;
  max_positions: number;
  max_daily_loss: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  risk_per_trade: number;
  trading_pairs: string[];
}

export interface SystemStatusResponse {
  status: SystemStatus;
  message: string;
  uptime?: number;
  active_positions: number;
  total_pnl: number;
  config?: SystemConfig;
}

export const systemApi = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    getSystemStatus: builder.query<SystemStatusResponse, void>({
      query: () => '/system/status',
      providesTags: ['System'],
    }),
    startSystem: builder.mutation<SystemStatusResponse, SystemConfig>({
      query: (config) => ({
        url: '/system/start',
        method: 'POST',
        body: config,
      }),
      invalidatesTags: ['System'],
    }),
    stopSystem: builder.mutation<SystemStatusResponse, void>({
      query: () => ({
        url: '/system/stop',
        method: 'POST',
      }),
      invalidatesTags: ['System'],
    }),
  }),
});

export const { useGetSystemStatusQuery, useStartSystemMutation, useStopSystemMutation } = systemApi;