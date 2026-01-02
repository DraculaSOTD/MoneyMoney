import { apiSlice } from './api';

export interface PerformanceMetrics {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  total_pnl: number;
  avg_win: number;
  avg_loss: number;
  best_trade: number;
  worst_trade: number;
  avg_trade_duration: string;
  current_streak: number;
}

export interface Alert {
  id: string;
  type: 'ERROR' | 'WARNING' | 'INFO';
  message: string;
  timestamp: string;
  details?: Record<string, any>;
  resolved: boolean;
}

export interface PerformanceReport {
  period: string;
  metrics: PerformanceMetrics;
  daily_returns: Array<{ date: string; return: number }>;
  symbol_performance: Record<string, { pnl: number; trades: number; win_rate: number }>;
  strategy_performance: Record<string, { pnl: number; trades: number; win_rate: number }>;
}

export const monitoringApi = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    getPerformance: builder.query<PerformanceMetrics, { period?: string }>({
      query: ({ period }) => ({
        url: '/monitoring/performance',
        params: period ? { period } : undefined,
      }),
    }),
    getAlerts: builder.query<Alert[], { limit?: number; type?: string }>({
      query: ({ limit, type }) => ({
        url: '/monitoring/alerts',
        params: { limit, type },
      }),
    }),
    generateReport: builder.query<PerformanceReport, { start_date?: string; end_date?: string }>({
      query: ({ start_date, end_date }) => ({
        url: '/monitoring/report',
        params: { start_date, end_date },
      }),
    }),
    exportMetrics: builder.query<Record<string, any>, { format?: 'json' | 'csv' }>({
      query: ({ format = 'json' }) => ({
        url: '/monitoring/metrics/export',
        params: { format },
      }),
    }),
  }),
});

export const {
  useGetPerformanceQuery,
  useGetAlertsQuery,
  useGenerateReportQuery,
  useExportMetricsQuery,
} = monitoringApi;