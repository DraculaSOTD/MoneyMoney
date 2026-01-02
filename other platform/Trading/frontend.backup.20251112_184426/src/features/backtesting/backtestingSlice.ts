import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

export interface BacktestResult {
  id: string;
  model_id: string;
  symbol: string;
  start_date: string;
  end_date: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  metrics?: {
    total_return: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    max_drawdown: number;
    win_rate: number;
    profit_factor: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    avg_win: number;
    avg_loss: number;
    calmar_ratio: number;
  };
  equity_curve?: Array<{ date: string; value: number }>;
  trades?: Array<{
    entry_time: string;
    exit_time: string;
    symbol: string;
    side: string;
    quantity: number;
    entry_price: number;
    exit_price: number;
    pnl: number;
  }>;
}

export interface BacktestConfig {
  model_id: string;
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission: number;
  slippage: number;
  risk_per_trade?: number;
  max_positions?: number;
}

interface BacktestingState {
  results: BacktestResult[];
  activeBacktest?: BacktestResult;
  config?: BacktestConfig;
}

const initialState: BacktestingState = {
  results: [],
};

const backtestingSlice = createSlice({
  name: 'backtesting',
  initialState,
  reducers: {
    setBacktestResults: (state, action: PayloadAction<BacktestResult[]>) => {
      state.results = action.payload;
    },
    updateBacktestResult: (state, action: PayloadAction<BacktestResult>) => {
      const index = state.results.findIndex(r => r.id === action.payload.id);
      if (index !== -1) {
        state.results[index] = action.payload;
      } else {
        state.results.push(action.payload);
      }
    },
    setActiveBacktest: (state, action: PayloadAction<BacktestResult | undefined>) => {
      state.activeBacktest = action.payload;
    },
    setBacktestConfig: (state, action: PayloadAction<BacktestConfig>) => {
      state.config = action.payload;
    },
  },
});

export const { setBacktestResults, updateBacktestResult, setActiveBacktest, setBacktestConfig } = backtestingSlice.actions;
export default backtestingSlice.reducer;