import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

export interface MarketData {
  symbol: string;
  price: number;
  bid: number;
  ask: number;
  volume: number;
  change24h: number;
  changePercent24h: number;
  high24h: number;
  low24h: number;
  timestamp: number;
}

export interface Kline {
  openTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closeTime: number;
}

interface MarketDataState {
  data: Record<string, MarketData>;
  klines: Record<string, Kline[]>;
  subscribedSymbols: string[];
  isConnected: boolean;
}

const initialState: MarketDataState = {
  data: {},
  klines: {},
  subscribedSymbols: [],
  isConnected: false,
};

const marketDataSlice = createSlice({
  name: 'marketData',
  initialState,
  reducers: {
    updateData: (state, action: PayloadAction<MarketData>) => {
      state.data[action.payload.symbol] = action.payload;
    },
    setKlines: (state, action: PayloadAction<{ symbol: string; klines: Kline[] }>) => {
      state.klines[action.payload.symbol] = action.payload.klines;
    },
    subscribeSymbols: (state, action: PayloadAction<string[]>) => {
      state.subscribedSymbols = [...new Set([...state.subscribedSymbols, ...action.payload])];
    },
    unsubscribeSymbols: (state, action: PayloadAction<string[]>) => {
      state.subscribedSymbols = state.subscribedSymbols.filter(s => !action.payload.includes(s));
    },
    setConnectionStatus: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },
  },
});

export const { updateData, setKlines, subscribeSymbols, unsubscribeSymbols, setConnectionStatus } = marketDataSlice.actions;
export default marketDataSlice.reducer;