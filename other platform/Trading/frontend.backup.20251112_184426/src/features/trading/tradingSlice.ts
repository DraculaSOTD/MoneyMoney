import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

export interface Position {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  entry_price: number;
  current_price?: number;
  unrealized_pnl?: number;
  realized_pnl?: number;
  status: 'OPEN' | 'CLOSED' | 'PARTIAL';
  open_time: string;
  close_time?: string;
  strategy?: string;
}

export interface Order {
  order_id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  order_type: 'LIMIT' | 'MARKET' | 'STOP_LOSS';
  quantity: number;
  price?: number;
  status: 'NEW' | 'PARTIALLY_FILLED' | 'FILLED' | 'CANCELED';
  created_at: string;
  strategy?: string;
}

interface TradingState {
  positions: Position[];
  orders: Order[];
  activeSymbols: string[];
}

const initialState: TradingState = {
  positions: [],
  orders: [],
  activeSymbols: [],
};

const tradingSlice = createSlice({
  name: 'trading',
  initialState,
  reducers: {
    setPositions: (state, action: PayloadAction<Position[]>) => {
      state.positions = action.payload;
    },
    updatePosition: (state, action: PayloadAction<Position>) => {
      const index = state.positions.findIndex(p => p.id === action.payload.id);
      if (index !== -1) {
        state.positions[index] = action.payload;
      } else {
        state.positions.push(action.payload);
      }
    },
    setOrders: (state, action: PayloadAction<Order[]>) => {
      state.orders = action.payload;
    },
    updateOrder: (state, action: PayloadAction<Order>) => {
      const index = state.orders.findIndex(o => o.order_id === action.payload.order_id);
      if (index !== -1) {
        state.orders[index] = action.payload;
      } else {
        state.orders.push(action.payload);
      }
    },
    setActiveSymbols: (state, action: PayloadAction<string[]>) => {
      state.activeSymbols = action.payload;
    },
  },
});

export const { setPositions, updatePosition, setOrders, updateOrder, setActiveSymbols } = tradingSlice.actions;
export default tradingSlice.reducer;