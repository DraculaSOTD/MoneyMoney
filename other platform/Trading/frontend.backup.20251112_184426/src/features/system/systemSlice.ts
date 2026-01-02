import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

export enum SystemStatus {
  RUNNING = 'RUNNING',
  STOPPED = 'STOPPED',
  ERROR = 'ERROR',
  INITIALIZING = 'INITIALIZING',
}

interface SystemState {
  status: SystemStatus;
  message: string;
  startTime?: string;
  metrics?: {
    totalPositions: number;
    openPositions: number;
    totalPnL: number;
    winRate: number;
  };
}

const initialState: SystemState = {
  status: SystemStatus.STOPPED,
  message: '',
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setSystemStatus: (state, action: PayloadAction<{ status: SystemStatus; message?: string }>) => {
      state.status = action.payload.status;
      if (action.payload.message) {
        state.message = action.payload.message;
      }
    },
    setSystemMetrics: (state, action: PayloadAction<SystemState['metrics']>) => {
      state.metrics = action.payload;
    },
    setStartTime: (state, action: PayloadAction<string>) => {
      state.startTime = action.payload;
    },
  },
});

export const { setSystemStatus, setSystemMetrics, setStartTime } = systemSlice.actions;
export default systemSlice.reducer;