import { configureStore } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query';
import { apiSlice } from '../services/api';
import authReducer from '../features/auth/authSlice';
import systemReducer from '../features/system/systemSlice';
import tradingReducer from '../features/trading/tradingSlice';
import marketDataReducer from '../features/marketData/marketDataSlice';
import mlModelsReducer from '../features/mlModels/mlModelsSlice';
import backtestingReducer from '../features/backtesting/backtestingSlice';
import profileReducer, { profileApi } from './slices/profileSlice';

export const store = configureStore({
  reducer: {
    [apiSlice.reducerPath]: apiSlice.reducer,
    [profileApi.reducerPath]: profileApi.reducer,
    auth: authReducer,
    system: systemReducer,
    trading: tradingReducer,
    marketData: marketDataReducer,
    mlModels: mlModelsReducer,
    backtesting: backtestingReducer,
    profile: profileReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['marketData/updateData'],
      },
    }).concat(apiSlice.middleware, profileApi.middleware),
});

setupListeners(store.dispatch);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;