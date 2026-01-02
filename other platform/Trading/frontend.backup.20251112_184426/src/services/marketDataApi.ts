import { apiSlice } from './api';
import { MarketData, Kline } from '../features/marketData/marketDataSlice';

export interface DataSubscription {
  symbols: string[];
  data_types: ('TICKER' | 'KLINE' | 'ORDERBOOK' | 'TRADES')[];
  interval?: string;
}

export interface TickerData {
  symbol: string;
  price: number;
  bid_price: number;
  ask_price: number;
  volume_24h: number;
  high_24h: number;
  low_24h: number;
  change_24h: number;
  change_percent_24h: number;
}

export const marketDataApi = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    subscribeToData: builder.mutation<{ message: string }, DataSubscription>({
      query: (subscription) => ({
        url: '/data/subscribe',
        method: 'POST',
        body: subscription,
      }),
    }),
    getSnapshot: builder.query<Record<string, MarketData>, void>({
      query: () => '/data/snapshot',
    }),
    getTicker: builder.query<TickerData, string>({
      query: (symbol) => `/data/${symbol}/ticker`,
    }),
    getKlines: builder.query<Kline[], { symbol: string; interval: string; limit?: number }>({
      query: ({ symbol, interval, limit }) => ({
        url: `/data/${symbol}/klines`,
        params: { interval, limit },
      }),
    }),
  }),
});

export const {
  useSubscribeToDataMutation,
  useGetSnapshotQuery,
  useGetTickerQuery,
  useGetKlinesQuery,
} = marketDataApi;