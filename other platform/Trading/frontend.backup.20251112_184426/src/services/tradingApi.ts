import { apiSlice } from './api';
import { Position, Order } from '../features/trading/tradingSlice';

export interface OrderRequest {
  symbol: string;
  side: 'BUY' | 'SELL';
  order_type: 'LIMIT' | 'MARKET' | 'STOP_LOSS';
  quantity: number;
  price?: number;
  strategy?: string;
  exchange: string;
}

export interface ClosePositionRequest {
  position_id: string;
  quantity?: number;
}

export const tradingApi = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    getPositions: builder.query<Position[], string | undefined>({
      query: (status) => ({
        url: '/positions',
        params: status ? { status } : undefined,
      }),
      providesTags: ['Position'],
    }),
    closePosition: builder.mutation<Position, ClosePositionRequest>({
      query: (request) => ({
        url: '/positions/close',
        method: 'POST',
        body: request,
      }),
      invalidatesTags: ['Position'],
    }),
    placeOrder: builder.mutation<Order, OrderRequest>({
      query: (order) => ({
        url: '/trading/order',
        method: 'POST',
        body: order,
      }),
      invalidatesTags: ['Order', 'Position'],
    }),
    cancelOrder: builder.mutation<void, string>({
      query: (orderId) => ({
        url: `/trading/order/${orderId}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['Order'],
    }),
  }),
});

export const {
  useGetPositionsQuery,
  useClosePositionMutation,
  usePlaceOrderMutation,
  useCancelOrderMutation,
} = tradingApi;