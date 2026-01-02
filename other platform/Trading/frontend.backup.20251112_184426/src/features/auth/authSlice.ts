import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

export interface AdminUser {
  username: string;
  email: string;
  is_superuser: boolean;
  expires_at?: string;
  full_name?: string | null;
}

interface AuthState {
  token: string | null;
  isAuthenticated: boolean;
  user: AdminUser | null;
  isLoading: boolean;
}

// Helper to get initial state from localStorage
const getInitialState = (): AuthState => {
  const token = localStorage.getItem('authToken');
  const userStr = localStorage.getItem('authUser');

  let user: AdminUser | null = null;
  if (userStr) {
    try {
      user = JSON.parse(userStr);

      // Check if token is expired
      if (user?.expires_at) {
        const expiresAt = new Date(user.expires_at);
        const now = new Date();

        if (now >= expiresAt) {
          // Token expired, clear everything
          localStorage.removeItem('authToken');
          localStorage.removeItem('authUser');
          return {
            token: null,
            isAuthenticated: false,
            user: null,
            isLoading: false,
          };
        }
      }
    } catch (e) {
      console.error('Failed to parse user from localStorage', e);
      localStorage.removeItem('authUser');
    }
  }

  return {
    token,
    isAuthenticated: !!token && !!user,
    user,
    isLoading: false,
  };
};

const initialState: AuthState = getInitialState();

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    setCredentials: (
      state,
      action: PayloadAction<{ token: string; user: AdminUser }>
    ) => {
      state.token = action.payload.token;
      state.isAuthenticated = true;
      state.user = action.payload.user;
      state.isLoading = false;

      // Persist to localStorage
      localStorage.setItem('authToken', action.payload.token);
      localStorage.setItem('authUser', JSON.stringify(action.payload.user));
    },

    updateUser: (state, action: PayloadAction<Partial<AdminUser>>) => {
      if (state.user) {
        state.user = { ...state.user, ...action.payload };
        localStorage.setItem('authUser', JSON.stringify(state.user));
      }
    },

    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },

    logout: (state) => {
      state.token = null;
      state.isAuthenticated = false;
      state.user = null;
      state.isLoading = false;

      // Clear localStorage
      localStorage.removeItem('authToken');
      localStorage.removeItem('authUser');
    },

    clearError: (state) => {
      state.isLoading = false;
    },
  },
});

export const { setCredentials, updateUser, setLoading, logout, clearError } = authSlice.actions;
export default authSlice.reducer;
