import axios from 'axios';
import { store } from '../store';
import { logout } from '../features/auth/authSlice';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
axiosInstance.interceptors.request.use(
  (config) => {
    const state = store.getState();
    const token = state.auth.token;

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle different error scenarios
    if (error.response) {
      const { status, data } = error.response;

      switch (status) {
        case 401:
          // Unauthorized - token invalid or expired
          console.warn('Session expired or unauthorized. Logging out...');
          store.dispatch(logout());

          // Redirect to login if not already there
          if (window.location.pathname !== '/login') {
            window.location.href = '/login';
          }
          break;

        case 403:
          // Forbidden - account locked, inactive, or insufficient permissions
          console.error('Access forbidden:', data?.detail || 'Insufficient permissions');
          break;

        case 404:
          // Not found
          console.error('Resource not found:', error.config?.url);
          break;

        case 429:
          // Too many requests
          console.error('Rate limit exceeded. Please try again later.');
          break;

        case 500:
        case 502:
        case 503:
        case 504:
          // Server errors
          console.error('Server error:', data?.detail || 'An error occurred on the server');
          break;

        default:
          console.error('API Error:', data?.detail || error.message);
      }
    } else if (error.request) {
      // Request made but no response received
      console.error('Network error: No response from server');
    } else {
      // Something else happened
      console.error('Error:', error.message);
    }

    return Promise.reject(error);
  }
);

export default axiosInstance;
