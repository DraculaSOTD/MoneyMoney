import axiosInstance from './axiosConfig';

export interface AdminLoginRequest {
  username: string;
  password: string;
}

export interface AdminLoginResponse {
  access_token: string;
  token_type: string;
  username: string;
  email: string;
  is_superuser: boolean;
  expires_at: string;
}

export interface AdminInfo {
  id: number;
  username: string;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  last_login: string | null;
  created_at: string;
}

/**
 * Admin Authentication API Service
 * Handles all admin authentication operations
 */
class AdminAuthService {
  private readonly BASE_PATH = '/admin/auth';

  /**
   * Login with username and password
   */
  async login(username: string, password: string): Promise<AdminLoginResponse> {
    try {
      const response = await axiosInstance.post<AdminLoginResponse>(
        `${this.BASE_PATH}/login`,
        { username, password }
      );
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 401) {
        throw new Error('Invalid username or password');
      } else if (error.response?.status === 403) {
        throw new Error(error.response.data.detail || 'Account is locked or inactive');
      } else {
        throw new Error(error.response?.data?.detail || 'Login failed. Please try again.');
      }
    }
  }

  /**
   * Verify the current token is still valid
   */
  async verifyToken(): Promise<AdminInfo> {
    try {
      const response = await axiosInstance.get<AdminInfo>(`${this.BASE_PATH}/verify`);
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 401 || error.response?.status === 403) {
        throw new Error('Session expired or invalid');
      }
      throw new Error('Failed to verify session');
    }
  }

  /**
   * Logout and invalidate session
   */
  async logout(): Promise<void> {
    try {
      await axiosInstance.post(`${this.BASE_PATH}/logout`);
    } catch (error) {
      // Even if logout fails on backend, we'll clear local state
      console.warn('Logout API call failed, but clearing local session');
    }
  }

  /**
   * Get current admin info (requires valid token)
   */
  async getCurrentAdmin(): Promise<AdminInfo> {
    return this.verifyToken();
  }

  /**
   * List all admins (superuser only)
   */
  async listAdmins(): Promise<AdminInfo[]> {
    try {
      const response = await axiosInstance.get<AdminInfo[]>(`${this.BASE_PATH}/list`);
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 403) {
        throw new Error('Superuser access required');
      }
      throw new Error('Failed to fetch admin list');
    }
  }
}

export const adminAuthService = new AdminAuthService();
