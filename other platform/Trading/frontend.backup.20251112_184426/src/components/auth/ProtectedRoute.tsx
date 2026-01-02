import React, { useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAppSelector, useAppDispatch } from '../../store/hooks';
import { logout, setCredentials } from '../../features/auth/authSlice';
import { adminAuthService } from '../../services/adminAuthApi';
import { Box, CircularProgress, Typography } from '@mui/material';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { isAuthenticated, token } = useAppSelector((state) => state.auth);
  const location = useLocation();
  const dispatch = useAppDispatch();
  const [isVerifying, setIsVerifying] = useState(true);
  const [isValid, setIsValid] = useState(false);

  useEffect(() => {
    const verifyToken = async () => {
      // If no token, skip verification
      if (!token || !isAuthenticated) {
        setIsVerifying(false);
        setIsValid(false);
        return;
      }

      try {
        // Verify token with backend
        const adminInfo = await adminAuthService.verifyToken();

        // Update user info in case anything changed
        dispatch(
          setCredentials({
            token,
            user: {
              username: adminInfo.username,
              email: adminInfo.email,
              is_superuser: adminInfo.is_superuser,
              full_name: adminInfo.full_name,
            },
          })
        );

        setIsValid(true);
      } catch (error) {
        console.error('Token verification failed:', error);
        // Token is invalid or expired, logout
        dispatch(logout());
        setIsValid(false);
      } finally {
        setIsVerifying(false);
      }
    };

    verifyToken();
  }, [token, isAuthenticated, dispatch]);

  // Show loading spinner while verifying
  if (isVerifying) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          gap: 2,
        }}
      >
        <CircularProgress size={50} />
        <Typography variant="body2" color="text.secondary">
          Verifying session...
        </Typography>
      </Box>
    );
  }

  // If not authenticated or token is invalid, redirect to login
  if (!isAuthenticated || !isValid) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
};
