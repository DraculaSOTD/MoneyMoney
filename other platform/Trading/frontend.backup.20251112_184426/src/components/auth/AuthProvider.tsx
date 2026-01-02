import React, { useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { websocketService } from '../../services/websocket';

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const dispatch = useAppDispatch();
  const { isAuthenticated } = useAppSelector((state) => state.auth);

  useEffect(() => {
    // WebSocket disabled temporarily until backend WebSocket endpoint is implemented
    // if (isAuthenticated) {
    //   websocketService.connect();
    // } else {
    //   websocketService.disconnect();
    // }

    // return () => {
    //   websocketService.disconnect();
    // };
  }, [isAuthenticated]);

  return <>{children}</>;
};