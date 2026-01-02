import React, { useEffect } from 'react';
import {
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Typography,
  Chip,
  CircularProgress,
  Avatar,
  ListItemText,
  ListItemAvatar,
  Skeleton,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  CurrencyBitcoin,
  ShowChart,
  CurrencyExchange,
  Inventory,
} from '@mui/icons-material';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store';
import { selectProfile } from '../store/slices/profileSlice';
import { useGetProfilesQuery } from '../store/slices/profileSlice';

interface ProfileSelectorProps {
  onProfileChange?: (profileId: number | null) => void;
  showMetrics?: boolean;
  fullWidth?: boolean;
}

const ProfileSelector: React.FC<ProfileSelectorProps> = ({
  onProfileChange,
  showMetrics = true,
  fullWidth = false,
}) => {
  const dispatch = useDispatch();
  const selectedProfileId = useSelector((state: RootState) => state.profile.selectedProfileId);
  const { data: profiles, isLoading, isError } = useGetProfilesQuery({ is_active: true });

  useEffect(() => {
    if (profiles && profiles.length > 0 && !selectedProfileId) {
      const defaultProfile = profiles[0];
      dispatch(selectProfile(defaultProfile.id));
      onProfileChange?.(defaultProfile.id);
    }
  }, [profiles, selectedProfileId, dispatch, onProfileChange]);

  const handleChange = (event: any) => {
    const profileId = event.target.value as number | '';
    const id = profileId === '' ? null : profileId;
    dispatch(selectProfile(id));
    onProfileChange?.(id);
  };

  const getProfileIcon = (profileType: string) => {
    switch (profileType) {
      case 'crypto':
        return <CurrencyBitcoin />;
      case 'stock':
        return <ShowChart />;
      case 'forex':
        return <CurrencyExchange />;
      case 'commodity':
        return <Inventory />;
      default:
        return <ShowChart />;
    }
  };

  const getPriceChangeIcon = (change: number | undefined) => {
    if (!change) return <TrendingFlat color="action" />;
    if (change > 0) return <TrendingUp color="success" />;
    if (change < 0) return <TrendingDown color="error" />;
    return <TrendingFlat color="action" />;
  };

  const formatPriceChange = (change: number | undefined) => {
    if (!change) return '0.00%';
    const formatted = change.toFixed(2);
    return `${change > 0 ? '+' : ''}${formatted}%`;
  };

  const formatPrice = (price: number | undefined) => {
    if (!price) return 'N/A';
    if (price < 0.01) return price.toExponential(4);
    if (price < 1) return price.toFixed(6);
    if (price < 100) return price.toFixed(4);
    return price.toFixed(2);
  };

  const selectedProfile = profiles?.find(p => p.id === selectedProfileId);

  if (isLoading) {
    return (
      <Box sx={{ minWidth: fullWidth ? '100%' : 240 }}>
        <Skeleton variant="rectangular" height={56} />
      </Box>
    );
  }

  if (isError || !profiles) {
    return (
      <Box sx={{ minWidth: fullWidth ? '100%' : 240 }}>
        <Typography color="error">Failed to load profiles</Typography>
      </Box>
    );
  }

  return (
    <FormControl fullWidth={fullWidth} sx={{ minWidth: fullWidth ? '100%' : 240 }}>
      <InputLabel id="profile-selector-label">Trading Profile</InputLabel>
      <Select
        labelId="profile-selector-label"
        id="profile-selector"
        value={selectedProfileId || ''}
        label="Trading Profile"
        onChange={handleChange}
        renderValue={(value) => {
          const profile = profiles.find(p => p.id === value);
          if (!profile) return 'Select Profile';
          
          return (
            <Box display="flex" alignItems="center" gap={1}>
              <Avatar sx={{ width: 24, height: 24, bgcolor: 'primary.main' }}>
                {getProfileIcon(profile.profile_type)}
              </Avatar>
              <Typography>{profile.name}</Typography>
              {showMetrics && profile.current_price && (
                <>
                  <Typography variant="body2" color="text.secondary">
                    ${formatPrice(profile.current_price)}
                  </Typography>
                  <Chip
                    size="small"
                    icon={getPriceChangeIcon(profile.price_change_24h)}
                    label={formatPriceChange(profile.price_change_24h)}
                    color={
                      profile.price_change_24h && profile.price_change_24h > 0
                        ? 'success'
                        : profile.price_change_24h && profile.price_change_24h < 0
                        ? 'error'
                        : 'default'
                    }
                    variant="outlined"
                  />
                </>
              )}
            </Box>
          );
        }}
      >
        <MenuItem value="">
          <ListItemText primary="None" secondary="No profile selected" />
        </MenuItem>
        {profiles.map((profile) => (
          <MenuItem key={profile.id} value={profile.id}>
            <ListItemAvatar>
              <Avatar sx={{ bgcolor: 'primary.main' }}>
                {getProfileIcon(profile.profile_type)}
              </Avatar>
            </ListItemAvatar>
            <ListItemText
              primary={
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography>{profile.name}</Typography>
                  <Chip
                    size="small"
                    label={profile.symbol}
                    variant="outlined"
                  />
                  <Chip
                    size="small"
                    label={profile.exchange}
                    color="primary"
                    variant="outlined"
                  />
                </Box>
              }
              secondary={
                <Box>
                  <Typography variant="caption" color="text.secondary" display="block">
                    {profile.profile_type.toUpperCase()} • {profile.base_currency}/{profile.quote_currency}
                  </Typography>
                  {showMetrics && (
                    <Box display="flex" gap={2} mt={0.5}>
                      <Typography variant="caption">
                        Price: ${formatPrice(profile.current_price)}
                      </Typography>
                      <Typography 
                        variant="caption" 
                        color={
                          profile.price_change_24h && profile.price_change_24h > 0
                            ? 'success.main'
                            : profile.price_change_24h && profile.price_change_24h < 0
                            ? 'error.main'
                            : 'text.secondary'
                        }
                      >
                        24h: {formatPriceChange(profile.price_change_24h)}
                      </Typography>
                      {profile.volume_24h && (
                        <Typography variant="caption">
                          Vol: ${(profile.volume_24h / 1e6).toFixed(2)}M
                        </Typography>
                      )}
                    </Box>
                  )}
                  <Box display="flex" gap={1} mt={0.5}>
                    <Chip
                      size="small"
                      label={`${profile.active_models} active models`}
                      color="info"
                      variant="outlined"
                    />
                    {profile.deployed_models > 0 && (
                      <Chip
                        size="small"
                        label={`${profile.deployed_models} deployed`}
                        color="success"
                        variant="outlined"
                      />
                    )}
                    {profile.total_trades > 0 && (
                      <Chip
                        size="small"
                        label={`${profile.total_trades} trades`}
                        variant="outlined"
                      />
                    )}
                  </Box>
                </Box>
              }
            />
          </MenuItem>
        ))}
      </Select>
      
      {selectedProfile && showMetrics && (
        <Box mt={2} p={2} bgcolor="background.default" borderRadius={1}>
          <Typography variant="subtitle2" gutterBottom>
            Profile Performance
          </Typography>
          <Box display="grid" gridTemplateColumns="1fr 1fr" gap={1}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Win Rate
              </Typography>
              <Typography variant="body2">
                {(selectedProfile.win_rate * 100).toFixed(1)}%
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Total P&L
              </Typography>
              <Typography 
                variant="body2" 
                color={selectedProfile.total_pnl >= 0 ? 'success.main' : 'error.main'}
              >
                ${selectedProfile.total_pnl.toFixed(2)}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Sharpe Ratio
              </Typography>
              <Typography variant="body2">
                {selectedProfile.sharpe_ratio.toFixed(2)}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Max Drawdown
              </Typography>
              <Typography variant="body2" color="error.main">
                {(selectedProfile.max_drawdown * 100).toFixed(1)}%
              </Typography>
            </Box>
          </Box>
        </Box>
      )}
    </FormControl>
  );
};

export default ProfileSelector;