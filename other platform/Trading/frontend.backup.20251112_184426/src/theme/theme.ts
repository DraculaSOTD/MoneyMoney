import { createTheme } from '@mui/material/styles';

// MoneyMoney Theme Colors - Exact Match
const moneyMoneyColors = {
  background: '#0C0B10',
  primaryAccent: '#E1007A', // Magenta/Pink
  secondaryAccent: '#8A2BE2', // Violet/Purple
  textPrimary: '#FFFFFF',
  textSecondary: '#B0B0B0',
  cardBackground: 'rgba(255, 255, 255, 0.05)',
  cardBorder: 'rgba(255, 255, 255, 0.1)',
  cardGlow: 'rgba(138, 43, 226, 0.3)',
  success: '#22c55e',
  error: '#ef4444',
  warning: '#f59e0b',
};

export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: moneyMoneyColors.primaryAccent,
      light: '#ff6b9d',
      dark: '#c1006a',
    },
    secondary: {
      main: moneyMoneyColors.secondaryAccent,
      light: '#a855f7',
      dark: '#7c3aed',
    },
    background: {
      default: moneyMoneyColors.background,
      paper: moneyMoneyColors.cardBackground,
    },
    success: {
      main: moneyMoneyColors.success,
    },
    error: {
      main: moneyMoneyColors.error,
    },
    warning: {
      main: moneyMoneyColors.warning,
    },
    info: {
      main: moneyMoneyColors.primaryAccent,
    },
    text: {
      primary: moneyMoneyColors.textPrimary,
      secondary: moneyMoneyColors.textSecondary,
    },
  },
  typography: {
    fontFamily: '"Poppins", "Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 500,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: '8px',
          transition: 'all 0.3s ease',
        },
        containedPrimary: {
          background: `linear-gradient(90deg, ${moneyMoneyColors.primaryAccent}, ${moneyMoneyColors.secondaryAccent})`,
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 5px 15px rgba(225, 0, 122, 0.3)',
          },
        },
        outlinedPrimary: {
          backgroundColor: 'transparent',
          border: `2px solid ${moneyMoneyColors.primaryAccent}`,
          color: moneyMoneyColors.primaryAccent,
          '&:hover': {
            backgroundColor: moneyMoneyColors.primaryAccent,
            color: moneyMoneyColors.textPrimary,
            transform: 'translateY(-2px)',
            border: `2px solid ${moneyMoneyColors.primaryAccent}`,
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: moneyMoneyColors.cardBackground,
          border: `1px solid ${moneyMoneyColors.cardBorder}`,
          backdropFilter: 'blur(10px)',
          WebkitBackdropFilter: 'blur(10px)',
          transition: 'transform 0.3s ease, box-shadow 0.3s ease',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: `0 10px 20px ${moneyMoneyColors.cardGlow}`,
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: moneyMoneyColors.cardBackground,
          border: `1px solid ${moneyMoneyColors.cardBorder}`,
          backdropFilter: 'blur(10px)',
          WebkitBackdropFilter: 'blur(10px)',
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '4px',
          height: '8px',
        },
        bar: {
          background: `linear-gradient(90deg, ${moneyMoneyColors.primaryAccent}, ${moneyMoneyColors.secondaryAccent})`,
          borderRadius: '4px',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          backgroundColor: moneyMoneyColors.cardBackground,
          border: `1px solid ${moneyMoneyColors.cardBorder}`,
          backdropFilter: 'blur(10px)',
        },
        colorPrimary: {
          backgroundColor: 'rgba(225, 0, 122, 0.2)',
          borderColor: moneyMoneyColors.primaryAccent,
          color: moneyMoneyColors.primaryAccent,
        },
        colorSuccess: {
          backgroundColor: 'rgba(34, 197, 94, 0.2)',
          borderColor: moneyMoneyColors.success,
          color: moneyMoneyColors.success,
        },
        colorError: {
          backgroundColor: 'rgba(239, 68, 68, 0.2)',
          borderColor: moneyMoneyColors.error,
          color: moneyMoneyColors.error,
        },
        colorWarning: {
          backgroundColor: 'rgba(245, 158, 11, 0.2)',
          borderColor: moneyMoneyColors.warning,
          color: moneyMoneyColors.warning,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: moneyMoneyColors.cardBackground,
            backdropFilter: 'blur(10px)',
            '& fieldset': {
              borderColor: moneyMoneyColors.cardBorder,
            },
            '&:hover fieldset': {
              borderColor: moneyMoneyColors.primaryAccent,
            },
            '&.Mui-focused fieldset': {
              borderColor: moneyMoneyColors.primaryAccent,
              borderWidth: '2px',
            },
          },
        },
      },
    },
  },
});