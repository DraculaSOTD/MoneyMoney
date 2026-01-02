"""
Configuration Management System for Crypto ML Trading
Handles YAML/JSON configuration loading with validation and environment variable support.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import copy

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


@dataclass
class ModelConfig:
    """Base configuration for ML models."""
    enabled: bool = True
    training: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.training is None:
            self.training = {}


@dataclass
class SystemConfig:
    """System-level configuration."""
    name: str = "Crypto ML Trading System"
    version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"


class ConfigManager:
    """
    Centralized configuration management system.
    
    Features:
    - YAML/JSON configuration loading
    - Environment variable substitution
    - Configuration validation
    - Hot reloading support
    - Default value management
    - Nested configuration access
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config_data = {}
        self.defaults = self._get_default_config()
        self.environment_variables = {}
        
        # Load configuration if path provided
        if self.config_path:
            self.load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "system": {
                "name": "Crypto ML Trading System",
                "version": "1.0.0",
                "environment": "development",
                "log_level": "INFO"
            },
            "data": {
                "source": "file",
                "data_dir": "./data/historical",
                "symbols": ["BTCUSDT"],
                "interval": "1m",
                "lookback_days": 30
            },
            "models": {
                "arima": {
                    "enabled": True,
                    "max_p": 5,
                    "max_d": 2,
                    "max_q": 5
                },
                "garch": {
                    "enabled": True,
                    "p": 1,
                    "q": 1
                }
            },
            "trading": {
                "mode": "simulation",
                "base_currency": "USDT"
            },
            "risk_management": {
                "max_position_size": 0.2,
                "max_portfolio_risk": 0.05,
                "kelly_fraction": 0.25
            }
        }
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Optional path to override default config path
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            logger.warning(f"Configuration file not found: {self.config_path}. Using defaults.")
            self.config_data = copy.deepcopy(self.defaults)
            return self.config_data
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    loaded_config = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {self.config_path.suffix}")
            
            # Merge with defaults
            self.config_data = self._merge_configs(self.defaults, loaded_config)
            
            # Substitute environment variables
            self.config_data = self._substitute_environment_variables(self.config_data)
            
            # Validate configuration
            self._validate_config()
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self.config_data
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge loaded configuration with defaults.
        
        Args:
            default: Default configuration
            loaded: Loaded configuration
            
        Returns:
            Merged configuration
        """
        merged = copy.deepcopy(default)
        
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _substitute_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration.
        
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {k: self._substitute_environment_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_environment_variables(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            var_expr = config[2:-1]  # Remove ${ and }
            
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                value = os.getenv(var_name, default_value)
            else:
                value = os.getenv(var_expr)
                if value is None:
                    raise ConfigurationError(f"Environment variable {var_expr} not found and no default provided")
            
            # Try to convert to appropriate type
            try:
                if value.lower() in ('true', 'false'):
                    return value.lower() == 'true'
                elif value.isdigit():
                    return int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    return float(value)
                else:
                    return value
            except:
                return value
        else:
            return config
    
    def _validate_config(self):
        """Validate configuration values."""
        # System validation
        system = self.config_data.get('system', {})
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if system.get('log_level', 'INFO') not in valid_log_levels:
            raise ConfigurationError(f"Invalid log level: {system.get('log_level')}")
        
        valid_environments = ['development', 'staging', 'production']
        if system.get('environment', 'development') not in valid_environments:
            raise ConfigurationError(f"Invalid environment: {system.get('environment')}")
        
        # Risk management validation
        risk_mgmt = self.config_data.get('risk_management', {})
        max_pos_size = risk_mgmt.get('max_position_size', 0.2)
        if not 0 < max_pos_size <= 1.0:
            raise ConfigurationError(f"Invalid max_position_size: {max_pos_size}. Must be between 0 and 1.")
        
        max_portfolio_risk = risk_mgmt.get('max_portfolio_risk', 0.05)
        if not 0 < max_portfolio_risk <= 1.0:
            raise ConfigurationError(f"Invalid max_portfolio_risk: {max_portfolio_risk}. Must be between 0 and 1.")
        
        # Trading validation
        trading = self.config_data.get('trading', {})
        valid_modes = ['simulation', 'paper', 'live']
        if trading.get('mode', 'simulation') not in valid_modes:
            raise ConfigurationError(f"Invalid trading mode: {trading.get('mode')}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'models.arima.enabled')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        current = self.config_data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'models.arima.enabled')
            value: Value to set
        """
        keys = key.split('.')
        current = self.config_data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        return self.get(f'models.{model_name}', {})
    
    def is_model_enabled(self, model_name: str) -> bool:
        """
        Check if a model is enabled.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is enabled
        """
        return self.get(f'models.{model_name}.enabled', False)
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.get('trading', {})
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return self.get('risk_management', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None):
        """
        Save current configuration to file.
        
        Args:
            output_path: Output file path (defaults to current config_path)
        """
        output_path = Path(output_path) if output_path else self.config_path
        
        if not output_path:
            raise ConfigurationError("No output path specified for saving configuration")
        
        try:
            with open(output_path, 'w') as f:
                if output_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
                elif output_path.suffix.lower() == '.json':
                    json.dump(self.config_data, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported output format: {output_path.suffix}")
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def reload_config(self):
        """Reload configuration from file."""
        if self.config_path:
            self.load_config()
            logger.info("Configuration reloaded")
        else:
            logger.warning("No configuration path set for reloading")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return copy.deepcopy(self.config_data)
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """
        Update configuration from dictionary.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self.config_data = self._merge_configs(self.config_data, updates)
        self._validate_config()
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Support dictionary-style assignment."""
        self.set(key, value)


# Global configuration manager instance
config_manager = ConfigManager()


def load_config(config_path: Union[str, Path]) -> ConfigManager:
    """
    Load configuration and return manager instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    global config_manager
    config_manager = ConfigManager(config_path)
    return config_manager


def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    return config_manager


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    config_path = Path(__file__).parent / "system_config.yaml"
    
    # Load configuration
    manager = load_config(config_path)
    
    # Test getting values
    print("System name:", manager.get("system.name"))
    print("ARIMA enabled:", manager.is_model_enabled("arima"))
    print("Max position size:", manager.get("risk_management.max_position_size"))
    
    # Test model config
    arima_config = manager.get_model_config("arima")
    print("ARIMA config:", arima_config)
    
    # Test configuration update
    manager.set("models.arima.max_p", 10)
    print("Updated ARIMA max_p:", manager.get("models.arima.max_p"))
    
    # Test environment variable substitution
    os.environ["TEST_VALUE"] = "42"
    manager.config_data["test"] = "${TEST_VALUE:default}"
    substituted = manager._substitute_environment_variables(manager.config_data)
    print("Environment substitution:", substituted.get("test"))
    
    print("Configuration management test completed successfully!")