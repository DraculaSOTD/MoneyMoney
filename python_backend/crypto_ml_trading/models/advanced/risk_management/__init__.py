"""
Advanced Risk Management System for Cryptocurrency Trading.

This module implements sophisticated risk management capabilities including
dynamic position sizing, portfolio risk assessment, stress testing, and
real-time risk monitoring with advanced hedging strategies.

Key Features:
- Dynamic position sizing with Kelly Criterion and risk parity
- Advanced VaR and CVaR calculations with multiple methodologies
- Stress testing and scenario analysis
- Real-time risk monitoring and alerting
- Portfolio optimization with risk constraints
- Advanced hedging strategies and tail risk protection
- Liquidity risk assessment and management
- Correlation risk monitoring and diversification analysis
"""

from models.advanced.risk_management.position_sizer import PositionSizer
from models.advanced.risk_management.portfolio_risk_manager import PortfolioRiskManager
from models.advanced.risk_management.stress_tester import StressTester
from models.advanced.risk_management.hedging_manager import HedgingManager
from models.advanced.risk_management.risk_coordinator import RiskCoordinator

__all__ = [
    'PositionSizer',
    'PortfolioRiskManager',
    'StressTester',
    'HedgingManager',
    'RiskCoordinator'
]