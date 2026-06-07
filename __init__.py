"""
Cooling System Health Monitor
A system for detecting cooling system degradation from OBD-II telemetry.
"""

from .data_ingestion import DataIngestion, DriveData, generate_synthetic_obd_data
from .parameter_extraction import CoolingSystemParameters
from .trend_analysis import TrendAnalyzer, AnomalyDetection
from .storage import DualStorage, VehicleState
from .visualization import create_health_plot, create_detailed_plot

__version__ = "1.0.0"
__all__ = [
    "DataIngestion",
    "DriveData",
    "generate_synthetic_obd_data",
    "CoolingSystemParameters",
    "TrendAnalyzer",
    "AnomalyDetection",
    "DualStorage",
    "VehicleState",
    "create_health_plot",
    "create_detailed_plot",
]
