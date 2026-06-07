"""
Dual-storage module for cooling system health monitoring.
Implements append-only CSV log + per-vehicle JSON state snapshot.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class VehicleState:
    """Compact per-vehicle state snapshot with 3-layer predictive maintenance."""
    vehicle_id: str
    last_updated: str
    total_drives: int
    current_health_score: float
    baseline_health_score: float
    health_trend: str  # 'improving', 'stable', 'degrading'
    last_anomaly: Optional[str]
    degradation_detected: bool
    degradation_drive_index: Optional[int]
    risk_index: float = 0.0  # Layer 2: Degradation Risk Index (0-1)
    drives_to_failure: Optional[int] = None  # Layer 3: Estimated drives until threshold breach
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VehicleState':
        return cls(**data)


class DualStorage:
    """
    Dual-storage pattern: append-only CSV log + compact JSON snapshot.
    
    CSV Log: Complete history of all drive analyses (append-only)
    JSON Snapshot: Current state summary (compact, frequently updated)
    """
    
    def __init__(self, output_dir: str = './output', vehicle_id: str = 'vehicle_001'):
        """
        Args:
            output_dir: Directory for storage files
            vehicle_id: Unique identifier for this vehicle
        """
        self.output_dir = Path(output_dir)
        self.vehicle_id = vehicle_id
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.csv_log_path = self.output_dir / f'{vehicle_id}_health_log.csv'
        self.json_snapshot_path = self.output_dir / f'{vehicle_id}_state.json'
        
        # Initialize CSV log with headers if it doesn't exist
        self._init_csv_log()
        
        # Load or initialize vehicle state
        self.vehicle_state = self._load_or_init_state()
    
    def _init_csv_log(self) -> None:
        """Initialize CSV log with headers if it doesn't exist."""
        if not self.csv_log_path.exists():
            headers = [
                'drive_id', 'timestamp', 'health_score', 'is_anomalous',
                'anomaly_type', 'confidence', 'severity', 'message',
                'risk_index', 'drives_to_failure',  # Layer 2 and 3 fields
                'avg_coolant_temp', 'eff_mean_deviation', 'eff_std_deviation',
                'eff_percent_in_danger_zone', 'resp_response_coefficient'
            ]
            pd.DataFrame(columns=headers).to_csv(self.csv_log_path, index=False)
    
    def _load_or_init_state(self) -> VehicleState:
        """Load existing state or initialize new state."""
        if self.json_snapshot_path.exists():
            with open(self.json_snapshot_path, 'r') as f:
                data = json.load(f)
                return VehicleState.from_dict(data)
        else:
            return VehicleState(
                vehicle_id=self.vehicle_id,
                last_updated=datetime.now().isoformat(),
                total_drives=0,
                current_health_score=100.0,
                baseline_health_score=100.0,
                health_trend='stable',
                last_anomaly=None,
                degradation_detected=False,
                degradation_drive_index=None,
                risk_index=0.0,
                drives_to_failure=None
            )
    
    def append_drive_log(self, drive_data: Dict[str, Any]) -> None:
        """
        Append a single drive analysis to the CSV log (append-only).
        
        Args:
            drive_data: Dictionary with drive analysis results
        """
        # Select only the columns we want in the log
        log_entry = {
            'drive_id': drive_data.get('drive_id'),
            'timestamp': drive_data.get('timestamp'),
            'health_score': drive_data.get('health_score'),
            'is_anomalous': drive_data.get('is_anomalous'),
            'anomaly_type': drive_data.get('anomaly_type'),
            'confidence': drive_data.get('confidence'),
            'severity': drive_data.get('severity'),
            'message': drive_data.get('message'),
            'risk_index': drive_data.get('risk_index', 0.0),  # Layer 2
            'drives_to_failure': drive_data.get('drives_to_failure'),  # Layer 3
            'avg_coolant_temp': drive_data.get('avg_coolant_temp'),
            'eff_mean_deviation': drive_data.get('eff_mean_deviation'),
            'eff_std_deviation': drive_data.get('eff_std_deviation'),
            'eff_percent_in_danger_zone': drive_data.get('eff_percent_in_danger_zone'),
            'resp_response_coefficient': drive_data.get('resp_response_coefficient'),
        }
        
        # Append to CSV
        df = pd.DataFrame([log_entry])
        df.to_csv(self.csv_log_path, mode='a', header=False, index=False)
    
    def append_batch_log(self, drive_data_list: list) -> None:
        """
        Append multiple drive analyses to the CSV log (append-only).
        
        Args:
            drive_data_list: List of dictionaries with drive analysis results
        """
        log_entries = []
        for drive_data in drive_data_list:
            log_entry = {
                'drive_id': drive_data.get('drive_id'),
                'timestamp': drive_data.get('timestamp'),
                'health_score': drive_data.get('health_score'),
                'is_anomalous': drive_data.get('is_anomalous'),
                'anomaly_type': drive_data.get('anomaly_type'),
                'confidence': drive_data.get('confidence'),
                'severity': drive_data.get('severity'),
                'message': drive_data.get('message'),
                'risk_index': drive_data.get('risk_index', 0.0),  # Layer 2
                'drives_to_failure': drive_data.get('drives_to_failure'),  # Layer 3
                'avg_coolant_temp': drive_data.get('avg_coolant_temp'),
                'eff_mean_deviation': drive_data.get('eff_mean_deviation'),
                'eff_std_deviation': drive_data.get('eff_std_deviation'),
                'eff_percent_in_danger_zone': drive_data.get('eff_percent_in_danger_zone'),
                'resp_response_coefficient': drive_data.get('resp_response_coefficient'),
            }
            log_entries.append(log_entry)
        
        # Append to CSV
        df = pd.DataFrame(log_entries)
        df.to_csv(self.csv_log_path, mode='a', header=False, index=False)
    
    def update_snapshot(self, result_df: pd.DataFrame, degradation_idx: Optional[int] = None) -> None:
        """
        Update the compact JSON state snapshot.
        
        Args:
            result_df: DataFrame with all drive analysis results
            degradation_idx: Index where degradation was first detected
        """
        if len(result_df) == 0:
            return
        
        # Calculate current state
        latest_drive = result_df.iloc[-1]
        current_health_score = latest_drive['health_score']
        
        # Calculate baseline (average of first 5 drives)
        baseline_window = min(5, len(result_df))
        baseline_health_score = result_df['health_score'].iloc[:baseline_window].mean()
        
        # Determine trend
        if len(result_df) >= 5:
            recent_scores = result_df['health_score'].tail(5).values
            slope, _, is_significant = self._calculate_simple_trend(recent_scores)
            
            if is_significant:
                if slope > 0.5:
                    health_trend = 'improving'
                elif slope < -0.5:
                    health_trend = 'degrading'
                else:
                    health_trend = 'stable'
            else:
                health_trend = 'stable'
        else:
            health_trend = 'stable'
        
        # Find last anomaly
        anomalous_drives = result_df[result_df['is_anomalous'] == True]
        if len(anomalous_drives) > 0:
            last_anomaly = anomalous_drives.iloc[-1]['message']
        else:
            last_anomaly = None
        
        # Update state
        self.vehicle_state = VehicleState(
            vehicle_id=self.vehicle_id,
            last_updated=datetime.now().isoformat(),
            total_drives=len(result_df),
            current_health_score=float(current_health_score),
            baseline_health_score=float(baseline_health_score),
            health_trend=health_trend,
            last_anomaly=last_anomaly,
            degradation_detected=degradation_idx is not None,
            degradation_drive_index=int(degradation_idx) if degradation_idx is not None else None,
            risk_index=float(latest_drive.get('risk_index', 0.0)),
            drives_to_failure=int(latest_drive.get('drives_to_failure')) if latest_drive.get('drives_to_failure') is not None else None
        )
        
        # Write to JSON
        with open(self.json_snapshot_path, 'w') as f:
            json.dump(self.vehicle_state.to_dict(), f, indent=2)
    
    def _calculate_simple_trend(self, values: np.ndarray) -> tuple:
        """Calculate simple linear trend."""
        if len(values) < 2:
            return 0.0, 0.0, False
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Calculate R-squared
        y_pred = np.polyval([slope, 0], x)
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Simple significance check: slope magnitude > 0.5
        is_significant = abs(slope) > 0.5
        
        return slope, r_squared, is_significant
    
    def get_log(self) -> pd.DataFrame:
        """Read the complete CSV log."""
        return pd.read_csv(self.csv_log_path)
    
    def get_snapshot(self) -> VehicleState:
        """Get the current vehicle state snapshot."""
        return self.vehicle_state
    
    def save_results(self, result_df: pd.DataFrame, degradation_idx: Optional[int] = None) -> None:
        """
        Convenience method to save both log and snapshot.
        
        Args:
            result_df: DataFrame with all drive analysis results
            degradation_idx: Index where degradation was first detected
        """
        # Append all drives to log
        drive_data_list = result_df.to_dict('records')
        self.append_batch_log(drive_data_list)
        
        # Update snapshot
        self.update_snapshot(result_df, degradation_idx)


if __name__ == '__main__':
    # Test dual-storage
    storage = DualStorage(output_dir='./test_output', vehicle_id='test_vehicle')
    
    # Simulate some drive data
    test_data = [
        {
            'drive_id': 'drive_001',
            'timestamp': '2024-01-01T09:00:00',
            'health_score': 95.0,
            'is_anomalous': False,
            'anomaly_type': None,
            'confidence': 0.0,
            'severity': 'low',
            'message': 'Normal operation',
            'avg_coolant_temp': 88.5,
            'eff_mean_deviation': 2.1,
            'eff_std_deviation': 1.8,
            'eff_percent_in_danger_zone': 0.0,
            'resp_response_coefficient': 0.75
        },
        {
            'drive_id': 'drive_002',
            'timestamp': '2024-01-02T10:00:00',
            'health_score': 92.0,
            'is_anomalous': False,
            'anomaly_type': None,
            'confidence': 0.0,
            'severity': 'low',
            'message': 'Normal operation',
            'avg_coolant_temp': 89.2,
            'eff_mean_deviation': 2.3,
            'eff_std_deviation': 2.0,
            'eff_percent_in_danger_zone': 0.0,
            'resp_response_coefficient': 0.72
        }
    ]
    
    storage.append_batch_log(test_data)
    
    # Update snapshot
    result_df = pd.DataFrame(test_data)
    storage.update_snapshot(result_df)
    
    print("Log saved to:", storage.csv_log_path)
    print("Snapshot saved to:", storage.json_snapshot_path)
    print("\nCurrent state:")
    print(storage.get_snapshot())
