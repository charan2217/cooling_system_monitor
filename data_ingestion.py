"""
Data ingestion module for OBD-II telemetry across multiple drives.
Handles CSV data with drive segmentation and basic validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class DriveData:
    """Container for a single drive's telemetry data."""
    
    def __init__(self, drive_id: str, df: pd.DataFrame, metadata: Optional[Dict] = None):
        self.drive_id = drive_id
        self.df = df
        self.metadata = metadata or {}
        self.start_time = df['timestamp'].min() if 'timestamp' in df.columns else None
        self.end_time = df['timestamp'].max() if 'timestamp' in df.columns else None
        self.duration = (self.end_time - self.start_time).total_seconds() if self.start_time else None
    
    def __len__(self):
        return len(self.df)
    
    def __repr__(self):
        return f"DriveData(id={self.drive_id}, samples={len(self)}, duration={self.duration:.1f}s)"


class DataIngestion:
    """Ingest and segment OBD-II telemetry data across multiple drives."""
    
    REQUIRED_COLUMNS = ['timestamp', 'rpm', 'coolant_temp', 'vehicle_speed', 
                       'throttle_pos', 'intake_air_temp']
    
    def __init__(self, time_gap_threshold: float = 300.0):
        """
        Args:
            time_gap_threshold: Seconds of inactivity to consider as drive boundary
        """
        self.time_gap_threshold = time_gap_threshold
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load OBD-II data from CSV file."""
        df = pd.read_csv(filepath)
        
        # Validate required columns
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def segment_drives(self, df: pd.DataFrame) -> List[DriveData]:
        """
        Segment continuous telemetry into individual drives based on time gaps.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            List of DriveData objects, one per drive
        """
        if len(df) == 0:
            return []
        
        # Calculate time gaps between consecutive readings
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        
        # Find drive boundaries (gaps exceeding threshold)
        drive_boundaries = [0]  # Start of first drive
        for i, gap in enumerate(time_diffs[1:], start=1):
            if gap > self.time_gap_threshold:
                drive_boundaries.append(i)
        drive_boundaries.append(len(df))  # End of last drive
        
        # Create DriveData objects for each segment
        drives = []
        for i in range(len(drive_boundaries) - 1):
            start_idx = drive_boundaries[i]
            end_idx = drive_boundaries[i + 1]
            
            if end_idx - start_idx < 10:  # Skip very short drives
                continue
            
            drive_df = df.iloc[start_idx:end_idx].copy()
            drive_id = f"drive_{i+1:03d}"
            
            metadata = {
                'start_time': drive_df['timestamp'].min().isoformat(),
                'end_time': drive_df['timestamp'].max().isoformat(),
                'sample_count': len(drive_df),
                'avg_rpm': drive_df['rpm'].mean(),
                'avg_speed': drive_df['vehicle_speed'].mean(),
                'avg_coolant_temp': drive_df['coolant_temp'].mean()
            }
            
            drives.append(DriveData(drive_id, drive_df, metadata))
        
        return drives
    
    def load_and_segment(self, filepath: str) -> List[DriveData]:
        """Convenience method to load CSV and segment into drives."""
        df = self.load_csv(filepath)
        return self.segment_drives(df)
    
    def validate_drive(self, drive: DriveData) -> bool:
        """Validate that a drive has sufficient data for analysis."""
        if len(drive) < 30:
            return False
        
        # Check for reasonable data ranges
        if drive.df['coolant_temp'].max() > 120 or drive.df['coolant_temp'].min() < -20:
            return False
        
        if drive.df['rpm'].max() > 10000 or drive.df['rpm'].min() < 0:
            return False
        
        return True


def generate_synthetic_obd_data(num_drives: int = 20, samples_per_drive: int = 500, 
                               degradation_start: int = 12, seed: int = 42) -> List[DriveData]:
    """
    Generate synthetic OBD-II data with injected cooling system degradation.
    
    This creates realistic driving patterns with gradual cooling system degradation
    starting at a specified drive.
    
    Args:
        num_drives: Number of drives to generate
        samples_per_drive: Average samples per drive
        degradation_start: Drive number where degradation begins
        seed: Random seed for reproducibility
        
    Returns:
        List of DriveData objects
    """
    np.random.seed(seed)
    
    drives = []
    current_time = datetime(2024, 1, 1, 9, 0, 0)
    
    for drive_num in range(1, num_drives + 1):
        # Vary drive length
        num_samples = int(samples_per_drive * np.random.uniform(0.8, 1.2))
        
        # Generate timestamps (1 Hz sampling)
        timestamps = [current_time + pd.Timedelta(seconds=i) for i in range(num_samples)]
        current_time = timestamps[-1] + pd.Timedelta(hours=np.random.uniform(2, 8))
        
        # Generate driving pattern (city vs highway mix)
        drive_type = np.random.choice(['city', 'highway'], p=[0.6, 0.4])
        
        if drive_type == 'city':
            rpm = np.random.normal(2000, 800, num_samples)
            speed = np.random.normal(30, 15, num_samples)
            throttle = np.random.normal(25, 15, num_samples)
        else:
            rpm = np.random.normal(2800, 400, num_samples)
            speed = np.random.normal(65, 10, num_samples)
            throttle = np.random.normal(35, 10, num_samples)
        
        # Ensure non-negative values
        rpm = np.maximum(rpm, 800)
        speed = np.maximum(speed, 0)
        throttle = np.clip(throttle, 0, 100)
        
        # Ambient temperature (seasonal variation)
        ambient_temp = 20 + 10 * np.sin(2 * np.pi * drive_num / 12) + np.random.normal(0, 3, num_samples)
        
        # Calculate degradation factor (0 = healthy, 1 = fully degraded)
        if drive_num >= degradation_start:
            degradation_progress = min((drive_num - degradation_start) / 8.0, 1.0)
        else:
            degradation_progress = 0.0
        
        # Coolant temperature with degradation effect
        # Base temperature depends on engine load (RPM, throttle) and ambient
        base_temp = 85 + 0.003 * rpm + 0.1 * throttle + 0.3 * (ambient_temp - 20)
        
        # Degradation causes: higher steady-state temp, slower cooling, more fluctuation
        temp_offset = 15 * degradation_progress  # Up to 15°C increase
        temp_noise = 2 + 5 * degradation_progress  # More noise as system degrades
        
        # Add thermal inertia (temperature doesn't change instantly)
        coolant_temp = np.zeros(num_samples)
        coolant_temp[0] = base_temp[0] + temp_offset
        
        for i in range(1, num_samples):
            # Thermal inertia: temp moves toward target with lag
            target_temp = base_temp[i] + temp_offset + np.random.normal(0, temp_noise)
            coolant_temp[i] = 0.9 * coolant_temp[i-1] + 0.1 * target_temp
        
        # Intake air temp (correlated with ambient but higher due to engine bay)
        intake_temp = ambient_temp + np.random.normal(15, 5, num_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'rpm': rpm,
            'coolant_temp': coolant_temp,
            'vehicle_speed': speed,
            'throttle_pos': throttle,
            'intake_air_temp': intake_temp
        })
        
        metadata = {
            'drive_type': drive_type,
            'degradation_progress': degradation_progress,
            'synthetic': True
        }
        
        drives.append(DriveData(f"drive_{drive_num:03d}", df, metadata))
    
    return drives


if __name__ == '__main__':
    # Test with synthetic data
    drives = generate_synthetic_obd_data(num_drives=5, samples_per_drive=100)
    for drive in drives:
        print(drive)
