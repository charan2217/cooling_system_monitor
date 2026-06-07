"""
Cooling system health parameter extraction module.
Designs parameters that are comparable across drives by normalizing for driving conditions.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from data_ingestion import DriveData


class CoolingSystemParameters:
    """
    Extract cooling system health parameters that are comparable across drives.
    
    Key insight: Raw coolant temperature depends heavily on driving conditions.
    We normalize by:
    1. Expected temperature given engine load (RPM, throttle) and ambient conditions
    2. Thermal response characteristics (how quickly temp changes)
    3. Statistical stability measures
    """
    
    def __init__(self, baseline_window: int = 10):
        """
        Args:
            baseline_window: Number of initial drives to establish baseline behavior
        """
        self.baseline_window = baseline_window
        self.baseline_params = None
        self.is_fitted = False
    
    def _calculate_engine_load(self, drive: DriveData) -> np.ndarray:
        """
        Calculate normalized engine load (0-1 scale).
        Combines RPM and throttle position into a single load metric.
        """
        df = drive.df
        rpm_norm = (df['rpm'] - 800) / (6000 - 800)  # Normalize RPM
        rpm_norm = np.clip(rpm_norm, 0, 1)
        
        throttle_norm = df['throttle_pos'] / 100.0  # Normalize throttle
        throttle_norm = np.clip(throttle_norm, 0, 1)
        
        # Combined load metric (weighted average)
        engine_load = 0.6 * rpm_norm + 0.4 * throttle_norm
        return engine_load.values
    
    def _calculate_expected_temp(self, drive: DriveData) -> np.ndarray:
        """
        Calculate expected coolant temperature given current conditions.
        
        Model: Expected temp = base_temp + load_effect + ambient_effect
        - Base temp: ~85°C when idling in normal conditions
        - Load effect: Higher RPM/throttle increases temp
        - Ambient effect: Higher ambient temp increases operating temp
        """
        df = drive.df
        engine_load = self._calculate_engine_load(drive)
        
        # Base temperature at idle
        base_temp = 85.0
        
        # Load contribution (up to +25°C at full load)
        load_effect = 25.0 * engine_load
        
        # Ambient contribution (0.3°C per degree above 20°C baseline)
        ambient_effect = 0.3 * (df['intake_air_temp'] - 20)
        
        expected_temp = base_temp + load_effect + ambient_effect
        return expected_temp.values
    
    def _calculate_thermal_efficiency(self, drive: DriveData) -> Dict[str, float]:
        """
        Calculate thermal efficiency metrics.
        
        A healthy cooling system maintains temperature close to expected
        and responds quickly to load changes.
        """
        df = drive.df
        actual_temp = df['coolant_temp'].values
        expected_temp = self._calculate_expected_temp(drive)
        
        # Temperature deviation from expected
        temp_deviation = actual_temp - expected_temp
        
        # Key metrics
        metrics = {
            'mean_deviation': np.mean(temp_deviation),
            'std_deviation': np.std(temp_deviation),
            'max_deviation': np.max(temp_deviation),
            'deviation_95th': np.percentile(np.abs(temp_deviation), 95),
            
            # Temperature stability (lower is better)
            'temp_variance': np.var(actual_temp),
            'temp_range': np.max(actual_temp) - np.min(actual_temp),
            
            # Correlation with load (healthy system shows predictable correlation)
            'load_temp_correlation': np.corrcoef(
                self._calculate_engine_load(drive), actual_temp
            )[0, 1] if len(actual_temp) > 1 else 0.0,
            
            # Percent of time in optimal range (80-95°C)
            'percent_in_optimal_range': np.mean(
                (actual_temp >= 80) & (actual_temp <= 95)
            ) * 100,
            
            # Percent of time in danger zone (>105°C)
            'percent_in_danger_zone': np.mean(actual_temp > 105) * 100,
            
            # Average temperature under different load conditions
            'avg_temp_low_load': np.mean(actual_temp[self._calculate_engine_load(drive) < 0.3]) if np.any(self._calculate_engine_load(drive) < 0.3) else 0.0,
            'avg_temp_high_load': np.mean(actual_temp[self._calculate_engine_load(drive) > 0.7]) if np.any(self._calculate_engine_load(drive) > 0.7) else 0.0,
        }
        
        # Handle NaN values
        for key, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                metrics[key] = 0.0
        
        return metrics
    
    def _calculate_thermal_response(self, drive: DriveData) -> Dict[str, float]:
        """
        Calculate thermal response characteristics.
        
        A healthy cooling system responds quickly to load changes.
        Degraded systems show sluggish response.
        """
        df = drive.df
        temp = df['coolant_temp'].values
        engine_load = self._calculate_engine_load(drive)
        
        # Calculate temperature change rate
        temp_change = np.diff(temp)
        load_change = np.diff(engine_load)
        
        # Response coefficient: how quickly temp follows load changes
        # Higher values indicate faster, more responsive cooling
        if len(temp_change) > 0 and len(load_change) > 0:
            # Cross-correlation at lag 0
            response_coeff = np.corrcoef(np.abs(load_change), np.abs(temp_change))[0, 1]
            if np.isnan(response_coeff):
                response_coeff = 0.0
        else:
            response_coeff = 0.0
        
        # Thermal inertia: autocorrelation of temperature
        # Higher values = more inertia (slower to change)
        if len(temp) > 10:
            inertia = np.corrcoef(temp[:-1], temp[1:])[0, 1]
            if np.isnan(inertia):
                inertia = 0.0
        else:
            inertia = 0.0
        
        return {
            'response_coefficient': response_coeff,
            'thermal_inertia': inertia,
            'avg_temp_change_rate': np.mean(np.abs(temp_change)) if len(temp_change) > 0 else 0.0,
        }
    
    def extract_drive_parameters(self, drive: DriveData) -> Dict[str, float]:
        """
        Extract all cooling system health parameters for a single drive.
        
        Returns a dictionary of normalized parameters that can be compared
        across drives with different conditions.
        """
        # Basic statistics
        df = drive.df
        
        params = {
            'drive_id': drive.drive_id,
            'timestamp': drive.start_time.isoformat() if drive.start_time else None,
            'sample_count': len(drive),
            'duration': drive.duration if drive.duration else 0,
            
            # Raw averages (for reference)
            'avg_coolant_temp': df['coolant_temp'].mean(),
            'avg_rpm': df['rpm'].mean(),
            'avg_speed': df['vehicle_speed'].mean(),
            'avg_throttle': df['throttle_pos'].mean(),
            'avg_intake_temp': df['intake_air_temp'].mean(),
        }
        
        # Thermal efficiency metrics
        efficiency = self._calculate_thermal_efficiency(drive)
        params.update({f'eff_{k}': v for k, v in efficiency.items()})
        
        # Thermal response metrics
        response = self._calculate_thermal_response(drive)
        params.update({f'resp_{k}': v for k, v in response.items()})
        
        return params
    
    def fit_baseline(self, drives: List[DriveData]) -> None:
        """
        Establish baseline parameters from initial drives (healthy system).
        
        This establishes what "normal" looks like for this vehicle,
        allowing us to detect deviations over time.
        """
        baseline_drives = drives[:self.baseline_window]
        
        # Extract parameters for baseline drives
        baseline_params_list = []
        for drive in baseline_drives:
            params = self.extract_drive_parameters(drive)
            baseline_params_list.append(params)
        
        # Calculate baseline statistics
        baseline_df = pd.DataFrame(baseline_params_list)
        
        # Store baseline statistics for each parameter
        self.baseline_params = {}
        for col in baseline_df.columns:
            if col in ['drive_id', 'timestamp']:
                continue
            self.baseline_params[col] = {
                'mean': baseline_df[col].mean(),
                'std': baseline_df[col].std(),
                'min': baseline_df[col].min(),
                'max': baseline_df[col].max(),
            }
        
        self.is_fitted = True
    
    def calculate_health_score(self, params: Dict[str, float]) -> float:
        """
        Calculate overall cooling system health score (0-100).
        
        100 = perfectly healthy (matches baseline)
        0 = severely degraded
        
        Based on deviation from baseline parameters.
        """
        if not self.is_fitted:
            return 50.0  # Neutral score if no baseline
        
        score_components = []
        
        # Key parameters for health assessment
        key_params = [
            'eff_mean_deviation',  # Higher deviation = worse
            'eff_std_deviation',   # Higher std = worse
            'eff_deviation_95th',  # Higher 95th percentile = worse
            'eff_percent_in_optimal_range',  # Lower = worse
            'eff_percent_in_danger_zone',     # Higher = worse
            'resp_response_coefficient',      # Lower = worse
            'resp_thermal_inertia',           # Higher = worse
        ]
        
        for param in key_params:
            if param not in params or param not in self.baseline_params:
                continue
            
            value = params[param]
            baseline = self.baseline_params[param]
            
            # Calculate z-score
            if baseline['std'] > 0:
                z_score = abs(value - baseline['mean']) / baseline['std']
            else:
                z_score = 0.0
            
            # Convert z-score to component score (0-100)
            # z_score of 0 = 100, z_score of 3 = 0
            component_score = max(0, 100 - (z_score * 33.33))
            score_components.append(component_score)
        
        if not score_components:
            return 50.0
        
        # Weighted average (some parameters more important)
        weights = [1.5, 1.2, 1.3, 1.0, 2.0, 1.0, 0.8]  # danger_zone weighted heavily
        weighted_sum = sum(s * w for s, w in zip(score_components, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight
    
    def extract_all_drives(self, drives: List[DriveData]) -> pd.DataFrame:
        """
        Extract parameters for all drives and calculate health scores.
        """
        # Fit baseline on first N drives
        self.fit_baseline(drives)
        
        # Extract parameters for all drives
        all_params = []
        for drive in drives:
            params = self.extract_drive_parameters(drive)
            health_score = self.calculate_health_score(params)
            params['health_score'] = health_score
            all_params.append(params)
        
        return pd.DataFrame(all_params)


if __name__ == '__main__':
    # Test with synthetic data
    from data_ingestion import generate_synthetic_obd_data
    
    drives = generate_synthetic_obd_data(num_drives=5, samples_per_drive=100)
    
    extractor = CoolingSystemParameters(baseline_window=3)
    params_df = extractor.extract_all_drives(drives)
    
    print("Extracted Parameters:")
    print(params_df[['drive_id', 'health_score', 'eff_mean_deviation', 'eff_percent_in_danger_zone']].head())
