"""
Trend analysis and anomaly detection module.
Detects cooling system degradation from parameter trends over time.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AnomalyDetection:
    """Result of anomaly detection for a single drive."""
    drive_id: str
    health_score: float
    is_anomalous: bool
    anomaly_type: Optional[str]  # 'spc', 'trend', 'change_point', 'isolation_forest', 'threshold'
    confidence: float  # 0-1
    severity: str  # 'low', 'medium', 'high'
    message: str
    risk_index: float = 0.0  # Layer 2: Degradation Risk Index (0-1)
    drives_to_failure: Optional[int] = None  # Layer 3: Estimated drives until threshold breach


class TrendAnalyzer:
    """
    Analyze trends in cooling system health parameters over multiple drives.
    
    Automotive-Style Predictive Maintenance System:
    
    Layer 1: Health Score (Current Condition)
    - 0-100 scale indicating current cooling system health
    
    Layer 2: Degradation Risk Index (Predictive)
    - 0-1 scale based on:
      * slope of health score
      * increase in mean temperature deviation
      * increase in danger-zone exposure
      * increase in thermal instability
    
    Layer 3: Remaining Drives to Failure
    - Estimated drives until health score crosses threshold
    
    Detection Pipeline (ordered by priority):
    1. Trend Detection (last 5 scores, slope < -1.5, R² > 0.5) - Early warning
    2. Parameter Drift (mean_deviation, danger_zone) - Confirms degradation
    3. SPC Control Chart (mean ± 3σ) - Statistical confirmation
    4. Threshold Detection (health_score < 65) - Final stage
    """
    
    def __init__(self, 
                 min_drives_for_trend: int = 5,
                 health_threshold: float = 65.0):
        """
        Args:
            min_drives_for_trend: Minimum drives before trend analysis is reliable
            health_threshold: Health score below which to flag as concerning
        """
        self.min_drives_for_trend = min_drives_for_trend
        self.health_threshold = health_threshold
    
    def _detect_trend(self, values: np.ndarray) -> Tuple[float, float, bool]:
        """
        Detect linear trend using linear regression.
        
        Returns:
            slope: Rate of change per drive
            r_squared: Fit quality
            is_significant: Whether trend is statistically significant
        """
        if len(values) < 3:
            return 0.0, 0.0, False
        
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Trend is significant if p-value < 0.05
        is_significant = p_value < 0.05
        
        return slope, r_value**2, is_significant
    
    
    def _calculate_control_limits(self, baseline_values: np.ndarray) -> Tuple[float, float]:
        """
        Calculate statistical process control limits.
        
        Returns:
            lower_limit: 3-sigma lower control limit
            upper_limit: 3-sigma upper control limit
        """
        mean = np.mean(baseline_values)
        std = np.std(baseline_values)
        
        # 3-sigma limits (standard in SPC)
        lower_limit = mean - 3 * std
        upper_limit = mean + 3 * std
        
        return lower_limit, upper_limit
    
    
    def _calculate_risk_index(self, 
                            health_scores: np.ndarray,
                            all_params: pd.DataFrame,
                            drive_idx: int) -> float:
        """
        Calculate Degradation Risk Index (Layer 2).
        
        Combines multiple factors:
        - Slope of health score (negative slope = higher risk)
        - Increase in mean temperature deviation
        - Increase in danger-zone exposure
        - Increase in thermal instability (std deviation)
        
        Returns risk index between 0 (low risk) and 1 (high risk).
        """
        if len(health_scores) < 3:
            return 0.0
        
        recent_scores = health_scores[:drive_idx + 1]
        
        # Factor 1: Slope of health score (normalized)
        slope, r_squared, _ = self._detect_trend(recent_scores)
        slope_risk = max(0, -slope / 5.0)  # Normalize negative slope to 0-1
        
        # Factor 2: Temperature deviation trend
        recent_deviations = all_params.iloc[:drive_idx + 1]['eff_mean_deviation'].values
        if len(recent_deviations) >= 3:
            dev_slope, _, _ = self._detect_trend(recent_deviations)
            dev_risk = max(0, dev_slope / 2.0)  # Normalize positive slope to 0-1
        else:
            dev_risk = 0.0
        
        # Factor 3: Danger zone exposure trend
        recent_danger = all_params.iloc[:drive_idx + 1]['eff_percent_in_danger_zone'].values
        if len(recent_danger) >= 3:
            danger_slope, _, _ = self._detect_trend(recent_danger)
            danger_risk = max(0, danger_slope / 10.0)  # Normalize to 0-1
        else:
            danger_risk = 0.0
        
        # Factor 4: Thermal instability (std deviation trend)
        recent_std = all_params.iloc[:drive_idx + 1]['eff_std_deviation'].values
        if len(recent_std) >= 3:
            std_slope, _, _ = self._detect_trend(recent_std)
            std_risk = max(0, std_slope / 1.0)  # Normalize to 0-1
        else:
            std_risk = 0.0
        
        # Combine factors with weights
        risk_index = (0.4 * slope_risk + 
                     0.2 * dev_risk + 
                     0.2 * danger_risk + 
                     0.2 * std_risk)
        
        return min(1.0, max(0.0, risk_index))
    
    def _estimate_drives_to_failure(self, 
                                   health_scores: np.ndarray,
                                   drive_idx: int) -> Optional[int]:
        """
        Estimate remaining drives until health score crosses threshold (Layer 3).
        
        Fits linear regression to recent health scores and extrapolates.
        
        Returns estimated number of drives, or None if trend is not significant.
        """
        if len(health_scores) < 5:
            return None
        
        recent_scores = health_scores[:drive_idx + 1]
        slope, r_squared, is_significant = self._detect_trend(recent_scores)
        
        current_score = health_scores[drive_idx]
        
        # If already below threshold
        if current_score <= self.health_threshold:
            return 0
        
        # Relaxed threshold: estimate even if trend is not very significant
        # Just need negative slope with reasonable R²
        if slope >= 0 or r_squared < 0.2:
            # If no clear negative trend, estimate based on current score
            # Simple heuristic: if score is close to threshold, estimate fewer drives
            if current_score > self.health_threshold + 20:
                return 10  # Far from threshold
            elif current_score > self.health_threshold + 10:
                return 5   # Moderately close
            else:
                return 3   # Very close
        
        # Estimate drives to threshold: (threshold - current) / slope
        drives_to_threshold = (self.health_threshold - current_score) / slope
        
        # Round to nearest integer and ensure positive
        drives_estimated = max(1, int(round(drives_to_threshold)))
        
        # Cap at reasonable maximum (e.g., 50 drives)
        return min(50, drives_estimated)
    
    
    def analyze_drive(self, 
                      drive_idx: int,
                      health_scores: np.ndarray,
                      all_params: pd.DataFrame) -> AnomalyDetection:
        """
        Analyze a single drive for anomalies using automotive-style predictive maintenance.
        
        Detection Pipeline (ordered by priority):
        1. Trend Detection (last 5 scores, slope < -2, R² > 0.7) - Early warning
        2. Parameter Drift (mean_deviation, danger_zone) - Confirms degradation
        3. SPC Control Chart (mean ± 3σ) - Statistical confirmation
        4. Threshold Detection (health_score < 65) - Final stage
        
        Isolation Forest is NOT used as a primary trigger to avoid false positives.
        
        Args:
            drive_idx: Index of current drive in the sequence
            health_scores: Array of health scores up to and including current drive
            all_params: DataFrame of all parameters extracted
            
        Returns:
            AnomalyDetection object with risk_index and drives_to_failure
        """
        current_score = health_scores[drive_idx]
        drive_id = all_params.iloc[drive_idx]['drive_id']
        
        # Calculate Layer 2: Risk Index
        risk_index = self._calculate_risk_index(health_scores, all_params, drive_idx)
        
        # Calculate Layer 3: Drives to Failure
        drives_to_failure = self._estimate_drives_to_failure(health_scores, drive_idx)
        
        # Not enough data for trend analysis
        if drive_idx < self.min_drives_for_trend - 1:
            return AnomalyDetection(
                drive_id=drive_id,
                health_score=current_score,
                is_anomalous=False,
                anomaly_type=None,
                confidence=0.0,
                severity='low',
                message='Insufficient data for trend analysis',
                risk_index=risk_index,
                drives_to_failure=drives_to_failure
            )
        
        # Get historical data (excluding current drive for trend calculation)
        historical_scores = health_scores[:drive_idx]
        
        # Calculate baseline statistics
        baseline_scores = historical_scores[:self.min_drives_for_trend]
        baseline_mean = np.mean(baseline_scores)
        baseline_std = np.std(baseline_scores)
        
        # Get baseline parameter values
        baseline_mean_dev = all_params['eff_mean_deviation'].iloc[:self.min_drives_for_trend].mean()
        baseline_std_dev = all_params['eff_mean_deviation'].iloc[:self.min_drives_for_trend].std()
        
        # === DETECTION PIPELINE (ordered by priority) ===
        
        # 1. Trend Detection (last 5 scores, slope < -1.5, R² > 0.5) - Early warning
        # Only check if we have at least 5 drives for trend analysis
        if len(historical_scores) >= 5:
            recent_scores = historical_scores[-5:]  # Last 5 scores
            slope, r_squared, is_significant = self._detect_trend(recent_scores)
            
            # Relaxed thresholds for earlier detection while maintaining reliability
            if is_significant and slope < -1.5 and r_squared > 0.5:
                # Early warning: trend detected while still above threshold
                if current_score > self.health_threshold:
                    return AnomalyDetection(
                        drive_id=drive_id,
                        health_score=current_score,
                        is_anomalous=True,
                        anomaly_type='early_warning',
                        confidence=min(0.9, r_squared),
                        severity='medium',
                        message=f'Early Warning: degrading trend (slope={slope:.2f}/drive, R²={r_squared:.2f})',
                        risk_index=risk_index,
                        drives_to_failure=drives_to_failure
                    )
        
        # 2. Parameter Drift Detection (mean_deviation, danger_zone)
        # Monitor if parameters are drifting away from baseline
        current_mean_dev = all_params.iloc[drive_idx]['eff_mean_deviation']
        current_danger_pct = all_params.iloc[drive_idx]['eff_percent_in_danger_zone']
        
        # Parameter drift: mean deviation significantly worse than baseline
        # Relaxed threshold: 1.5 sigma instead of 2 sigma for earlier detection
        if current_mean_dev > baseline_mean_dev + 1.5 * baseline_std_dev:
            # Only flag if health score is still above threshold (early stage)
            if current_score > self.health_threshold:
                return AnomalyDetection(
                    drive_id=drive_id,
                    health_score=current_score,
                    is_anomalous=True,
                    anomaly_type='parameter_drift',
                    confidence=0.8,
                    severity='medium',
                    message='Cooling behavior deviating from baseline thermal profile',
                    risk_index=risk_index,
                    drives_to_failure=drives_to_failure
                )
        
        # Danger zone exposure: significant time spent in danger zone
        if current_danger_pct > 10:  # More than 10% time in danger zone
            if current_score > self.health_threshold:
                return AnomalyDetection(
                    drive_id=drive_id,
                    health_score=current_score,
                    is_anomalous=True,
                    anomaly_type='danger_zone',
                    confidence=0.85,
                    severity='medium',
                    message=f'Danger Zone: {current_danger_pct:.1f}% time above 105°C',
                    risk_index=risk_index,
                    drives_to_failure=drives_to_failure
                )
        
        # 3. SPC Control Chart (mean ± 3σ) - Statistical confirmation
        lower_limit, upper_limit = self._calculate_control_limits(baseline_scores)
        if current_score < lower_limit:
            return AnomalyDetection(
                drive_id=drive_id,
                health_score=current_score,
                is_anomalous=True,
                anomaly_type='spc',
                confidence=0.85,
                severity='medium',
                message=f'SPC anomaly: score {current_score:.1f} below 3-sigma limit {lower_limit:.1f}',
                risk_index=risk_index,
                drives_to_failure=drives_to_failure
            )
        
        # 4. Threshold Detection (last stage, not first)
        if current_score < self.health_threshold:
            severity = 'high' if current_score < 40 else 'medium'
            return AnomalyDetection(
                drive_id=drive_id,
                health_score=current_score,
                is_anomalous=True,
                anomaly_type='threshold',
                confidence=0.9,
                severity=severity,
                message=f'Threshold breach: score {current_score:.1f} below {self.health_threshold}',
                risk_index=risk_index,
                drives_to_failure=drives_to_failure
            )
        
        # No anomaly detected
        return AnomalyDetection(
            drive_id=drive_id,
            health_score=current_score,
            is_anomalous=False,
            anomaly_type=None,
            confidence=0.0,
            severity='low',
            message='Normal operation',
            risk_index=risk_index,
            drives_to_failure=drives_to_failure
        )
    
    def analyze_all_drives(self, params_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze all drives for anomalies and trends using automotive-style predictive maintenance.
        
        Returns DataFrame with original parameters plus anomaly detection results,
        including risk_index and drives_to_failure.
        """
        # Isolation Forest is NOT used as primary trigger to avoid false positives
        # Only fit if needed for supporting evidence (not implemented here)
        
        health_scores = params_df['health_score'].values
        results = []
        
        for i in range(len(params_df)):
            detection = self.analyze_drive(i, health_scores, params_df)
            results.append({
                'drive_id': detection.drive_id,
                'health_score': detection.health_score,
                'is_anomalous': detection.is_anomalous,
                'anomaly_type': detection.anomaly_type,
                'confidence': detection.confidence,
                'severity': detection.severity,
                'message': detection.message,
                'risk_index': detection.risk_index,
                'drives_to_failure': detection.drives_to_failure
            })
        
        # Merge with original parameters
        anomaly_df = pd.DataFrame(results)
        result_df = pd.concat([params_df.reset_index(drop=True), anomaly_df], axis=1)
        
        # Remove duplicate columns
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        
        return result_df
    
    def get_degradation_start_point(self, result_df: pd.DataFrame) -> Optional[int]:
        """
        Identify the drive where degradation first became detectable.
        
        Returns the drive index where degradation was first flagged, or None if no degradation.
        """
        anomalous_drives = result_df[result_df['is_anomalous'] == True]
        
        if len(anomalous_drives) == 0:
            return None
        
        # Return the first anomalous drive
        first_anomaly_idx = anomalous_drives.index[0]
        return first_anomaly_idx


if __name__ == '__main__':
    # Test with synthetic data
    from data_ingestion import generate_synthetic_obd_data
    from parameter_extraction import CoolingSystemParameters
    
    # Generate data with degradation
    drives = generate_synthetic_obd_data(num_drives=20, samples_per_drive=200, degradation_start=12)
    
    # Extract parameters
    extractor = CoolingSystemParameters(baseline_window=5)
    params_df = extractor.extract_all_drives(drives)
    
    # Analyze trends
    analyzer = TrendAnalyzer(min_drives_for_trend=5, health_threshold=65.0)
    result_df = analyzer.analyze_all_drives(params_df)
    
    print("Trend Analysis Results:")
    print(result_df[['drive_id', 'health_score', 'is_anomalous', 'anomaly_type', 'message']].tail(10))
    
    degradation_point = analyzer.get_degradation_start_point(result_df)
    print(f"\nDegradation first detected at drive index: {degradation_point}")
