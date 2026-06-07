"""
Visualization module for cooling system health monitoring.
Creates plots showing health parameter trends and degradation detection.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def create_health_plot(result_df: pd.DataFrame, 
                      degradation_idx: Optional[int],
                      output_path: str) -> None:
    """
    Create a plot showing health parameter trends and multi-stage degradation detection.
    
    Shows the 3-layer predictive maintenance system:
    - Layer 1: Health Score (current condition)
    - Layer 2: Risk Index (predictive)
    - Multi-stage detection: SPC → Trend → Change Point → Isolation Forest → Threshold
    
    Args:
        result_df: DataFrame with analysis results including health_score, risk_index, anomaly_type
        degradation_idx: Index where degradation was first detected
        output_path: Path to save the plot
    """
    # Set up the figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('3-Layer Predictive Maintenance System - Cooling System Health Monitor', 
                 fontsize=16, fontweight='bold')
    
    # Extract data
    drive_indices = range(len(result_df))
    health_scores = result_df['health_score'].values
    anomalous = result_df['is_anomalous'].values
    anomaly_types = result_df['anomaly_type'].values
    
    # Plot 1: Health Score Trend with Multi-Stage Detection
    ax1.plot(drive_indices, health_scores, 'o-', linewidth=2, markersize=6, 
             color='#2E86AB', label='Health Score')
    
    # Add threshold line
    threshold = 65.0
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, 
                label=f'Threshold ({threshold})')
    
    # Add warning zone (threshold + 10)
    warning_zone = threshold + 10
    ax1.axhline(y=warning_zone, color='orange', linestyle=':', linewidth=1, 
                label=f'Warning Zone ({warning_zone})')
    
    # Highlight different anomaly types with different markers
    anomaly_markers = {
        'early_warning': ('^', '#FFA500', 'Early Warning'),
        'parameter_drift': ('s', '#9B2335', 'Parameter Drift'),
        'danger_zone': ('d', '#8B4513', 'Danger Zone'),
        'spc': ('o', '#6B8E23', 'SPC Anomaly'),
        'threshold': ('X', '#FF0000', 'Threshold Breach')
    }
    
    for anomaly_type, (marker, color, label) in anomaly_markers.items():
        indices = [i for i, at in enumerate(anomaly_types) if at == anomaly_type]
        if indices:
            ax1.scatter(indices, health_scores[indices], 
                       color=color, s=120, marker=marker, zorder=5, 
                       label=label, edgecolors='black', linewidth=1)
    
    # Mark degradation point
    if degradation_idx is not None:
        ax1.axvline(x=degradation_idx, color='orange', linestyle=':', linewidth=2,
                   label=f'Degradation Start (Drive {degradation_idx + 1})')
        ax1.text(degradation_idx, health_scores[degradation_idx], '  DEGRADATION', 
                fontsize=10, fontweight='bold', color='orange',
                verticalalignment='bottom')
    
    ax1.set_xlabel('Drive Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Health Score (0-100)', fontsize=12, fontweight='bold')
    ax1.set_title('Layer 1: Health Score (Current Condition)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Add baseline region annotation
    baseline_window = min(5, len(result_df))
    ax1.axvspan(0, baseline_window, alpha=0.1, color='green', 
                label='Baseline Period')
    
    # Plot 2: Layer 2 - Risk Index
    if 'risk_index' in result_df.columns:
        risk_indices = result_df['risk_index'].values
        ax2.plot(drive_indices, risk_indices, 'o-', linewidth=2, markersize=6,
                color='#A23B72', label='Degradation Risk Index')
        ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5,
                   label='High Risk Threshold (0.5)')
        ax2.fill_between(drive_indices, risk_indices, 0, 
                        where=risk_indices > 0.5, alpha=0.3, color='red')
        
        # Mark anomalous drives on risk index plot
        for anomaly_type, (marker, color, label) in anomaly_markers.items():
            indices = [i for i, at in enumerate(anomaly_types) if at == anomaly_type]
            if indices:
                ax2.scatter(indices, risk_indices[indices], 
                           color=color, s=100, marker=marker, zorder=5,
                           edgecolors='black', linewidth=1)
        
        if degradation_idx is not None:
            ax2.axvline(x=degradation_idx, color='orange', linestyle=':', linewidth=2)
        
        ax2.set_xlabel('Drive Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Risk Index (0-1)', fontsize=12, fontweight='bold')
        ax2.set_title('Layer 2: Degradation Risk Index (Predictive)', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])
    
    # Plot 3: Key Parameters Over Time
    # Mean deviation from expected temperature
    if 'eff_mean_deviation' in result_df.columns:
        mean_dev = result_df['eff_mean_deviation'].values
        ax3.plot(drive_indices, mean_dev, 's-', linewidth=2, markersize=5,
                color='#A23B72', label='Mean Temp Deviation (°C)')
    
    # Percent in danger zone
    if 'eff_percent_in_danger_zone' in result_df.columns:
        danger_pct = result_df['eff_percent_in_danger_zone'].values
        ax3_twin = ax3.twinx()
        ax3_twin.plot(drive_indices, danger_pct, '^-', linewidth=2, markersize=5,
                     color='#F18F01', label='% Time in Danger Zone')
        ax3_twin.set_ylabel('% Time in Danger Zone', fontsize=12, fontweight='bold', 
                           color='#F18F01')
        ax3_twin.tick_params(axis='y', labelcolor='#F18F01')
        ax3_twin.set_ylim([0, max(10, danger_pct.max() * 1.2)])
    
    # Mark degradation point on third plot
    if degradation_idx is not None:
        ax3.axvline(x=degradation_idx, color='orange', linestyle=':', linewidth=2)
    
    ax3.set_xlabel('Drive Number', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Temperature Deviation (°C)', fontsize=12, fontweight='bold',
                  color='#A23B72')
    ax3.set_title('Key Cooling System Parameters', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='#A23B72')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def create_detailed_plot(result_df: pd.DataFrame,
                        degradation_idx: Optional[int],
                        output_path: str) -> None:
    """
    Create a more detailed multi-panel plot showing multiple parameters.
    
    Args:
        result_df: DataFrame with analysis results
        degradation_idx: Index where degradation was first detected
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Cooling System Health Monitor - Detailed Analysis', 
                 fontsize=16, fontweight='bold')
    
    drive_indices = range(len(result_df))
    
    # Panel 1: Health Score
    ax = axes[0, 0]
    ax.plot(drive_indices, result_df['health_score'].values, 'o-', 
            linewidth=2, markersize=5, color='#2E86AB')
    ax.axhline(y=65.0, color='red', linestyle='--', linewidth=1.5)
    if degradation_idx is not None:
        ax.axvline(x=degradation_idx, color='orange', linestyle=':', linewidth=2)
    ax.set_ylabel('Health Score', fontweight='bold')
    ax.set_title('Health Score Trend', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Panel 2: Mean Temperature Deviation
    ax = axes[0, 1]
    if 'eff_mean_deviation' in result_df.columns:
        ax.plot(drive_indices, result_df['eff_mean_deviation'].values, 's-',
                linewidth=2, markersize=5, color='#A23B72')
    if degradation_idx is not None:
        ax.axvline(x=degradation_idx, color='orange', linestyle=':', linewidth=2)
    ax.set_ylabel('Mean Deviation (°C)', fontweight='bold')
    ax.set_title('Temperature Deviation from Expected', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Standard Deviation
    ax = axes[1, 0]
    if 'eff_std_deviation' in result_df.columns:
        ax.plot(drive_indices, result_df['eff_std_deviation'].values, '^-',
                linewidth=2, markersize=5, color='#F18F01')
    if degradation_idx is not None:
        ax.axvline(x=degradation_idx, color='orange', linestyle=':', linewidth=2)
    ax.set_ylabel('Std Deviation (°C)', fontweight='bold')
    ax.set_title('Temperature Stability', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Percent in Danger Zone
    ax = axes[1, 1]
    if 'eff_percent_in_danger_zone' in result_df.columns:
        ax.plot(drive_indices, result_df['eff_percent_in_danger_zone'].values, 'd-',
                linewidth=2, markersize=5, color='#C73E1D')
    if degradation_idx is not None:
        ax.axvline(x=degradation_idx, color='orange', linestyle=':', linewidth=2)
    ax.set_ylabel('% Time > 105°C', fontweight='bold')
    ax.set_title('Danger Zone Exposure', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 5: Response Coefficient
    ax = axes[2, 0]
    if 'resp_response_coefficient' in result_df.columns:
        ax.plot(drive_indices, result_df['resp_response_coefficient'].values, 'v-',
                linewidth=2, markersize=5, color='#6B705C')
    if degradation_idx is not None:
        ax.axvline(x=degradation_idx, color='orange', linestyle=':', linewidth=2)
    ax.set_ylabel('Response Coeff', fontweight='bold')
    ax.set_title('Thermal Response', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 6: Anomaly Flags
    ax = axes[2, 1]
    anomalous = result_df['is_anomalous'].astype(int).values
    colors = ['green' if a == 0 else 'red' for a in anomalous]
    ax.bar(drive_indices, anomalous, color=colors, alpha=0.7)
    if degradation_idx is not None:
        ax.axvline(x=degradation_idx, color='orange', linestyle=':', linewidth=2)
    ax.set_ylabel('Anomaly Flag', fontweight='bold')
    ax.set_title('Anomaly Detection', fontweight='bold')
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Normal', 'Anomalous'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add common x-label
    for ax in axes.flat:
        ax.set_xlabel('Drive Number', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed plot saved to: {output_path}")


if __name__ == '__main__':
    # Test with synthetic data
    from data_ingestion import generate_synthetic_obd_data
    from parameter_extraction import CoolingSystemParameters
    from trend_analysis import TrendAnalyzer
    
    # Generate data
    drives = generate_synthetic_obd_data(num_drives=20, samples_per_drive=200, degradation_start=12)
    
    # Extract parameters
    extractor = CoolingSystemParameters(baseline_window=5)
    params_df = extractor.extract_all_drives(drives)
    
    # Analyze trends
    analyzer = TrendAnalyzer(min_drives_for_trend=5, health_threshold=65.0)
    result_df = analyzer.analyze_all_drives(params_df)
    
    # Get degradation point
    degradation_idx = analyzer.get_degradation_start_point(result_df)
    
    # Create plots
    create_health_plot(result_df, degradation_idx, './test_output/test_health_plot.png')
    create_detailed_plot(result_df, degradation_idx, './test_output/test_detailed_plot.png')
