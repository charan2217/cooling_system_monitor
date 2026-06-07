"""
Main entry point for Cooling System Health Monitor.
Orchestrates data ingestion, parameter extraction, trend analysis, and storage.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from data_ingestion import DataIngestion, DriveData, generate_synthetic_obd_data
from parameter_extraction import CoolingSystemParameters
from trend_analysis import TrendAnalyzer
from storage import DualStorage
from visualization import create_health_plot


def process_real_data(input_csv: str, output_dir: str = './output', vehicle_id: str = 'vehicle_001'):
    """
    Process real OBD-II data from CSV file.
    
    Args:
        input_csv: Path to input CSV file with OBD-II telemetry
        output_dir: Directory for output files
        vehicle_id: Unique vehicle identifier
    """
    print(f"Processing real data from: {input_csv}")
    
    # Ingest and segment data
    ingestion = DataIngestion(time_gap_threshold=300.0)
    drives = ingestion.load_and_segment(input_csv)
    
    print(f"Segmented into {len(drives)} drives")
    
    # Validate drives
    valid_drives = [d for d in drives if ingestion.validate_drive(d)]
    print(f"Valid drives: {len(valid_drives)}/{len(drives)}")
    
    if len(valid_drives) < 5:
        print("Warning: Fewer than 5 valid drives. Results may be unreliable.")
    
    # Extract parameters
    print("Extracting cooling system parameters...")
    extractor = CoolingSystemParameters(baseline_window=min(5, len(valid_drives)))
    params_df = extractor.extract_all_drives(valid_drives)
    
    # Analyze trends using 3-layer predictive maintenance system
    print("Analyzing trends and detecting anomalies...")
    analyzer = TrendAnalyzer(
        min_drives_for_trend=min(5, len(valid_drives)),
        health_threshold=65.0
    )
    result_df = analyzer.analyze_all_drives(params_df)
    
    # Detect degradation start point
    degradation_idx = analyzer.get_degradation_start_point(result_df)
    if degradation_idx is not None:
        print(f"Degradation detected at drive index: {degradation_idx}")
    else:
        print("No degradation detected")
    
    # Store results
    print("Storing results...")
    storage = DualStorage(output_dir=output_dir, vehicle_id=vehicle_id)
    storage.save_results(result_df, degradation_idx)
    
    # Create visualization
    print("Creating visualization...")
    plot_path = Path(output_dir) / f'{vehicle_id}_health_plot.png'
    create_health_plot(result_df, degradation_idx, str(plot_path))
    
    print(f"\nResults saved to:")
    print(f"  - CSV log: {storage.csv_log_path}")
    print(f"  - JSON snapshot: {storage.json_snapshot_path}")
    print(f"  - Health plot: {plot_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total drives analyzed: {len(result_df)}")
    print(f"  Current health score: {result_df['health_score'].iloc[-1]:.1f}")
    print(f"  Anomalous drives: {result_df['is_anomalous'].sum()}")
    print(f"  Degradation detected: {degradation_idx is not None}")


def process_synthetic_data(num_drives: int = 20, samples_per_drive: int = 200, 
                          degradation_start: int = 12, output_dir: str = './output',
                          vehicle_id: str = 'vehicle_001'):
    """
    Process synthetic OBD-II data with injected degradation.
    
    Args:
        num_drives: Number of drives to generate
        samples_per_drive: Average samples per drive
        degradation_start: Drive number where degradation begins
        output_dir: Directory for output files
        vehicle_id: Unique vehicle identifier
    """
    print(f"Generating synthetic data with {num_drives} drives")
    print(f"Degradation starts at drive {degradation_start}")
    
    # Generate synthetic data
    drives = generate_synthetic_obd_data(
        num_drives=num_drives,
        samples_per_drive=samples_per_drive,
        degradation_start=degradation_start
    )
    
    # Extract parameters
    print("Extracting cooling system parameters...")
    extractor = CoolingSystemParameters(baseline_window=min(5, len(drives)))
    params_df = extractor.extract_all_drives(drives)
    
    # Analyze trends
    print("Analyzing trends and detecting anomalies...")
    analyzer = TrendAnalyzer(
        min_drives_for_trend=min(5, len(drives)),
        health_threshold=65.0
    )
    result_df = analyzer.analyze_all_drives(params_df)
    
    # Detect degradation start point
    degradation_idx = analyzer.get_degradation_start_point(result_df)
    if degradation_idx is not None:
        print(f"Degradation detected at drive index: {degradation_idx} (expected: {degradation_start - 1})")
    else:
        print("No degradation detected")
    
    # Store results
    print("Storing results...")
    storage = DualStorage(output_dir=output_dir, vehicle_id=vehicle_id)
    storage.save_results(result_df, degradation_idx)
    
    # Create visualization
    print("Creating visualization...")
    plot_path = Path(output_dir) / f'{vehicle_id}_health_plot.png'
    create_health_plot(result_df, degradation_idx, str(plot_path))
    
    print(f"\nResults saved to:")
    print(f"  - CSV log: {storage.csv_log_path}")
    print(f"  - JSON snapshot: {storage.json_snapshot_path}")
    print(f"  - Health plot: {plot_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total drives analyzed: {len(result_df)}")
    print(f"  Current health score: {result_df['health_score'].iloc[-1]:.1f}")
    print(f"  Anomalous drives: {result_df['is_anomalous'].sum()}")
    print(f"  Degradation detected: {degradation_idx is not None}")


def main():
    parser = argparse.ArgumentParser(
        description='Cooling System Health Monitor - Detect degradation from OBD-II telemetry'
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-csv',
        type=str,
        help='Path to input CSV file with OBD-II telemetry'
    )
    input_group.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data with injected degradation'
    )
    
    # Synthetic data options
    parser.add_argument(
        '--num-drives',
        type=int,
        default=20,
        help='Number of synthetic drives to generate (default: 20)'
    )
    parser.add_argument(
        '--samples-per-drive',
        type=int,
        default=200,
        help='Average samples per synthetic drive (default: 200)'
    )
    parser.add_argument(
        '--degradation-start',
        type=int,
        default=12,
        help='Drive number where degradation begins (default: 12)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Directory for output files (default: ./output)'
    )
    parser.add_argument(
        '--vehicle-id',
        type=str,
        default='vehicle_001',
        help='Unique vehicle identifier (default: vehicle_001)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.synthetic:
            process_synthetic_data(
                num_drives=args.num_drives,
                samples_per_drive=args.samples_per_drive,
                degradation_start=args.degradation_start,
                output_dir=args.output_dir,
                vehicle_id=args.vehicle_id
            )
        else:
            process_real_data(
                input_csv=args.input_csv,
                output_dir=args.output_dir,
                vehicle_id=args.vehicle_id
            )
        
        print("\n✓ Processing complete")
        return 0
    
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
