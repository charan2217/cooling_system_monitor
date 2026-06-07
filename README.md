

# Cooling System Health Monitor

An automotive predictive maintenance system that detects cooling system degradation from OBD-II telemetry data using a 3-layer detection pipeline.

## Overview

This system analyzes vehicle cooling system health by processing OBD-II telemetry across multiple drives. It uses physics-informed statistical parameters normalized for driving conditions to detect early signs of degradation before threshold breaches occur.

### Key Features

- **3-Layer Predictive Maintenance System**
  - Layer 1: Health Score (0-100 scale for current condition)
  - Layer 2: Degradation Risk Index (0-1 scale for predictive risk)
  - Layer 3: Drives to Failure (estimated remaining drives until threshold breach)

- **Multi-Stage Detection Pipeline**
  1. Trend Detection (early warning based on last 5 scores)
  2. Parameter Drift (thermal profile deviation)
  3. SPC Control Chart (statistical confirmation)
  4. Threshold Detection (final stage)

- **Physics-Informed Parameters**
  - Normalized for driving conditions (city vs highway)
  - Accounts for engine load, ambient temperature, thermal inertia
  - Comparable across different operating conditions

## Project Structure

```
cooling_system_monitor/
├── main.py                      # Entry point and orchestration
├── data_ingestion.py            # OBD-II data loading and segmentation
├── parameter_extraction.py      # Health metric calculation
├── trend_analysis.py            # Anomaly detection engine
├── storage.py                   # Dual-storage system (CSV + JSON)
├── visualization.py             # Plotting and visualization
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker container configuration
├── .dockerignore                # Docker build exclusions
├── sample_obd_data.csv          # Sample OBD-II data for testing
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Local Installation

1. Clone or download the project:
```bash
cd cooling_system_monitor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using Sample Data (Synthetic)

Generate and process synthetic OBD-II data with injected degradation:

```bash
python main.py --synthetic --num-drives 20 --degradation-start 12 --output-dir ./output
```

**Parameters:**
- `--synthetic`: Use synthetic data instead of real CSV
- `--num-drives`: Number of drives to generate (default: 20)
- `--degradation-start`: Drive number where degradation begins (default: 12)
- `--output-dir`: Directory for output files (default: ./output)

### Option 2: Using Real OBD-II Data

Process your own OBD-II CSV file:

```bash
python main.py --input-csv your_data.csv --output-dir ./output
```

**CSV Format Requirements:**
The input CSV must contain the following columns:
- `timestamp`: ISO format datetime
- `rpm`: Engine RPM
- `coolant_temp`: Coolant temperature in Celsius
- `vehicle_speed`: Vehicle speed in km/h
- `throttle_pos`: Throttle position percentage (0-100)
- `intake_air_temp`: Intake air temperature in Celsius

### Option 3: Using Docker

#### Build the Docker Image

```bash
docker build -t cooling-system-monitor .
```

#### Run with Sample Data

```bash
docker run -v "${PWD}\output:/app/output" cooling-system-monitor
```

#### Run with Custom Data

```bash
docker run -v "${PWD}\your_data.csv:/app/data.csv" -v "${PWD}\output:/app/output" cooling-system-monitor python main.py --input-csv data.csv --output-dir /app/output
```

**Note:** On Windows Command Prompt, use `%cd%` instead of `${PWD}`.

## Output Files

After running the system, the following files are generated in the output directory:

### 1. Health Log CSV (`{vehicle_id}_health_log.csv`)
Complete analysis history with columns:
- `drive_id`: Unique drive identifier
- `timestamp`: Drive start time
- `health_score`: Overall health score (0-100)
- `is_anomalous`: Boolean flag for anomaly detection
- `anomaly_type`: Type of anomaly detected (early_warning, parameter_drift, danger_zone, spc, threshold)
- `confidence`: Detection confidence (0-1)
- `severity`: Severity level (low, medium, high)
- `message`: Human-readable detection message
- `risk_index`: Degradation risk index (0-1)
- `drives_to_failure`: Estimated drives until threshold breach
- Additional cooling system parameters

### 2. State Snapshot JSON (`{vehicle_id}_state.json`)
Current vehicle state summary:
```json
{
  "vehicle_id": "vehicle_001",
  "last_updated": "2024-01-05T20:50:30",
  "total_drives": 20,
  "current_health_score": 54.3,
  "baseline_health_score": 80.1,
  "health_trend": "degrading",
  "last_anomaly": "SPC anomaly: score 54.3 below 3-sigma limit",
  "degradation_detected": true,
  "degradation_drive_index": 10,
  "risk_index": 0.308,
  "drives_to_failure": 0
}
```

### 3. Health Plot PNG (`{vehicle_id}_health_plot.png`)
3-panel visualization showing:
- Health score trend with anomaly markers
- Risk index over time
- Key cooling system parameters

## Detection Pipeline

The system uses a prioritized detection pipeline to identify degradation:

1. **Trend Detection** (Early Warning)
   - Analyzes last 5 health scores
   - Triggers if: slope < -1.5 AND R² > 0.5
   - Detects degradation while still above threshold

2. **Parameter Drift** (Confirmation)
   - Monitors mean temperature deviation from baseline
   - Triggers if: deviation > baseline + 1.5σ
   - Monitors danger zone exposure (>10% time above 105°C)

3. **SPC Control Chart** (Statistical Confirmation)
   - Uses 3-sigma control limits
   - Triggers if: score below lower control limit

4. **Threshold Detection** (Final Stage)
   - Triggers if: health_score < 65
   - Severity: high if <40, medium if 40-65

## Technical Details

### Health Score Calculation

The health score (0-100) is calculated by comparing current parameters to baseline behavior:
- Baseline established from first 5 drives
- Z-score normalization for each parameter
- Weighted average of key parameters:
  - Mean temperature deviation (weight: 1.5)
  - Temperature stability (weight: 1.2)
  - 95th percentile deviation (weight: 1.3)
  - Optimal range percentage (weight: 1.0)
  - Danger zone exposure (weight: 2.0)
  - Thermal response coefficient (weight: 1.0)
  - Thermal inertia (weight: 0.8)

### Risk Index Calculation

The degradation risk index (0-1) combines:
- Health score slope (40% weight)
- Temperature deviation trend (20% weight)
- Danger zone trend (20% weight)
- Thermal instability (20% weight)

### Drives to Failure Estimation

Estimated drives until health score crosses threshold:
- Linear regression extrapolation
- Heuristic estimates when trend is not significant
- Capped at 50 drives maximum

## Dependencies

- numpy>=1.23.0
- pandas>=1.5.0
- scipy>=1.9.0
- matplotlib>=3.5.0

## Example Output

```
Generating synthetic data with 20 drives
Degradation starts at drive 12
Extracting cooling system parameters...
Analyzing trends and detecting anomalies...
Degradation detected at drive index: 10 (expected: 11)
Storing results...
Creating visualization...
Plot saved to: output\vehicle_001_health_plot.png

Results saved to:
  - CSV log: output\vehicle_001_health_log.csv
  - JSON snapshot: output\vehicle_001_state.json
  - Health plot: output\vehicle_001_health_plot.png

Summary:
  Total drives analyzed: 20
  Current health score: 54.3
  Anomalous drives: 9
  Degradation detected: True
```

## Contact

For questions or issues, please refer to the project documentation or contact the development team.
