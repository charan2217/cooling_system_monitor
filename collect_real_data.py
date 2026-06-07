"""
Collect real OBD-II data from a vehicle and save to CSV format.
Run this while driving to collect data for the cooling system monitor.
"""

import obd
import time
import pandas as pd
from datetime import datetime
import sys


class RealOBDCollector:
    """Collect real OBD-II data from a vehicle."""
    
    def __init__(self, port_str=None, interval=1.0):
        """
        Args:
            port_str: OBD port (e.g., 'COM10' on Windows, '/dev/ttyUSB0' on Linux)
                     If None, will auto-detect
            interval: Seconds between readings
        """
        self.interval = interval
        self.data = []
        
        # Connect to OBD adapter
        print("Connecting to OBD adapter...")
        if port_str:
            self.connection = obd.OBD(port_str)
        else:
            self.connection = obd.OBD()  # Auto-detect
        
        if not self.connection.is_connected():
            print("✗ Failed to connect to OBD adapter")
            print("  - Check that the adapter is plugged in")
            print("  - Check that the car ignition is ON")
            print("  - Try specifying the port explicitly")
            sys.exit(1)
        
        print(f"✓ Connected to OBD adapter on port: {self.connection.port_name()}")
    
    def _get_value(self, command):
        """Get value from OBD command."""
        response = self.connection.query(command)
        return response.value.magnitude if response.value is not None else None
    
    def collect_reading(self):
        """Collect a single reading from all sensors."""
        reading = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rpm': self._get_value(obd.commands.RPM),
            'coolant_temp': self._get_value(obd.commands.COOLANT_TEMP),
            'vehicle_speed': self._get_value(obd.commands.SPEED),
            'throttle_pos': self._get_value(obd.commands.THROTTLE_POS),
            'intake_air_temp': self._get_value(obd.commands.INTAKE_TEMP),
        }
        return reading
    
    def collect_duration(self, duration_minutes=10):
        """
        Collect data for a specified duration.
        
        Args:
            duration_minutes: How long to collect data (in minutes)
        """
        print(f"\nCollecting data for {duration_minutes} minutes...")
        print("Press Ctrl+C to stop early\n")
        
        duration_seconds = duration_minutes * 60
        start_time = time.time()
        reading_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                reading = self.collect_reading()
                self.data.append(reading)
                reading_count += 1
                
                # Print progress
                elapsed = time.time() - start_time
                remaining = duration_seconds - elapsed
                print(f"\rReading {reading_count:4d} | Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s | "
                      f"Temp: {reading['coolant_temp']:.1f}°C | RPM: {reading['rpm']:.0f}", end='')
                
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            print("\n\n✓ Data collection stopped by user")
        
        print(f"\n✓ Collected {len(self.data)} readings")
    
    def collect_driving_session(self, min_duration_minutes=5):
        """
        Collect data during a driving session.
        Automatically stops when the vehicle stops for more than 30 seconds.
        
        Args:
            min_duration_minutes: Minimum collection time
        """
        print(f"\nStarting driving session collection (minimum {min_duration_minutes} minutes)...")
        print("Start driving now. Data will stop automatically when you park.")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        reading_count = 0
        last_speed = 0
        stationary_start = None
        
        try:
            while True:
                reading = self.collect_reading()
                self.data.append(reading)
                reading_count += 1
                
                current_speed = reading['vehicle_speed'] if reading['vehicle_speed'] else 0
                
                # Detect if vehicle is stationary
                if current_speed < 1:  # Less than 1 km/h
                    if stationary_start is None:
                        stationary_start = time.time()
                    elif time.time() - stationary_start > 30:  # Stationary for 30 seconds
                        # Check minimum duration
                        if time.time() - start_time >= min_duration_minutes * 60:
                            print("\n\n✓ Vehicle stationary for 30 seconds - ending collection")
                            break
                else:
                    stationary_start = None
                
                # Print progress
                elapsed = time.time() - start_time
                print(f"\rReading {reading_count:4d} | Elapsed: {elapsed:.0f}s | "
                      f"Speed: {current_speed:.0f} km/h | Temp: {reading['coolant_temp']:.1f}°C | RPM: {reading['rpm']:.0f}", end='')
                
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            print("\n\n✓ Data collection stopped by user")
        
        print(f"\n✓ Collected {len(self.data)} readings")
    
    def save_to_csv(self, filename):
        """Save collected data to CSV file."""
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"✓ Data saved to: {filename}")
        return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect real OBD-II data from vehicle')
    parser.add_argument('--port', type=str, default=None,
                       help='OBD port (e.g., COM10 on Windows, /dev/ttyUSB0 on Linux)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Collection duration in minutes (default: 10)')
    parser.add_argument('--mode', type=str, choices=['duration', 'driving'], default='duration',
                       help='Collection mode: duration (fixed time) or driving (auto-stop when parked)')
    parser.add_argument('--output', type=str, default='obd_data.csv',
                       help='Output CSV filename (default: obd_data.csv)')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Seconds between readings (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create collector
    collector = RealOBDCollector(port_str=args.port, interval=args.interval)
    
    # Collect data
    if args.mode == 'duration':
        collector.collect_duration(duration_minutes=args.duration)
    else:
        collector.collect_driving_session(min_duration_minutes=args.duration)
    
    # Save to CSV
    df = collector.save_to_csv(args.output)
    
    # Show summary
    print(f"\nData Summary:")
    print(f"  Total readings: {len(df)}")
    print(f"  Duration: {len(df) * args.interval:.1f} seconds")
    print(f"  Avg coolant temp: {df['coolant_temp'].mean():.1f}°C")
    print(f"  Avg RPM: {df['rpm'].mean():.0f}")
    print(f"  Avg speed: {df['vehicle_speed'].mean():.1f} km/h")
    
    print(f"\nNext step: Run the cooling system monitor on this data:")
    print(f"  python main.py --input-csv {args.output} --output-dir ./output")


if __name__ == '__main__':
    main()
