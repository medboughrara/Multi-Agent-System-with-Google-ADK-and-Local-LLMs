"""Test GPU monitoring functionality."""
from system_monitor import SystemMonitor
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_gpu_monitoring():
    # Initialize monitor
    monitor = SystemMonitor(interval=1.0)
    
    # Get system info
    sys_info = monitor.get_system_info()
    print("\nSystem Information:")
    for key, value in sys_info.items():
        print(f"{key}: {value}")
    
    # Start monitoring
    monitor.start_monitoring()
    
    print("\nMonitoring GPU for 5 seconds...")
    # Get usage every second for 5 seconds
    for _ in range(5):
        usage = monitor.get_current_usage()
        print("\nCurrent Usage:")
        for key, value in usage.items():
            print(f"{key}: {value:.2f}%")
        time.sleep(1)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Get history
    history = monitor.get_resource_history()
    print("\nResource History:")
    for key, values in history.items():
        if values:
            print(f"{key} average: {sum(values) / len(values):.2f}%")

if __name__ == "__main__":
    test_gpu_monitoring()
