import requests
import pandas as pd
import os
import time
from datetime import datetime

# Configuration
LAT, LON = 42.6629, 21.1655  # Prishtina coordinates

# Create organized folder structure
TRAFFIC_DIR = "data/raw/traffic"
WEATHER_DIR = "data/raw/weather"
os.makedirs(TRAFFIC_DIR, exist_ok=True)
os.makedirs(WEATHER_DIR, exist_ok=True)

# Months to download 
MONTHS = [
    ("2025-03-01", "2025-03-31"),
    ("2025-04-01", "2025-04-30"), 
    ("2025-05-01", "2025-05-31"),
    ("2025-06-01", "2025-06-30"),
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-30"),
    ("2025-10-01", "2025-10-31"), 
]

def download_weather_data():
    """Download weather data from Open-Meteo API"""
    print("Starting weather download...")
    
    for start, end in MONTHS:
        month = start[:7]
        URL = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={LAT}&longitude={LON}"
            f"&start_date={start}&end_date={end}"
            "&hourly=temperature_2m,precipitation"
        )

        try:
            response = requests.get(URL, timeout=60)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame({
                "time": data["hourly"]["time"],
                "temperature": data["hourly"]["temperature_2m"],
                "precipitation": data["hourly"]["precipitation"],
            })

            filename = os.path.join(WEATHER_DIR, f"weather_{month}.csv")
            df.to_csv(filename, index=False)
            print(f"Weather {month}: {len(df)} records")

            time.sleep(1) 

        except Exception as e:
            print(f"Weather {month}: {e}")

def download_traffic_data():
    """Download traffic data from IntraTraffic API"""
    print("Starting traffic download...")
    
    for start, end in MONTHS:
        URL = f"https://intrastraffic.com/api/traffic-data?from={start}T00:00:00.000Z&to={end}T23:59:59.999Z"
        
        try:
            response = requests.get(URL, timeout=60)
            response.raise_for_status()  # Check for HTTP errors
            data = response.json()
            df = pd.DataFrame(data)
            
            filename = os.path.join(TRAFFIC_DIR, f"traffic_{start[:7]}.csv")
            df.to_csv(filename, index=False)
            print(f"Traffic {start[:7]}: {len(df)} rows")
            
        except Exception as e:
            print(f"Traffic {start[:7]}: {e}")

def main():
    """Main function to run both downloads"""
    print("Starting data download process...")
    print("-"*50)
    
    # Download both datasets
    download_weather_data()
    print("-" * 30)
    download_traffic_data()
    
    print("\All downloads complete!")
    print(f"Check files in:{os.path.abspath(TRAFFIC_DIR)} and {os.path.abspath(WEATHER_DIR)}")

if __name__ == "__main__":
    main()