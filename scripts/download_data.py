"""
NYC Traffic and NOAA Weather Data Downloader

Downloads traffic speed data from NYC Open Data and weather data from NOAA Climate API.
Saves raw data to data/raw/ directory for further processing.

Usage:
    python scripts/download_data.py --year 2023
    python scripts/download_data.py --test --limit 1000
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
NYC_TRAFFIC_BASE_URL = "https://data.cityofnewyork.us/resource/i4gi-tjb9.csv"
NOAA_API_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
NOAA_STATION_ID = "GHCND:USW00094728"  # JFK Airport, NYC
NOAA_DATASET_ID = "GHCND"  # Global Historical Climatology Network Daily

# Rate limiting
NOAA_RATE_LIMIT_DELAY = 0.2  # 5 requests per second max


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def download_nyc_traffic(
    start_date: str,
    end_date: str,
    limit: int = 50000,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Download NYC traffic speed data from NYC Open Data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        limit: Maximum number of records to download
        output_path: Path to save the CSV file
        
    Returns:
        DataFrame with traffic data
    """
    logger.info(f"Downloading NYC traffic data from {start_date} to {end_date}")
    
    all_data = []
    offset = 0
    batch_size = min(limit, 50000)
    
    with tqdm(desc="Downloading NYC Traffic Data", unit=" records") as pbar:
        while True:
            # Build query with SoQL
            params = {
                "$where": f"data_as_of >= '{start_date}T00:00:00' AND data_as_of <= '{end_date}T23:59:59'",
                "$limit": batch_size,
                "$offset": offset,
                "$order": "data_as_of ASC"
            }
            
            try:
                response = requests.get(NYC_TRAFFIC_BASE_URL, params=params, timeout=60)
                response.raise_for_status()
                
                # Parse CSV response
                from io import StringIO
                batch_df = pd.read_csv(StringIO(response.text))
                
                if batch_df.empty:
                    break
                    
                all_data.append(batch_df)
                records_fetched = len(batch_df)
                pbar.update(records_fetched)
                
                # Check if we've hit the limit or got less than batch size
                total_records = sum(len(df) for df in all_data)
                if total_records >= limit or records_fetched < batch_size:
                    break
                    
                offset += batch_size
                
            except requests.RequestException as e:
                logger.error(f"Error downloading NYC traffic data: {e}")
                if all_data:
                    logger.warning("Returning partial data")
                    break
                raise
    
    if not all_data:
        logger.warning("No NYC traffic data found for the specified date range")
        return pd.DataFrame()
    
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Downloaded {len(df)} NYC traffic records")
    
    # Save to file if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved NYC traffic data to {output_path}")
    
    return df


def download_noaa_weather(
    start_date: str,
    end_date: str,
    token: str,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Download NOAA weather data from Climate Data Online API.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        token: NOAA API token
        output_path: Path to save the CSV file
        
    Returns:
        DataFrame with weather data
    """
    logger.info(f"Downloading NOAA weather data from {start_date} to {end_date}")
    
    if not token:
        raise ValueError("NOAA API token is required. Set NOAA_API_TOKEN environment variable.")
    
    headers = {"token": token}
    all_data = []
    
    # NOAA API limits date range to 1 year, so we need to chunk if necessary
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Calculate number of days for progress bar
    total_days = (end_dt - start_dt).days + 1
    
    with tqdm(total=total_days, desc="Downloading NOAA Weather Data", unit=" days") as pbar:
        current_start = start_dt
        
        while current_start <= end_dt:
            # NOAA API allows max 1 year range
            current_end = min(current_start + timedelta(days=364), end_dt)
            
            offset = 0
            limit = 1000  # NOAA API max limit per request
            
            while True:
                params = {
                    "datasetid": NOAA_DATASET_ID,
                    "stationid": NOAA_STATION_ID,
                    "startdate": current_start.strftime("%Y-%m-%d"),
                    "enddate": current_end.strftime("%Y-%m-%d"),
                    "limit": limit,
                    "offset": offset,
                    "units": "metric"
                }
                
                try:
                    response = requests.get(
                        NOAA_API_BASE_URL,
                        headers=headers,
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code == 429:
                        logger.warning("Rate limited, waiting 1 second...")
                        time.sleep(1)
                        continue
                        
                    response.raise_for_status()
                    data = response.json()
                    
                    if "results" not in data or len(data["results"]) == 0:
                        break
                    
                    all_data.extend(data["results"])
                    
                    if len(data["results"]) < limit:
                        break
                        
                    offset += limit
                    time.sleep(NOAA_RATE_LIMIT_DELAY)
                    
                except requests.RequestException as e:
                    logger.error(f"Error downloading NOAA data: {e}")
                    if all_data:
                        logger.warning("Returning partial data")
                        break
                    raise
            
            days_processed = (current_end - current_start).days + 1
            pbar.update(days_processed)
            current_start = current_end + timedelta(days=1)
            time.sleep(NOAA_RATE_LIMIT_DELAY)
    
    if not all_data:
        logger.warning("No NOAA weather data found for the specified date range")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    logger.info(f"Downloaded {len(df)} NOAA weather records")
    
    # Pivot data to have one row per date with columns for each data type
    if not df.empty and 'datatype' in df.columns:
        df_pivot = df.pivot_table(
            index='date',
            columns='datatype',
            values='value',
            aggfunc='first'
        ).reset_index()
        df_pivot.columns.name = None
        df = df_pivot
    
    # Save to file if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved NOAA weather data to {output_path}")
    
    return df


def main():
    """Main entry point for data download script."""
    parser = argparse.ArgumentParser(
        description="Download NYC traffic and NOAA weather data"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Year to download data for (default: 2023)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format (overrides --year)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format (overrides --year)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000000,
        help="Maximum number of traffic records to download (default: 1000000)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with limited data"
    )
    parser.add_argument(
        "--traffic-only",
        action="store_true",
        help="Download only traffic data"
    )
    parser.add_argument(
        "--weather-only",
        action="store_true",
        help="Download only weather data"
    )
    
    args = parser.parse_args()
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        start_date = f"{args.year}-01-01"
        end_date = f"{args.year}-12-31"
    
    # Test mode uses limited data
    if args.test:
        args.limit = 1000
        # Use just one month for testing
        end_dt = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=30)
        end_date = end_dt.strftime("%Y-%m-%d")
        logger.info("Running in TEST mode with limited data")
    
    # Get project root and data paths
    project_root = get_project_root()
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get NOAA token
    noaa_token = os.getenv("NOAA_API_TOKEN")
    
    logger.info(f"Data will be saved to: {raw_data_dir}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Download NYC traffic data
    if not args.weather_only:
        try:
            traffic_path = raw_data_dir / f"nyc_traffic_{args.year}.csv"
            download_nyc_traffic(
                start_date=start_date,
                end_date=end_date,
                limit=args.limit,
                output_path=traffic_path
            )
        except Exception as e:
            logger.error(f"Failed to download NYC traffic data: {e}")
            if not args.traffic_only:
                logger.info("Continuing with weather download...")
    
    # Download NOAA weather data
    if not args.traffic_only:
        if not noaa_token:
            logger.error("NOAA_API_TOKEN not found in environment variables!")
            logger.error("Please create a .env file with: NOAA_API_TOKEN=your_token_here")
            sys.exit(1)
            
        try:
            weather_path = raw_data_dir / f"noaa_weather_{args.year}.csv"
            download_noaa_weather(
                start_date=start_date,
                end_date=end_date,
                token=noaa_token,
                output_path=weather_path
            )
        except Exception as e:
            logger.error(f"Failed to download NOAA weather data: {e}")
    
    logger.info("Data download complete!")
    logger.info(f"Files saved to: {raw_data_dir}")


if __name__ == "__main__":
    main()
