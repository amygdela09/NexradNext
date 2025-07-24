import os
import io
import gzip
import boto3
import requests
import numpy as np
import pyart
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from osgeo import gdal, osr
import matplotlib.pyplot as plt
from urllib.parse import urljoin
import tempfile
import logging
import json
import re
import shutil # For cleaning up temporary directories

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Create a dedicated directory for NEXRAD GeoTIFF outputs
NEXRAD_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), 'nexrad_data_output')
os.makedirs(NEXRAD_OUTPUT_DIR, exist_ok=True)
logger.info(f"NEXRAD GeoTIFFs will be stored in: {NEXRAD_OUTPUT_DIR}")

# --- Classes from provided files ---

class DataSource(Enum):
    """
    Enumeration for the different NEXRAD data sources.
    """
    AWS = "s3://noaa-nexrad-level2/"
    GOOGLE = "gs://gcp-public-data-nexrad-l2/"
    NCEI = "https://www.ncei.noaa.gov/data/nexrad-level-ii/access/"

class NexradDownloader:
    """
    A library for downloading and processing NEXRAD Level 2 data.
    (From nexrad_toolkit.py, with minor adjustments)
    """

    def __init__(self):
        self.session = requests.Session()
        # Initialize boto3 client only once
        if not hasattr(self, 's3_client'):
            try:
                self.s3_client = boto3.client('s3')
            except Exception as e:
                logger.error(f"Failed to initialize boto3 S3 client: {e}. AWS downloads may not work.")
                self.s3_client = None

    def get_available_sites(self):
        """
        Returns a comprehensive list of NEXRAD radar sites.
        This list is based on the official list from the National Weather Service.
        """
        return {
            'KABR': 'Aberdeen, SD', 'KABX': 'Albuquerque, NM', 'KAKQ': 'Norfolk/Wakefield, VA',
            'KAMA': 'Amarillo, TX', 'KAMX': 'Miami, FL', 'KAPX': 'Gaylord, MI',
            'KARX': 'La Crosse, WI', 'KATX': 'Seattle/Tacoma, WA', 'KBBX': 'Beale AFB, CA',
            'KBGM': 'Binghamton, NY', 'KBHX': 'Eureka, CA', 'KBIS': 'Bismarck, ND',
            'KBLX': 'Billings, MT', 'KBMX': 'Birmingham, AL', 'KBOX': 'Boston, MA',
            'KBRO': 'Brownsville, TX', 'KBUF': 'Buffalo, NY', 'KBYX': 'Key West, FL',
            'KCAE': 'Columbia, SC', 'KCBW': 'Caribou, ME', 'KCBX': 'Boise, ID',
            'KCCX': 'State College, PA', 'KCLE': 'Cleveland, OH', 'KCLX': 'Charleston, SC',
            'KCRP': 'Corpus Christi, TX', 'KCXX': 'Burlington, VT', 'KCYS': 'Cheyenne, WY',
            'KDAX': 'Sacramento, CA', 'KDDC': 'Dodge City, KS', 'KDFX': 'Laughlin AFB, TX',
            'KDGX': 'Jackson, MS', 'KDIX': 'Philadelphia, PA', 'KDLH': 'Duluth, MN',
            'KDMX': 'Des Moines, IA', 'KDOX': 'Dover AFB, DE', 'KDTX': 'Detroit, MI',
            'KDVN': 'Davenport, IA', 'KDYX': 'Dyess AFB, TX', 'KEAX': 'Kansas City, MO',
            'KEMX': 'Tucson, AZ', 'KENX': 'Albany, NY', 'KEOX': 'Fort Rucker, AL',
            'KEPZ': 'El Paso, TX', 'KESX': 'Las Vegas, NV', 'KEVX': 'Eglin AFB, FL',
            'KEWX': 'San Antonio, TX', 'KEYX': 'Edwards AFB, CA', 'KFCX': 'Roanoke, VA',
            'KFDR': 'Altus AFB, OK', 'KFFC': 'Atlanta, GA', 'KFSD': 'Sioux Falls, SD',
            'KFSX': 'Flagstaff, AZ', 'KFTG': 'Denver, CO', 'KFWD': 'Dallas/Fort Worth, TX',
            'KFWS': 'Dallas/Fort Worth, TX', 'KGGW': 'Glasgow, MT', 'KGJX': 'Grand Junction, CO',
            'KGLD': 'Goodland, KS', 'KGRB': 'Green Bay, WI', 'KGRK': 'Fort Hood, TX',
            'KGRR': 'Grand Rapids, MI', 'KGSP': 'Greenville/Spartanburg, SC', 'KGUA': 'Andersen AFB, Guam',
            'KGX': 'Minot AFB, ND', 'KGYX': 'Portland, ME', 'KHDX': 'Holloman AFB, NM',
            'KHGX': 'Houston, TX', 'KHNX': 'San Joaquin Valley, CA', 'KHPX': 'Fort Campbell, KY',
            'KHTX': 'Huntsville, AL', 'KICT': 'Wichita, KS', 'KICX': 'Cedar City, UT',
            'KILN': 'Cincinnati/Wilmington, OH', 'KILX': 'Lincoln, IL', 'KIND': 'Indianapolis, IN',
            'KINX': 'Tulsa, OK', 'KIWA': 'Phoenix, AZ', 'KIWX': 'Fort Wayne, IN',
            'KJAX': 'Jacksonville, FL', 'KJGX': 'Robins AFB, GA', 'KJKL': 'Jackson, KY',
            'KLBB': 'Lubbock, TX', 'KLCH': 'Lake Charles, LA', 'KLIX': 'New Orleans, LA',
            'KLNX': 'North Platte, NE', 'KLOT': 'Chicago, IL', 'KLRX': 'Elko, NV',
            'KLSX': 'St. Louis, MO', 'KLTX': 'Wilmington, NC', 'KLVX': 'Las Vegas, NV',
            'KLWX': 'Sterling, VA', 'KMAF': 'Midland/Odessa, TX', 'KMAX': 'Medford, OR',
            'KMBX': 'Minot AFB, ND', 'KMHX': 'Morehead City, NC', 'KMKX': 'Milwaukee, WI',
            'KMLB': 'Melbourne, FL', 'KMOB': 'Mobile, AL', 'KMPX': 'Minneapolis/St. Paul, MN',
            'KMQT': 'Marquette, MI', 'KMRX': 'Knoxville, TN', 'KMSX': 'Missoula, MT',
            'KMTX': 'Salt Lake City, UT', 'KMUX': 'San Francisco, CA', 'KMVX': 'Grand Forks, ND',
            'KMWX': 'Minneapolis, MN', 'KNKX': 'San Diego, CA', 'KNQA': 'Memphis, TN',
            'KOAX': 'Omaha, NE', 'KOHX': 'Nashville, TN', 'KOKX': 'New York City, NY',
            'KOTX': 'Spokane, WA', 'KOUN': 'Norman, OK', 'KPAH': 'Paducah, KY',
            'KPBZ': 'Pittsburgh, PA', 'KPDT': 'Pendleton, OR', 'KPOE': 'Fort Polk, LA',
            'KPUX': 'Pueblo, CO', 'KRAX': 'Raleigh/Durham, NC', 'KRGX': 'Reno, NV',
            'KRIW': 'Riverton, WY', 'KRLX': 'Charleston, WV', 'KRTX': 'Portland, OR',
            'KSFX': 'Pocatello/Idaho Falls, ID', 'KSGF': 'Springfield, MO', 'KSHV': 'Shreveport, LA',
            'KSJT': 'San Angelo, TX', 'KSOX': 'Santa Ana Mountains, CA', 'KSRX': 'Fort Smith, AR',
            'KTBW': 'Tampa, FL', 'KTFX': 'Great Falls, MT', 'KTLH': 'Tallahassee, FL',
            'KTLX': 'Oklahoma City, OK', 'KTWX': 'Topeka, KS', 'KTYX': 'Montague, NY',
            'KUDX': 'Rapid City, SD', 'KUEX': 'Hastings, NE', 'KVNX': 'Vance AFB, OK',
            'KVTX': 'Roanoke, VA', 'KVWX': 'Evansville, IN', 'KYUX': 'Yuma, AZ'
        }

    def _get_aws_urls(self, station, start_time, end_time):
        """Generates file URLs for the AWS source."""
        urls = []
        current_time = start_time.replace(minute=0, second=0, microsecond=0) # Start from hour for listing
        while current_time <= end_time:
            prefix = f"{current_time.strftime('%Y/%m/%d')}/{station}/"
            if self.s3_client:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                try:
                    pages = paginator.paginate(Bucket='noaa-nexrad-level2', Prefix=prefix)
                    for page in pages:
                        for obj in page.get('Contents', []):
                            key = obj['Key']
                            if key.endswith('.gz') or not key.endswith('/'): # Exclude directories
                                # Extract time from filename (e.g., KTLX20240315_000000_V06.gz)
                                match = re.search(r'(\d{8})_(\d{6})_V06\.gz$', key) # Common pattern
                                if not match: # Try alternative common pattern
                                    match = re.search(r'(\d{8})_(\d{6})\.gz$', key)
                                if match:
                                    file_time_str = f"{match.group(1)}{match.group(2)}"
                                    try:
                                        file_time = datetime.strptime(file_time_str, "%Y%m%d%H%M%S")
                                        if start_time <= file_time <= end_time:
                                            urls.append(f"s3://noaa-nexrad-level2/{key}")
                                    except ValueError:
                                        logger.warning(f"Could not parse time from filename: {key}")
                                        continue
                except Exception as e:
                    logger.error(f"Error listing AWS S3 objects for {prefix}: {e}")
            current_time += timedelta(hours=1) # Iterate hourly
        return list(set(urls)) # Remove duplicates

    def _get_google_urls(self, station, start_time, end_time):
        """Generates file URLs for the Google Cloud source. (Simplified)"""
        # Note: A full implementation would use the Google Cloud Storage client library
        # to list files. This is a simplified example based on common patterns for public data.
        urls = []
        current_time = start_time.replace(minute=0, second=0, microsecond=0)
        while current_time <= end_time:
            # Google Cloud stores files in hourly tarballs or individual files
            # For direct access, individual files are easier. Tarballs require extraction.
            # This attempts common individual file patterns.
            for minute_step in range(0, 60, 5): # Check every 5 minutes
                check_time = current_time.replace(minute=minute_step)
                if start_time <= check_time <= end_time:
                    # Common file patterns on Google Cloud Storage
                    filename_v06_gz = f"{station}{check_time.strftime('%Y%m%d_%H%M%S')}_V06.gz"
                    filename_gz = f"{station}{check_time.strftime('%Y%m%d_%H%M%S')}.gz"
                    filename_no_ext = f"{station}{check_time.strftime('%Y%m%d_%H%M%S')}"

                    urls.append(f"gs://gcp-public-data-nexrad-l2/{check_time.year}/{check_time.month:02d}/{check_time.day:02d}/{station}/{filename_v06_gz}")
                    urls.append(f"gs://gcp-public-data-nexrad-l2/{check_time.year}/{check_time.month:02d}/{check_time.day:02d}/{station}/{filename_gz}")
                    urls.append(f"gs://gcp-public-data-nexrad-l2/{check_time.year}/{check_time.month:02d}/{check_time.day:02d}/{station}/{filename_no_ext}")
            current_time += timedelta(hours=1)
        return list(set(urls))

    def _get_ncei_urls(self, station, start_time, end_time):
        """Generates file URLs for the NCEI source."""
        urls = []
        current_time = start_time.replace(minute=0, second=0, microsecond=0)
        while current_time <= end_time:
            # NCEI has a predictable URL structure, but filenames can vary.
            # Check common scan times within each hour.
            for minute_step in range(0, 60, 5):
                check_time = current_time.replace(minute=minute_step)
                if start_time <= check_time <= end_time:
                    date_path = check_time.strftime('%Y%m')
                    day_path = check_time.strftime('%Y%m%d')
                    
                    file_name_base = f"{station}{check_time.strftime('%Y%m%d_%H%M%S')}"
                    possible_suffixes = ["", "_V06"]
                    possible_compressions = ["", ".gz"]

                    for suffix in possible_suffixes:
                        for comp in possible_compressions:
                            full_file_name = f"{file_name_base}{suffix}{comp}"
                            url = f"{DataSource.NCEI.value}{date_path}/{day_path}/{full_file_name}"
                            urls.append(url)
            current_time += timedelta(hours=1)
        return list(set(urls))

    def _download_file_from_url(self, url, output_path):
        """Downloads a single file from a given URL."""
        try:
            # Check if a decompressed version already exists
            if output_path.endswith('.gz'):
                decompressed_output_path = output_path[:-3]
                if os.path.exists(decompressed_output_path):
                    logger.info(f"Decompressed file already exists: {decompressed_output_path}")
                    return decompressed_output_path
            elif os.path.exists(output_path):
                 logger.info(f"File already exists: {output_path}")
                 return output_path

            logger.info(f"Attempting to download {url} to {output_path}")
            
            if url.startswith('s3://'):
                if not self.s3_client:
                    raise RuntimeError("Boto3 S3 client not initialized. Cannot download from AWS.")
                bucket_name = url.split('/')[2]
                key = '/'.join(url.split('/')[3:])
                self.s3_client.download_file(bucket_name, key, output_path)
            elif url.startswith('gs://'):
                # For Google Cloud Storage, direct HTTP access is often available for public data
                http_url = url.replace('gs://', 'https://storage.googleapis.com/')
                response = self.session.get(http_url, stream=True, timeout=30)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else: # Assume http/https for NCEI and other direct URLs
                response = self.session.get(url, stream=True, timeout=30)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Decompress if necessary
            if output_path.endswith('.gz'):
                decompressed_path = output_path[:-3]
                with gzip.open(output_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                os.remove(output_path) # Remove the compressed file
                logger.info(f"Decompressed {output_path} to {decompressed_path}")
                return decompressed_path
            return output_path
        except requests.exceptions.RequestException as e:
            logger.warning(f"HTTP/S error downloading {url}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
        except Exception as e:
            logger.warning(f"Failed to download {url} due to an unexpected error: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
            
    def download_time_series(self, source: DataSource, station: str, start_time: datetime, end_time: datetime, output_dir: str):
        """
        Downloads NEXRAD data for a time series from the specified source.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        urls = []
        if source == DataSource.AWS:
            urls = self._get_aws_urls(station, start_time, end_time)
        elif source == DataSource.GOOGLE:
            urls = self._get_google_urls(station, start_time, end_time)
        elif source == DataSource.NCEI:
            urls = self._get_ncei_urls(station, start_time, end_time)
        else:
            raise ValueError("Invalid data source specified.")

        downloaded_files = []
        unique_urls = sorted(list(set(urls))) # Sort to maintain some time order, remove duplicates
        
        max_workers = 5 # Limit parallel downloads to avoid overwhelming sources or local resources

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {}
            for url in unique_urls:
                # Create a unique filename for the downloaded raw file
                # Use a hash or a more robust naming convention if files might have same names from different sources/times
                file_name = os.path.basename(url.split('?')[0]) # Remove query parameters if any
                if file_name.endswith('.tar'): # Handle tar files from Google (will need extraction)
                    # For simplicity, we'll try to download .tar and assume it contains .gz inside
                    # A more complete solution would extract individual files from .tar
                    file_name = file_name.replace('.tar', '') # Treat tar as a container for now
                
                output_path_base = os.path.join(output_dir, file_name)
                
                # Check for existing decompressed or raw files before scheduling download
                if not (os.path.exists(output_path_base) or os.path.exists(output_path_base + '.gz')):
                    future_to_url[executor.submit(self._download_file_from_url, url, output_path_base + '.gz')] = url # Assume .gz for now

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result and os.path.exists(result): # Ensure the file actually exists
                        downloaded_files.append(result)
                except Exception as exc:
                    logger.error(f'{url} generated an exception during download: {exc}')

        logger.info(f"Finished downloading. Downloaded {len(downloaded_files)} raw files.")
        return downloaded_files


class NEXRADProcessor:
    """
    NEXRAD Level 2 Data Processor for GeoTIFF conversion.
    (Adapted from NexradDataProcessor.py, focusing on conversion)
    """

    def nexrad_to_geotiff(self, nexrad_file, output_geotiff, product='reflectivity'):
        """Convert NEXRAD Level 2 data to GeoTIFF"""
        logger.info(f"Processing NEXRAD file: {nexrad_file} for product: {product}")
        
        try:
            # Read NEXRAD data using PyART
            radar = pyart.io.read(nexrad_file)
            
            # Get the specified product (default: reflectivity)
            field_name = product
            
            # Check available fields
            available_fields = list(radar.fields.keys())
            
            if field_name not in available_fields:
                if 'reflectivity' in available_fields:
                    field_name = 'reflectivity'
                    logger.warning(f"Field '{product}' not found. Using 'reflectivity' instead.")
                elif 'velocity' in available_fields:
                    field_name = 'velocity'
                    logger.warning(f"Field '{product}' not found. Using 'velocity' instead.")
                elif available_fields:
                    field_name = available_fields[0]
                    logger.warning(f"Field '{product}' not found. Using first available field: '{field_name}'.")
                else:
                    raise ValueError("No valid radar fields found in the file.")
            
            # Get radar location
            radar_lat = radar.latitude['data'][0]
            radar_lon = radar.longitude['data'][0]
            
            logger.info(f"Radar location: {radar_lat:.4f}, {radar_lon:.4f}")
            
            # Get sweep data (use first sweep/elevation angle)
            sweep_idx = 0
            if radar.nsweeps == 0:
                raise ValueError("Radar object contains no sweeps.")
            if sweep_idx >= radar.nsweeps:
                logger.warning(f"Sweep index {sweep_idx} out of bounds. Using sweep 0.")
                sweep_idx = 0

            sweep_slice = radar.get_slice(sweep_idx)
            
            # Get data
            data = radar.fields[field_name]['data'][sweep_slice]
            azimuth = radar.azimuth['data'][sweep_slice]
            range_bins = radar.range['data']
            
            # Convert polar to cartesian coordinates
            az_rad = np.deg2rad(azimuth)
            range_2d, az_2d = np.meshgrid(range_bins, az_rad)
            
            # Calculate x, y coordinates (in meters from radar)
            x = range_2d * np.sin(az_2d)
            y = range_2d * np.cos(az_2d)
            
            # Create regular grid for interpolation
            max_range = np.max(range_bins)
            if max_range == 0:
                 logger.warning("Max range is zero, cannot create grid.")
                 return None
            grid_size = 500  # 500x500 grid for image output
            xi = np.linspace(-max_range, max_range, grid_size)
            yi = np.linspace(-max_range, max_range, grid_size)
            xi_2d, yi_2d = np.meshgrid(xi, yi)
            
            # Interpolate data to regular grid
            from scipy.interpolate import griddata
            
            # Flatten arrays for interpolation
            points = np.column_stack((x.flatten(), y.flatten()))
            values = data.flatten()
            
            # Remove masked/invalid data
            valid_mask = ~np.ma.is_masked(values) & np.isfinite(values)
            if not np.any(valid_mask):
                logger.warning(f"No valid data points found for interpolation in {nexrad_file}. Returning None.")
                return None
            
            points = points[valid_mask]
            values = values[valid_mask]
            
            # Interpolate
            grid_data = griddata(points, values, (xi_2d, yi_2d), method='linear', fill_value=np.nan)
            
            # Calculate pixel size in degrees (approximate)
            pixel_size_m = (2 * max_range) / grid_size
            # Rough conversion factors
            meters_per_degree_at_equator_lon = 111320
            meters_per_degree_lat = 110570

            # Adjust for latitude for longitude pixel size
            pixel_size_deg_lon = pixel_size_m / (meters_per_degree_at_equator_lon * np.cos(np.deg2rad(radar_lat)))
            pixel_size_deg_lat = pixel_size_m / meters_per_degree_lat
            
            # Create GeoTIFF
            driver = gdal.GetDriverByName('GTiff')
            
            # Create dataset
            dataset = driver.Create(
                output_geotiff,
                grid_size,
                grid_size,
                1, # Number of bands
                gdal.GDT_Float32, # Data type
                options=['COMPRESS=LZW', 'TILED=YES'] # Compression and tiling for better performance
            )
            
            # Set geotransform (defines pixel coordinates)
            # [top-left x, pixel width, rotation x, top-left y, rotation y, pixel height (negative)]
            # Top-left corner is calculated to center the radar at (0,0) in grid and then convert to lat/lon
            top_left_lon = radar_lon - (grid_size / 2) * pixel_size_deg_lon
            top_left_lat = radar_lat + (grid_size / 2) * pixel_size_deg_lat
            
            geotransform = [
                top_left_lon,
                pixel_size_deg_lon,
                0,
                top_left_lat,
                0,
                -pixel_size_deg_lat # Negative pixel height for GeoTIFFs (origin is top-left)
            ]
            
            dataset.SetGeoTransform(geotransform)
            
            # Set projection to WGS84
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)  # WGS84
            dataset.SetProjection(srs.ExportToWkt())
            
            # Write data
            band = dataset.GetRasterBand(1)
            # Ensure grid_data is not all NaNs, replace with a default if so to avoid GDAL error
            if np.all(np.isnan(grid_data)):
                logger.warning("Interpolated grid data is all NaNs for output. Setting to 0s.")
                band.WriteArray(np.zeros_like(grid_data, dtype=np.float32))
                band.SetNoDataValue(0) # Set NoDataValue explicitly if data is empty
            else:
                # Replace NaNs with a value outside the expected data range if you want them masked by client
                # Or set as NoDataValue if client supports it well.
                # For now, let's keep NaN and rely on client-side handling or set a specific NoData value
                band.WriteArray(np.flipud(grid_data).astype(np.float32))  # Flip vertically for correct orientation
                band.SetNoDataValue(np.nan) # Mark NaN values as no-data

            # Set metadata
            dataset.SetMetadataItem('DESCRIPTION', f'NEXRAD {field_name} data')
            dataset.SetMetadataItem('RADAR_SITE', radar.metadata.get('instrument_name', 'Unknown'))
            
            # Flush data to disk and close dataset
            dataset.FlushCache()
            dataset = None
            
            logger.info(f"GeoTIFF created: {output_geotiff}")
            
            return {
                'geotiff_path': output_geotiff,
                'radar_lat': float(radar_lat),
                'radar_lon': float(radar_lon),
                'field_name': field_name,
                'max_range_km': float(max_range / 1000),
                'data_min': float(np.nanmin(grid_data)) if not np.all(np.isnan(grid_data)) else None,
                'data_max': float(np.nanmax(grid_data)) if not np.all(np.isnan(grid_data)) else None
            }
            
        except Exception as e:
            logger.error(f"Error processing NEXRAD data to GeoTIFF for {nexrad_file}: {e}")
            # Ensure output_geotiff is removed if creation failed
            if os.path.exists(output_geotiff):
                os.remove(output_geotiff)
            return None


class SPCDataFetcher:
    """Handles fetching and parsing SPC outlook data (From OutlookBackend.py)"""
    
    BASE_URL = "https://www.spc.noaa.gov/products/outlook"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_outlook_url(self, date_str, outlook_type, day=1):
        """
        Construct SPC outlook URL
        
        Args:
            date_str: Date in YYYYMMDD format
            outlook_type: 'categorical', 'tornado', 'wind', 'hail'
            day: Outlook day (1-8)
        """
        year = date_str[:4]
        
        # Different URL patterns for different outlook types
        if outlook_type == 'categorical':
            return f"{self.BASE_URL}/archive/{year}/day{day}otlk_{date_str}_1200_cat.kml"
        else:
            return f"{self.BASE_URL}/archive/{year}/day{day}otlk_{date_str}_1200_{outlook_type}.kml"
    
    def fetch_kml_data(self, url):
        """Fetch KML data from SPC"""
        try:
            logger.info(f"Fetching data from: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching KML data from {url}: {e}")
            return None
    
    def parse_kml_to_geojson(self, kml_content, outlook_type):
        """
        Parse KML content and convert to GeoJSON format
        
        Args:
            kml_content: Raw KML content
            outlook_type: Type of outlook for proper styling
        """
        try:
            # Parse XML
            root = ET.fromstring(kml_content)
            
            # Define namespaces
            namespaces = {
                'kml': 'http://www.opengis.net/kml/2.2',
                'gx': 'http://www.google.com/kml/ext/2.2'
            }
            
            features = []
            
            # Find all Placemark elements
            placemarks = root.findall('.//kml:Placemark', namespaces)
            
            for placemark in placemarks:
                feature = self.parse_placemark(placemark, namespaces, outlook_type)
                if feature:
                    features.append(feature)
            
            return {
                "type": "FeatureCollection",
                "features": features
            }
            
        except ET.ParseError as e:
            logger.error(f"Error parsing KML: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing KML: {e}")
            return None
    
    def parse_placemark(self, placemark, namespaces, outlook_type):
        """Parse individual placemark to GeoJSON feature"""
        try:
            # Get name/description
            name_elem = placemark.find('kml:name', namespaces)
            desc_elem = placemark.find('kml:description', namespaces)
            
            name = name_elem.text if name_elem is not None else ""
            description = desc_elem.text if desc_elem is not None else ""
            
            # Extract risk level from name or description
            risk_level = self.extract_risk_level(name, description, outlook_type)
            
            # Find polygon coordinates
            polygon_elem = placemark.find('.//kml:Polygon', namespaces)
            if polygon_elem is not None:
                coords = self.parse_polygon_coordinates(polygon_elem, namespaces)
                if coords:
                    return {
                        "type": "Feature",
                        "properties": {
                            "name": name,
                            "description": description,
                            "risk_level": risk_level,
                            "outlook_type": outlook_type
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": coords
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing placemark: {e}")
            return None
    
    def parse_polygon_coordinates(self, polygon_elem, namespaces):
        """Parse polygon coordinates from KML"""
        try:
            # Find outer boundary coordinates
            outer_boundary = polygon_elem.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespaces)
            
            if outer_boundary is not None and outer_boundary.text:
                coord_text = outer_boundary.text.strip()
                coordinates = []
                
                # Parse coordinate pairs (lon,lat,alt format in KML)
                for coord_pair in coord_text.split():
                    parts = coord_pair.split(',')
                    if len(parts) >= 2:
                        lon, lat = float(parts[0]), float(parts[1])
                        coordinates.append([lon, lat])
                
                # Return as nested array for GeoJSON polygon format
                return [coordinates] if coordinates else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing coordinates: {e}")
            return None
    
    def extract_risk_level(self, name, description, outlook_type):
        """Extract risk level from name or description"""
        text = f"{name} {description}".upper()
        
        if outlook_type == 'categorical':
            risk_levels = ['HIGH', 'MDT', 'ENH', 'SLGT', 'MRGL', 'TSTM']
            for level in risk_levels:
                if level in text:
                    return level
        else:
            # For probability-based outlooks, look for percentages
            percent_match = re.search(r'(\d+)%', text)
            if percent_match:
                return percent_match.group(1)
        
        return "UNKNOWN"

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

nexrad_downloader = NexradDownloader()
nexrad_processor = NEXRADProcessor()
spc_fetcher = SPCDataFetcher()

# --- Flask Routes for SPC Outlooks (from OutlookBackend.py) ---

@app.route('/api/spc/outlook')
def get_outlook():
    """
    Get SPC outlook data for a specific date and type
    
    Query parameters:
    - date: Date in YYYY-MM-DD format
    - type: Outlook type (categorical, tornado, wind, hail)
    - day: Outlook day (default: 1)
    """
    try:
        # Get parameters
        date_param = request.args.get('date')
        outlook_type = request.args.get('type', 'categorical')
        day = int(request.args.get('day', 1))
        
        # Validate parameters
        if not date_param:
            return jsonify({"error": "Date parameter is required"}), 400
        
        if outlook_type not in ['categorical', 'tornado', 'wind', 'hail']:
            return jsonify({"error": "Invalid outlook type"}), 400
        
        # Convert date format
        try:
            date_obj = datetime.strptime(date_param, '%Y-%m-%d')
            date_str = date_obj.strftime('%Y%m%d')
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        # Check if date is too recent (data might not be available)
        if date_obj > datetime.now():
             return jsonify({"error": "Data not available for future dates"}), 400
        
        # Get SPC URL and fetch data
        url = spc_fetcher.get_outlook_url(date_str, outlook_type, day)
        kml_content = spc_fetcher.fetch_kml_data(url)
        
        if not kml_content:
            return jsonify({"error": f"Could not fetch data from SPC for {date_param} type {outlook_type}"}), 404
        
        # Parse KML to GeoJSON
        geojson = spc_fetcher.parse_kml_to_geojson(kml_content, outlook_type)
        
        if not geojson:
            return jsonify({"error": "Could not parse KML data"}), 500
        
        return jsonify({
            "success": True,
            "data": geojson,
            "source_url": url,
            "date": date_param,
            "outlook_type": outlook_type,
            "day": day
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in SPC outlook endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/spc/available-dates')
def get_available_spc_dates():
    """Get available dates for SPC data (last 30 days)"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        dates = []
        current_date = start_date
        
        while current_date <= end_date:
            dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        return jsonify({
            "success": True,
            "dates": dates
        })
        
    except Exception as e:
        logger.error(f"Error getting available SPC dates: {e}")
        return jsonify({"error": "Internal server error"}), 500

# --- Flask Routes for NEXRAD Radar Data ---

@app.route('/api/nexrad/stations')
def get_nexrad_stations():
    """Returns a list of available NEXRAD radar stations."""
    try:
        stations = nexrad_downloader.get_available_sites()
        return jsonify({"success": True, "stations": stations})
    except Exception as e:
        logger.error(f"Error getting NEXRAD stations: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/nexrad/available_times')
def get_nexrad_available_times():
    """
    Get available NEXRAD scan times for a given station and date.
    This generates a list of common scan times rather than querying exact file existence
    for performance reasons. For exact availability, `single_frame` endpoint should be used.
    """
    station = request.args.get('station')
    date_str = request.args.get('date')

    if not station or not date_str:
        return jsonify({"error": "Station and date parameters are required."}), 400

    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    times = []
    # Generate common 5-minute intervals throughout the day
    for hour in range(24):
        for minute in range(0, 60, 5): 
            dt_time = date_obj.replace(hour=hour, minute=minute, second=0, microsecond=0)
            times.append({
                "time": dt_time.strftime('%H:%M'),
                "timestamp": dt_time.timestamp()
            })
    
    return jsonify({"success": True, "times": sorted(times, key=lambda x: x['timestamp'])})


@app.route('/api/nexrad/single_frame')
def get_nexrad_single_frame():
    """
    Downloads and processes a single NEXRAD file into a GeoTIFF.
    Returns the URL to the generated GeoTIFF.
    """
    station = request.args.get('station')
    date_str = request.args.get('date')
    product = request.args.get('product', 'reflectivity')
    hour_str = request.args.get('hour')
    minute_str = request.args.get('minute')
    source_param = request.args.get('source', 'AWS') # Default to AWS

    if not all([station, date_str, hour_str, minute_str]):
        return jsonify({"error": "Station, date, hour, and minute parameters are required."}), 400

    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        hour = int(hour_str)
        minute = int(minute_str)
        scan_time = date_obj.replace(hour=hour, minute=minute, second=0, microsecond=0)
        source = DataSource[source_param.upper()]
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid date/time/source format: {e}"}), 400

    # Create a temporary directory for this specific download
    # This ensures isolation and easier cleanup for single files
    request_temp_dir = tempfile.mkdtemp(dir=NEXRAD_OUTPUT_DIR)

    try:
        downloaded_file_path = None
        # Attempt to get direct URL for the specific time and download
        # This part tries to build a highly specific URL based on common patterns
        # and then uses the general download function, which includes `requests.head` checks.
        
        # Build potential file name patterns for download attempt
        file_base_name = f"{station}{scan_time.strftime('%Y%m%d_%H%M%S')}"
        potential_raw_filepath = os.path.join(request_temp_dir, file_base_name + '.ar2v.gz') # Common PyART readable format
        
        # Use download_time_series for the actual download, limiting the range to ensure only one file
        downloaded_files = nexrad_downloader.download_time_series(
            source, station, scan_time, scan_time, output_dir=request_temp_dir
        )
        
        if downloaded_files:
            # Sort by proximity to requested time if multiple files were found
            closest_file = None
            min_diff_seconds = float('inf')
            for f_path in downloaded_files:
                try:
                    # Extract datetime from filename (e.g., KTLX20240520_120000_V06.gz -> 20240520_120000)
                    fname_match = re.search(r'(\d{8})_(\d{6})', os.path.basename(f_path))
                    if fname_match:
                        file_dt_str = f"{fname_match.group(1)}{fname_match.group(2)}"
                        file_dt_obj = datetime.strptime(file_dt_str, '%Y%m%d%H%M%S')
                        diff = abs((file_dt_obj - scan_time).total_seconds())
                        if diff < min_diff_seconds:
                            min_diff_seconds = diff
                            closest_file = f_path
                except ValueError:
                    logger.warning(f"Could not parse datetime from filename: {os.path.basename(f_path)}")
            downloaded_file_path = closest_file
        
        if not downloaded_file_path or not os.path.exists(downloaded_file_path):
            return jsonify({"error": f"Failed to download NEXRAD data for {station} on {date_str} at {hour:02d}:{minute:02d} from {source.name}. It might not be available or the specific timestamp is not found."}), 404

        # Process the downloaded file to GeoTIFF
        output_geotiff_name = f"{os.path.basename(downloaded_file_path).split('.')[0]}_{product}.tif"
        output_geotiff_path = os.path.join(NEXRAD_OUTPUT_DIR, output_geotiff_name) # Save to main output dir
        
        metadata = nexrad_processor.nexrad_to_geotiff(downloaded_file_path, output_geotiff_path, product)
        
        if not metadata:
            return jsonify({"error": "Failed to convert NEXRAD data to GeoTIFF."}), 500
        
        # Return URL to the GeoTIFF
        return jsonify({
            "success": True,
            "image_url": f"/nexrad_static/{os.path.basename(output_geotiff_path)}",
            "metadata": metadata
        })

    except Exception as e:
        logger.error(f"Error in single frame NEXRAD processing: {e}")
        return jsonify({"error": "Internal server error during NEXRAD processing."}), 500
    finally:
        # Clean up the temporary download directory
        if os.path.exists(request_temp_dir):
            shutil.rmtree(request_temp_dir)
            logger.info(f"Cleaned up temporary directory: {request_temp_dir}")


@app.route('/api/nexrad/animation_frames')
def get_nexrad_animation_frames():
    """
    Downloads and processes multiple NEXRAD files for animation.
    Returns a list of URLs to the generated GeoTIFFs.
    """
    station = request.args.get('station')
    date_str = request.args.get('date')
    product = request.args.get('product', 'reflectivity')
    start_time_str = request.args.get('start_time')
    end_time_str = request.args.get('end_time')
    source_param = request.args.get('source', 'AWS') # Default to AWS

    if not all([station, date_str, start_time_str, end_time_str]):
        return jsonify({"error": "Station, date, start_time, and end_time parameters are required."}), 400

    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        start_hour, start_minute = map(int, start_time_str.split(':'))
        end_hour, end_minute = map(int, end_time_str.split(':'))
        
        start_dt = date_obj.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        end_dt = date_obj.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
        source = DataSource[source_param.upper()]
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid date/time/source format: {e}"}), 400

    # Create a unique temporary directory for this animation batch's raw downloads
    animation_temp_dir = tempfile.mkdtemp(dir=NEXRAD_OUTPUT_DIR)

    try:
        # Download files for the specified time range
        downloaded_raw_files = nexrad_downloader.download_time_series(source, station, start_dt, end_dt, output_dir=animation_temp_dir)
        
        if not downloaded_raw_files:
            return jsonify({"error": f"No NEXRAD data found for {station} on {date_str} between {start_time_str} and {end_time_str} from {source.name}"}), 404

        geotiff_urls = []
        # Sort files by their timestamp to ensure animation order
        downloaded_raw_files.sort(key=lambda f: datetime.strptime(re.search(r'(\d{8})_(\d{6})', os.path.basename(f)).group(1) + re.search(r'(\d{8})_(\d{6})', os.path.basename(f)).group(2), '%Y%m%d%H%M%S') if re.search(r'(\d{8})_(\d{6})', os.path.basename(f)) else datetime.min)

        for i, raw_file_path in enumerate(downloaded_raw_files):
            try:
                # Extract original timestamp from filename for consistent naming
                fname = os.path.basename(raw_file_path)
                match = re.search(r'(\d{8})_(\d{6})', fname)
                timestamp_part = f"frame_{i}" # Default if regex fails
                if match:
                    timestamp_part = f"{match.group(1)}_{match.group(2)}"
                
                output_geotiff_name = f"{station}_{timestamp_part}_{product}.tif"
                output_geotiff_path = os.path.join(NEXRAD_OUTPUT_DIR, output_geotiff_name)
                
                metadata = nexrad_processor.nexrad_to_geotiff(raw_file_path, output_geotiff_path, product)
                
                if metadata:
                    geotiff_urls.append({
                        "url": f"/nexrad_static/{os.path.basename(output_geotiff_path)}",
                        "metadata": metadata,
                        "time": datetime.strptime(timestamp_part, '%Y%m%d_%H%M%S').strftime('%H:%M') if '_' in timestamp_part else None
                    })
            except Exception as proc_e:
                logger.error(f"Error processing {raw_file_path}: {proc_e}")
            
        if not geotiff_urls:
            return jsonify({"error": "No GeoTIFFs could be generated from the downloaded files."}), 500

        return jsonify({
            "success": True,
            "animation_frames": geotiff_urls
        })

    except Exception as e:
        logger.error(f"Error in NEXRAD animation processing: {e}")
        return jsonify({"error": "Internal server error during NEXRAD animation processing."}), 500
    finally:
        # Clean up the temporary raw download directory
        if os.path.exists(animation_temp_dir):
            shutil.rmtree(animation_temp_dir)
            logger.info(f"Cleaned up temporary directory: {animation_temp_dir}")


# --- Serve Static NEXRAD GeoTIFFs ---
@app.route('/nexrad_static/<path:filename>')
def serve_nexrad_static(filename):
    """Serve dynamically generated NEXRAD GeoTIFFs."""
    return send_from_directory(NEXRAD_OUTPUT_DIR, filename)

# --- Health Check and Root ---
@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        "name": "Weather Data API (SPC & NEXRAD)",
        "version": "1.0.0",
        "endpoints": {
            "/api/spc/outlook": {
                "method": "GET",
                "description": "Get SPC outlook data",
                "parameters": {
                    "date": "Date in YYYY-MM-DD format (required)",
                    "type": "Outlook type: categorical, tornado, wind, hail (default: categorical)",
                    "day": "Outlook day 1-8 (default: 1)"
                }
            },
            "/api/spc/available-dates": {
                "method": "GET",
                "description": "Get list of available SPC outlook dates"
            },
            "/api/nexrad/stations": {
                "method": "GET",
                "description": "Get list of available NEXRAD radar stations"
            },
            "/api/nexrad/available_times": {
                "method": "GET",
                "description": "Get available scan times for a NEXRAD station and date (approximate)",
                "parameters": {
                    "station": "NEXRAD station code (e.g., KTLX)",
                    "date": "Date in YYYY-MM-DD format"
                }
            },
            "/api/nexrad/single_frame": {
                "method": "GET",
                "description": "Get a single NEXRAD radar frame as GeoTIFF",
                "parameters": {
                    "station": "NEXRAD station code",
                    "date": "Date in YYYY-MM-DD format",
                    "hour": "Hour (0-23)",
                    "minute": "Minute (0-59)",
                    "product": "Radar product: reflectivity or velocity (default: reflectivity)",
                    "source": "Data source: AWS, GOOGLE, NCEI (default: AWS)"
                }
            },
            "/api/nexrad/animation_frames": {
                "method": "GET",
                "description": "Get multiple NEXRAD radar frames for animation as GeoTIFFs",
                "parameters": {
                    "station": "NEXRAD station code",
                    "date": "Date in YYYY-MM-DD format",
                    "start_time": "Start time in HH:MM format",
                    "end_time": "End time in HH:MM format",
                    "product": "Radar product: reflectivity or velocity (default: reflectivity)",
                    "source": "Data source: AWS, GOOGLE, NCEI (default: AWS)"
                }
            }
        },
        "notes": "Ensure required Python packages (pyart, scipy, boto3, requests, numpy, matplotlib, gdal) are installed."
    })

if __name__ == '__main__':
    # Clean up any residual old GeoTIFFs before starting
    if os.path.exists(NEXRAD_OUTPUT_DIR):
        shutil.rmtree(NEXRAD_OUTPUT_DIR)
        logger.info(f"Cleaned up existing NEXRAD output directory: {NEXRAD_OUTPUT_DIR}")
    os.makedirs(NEXRAD_OUTPUT_DIR, exist_ok=True) # Recreate fresh

    print(f"Starting Weather Data Backend Server. NEXRAD GeoTIFFs will be stored in: {NEXRAD_OUTPUT_DIR}")
    print("API will be available at: http://localhost:5000")
    print("\nTo use this with your HTML files:")
    print("1. Ensure your HTML files (NexradMapViewer.html, OutlookMapTest.html) are configured to fetch data from http://localhost:5000/api/...")
    print("2. For NexradMapViewer.html, you'll need to update its JavaScript functions (e.g., loadRadarData, generateAnimationFrames) to make actual fetch requests to /api/nexrad/single_frame and /api/nexrad/animation_frames respectively.")
    print("3. For OutlookMapTest.html, it already uses /api/spc/outlook, so it should work out of the box with this backend.")
    print("4. You can open NexradMapViewer.html or OutlookMapTest.html directly in your browser after starting this server.")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) # use_reloader=False to prevent multiple runs
