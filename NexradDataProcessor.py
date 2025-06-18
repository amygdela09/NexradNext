#!/usr/bin/env python3
"""
NEXRAD Level 2 Data Processor
Downloads NEXRAD Level 2 data and converts to GeoTIFF format
"""

import os
import sys
import requests
from datetime import datetime, timedelta
import numpy as np
import pyart
from osgeo import gdal, osr
import matplotlib.pyplot as plt
from urllib.parse import urljoin
import gzip
import tempfile

class NEXRADProcessor:
    def __init__(self):
        # NOAA NEXRAD Level 2 Data Archive URL
        self.base_url = "https://noaa-nexrad-level2.s3.amazonaws.com/"
        
    def get_available_stations(self):
        """Get list of available NEXRAD stations"""
        # Common NEXRAD stations (subset)
        stations = {
            'KTLX': 'Oklahoma City, OK',
            'KOUN': 'Norman, OK', 
            'KFWS': 'Dallas/Fort Worth, TX',
            'KEWX': 'Austin/San Antonio, TX',
            'KHGX': 'Houston, TX',
            'KLCH': 'Lake Charles, LA',
            'KLIX': 'New Orleans, LA',
            'KMOB': 'Mobile, AL',
            'KBMX': 'Birmingham, AL',
            'KHTX': 'Huntsville, AL'
        }
        return stations
    
    def build_file_url(self, station, date, hour=None, minute=None):
        """Build URL for NEXRAD file download"""
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
        
        # If specific time not provided, try to find any file for that day
        if hour is None or minute is None:
            # Try noon as a reasonable default
            hour = hour or 12
            minute = minute or 0
            
        timestamp = f"{hour:02d}{minute:02d}"
        
        # NEXRAD file naming convention
        filename = f"{station}{year}{month}{day}_{timestamp}"
        
        # Try different possible extensions and compression
        possible_files = [
            f"{filename}.gz",
            f"{filename}",
            f"{filename}_V06.gz",
            f"{filename}_V06"
        ]
        
        base_path = f"{year}/{month}/{day}/{station}/"
        
        return [(urljoin(self.base_url, base_path + f), f) for f in possible_files]
    
    def get_available_times(self, station, date):
        """Get list of available times for a given station and date"""
        available_times = []
        
        # NEXRAD typically scans every 5-10 minutes during active weather
        # Check common scan times throughout the day
        for hour in range(24):
            for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                urls_to_try = self.build_file_url(station, date, hour, minute)
                
                for url, filename in urls_to_try:
                    try:
                        response = requests.head(url, timeout=5)
                        if response.status_code == 200:
                            time_str = f"{hour:02d}:{minute:02d}"
                            available_times.append({
                                'time': time_str,
                                'hour': hour,
                                'minute': minute,
                                'url': url,
                                'filename': filename
                            })
                            break  # Found file for this time
                    except:
                        continue
                        
        return sorted(available_times, key=lambda x: (x['hour'], x['minute']))
    
    def download_time_series(self, station, date, start_hour=None, end_hour=None, output_dir="./nexrad_data"):
        """Download multiple NEXRAD files for animation"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Searching for time series data for {station} on {date.strftime('%Y-%m-%d')}")
        
        # Get available times
        available_times = self.get_available_times(station, date)
        
        if not available_times:
            raise FileNotFoundError(f"No NEXRAD data found for {station} on {date.strftime('%Y-%m-%d')}")
        
        # Filter by time range if specified
        if start_hour is not None or end_hour is not None:
            start_h = start_hour or 0
            end_h = end_hour or 23
            available_times = [t for t in available_times if start_h <= t['hour'] <= end_h]
        
        downloaded_files = []
        
        for time_info in available_times:
            try:
                print(f"Downloading {time_info['filename']} for {time_info['time']}")
                
                response = requests.get(time_info['url'], timeout=60)
                response.raise_for_status()
                
                output_path = os.path.join(output_dir, time_info['filename'])
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                # If it's gzipped, decompress it
                if time_info['filename'].endswith('.gz'):
                    decompressed_path = output_path[:-3]
                    with gzip.open(output_path, 'rb') as gz_file:
                        with open(decompressed_path, 'wb') as out_file:
                            out_file.write(gz_file.read())
                    os.remove(output_path)  # Remove compressed file
                    output_path = decompressed_path
                
                downloaded_files.append({
                    'time': time_info['time'],
                    'hour': time_info['hour'],
                    'minute': time_info['minute'],
                    'file_path': output_path
                })
                
            except Exception as e:
                print(f"Failed to download {time_info['filename']}: {e}")
                continue
        
        print(f"Downloaded {len(downloaded_files)} files")
        return downloaded_files
    
    def create_animation_geotiffs(self, nexrad_files, output_dir="./geotiffs", product='reflectivity'):
        """Convert multiple NEXRAD files to GeoTIFFs for animation"""
        os.makedirs(output_dir, exist_ok=True)
        
        geotiff_files = []
        
        for i, file_info in enumerate(nexrad_files):
            try:
                print(f"Processing {file_info['file_path']} ({i+1}/{len(nexrad_files)})")
                
                # Create output filename
                base_name = f"radar_{file_info['hour']:02d}{file_info['minute']:02d}_{product}.tif"
                output_geotiff = os.path.join(output_dir, base_name)
                
                # Convert to GeoTIFF
                metadata = self.nexrad_to_geotiff(file_info['file_path'], output_geotiff, product)
                
                if metadata:
                    geotiff_files.append({
                        'time': file_info['time'],
                        'hour': file_info['hour'],
                        'minute': file_info['minute'],
                        'geotiff_path': output_geotiff,
                        'metadata': metadata
                    })
                    
            except Exception as e:
                print(f"Failed to process {file_info['file_path']}: {e}")
                continue
        
        return geotiff_files
    
    def download_nexrad_data(self, station, date, output_dir="./nexrad_data"):
        """Download NEXRAD Level 2 data for given station and date"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Searching for NEXRAD data for {station} on {date.strftime('%Y-%m-%d')}")
        
        # Try different hours throughout the day to find available data
        for hour in [12, 18, 6, 0, 15, 21, 9, 3]:  # Try common radar scan times
            for minute in [0, 30]:
                urls_to_try = self.build_file_url(station, date, hour, minute)
                
                for url, filename in urls_to_try:
                    try:
                        print(f"Trying: {url}")
                        response = requests.head(url, timeout=10)
                        
                        if response.status_code == 200:
                            print(f"Found file: {filename}")
                            # Download the file
                            response = requests.get(url, timeout=60)
                            response.raise_for_status()
                            
                            output_path = os.path.join(output_dir, filename)
                            with open(output_path, 'wb') as f:
                                f.write(response.content)
                            
                            # If it's gzipped, decompress it
                            if filename.endswith('.gz'):
                                decompressed_path = output_path[:-3]
                                with gzip.open(output_path, 'rb') as gz_file:
                                    with open(decompressed_path, 'wb') as out_file:
                                        out_file.write(gz_file.read())
                                os.remove(output_path)  # Remove compressed file
                                output_path = decompressed_path
                            
                            print(f"Downloaded to: {output_path}")
                            return output_path
                            
                    except requests.exceptions.RequestException as e:
                        continue  # Try next URL
                        
        raise FileNotFoundError(f"No NEXRAD data found for {station} on {date.strftime('%Y-%m-%d')}")
    
    def nexrad_to_geotiff(self, nexrad_file, output_geotiff, product='reflectivity'):
        """Convert NEXRAD Level 2 data to GeoTIFF"""
        print(f"Processing NEXRAD file: {nexrad_file}")
        
        try:
            # Read NEXRAD data using PyART
            radar = pyart.io.read(nexrad_file)
            
            # Get the specified product (default: reflectivity)
            if product == 'reflectivity':
                field_name = 'reflectivity'
            elif product == 'velocity':
                field_name = 'velocity'
            else:
                field_name = 'reflectivity'  # fallback
            
            # Check available fields
            available_fields = list(radar.fields.keys())
            print(f"Available fields: {available_fields}")
            
            # Use the first available field if specified field not found
            if field_name not in available_fields:
                field_name = available_fields[0]
                print(f"Using field: {field_name}")
            
            # Get radar location
            radar_lat = radar.latitude['data'][0]
            radar_lon = radar.longitude['data'][0]
            
            print(f"Radar location: {radar_lat:.4f}, {radar_lon:.4f}")
            
            # Get sweep data (use first sweep/elevation angle)
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
            grid_size = 500  # 500x500 grid
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
            points = points[valid_mask]
            values = values[valid_mask]
            
            # Interpolate
            grid_data = griddata(points, values, (xi_2d, yi_2d), method='linear', fill_value=np.nan)
            
            # Calculate pixel size in degrees (approximate)
            pixel_size_m = (2 * max_range) / grid_size
            # Rough conversion: 1 degree ≈ 111km at equator
            pixel_size_deg = pixel_size_m / 111000.0
            
            # Create GeoTIFF
            driver = gdal.GetDriverByName('GTiff')
            
            # Create dataset
            dataset = driver.Create(
                output_geotiff,
                grid_size,
                grid_size,
                1,
                gdal.GDT_Float32,
                options=['COMPRESS=LZW']
            )
            
            # Set geotransform (defines pixel coordinates)
            # Format: [top-left x, pixel width, rotation, top-left y, rotation, pixel height]
            geotransform = [
                radar_lon - pixel_size_deg * grid_size / 2,  # top-left longitude
                pixel_size_deg,                               # pixel width
                0,                                           # rotation
                radar_lat + pixel_size_deg * grid_size / 2,  # top-left latitude  
                0,                                           # rotation
                -pixel_size_deg                              # pixel height (negative)
            ]
            
            dataset.SetGeoTransform(geotransform)
            
            # Set projection to WGS84
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)  # WGS84
            dataset.SetProjection(srs.ExportToWkt())
            
            # Write data
            band = dataset.GetRasterBand(1)
            band.WriteArray(np.flipud(grid_data))  # Flip vertically for correct orientation
            band.SetNoDataValue(np.nan)
            
            # Set metadata
            dataset.SetMetadataItem('DESCRIPTION', f'NEXRAD {field_name} data')
            dataset.SetMetadataItem('RADAR_SITE', radar.metadata['instrument_name'])
            
            # Close dataset
            dataset = None
            
            print(f"GeoTIFF created: {output_geotiff}")
            
            # Return metadata for web display
            return {
                'geotiff_path': output_geotiff,
                'radar_lat': float(radar_lat),
                'radar_lon': float(radar_lon),
                'field_name': field_name,
                'max_range_km': float(max_range / 1000),
                'data_min': float(np.nanmin(grid_data)),
                'data_max': float(np.nanmax(grid_data))
            }
            
        except Exception as e:
            print(f"Error processing NEXRAD data: {e}")
            raise

def main():
    """Main execution function"""
    processor = NEXRADProcessor()
    
    # Configuration
    station = "KTLX"  # Oklahoma City radar
    date = datetime(2024, 5, 20)  # Example date - adjust as needed
    start_hour = 12  # Start time for animation (optional)
    end_hour = 18    # End time for animation (optional)
    
    print("Available NEXRAD stations:")
    stations = processor.get_available_stations()
    for code, name in stations.items():
        print(f"  {code}: {name}")
    
    try:
        # Check for single file or time series
        mode = input("\nDownload mode: (1) Single file, (2) Time series for animation: ").strip()
        
        if mode == "2":
            # Download time series for animation
            print(f"\nDownloading time series data for {station} from {start_hour:02d}:00 to {end_hour:02d}:00")
            nexrad_files = processor.download_time_series(station, date, start_hour, end_hour)
            
            if nexrad_files:
                print(f"\nCreating animation GeoTIFFs...")
                geotiff_files = processor.create_animation_geotiffs(nexrad_files, product='reflectivity')
                
                print(f"\nAnimation processing complete!")
                print(f"Created {len(geotiff_files)} GeoTIFF frames")
                
                # Create animation metadata file
                animation_metadata = {
                    'station': station,
                    'date': date.strftime('%Y-%m-%d'),
                    'product': 'reflectivity',
                    'frames': geotiff_files,
                    'total_frames': len(geotiff_files)
                }
                
                import json
                with open('animation_metadata.json', 'w') as f:
                    json.dump(animation_metadata, f, indent=2, default=str)
                
                print("Animation metadata saved to animation_metadata.json")
                
                return animation_metadata
            else:
                print("No files downloaded for animation")
                return None
        else:
            # Single file download (original functionality)
            nexrad_file = processor.download_nexrad_data(station, date)
            
            # Convert to GeoTIFF
            geotiff_file = f"{station}_{date.strftime('%Y%m%d')}_reflectivity.tif"
            metadata = processor.nexrad_to_geotiff(nexrad_file, geotiff_file)
            
            print("\nProcessing complete!")
            print(f"GeoTIFF file: {geotiff_file}")
            print(f"Radar location: {metadata['radar_lat']:.4f}, {metadata['radar_lon']:.4f}")
            print(f"Data range: {metadata['data_min']:.2f} to {metadata['data_max']:.2f}")
            
            return metadata
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Install required packages if running directly
    try:
        import pyart
        import scipy
        from osgeo import gdal
    except ImportError as e:
        print("Missing required packages. Install with:")
        print("pip install arm-pyart scipy gdal requests numpy matplotlib")
        sys.exit(1)
    
    main()
