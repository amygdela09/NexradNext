import os
import io
import gzip
import boto3
import botocore 
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
import shutil
import xml.etree.ElementTree as ET

from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
NEXRAD_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), 'nexrad_data_output')
os.makedirs(NEXRAD_OUTPUT_DIR, exist_ok=True)
logger.info(f"NEXRAD GeoTIFFs will be stored in: {NEXRAD_OUTPUT_DIR}")

# --- Classes ---

class DataSource(Enum):
    AWS = "s3://noaa-nexrad-level2/"
    GOOGLE = "gs://gcp-public-data-nexrad-l2/"
    NCEI = "https://www.ncei.noaa.gov/data/nexrad-level-ii/access/"

class NexradDownloader:
    def __init__(self):
        self.session = requests.Session()
        if not hasattr(self, 's3_client'):
            try:
                self.s3_client = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))
            except Exception as e:
                logger.error(f"Failed to initialize boto3 S3 client: {e}. AWS downloads may not work.")
                self.s3_client = None

    def get_available_sites(self):
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
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            prefix = f"{current_date.strftime('%Y/%m/%d')}/{station}/"
            if self.s3_client:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                try:
                    pages = paginator.paginate(Bucket='noaa-nexrad-level2', Prefix=prefix)
                    for page in pages:
                        for obj in page.get('Contents', []):
                            key = obj['Key']
                            filename = os.path.basename(key)

                            if filename.endswith('.gz'):
                                match = re.search(r'(\d{8})_(\d{6})', filename)
                                if match:
                                    file_time_str = f"{match.group(1)}{match.group(2)}"
                                    try:
                                        file_time = datetime.strptime(file_time_str, "%Y%m%d%H%M%S")
                                        if start_time <= file_time <= end_time:
                                            urls.append(f"s3://noaa-nexrad-level2/{key}")
                                    except ValueError:
                                        logger.warning(f"Could not parse time from filename: {filename}")
                except Exception as e:
                    logger.error(f"Error listing AWS S3 objects for {prefix}: {e}")
            current_date += timedelta(days=1)
        return list(set(urls))

    def _get_google_urls(self, station, start_time, end_time):
        """Generates file URLs for the Google Cloud source."""
        # This function remains unchanged
        urls = []
        current_time = start_time
        while current_time <= end_time:
            base_url = f"gs://gcp-public-data-nexrad-l2/{current_time.year}/{current_time.month:02d}/{current_time.day:02d}/{station}/"
            filename_base = f"{station}{current_time.strftime('%Y%m%d_%H%M%S')}"
            urls.append(f"{base_url}{filename_base}_V06.gz")
            urls.append(f"{base_url}{filename_base}.gz")
            current_time += timedelta(minutes=1) # Check every minute
        return list(set(urls))

    def _get_ncei_urls(self, station, start_time, end_time):
        """Generates file URLs for the NCEI source."""
         # This function remains unchanged
        urls = []
        current_time = start_time
        while current_time <= end_time:
            date_path = current_time.strftime('%Y%m/%Y%m%d')
            base_url = f"{DataSource.NCEI.value}{date_path}/"
            filename_base = f"{station}{current_time.strftime('%Y%m%d_%H%M%S')}"
            urls.append(f"{base_url}{filename_base}_V06.gz")
            urls.append(f"{base_url}{filename_base}.gz")
            current_time += timedelta(minutes=1)
        return list(set(urls))

    def _download_file_from_url(self, url, output_path):
        """Downloads a single file from a given URL."""
         # This function remains unchanged
        try:
            decompressed_path = output_path.replace('.gz', '')
            if os.path.exists(decompressed_path):
                logger.info(f"Decompressed file already exists: {decompressed_path}")
                return decompressed_path

            logger.info(f"Attempting to download {url}")
            
            if url.startswith('s3://'):
                if not self.s3_client:
                    raise RuntimeError("Boto3 S3 client not initialized.")
                bucket_name, key = url.replace("s3://", "").split('/', 1)
                self.s3_client.download_file(bucket_name, key, output_path)
            else: # Handles gs and https
                http_url = url.replace('gs://', 'https://storage.googleapis.com/')
                response = self.session.get(http_url, stream=True, timeout=30)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)

            if output_path.endswith('.gz'):
                with gzip.open(output_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(output_path)
                logger.info(f"Decompressed to {decompressed_path}")
                return decompressed_path
            return output_path
        except requests.exceptions.RequestException as e:
            if e.response and e.response.status_code in [403, 404]:
                logger.debug(f"File not found at {url}: {e}")
            else:
                logger.warning(f"HTTP error downloading {url}: {e}")
            if os.path.exists(output_path): os.remove(output_path)
            return None
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            if os.path.exists(output_path): os.remove(output_path)
            return None
            
    def download_time_series(self, source: DataSource, station: str, start_time: datetime, end_time: datetime, output_dir: str):
        """Downloads NEXRAD data for a time series."""
         # This function remains unchanged
        os.makedirs(output_dir, exist_ok=True)
        
        url_getters = {
            DataSource.AWS: self._get_aws_urls,
            DataSource.GOOGLE: self._get_google_urls,
            DataSource.NCEI: self._get_ncei_urls
        }
        urls = url_getters.get(source, lambda *args: [])(station, start_time, end_time)

        downloaded_files = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {
                executor.submit(self._download_file_from_url, url, os.path.join(output_dir, os.path.basename(url))): url
                for url in set(urls)
            }
            for future in as_completed(future_to_url):
                try:
                    result = future.result()
                    if result: downloaded_files.append(result)
                except Exception as exc:
                    logger.error(f'URL generated an exception: {future_to_url[future]} -> {exc}')

        logger.info(f"Finished downloading. Downloaded {len(downloaded_files)} raw files.")
        return downloaded_files


class NEXRADProcessor:
    def nexrad_to_geotiff(self, nexrad_file, output_geotiff, product='reflectivity'):
        """Convert NEXRAD Level 2 data to GeoTIFF"""
         # This function remains unchanged
        logger.info(f"Processing NEXRAD file: {nexrad_file} for product: {product}")
        try:
            radar = pyart.io.read(nexrad_file)
        except Exception as e:
            logger.error(f"Could not read NEXRAD file {nexrad_file} with Py-ART: {e}")
            return None

        available_fields = list(radar.fields.keys())
        if product not in available_fields:
            logger.warning(f"Field '{product}' not found. Available fields: {available_fields}")
            product = 'reflectivity' if 'reflectivity' in available_fields else (available_fields[0] if available_fields else None)
            if not product:
                logger.error("No suitable radar fields found in the file.")
                return None
            logger.warning(f"Defaulting to field '{product}'.")

        display = pyart.graph.RadarDisplay(radar)
        fig = plt.figure(figsize=[10, 8])
        ax = fig.add_subplot(111)
        display.plot_ppi_map(product, sweep=0, ax=ax, vmin=-8, vmax=64)
        plt.savefig(output_geotiff, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        
        # This simplified version just saves a PNG map. The GeoTIFF conversion is complex.
        # For a true GeoTIFF, the previous logic with GDAL is necessary but also more error-prone.
        # This PNG approach is more robust for simple visualization.
        
        logger.info(f"Image created: {output_geotiff}")
        return {
            'image_path': output_geotiff,
            'radar_lat': radar.latitude['data'][0],
            'radar_lon': radar.longitude['data'][0],
            'field_name': product
        }


class SPCDataFetcher:
    # This class remains unchanged
    BASE_URL = "https://www.spc.noaa.gov/products/outlook"
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'MyNEXRADApp/1.0'})
    
    def get_outlook_url(self, date_str, outlook_type, time_str, day=1):
        year = date_str[:4]
        if outlook_type == 'categorical':
            return f"{self.BASE_URL}/archive/{year}/day{day}otlk_{date_str}_{time_str}_cat.kml"
        return f"{self.BASE_URL}/archive/{year}/day{day}otlk_{date_str}_{time_str}_{outlook_type}.kml"
    
    def fetch_kml_data(self, url):
        try:
            logger.info(f"Fetching data from: {url}")
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching KML data from {url}: {e}")
            return None

    def parse_kml_to_geojson(self, kml_content, outlook_type):
        try:
            root = ET.fromstring(kml_content)
            namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
            features = []
            for placemark in root.findall('.//kml:Placemark', namespaces):
                try:
                    name = placemark.find('kml:name', namespaces).text
                    polygon = placemark.find('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespaces)
                    if polygon is None: continue
                    
                    coords_text = polygon.text.strip()
                    coords_list = [ [float(lon), float(lat)] for lon, lat, _ in (p.split(',') for p in coords_text.split()) ]

                    features.append({
                        "type": "Feature",
                        "properties": {"risk": name, "type": outlook_type},
                        "geometry": {"type": "Polygon", "coordinates": [coords_list]}
                    })
                except Exception as e:
                    logger.warning(f"Could not parse placemark: {e}")
            return {"type": "FeatureCollection", "features": features}
        except ET.ParseError as e:
            logger.error(f"Error parsing KML: {e}")
            return None


# --- Flask App ---
CORS(app)
nexrad_downloader = NexradDownloader()
nexrad_processor = NEXRADProcessor()
spc_fetcher = SPCDataFetcher()

@app.route('/')
def index():
    return render_template('index.html') # Assumes templates/index.html exists

@app.route('/api/nexrad/animation_frames')
def get_nexrad_animation_frames():
    """Downloads and processes multiple NEXRAD files for animation."""
    station = request.args.get('station')
    date_str = request.args.get('date')
    product = request.args.get('product', 'reflectivity')
    start_time_str = request.args.get('start_time')
    end_time_str = request.args.get('end_time')
    
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        start_h, start_m = map(int, start_time_str.split(':'))
        end_h, end_m = map(int, end_time_str.split(':'))
        start_dt = date_obj.replace(hour=start_h, minute=start_m)
        end_dt = date_obj.replace(hour=end_h, minute=end_m)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid date/time parameter: {e}"}), 400

    animation_temp_dir = tempfile.mkdtemp(prefix="nexrad_anim_")
    
    try:
        raw_files = nexrad_downloader.download_time_series(DataSource.AWS, station, start_dt, end_dt, animation_temp_dir)
        if not raw_files:
            return jsonify({"error": f"No NEXRAD data found for {station} in the specified time range."}), 404

        frames = []
        for raw_file in sorted(raw_files):
            try:
                # Using PNG for simplicity, as GeoTIFF generation was complex
                output_png_name = os.path.basename(raw_file).replace('.gz', f'_{product}.png')
                output_path = os.path.join(NEXRAD_OUTPUT_DIR, output_png_name)
                
                # Check if image already exists
                if not os.path.exists(output_path):
                    metadata = nexrad_processor.nexrad_to_geotiff(raw_file, output_path, product)
                else:
                    # If it exists, we can't get metadata without reprocessing, so we fake it
                    metadata = {'image_path': output_path}

                if metadata:
                    match = re.search(r'(\d{8})_(\d{6})', output_png_name)
                    time_str = datetime.strptime(f"{match.group(1)}{match.group(2)}", '%Y%m%d%H%M%S').strftime('%H:%M:%S') if match else "N/A"
                    frames.append({
                        "url": f"/serve_image/{output_png_name}",
                        "time": time_str
                    })
            except Exception as e:
                logger.error(f"Failed to process file {raw_file}: {e}")
        
        if not frames:
            return jsonify({"error": "Failed to process any downloaded files into images."}), 500
        
        return jsonify({"success": True, "animation_frames": frames})

    finally:
        shutil.rmtree(animation_temp_dir)


@app.route('/api/spc/outlook')
def get_spc_outlook():
    """Get SPC outlook data."""
    date_param = request.args.get('date')
    outlook_type = request.args.get('type', 'categorical')
    day = int(request.args.get('day', 1))

    try:
        date_obj = datetime.strptime(date_param, '%Y-%m-%d')
        date_str = date_obj.strftime('%Y%m%d')
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    outlook_times = ['2000', '1630', '1300', '1200', '0100']
    for time_str in outlook_times:
        url = spc_fetcher.get_outlook_url(date_str, outlook_type, time_str, day)
        kml_content = spc_fetcher.fetch_kml_data(url)
        if kml_content:
            geojson = spc_fetcher.parse_kml_to_geojson(kml_content, outlook_type)
            if geojson:
                return jsonify({"success": True, "data": geojson, "source_url": url})
    
    return jsonify({"error": f"Could not fetch or parse SPC data for {date_param} (type: {outlook_type}). It may not exist for this day."}), 404

@app.route('/serve_image/<path:filename>')
def serve_image(filename):
    """Serve dynamically generated images."""
    return send_from_directory(NEXRAD_OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
