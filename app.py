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
import tempfile
import logging
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from urllib.parse import urlparse

from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS

app = Flask(__name__)

# --- App Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NEXRAD_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), 'nexrad_data_output')
os.makedirs(NEXRAD_OUTPUT_DIR, exist_ok=True)
logger.info(f"NEXRAD output will be stored in: {NEXRAD_OUTPUT_DIR}")


# --- Data Access and Processing Classes ---

class NexradDownloader:
    """Handles downloading of NEXRAD data."""
    def __init__(self):
        self.session = requests.Session()
        self.s3_client = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))

    def get_available_sites(self):
        """FIX: Restored this function to populate the station dropdown."""
        return {
            'KABR': 'Aberdeen, SD', 'KABX': 'Albuquerque, NM', 'KAKQ': 'Norfolk/Wakefield, VA',
            'KAMA': 'Amarillo, TX', 'KAMX': 'Miami, FL', 'KAPX': 'Gaylord, MI', 'KARX': 'La Crosse, WI',
            'KATX': 'Seattle/Tacoma, WA', 'KBBX': 'Beale AFB, CA', 'KBGM': 'Binghamton, NY',
            'KBHX': 'Eureka, CA', 'KBIS': 'Bismarck, ND', 'KBLX': 'Billings, MT', 'KBMX': 'Birmingham, AL',
            'KBOX': 'Boston, MA', 'KBRO': 'Brownsville, TX', 'KBUF': 'Buffalo, NY', 'KBYX': 'Key West, FL',
            'KCAE': 'Columbia, SC', 'KCBW': 'Caribou, ME', 'KCBX': 'Boise, ID', 'KCCX': 'State College, PA',
            'KCLE': 'Cleveland, OH', 'KCLX': 'Charleston, SC', 'KCRP': 'Corpus Christi, TX',
            'KCXX': 'Burlington, VT', 'KCYS': 'Cheyenne, WY', 'KDAX': 'Sacramento, CA',
            'KDDC': 'Dodge City, KS', 'KDFX': 'Laughlin AFB, TX', 'KDGX': 'Jackson, MS',
            'KDIX': 'Philadelphia, PA', 'KDLH': 'Duluth, MN', 'KDMX': 'Des Moines, IA',
            'KDOX': 'Dover AFB, DE', 'KDTX': 'Detroit, MI', 'KDVN': 'Davenport, IA',
            'KDYX': 'Dyess AFB, TX', 'KEAX': 'Kansas City, MO', 'KEMX': 'Tucson, AZ', 'KENX': 'Albany, NY',
            'KEOX': 'Fort Rucker, AL', 'KEPZ': 'El Paso, TX', 'KESX': 'Las Vegas, NV',
            'KEVX': 'Eglin AFB, FL', 'KEWX': 'San Antonio, TX', 'KEYX': 'Edwards AFB, CA',
            'KFCX': 'Roanoke, VA', 'KFDR': 'Altus AFB, OK', 'KFFC': 'Atlanta, GA',
            'KFSD': 'Sioux Falls, SD', 'KFSX': 'Flagstaff, AZ', 'KFTG': 'Denver, CO',
            'KFWD': 'Dallas/Fort Worth, TX', 'KGGW': 'Glasgow, MT', 'KGJX': 'Grand Junction, CO',
            'KGLD': 'Goodland, KS', 'KGRB': 'Green Bay, WI', 'KGRK': 'Fort Hood, TX',
            'KGRR': 'Grand Rapids, MI', 'KGSP': 'Greenville/Spartanburg, SC', 'KGWX': 'Columbus AFB, MS',
            'KGYX': 'Portland, ME', 'KHGX': 'Houston, TX', 'KHNX': 'San Joaquin Valley, CA',
            'KHPX': 'Fort Campbell, KY', 'KHTX': 'Huntsville, AL', 'KICT': 'Wichita, KS',
            'KICX': 'Cedar City, UT', 'KILN': 'Cincinnati/Wilmington, OH', 'KILX': 'Lincoln, IL',
            'KIND': 'Indianapolis, IN', 'KINX': 'Tulsa, OK', 'KIWA': 'Phoenix, AZ',
            'KIWX': 'Fort Wayne, IN', 'KJAX': 'Jacksonville, FL', 'KJGX': 'Robins AFB, GA',
            'KJKL': 'Jackson, KY', 'KLBB': 'Lubbock, TX', 'KLCH': 'Lake Charles, LA',
            'KLIX': 'New Orleans, LA', 'KLNX': 'North Platte, NE', 'KLOT': 'Chicago, IL',
            'KLRX': 'Elko, NV', 'KLSX': 'St. Louis, MO', 'KLTX': 'Wilmington, NC',
            'KLVX': 'Las Vegas, NV', 'KLWX': 'Sterling, VA', 'KMAF': 'Midland/Odessa, TX',
            'KMAX': 'Medford, OR', 'KMHX': 'Morehead City, NC', 'KMKX': 'Milwaukee, WI',
            'KMLB': 'Melbourne, FL', 'KMOB': 'Mobile, AL', 'KMPX': 'Minneapolis/St. Paul, MN',
            'KMQT': 'Marquette, MI', 'KMRX': 'Knoxville, TN', 'KMSX': 'Missoula, MT',
            'KMTX': 'Salt Lake City, UT', 'KMUX': 'San Francisco, CA', 'KMVX': 'Grand Forks, ND',
            'KNKX': 'San Diego, CA', 'KNQA': 'Memphis, TN', 'KOAX': 'Omaha, NE',
            'KOHX': 'Nashville, TN', 'KOKX': 'New York City, NY', 'KOTX': 'Spokane, WA',
            'KPAH': 'Paducah, KY', 'KPBZ': 'Pittsburgh, PA', 'KPDT': 'Pendleton, OR',
            'KPOE': 'Fort Polk, LA', 'KPUX': 'Pueblo, CO', 'KRAX': 'Raleigh/Durham, NC',
            'KRGX': 'Reno, NV', 'KRIW': 'Riverton, WY', 'KRLX': 'Charleston, WV',
            'KRTX': 'Portland, OR', 'KSFX': 'Pocatello/Idaho Falls, ID', 'KSGF': 'Springfield, MO',
            'KSHV': 'Shreveport, LA', 'KSJT': 'San Angelo, TX', 'KSOX': 'Santa Ana Mountains, CA',
            'KSRX': 'Fort Smith, AR', 'KTBW': 'Tampa, FL', 'KTFX': 'Great Falls, MT',
            'KTLH': 'Tallahassee, FL', 'KTLX': 'Oklahoma City, OK', 'KTWX': 'Topeka, KS',
            'KTYX': 'Montague, NY', 'KUDX': 'Rapid City, SD', 'KUEX': 'Hastings, NE',
            'KVNX': 'Vance AFB, OK', 'KVTX': 'Roanoke, VA', 'KVWX': 'Evansville, IN', 'KYUX': 'Yuma, AZ'
        }

    def _get_aws_urls(self, station, start_time, end_time):
        urls = []
        current_date = start_time.date()
        while current_date <= end_time.date():
            prefix = f"{current_date.strftime('%Y/%m/%d')}/{station}/"
            try:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket='noaa-nexrad-level2', Prefix=prefix):
                    for obj in page.get('Contents', []):
                        if obj['Key'].endswith('.gz') and '_MDM' not in obj['Key']:
                            match = re.search(r'(\d{8})_(\d{6})', obj['Key'])
                            if match:
                                try:
                                    file_time = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S")
                                    if start_time <= file_time <= end_time:
                                        urls.append(f"s3://noaa-nexrad-level2/{obj['Key']}")
                                except ValueError:
                                    continue
            except Exception as e:
                logger.error(f"Error listing AWS S3 objects for {prefix}: {e}")
            current_date += timedelta(days=1)
        return list(set(urls))

    def _download_and_decompress_file(self, url, output_dir):
        try:
            parsed_url = urlparse(url)
            filename_gz = os.path.basename(parsed_url.path)
            output_path_gz = os.path.join(output_dir, filename_gz)
            decompressed_path = output_path_gz.replace('.gz', '')

            if os.path.exists(decompressed_path):
                return decompressed_path

            bucket_name, key = parsed_url.netloc, parsed_url.path.lstrip('/')
            self.s3_client.download_file(bucket_name, key, output_path_gz)

            with gzip.open(output_path_gz, 'rb') as f_in, open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(output_path_gz)
            
            logger.info(f"Successfully created file: {decompressed_path}")
            return decompressed_path
        except Exception as e:
            logger.error(f"Failed during download/decompression of {url}: {e}")
            return None

    def download_time_series(self, station, start_time, end_time, output_dir):
        urls = self._get_aws_urls(station, start_time, end_time)
        if not urls:
            return []

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self._download_and_decompress_file, url, output_dir) for url in urls]
            results = [future.result() for future in as_completed(futures)]

        successful_downloads = [res for res in results if res]
        logger.info(f"Successfully processed {len(successful_downloads)} raw files.")
        return successful_downloads

class NEXRADProcessor:
    def nexrad_to_geotiff(self, nexrad_file, output_geotiff, product='reflectivity'):
        try:
            radar = pyart.io.read(nexrad_file)
        except Exception as e:
            logger.error(f"Py-ART could not read file {nexrad_file}: {e}")
            return None

        available_fields = list(radar.fields.keys())
        if product not in available_fields:
            fallback = 'reflectivity' if 'reflectivity' in available_fields else (available_fields[0] if available_fields else None)
            if not fallback: return None
            product = fallback

        grid = radar.to_grid(
            (1, 500, 500),
            ((-230000, 230000), (-230000, 230000), (0, 10000)),
            fields=[product],
            grid_origin=(radar.latitude['data'][0], radar.longitude['data'][0]),
            projparams={'proj': 'aeqd', 'lat_0': radar.latitude['data'][0], 'lon_0': radar.longitude['data'][0]}
        )
        data = grid.fields[product]['data'][0]

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_geotiff, 500, 500, 1, gdal.GDT_Float32, options=['COMPRESS=LZW'])
        dataset.SetGeoTransform([grid.point_x['data'][0], grid.x['data'][1] - grid.x['data'][0], 0, 
                                 grid.point_y['data'][-1], 0, grid.y['data'][0] - grid.y['data'][1]])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())
        band = dataset.GetRasterBand(1)
        band.WriteArray(np.ma.filled(data, fill_value=np.nan))
        band.SetNoDataValue(np.nan)
        dataset.FlushCache()
        dataset = None
        return {'geotiff_path': output_geotiff}

class SPCDataFinder:
    """Finds the URL for the correct SPC outlook KMZ file."""
    BASE_URL = "https://www.spc.noaa.gov/products/outlook/"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'MyWeatherApp/1.0'})

    def find_latest_outlook_url(self, date_obj, day=1):
        """Tries common issuance times to find the latest available KMZ URL."""
        outlook_times = ['2000', '1630', '1300', '1200', '0100']
        for time_str in outlook_times:
            date_str_url = date_obj.strftime('%Y%m%d')
            url = f"{self.BASE_URL}archive/{date_obj.year}/day{day}otlk_{date_str_url}_{time_str}.kmz"
            try:
                # Use a HEAD request to check for existence without downloading the whole file
                response = self.session.head(url, timeout=5, allow_redirects=True)
                if response.status_code == 200:
                    logger.info(f"Found valid SPC outlook URL: {url}")
                    return url
            except requests.exceptions.RequestException:
                continue
        return None


# --- Flask API Endpoints ---
CORS(app)
nexrad_downloader = NexradDownloader()
nexrad_processor = NEXRADProcessor()
spc_finder = SPCDataFinder()

@app.route('/')
def serve_vue_app():
    return render_template('index.html')

@app.route('/api/nexrad/stations')
def get_nexrad_stations():
    """FIX: Added back this endpoint to populate the station dropdown."""
    stations = nexrad_downloader.get_available_sites()
    return jsonify({"success": True, "stations": stations})

@app.route('/api/nexrad/animation_frames')
def get_nexrad_animation_frames():
    station = request.args.get('station')
    date_str = request.args.get('date')
    start_time_str = request.args.get('start_time')
    end_time_str = request.args.get('end_time')
    product = request.args.get('product', 'reflectivity')
    
    try:
        start_dt = datetime.strptime(f"{date_str} {start_time_str}", '%Y-%m-%d %H:%M')
        end_dt = datetime.strptime(f"{date_str} {end_time_str}", '%Y-%m-%d %H:%M')
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid date/time format."}), 400

    temp_dir = tempfile.mkdtemp(prefix="nexrad_anim_")
    try:
        downloaded_files = nexrad_downloader.download_time_series(station, start_dt, end_dt, temp_dir)
        if not downloaded_files:
            return jsonify({"error": "No NEXRAD data found for this time range."}), 404

        frames = []
        for raw_filepath in sorted(downloaded_files):
            filename_base = os.path.basename(raw_filepath)
            output_geotiff = os.path.join(NEXRAD_OUTPUT_DIR, f"{filename_base}_{product}.tif")
            if not os.path.exists(output_geotiff):
                nexrad_processor.nexrad_to_geotiff(raw_filepath, output_geotiff, product)
            
            if os.path.exists(output_geotiff):
                match = re.search(r'(\d{8})_(\d{6})', filename_base)
                time_str = datetime.strptime(f"{match.group(1)}{match.group(2)}", '%Y%m%d%H%M%S').strftime('%H:%M:%S') if match else "N/A"
                frames.append({
                    "url": f"/geotiff/{os.path.basename(output_geotiff)}",
                    "time": time_str
                })
        
        if not frames:
            return jsonify({"error": "Data downloaded, but failed to process into images."}), 500
        
        return jsonify({"success": True, "frames": frames})
    finally:
        shutil.rmtree(temp_dir)

@app.route('/api/spc/outlook_url')
def get_spc_outlook_url():
    """FIX: Returns a direct URL to the latest valid KMZ file for the frontend to handle."""
    date_param = request.args.get('date')
    if not date_param:
        return jsonify({"error": "Date parameter is required"}), 400
    
    try:
        date_obj = datetime.strptime(date_param, '%Y-%m-%d')
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    url = spc_finder.find_latest_outlook_url(date_obj)
    if url:
        return jsonify({"success": True, "url": url})
    else:
        return jsonify({"error": f"Could not find any SPC outlook for {date_param}."}), 404

@app.route('/geotiff/<path:filename>')
def serve_geotiff(filename):
    return send_from_directory(NEXRAD_OUTPUT_DIR, filename, mimetype='image/tiff')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
