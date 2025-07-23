import os
import io
import gzip
import boto3
import requests
import numpy as np
import pyart
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import tempfile
import shutil
import json
import re
import xml.etree.ElementTree as ET
import logging

from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS

# Ensure PyART colormaps are registered
import pyart.graph.cm_colorblind 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NEXRAD Classes/Functions ---
class DataSource(Enum):
    AWS = "s3://noaa-nexrad-level2/"
    NCEI = "https://www.ncei.noaa.gov/data/nexrad-level-ii/access/"

class NexradProcessor:
    def __init__(self):
        self.session = requests.Session()
        self.s3_client = boto3.client('s3')

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

    def _build_possible_file_urls(self, source, station, date, hour, minute):
        """Builds possible URLs for a given time for both AWS and NCEI."""
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
        # NEXRAD Level 2 files are usually on 00 seconds, but sometimes have variations
        timestamps = [f"{hour:02d}{minute:02d}00", f"{hour:02d}{minute:02d}01", f"{hour:02d}{minute:02d}02"]

        urls = []
        if source == DataSource.AWS:
            base_path = f"{year}/{month}/{day}/{station}/"
            for timestamp in timestamps:
                # Try V06 and non-V06, gzipped and not gzipped
                possible_filenames = [
                    f"{station}{year}{month}{day}_{timestamp}_V06.gz",
                    f"{station}{year}{month}{day}_{timestamp}.gz",
                    f"{station}{year}{month}{day}_{timestamp}_V06",
                    f"{station}{year}{month}{day}_{timestamp}",
                ]
                for fname in possible_filenames:
                    urls.append((f"s3://noaa-nexrad-level2/{base_path}{fname}", fname))
        elif source == DataSource.NCEI:
            base_path_ncei = f"{date.strftime('%Y%m')}/{date.strftime('%Y%m%d')}/"
            for timestamp in timestamps:
                possible_filenames = [
                    f"{station}{year}{month}{day}_{timestamp}_V06",
                    f"{station}{year}{month}{day}_{timestamp}",
                ]
                for fname in possible_filenames:
                    urls.append((f"{DataSource.NCEI.value}{base_path_ncei}{fname}", fname))
        return urls

    def _download_single_file(self, url, output_path):
        """Downloads a single file from S3 or HTTP and decompresses if needed."""
        try:
            if url.startswith('s3://'):
                bucket_name = url.split('/')[2]
                key = '/'.join(url.split('/')[3:])
                self.s3_client.download_file(bucket_name, key, output_path)
            else: # Assume HTTP
                response = self.session.get(url, stream=True, timeout=10)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Decompress if necessary
            if output_path.endswith('.gz'):
                decompressed_path = output_path[:-3]
                with gzip.open(output_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(output_path)
                return decompressed_path
            return output_path
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    def download_nexrad_file(self, station, date_obj, hour, minute):
        """Downloads a single NEXRAD Level 2 file for a specific time."""
        for source in [DataSource.AWS, DataSource.NCEI]:
            possible_urls = self._build_possible_file_urls(source, station, date_obj, hour, minute)
            for url, filename in possible_urls:
                try:
                    logger.info(f"Attempting to download from {url}")
                    
                    if url.startswith('s3://'):
                        bucket_name = url.split('/')[2]
                        key = '/'.join(url.split('/')[3:])
                        self.s3_client.head_object(Bucket=bucket_name, Key=key)
                    else:
                        response = self.session.head(url, timeout=5)
                        response.raise_for_status()

                    temp_file_fd, temp_file_path = tempfile.mkstemp(suffix=os.path.basename(filename))
                    os.close(temp_file_fd)

                    downloaded_path = self._download_single_file(url, temp_file_path)
                    if downloaded_path:
                        logger.info(f"Successfully downloaded {downloaded_path}")
                        return downloaded_path
                except Exception as e:
                    logger.debug(f"File not found or error with {url}: {e}")
                    continue

        raise FileNotFoundError(f"No NEXRAD data found for {station} on {date_obj.strftime('%Y-%m-%d')} at {hour:02d}:{minute:02d}")

    def download_time_series_files(self, station, date_obj, start_hour, end_hour):
        """
        Downloads a time series of NEXRAD files for animation.
        This attempts to find files roughly every 5 minutes within the range.
        """
        downloaded_files_info = []
        target_times_list = []

        current_time = datetime(date_obj.year, date_obj.month, date_obj.day, start_hour, 0)
        end_dt = datetime(date_obj.year, date_obj.month, date_obj.day, end_hour, 59) # up to the end of the end_hour

        while current_time <= end_dt:
            if start_hour <= current_time.hour <= end_hour:
                target_times_list.append((current_time.hour, current_time.minute))
            current_time += timedelta(minutes=5)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for hour, minute in target_times_list:
                futures.append(executor.submit(self.download_nexrad_file, station, date_obj, hour, minute))

            # Store results with their original target_time for sorting
            results_with_times = []
            for i, future in enumerate(as_completed(futures)):
                try:
                    file_path = future.result()
                    if file_path:
                        results_with_times.append({
                            'time': f"{target_times_list[i][0]:02d}:{target_times_list[i][1]:02d}",
                            'hour': target_times_list[i][0],
                            'minute': target_times_list[i][1],
                            'file_path': file_path
                        })
                except FileNotFoundError:
                    logger.debug(f"No file found for {station} at {target_times_list[i][0]:02d}:{target_times_list[i][1]:02d}")
                except Exception as e:
                    logger.error(f"Error downloading file in time series for {station} at {target_times_list[i][0]:02d}:{target_times_list[i][1]:02d}: {e}")

        # Sort by time before returning
        downloaded_files_info = sorted(results_with_times, key=lambda x: datetime.strptime(x['time'], '%H:%M'))
        return downloaded_files_info


    def nexrad_to_png(self, nexrad_file_path, output_png_path, product='reflectivity'):
        """
        Converts a NEXRAD file to a PNG image for web display and returns metadata.
        Uses PyART and Matplotlib.
        """
        logger.info(f"Processing NEXRAD file to PNG: {nexrad_file_path}")
        try:
            radar = pyart.io.read(nexrad_file_path)

            field_to_plot = product
            if field_to_plot not in radar.fields:
                logger.warning(f"Field '{field_to_plot}' not found in radar data. Available: {list(radar.fields.keys())}. Falling back to reflectivity if available.")
                field_to_plot = 'reflectivity'
                if field_to_plot not in radar.fields:
                    if radar.fields: # Check if any fields exist at all
                        field_to_plot = list(radar.fields.keys())[0]
                        logger.warning(f"Using first available field: {field_to_plot}")
                    else:
                        logger.error(f"No fields found in radar data for {nexrad_file_path}")
                        return None
                    
            if not radar.fields.get(field_to_plot):
                logger.error(f"No valid field to plot after fallback for {nexrad_file_path}")
                return None

            display = pyart.graph.RadarDisplay(radar)
            fig = plt.figure(figsize=[10, 8])
            ax = fig.add_subplot(111)

            vmin, vmax, cmap_name = -32, 64, 'pyart_NWSStormClearRef' 
            if product == 'velocity':
                vmin, vmax, cmap_name = -30, 30, 'pyart_RdBu'

            # Assuming single sweep (sweep_idx=0) for simplicity for a 2D image
            display.plot_ppi(field_to_plot, 0, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap_name)

            radar_lat = radar.latitude['data'][0]
            radar_lon = radar.longitude['data'][0]
            
            x_min_km, x_max_km = ax.get_xlim()
            y_min_km, y_max_km = ax.get_ylim()

            lat_per_km = 1.0 / 111.0
            lon_per_km_at_radar_lat = 1.0 / (111.0 * np.cos(np.deg2rad(radar_lat)))

            south_lat = radar_lat + (y_min_km * lat_per_km)
            north_lat = radar_lat + (y_max_km * lat_per_km)
            west_lon = radar_lon + (x_min_km * lon_per_km_at_radar_lat)
            east_lon = radar_lon + (x_max_km * lon_per_km_at_radar_lat)

            bounds = [[min(south_lat, north_lat), min(west_lon, east_lon)],
                      [max(south_lat, north_lat), max(west_lon, east_lon)]]

            ax.set_title(f"NEXRAD {radar.metadata.get('instrument_name', '')} - {product.title()} Data\nScan Time: {radar.time['units'].split(' ')[-1].replace('Z', ' UTC')}", fontsize=12)
            ax.set_xlabel("Distance from Radar (km)")
            ax.set_ylabel("Distance from Radar (km)")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            buf.seek(0)

            with open(output_png_path, 'wb') as f:
                f.write(buf.getvalue())

            plt.close(fig)

            logger.info(f"PNG created: {output_png_path}")
            return {
                'png_path': output_png_path,
                'radar_lat': float(radar_lat),
                'radar_lon': float(radar_lon),
                'field_name': product,
                'max_range_km': float(max(abs(x_min_km), abs(x_max_km), abs(y_min_km), abs(y_max_km))),
                'data_min': float(np.nanmin(radar.fields[field_to_plot]['data']) if np.any(~np.isnan(radar.fields[field_to_plot]['data'])) else 0),
                'data_max': float(np.nanmax(radar.fields[field_to_plot]['data']) if np.any(~np.isnan(radar.fields[field_to_plot]['data'])) else 0),
                'bounds': bounds
            }
        except Exception as e:
            logger.error(f"Error converting NEXRAD file to PNG: {e}")
            return None

# --- SPC Classes/Functions ---
class SPCDataFetcher:
    BASE_URL = "https://www.spc.noaa.gov/products/outlook"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_outlook_url(self, date_str, outlook_type, day=1):
        year = date_str[:4]
        if outlook_type == 'categorical':
            return f"{self.BASE_URL}/archive/{year}/day{day}otlk_{date_str}_1200_cat.kml"
        else:
            return f"{self.BASE_URL}/archive/{year}/day{day}otlk_{date_str}_1200_{outlook_type}.kml"
    
    def fetch_kml_data(self, url):
        try:
            logger.info(f"Fetching data from: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching KML data: {e}")
            return None
    
    def parse_kml_to_geojson(self, kml_content, outlook_type):
        try:
            root = ET.fromstring(kml_content)
            namespaces = {
                'kml': 'http://www.opengis.net/kml/2.2',
                'gx': 'http://www.google.com/kml/ext/2.2'
            }
            features = []
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
        try:
            name_elem = placemark.find('kml:name', namespaces)
            desc_elem = placemark.find('kml:description', namespaces)
            name = name_elem.text if name_elem is not None else ""
            description = desc_elem.text if desc_elem is not None else ""
            risk_level = self.extract_risk_level(name, description, outlook_type)
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
        try:
            outer_boundary = polygon_elem.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', namespaces)
            if outer_boundary is not None and outer_boundary.text:
                coord_text = outer_boundary.text.strip()
                coordinates = []
                for coord_pair in coord_text.split():
                    parts = coord_pair.split(',')
                    if len(parts) >= 2:
                        lon, lat = float(parts[0]), float(parts[1])
                        coordinates.append([lon, lat])
                return [coordinates] if coordinates else None
            return None
        except Exception as e:
            logger.error(f"Error parsing coordinates: {e}")
            return None
    
    def extract_risk_level(self, name, description, outlook_type):
        text = f"{name} {description}".upper()
        if outlook_type == 'categorical':
            risk_levels = ['HIGH', 'MDT', 'ENH', 'SLGT', 'MRGL', 'TSTM']
            for level in risk_levels:
                if level in text:
                    return level
        else:
            percent_match = re.search(r'(\d+)%', text)
            if percent_match:
                return percent_match.group(1)
        return "UNKNOWN"

# --- Flask Application Setup ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

spc_fetcher = SPCDataFetcher()
nexrad_processor = NexradProcessor()

NEXRAD_DATA_DIR = os.path.join(app.static_folder, 'nexrad_data')
os.makedirs(NEXRAD_DATA_DIR, exist_ok=True)

# --- Utility for cleaning up old files ---
def cleanup_old_files(directory, retention_minutes=30):
    now = datetime.now()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if (now - file_mod_time) > timedelta(minutes=retention_minutes):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error cleaning up file {file_path}: {e}")

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

# SPC Routes
@app.route('/api/spc/outlook')
def get_outlook():
    try:
        date_param = request.args.get('date')
        outlook_type = request.args.get('type', 'categorical')
        day = int(request.args.get('day', 1))
        
        if not date_param:
            return jsonify({"error": "Date parameter is required"}), 400
        
        if outlook_type not in ['categorical', 'tornado', 'wind', 'hail']:
            return jsonify({"error": "Invalid outlook type"}), 400
        
        try:
            date_obj = datetime.strptime(date_param, '%Y-%m-%d')
            date_str = date_obj.strftime('%Y%m%d')
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        if date_obj > datetime.now() - timedelta(days=1):
            return jsonify({"error": "Data not available for future dates or very recent dates"}), 400
        
        url = spc_fetcher.get_outlook_url(date_str, outlook_type, day)
        kml_content = spc_fetcher.fetch_kml_data(url)
        
        if not kml_content:
            return jsonify({"error": "Could not fetch data from SPC"}), 404
        
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
        logger.error(f"Unexpected error in SPC outlook: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/spc/available-dates')
def get_available_dates_spc():
    try:
        end_date = datetime.now() - timedelta(days=1)
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
        logger.error(f"Error getting SPC available dates: {e}")
        return jsonify({"error": "Internal server error"}), 500

# NEXRAD Routes
@app.route('/api/nexrad/stations')
def get_nexrad_stations():
    return jsonify(nexrad_processor.get_available_sites())

@app.route('/api/nexrad/data')
def get_nexrad_data():
    station = request.args.get('station')
    date_str = request.args.get('date')
    product = request.args.get('product', 'reflectivity')
    mode = request.args.get('mode', 'single') 
    start_time_str = request.args.get('startTime')
    end_time_str = request.args.get('endTime')
    specific_time_str = request.args.get('time')

    if not all([station, date_str]):
        return jsonify({"error": "Station and date parameters are required"}), 400

    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    cleanup_old_files(NEXRAD_DATA_DIR)

    if mode == 'animation':
        if not all([start_time_str, end_time_str]):
             return jsonify({"error": "startTime and endTime are required for animation mode"}), 400

        try:
            start_hour, start_minute = map(int, start_time_str.split(':'))
            end_hour, end_minute = map(int, end_time_str.split(':'))
        except ValueError:
            return jsonify({"error": "Invalid time format. Use HH:MM"}), 400

        try:
            downloaded_files_info = nexrad_processor.download_time_series_files(station, date_obj, start_hour, end_hour)
            if not downloaded_files_info:
                return jsonify({"error": "No NEXRAD files found for the specified time range"}), 404

            png_data_list = []
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
                futures = []
                # Store original index to maintain order after parallel processing
                for i, file_info in enumerate(downloaded_files_info):
                    png_filename = f"{station}_{date_str}_{file_info['hour']:02d}{file_info['minute']:02d}_{product}_{i}.png"
                    output_png_path = os.path.join(NEXRAD_DATA_DIR, png_filename)
                    futures.append((executor.submit(nexrad_processor.nexrad_to_png, file_info['file_path'], output_png_path, product), i))

                # Collect results in order of submission
                results_in_order = [None] * len(futures)
                for future, original_index in futures:
                    try:
                        metadata = future.result()
                        if metadata:
                            results_in_order[original_index] = {
                                'time': downloaded_files_info[original_index]['time'],
                                'png_url': f'/static/nexrad_data/{os.path.basename(metadata["png_path"])}',
                                'radar_lat': metadata['radar_lat'],
                                'radar_lon': metadata['radar_lon'],
                                'max_range_km': metadata['max_range_km'],
                                'field_name': metadata['field_name'],
                                'data_min': metadata['data_min'],
                                'data_max': metadata['data_max'],
                                'bounds': metadata['bounds']
                            }
                    except Exception as e:
                        logger.error(f"Error processing animation frame {original_index}: {e}")
                    finally:
                        # Clean up raw NEXRAD file after processing
                        if os.path.exists(downloaded_files_info[original_index]['file_path']):
                            os.remove(downloaded_files_info[original_index]['file_path'])
            
            # Filter out any failed frames
            png_data_list = [res for res in results_in_order if res is not None]

            return jsonify({
                "success": True,
                "mode": "animation",
                "frames": png_data_list
            })

        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            logger.error(f"Error getting NEXRAD animation data: {e}", exc_info=True)
            return jsonify({"error": "Internal server error during animation data processing"}), 500

    else: # Single file mode
        if not specific_time_str:
            specific_time_str = "12:00" # Default to noon if no specific time is provided

        try:
            hour, minute = map(int, specific_time_str.split(':'))
        except ValueError:
            return jsonify({"error": "Invalid time format for single mode. Use HH:MM"}), 400

        try:
            nexrad_file_path = nexrad_processor.download_nexrad_file(station, date_obj, hour, minute)
            if not nexrad_file_path:
                return jsonify({"error": "Could not download NEXRAD file"}), 404

            png_filename = f"{station}_{date_str}_{hour:02d}{minute:02d}_{product}.png"
            output_png_path = os.path.join(NEXRAD_DATA_DIR, png_filename)
            
            metadata = nexrad_processor.nexrad_to_png(nexrad_file_path, output_png_path, product)
            
            if os.path.exists(nexrad_file_path):
                os.remove(nexrad_file_path)

            if not metadata:
                return jsonify({"error": "Could not process NEXRAD file to PNG"}), 500

            return jsonify({
                "success": True,
                "mode": "single",
                "png_url": f'/static/nexrad_data/{png_filename}',
                "radar_lat": metadata['radar_lat'],
                "radar_lon": metadata['radar_lon'],
                "max_range_km": metadata['max_range_km'],
                "field_name": metadata['field_name'],
                "data_min": metadata['data_min'],
                "data_max": metadata['data_max'],
                "bounds": metadata['bounds']
            })

        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            logger.error(f"Error getting NEXRAD data: {e}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    cleanup_old_files(NEXRAD_DATA_DIR, retention_minutes=15)

    print("Starting NEXRAD and SPC Weather Map Backend Server...")
    print("API will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)