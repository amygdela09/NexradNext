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
    """

    def __init__(self):
        self.session = requests.Session()
        self.s3_client = boto3.client('s3')

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
        current_time = start_time
        while current_time <= end_time:
            prefix = f"{current_time.strftime('%Y/%m/%d')}/{station}/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket='noaa-nexrad-level2', Prefix=prefix)
            for page in pages:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.gz'):
                        file_time_str = key.split('_')[-2]
                        file_time = datetime.strptime(f"{current_time.strftime('%Y%m%d')}{file_time_str}", "%Y%m%d%H%M%S")
                        if start_time <= file_time <= end_time:
                            urls.append(f"s3://noaa-nexrad-level2/{key}")
            current_time += timedelta(days=1)
        return urls


    def _get_google_urls(self, station, start_time, end_time):
        """Generates file URLs for the Google Cloud source."""
        # Note: This is a simplified example. A more robust implementation
        # would use the Google Cloud Storage client library to list files.
        # For this example, we'll construct URLs based on the known pattern.
        urls = []
        current_time = start_time
        while current_time <= end_time:
            # Google Cloud stores files in hourly tarballs
            for hour in range(current_time.hour, end_time.hour + 1 if current_time.date() == end_time.date() else 24):
                 # Construct the tar file name based on the documentation
                file_name = f"gs://gcp-public-data-nexrad-l2/{current_time.year}/{current_time.month:02d}/{current_time.day:02d}/{station}/NWS_NEXRAD_NXL2LG_{station}_{current_time.year}{current_time.month:02d}{current_time.day:02d}{hour:02d}0000_{current_time.year}{current_time.month:02d}{current_time.day:02d}{hour:02d}5959.tar"
                urls.append(file_name)
            current_time += timedelta(days=1)
        return urls

    def _get_ncei_urls(self, station, start_time, end_time):
        """Generates file URLs for the NCEI source."""
        # NCEI has a predictable URL structure
        urls = []
        current_time = start_time
        while current_time <= end_time:
            date_str = current_time.strftime('%Y%m%d')
            # Iterate through possible filenames (as NCEI has some variability)
            for minute in range(0, 60, 5): # Check every 5 minutes
                for second in range(0, 60, 1):
                    time_str = current_time.strftime('%H') + f"{minute:02d}" + f"{second:02d}"
                    file_name = f"{station}{date_str}_{time_str}_V06"
                    url = f"{DataSource.NCEI.value}{current_time.strftime('%Y%m')}/{date_str}/{file_name}"
                    urls.append(url)
            current_time += timedelta(minutes=5)
        return urls

    def _download_file(self, url, output_dir):
        """Downloads a single file."""
        try:
            file_name = os.path.basename(url)
            output_path = os.path.join(output_dir, file_name)

            if url.startswith('s3://'):
                bucket_name = url.split('/')[2]
                key = '/'.join(url.split('/')[3:])
                self.s3_client.download_file(bucket_name, key, output_path)
            else:
                response = self.session.get(url, stream=True)
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
                os.remove(output_path)
                return decompressed_path
            return output_path
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None

    def nexrad_to_geotiff(self, nexrad_file, output_geotiff, product='reflectivity'):
        """
        Converts a NEXRAD file to a GeoTIFF.
        This function is an adaptation of the original NexradDataProcessor.py script.
        """
        try:
            radar = pyart.io.read(nexrad_file)
            display = pyart.graph.RadarDisplay(radar)
            fig = plt.figure(figsize=[10, 8])
            ax = fig.add_subplot(111)
            display.plot_ppi(product, 0, ax=ax, vmin=-32, vmax=64.)

            # Create an in-memory buffer for the image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)

            # Use GDAL to convert the PNG to a GeoTIFF
            src_ds = gdal.Open(f'/vsimem/{os.path.basename(nexrad_file)}.png', gdal.GA_ReadOnly)
            gdal.GetDriverByName('GTiff').CreateCopy(output_geotiff, src_ds)

            plt.close(fig)
            return output_geotiff
        except Exception as e:
            print(f"Failed to convert {nexrad_file} to GeoTIFF: {e}")
            return None


    def download(self, source, station, start_time, end_time, convert_to_geotiff=False, output_dir='nexrad_data'):
        """
        Downloads NEXRAD data from the specified source.
        """
        os.makedirs(output_dir, exist_ok=True)

        if source == DataSource.AWS:
            urls = self._get_aws_urls(station, start_time, end_time)
        elif source == DataSource.GOOGLE:
            urls = self._get_google_urls(station, start_time, end_time)
        elif source == DataSource.NCEI:
            urls = self._get_ncei_urls(station, start_time, end_time)
        else:
            raise ValueError("Invalid data source specified.")

        downloaded_files = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(self._download_file, url, output_dir): url for url in urls}
            for future in as_completed(future_to_url):
                result = future.result()
                if result:
                    downloaded_files.append(result)

        if convert_to_geotiff:
            geotiff_files = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_file = {executor.submit(self.nexrad_to_geotiff, f, f.replace('.ar2v', '.tif')): f for f in downloaded_files}
                for future in as_completed(future_to_file):
                    result = future.result()
                    if result:
                        geotiff_files.append(result)
            return geotiff_files

        return downloaded_files
