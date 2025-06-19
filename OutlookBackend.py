#!/usr/bin/env python3
"""
SPC Severe Weather Outlook Data Backend
Fetches KML/XML data from NOAA Storm Prediction Center and serves via API
"""

import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import re
import zipfile
import io
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SPCDataFetcher:
    """Handles fetching and parsing SPC outlook data"""
    
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
            logger.error(f"Error fetching KML data: {e}")
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

# Initialize the data fetcher
spc_fetcher = SPCDataFetcher()

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
        if date_obj > datetime.now() - timedelta(days=1):
            return jsonify({"error": "Data not available for future dates or very recent dates"}), 400
        
        # Get SPC URL and fetch data
        url = spc_fetcher.get_outlook_url(date_str, outlook_type, day)
        kml_content = spc_fetcher.fetch_kml_data(url)
        
        if not kml_content:
            return jsonify({"error": "Could not fetch data from SPC"}), 404
        
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
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/spc/available-dates')
def get_available_dates():
    """Get available dates for SPC data (last 30 days)"""
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
        logger.error(f"Error getting available dates: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        "name": "SPC Outlook Data API",
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
                "description": "Get list of available dates"
            },
            "/api/health": {
                "method": "GET",
                "description": "Health check"
            }
        },
        "example": "/api/spc/outlook?date=2024-06-15&type=tornado"
    })

if __name__ == '__main__':
    print("Starting SPC Data Backend Server...")
    print("API will be available at: http://localhost:5000")
    print("Example: http://localhost:5000/api/spc/outlook?date=2024-06-15&type=categorical")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
