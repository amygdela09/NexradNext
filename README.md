NEXRADNext

This project provides a web-based application capable of visualizing real-time (or near-real-time) NEXRAD radar data and Storm Prediction Center (SPC) severe weather outlooks on a single interactive Leaflet map. It integrates various functionalities previously developed as mini-projects into a cohesive system, offering a unified platform for weather enthusiasts and meteorologists to observe severe weather threats.

The application consists of a Python Flask backend that handles data fetching and processing, and a single HTML/JavaScript frontend that renders the data on a map in the browser.
Features
Current Capabilities:

    Combined Map Display: Visualize both NEXRAD radar imagery and SPC severe weather outlook polygons on the same Leaflet map.

    NEXRAD Radar Data:

        Fetch Level 2 NEXRAD radar data (reflectivity and velocity products) from NOAA's AWS S3 bucket (default), Google Cloud Storage, or NCEI archives.

        Process raw NEXRAD data (.ar2v files) into georeferenced GeoTIFF images suitable for web display.

        Select from a comprehensive list of NEXRAD radar stations across the United States.

        Single Frame View: Display a single radar scan for a specific date and time.

        Animation Mode: Download and animate a sequence of radar scans over a user-defined time range.

        Dynamic Radar Legend: Displays a color scale for the selected radar product (reflectivity or velocity).

        Radar Information Display: Shows metadata about the currently displayed radar frame (station, product, data range).

        Opacity Control: Adjust the transparency of the radar overlay on the map.

    SPC Severe Weather Outlooks:

        Fetch SPC Day 1 Convective Outlooks (Categorical, Tornado, Wind, Hail) in GEOJson format.

        Display outlook polygons with appropriate color-coding based on risk levels.

        Dynamic SPC Legend: Shows the color scheme and corresponding risk levels for the selected outlook type.

        Interactive Popups: Click on outlook polygons to view detailed information (risk level, description).

    User Interface:

        Intuitive web interface with a sidebar for controls (date, station, outlook type, animation settings).

        Date and time pickers for precise data selection.

        Play/pause, slider, and speed controls for radar animation.

        Status messages to provide user feedback on data loading and errors.

    Backend API: A Flask-based RESTful API serving both NEXRAD GeoTIFF URLs and SPC GeoJSON data, enabling a clean separation of concerns.

    Temporary File Management: Automatically handles temporary downloads and processed GeoTIFFs, cleaning up raw files after processing.

Installation and Setup

To get this project up and running, follow these steps:
Prerequisites

    Python 3.x: Ensure you have Python installed.

    pip: Python package installer (usually comes with Python).

    Required Python Libraries:

        Flask

        Flask-Cors

        requests

        numpy

        pyart

        scipy

        boto3 (for AWS S3 access)

        matplotlib

        GDAL (crucial for GeoTIFF processing)

    You can install most of these using pip:

    pip install Flask Flask-Cors requests numpy pyart scipy boto3 matplotlib

    Important Note on GDAL: Installing GDAL can be tricky, especially on Windows or macOS. If pip install GDAL fails, you might need to install pre-compiled GDAL binaries for your system first, or use a package manager like conda.

        For Conda users:

        conda install -c conda-forge gdal

        For manual installation (e.g., Windows): Download the appropriate wheel file (GDAL‑x.x.x‑cpXX‑cpXXm‑win_amd64.whl) from a reliable source like Gohlke's Python Wheels and install it using pip install path/to/your/GDAL‑x.x.x‑cpXX‑cpXXm‑win_amd64.whl.

API Endpoints

The Flask backend exposes the following API endpoints:

    /api/spc/outlook (GET)

        Description: Fetches SPC severe weather outlook data (GeoJSON).

        Parameters:

            date (required): Date in YYYY-MM-DD format.

            type (optional, default: categorical): Outlook type (categorical, tornado, wind, hail).

            day (optional, default: 1): Outlook day (1-8).

    /api/spc/available-dates (GET)

        Description: Returns a list of available dates for SPC outlooks (last 30 days).

    /api/nexrad/stations (GET)

        Description: Returns a list of available NEXRAD radar station codes and names.

    /api/nexrad/available_times (GET)

        Description: Returns a list of approximate available scan times for a given NEXRAD station and date. (Note: This is an approximation for performance; actual file existence is checked during download).

        Parameters:

            station (required): NEXRAD station code (e.g., KTLX).

            date (required): Date in YYYY-MM-DD format.

    /api/nexrad/single_frame (GET)

        Description: Downloads and processes a single NEXRAD Level 2 file into a GeoTIFF image.

        Parameters:

            station (required): NEXRAD station code.

            date (required): Date in YYYY-MM-DD format.

            hour (required): Hour (0-23).

            minute (required): Minute (0-59).

            product (optional, default: reflectivity): Radar product (reflectivity or velocity).

            source (optional, default: AWS): Data source (AWS, GOOGLE, NCEI).

        Returns: A JSON object containing the URL to the generated GeoTIFF and its metadata.

    /api/nexrad/animation_frames (GET)

        Description: Downloads and processes multiple NEXRAD Level 2 files within a time range into GeoTIFF images for animation.

        Parameters:

            station (required): NEXRAD station code.

            date (required): Date in YYYY-MM-DD format.

            start_time (required): Start time in HH:MM format.

            end_time (required): End time in HH:MM format.

            product (optional, default: reflectivity): Radar product (reflectivity or velocity).

            source (optional, default: AWS): Data source (AWS, GOOGLE, NCEI).

        Returns: A JSON object containing a list of URLs to the generated GeoTIFF frames and their metadata.

    /nexrad_static/<path:filename> (GET)

        Description: Serves the dynamically generated NEXRAD GeoTIFF image files.

    /api/health (GET)

        Description: A simple health check endpoint to verify the backend is running.

    / (GET)

        Description: Provides basic API documentation.

Potential Future Contributions

Data & Content Enhancements:

    More Weather Data Layers:

        Severe Reports: Overlay historical severe weather reports (tornadoes, hail, wind damage).

        Watches/Warnings: Display active NWS severe thunderstorm and tornado watches/warnings.

        Satellite Imagery: Integrate visible, infrared, or water vapor satellite imagery.

        Surface Observations: Show current temperature, dew point, wind, and pressure data from weather stations.

        Model Forecasts: Overlay short-range numerical weather prediction model outputs (e.g., HRRR, NAM).

    NEXRAD Product Expansion:

        Support for more Level 2 products (e.g., Differential Reflectivity, Correlation Coefficient, Specific Differential Phase).

        Implement Level 3 products (e.g., Base Reflectivity, Composite Reflectivity, Radial Velocity).

    Historical Data Access: Enhance the date selection to easily browse historical radar and outlook data beyond the immediate past.

NEXRAD Processing & Rendering Improvements:

    Advanced GeoTIFF Generation:

        Utilize pyresample or similar libraries for more accurate and efficient reprojection of radar data to a geographic grid, reducing distortion.

        Implement more sophisticated interpolation methods (e.g., nearest neighbor, cubic spline) for better image quality.

    Client-Side GeoTIFF Rendering: Explore libraries like geotiff.js to render GeoTIFFs directly in the browser, potentially reducing server load and improving interactivity.

    Performance Optimization: Implement caching mechanisms in the Flask backend for frequently requested radar files or processed GeoTIFFs to speed up response times.

User Interface & Experience (UI/UX):

    Enhanced Animation Controls:

        Add "step forward/backward" buttons for frame-by-frame navigation.

        Include preset speed options (e.g., 0.5x, 1x, 2x).

        Implement a "loop count" option for animation.

    Time Zone Handling: Allow users to select their preferred time zone for displaying radar and outlook times.

    Location Search: Add a search bar to quickly pan the map to a specific city or address.

    Mobile Optimizations: Further enhance responsiveness and touch-friendly interactions for mobile devices.

    Custom Styling: Allow users to customize map themes, radar color palettes, or outlook polygon styles.

    Interactive Radar Data: Enable clicking on radar pixels to retrieve specific dBZ/velocity values and associated metadata.

Backend Robustness & Scalability:

    Error Handling & Logging: Implement more detailed error logging to files for easier debugging in production.

    Rate Limiting: Add rate limiting to API endpoints to prevent abuse and protect external data sources.

    Containerization: Package the Flask application using Docker for easier deployment and scalability.

    Database Integration: For persistent storage of user preferences, saved locations, or custom configurations.

Advanced Interactivity:

    Storm Tracking & Forecasting: Integrate algorithms to identify and track storm cells, and potentially display short-term forecast tracks.

    Cross-Section Views: Allow users to draw a line on the map and view a vertical cross-section of radar data along that line (this would be a significant undertaking).

    User Feedback/Reporting: A simple mechanism for users to report issues or provide feedback.

License:
    MIT License. Check the license file for details.
