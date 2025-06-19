// Cargo.toml
/*
[package]
name = "spc-backend"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
axum = "0.7"
tower = "0.4"
tower-http = { version = "0.5", features = ["cors"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json"] }
chrono = { version = "0.4", features = ["serde"] }
regex = "1.0"
quick-xml = { version = "0.31", features = ["serialize"] }
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
*/

use axum::{
    extract::Query,
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use chrono::{DateTime, Duration, NaiveDate, Utc};
use quick_xml::events::Event;
use quick_xml::Reader;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};

// Data structures
#[derive(Debug, Serialize, Deserialize)]
struct OutlookRequest {
    date: String,
    #[serde(rename = "type")]
    outlook_type: Option<String>,
    day: Option<u8>,
}

#[derive(Debug, Serialize)]
struct ApiResponse<T> {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outlook_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    day: Option<u8>,
}

#[derive(Debug, Serialize)]
struct GeoJsonFeatureCollection {
    #[serde(rename = "type")]
    feature_type: String,
    features: Vec<GeoJsonFeature>,
}

#[derive(Debug, Serialize)]
struct GeoJsonFeature {
    #[serde(rename = "type")]
    feature_type: String,
    properties: FeatureProperties,
    geometry: Geometry,
}

#[derive(Debug, Serialize)]
struct FeatureProperties {
    name: String,
    description: String,
    risk_level: String,
    outlook_type: String,
}

#[derive(Debug, Serialize)]
struct Geometry {
    #[serde(rename = "type")]
    geometry_type: String,
    coordinates: Vec<Vec<Vec<f64>>>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    timestamp: String,
}

#[derive(Debug, Serialize)]
struct AvailableDatesResponse {
    success: bool,
    dates: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ApiDocumentation {
    name: String,
    version: String,
    endpoints: HashMap<String, EndpointInfo>,
    example: String,
}

#[derive(Debug, Serialize)]
struct EndpointInfo {
    method: String,
    description: String,
    parameters: Option<HashMap<String, String>>,
}

// SPC Data Fetcher
#[derive(Clone)]
struct SPCDataFetcher {
    client: reqwest::Client,
    base_url: String,
}

impl SPCDataFetcher {
    fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: "https://www.spc.noaa.gov/products/outlook".to_string(),
        }
    }

    fn get_outlook_url(&self, date_str: &str, outlook_type: &str, day: u8) -> String {
        let year = &date_str[..4];
        
        match outlook_type {
            "categorical" => {
                format!("{}/archive/{}/day{}otlk_{}_1200_cat.kml", 
                    self.base_url, year, day, date_str)
            }
            _ => {
                format!("{}/archive/{}/day{}otlk_{}_1200_{}.kml", 
                    self.base_url, year, day, date_str, outlook_type)
            }
        }
    }

    async fn fetch_kml_data(&self, url: &str) -> Result<String, anyhow::Error> {
        info!("Fetching data from: {}", url);
        
        let response = self.client.get(url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("HTTP error: {}", response.status()));
        }
        
        let content = response.text().await?;
        Ok(content)
    }

    fn parse_kml_to_geojson(&self, kml_content: &str, outlook_type: &str) -> Result<GeoJsonFeatureCollection, anyhow::Error> {
        let mut reader = Reader::from_str(kml_content);
        reader.trim_text(true);
        
        let mut features = Vec::new();
        let mut buf = Vec::new();
        let mut current_placemark: Option<PlacemarkData> = None;
        let mut in_placemark = false;
        let mut current_element = String::new();
        let mut text_content = String::new();

        loop {
            match reader.read_event_into(&mut buf)? {
                Event::Start(ref e) => {
                    current_element = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    if current_element == "Placemark" {
                        in_placemark = true;
                        current_placemark = Some(PlacemarkData::default());
                    }
                    text_content.clear();
                }
                Event::Text(e) => {
                    text_content.push_str(&e.unescape()?);
                }
                Event::End(ref e) => {
                    let element_name = String::from_utf8_lossy(e.name().as_ref());
                    
                    if in_placemark {
                        if let Some(ref mut placemark) = current_placemark {
                            match element_name.as_ref() {
                                "name" => placemark.name = text_content.clone(),
                                "description" => placemark.description = text_content.clone(),
                                "coordinates" => placemark.coordinates = text_content.clone(),
                                "Placemark" => {
                                    if let Ok(feature) = self.parse_placemark(placemark, outlook_type) {
                                        features.push(feature);
                                    }
                                    in_placemark = false;
                                    current_placemark = None;
                                }
                                _ => {}
                            }
                        }
                    }
                    text_content.clear();
                }
                Event::Eof => break,
                _ => {}
            }
            buf.clear();
        }

        Ok(GeoJsonFeatureCollection {
            feature_type: "FeatureCollection".to_string(),
            features,
        })
    }

    fn parse_placemark(&self, placemark: &PlacemarkData, outlook_type: &str) -> Result<GeoJsonFeature, anyhow::Error> {
        let coordinates = self.parse_coordinates(&placemark.coordinates)?;
        let risk_level = self.extract_risk_level(&placemark.name, &placemark.description, outlook_type);

        Ok(GeoJsonFeature {
            feature_type: "Feature".to_string(),
            properties: FeatureProperties {
                name: placemark.name.clone(),
                description: placemark.description.clone(),
                risk_level,
                outlook_type: outlook_type.to_string(),
            },
            geometry: Geometry {
                geometry_type: "Polygon".to_string(),
                coordinates: vec![coordinates],
            },
        })
    }

    fn parse_coordinates(&self, coord_text: &str) -> Result<Vec<Vec<f64>>, anyhow::Error> {
        let mut coordinates = Vec::new();
        
        for coord_pair in coord_text.split_whitespace() {
            let parts: Vec<&str> = coord_pair.split(',').collect();
            if parts.len() >= 2 {
                let lon: f64 = parts[0].parse()?;
                let lat: f64 = parts[1].parse()?;
                coordinates.push(vec![lon, lat]);
            }
        }
        
        if coordinates.is_empty() {
            return Err(anyhow::anyhow!("No valid coordinates found"));
        }
        
        Ok(coordinates)
    }

    fn extract_risk_level(&self, name: &str, description: &str, outlook_type: &str) -> String {
        let text = format!("{} {}", name, description).to_uppercase();
        
        match outlook_type {
            "categorical" => {
                let risk_levels = ["HIGH", "MDT", "ENH", "SLGT", "MRGL", "TSTM"];
                for level in risk_levels {
                    if text.contains(level) {
                        return level.to_string();
                    }
                }
            }
            _ => {
                if let Ok(re) = Regex::new(r"(\d+)%") {
                    if let Some(cap) = re.captures(&text) {
                        return cap[1].to_string();
                    }
                }
            }
        }
        
        "UNKNOWN".to_string()
    }
}

#[derive(Debug, Default)]
struct PlacemarkData {
    name: String,
    description: String,
    coordinates: String,
}

// API Handlers
async fn get_outlook(Query(params): Query<OutlookRequest>) -> Result<Json<ApiResponse<GeoJsonFeatureCollection>>, StatusCode> {
    let outlook_type = params.outlook_type.unwrap_or_else(|| "categorical".to_string());
    let day = params.day.unwrap_or(1);
    
    // Validate outlook type
    if !["categorical", "tornado", "wind", "hail"].contains(&outlook_type.as_str()) {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Invalid outlook type".to_string()),
            source_url: None,
            date: None,
            outlook_type: None,
            day: None,
        }));
    }
    
    // Parse and validate date
    let date_obj = match NaiveDate::parse_from_str(&params.date, "%Y-%m-%d") {
        Ok(date) => date,
        Err(_) => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Invalid date format. Use YYYY-MM-DD".to_string()),
                source_url: None,
                date: None,
                outlook_type: None,
                day: None,
            }));
        }
    };
    
    // Check if date is too recent
    let now = Utc::now().naive_utc().date();
    if date_obj > now - Duration::days(1) {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Data not available for future dates or very recent dates".to_string()),
            source_url: None,
            date: None,
            outlook_type: None,
            day: None,
        }));
    }
    
    let fetcher = SPCDataFetcher::new();
    let date_str = date_obj.format("%Y%m%d").to_string();
    let url = fetcher.get_outlook_url(&date_str, &outlook_type, day);
    
    match fetcher.fetch_kml_data(&url).await {
        Ok(kml_content) => {
            match fetcher.parse_kml_to_geojson(&kml_content, &outlook_type) {
                Ok(geojson) => {
                    Ok(Json(ApiResponse {
                        success: true,
                        data: Some(geojson),
                        error: None,
                        source_url: Some(url),
                        date: Some(params.date),
                        outlook_type: Some(outlook_type),
                        day: Some(day),
                    }))
                }
                Err(e) => {
                    error!("Failed to parse KML: {}", e);
                    Ok(Json(ApiResponse {
                        success: false,
                        data: None,
                        error: Some("Could not parse KML data".to_string()),
                        source_url: None,
                        date: None,
                        outlook_type: None,
                        day: None,
                    }))
                }
            }
        }
        Err(e) => {
            error!("Failed to fetch KML data: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Could not fetch data from SPC".to_string()),
                source_url: None,
                date: None,
                outlook_type: None,
                day: None,
            }))
        }
    }
}

async fn get_available_dates() -> Json<AvailableDatesResponse> {
    let end_date = Utc::now().naive_utc().date() - Duration::days(1);
    let start_date = end_date - Duration::days(30);
    
    let mut dates = Vec::new();
    let mut current_date = start_date;
    
    while current_date <= end_date {
        dates.push(current_date.format("%Y-%m-%d").to_string());
        current_date += Duration::days(1);
    }
    
    Json(AvailableDatesResponse {
        success: true,
        dates,
    })
}

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: Utc::now().to_rfc3339(),
    })
}

async fn api_documentation() -> Json<ApiDocumentation> {
    let mut endpoints = HashMap::new();
    
    let mut outlook_params = HashMap::new();
    outlook_params.insert("date".to_string(), "Date in YYYY-MM-DD format (required)".to_string());
    outlook_params.insert("type".to_string(), "Outlook type: categorical, tornado, wind, hail (default: categorical)".to_string());
    outlook_params.insert("day".to_string(), "Outlook day 1-8 (default: 1)".to_string());
    
    endpoints.insert("/api/spc/outlook".to_string(), EndpointInfo {
        method: "GET".to_string(),
        description: "Get SPC outlook data".to_string(),
        parameters: Some(outlook_params),
    });
    
    endpoints.insert("/api/spc/available-dates".to_string(), EndpointInfo {
        method: "GET".to_string(),
        description: "Get list of available dates".to_string(),
        parameters: None,
    });
    
    endpoints.insert("/api/health".to_string(), EndpointInfo {
        method: "GET".to_string(),
        description: "Health check".to_string(),
        parameters: None,
    });
    
    Json(ApiDocumentation {
        name: "SPC Outlook Data API".to_string(),
        version: "1.0.0".to_string(),
        endpoints,
        example: "/api/spc/outlook?date=2024-06-15&type=tornado".to_string(),
    })
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting SPC Data Backend Server...");
    
    // Build the application with routes
    let app = Router::new()
        .route("/", get(api_documentation))
        .route("/api/spc/outlook", get(get_outlook))
        .route("/api/spc/available-dates", get(get_available_dates))
        .route("/api/health", get(health_check))
        .layer(CorsLayer::permissive());
    
    // Start the server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:5000")
        .await
        .expect("Failed to bind to address");
    
    info!("API server running at: http://localhost:5000");
    info!("Example: http://localhost:5000/api/spc/outlook?date=2024-06-15&type=categorical");
    
    axum::serve(listener, app)
        .await
        .expect("Server failed to start");
}
