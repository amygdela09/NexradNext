// Cargo.toml
/*
[package]
name = "nexrad-backend"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
axum = "0.7"
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "fs"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json", "stream"] }
chrono = { version = "0.4", features = ["serde"] }
regex = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
uuid = { version = "1.0", features = ["v4"] }
tokio-util = { version = "0.7", features = ["io"] }
futures-util = "0.3"
base64 = "0.21"
url = "2.0"
bytes = "1.0"
flate2 = "1.0"
tar = "0.4"
*/

use axum::{
    extract::{Path, Query},
    http::{HeaderMap, StatusCode},
    response::{Json, Response},
    routing::get,
    Router,
};
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};
use uuid::Uuid;

// Data structures
#[derive(Debug, Serialize, Deserialize)]
struct NexradRequest {
    site: String,
    date: String,
    time: Option<String>,
    source: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct NexradListRequest {
    site: String,
    date: String,
    source: Option<String>,
}

#[derive(Debug, Serialize)]
struct ApiResponse<T> {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<NexradMetadata>,
}

#[derive(Debug, Serialize)]
struct NexradMetadata {
    site: String,
    date: String,
    time: String,
    filename: String,
    size_bytes: Option<u64>,
    source_url: String,
    data_format: String,
}

#[derive(Debug, Serialize)]
struct NexradFile {
    filename: String,
    datetime: String,
    size_bytes: Option<u64>,
    download_url: String,
    source: String,
}

#[derive(Debug, Serialize)]
struct NexradSite {
    id: String,
    name: String,
    state: String,
    lat: f64,
    lon: f64,
    elevation_m: f64,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    timestamp: String,
    sources: HashMap<String, String>,
}

// NEXRAD Data Sources
#[derive(Clone)]
enum DataSource {
    AWS,
    GoogleCloud,
    DataGov,
    NOAA,
}

impl DataSource {
    fn from_string(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "aws" => Some(DataSource::AWS),
            "gcp" | "google" | "googlecloud" => Some(DataSource::GoogleCloud),
            "datagov" | "data.gov" => Some(DataSource::DataGov),
            "noaa" => Some(DataSource::NOAA),
            _ => None,
        }
    }

    fn to_string(&self) -> String {
        match self {
            DataSource::AWS => "aws".to_string(),
            DataSource::GoogleCloud => "gcp".to_string(),
            DataSource::DataGov => "datagov".to_string(),
            DataSource::NOAA => "noaa".to_string(),
        }
    }
}

// NEXRAD Data Fetcher
#[derive(Clone)]
struct NexradDataFetcher {
    client: reqwest::Client,
}

impl NexradDataFetcher {
    fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent("NEXRAD-Backend/1.0")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self { client }
    }

    async fn list_files(&self, site: &str, date: &str, source: DataSource) -> Result<Vec<NexradFile>, anyhow::Error> {
        match source {
            DataSource::AWS => self.list_aws_files(site, date).await,
            DataSource::GoogleCloud => self.list_gcp_files(site, date).await,
            DataSource::DataGov => self.list_datagov_files(site, date).await,
            DataSource::NOAA => self.list_noaa_files(site, date).await,
        }
    }

    async fn list_aws_files(&self, site: &str, date: &str) -> Result<Vec<NexradFile>, anyhow::Error> {
        // AWS S3 NEXRAD bucket: noaa-nexrad-level2
        let year = &date[..4];
        let month = &date[5..7];
        let day = &date[8..10];
        
        let url = format!(
            "https://noaa-nexrad-level2.s3.amazonaws.com/?list-type=2&prefix={}/{}/{}/{}/",
            year, month, day, site
        );

        info!("Fetching AWS NEXRAD files from: {}", url);
        
        let response = self.client.get(&url).send().await?;
        let xml_content = response.text().await?;
        
        self.parse_aws_s3_response(&xml_content, site, date)
    }

    async fn list_gcp_files(&self, site: &str, date: &str) -> Result<Vec<NexradFile>, anyhow::Error> {
        // Google Cloud Public Dataset: gcp-public-data-nexrad-l2
        let year = &date[..4];
        let month = &date[5..7];
        let day = &date[8..10];
        
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/gcp-public-data-nexrad-l2/o?prefix={}/{}/{}/{}/",
            year, month, day, site
        );

        info!("Fetching GCP NEXRAD files from: {}", url);
        
        let response = self.client.get(&url).send().await?;
        let json_response: serde_json::Value = response.json().await?;
        
        self.parse_gcp_response(&json_response, site, date)
    }

    async fn list_datagov_files(&self, site: &str, date: &str) -> Result<Vec<NexradFile>, anyhow::Error> {
        // Data.gov CKAN API for NEXRAD data
        let url = format!(
            "https://catalog.data.gov/api/3/action/package_search?q=nexrad+{}&fq=res_format:Level2",
            site
        );

        info!("Fetching Data.gov NEXRAD files from: {}", url);
        
        let response = self.client.get(&url).send().await?;
        let json_response: serde_json::Value = response.json().await?;
        
        self.parse_datagov_response(&json_response, site, date)
    }

    async fn list_noaa_files(&self, site: &str, date: &str) -> Result<Vec<NexradFile>, anyhow::Error> {
        // NOAA NCEI Archive
        let year = &date[..4];
        let month = &date[5..7];
        let day = &date[8..10];
        
        let url = format!(
            "https://www.ncei.noaa.gov/data/nexrad-level-2/access/{}/{}/{}/{}",
            year, month, day, site
        );

        info!("Fetching NOAA NEXRAD files from: {}", url);
        
        let response = self.client.get(&url).send().await?;
        let html_content = response.text().await?;
        
        self.parse_noaa_html_response(&html_content, site, date)
    }

    fn parse_aws_s3_response(&self, xml_content: &str, site: &str, date: &str) -> Result<Vec<NexradFile>, anyhow::Error> {
        let mut files = Vec::new();
        
        // Simple XML parsing for S3 ListBucketResult
        let key_regex = Regex::new(r"<Key>([^<]+)</Key>")?;
        let size_regex = Regex::new(r"<Size>(\d+)</Size>")?;
        let modified_regex = Regex::new(r"<LastModified>([^<]+)</LastModified>")?;
        
        let keys: Vec<&str> = key_regex.find_iter(xml_content)
            .map(|m| m.as_str())
            .collect();
        let sizes: Vec<&str> = size_regex.find_iter(xml_content)
            .map(|m| m.as_str())
            .collect();
        let modified_times: Vec<&str> = modified_regex.find_iter(xml_content)
            .map(|m| m.as_str())
            .collect();

        for (i, key_match) in keys.iter().enumerate() {
            if let Some(caps) = key_regex.captures(key_match) {
                let key = caps.get(1).unwrap().as_str();
                if key.ends_with(".gz") || key.ends_with(".bz2") {
                    let filename = key.split('/').last().unwrap_or(key);
                    
                    let size = if i < sizes.len() {
                        size_regex.captures(sizes[i])
                            .and_then(|c| c.get(1))
                            .and_then(|m| m.as_str().parse().ok())
                    } else {
                        None
                    };

                    let datetime = if i < modified_times.len() {
                        modified_regex.captures(modified_times[i])
                            .and_then(|c| c.get(1))
                            .map(|m| m.as_str().to_string())
                            .unwrap_or_else(|| "unknown".to_string())
                    } else {
                        "unknown".to_string()
                    };

                    files.push(NexradFile {
                        filename: filename.to_string(),
                        datetime,
                        size_bytes: size,
                        download_url: format!("https://noaa-nexrad-level2.s3.amazonaws.com/{}", key),
                        source: "aws".to_string(),
                    });
                }
            }
        }

        Ok(files)
    }

    fn parse_gcp_response(&self, json_response: &serde_json::Value, site: &str, date: &str) -> Result<Vec<NexradFile>, anyhow::Error> {
        let mut files = Vec::new();
        
        if let Some(items) = json_response["items"].as_array() {
            for item in items {
                if let (Some(name), Some(size), Some(updated)) = (
                    item["name"].as_str(),
                    item["size"].as_str().and_then(|s| s.parse::<u64>().ok()),
                    item["updated"].as_str(),
                ) {
                    if name.ends_with(".gz") || name.ends_with(".bz2") {
                        let filename = name.split('/').last().unwrap_or(name);
                        
                        files.push(NexradFile {
                            filename: filename.to_string(),
                            datetime: updated.to_string(),
                            size_bytes: Some(size),
                            download_url: format!("https://storage.googleapis.com/gcp-public-data-nexrad-l2/{}", name),
                            source: "gcp".to_string(),
                        });
                    }
                }
            }
        }

        Ok(files)
    }

    fn parse_datagov_response(&self, json_response: &serde_json::Value, site: &str, date: &str) -> Result<Vec<NexradFile>, anyhow::Error> {
        let mut files = Vec::new();
        
        if let Some(result) = json_response["result"].as_object() {
            if let Some(results) = result["results"].as_array() {
                for dataset in results {
                    if let Some(resources) = dataset["resources"].as_array() {
                        for resource in resources {
                            if let (Some(name), Some(url)) = (
                                resource["name"].as_str(),
                                resource["url"].as_str(),
                            ) {
                                if name.contains(site) && (name.ends_with(".gz") || name.ends_with(".bz2")) {
                                    files.push(NexradFile {
                                        filename: name.to_string(),
                                        datetime: resource["created"].as_str().unwrap_or("unknown").to_string(),
                                        size_bytes: resource["size"].as_str().and_then(|s| s.parse().ok()),
                                        download_url: url.to_string(),
                                        source: "datagov".to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(files)
    }

    fn parse_noaa_html_response(&self, html_content: &str, site: &str, date: &str) -> Result<Vec<NexradFile>, anyhow::Error> {
        let mut files = Vec::new();
        
        // Parse HTML directory listing
        let link_regex = Regex::new(r#"<a href="([^"]+\.(?:gz|bz2))"[^>]*>([^<]+)</a>"#)?;
        
        for caps in link_regex.captures_iter(html_content) {
            let filename = caps.get(1).unwrap().as_str();
            let display_name = caps.get(2).unwrap().as_str();
            
            if filename.contains(site) {
                files.push(NexradFile {
                    filename: display_name.to_string(),
                    datetime: "unknown".to_string(),
                    size_bytes: None,
                    download_url: format!("https://www.ncei.noaa.gov/data/nexrad-level-2/access/{}", filename),
                    source: "noaa".to_string(),
                });
            }
        }

        Ok(files)
    }

    async fn download_file(&self, url: &str) -> Result<bytes::Bytes, anyhow::Error> {
        info!("Downloading file from: {}", url);
        
        let response = self.client.get(url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("HTTP error: {}", response.status()));
        }
        
        let bytes = response.bytes().await?;
        Ok(bytes)
    }

    fn get_nexrad_sites(&self) -> Vec<NexradSite> {
        // Sample of major NEXRAD sites - in production, load from comprehensive database
        vec![
            NexradSite { id: "KTLX".to_string(), name: "Norman, OK".to_string(), state: "OK".to_string(), lat: 35.3331, lon: -97.2775, elevation_m: 370.0 },
            NexradSite { id: "KFWS".to_string(), name: "Dallas/Fort Worth, TX".to_string(), state: "TX".to_string(), lat: 32.5730, lon: -97.3031, elevation_m: 208.0 },
            NexradSite { id: "KEWX".to_string(), name: "Austin/San Antonio, TX".to_string(), state: "TX".to_string(), lat: 29.7038, lon: -98.0289, elevation_m: 193.0 },
            NexradSite { id: "KSHV".to_string(), name: "Shreveport, LA".to_string(), state: "LA".to_string(), lat: 32.4508, lon: -93.8414, elevation_m: 83.0 },
            NexradSite { id: "KLCH".to_string(), name: "Lake Charles, LA".to_string(), state: "LA".to_string(), lat: 30.1253, lon: -93.2161, elevation_m: 4.0 },
            NexradSite { id: "KLIX".to_string(), name: "New Orleans, LA".to_string(), state: "LA".to_string(), lat: 30.3367, lon: -89.8256, elevation_m: 7.0 },
            NexradSite { id: "KMOB".to_string(), name: "Mobile, AL".to_string(), state: "AL".to_string(), lat: 30.6794, lon: -88.2397, elevation_m: 63.0 },
            NexradSite { id: "KBMX".to_string(), name: "Birmingham, AL".to_string(), state: "AL".to_string(), lat: 33.1717, lon: -86.7700, elevation_m: 197.0 },
            NexradSite { id: "KHTX".to_string(), name: "Huntsville, AL".to_string(), state: "AL".to_string(), lat: 34.9306, lon: -86.0833, elevation_m: 537.0 },
            NexradSite { id: "KOHX".to_string(), name: "Nashville, TN".to_string(), state: "TN".to_string(), lat: 36.2472, lon: -86.5625, elevation_m: 176.0 },
        ]
    }
}

// API Handlers
async fn list_nexrad_files(Query(params): Query<NexradListRequest>) -> Result<Json<ApiResponse<Vec<NexradFile>>>, StatusCode> {
    let source = params.source.unwrap_or_else(|| "aws".to_string());
    let data_source = match DataSource::from_string(&source) {
        Some(ds) => ds,
        None => {
            return Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some("Invalid data source. Use: aws, gcp, datagov, or noaa".to_string()),
                source: None,
                metadata: None,
            }));
        }
    };

    // Validate date format
    if NaiveDate::parse_from_str(&params.date, "%Y-%m-%d").is_err() {
        return Ok(Json(ApiResponse {
            success: false,
            data: None,
            error: Some("Invalid date format. Use YYYY-MM-DD".to_string()),
            source: None,
            metadata: None,
        }));
    }

    let fetcher = NexradDataFetcher::new();
    
    match fetcher.list_files(&params.site, &params.date, data_source).await {
        Ok(files) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(files),
                error: None,
                source: Some(source),
                metadata: None,
            }))
        }
        Err(e) => {
            error!("Failed to list NEXRAD files: {}", e);
            Ok(Json(ApiResponse {
                success: false,
                data: None,
                error: Some(format!("Failed to retrieve files: {}", e)),
                source: None,
                metadata: None,
            }))
        }
    }
}

async fn download_nexrad_file(Path(file_id): Path<String>) -> Result<Response, StatusCode> {
    // In a real implementation, you'd decode the file_id to get the actual download URL
    // For demo purposes, return a placeholder response
    
    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/octet-stream")
        .header("Content-Disposition", "attachment; filename=\"nexrad_data.gz\"")
        .body("Binary NEXRAD data would be here".into())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(response)
}

async fn get_nexrad_sites() -> Json<ApiResponse<Vec<NexradSite>>> {
    let fetcher = NexradDataFetcher::new();
    let sites = fetcher.get_nexrad_sites();
    
    Json(ApiResponse {
        success: true,
        data: Some(sites),
        error: None,
        source: None,
        metadata: None,
    })
}

async fn health_check() -> Json<HealthResponse> {
    let mut sources = HashMap::new();
    sources.insert("aws".to_string(), "AWS S3 NEXRAD Level 2".to_string());
    sources.insert("gcp".to_string(), "Google Cloud Public Data".to_string());
    sources.insert("datagov".to_string(), "Data.gov CKAN API".to_string());
    sources.insert("noaa".to_string(), "NOAA NCEI Archive".to_string());
    
    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: Utc::now().to_rfc3339(),
        sources,
    })
}

async fn api_documentation() -> Json<serde_json::Value> {
    serde_json::json!({
        "name": "NEXRAD Level 2 Data API",
        "version": "1.0.0",
        "description": "Retrieve NEXRAD Level 2 radar data from multiple sources",
        "endpoints": {
            "/api/nexrad/files": {
                "method": "GET",
                "description": "List available NEXRAD files",
                "parameters": {
                    "site": "NEXRAD site ID (e.g., KTLX) (required)",
                    "date": "Date in YYYY-MM-DD format (required)",
                    "source": "Data source: aws, gcp, datagov, noaa (default: aws)"
                }
            },
            "/api/nexrad/sites": {
                "method": "GET",
                "description": "Get list of NEXRAD sites"
            },
            "/api/nexrad/download/{file_id}": {
                "method": "GET",
                "description": "Download a specific NEXRAD file"
            },
            "/api/health": {
                "method": "GET",
                "description": "Health check and data source status"
            }
        },
        "data_sources": {
            "aws": "AWS S3 noaa-nexrad-level2 bucket",
            "gcp": "Google Cloud gcp-public-data-nexrad-l2 bucket",  
            "datagov": "Data.gov CKAN API",
            "noaa": "NOAA NCEI Archive"
        },
        "examples": {
            "list_files": "/api/nexrad/files?site=KTLX&date=2024-06-15&source=aws",
            "get_sites": "/api/nexrad/sites"
        }
    })
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting NEXRAD Level 2 Data Backend Server...");
    
    // Build the application with routes
    let app = Router::new()
        .route("/", get(api_documentation))
        .route("/api/nexrad/files", get(list_nexrad_files))
        .route("/api/nexrad/sites", get(get_nexrad_sites))
        .route("/api/nexrad/download/:file_id", get(download_nexrad_file))
        .route("/api/health", get(health_check))
        .layer(CorsLayer::permissive());
    
    // Start the server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:5001")
        .await
        .expect("Failed to bind to address");
    
    info!("NEXRAD API server running at: http://localhost:5001");
    info!("Example: http://localhost:5001/api/nexrad/files?site=KTLX&date=2024-06-15&source=aws");
    
    axum::serve(listener, app)
        .await
        .expect("Server failed to start");
}
