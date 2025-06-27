use anyhow::{Result, Context};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::Parser;
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;
use tiff::encoder::{TiffEncoder, colortype};
use tiff::tags::Tag;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input NEXRAD Level 2 file (can be gzipped)
    input: String,
    
    /// Output directory for GeoTIFF files
    #[arg(short, long, default_value = ".")]
    output: String,
    
    /// Grid resolution in meters
    #[arg(short, long, default_value = "1000")]
    resolution: u32,
    
    /// Maximum range in km
    #[arg(short, long, default_value = "230")]
    max_range: f64,
    
    /// Process only specific sweep (0-based index)
    #[arg(short, long)]
    sweep: Option<usize>,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Clone)]
struct RadarSite {
    id: String,
    latitude: f64,
    longitude: f64,
    elevation: f64,
}

#[derive(Debug)]
struct RadialData {
    azimuth: f64,
    elevation: f64,
    reflectivity: Vec<f32>,
    velocity: Vec<f32>,
    range_gate_size: f32,
    first_gate_range: f32,
}

#[derive(Debug)]
struct SweepData {
    radials: Vec<RadialData>,
    sweep_number: u16,
    elevation_angle: f32,
}

#[derive(Debug)]
struct GeoTransform {
    pixel_size_x: f64,
    rotation_x: f64,
    top_left_x: f64,
    pixel_size_y: f64,
    rotation_y: f64,
    top_left_y: f64,
}

struct NexradReader<R: Read + Seek> {
    reader: R,
    radar_site: Option<RadarSite>,
}

impl<R: Read + Seek> NexradReader<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            radar_site: None,
        }
    }

    fn read_volume_header(&mut self) -> Result<()> {
        let mut buffer = [0u8; 24];
        self.reader.read_exact(&mut buffer)?;
        
        // Extract radar site ID from tape header
        let tape_header = String::from_utf8_lossy(&buffer[0..9]);
        if tape_header.starts_with("AR2V") {
            // This is a valid NEXRAD file
            let site_id = String::from_utf8_lossy(&buffer[20..24]).trim().to_string();
            
            // Use predefined radar site locations (simplified)
            self.radar_site = Some(get_radar_site(&site_id));
        }
        
        Ok(())
    }

    fn read_message(&mut self) -> Result<Option<SweepData>> {
        let mut size_bytes = [0u8; 2];
        if self.reader.read_exact(&mut size_bytes).is_err() {
            return Ok(None); // EOF
        }
        
        let message_size = u16::from_be_bytes(size_bytes) as usize;
        if message_size == 0 {
            return Ok(None);
        }

        let mut message_data = vec![0u8; message_size * 2 - 2];
        self.reader.read_exact(&mut message_data)?;
        
        let mut cursor = Cursor::new(&message_data);
        
        // Skip redundant channel and message type
        cursor.seek(SeekFrom::Start(2))?;
        let message_type = cursor.read_u8()?;
        
        if message_type == 31 {
            // Digital Radar Data message
            return self.parse_message_31(&mut cursor);
        }
        
        Ok(None)
    }

    fn parse_message_31(&mut self, cursor: &mut Cursor<&Vec<u8>>) -> Result<Option<SweepData>> {
        cursor.seek(SeekFrom::Start(12))?; // Skip to radial data
        
        let _radial_number = cursor.read_u16::<BigEndian>()?;
        let _radial_status = cursor.read_u16::<BigEndian>()?;
        let elevation_angle = cursor.read_u16::<BigEndian>()? as f32 / 8.0 * (180.0 / 32768.0);
        let azimuth_angle = cursor.read_u16::<BigEndian>()? as f32 / 8.0 * (180.0 / 32768.0);
        
        cursor.seek(SeekFrom::Current(38))?; // Skip to data moment pointers
        
        let ref_pointer = cursor.read_u32::<BigEndian>()?;
        let vel_pointer = cursor.read_u32::<BigEndian>()?;
        let _sw_pointer = cursor.read_u32::<BigEndian>()?;
        
        let mut reflectivity = Vec::new();
        let mut velocity = Vec::new();
        let mut range_gate_size = 250.0; // Default
        let mut first_gate_range = 0.0;
        
        // Read reflectivity data
        if ref_pointer > 0 {
            cursor.seek(SeekFrom::Start(ref_pointer as u64))?;
            let block_id = cursor.read_u32::<BigEndian>()?;
            if block_id == 1 { // Reflectivity block
                let gate_count = cursor.read_u16::<BigEndian>()?;
                first_gate_range = cursor.read_u16::<BigEndian>()? as f32;
                range_gate_size = cursor.read_u16::<BigEndian>()? as f32;
                let _tover = cursor.read_u16::<BigEndian>()?;
                let _snr_threshold = cursor.read_u16::<BigEndian>()?;
                let _control_flags = cursor.read_u8()?;
                let _data_size = cursor.read_u8()?;
                let _scale = cursor.read_f32::<BigEndian>()?;
                let _offset = cursor.read_f32::<BigEndian>()?;
                
                for _ in 0..gate_count {
                    let raw_value = cursor.read_u8()?;
                    let dbz = if raw_value == 0 {
                        f32::NAN
                    } else {
                        (raw_value as f32 - 2.0) / 2.0 - 32.0
                    };
                    reflectivity.push(dbz);
                }
            }
        }
        
        // Read velocity data
        if vel_pointer > 0 {
            cursor.seek(SeekFrom::Start(vel_pointer as u64))?;
            let block_id = cursor.read_u32::<BigEndian>()?;
            if block_id == 1 { // Velocity block
                let gate_count = cursor.read_u16::<BigEndian>()?;
                cursor.seek(SeekFrom::Current(10))?; // Skip header fields
                let _scale = cursor.read_f32::<BigEndian>()?;
                let _offset = cursor.read_f32::<BigEndian>()?;
                
                for _ in 0..gate_count {
                    let raw_value = cursor.read_u8()?;
                    let vel = if raw_value == 0 || raw_value == 1 {
                        f32::NAN
                    } else {
                        (raw_value as f32 - 129.0) / 2.0
                    };
                    velocity.push(vel);
                }
            }
        }
        
        let radial = RadialData {
            azimuth: azimuth_angle as f64,
            elevation: elevation_angle as f64,
            reflectivity,
            velocity,
            range_gate_size,
            first_gate_range,
        };
        
        // For simplicity, return each radial as a separate sweep
        // In practice, you'd accumulate radials into complete sweeps
        Ok(Some(SweepData {
            radials: vec![radial],
            sweep_number: 0,
            elevation_angle,
        }))
    }
}

fn get_radar_site(site_id: &str) -> RadarSite {
    // Simplified radar site database - in practice, load from file
    let sites = HashMap::from([
        ("KTLX", (35.3331, -97.2778, 370.0)),
        ("KOUN", (35.2366, -97.4608, 384.0)),
        ("KDDC", (37.7608, -99.9689, 789.0)),
        ("KFDR", (34.3623, -98.9763, 386.0)),
        ("KINX", (36.1750, -95.5644, 204.0)),
        ("KICT", (37.6544, -97.4428, 407.0)),
        ("KVNX", (36.7406, -98.1278, 369.0)),
        // Add more sites as needed
    ]);
    
    let (lat, lon, elev) = sites.get(site_id).unwrap_or(&(35.0, -97.0, 300.0));
    
    RadarSite {
        id: site_id.to_string(),
        latitude: *lat,
        longitude: *lon,
        elevation: *elev,
    }
}

fn grid_radar_data(
    sweeps: &[SweepData],
    radar_site: &RadarSite,
    resolution: u32,
    max_range: f64,
) -> Result<(Vec<f32>, Vec<f32>, GeoTransform, (usize, usize))> {
    let grid_size = ((max_range * 2000.0) / resolution as f64) as usize;
    let mut ref_grid = vec![f32::NAN; grid_size * grid_size];
    let mut vel_grid = vec![f32::NAN; grid_size * grid_size];
    
    let center_x = grid_size / 2;
    let center_y = grid_size / 2;
    
    for sweep in sweeps {
        for radial in &sweep.radials {
            let az_rad = radial.azimuth * PI / 180.0;
            
            for (i, (&ref_val, &vel_val)) in radial.reflectivity.iter()
                .zip(radial.velocity.iter()).enumerate() {
                
                let range_km = (radial.first_gate_range + i as f32 * radial.range_gate_size) / 1000.0;
                if range_km > max_range as f32 {
                    break;
                }
                
                let x = range_km as f64 * az_rad.sin();
                let y = range_km as f64 * az_rad.cos();
                
                let grid_x = (center_x as f64 + x * 1000.0 / resolution as f64) as isize;
                let grid_y = (center_y as f64 - y * 1000.0 / resolution as f64) as isize;
                
                if grid_x >= 0 && grid_x < grid_size as isize && 
                   grid_y >= 0 && grid_y < grid_size as isize {
                    let idx = grid_y as usize * grid_size + grid_x as usize;
                    
                    if !ref_val.is_nan() {
                        ref_grid[idx] = ref_val;
                    }
                    if !vel_val.is_nan() {
                        vel_grid[idx] = vel_val;
                    }
                }
            }
        }
    }
    
    // Calculate geotransform
    let pixel_size = resolution as f64 / 111320.0; // Convert meters to degrees (approximate)
    let top_left_lon = radar_site.longitude - (max_range / 111.32);
    let top_left_lat = radar_site.latitude + (max_range / 111.32);
    
    let geotransform = GeoTransform {
        pixel_size_x: pixel_size,
        rotation_x: 0.0,
        top_left_x: top_left_lon,
        pixel_size_y: -pixel_size,
        rotation_y: 0.0,
        top_left_y: top_left_lat,
    };
    
    Ok((ref_grid, vel_grid, geotransform, (grid_size, grid_size)))
}

fn write_geotiff(
    data: &[f32],
    filename: &str,
    geotransform: &GeoTransform,
    size: (usize, usize),
) -> Result<()> {
    let file = File::create(filename)?;
    let mut tiff = TiffEncoder::new(BufWriter::new(file))?;
    
    // Create image with proper dimensions and sample format
    let mut image = tiff.new_image::<colortype::Gray32Float>(size.0 as u32, size.1 as u32)?;
    
    // Write tags one by one to avoid borrowing issues
    let encoder = image.encoder();
    
    // Write basic TIFF tags
    encoder.write_tag(Tag::Compression, 1u16)?; // No compression
    encoder.write_tag(Tag::PlanarConfiguration, 1u16)?; // Chunky
    encoder.write_tag(Tag::SampleFormat, 3u16)?; // IEEE floating point
    
    // ModelPixelScaleTag - write individual values
    encoder.write_tag(Tag::Unknown(33550), vec![
        geotransform.pixel_size_x,
        geotransform.pixel_size_y.abs(),
        0.0
    ])?;
    
    // ModelTiepointTag
    encoder.write_tag(Tag::Unknown(33922), vec![
        0.0, 0.0, 0.0,
        geotransform.top_left_x, geotransform.top_left_y, 0.0
    ])?;
    
    // GeoKeyDirectoryTag
    encoder.write_tag(Tag::Unknown(34735), vec![
        1u16, 1, 0, 4,      // Header: version=1, revision=1, minor=0, numberOfKeys=4
        1024, 0, 1, 1,      // GTModelTypeGeoKey = ModelTypeGeographic
        1025, 0, 1, 1,      // GTRasterTypeGeoKey = RasterPixelIsArea  
        2048, 0, 1, 4326,   // GeographicTypeGeoKey = GCS_WGS_84
        2054, 0, 1, 9102    // GeogAngularUnitsGeoKey = Angular_Degree
    ])?;
    
    // Write the image data and finish - these methods consume the image
    image.write_data(data)?;
    
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner()
        .template("{spinner:.green} {msg}")
        .unwrap());
    
    pb.set_message("Opening NEXRAD file...");
    
    // Open and decompress file if needed
    let file = File::open(&args.input)
        .context("Failed to open input file")?;
    
    // Read all data into memory first (simpler approach)
    let mut data = Vec::new();
    if args.input.ends_with(".gz") {
        let mut decoder = GzDecoder::new(file);
        decoder.read_to_end(&mut data)?;
    } else {
        let mut reader = BufReader::new(file);
        reader.read_to_end(&mut data)?;
    };
    
    let cursor = Cursor::new(data);
    
    let mut nexrad = NexradReader::new(cursor);
    nexrad.read_volume_header()?;
    
    pb.set_message("Reading radar data...");
    
    let mut sweeps = Vec::new();
    while let Some(sweep) = nexrad.read_message()? {
        sweeps.push(sweep);
        if args.verbose {
            pb.set_message(&format!("Read {} sweeps", sweeps.len()));
        }
    }
    
    if sweeps.is_empty() {
        anyhow::bail!("No valid radar data found in file");
    }
    
    let radar_site = nexrad.radar_site
        .ok_or_else(|| anyhow::anyhow!("Could not determine radar site"))?;
    
    pb.set_message("Gridding data...");
    
    let (ref_grid, vel_grid, geotransform, size) = grid_radar_data(
        &sweeps,
        &radar_site,
        args.resolution,
        args.max_range,
    )?;
    
    // Generate output filenames
    let input_stem = Path::new(&args.input)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("nexrad");
    
    let ref_filename = format!("{}/{}_reflectivity.tif", args.output, input_stem);
    let vel_filename = format!("{}/{}_velocity.tif", args.output, input_stem);
    
    pb.set_message("Writing reflectivity GeoTIFF...");
    write_geotiff(&ref_grid, &ref_filename, &geotransform, size)?;
    
    pb.set_message("Writing velocity GeoTIFF...");
    write_geotiff(&vel_grid, &vel_filename, &geotransform, size)?;
    
    pb.finish_with_message(&format!(
        "✓ Conversion complete!\n  Reflectivity: {}\n  Velocity: {}",
        ref_filename, vel_filename
    ));
    
    println!("Radar Site: {} ({:.4}, {:.4})", 
             radar_site.id, radar_site.latitude, radar_site.longitude);
    println!("Processed {} sweeps", sweeps.len());
    println!("Grid size: {}x{} at {}m resolution", size.0, size.1, args.resolution);
    
    Ok(())
}
