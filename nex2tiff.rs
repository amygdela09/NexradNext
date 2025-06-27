use anyhow::{Context, Result};
use byteorder::{BigEndian, ReadBytesExt};
use clap::Parser;
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use tiff::encoder::{colortype, TiffEncoder};
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

#[derive(Debug, Clone)]
struct RadialData {
    azimuth: f64,
    elevation: f64,
    reflectivity: Vec<f32>,
    velocity: Vec<f32>,
    range_gate_size: f32,
    first_gate_range: f32,
    sweep_number: u8,
}

#[derive(Debug, Default, Clone)]
struct SweepData {
    radials: Vec<RadialData>,
    sweep_number: u8,
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

struct NexradFile {
    reader: Cursor<Vec<u8>>,
    radar_site: RadarSite,
}

impl NexradFile {
    fn new(data: Vec<u8>) -> Result<Self> {
        let mut reader = Cursor::new(data);

        let mut buffer = [0u8; 24];
        reader.read_exact(&mut buffer)?;

        let tape_header = String::from_utf8_lossy(&buffer[0..9]);
        if !tape_header.starts_with("AR2V") {
            anyhow::bail!("Not a valid NEXRAD Level 2 file (missing AR2V header).");
        }
        let site_id = String::from_utf8_lossy(&buffer[20..24]).trim().to_string();
        let radar_site = get_radar_site(&site_id);

        Ok(Self { reader, radar_site })
    }

    fn read_sweeps(&mut self) -> Result<Vec<SweepData>> {
        let mut sweeps: HashMap<u8, SweepData> = HashMap::new();
        self.reader.seek(SeekFrom::Start(24))?;

        while let Ok(record_size) = self.reader.read_i32::<BigEndian>() {
            let record_size = record_size.abs() as usize;
            if record_size == 0 { continue; }

            // The record size is for the LDM (Local Data Manager) wrapper, skip it.
            // The actual message is what follows.
            let mut message_buffer = vec![0; record_size];
            if self.reader.read_exact(&mut message_buffer).is_err() {
                break; // End of file or incomplete record
            }
            
            let mut cursor = Cursor::new(message_buffer);
            cursor.seek(SeekFrom::Start(12))?; // Skip CTM header

            let message_type = cursor.read_u8()?;
            if message_type == 31 {
                // We have a digital radar data message, parse it
                if let Ok(Some(radial)) = parse_radial(cursor.into_inner()) {
                    let sweep = sweeps.entry(radial.sweep_number).or_insert_with(|| SweepData {
                        sweep_number: radial.sweep_number,
                        elevation_angle: radial.elevation as f32,
                        ..Default::default()
                    });
                    sweep.radials.push(radial);
                }
            }
        }
        Ok(sweeps.into_values().collect())
    }
}

fn parse_radial(message_data: Vec<u8>) -> Result<Option<RadialData>> {
    let mut cursor = Cursor::new(message_data);
    cursor.seek(SeekFrom::Start(12))?; // Skip CTM, go to start of message body
    
    let _message_type = cursor.read_u8()?; // Should be 31
    let _id_sequence = cursor.read_u16::<BigEndian>()?;
    let _julian_date = cursor.read_u16::<BigEndian>()?;
    let _milliseconds_of_day = cursor.read_u32::<BigEndian>()?;
    let _unambiguous_range = cursor.read_u16::<BigEndian>()?;

    let azimuth = cursor.read_u16::<BigEndian>()? as f32 * (360.0 / 65536.0);
    let _azimuth_spacing = cursor.read_u8()? as f32 * 0.125;
    let _radial_status = cursor.read_u8()?;
    let elevation = cursor.read_u16::<BigEndian>()? as f32 * (180.0 / 32768.0);
    let sweep_number = cursor.read_u8()?;
    
    cursor.seek(SeekFrom::Start(54))?; // Absolute position of data pointers
    let ref_pointer = cursor.read_u32::<BigEndian>()? as u64;
    let vel_pointer = cursor.read_u32::<BigEndian>()? as u64;
    let _sw_pointer = cursor.read_u32::<BigEndian>()? as u64;

    let mut reflectivity = Vec::new();
    let mut velocity = Vec::new();
    let mut range_gate_size = 1000.0;
    let mut first_gate_range = 0.0;

    if ref_pointer > 0 {
        cursor.seek(SeekFrom::Start(12 + ref_pointer))?; // Pointers are relative to CTM header
        let _data_block_type = cursor.read_u8()?;
        let _data_name = cursor.read_u32::<BigEndian>()?;
        let _reserved = cursor.read_u32::<BigEndian>()?;
        let gate_count = cursor.read_u16::<BigEndian>()?;
        first_gate_range = cursor.read_u16::<BigEndian>()? as f32;
        range_gate_size = cursor.read_u16::<BigEndian>()? as f32;
        let _tover = cursor.read_u16::<BigEndian>()?;
        let _snr_threshold = cursor.read_u16::<BigEndian>()?;
        let _control_flags = cursor.read_u8()?;
        let _data_size = cursor.read_u8()?;
        let scale = cursor.read_f32::<BigEndian>()?;
        let offset = cursor.read_f32::<BigEndian>()?;

        for _ in 0..gate_count {
            let raw_value = cursor.read_u8()?;
            reflectivity.push(if raw_value <= 1 { f32::NAN } else { (raw_value as f32 - offset) / scale });
        }
    }

    if vel_pointer > 0 {
        cursor.seek(SeekFrom::Start(12 + vel_pointer))?;
        let _data_block_type = cursor.read_u8()?;
        let _data_name = cursor.read_u32::<BigEndian>()?;
        let _reserved = cursor.read_u32::<BigEndian>()?;
        let gate_count = cursor.read_u16::<BigEndian>()?;
        // Not using these, but need to read to advance cursor
        let _first_gate_range = cursor.read_u16::<BigEndian>()?;
        let _range_gate_size = cursor.read_u16::<BigEndian>()?;
        let _tover = cursor.read_u16::<BigEndian>()?;
        let _snr_threshold = cursor.read_u16::<BigEndian>()?;
        let _control_flags = cursor.read_u8()?;
        let _data_size = cursor.read_u8()?;
        let scale = cursor.read_f32::<BigEndian>()?;
        let offset = cursor.read_f32::<BigEndian>()?;

        for _ in 0..gate_count {
            let raw_value = cursor.read_u8()?;
            velocity.push(if raw_value <= 1 { f32::NAN } else { (raw_value as f32 - offset) / scale });
        }
    }
    
    let max_len = reflectivity.len().max(velocity.len());
    reflectivity.resize(max_len, f32::NAN);
    velocity.resize(max_len, f32::NAN);

    Ok(Some(RadialData {
        azimuth: azimuth as f64,
        elevation: elevation as f64,
        reflectivity,
        velocity,
        range_gate_size,
        first_gate_range,
        sweep_number,
    }))
}

fn get_radar_site(site_id: &str) -> RadarSite {
    let sites = HashMap::from([
        ("KTLX", (35.3331, -97.2778, 370.0)),
        ("KOUN", (35.2366, -97.4608, 384.0)),
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
            for i in 0..radial.reflectivity.len() {
                let ref_val = radial.reflectivity[i];
                let vel_val = radial.velocity[i];
                let range_m = radial.first_gate_range + i as f32 * radial.range_gate_size;
                if range_m / 1000.0 > max_range as f32 { break; }

                let x = range_m as f64 * az_rad.sin();
                let y = range_m as f64 * az_rad.cos();
                let grid_x = (center_x as f64 + x / resolution as f64) as isize;
                let grid_y = (center_y as f64 - y / resolution as f64) as isize;

                if grid_x >= 0 && grid_x < grid_size as isize && grid_y >= 0 && grid_y < grid_size as isize {
                    let idx = grid_y as usize * grid_size + grid_x as usize;
                    if !ref_val.is_nan() { ref_grid[idx] = ref_val; }
                    if !vel_val.is_nan() { vel_grid[idx] = vel_val; }
                }
            }
        }
    }

    let pixel_size_deg = resolution as f64 / 111_320.0;
    let top_left_lon = radar_site.longitude - (grid_size as f64 / 2.0 * pixel_size_deg);
    let top_left_lat = radar_site.latitude + (grid_size as f64 / 2.0 * pixel_size_deg);

    let geotransform = GeoTransform {
        pixel_size_x: pixel_size_deg,
        rotation_x: 0.0,
        top_left_x: top_left_lon,
        pixel_size_y: -pixel_size_deg,
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
    let mut image = tiff.new_image::<colortype::Gray32Float>(size.0 as u32, size.1 as u32)?;

    image.encoder().write_tag(Tag::Compression, 1u16)?;
    image.encoder().write_tag(Tag::PlanarConfiguration, 1u16)?;
    image.encoder().write_tag(Tag::SampleFormat, 3u16)?;

    let pixel_scale = vec![geotransform.pixel_size_x, geotransform.pixel_size_y.abs(), 0.0];
    image.encoder().write_tag(Tag::Unknown(33550), &*pixel_scale)?;

    let tiepoint = vec![0.0, 0.0, 0.0, geotransform.top_left_x, geotransform.top_left_y, 0.0];
    image.encoder().write_tag(Tag::Unknown(33922), &*tiepoint)?;

    let geo_key_directory: Vec<u16> = vec![
        1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 2048, 0, 1, 4326, 2054, 0, 1, 9102,
    ];
    image.encoder().write_tag(Tag::Unknown(34735), &*geo_key_directory)?;

    image.write_data(data)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());

    pb.set_message("Opening NEXRAD file...");
    let file = File::open(&args.input).context("Failed to open input file")?;
    let mut data = Vec::new();
    if args.input.ends_with(".gz") {
        GzDecoder::new(file).read_to_end(&mut data)?;
    } else {
        BufReader::new(file).read_to_end(&mut data)?;
    }

    let mut nexrad_file = NexradFile::new(data)?;
    let radar_site = nexrad_file.radar_site.clone();

    pb.set_message("Reading radar data...");
    let sweeps = nexrad_file.read_sweeps()?;

    if sweeps.is_empty() {
        anyhow::bail!("No valid radar sweeps found in file. The file may be empty or in an unsupported format.");
    }

    pb.set_message("Gridding data...");
    let (ref_grid, vel_grid, geotransform, size) =
        grid_radar_data(&sweeps, &radar_site, args.resolution, args.max_range)?;

    let input_stem = Path::new(&args.input).file_stem().and_then(|s| s.to_str()).unwrap_or("nexrad");
    let ref_filename = format!("{}/{}_reflectivity.tif", args.output, input_stem);
    let vel_filename = format!("{}/{}_velocity.tif", args.output, input_stem);

    pb.set_message("Writing reflectivity GeoTIFF...");
    write_geotiff(&ref_grid, &ref_filename, &geotransform, size)?;

    pb.set_message("Writing velocity GeoTIFF...");
    write_geotiff(&vel_grid, &vel_filename, &geotransform, size)?;

    pb.finish_with_message(format!(
        "✓ Conversion complete!\n  Reflectivity: {}\n  Velocity: {}",
        ref_filename, vel_filename
    ));

    println!("Radar Site: {} ({:.4}, {:.4})", radar_site.id, radar_site.latitude, radar_site.longitude);
    println!("Processed {} sweeps", sweeps.len());
    println!("Grid size: {}x{} at {}m resolution", size.0, size.1, args.resolution);

    Ok(())
}
