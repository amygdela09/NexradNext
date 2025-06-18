#!/usr/bin/env python3
"""
NEXRAD Level 2 3D Visualizer
A Python program to visualize NEXRAD Level 2 radar data in 3D
Similar functionality to GR2Analyst
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import pyart
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NEXRADViewer3D:
    def __init__(self):
        self.radar = None
        self.fig = None
        self.ax = None
        self.current_sweep = 0
        self.current_field = 'reflectivity'
        self.elevation_angles = []
        
        # Reflectivity colormap and bounds (dBZ)
        self.ref_colors = ['#646464', '#04e9e7', '#019ff4', '#0300f4', '#02fd02',
                          '#01c501', '#008e00', '#fdf802', '#e5bc00', '#fd9500',
                          '#fd0000', '#d40000', '#bc0000', '#f800fd', '#9854c6']
        self.ref_bounds = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 
                          30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
        
        # Velocity colormap (m/s)
        self.vel_colors = ['#730026', '#e6194b', '#911eb4', '#4363d8', '#42d4f4',
                          '#469990', '#9a6324', '#800000', '#000075', '#dcbeff']
        
    def load_radar_file(self, filepath):
        """Load NEXRAD Level 2 file using PyART"""
        try:
            print(f"Loading radar file: {filepath}")
            self.radar = pyart.io.read_nexrad_archive(filepath)
            
            # Get elevation angles
            self.elevation_angles = np.round(self.radar.elevation['data'], 1)
            unique_elevations = np.unique(self.elevation_angles)
            
            print(f"Loaded radar data with {len(unique_elevations)} elevation angles:")
            print(f"Elevations: {unique_elevations}")
            print(f"Available fields: {list(self.radar.fields.keys())}")
            
            # Set default field
            if 'reflectivity' in self.radar.fields:
                self.current_field = 'reflectivity'
            elif 'velocity' in self.radar.fields:
                self.current_field = 'velocity'
            else:
                self.current_field = list(self.radar.fields.keys())[0]
                
            print(f"Using field: {self.current_field}")
            return True
            
        except Exception as e:
            print(f"Error loading radar file: {e}")
            return False
    
    def get_3d_coordinates(self, sweep_idx=None):
        """Convert radar data to 3D Cartesian coordinates"""
        if self.radar is None:
            return None, None, None, None
            
        if sweep_idx is None:
            # Use all sweeps
            sweeps = range(self.radar.nsweeps)
        else:
            sweeps = [sweep_idx]
            
        x_coords, y_coords, z_coords, values = [], [], [], []
        
        for sweep in sweeps:
            # Get sweep data
            sweep_start = self.radar.sweep_start_ray_index['data'][sweep]
            sweep_end = self.radar.sweep_end_ray_index['data'][sweep]
            
            # Get ranges, azimuths, and elevation
            ranges = self.radar.range['data']
            azimuths = self.radar.azimuth['data'][sweep_start:sweep_end+1]
            elevation = self.radar.elevation['data'][sweep_start:sweep_end+1]
            
            # Get field data
            field_data = self.radar.fields[self.current_field]['data'][sweep_start:sweep_end+1]
            
            # Convert to Cartesian coordinates
            for i, (az, el) in enumerate(zip(azimuths, elevation)):
                az_rad = np.deg2rad(az)
                el_rad = np.deg2rad(el)
                
                # Calculate 3D coordinates for each range gate
                for j, r in enumerate(ranges):
                    if not np.ma.is_masked(field_data[i, j]):
                        x = r * np.cos(el_rad) * np.sin(az_rad)
                        y = r * np.cos(el_rad) * np.cos(az_rad)
                        z = r * np.sin(el_rad)
                        
                        x_coords.append(x)
                        y_coords.append(y)
                        z_coords.append(z)
                        values.append(field_data[i, j])
        
        return np.array(x_coords), np.array(y_coords), np.array(z_coords), np.array(values)
    
    def create_colormap(self, field_type):
        """Create appropriate colormap for the field"""
        if field_type == 'reflectivity':
            cmap = colors.ListedColormap(self.ref_colors)
            norm = colors.BoundaryNorm(self.ref_bounds, cmap.N)
            return cmap, norm
        elif field_type == 'velocity':
            cmap = plt.cm.RdYlBu_r
            norm = colors.Normalize(vmin=-30, vmax=30)
            return cmap, norm
        else:
            cmap = plt.cm.viridis
            norm = colors.Normalize()
            return cmap, norm
    
    def plot_3d_volume(self, max_points=50000, elevation_filter=None):
        """Create 3D volume visualization"""
        if self.radar is None:
            print("No radar data loaded!")
            return
            
        print("Generating 3D coordinates...")
        x, y, z, values = self.get_3d_coordinates()
        
        if x is None or len(x) == 0:
            print("No valid data points found!")
            return
            
        # Filter by elevation if specified
        if elevation_filter is not None:
            mask = np.abs(np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2))) - elevation_filter) < 0.5
            x, y, z, values = x[mask], y[mask], z[mask], values[mask]
        
        # Subsample if too many points
        if len(x) > max_points:
            indices = np.random.choice(len(x), max_points, replace=False)
            x, y, z, values = x[indices], y[indices], z[indices], values[indices]
            print(f"Subsampled to {max_points} points")
        
        # Create 3D plot
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Get colormap
        cmap, norm = self.create_colormap(self.current_field)
        
        # Create scatter plot
        scatter = self.ax.scatter(x/1000, y/1000, z/1000, c=values, 
                                cmap=cmap, norm=norm, alpha=0.6, s=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=self.ax, shrink=0.8)
        if self.current_field == 'reflectivity':
            cbar.set_label('Reflectivity (dBZ)', fontsize=12)
        elif self.current_field == 'velocity':
            cbar.set_label('Velocity (m/s)', fontsize=12)
        else:
            cbar.set_label(f'{self.current_field}', fontsize=12)
        
        # Set labels and title
        self.ax.set_xlabel('East-West Distance (km)', fontsize=12)
        self.ax.set_ylabel('North-South Distance (km)', fontsize=12)
        self.ax.set_zlabel('Height (km)', fontsize=12)
        
        # Add radar location
        self.ax.scatter([0], [0], [0], c='red', s=100, marker='^', 
                       label='Radar Location')
        
        # Get radar info
        site_name = self.radar.metadata.get('instrument_name', 'Unknown')
        scan_time = self.radar.time['units'].split(' ')[-1]
        
        self.ax.set_title(f'NEXRAD {site_name} - {self.current_field.title()}\n'
                         f'Scan Time: {scan_time}', fontsize=14)
        
        # Set aspect ratio
        max_range = 250  # km
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([0, 20])
        
        plt.legend()
        plt.tight_layout()
        print(f"Plotted {len(x)} data points")
        
    def plot_ppi_3d(self, sweep_idx=0):
        """Plot Plan Position Indicator (PPI) in 3D"""
        if self.radar is None:
            print("No radar data loaded!")
            return
            
        print(f"Plotting PPI for sweep {sweep_idx}")
        
        # Get sweep data
        sweep_start = self.radar.sweep_start_ray_index['data'][sweep_idx]
        sweep_end = self.radar.sweep_end_ray_index['data'][sweep_idx]
        
        # Get coordinates for this sweep
        ranges = self.radar.range['data']
        azimuths = self.radar.azimuth['data'][sweep_start:sweep_end+1]
        elevation = self.radar.elevation['data'][sweep_start]
        
        # Get field data
        field_data = self.radar.fields[self.current_field]['data'][sweep_start:sweep_end+1]
        
        # Create mesh grid
        R, A = np.meshgrid(ranges, azimuths)
        
        # Convert to Cartesian
        el_rad = np.deg2rad(elevation)
        X = R * np.cos(el_rad) * np.sin(np.deg2rad(A))
        Y = R * np.cos(el_rad) * np.cos(np.deg2rad(A))
        Z = R * np.sin(el_rad)
        
        # Create plot
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Get colormap
        cmap, norm = self.create_colormap(self.current_field)
        
        # Plot surface
        surf = self.ax.plot_surface(X/1000, Y/1000, Z/1000, facecolors=cmap(norm(field_data)),
                                   alpha=0.8, linewidth=0, antialiased=True)
        
        # Set labels
        self.ax.set_xlabel('East-West Distance (km)')
        self.ax.set_ylabel('North-South Distance (km)')
        self.ax.set_zlabel('Height (km)')
        self.ax.set_title(f'PPI Sweep {sweep_idx} - Elevation {elevation:.1f}°')
        
        # Add colorbar
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(field_data)
        plt.colorbar(mappable, ax=self.ax, shrink=0.8)
        
        plt.tight_layout()
    
    def animate_sweeps(self, interval=1000):
        """Animate through elevation sweeps"""
        if self.radar is None:
            print("No radar data loaded!")
            return
            
        self.fig, self.ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': '3d'})
        
        def update(frame):
            self.ax.clear()
            self.current_sweep = frame
            
            # Plot current sweep
            sweep_start = self.radar.sweep_start_ray_index['data'][frame]
            sweep_end = self.radar.sweep_end_ray_index['data'][frame]
            
            ranges = self.radar.range['data']
            azimuths = self.radar.azimuth['data'][sweep_start:sweep_end+1]
            elevation = self.radar.elevation['data'][sweep_start]
            field_data = self.radar.fields[self.current_field]['data'][sweep_start:sweep_end+1]
            
            # Convert to 3D coordinates
            R, A = np.meshgrid(ranges, azimuths)
            el_rad = np.deg2rad(elevation)
            X = R * np.cos(el_rad) * np.sin(np.deg2rad(A))
            Y = R * np.cos(el_rad) * np.cos(np.deg2rad(A))
            Z = R * np.sin(el_rad)
            
            # Plot
            cmap, norm = self.create_colormap(self.current_field)
            self.ax.plot_surface(X/1000, Y/1000, Z/1000, facecolors=cmap(norm(field_data)),
                               alpha=0.8, linewidth=0)
            
            self.ax.set_xlabel('East-West (km)')
            self.ax.set_ylabel('North-South (km)')
            self.ax.set_zlabel('Height (km)')
            self.ax.set_title(f'Sweep {frame} - Elevation {elevation:.1f}°')
            
            # Set limits
            self.ax.set_xlim([-250, 250])
            self.ax.set_ylim([-250, 250])
            self.ax.set_zlim([0, 20])
        
        anim = FuncAnimation(self.fig, update, frames=self.radar.nsweeps, 
                           interval=interval, repeat=True)
        return anim
    
    def show(self):
        """Display the plot"""
        if self.fig is not None:
            plt.show()
        else:
            print("No plot to display!")
    
    def save_plot(self, filename):
        """Save the current plot"""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {filename}")
        else:
            print("No plot to save!")

def main():
    parser = argparse.ArgumentParser(description='NEXRAD Level 2 3D Visualizer')
    parser.add_argument('file', help='Path to NEXRAD Level 2 file')
    parser.add_argument('--field', default='reflectivity', 
                       help='Field to visualize (reflectivity, velocity, etc.)')
    parser.add_argument('--mode', default='volume', choices=['volume', 'ppi', 'animate'],
                       help='Visualization mode')
    parser.add_argument('--sweep', type=int, default=0,
                       help='Sweep index for PPI mode')
    parser.add_argument('--max-points', type=int, default=50000,
                       help='Maximum points for volume rendering')
    parser.add_argument('--save', help='Save plot to file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return
    
    # Create viewer
    viewer = NEXRADViewer3D()
    
    # Load radar data
    if not viewer.load_radar_file(args.file):
        return
    
    # Set field
    if args.field in viewer.radar.fields:
        viewer.current_field = args.field
    else:
        print(f"Field '{args.field}' not available. Using {viewer.current_field}")
    
    # Create visualization
    if args.mode == 'volume':
        viewer.plot_3d_volume(max_points=args.max_points)
    elif args.mode == 'ppi':
        viewer.plot_ppi_3d(args.sweep)
    elif args.mode == 'animate':
        anim = viewer.animate_sweeps()
        # Keep reference to animation to prevent garbage collection
        viewer._animation = anim
    
    # Save if requested
    if args.save:
        viewer.save_plot(args.save)
    
    # Show plot
    viewer.show()

if __name__ == "__main__":
    # Example usage without command line arguments
    print("NEXRAD Level 2 3D Visualizer")
    print("============================")
    print("\nTo use this program:")
    print("1. Install required packages:")
    print("   pip install pyart matplotlib numpy")
    print("2. Download NEXRAD Level 2 data from NCEI or AWS")
    print("3. Run: python nexrad_3d_viewer.py <path_to_nexrad_file>")
    print("\nExample:")
    print("python nexrad_3d_viewer.py KTLX20240315_000000_V06")
    print("python nexrad_3d_viewer.py data.nexrad --mode volume --field reflectivity")
    print("python nexrad_3d_viewer.py data.nexrad --mode ppi --sweep 2")
    print("python nexrad_3d_viewer.py data.nexrad --mode animate")
    
    # If running directly, provide demo functionality
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print("\nFor interactive use, create a NEXRADViewer3D instance:")
        print("viewer = NEXRADViewer3D()")
        print("viewer.load_radar_file('path_to_file')")
        print("viewer.plot_3d_volume()")
        print("viewer.show()")
