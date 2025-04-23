import os
import glob
import re
from astropy.io import fits
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Root path for the light leak test data
root_path = '/Users/masonjp2/Dropbox/suncet_dropbox/7000 Testing/7031-2 Telescope Light Leak Test/2025-01-17_Light_Leak_Test/'

# Get all subdirectories
subdirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

# Dictionary to store FITS files for each folder
fits_files_by_folder = {}

# Regular expression to extract date and time from filename
# Format: Frame_250117_181703_907.fits
filename_pattern = re.compile(r'Frame_(\d{6})_(\d{6})_\d+\.fits')

# Track the earliest timestamp across all folders
global_first_timestamp = None

# Process each subdirectory
for subdir in subdirs:
    subdir_path = os.path.join(root_path, subdir)
    fits_files = glob.glob(os.path.join(subdir_path, '*.fits'))
    
    # Store the list of FITS files
    fits_files_by_folder[subdir] = []
    
    for fits_file in fits_files:
        filename = os.path.basename(fits_file)
        match = filename_pattern.match(filename)
        
        if match:
            date_str = match.group(1)  # yymmdd
            time_str = match.group(2)  # hhmmss
            
            # Convert to datetime object
            timestamp = datetime.strptime(f"20{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            
            # Update global first timestamp if needed
            if global_first_timestamp is None or timestamp < global_first_timestamp:
                global_first_timestamp = timestamp
            
            # Load FITS file
            with fits.open(fits_file) as hdul:
                image_data = hdul[0].data
                header = hdul[0].header
            
            # Compute sum and mean of the image
            image_sum = np.sum(image_data)
            image_mean = np.mean(image_data)
            
            # Store file info in the dictionary
            fits_files_by_folder[subdir].append({
                'filepath': fits_file,
                'filename': filename,
                'timestamp': timestamp,
                'data': image_data,
                'header': header,
                'sum': image_sum,
                'mean': image_mean
            })
    
    # Sort files by timestamp
    fits_files_by_folder[subdir].sort(key=lambda x: x['timestamp'])
    
    print(f"Loaded {len(fits_files_by_folder[subdir])} FITS files from {subdir}")

# Calculate seconds from first image for all files
for folder_name, files_list in fits_files_by_folder.items():
    for file_info in files_list:
        time_diff = (file_info['timestamp'] - global_first_timestamp).total_seconds()
        file_info['seconds_from_first'] = time_diff

# Example of accessing the data
for folder_name, files_list in fits_files_by_folder.items():
    if files_list:
        first_file = files_list[0]
        print(f"\nFolder: {folder_name}")
        print(f"  First file: {first_file['filename']}")
        print(f"  Timestamp: {first_file['timestamp']}")
        print(f"  Image shape: {first_file['data'].shape}")
        print(f"  Image sum: {first_file['sum']}")
        print(f"  Image mean: {first_file['mean']}")
        print(f"  Seconds from first image: {first_file['seconds_from_first']}")

# Create a plot of image sum versus time
plt.figure(figsize=(12, 8))

# Sort folders by their first timestamp to ensure chronological order in the legend
folder_order = []
for folder_name, files_list in fits_files_by_folder.items():
    if files_list:
        first_timestamp = files_list[0]['timestamp']
        folder_order.append((folder_name, first_timestamp))

# Sort folders by their first timestamp
folder_order.sort(key=lambda x: x[1])

# Collect data for linear fit (excluding 'Initial_Alt_Stim_On')
fit_seconds = []
fit_sums = []

# Use a different color for each folder
colors = plt.cm.tab10.colors
for i, (folder_name, first_timestamp) in enumerate(folder_order):
    files_list = fits_files_by_folder[folder_name]
    if files_list:
        # Extract timestamps and sums
        timestamps = [file_info['timestamp'] for file_info in files_list]
        sums = [file_info['sum'] for file_info in files_list]
        seconds = [file_info['seconds_from_first'] for file_info in files_list]
        
        # Plot with a different color for each folder
        color_idx = i % len(colors)
        plt.plot(timestamps, sums, 'o-', label=folder_name, color=colors[color_idx], alpha=0.7)
        
        # Collect data for fit if not from 'Initial_Alt_Stim_On'
        if folder_name != 'Initial_Alt_Stim_On':
            fit_seconds.extend(seconds)
            fit_sums.extend(sums)

# Perform linear fit if we have data
if fit_seconds and fit_sums:
    # Convert to numpy arrays for fitting
    fit_seconds = np.array(fit_seconds)
    fit_sums = np.array(fit_sums)
    
    # Fit a line to the data
    slope, intercept = np.polyfit(fit_seconds, fit_sums, 1)
    
    # Create x values for the fit line (spanning the entire time range)
    all_seconds = []
    for files_list in fits_files_by_folder.values():
        all_seconds.extend([file_info['seconds_from_first'] for file_info in files_list])
    
    min_seconds = min(all_seconds)
    max_seconds = max(all_seconds)
    fit_x_seconds = np.array([min_seconds, max_seconds])
    fit_y = slope * fit_x_seconds + intercept
    
    # Convert seconds to datetime for plotting
    fit_x_datetime = [global_first_timestamp + timedelta(seconds=s) for s in fit_x_seconds]
    
    # Plot the fit line
    plt.plot(fit_x_datetime, fit_y, 'k--', linewidth=1, label=f'Fit: {slope:.2e}x + {intercept:.2e}')

# Format the plot
plt.xlabel('Time')
plt.ylabel('Image Sum')
plt.title('Image Sum vs. Time for Light Leak Test')
plt.grid(True, alpha=0.3)
plt.legend()

# Format the x-axis to show time nicely
ax1 = plt.gca()
ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()

# Add a second x-axis showing seconds from first image
ax2 = plt.gca().twiny()

# Collect all timestamps and their corresponding seconds from first
all_timestamps = []
all_seconds = []
for files_list in fits_files_by_folder.values():
    for file_info in files_list:
        all_timestamps.append(file_info['timestamp'])
        all_seconds.append(file_info['seconds_from_first'])

if all_timestamps:
    # Create a function to convert from datetime to seconds from first
    def datetime_to_seconds(x):
        # Convert datetime to seconds from first image
        # Handle timezone-aware datetime by converting to naive datetime
        if hasattr(x, 'tzinfo') and x.tzinfo is not None:
            # Convert to naive datetime in the same representation
            naive_x = x.replace(tzinfo=None)
        else:
            naive_x = x
        return (naive_x - global_first_timestamp).total_seconds()
    
    # Create a function to convert from seconds to datetime
    def seconds_to_datetime(x):
        # Convert seconds from first image to datetime
        return global_first_timestamp + timedelta(seconds=x)
    
    # Set the secondary x-axis limits based on the primary axis
    ax2.set_xlim(ax1.get_xlim())
    
    # Set the formatter for the secondary axis to show seconds
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{datetime_to_seconds(mdates.num2date(x)):.0f}"))
    ax2.set_xlabel('Seconds from First Image')

# Save the plot
plt.tight_layout()
plt.savefig('light_leak_sum_vs_time.png', dpi=300)
plt.show()

# Create a plot of image mean versus time
plt.figure(figsize=(12, 8))

# Collect data for linear fit (excluding 'Initial_Alt_Stim_On')
fit_seconds = []
fit_means = []

# Use the same folder order for consistency
for i, (folder_name, first_timestamp) in enumerate(folder_order):
    files_list = fits_files_by_folder[folder_name]
    if files_list:
        # Extract timestamps and means
        timestamps = [file_info['timestamp'] for file_info in files_list]
        means = [file_info['mean'] for file_info in files_list]
        seconds = [file_info['seconds_from_first'] for file_info in files_list]
        
        # Plot with a different color for each folder
        color_idx = i % len(colors)
        plt.plot(timestamps, means, 'o-', label=folder_name, color=colors[color_idx], alpha=0.7)
        
        # Collect data for fit if not from 'Initial_Alt_Stim_On'
        if folder_name != 'Initial_Alt_Stim_On':
            fit_seconds.extend(seconds)
            fit_means.extend(means)

# Perform linear fit if we have data
if fit_seconds and fit_means:
    # Convert to numpy arrays for fitting
    fit_seconds = np.array(fit_seconds)
    fit_means = np.array(fit_means)
    
    # Fit a line to the data
    slope, intercept = np.polyfit(fit_seconds, fit_means, 1)
    
    # Create x values for the fit line (spanning the entire time range)
    all_seconds = []
    for files_list in fits_files_by_folder.values():
        all_seconds.extend([file_info['seconds_from_first'] for file_info in files_list])
    
    min_seconds = min(all_seconds)
    max_seconds = max(all_seconds)
    fit_x_seconds = np.array([min_seconds, max_seconds])
    fit_y = slope * fit_x_seconds + intercept
    
    # Convert seconds to datetime for plotting
    fit_x_datetime = [global_first_timestamp + timedelta(seconds=s) for s in fit_x_seconds]
    
    # Plot the fit line
    plt.plot(fit_x_datetime, fit_y, 'k--', linewidth=1, label=f'Fit: {slope:.2e}x + {intercept:.2e}')

# Format the plot
plt.xlabel('Time')
plt.ylabel('Image Mean')
plt.title('Image Mean vs. Time for Light Leak Test')
plt.grid(True, alpha=0.3)
plt.legend()

# Format the x-axis to show time nicely
ax1 = plt.gca()
ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()

# Add a second x-axis showing seconds from first image
ax2 = plt.gca().twiny()

# Set the secondary x-axis limits based on the primary axis
ax2.set_xlim(ax1.get_xlim())

# Set the formatter for the secondary axis to show seconds
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{datetime_to_seconds(mdates.num2date(x)):.0f}"))
ax2.set_xlabel('Seconds from First Image')

# Save the plot
plt.tight_layout()
plt.savefig('light_leak_mean_vs_time.png', dpi=300)
plt.show()

pass