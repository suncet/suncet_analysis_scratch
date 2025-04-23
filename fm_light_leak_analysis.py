import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import csv
from tqdm import tqdm
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider

# Configuration
do_histograms = False
do_dark_vs_light = True
do_dark_vs_light_post_process_analysis = False

# Direct Directories
before_image_dir = '/Users/masonjp2/Library/CloudStorage/Box-Box/SunCET/7000 Testing/7105 CSIE/CSIE Images/2024-10-30 APL Telescope Integration CSIE Images/3 In chamber with both filters installed (light leak test)/'
after_image_dir = '/Users/masonjp2/Library/CloudStorage/Box-Box/SunCET/7000 Testing/7105 CSIE/CSIE Images/2024-10-30 APL Telescope Integration CSIE Images/4 At LASP In chamber with both filters installed (post ship light leak test)/'
after_filter_replace_dir = '/Users/masonjp2/Dropbox/suncet_dropbox/7000 Testing/7031-2 Telescope Light Leak Test/2025-01-17_Light_Leak_Test/'
output_dir = '/Users/masonjp2/Dropbox/suncet_dropbox/9500 Science Results or something/light_leak_analysis/before_after_telescope_delivery/'

# Regular expression to extract exposure time from the filename
filename_pattern = re.compile(r'.*_(\d+\.\d+)s\.fits')

# Get all image file paths
before_image_filepaths = glob.glob(os.path.join(before_image_dir, '*.fits'))
after_image_filepaths = glob.glob(os.path.join(after_image_dir, '*.fits'))
if 'Stim_On' not in after_filter_replace_dir:
    after_filter_replace_image_filepaths = glob.glob(os.path.join(after_filter_replace_dir, '*', '*.fits'))

# Prepare output directory
os.makedirs(output_dir, exist_ok=True)

def create_csv_writer(label):
    specific_csv_filename = os.path.join(output_dir, f'image_statistics_{label}.csv')
    csv_file = open(specific_csv_filename, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    return csv_writer, csv_file

def load_image(image_filepath):
    with fits.open(image_filepath) as hdul:
        image = hdul[0].data
        image = np.where(image < 0, image + 65536, image)
        return image, hdul[0].header

def extract_header_values(header):
    csie_current = header.get('HIINST', 0)
    integration_time = header.get('PINTG', 1) / 1000  # Default to 1 if not found, convert from ms to seconds
    detector_temperature = header.get('HTDETB', 'N/A')
    return csie_current, integration_time, detector_temperature

def compute_dark_current(detector_temperature):
    dark_current_mean = 20 * 2**((detector_temperature - 20) / 5.5)
    dark_current_std = 12 * 2**((detector_temperature - 20) / 5.5)
    return dark_current_mean, dark_current_std

def normalize_image(image, integration_time, gain=1.8):
    return (image / gain) / integration_time

def save_figure(fig, image_filepath, label):
    figure_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_filepath))[0] + f'_{label}_analysis.png')
    plt.savefig(figure_filename)
    plt.close(fig)

def calculate_statistics(image):
    mean_val = np.mean(image)
    median_val = np.median(image)
    std_val = np.std(image)
    max_val = np.max(image)
    return mean_val, median_val, std_val, max_val

def make_histograms(image_filepaths, label, vmin=0, vmax=100000):
    csv_writer, csv_file = create_csv_writer(label)
    
    for image_filepath in tqdm(image_filepaths, desc=f"Processing {label} images"):
        image, header = load_image(image_filepath)
        csie_current, integration_time, detector_temperature = extract_header_values(header)
        
        if csie_current >= 0.464:
            continue
        
        result_image = normalize_image(image, integration_time)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(result_image, cmap='hot', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Dark Image at {detector_temperature} ºC')
        cbar = fig.colorbar(ax1.imshow(result_image, cmap='hot', vmin=vmin, vmax=vmax), ax=ax1)
        cbar.set_label('electrons/second')
        
        ax2.hist(result_image.ravel(), bins=256, histtype='step', color='dodgerblue', alpha=0.75)
        ax2.set_xlim(vmin, vmax)
        ax2.set_xlabel('electrons/second')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Histogram of Image')
        
        save_figure(fig, image_filepath, label)
        
        mean_val, median_val, std_val, max_val = calculate_statistics(result_image)
        csv_writer.writerow([os.path.basename(image_filepath), mean_val, median_val, std_val, max_val])
    
    csv_file.close()

def compare_dark_vs_light(image_filepaths, label, baseline_filename):
    baseline_image_filepath = next((filepath for filepath in image_filepaths if baseline_filename in filepath), None)
    
    if baseline_image_filepath is None:
        raise FileNotFoundError(f"Baseline image with filename containing '{baseline_filename}' not found.")
    
    baseline_image, baseline_header = load_image(baseline_image_filepath)
    baseline_integration_time = extract_header_values(baseline_header)[1]
    
    diff_csv_filename = os.path.join(output_dir, 'image_statistics_difference.csv')
    with open(diff_csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Filename', 'Mean', 'Median', 'Standard Deviation', 'Maximum Value', 'Detector Temperature'])
        
        for image_filepath in tqdm(image_filepaths, desc="Processing images"):
            if baseline_filename in image_filepath:
                continue
            
            image, header = load_image(image_filepath)
            integration_time, csie_current, detector_temperature = extract_header_values(header)
            
            # Skip images with integration_time not equal to 15 if label is 'after'
            if label == 'after' and integration_time != 15:
                continue
            
            difference_image = image - baseline_image
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            custom_cmap = create_custom_colormap()
            max_abs_val = max(abs(np.min(difference_image)), abs(np.max(difference_image)))
            
            im = ax1.imshow(difference_image, cmap=custom_cmap, vmin=-max_abs_val, vmax=max_abs_val)
            if label == 'after_filter_replace':
                subfolder = os.path.basename(os.path.dirname(image_filepath))
                filename_suffix = os.path.basename(image_filepath)[-8:-5]
                ax1.set_title(f'Difference Image\n{subfolder} {filename_suffix}')
            else:
                ax1.set_title(f'Difference Image\nCurrent Integration Time: {integration_time}, Baseline: {baseline_integration_time}')
            cbar = fig.colorbar(im, ax=ax1)
            cbar.set_label('DN')
            
            add_interactive_colorbar(fig, im, max_abs_val)
            
            ax2.hist(difference_image.ravel(), bins=256, histtype='step', color='dodgerblue', alpha=0.75)
            ax2.set_xlabel('DN')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Histogram of Difference Image')
            
            mean_val, median_val, std_val, max_val = calculate_statistics(difference_image)

            minmax_val = 6 * std_val
            im.set_clim([-minmax_val, minmax_val])
            ax2.set_xlim([-minmax_val, minmax_val])
            
            if (label == 'after' and max_val > 50000) or (label == 'before' and max_val > 1000):
               print(f"Warning: {image_filepath} has at least one pixel that is {max_val} DN brighter than baseline dark, which is suspiciously high.")
               plt.show()
            
            save_figure(fig, image_filepath, label)
            csv_writer.writerow([os.path.basename(image_filepath), mean_val, median_val, std_val, max_val, detector_temperature])

def create_custom_colormap():
    cdict = {
        'red':   [(0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.12, 0.12)],
        'green': [(0.0, 0.39, 0.39), (0.5, 0.0, 0.0), (1.0, 0.56, 0.56)],
        'blue':  [(0.0, 0.28, 0.28), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)]
    }
    return LinearSegmentedColormap('CustomMap', cdict)

def add_interactive_colorbar(fig, im, max_abs_val):
    axcolor = 'lightgoldenrodyellow'
    axmin = plt.axes([0.15, 0.01, 0.65, 0.03], facecolor=axcolor)
    axmax = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
    
    smin = Slider(axmin, 'Min', -max_abs_val, max_abs_val, valinit=-max_abs_val)
    smax = Slider(axmax, 'Max', -max_abs_val, max_abs_val, valinit=max_abs_val)
    
    def update(val):
        im.set_clim([smin.val, smax.val])
        fig.canvas.draw_idle()
    
    smin.on_changed(update)
    smax.on_changed(update)

def analyze_dark_vs_light_processed_data(diff_csv_filename):
    data = pd.read_csv(diff_csv_filename)
    
    max_of_max_values = data['Maximum Value'].max()
    print(f"The biggest difference in any lights-on image from the lights-off image is {max_of_max_values} DN")
    
    count_above_1000 = (data['Maximum Value'] > 1000).sum()
    total_images = len(data)
    print(f"The number of images with maximum values above 1000 DN is {count_above_1000}")
    print(f"The total number of images is {total_images}")
    
    std_min = data['Standard Deviation'].min()
    std_max = data['Standard Deviation'].max()
    std_median = data['Standard Deviation'].median()
    
    print(f"Standard Deviation - Min: {std_min}, Max: {std_max}, Median: {std_median}")

def load_image_and_compute_dark_current(image_filepath):
    with fits.open(image_filepath) as hdul:
        image = hdul[0].data
        detector_temperature = hdul[0].header.get('HTDETB', 'N/A')
        
        if detector_temperature == 'N/A':
            raise ValueError("Detector temperature not found in FITS header.")
        
        dark_current_mean, dark_current_std = compute_dark_current(detector_temperature)
    
    return dark_current_mean, dark_current_std, detector_temperature

def main():
    if do_histograms:
        for label in ['before', 'after']:
            csv_writer, csv_file = create_csv_writer(label)
            csv_writer.writerow(['Filename', 'Mean', 'Median', 'Standard Deviation', 'Maximum Value'])
            csv_file.close()
        
        make_histograms(before_image_filepaths, 'before')
        make_histograms(after_image_filepaths, 'after')
        
        before_data = pd.read_csv(os.path.join(output_dir, 'image_statistics_before.csv'))
        after_data = pd.read_csv(os.path.join(output_dir, 'image_statistics_after.csv'))
        
        comparison_results = {
            'Statistic': ['Mean', 'Median', 'Standard Deviation', 'Maximum Value'],
            'Before': [before_data['Mean'].mean(), before_data['Median'].mean(), before_data['Standard Deviation'].mean(), before_data['Maximum Value'].mean()],
            'After': [after_data['Mean'].mean(), after_data['Median'].mean(), after_data['Standard Deviation'].mean(), after_data['Maximum Value'].mean()],
            'Difference': [
                after_data['Mean'].mean() - before_data['Mean'].mean(),
                after_data['Median'].mean() - before_data['Median'].mean(),
                after_data['Standard Deviation'].mean() - before_data['Standard Deviation'].mean(),
                after_data['Maximum Value'].mean() - before_data['Maximum Value'].mean()
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_results)
        print("Comparison of Statistics Before and After:")
        print(comparison_df)
        
        comparison_csv_filename = os.path.join(output_dir, 'comparison_statistics.csv')
        comparison_df.to_csv(comparison_csv_filename, index=False)
    
    diff_csv_filename = os.path.join(output_dir, 'image_statistics_difference.csv')
    if do_dark_vs_light:
        with open(diff_csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Filename', 'Mean', 'Median', 'Standard Deviation', 'Maximum Value', 'LED Status'])
        
        #compare_dark_vs_light(before_image_filepaths, 'before', 'FrameData_20241030_150700273.fits')
        #compare_dark_vs_light(after_image_filepaths, 'after', 'FrameData_20241107_124513938.fits')
        compare_dark_vs_light(after_filter_replace_image_filepaths, 'after_filter_replace', 'Frame_250117_174140_485.fits')
    
    if do_dark_vs_light_post_process_analysis:
        analyze_dark_vs_light_processed_data(diff_csv_filename)
    
    sample_image_filepath = before_image_filepaths[0] if before_image_filepaths else None
    
    if sample_image_filepath:
        dark_current_mean, dark_current_std, detector_temperature = load_image_and_compute_dark_current(sample_image_filepath)
        print(f"Detector Temperature (ºC): {detector_temperature}")
        print(f"Dark Current Mean (e-/s): {dark_current_mean}, Dark Current Std (e-/s): {dark_current_std}")
        
        gain = 1.8
        integration_time_s = 0.05
        dark_current_mean_dn = dark_current_mean * gain * integration_time_s
        dark_current_std_dn = dark_current_std * gain * integration_time_s
        print(f"Dark Current Mean (DN): {dark_current_mean_dn}, Dark Current Std (DN): {dark_current_std_dn}")
    else:
        print("No image file paths provided for dark current computation.")

if __name__ == "__main__":
    main()