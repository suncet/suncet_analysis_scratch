import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from scipy.io import loadmat

# Directory containing the .png and .mat files
data_dir = '/Users/masonjp2/Library/CloudStorage/Box-Box/SunCET/7000 Testing/7105 CSIE/CSIE Images/2024-05-20 Darks over temp/'

# Initialize lists to store the statistics and temperatures
image_stats = []
temperatures = []

# Loop over all .png files in the directory
for image_path in glob.glob(os.path.join(data_dir, '*.png')):
    # Extract the base filename (without extension) to match with .mat file
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    mat_filename = f"{base_filename}.mat"
    mat_path = os.path.join(data_dir, mat_filename)
    
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to a numpy array
        img_array = np.array(img)
        
        # Calculate basic statistics
        mean_val = np.mean(img_array)
        median_val = np.median(img_array)
        std_val = np.std(img_array)
        max_val = np.max(img_array)
        
        # Append the statistics to the list
        image_stats.append([mean_val, median_val, std_val, max_val])
    
    # Read the corresponding .mat file for temperature data
    if os.path.exists(mat_path):
        mat_data = loadmat(mat_path)
        temperature = mat_data['HKD_THRM_DET0'][0][0]  # Adjust indexing if necessary
        temperatures.append(temperature)
    else:
        print(f"Warning: No corresponding .mat file found for {image_path}")

# Convert lists to numpy arrays for easier manipulation
image_stats = np.array(image_stats)
temperatures = np.array(temperatures)

# Print the array of statistics and temperatures
print("Image Statistics:\n", image_stats)
print("Temperatures:\n", temperatures)

# Optional: Save the statistics and temperatures to a file for later use
np.savetxt('image_statistics_with_temperatures.csv', np.column_stack((image_stats, temperatures)), 
           delimiter=',', header='Mean,Median,Std,Max,Temperature', comments='')

# Plotting the statistics against temperature
plt.figure(figsize=(10, 8))

# Plot mean values against temperature
plt.subplot(2, 2, 1)
plt.plot(temperatures, image_stats[:, 0], 'o-')
plt.title('Mean Values vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Mean')

# Plot median values against temperature
plt.subplot(2, 2, 2)
plt.plot(temperatures, image_stats[:, 1], 'o-')
plt.title('Median Values vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Median')

# Plot standard deviation values against temperature
plt.subplot(2, 2, 3)
plt.plot(temperatures, image_stats[:, 2], 'o-')
plt.title('Standard Deviation vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Std Dev')

# Plot maximum values against temperature
plt.subplot(2, 2, 4)
plt.plot(temperatures, image_stats[:, 3], 'o-')
plt.title('Maximum Values vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Max Value')

plt.tight_layout()
plt.show()



pass