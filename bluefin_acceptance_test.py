"""
Bluefin Acceptance Test Data Analysis

This script reads CSV files from the Bluefin acceptance test and creates
various plots to analyze the data versus time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import glob
import os

def read_csv_files(data_directory: str) -> Dict[str, pd.DataFrame]:
    """
    Read all CSV files from the specified directory.
    
    Args:
        data_directory: Path to directory containing CSV files
        
    Returns:
        Dictionary mapping filename to DataFrame
    """
    csv_files = glob.glob(f"{data_directory}/*.csv")
    dataframes = {}
    
    for file_path in csv_files:
        filename = Path(file_path).name
        try:
            df = pd.read_csv(file_path)
            dataframes[filename] = df
            print(f"Successfully loaded: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return dataframes


def correct_time_resets(df: pd.DataFrame, time_column: str = None) -> pd.DataFrame:
    """
    Correct for time variable resets by detecting discontinuities and adjusting time values.
    
    When the time value decreases (indicating a power cycle reset), this function
    continues the time progression by adding 60 seconds to the previous value
    and maintaining that offset for subsequent values.
    
    Args:
        df: DataFrame containing the time series data
        time_column: Name of the column containing time values. If None, will auto-detect
                    columns starting with 'ccsdsSecHeader2_sec'
        
    Returns:
        DataFrame with a new 'corrected_time' column
    """
    # Auto-detect time column if not provided
    if time_column is None:
        time_columns = [col for col in df.columns if col.startswith('ccsdsSecHeader2_sec')]
        if len(time_columns) == 0:
            print("Warning: No time column starting with 'ccsdsSecHeader2_sec' found in DataFrame")
            return df
        elif len(time_columns) > 1:
            print(f"Warning: Multiple time columns found: {time_columns}. Using the first one: {time_columns[0]}")
            time_column = time_columns[0]
        else:
            time_column = time_columns[0]
            print(f"Auto-detected time column: {time_column}")
    
    if time_column not in df.columns:
        print(f"Warning: Time column '{time_column}' not found in DataFrame")
        return df
    
    # Create a copy to avoid modifying the original
    df_corrected = df.copy()
    
    # Initialize corrected time with original values
    corrected_time = df_corrected[time_column].values.copy()
    time_offset = 0.0
    
    # Iterate through time values to detect and correct resets
    for i in range(1, len(corrected_time)):
        current_time = corrected_time[i]
        previous_time = corrected_time[i-1]
        
        # If current time is less than previous time, we have a reset
        if current_time < previous_time:
            # Calculate the offset needed to continue from previous time + 60 seconds
            time_offset += (previous_time + 60.0) - current_time
            print(f"Time reset detected at index {i}: {previous_time} -> {current_time}")
            print(f"  Applying offset: {time_offset:.2f} seconds")
            
            # Apply the offset to all subsequent time values immediately
            for j in range(i, len(corrected_time)):
                corrected_time[j] += time_offset
    
    # Add the corrected time column
    df_corrected['corrected_time'] = corrected_time
    
    return df_corrected


def create_temperature_plot(dataframes: Dict[str, pd.DataFrame]) -> None:
    """
    Create a simple plot of the temperature variable from the xband_hk_pkt dataframe.
    
    Args:
        dataframes: Dictionary of DataFrames
    """
    # Find the xband_hk_pkt dataframe
    xband_df = None
    for filename, df in dataframes.items():
        if 'xband_hk_pkt' in filename.lower():
            xband_df = df
            print(f"Found xband_hk_pkt dataframe: {filename}")
            break
    
    if xband_df is None:
        print("Error: Could not find xband_hk_pkt dataframe")
        return
    
    # Check if the temperature column exists
    temp_column = 'xband_hk_power_amp_temp_degc'
    if temp_column not in xband_df.columns:
        print(f"Error: Temperature column '{temp_column}' not found in xband_hk_pkt dataframe")
        print(f"Available columns: {list(xband_df.columns)}")
        return
    
    # Get the time column (corrected_time if available, otherwise the original time column)
    if 'corrected_time' in xband_df.columns:
        time_col = 'corrected_time'
    else:
        time_columns = [col for col in xband_df.columns if col.startswith('ccsdsSecHeader2_sec')]
        if not time_columns:
            print("Error: No time column found in xband_hk_pkt dataframe")
            return
        time_col = time_columns[0]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(xband_df[time_col], xband_df[temp_column], linewidth=3, alpha=0.8)
    plt.title('X-Band Power Amplifier Temperature vs Time', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Temperature (°C)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to orchestrate the data analysis and plotting.
    """
    # Configuration
    data_directory = os.path.join(os.getenv('suncet_data'), 'test_data/2025_177_10_06_25_bluefin_fm1_acceptance_part1/decoded/csv/')
    
    # Read data files
    print("Reading CSV files...")
    dataframes = read_csv_files(data_directory)
    
    if not dataframes:
        print("No CSV files found or loaded successfully.")
        return
    
    print(f"Loaded {len(dataframes)} CSV files.")
    
    # Apply time correction to all dataframes
    print("Applying time corrections for power cycle resets...")
    corrected_dataframes = {}
    for filename, df in dataframes.items():
        corrected_df = correct_time_resets(df)
        corrected_dataframes[filename] = corrected_df
        print(f"Applied time correction to: {filename}")
    
    # Create the temperature plot
    print("Creating temperature plot...")
    create_temperature_plot(corrected_dataframes)
    
    print("Plot generation complete.")


if __name__ == "__main__":
    main()
