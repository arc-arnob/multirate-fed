import pandas as pd
import numpy as np
import random

def generate_simulated_time_series(data, date_column, columns, n_tasks=4, resample_frequency='3D'):
    """
    Generate simulated time series data with resampling and random translations.

    Args:
        data_path (str): Path to the time series data CSV file.
        date_column (str): Name of the column containing datetime information.
        columns (list): List of columns to apply transformations (e.g., forecasting targets).
        n_tasks (int): Number of tasks (splits) to create in the dataset.
        resample_frequency (str): Frequency for resampling (e.g., '3D' for 3 days).

    Returns:
        list: List of DataFrames, each corresponding to a task with unique transformations.
    """
    # Read the dataset
    
    # Ensure the date column is in datetime format
    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)
    
    # Resample target columns
    for column in columns:
        resampled_column = data[column].resample(resample_frequency).mean()
        data[f"{column}_{resample_frequency}"] = resampled_column.reindex(data.index)
    
    data.reset_index(inplace=True)
    
    # Split the dataset into `n_tasks` equal parts
    task_size = len(data) // n_tasks
    task_data = []
    
    # Dynamically generate translation ranges for each task
    # Seed for reproducibility
    random.seed(42)

    # Define a base range template
    base_ranges = [(-30, -20), (-40, -10), (10, 20), (5, 15)]

    # Shuffle the order of base ranges
    random.shuffle(base_ranges)

    # Expand or repeat ranges to match the number of tasks, with random factors
    translation_ranges = [random.choice(base_ranges) for _ in range(n_tasks)]
    
    for i in range(n_tasks):
        # Determine start and end indices for the split
        start_idx = i * task_size
        end_idx = len(data) if i == n_tasks - 1 else (i + 1) * task_size
        
        # Extract the slice
        task_slice = data.iloc[start_idx:end_idx].copy()
        
        # Assign a unique cluster_id to each slice
        task_slice['cluster_id'] = i + 1
        
        # Apply random translation within the unique range
        low, high = translation_ranges[i]
        random_translation = np.random.uniform(low, high)
        for column in columns:
            task_slice[column] += random_translation
        
        # Collect the processed task slice
        task_data.append(task_slice)
    
    return task_data
