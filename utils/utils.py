# add imports
import numpy as np

def create_time_series(data, target_col, n_past, n_future):
    """
    Creates input (X) and output (y) time-series data using n_past time steps to predict n_future steps ahead.
    :param data: Preprocessed DataFrame.
    :param target_col: Target column name (e.g., 'PM2.5').
    :param n_past: Number of past time steps used for the input.
    :param n_future: Number of future steps to predict.
    :return: X (inputs), y (outputs)
    """
    # Ensure only numeric columns are used for X
    X_data = data.drop(columns=['date'], errors='ignore')  # Drop the date column if it exists
    y_data = data[target_col]  # Target column

    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        # Use the past n_past time steps for input
        X.append(X_data.iloc[i - n_past:i].values)  
        # Predict the value n_future steps ahead
        y.append(y_data.iloc[i + n_future - 1])  

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)