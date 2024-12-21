import warnings

warnings.filterwarnings("ignore")

import os
import time
import random
import pandas as pd
import pickle
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from itertools import product
import torch
from torch import nn
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.utils.losses import SmapeLoss
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae
from darts.utils.utils import SeasonalityMode, TrendMode, ModelMode
from darts.models import *


import pandas as pd
import numpy as np
import os
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MaxAbsScaler
from typing import Tuple, List
from tqdm import tqdm

HORIZON = 32  # Forecast horizon, e.g., last 48 hours as test set

def load_multiple_sites_data(data_dir: str) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    """
    Load and preprocess time series data for multiple sites (one file per site).
    
    Args:
        data_dir: Path to the directory containing CSV files for each site.
    
    Returns:
        Tuple containing lists of scaled train and test TimeSeries.
    """
    print("Loading Beijing Air Quality TimeSeries from multiple files...")

    site_series = []
    
    # Iterate through all files in the directory
    for file in tqdm(os.listdir(data_dir)):
        if file.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(data_dir, file)
            print(f"Processing {file_path}...")
            
            # Load data
            df = pd.read_csv(file_path)

            # Combine year, month, day, and hour into a datetime index
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            df.set_index('datetime', inplace=True)

            # Drop unnecessary columns
            df.drop(columns=['No', 'year', 'month', 'day', 'hour', 'wd', 'station'], errors='ignore', inplace=True)

            # Handle missing values (forward fill, backward fill)
            df = df.ffill().bfill()

            # Convert to Darts TimeSeries
            series = TimeSeries.from_dataframe(df).astype(np.float32)
            site_series.append(series)
    
    print("\nThere are {} series in the dataset.".format(len(site_series)))

    # Split train, test, and unseen
    site_train, site_test, site_unseen = [], [], []

    # for s in site_series:
    #     total_length = len(s)
    #     train_end = int(0.7 * total_length)
    #     test_end = int(0.2 * total_length)
    #     # val_end = int

    #     # Correctly apply the HORIZON
    #     unseen_start = total_length - HORIZON

    #     site_train.append(s[:train_end])       # 70% Training data
    #     site_test.append(s[train_end:test_end]) # 20% Test data
    #     site_unseen.append(s[unseen_start:])            # 10% Unseen data # By Horizon as well FIX

    # # Scale data
    # print("Scaling...")
    # scaler = Scaler(scaler=MaxAbsScaler())
    # site_train_scaled = scaler.fit_transform(site_train)
    # site_test_scaled = scaler.transform(site_test)
    # site_unseen_scaled = scaler.transform(site_unseen)
    # Split train/test
    
    print("Splitting train/test...")
    site_train = [s[:-HORIZON] for s in site_series]
    site_test = [s[-HORIZON:] for s in site_series]

    # Scale data so that the largest value is 1
    print("Scaling...")
    scaler = Scaler(scaler=MaxAbsScaler())
    site_train_scaled: List[TimeSeries] = scaler.fit_transform(site_train)
    site_test_scaled: List[TimeSeries] = scaler.transform(site_test)


    print(
        "Done. There are {} series, with average training length {:.1f}".format(
            len(site_train_scaled), np.mean([len(s) for s in site_train_scaled])
        )
    )
    return site_train_scaled, site_test_scaled # , site_unseen_scaled


def eval_forecasts(
    pred_series: List[TimeSeries], test_series: List[TimeSeries]
) -> List[float]:


    print("******* DEBUGGING ******")
    print(type(pred_series), type(test_series))
    print("******* DEBUGGING ENDS ******")
    
    print("computing MAE...")
    mae_ = mae(test_series, pred_series) # test_series)
    print("Median MAE", mae_)
    plt.figure()
    plt.hist(mae_, bins=12)
    plt.ylabel("Count")
    plt.xlabel("MAE")
    plt.title("Median MAE: %.3f" % np.median(mae_))
    plt.show()
    plt.close()
    return mae_

def eval_local_model(
    train_series: List[TimeSeries], test_series: List[TimeSeries], model_cls, **kwargs
) -> Tuple[List[float], float]:
    preds = []
    start_time = time.time()
    for series in tqdm(train_series):
        model = model_cls(**kwargs)
        model.fit(series)
        pred = model.predict(n=HORIZON)
        preds.append(pred)
    elapsed_time = time.time() - start_time
    mae = eval_forecasts(preds, test_series)
    return mae, elapsed_time, preds, test_series

air_quality_all_sites_train, air_quality_all_sites_test = load_multiple_sites_data("/kaggle/input/beijing-multisite-airquality-data-set")


def eval_global_model(
    train_series: List[TimeSeries], test_series: List[TimeSeries], model_cls, **kwargs
) -> Tuple[List[float], float]:

    start_time = time.time()

    model = model_cls(**kwargs)
    model.fit(train_series)
    preds = model.predict(n=HORIZON, series=train_series)

    elapsed_time = time.time() - start_time

    smapes = eval_forecasts(preds, test_series)
    return smapes, elapsed_time,  preds, test_series


naive1_mae, naive1_time, pred, test_series = eval_local_model(
    air_quality_all_sites_train, 
    air_quality_all_sites_test, 
    RNNModel, 
    model="LSTM",
    hidden_dim=64,
    dropout=0.2,
    batch_size=1024,
    n_epochs=1,
    optimizer_kwargs={"lr": 1e-3},
    model_name="Air_RNN",
    log_tensorboard=False,
    random_state=42,
    training_length=752,
    input_chunk_length=32,
    force_reset=True,
    save_checkpoints=False,
    pl_trainer_kwargs={
        "enable_progress_bar": True,
        # change this one to "gpu" if your notebook does run in a GPU environment:
        "accelerator": "gpu",
        "callbacks": [my_stopper]
    },)


lr_smapes, lr_time = eval_global_model(
    air_quality_all_sites_train,
    air_quality_all_sites_test, 
    RNNModel, model="LSTM",
    hidden_dim=64,
    dropout=0.2,
    batch_size=1024,
    n_epochs=10,
    optimizer_kwargs={"lr": 1e-3},
    model_name="Air_RNN",
    log_tensorboard=False,
    random_state=42,
    training_length=752,
    input_chunk_length=32,
    force_reset=True,
    save_checkpoints=False,
    pl_trainer_kwargs={
        "enable_progress_bar": True,
        # change this one to "gpu" if your notebook does run in a GPU environment:
        "accelerator": "gpu",
        "callbacks": [my_stopper]
    },
)


## Possible N-BEATS hyper-parameters

# Slicing hyper-params:
IN_LEN = 32
OUT_LEN = 8

# Architecture hyper-params:
NUM_STACKS = 20
NUM_BLOCKS = 1
NUM_LAYERS = 2
LAYER_WIDTH = 136
COEFFS_DIM = 11

# Training settings:
LR = 1e-3
BATCH_SIZE = 1024
MAX_SAMPLES_PER_TS = 10
NUM_EPOCHS = 10


# reproducibility
np.random.seed(42)
torch.manual_seed(42)

start_time = time.time()

nbeats_model_air = NBEATSModel(
    input_chunk_length=IN_LEN,
    output_chunk_length=OUT_LEN,
    num_stacks=NUM_STACKS,
    num_blocks=NUM_BLOCKS,
    num_layers=NUM_LAYERS,
    layer_widths=LAYER_WIDTH,
    expansion_coefficient_dim=COEFFS_DIM,
    loss_fn=SmapeLoss(),
    batch_size=BATCH_SIZE,
    optimizer_kwargs={"lr": LR},
    pl_trainer_kwargs={
        "enable_progress_bar": True,
        # change this one to "gpu" if your notebook does run in a GPU environment:
        "accelerator": "gpu",
    },
)

nbeats_model_air.fit(air_quality_all_sites_train, dataloader_kwargs={"num_workers":4}, epochs=NUM_EPOCHS)

# get predictions
nb_preds = nbeats_model_air.predict(series=air_quality_all_sites_train, n=HORIZON)
nbeats_elapsed_time = time.time() - start_time

nbeats_smapes = eval_forecasts(nb_preds, air_quality_all_sites_test)


# reproducibility
np.random.seed(42)
torch.manual_seed(42)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
# a period of 5 epochs (`patience`)
my_stopper = EarlyStopping(
    monitor="train_loss",
    patience=5,
    min_delta=0.05,
    mode='min',
)

start_time = time.time()

rnn_model_air = RNNModel(
    model="LSTM",
    hidden_dim=64,
    dropout=0.2,
    batch_size=1024,
    n_epochs=30,
    optimizer_kwargs={"lr": 1e-3},
    model_name="Air_RNN",
    log_tensorboard=False,
    random_state=42,
    training_length=40,
    input_chunk_length=32,
    force_reset=True,
    save_checkpoints=False,
    pl_trainer_kwargs={
        "enable_progress_bar": True,
        # change this one to "gpu" if your notebook does run in a GPU environment:
        "accelerator": "gpu",
        "callbacks": [my_stopper]
    },
)

# rnn_model_air.fit(air_quality_all_sites_train, dataloader_kwargs={"num_workers":4})

# # get predictions
# rnn_preds = rnn_model_air.predict(series=air_quality_all_sites_train, n=8)
# rnn_elapsed_time = time.time() - start_time

# rnn_smapes = eval_forecasts(rnn_preds, air_quality_all_sites_test)

# Transformer model
lr_smapes, lr_time, preds, test_990 = eval_global_model(
    air_quality_all_sites_train,
    air_quality_all_sites_test, 
    TransformerModel, 
    d_model=64,
    dropout=0.2,
    batch_size=1024,
    n_epochs=10,
    optimizer_kwargs={"lr": 1e-3},
    model_name="Air_Transformer",
    random_state=42,
    output_chunk_length=16,
    input_chunk_length=32,
    pl_trainer_kwargs={
        "enable_progress_bar": True,
        # change this one to "gpu" if your notebook does run in a GPU environment:
        "accelerator": "gpu",
        "callbacks": [my_stopper]
    },
)