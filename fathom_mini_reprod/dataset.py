import os
import json
from abc import ABC, abstractmethod

import torch
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class ExperimentDataset(Dataset, ABC):
    def __init__(self,
                 folder,
                 x_columns,
                 y_columns,
                 clients,
                 min_date=None,
                 max_date=None,
                 filters=[],
                 batch_size=1,
                 sliding_window=1,
                 pred_window=1,
                 percentage=0.8,
                 standardize=True,
                 dtype=torch.float64,
                 device=torch.device("cpu"),
                 dynamic_slide=False,
                 concatenate=False):
        self.folder = folder
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.clients = clients
        self.min_date = min_date
        self.max_date = max_date
        self.filters = filters
        self.batch_size = batch_size
        self.sliding_window = sliding_window
        self.pred_window = pred_window
        self.window = sliding_window + pred_window
        self.percentage = percentage
        self.standardize = standardize
        self.dtype = dtype
        self.device = device
        self.dynamic_slide = dynamic_slide
        self.concatenate = concatenate

        self.dataframes = [[]]
        self.train_length = -1
        self.test_length = -1
        self.train = True
        self.cluster = np.arange(1, self.clients + 1)

    @abstractmethod
    def preprocess_dataframe(self, df):
        pass

    def __len__(self):
        # calculate the length of the dataset, every time we switch between train and test
        if self.train_length == -1 or self.test_length == -1:
            # length = the amount of batched samples we can fit in the dataset
            correction = self.window if self.dynamic_slide else 0
            denominator = self.batch_size if self.dynamic_slide else self.batch_size * self.window

            train_l = []
            test_l = []

            for client in self.cluster:
                df = self.dataframes[client - 1]
                length_df = int((len(df) - correction) / denominator) - 1
                l_perc = int(length_df * self.percentage)

                train_l.append(l_perc)
                test_l.append(length_df - l_perc)

            # concatenate the samples for centralised training
            self.train_length = sum(train_l) if self.concatenate else min(train_l)
            self.test_length = min(test_l)

        return self.train_length if self.train else self.test_length

    def get_df_mask(self, df):
        mask = pd.Series(True, index=df.index)

        for f in self.filters:
            column_name, operator, value = f
            mask = mask & df[column_name].apply(lambda x: eval(f"x {operator} {value}"))

        return mask

    def standardize_df(self, df):
        return (df - df.mean()) / (df.std() + 1e-8)

    def to_dataframe(self, path):
        df = pd.read_csv(path)
        df = self.preprocess_dataframe(df)

        return self.filter_and_normalize(df)

    def filter_and_normalize(self, df):
        if not isinstance(df, list):
            df = [df]

        return_df = []

        for df_ in df:
            filter_mask = self.get_df_mask(df_)
            filtered_df = df_[filter_mask]

            if self.standardize:
                filtered_df = self.standardize_df(filtered_df)

            num_rows = len(filtered_df)

            return_df.append(
                filtered_df.head(round(num_rows) - (round(num_rows) % (self.window * self.batch_size))).copy())

        return return_df if len(return_df) > 1 else return_df[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx < 0:
            idx += len(self)

        if idx > len(self) or idx < 0:
            raise IndexError("Index out of range")

        # if concatenate we return the sample from the correct client
        if self.concatenate and self.train:
            length_of_one_df = self.train_length / len(self.cluster)
            df_number = math.floor(idx / length_of_one_df)
            index = int(idx % length_of_one_df)

            batches_x, batches_y = self.get_item_helper(index, df_number)
        else:
            if self.train:
                # return batch for x and y with correct input windows and prediction windows, resp.
                batches_x, batches_y = self.get_item_helper(idx)
            else:
                if self.concatenate:
                    batches_x, batches_y = self.get_item_helper(idx + int(self.train_length / len(self.cluster)))
                else:
                    # if testing, we return the samples that start after training samples
                    batches_x, batches_y = self.get_item_helper(idx + self.train_length)

        items_x_tensor = torch.tensor(np.array(batches_x), dtype=self.dtype).to(self.device)
        items_y_tensor = torch.tensor(np.array(batches_y), dtype=self.dtype).to(self.device)

        return items_x_tensor, items_y_tensor

    def get_item_helper(self, index, df_id=None):
        start_index = index * self.batch_size
        batches_x = []
        batches_y = []

        for bn in range(self.batch_size):
            items_x = []
            items_y = []

            x_left = start_index + bn
            # If we have fixed windows
            # dynamic slide = increase window starting point by 1
            if not self.dynamic_slide:
                x_left *= self.window

            # In case of pred_window we want to predict future value and should
            # take this into account when dividing the frames
            if self.pred_window:
                x_right = x_left + self.sliding_window
                y_left = x_right
                y_right = x_right + self.pred_window
            else:
                x_right = x_left + self.sliding_window
                y_left = x_left
                y_right = x_right

            frames = [self.dataframes[client - 1] for client in self.cluster]

            if df_id is not None:
                frames = [frames[df_id]]

            for frame in frames:
                items_x.append(frame[self.x_columns].iloc[x_left:x_right].values)
                items_y.append(frame[self.y_columns].iloc[y_left:y_right].values)

            batches_x.append(np.array(items_x))
            batches_y.append(np.array(items_y))

        return batches_x, batches_y

    def train_data(self, cluster=None):
        # we return the training data (first part of the dataset)
        # cluster indicates which clients we want to include during the retrieval of data
        self.train = True
        if cluster is not None:
            self.train_length = -1
            self.cluster = cluster

        return self

    def test_data(self, cluster=None):
        # we return the test data (last part of the dataset)
        self.train = False
        if cluster is not None:
            self.test_length = -1
            self.cluster = cluster

        return self

    def get_correlation(self):
        df = pd.DataFrame([])

        for j in range(len(self.dataframes)):
            df[f'client {j + 1}'] = self.dataframes[j][self.y_columns]

        return df.corr().abs().values.tolist()


class AirQualityDataset(ExperimentDataset):
    def __init__(self,
                 folder,
                 files,
                 x_columns,
                 y_columns,
                 clients,
                 min_date=None,
                 max_date=None,
                 filters=[],
                 batch_size=1,
                 sliding_window=1,
                 pred_window=1,
                 percentage=1.0,
                 standardize=True,
                 dtype=torch.float64,
                 device=torch.device("cpu"),
                 dynamic_slide=False,
                 concatenate=False,
                 offset=0):

        super().__init__(folder,
                         x_columns,
                         y_columns,
                         clients,
                         min_date,
                         max_date,
                         filters,
                         batch_size,
                         sliding_window,
                         pred_window,
                         percentage,
                         standardize,
                         dtype,
                         device,
                         dynamic_slide,
                         concatenate)

        self.files = files[offset:clients + offset]

        if clients > len(self.files):
            self.clients = len(self.files)
            raise RuntimeWarning(f'Number of clients is too large and set to {self.clients}')

        self.dataframes = [self.to_dataframe(os.path.join(self.folder, file)) for file in self.files]

    def preprocess_dataframe(self, df):
        df['Time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

        if self.min_date:
            df = df[df['Time'] > self.min_date]

        if self.max_date:
            df = df[df['Time'] <= self.max_date]

        label_encoder = LabelEncoder()
        df['wd'] = label_encoder.fit_transform(df['wd'])

        non_dup_columns = self.x_columns + list(set(self.y_columns) - set(self.x_columns))

        return df[non_dup_columns].interpolate().fillna(0)


class SolarDataset(ExperimentDataset):
    def __init__(self,
                 folder,
                 files,
                 x_columns,
                 y_columns,
                 clients,
                 min_date=None,
                 max_date=None,
                 filters=[],
                 batch_size=1,
                 sliding_window=1,
                 pred_window=1,
                 percentage=1.0,
                 standardize=True,
                 dtype=torch.float64,
                 device=torch.device("cpu"),
                 dynamic_slide=False,
                 concatenate=False,
                 offset=0):

        super().__init__(folder,
                         x_columns,
                         y_columns,
                         clients,
                         min_date,
                         max_date,
                         filters,
                         batch_size,
                         sliding_window,
                         pred_window,
                         percentage,
                         standardize,
                         dtype,
                         device,
                         dynamic_slide,
                         concatenate)

        self.files = files[:clients]

        if clients > len(self.files):
            self.clients = len(self.files)
            raise RuntimeWarning(f'Number of clients is too large and set to {self.clients}')

        self.dataframes = self.collect_dataframes()

    def collect_dataframes(self):
        non_dup_columns = self.x_columns + list(set(self.y_columns) - set(self.x_columns))
        dfs = []

        if not isinstance(self.files[0], list):
            dfs = [self.to_dataframe(os.path.join(self.folder, file)) for file in self.files]
        else:
            for files_ in self.files:
                dfs_ = [self.to_dataframe(os.path.join(self.folder, file)) for file in files_]
                dfs_merged = pd.concat(dfs_, axis=1)
                dfs_merged.columns = [f'panel{i + 1}' for i in range(len(dfs_))]

                dfs.append(dfs_merged)

        return [df[non_dup_columns] for df in dfs]

    def preprocess_dataframe(self, df):
        df['LocalTime'] = pd.to_datetime(df['LocalTime'], format='%m/%d/%y %H:%M')

        if self.min_date:
            df = df[df['LocalTime'] > self.min_date]

        if self.max_date:
            df = df[df['LocalTime'] <= self.max_date]

        return df[['Power(MW)']].interpolate().fillna(0)


class SalesDataset(ExperimentDataset):
    def __init__(self,
                 folder,
                 file,
                 x_columns,
                 y_columns,
                 clients,
                 min_date=None,
                 max_date=None,
                 filters=[],
                 batch_size=1,
                 sliding_window=1,
                 pred_window=1,
                 percentage=1.0,
                 standardize=True,
                 dtype=torch.float64,
                 device=torch.device("cpu"),
                 dynamic_slide=False,
                 concatenate=False):

        super().__init__(folder,
                         x_columns,
                         y_columns,
                         clients,
                         min_date,
                         max_date,
                         filters,
                         batch_size,
                         sliding_window,
                         pred_window,
                         percentage,
                         standardize,
                         dtype,
                         device,
                         dynamic_slide,
                         concatenate)

        self.file = file
        self.dataframes = self.collect_dataframes()

    def collect_dataframes(self):
        df = pd.read_csv(os.path.join(self.folder, self.file))
        df = self.preprocess_dataframe(df)

        non_dup_columns = self.x_columns + list(set(self.y_columns) - set(self.x_columns))

        dfs_per_client = []
        for i in range(self.clients):
            df_ = df[df['Store'] == i + 1][non_dup_columns]
            dfs_per_client.append(self.filter_and_normalize(df_))

        return dfs_per_client

    def preprocess_dataframe(self, df):
        df['Date'] = pd.to_datetime(df['Date'])  # , format='%y-%m-%d')

        if self.min_date:
            df = df[df['Date'] > self.min_date]

        if self.max_date:
            df = df[df['Date'] <= self.max_date]

        df['Store'] = df['Store'].astype(int)

        return df.interpolate().fillna(0)


class CryptoDataset(ExperimentDataset):
    def __init__(self,
                 folder,
                 files,
                 x_columns,
                 y_columns,
                 clients,
                 min_date=None,
                 max_date=None,
                 filters=[],
                 batch_size=1,
                 sliding_window=1,
                 pred_window=1,
                 percentage=1.0,
                 standardize=True,
                 dtype=torch.float64,
                 device=torch.device("cpu"),
                 dynamic_slide=False,
                 concatenate=False,
                 offset=0):

        super().__init__(folder,
                         x_columns,
                         y_columns,
                         clients,
                         min_date,
                         max_date,
                         filters,
                         batch_size,
                         sliding_window,
                         pred_window,
                         percentage,
                         standardize,
                         dtype,
                         device,
                         dynamic_slide,
                         concatenate)

        self.files = files[:clients]

        if clients > len(self.files):
            self.clients = len(self.files)
            raise RuntimeWarning(f'Number of clients is too large and set to {self.clients}')

        self.dataframes = [self.to_dataframe(os.path.join(self.folder, file)) for file in self.files]

    def preprocess_dataframe(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        if self.min_date:
            df = df[df['timestamp'] > self.min_date]

        if self.max_date:
            df = df[df['timestamp'] <= self.max_date]

        all_dates = pd.date_range(start=self.min_date, end=self.max_date, freq='h')

        agg_dict = {
            'Asset_ID': 'first',  # Assuming the ID doesn't change
            'Count': 'sum',  # Sum of counts
            'Open': 'first',  # Opening price
            'High': 'max',  # Highest price
            'Low': 'min',  # Lowest price
            'Close': 'last',  # Closing price
            'Volume': 'sum',  # Total volume
            'VWAP': 'mean',  # Average VWAP
            'Target': 'mean'  # Average target
        }

        # resample
        df = df.set_index('timestamp').resample('h').agg(agg_dict).reindex(
            all_dates).interpolate().reset_index().fillna(0)

        non_dup_columns = self.x_columns + list(set(self.y_columns) - set(self.x_columns))

        return df[non_dup_columns].interpolate().fillna(0)


class IndustryDataset(ExperimentDataset):
    def __init__(self,
                 folder,
                 x_columns,
                 y_columns,
                 clients,
                 min_date=None,
                 max_date=None,
                 filters=[],
                 batch_size=1,
                 sliding_window=1,
                 pred_window=1,
                 percentage=1.0,
                 standardize=True,
                 dtype=torch.float64,
                 device=torch.device("cpu"),
                 dynamic_slide=False,
                 concatenate=False,
                 offset=0,
                 customer_only=False):

        super().__init__(folder,
                         x_columns,
                         y_columns,
                         clients,
                         min_date,
                         max_date,
                         filters,
                         batch_size,
                         sliding_window,
                         pred_window,
                         percentage,
                         standardize,
                         dtype,
                         device,
                         dynamic_slide,
                         concatenate)

        self.x_columns = x_columns
        self.y_columns = y_columns
        self.pred_window = pred_window

        # load factory and customer files separately
        self.factory_files = [os.path.join(folder, file) for file in
                              json.loads(os.getenv("FACTORY_FILES"))]  # [:clients]
        self.customer_files = [os.path.join(folder, file) for file in
                               json.loads(os.getenv("CUSTOMER_FILES"))]  # [:math.ceil(clients/2)]

        if clients > len(self.customer_files) * 2:
            self.clients = len(self.customer_files) * 2
            raise RuntimeWarning(f'Number of clients is too large and set to {self.clients}')

        self.dataframes = self.collect_dataframes(offset, customer_only)

    def collect_dataframes(self, offset, customer_only):
        customer_dataframes = []
        for file in self.customer_files:
            customer_dataframes += self.to_dataframe(file)

        if customer_only:
            return customer_dataframes

        factory_dataframes = [self.to_dataframe(file) for file in self.factory_files]

        # merge factory and customer dataframes
        dataframes = []
        if len(factory_dataframes[0]) > 0:
            for df_i in range(offset, offset + self.clients):
                dataframes.append(
                    pd.merge(factory_dataframes[df_i].reset_index(), customer_dataframes[df_i].reset_index(),
                             how="left", on=["index", "index"]))
        elif len(customer_dataframes[0]) > 0:
            for df_i in range(offset, offset + self.clients):
                dataframes.append(
                    pd.merge(customer_dataframes[df_i].reset_index(), factory_dataframes[df_i].reset_index(),
                             how="left", on=["index", "index"]))

        return dataframes

    def preprocess_dataframe(self, df):
        # identify factory and customer dataframes by specific column
        is_factory = (os.getenv("FACTORY_COLUMN") in df.columns)
        all_dates = pd.date_range(start=self.min_date, end=self.max_date, freq='D')
        non_dup_columns = self.x_columns + list(set(self.y_columns) - set(self.x_columns))
        columns_in_df = list(set(df.columns) & set(non_dup_columns))

        if len(columns_in_df) == 0:
            return pd.DataFrame()

        # factory dfs require different pre-processing than customer
        if is_factory:
            df['Time'] = pd.to_datetime(df['Time'])

            if self.min_date:
                df = df[df['Time'] > self.min_date]

            if self.max_date:
                df = df[df['Time'] <= self.max_date]

            df_ = df[['Time'] + columns_in_df].set_index('Time').resample('D').mean().reindex(
                all_dates).interpolate().reset_index().fillna(0)

            return df_[columns_in_df]
        else:
            df['Time'] = pd.to_datetime(df['Time'])
            df.sort_values(by='Time', inplace=True)

            df = df[df['Control'] == 0]

            if self.min_date:
                df = df[df['Time'] > self.min_date]

            if self.max_date:
                df = df[df['Time'] <= self.max_date]

            df1 = df[df['ChuckId'] == 'CHUCK_ID_1']
            df2 = df[df['ChuckId'] == 'CHUCK_ID_2']

            df1_ = df1[['Time'] + columns_in_df].set_index('Time').resample('D').mean().reindex(
                all_dates).interpolate().reset_index().fillna(0)
            df2_ = df2[['Time'] + columns_in_df].set_index('Time').resample('D').mean().reindex(
                all_dates).interpolate().reset_index().fillna(0)

            return [df1_[columns_in_df], df2_[columns_in_df]]

class ElectricityDataset(ExperimentDataset):
    def __init__(self,
                 folder,
                 files,
                 clients,
                 min_date=None,
                 max_date=None,
                 filters=[],
                 batch_size=1,
                 sliding_window=1,
                 pred_window=1,
                 percentage=1.0,
                 standardize=True,
                 dtype=torch.float64,
                 device=torch.device("cpu"),
                 dynamic_slide=False,
                 concatenate=False,
                 offset=0):

        super().__init__(folder,
                         ['consumption'],
                         ['consumption'],
                         clients,
                         min_date,
                         max_date,
                         filters,
                         batch_size,
                         sliding_window,
                         pred_window,
                         percentage,
                         standardize,
                         dtype,
                         device,
                         dynamic_slide,
                         concatenate)

        self.files = files[offset:clients + offset]

        if clients > len(self.files):
            self.clients = len(self.files)
            raise RuntimeWarning(f'Number of clients is too large and set to {self.clients}')

        self.dataframes = [self.to_dataframe(os.path.join(self.folder, file)) for file in self.files]

    def preprocess_dataframe(self, df):
        df['Time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])

        if self.min_date:
            df = df[df['Time'] > self.min_date]

        if self.max_date:
            df = df[df['Time'] <= self.max_date]

        return df['consumption'].to_frame().fillna(0)

def concat_dataframes(index, set_of_dfs):
    transformed_dfs = [list(item) for item in zip(*set_of_dfs)]
    result_dfs = []

    for dfs in transformed_dfs:
        for df in dfs:
            df.set_index(index, inplace=True)
        result_dfs += pd.concat(dfs, axis=1).reset_index

    return result_dfs
