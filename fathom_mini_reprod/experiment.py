import os
from abc import ABC, abstractmethod

import dateutil.parser
import torch
from .dataset import AirQualityDataset

from fathom_mini_reprod.const import AIR_QUALITY_PATH, ELECTRICITY_PATH, SOLAR_PATH, AIR_QUALITY_FILES, SOLAR_FILES, \
    ELECTRICITY_PRO_FILES, SOLAR_FILES_MV, ROSSMAN_PRO_FILES, ROSSMAN_PATH, CRYPTO_PRO_FILES, CRYPTO_PATH


class Experiment(ABC):
    def __init__(self,
                 path,
                 batch_size,
                 num_tasks,
                 window,
                 problem_type="classification",
                 dtype=torch.float64,
                 seasonality=12):
        self.batch_size = batch_size
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        self.num_tasks = num_tasks
        self.window = window
        self.dtype = dtype
        self.seasonality = seasonality

        self.problem_type = problem_type

    
    @abstractmethod
    def get_dataset(self,
                    percentage=1.0,
                    device=torch.device("cpu"),
                    dynamic_slide=False,
                    concatenate=False,
                    offset=0):
        pass

class AirQuality(Experiment):
    def __init__(self,
                 batch_size=None,
                 num_tasks=None,
                 window=None,
                 pred_window=None,
                 features=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'],
                 labels=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']):
        batch_size = batch_size if batch_size else 8
        num_tasks = num_tasks if num_tasks else 9
        window = window if window else 13
        pred_window = pred_window if pred_window else 1

        super().__init__(AIR_QUALITY_PATH, batch_size, num_tasks, window, "regression", seasonality=24)

        self.files = AIR_QUALITY_FILES

        self.features = features
        self.features_for_dataset = features
        self.labels = labels

        self.distributed = False
        self.num_features = len(self.features)

        if isinstance(self.features[0], list):
            self.num_features = [len(f) for f in self.features]
            self.distributed = True
            self.features_for_dataset = [item for sublist in features for item in sublist]

        self.num_labels = len(self.labels)

        self.pred_window = pred_window

    def get_dataset(self,
                    percentage=1.0,
                    device=torch.device("cpu"),
                    dynamic_slide=False,
                    concatenate=False,
                    offset=0):
        return AirQualityDataset(self.path,
                                 self.files,
                                 self.features_for_dataset,
                                 self.labels,
                                 self.num_tasks,
                                 min_date=dateutil.parser.parse('2014-09-01'),
                                 max_date=dateutil.parser.parse('2014-11-12 19:00'),
                                 batch_size=self.batch_size,
                                 sliding_window=self.window,
                                 pred_window=self.pred_window,
                                 percentage=percentage,
                                 dtype=self.dtype,
                                 device=device,
                                 dynamic_slide=dynamic_slide,
                                 concatenate=concatenate,
                                 offset=offset
                                 )