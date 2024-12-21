import os
import warnings
import pandas as pd
import numpy as np
from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

# Suppress warnings
warnings.filterwarnings("ignore", module="torch")

# Set seed for reproducibility
set_seed(2023)


class TimeSeriesTrainer:
    def __init__(
        self,
        dataset_path,
        timestamp_column,
        forecast_columns,
        context_length,
        forecast_horizon,
        batch_size,
        num_workers,
    ):
        self.dataset_path = dataset_path
        self.timestamp_column = timestamp_column
        self.forecast_columns = forecast_columns
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = pd.read_csv(self.dataset_path, parse_dates=[self.timestamp_column])

    def prepare_datasets(self, train_split=0.7, test_split=0.2):
        """Splits the data into train, validation, and test sets."""
        num_train = int(len(self.data) * train_split)
        num_test = int(len(self.data) * test_split)
        num_valid = len(self.data) - num_train - num_test

        # Define split indices
        border1s = [
            0, 
            num_train - self.context_length,
            len(self.data) - num_test - self.context_length,
        ]
        border2s = [
            num_train,
            num_train + num_valid,
            len(self.data),
        ]

        # Train/Validation/Test splits
        self.train_data = select_by_index(self.data, start_index=border1s[0], end_index=border2s[0])
        self.valid_data = select_by_index(self.data, start_index=border1s[1], end_index=border2s[1])
        self.test_data = select_by_index(self.data, start_index=border1s[2], end_index=border2s[2])
        print(len(self.train_data), len(self.valid_data), len(self.test_data))


    def preprocess_airquality_data(self):
        """Preprocess the Air Quality dataset."""
        self.data['date'] = pd.to_datetime(self.data[['year', 'month', 'day', 'hour']])
        self.data.set_index('date', inplace=True)
        self.data.drop(columns=['No', 'year', 'month', 'day', 'hour', 'wd', 'station'], inplace=True)
        self.data.reset_index(inplace=True)
        # Drop rows with missing values
        self.data.dropna(inplace=True)

    def preprocess_data(self):
        """Preprocesses data and creates datasets."""
        self.preprocessor = TimeSeriesPreprocessor(
            timestamp_column=self.timestamp_column,
            input_columns=forecast_columns,
            output_columns=forecast_columns,
            scaling=True,
        )
        self.preprocessor = self.preprocessor.train(self.train_data)

        self.train_dataset = ForecastDFDataset(
            self.preprocessor.preprocess(self.train_data),
            timestamp_column=self.timestamp_column,
            target_columns=self.forecast_columns,
            context_length=self.context_length,
            prediction_length=self.forecast_horizon,
        )
        self.valid_dataset = ForecastDFDataset(
            self.preprocessor.preprocess(self.valid_data),
            timestamp_column=self.timestamp_column,
            target_columns=self.forecast_columns,
            context_length=self.context_length,
            prediction_length=self.forecast_horizon,
        )
        self.test_dataset = ForecastDFDataset(
            self.preprocessor.preprocess(self.test_data),
            timestamp_column=self.timestamp_column,
            target_columns=self.forecast_columns,
            context_length=self.context_length,
            prediction_length=self.forecast_horizon,
        )
        print("Data preprocessing completed.")

    def train_model(self, model_config, output_dir, num_epochs=10):
        """Trains the model and evaluates on validation data."""
        model = PatchTSTForPrediction(model_config)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            # learning_rate=0.001,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            dataloader_num_workers=self.num_workers,
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=3,
            logging_dir="./checkpoint/patchtst/electricity/pretrain/logs/",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss", # Metric to monitor for early stopping
            greater_is_better=False,
            label_names=["future_values"],
            report_to="none",
        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=0.0001
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            callbacks=[early_stopping_callback],
        )
        trainer.train()
        return trainer

    def finetune_model(self, pretrained_model_path, output_dir, num_epochs=20, learning_rate=1e-4):
        """Finetunes a pretrained model on a new dataset."""
        model = PatchTSTForPrediction.from_pretrained(
            pretrained_model_path,
            num_input_channels=len(self.forecast_columns),
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            dataloader_num_workers=self.num_workers,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            callbacks=[early_stopping_callback],
        )
        trainer.train()
        return trainer

    def evaluate_model(self, trainer):
        """Evaluates the model on test data."""
        results = trainer.evaluate(self.test_dataset)
        print("Test Results:", results)
        return results


if __name__ == "__main__":
    # Configurations
    dataset_path = "../data/ECL/ECL.csv"
    timestamp_column = "date"
    forecast_columns = None # ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']  # Update with actual columns
    context_length = 512
    forecast_horizon = 96
    batch_size = 16
    num_workers = 1

    # Foerecast columns
    aux_load = pd.read_csv(dataset_path)
    forecast_columns = list(aux_load.columns[1:])

    # Initialize and run training
    trainer_obj = TimeSeriesTrainer(
        dataset_path,
        timestamp_column,
        forecast_columns,
        context_length,
        forecast_horizon,
        batch_size,
        num_workers,
    )
    # trainer_obj.preprocess_airquality_data()
    trainer_obj.prepare_datasets()
    trainer_obj.preprocess_data()

    # Model Configuration for Pretraining
    pretrain_config = PatchTSTConfig(
        num_input_channels=len(forecast_columns),
        context_length=context_length,
        patch_length=16,
        patch_stride=16,
        prediction_length=forecast_horizon,
        random_mask_ratio=0.4,
        d_model=128,
        num_attention_heads=16,
        num_hidden_layers=3,
        ffn_dim=256,
        dropout=0.2,
        head_dropout=0.2,
        pooling_type=None,
        channel_attention=False,
        scaling="std",
        loss="mse",
        pre_norm=True,
        norm_type="batchnorm",
    )

    pretrain_output_dir = "./checkpoint/patchtst/pretrain/"
    pretrainer = trainer_obj.train_model(pretrain_config, pretrain_output_dir)
    print("Evaluating now....")
    trainer_obj.evaluate_model(pretrainer)

    # Finetuning on a New Dataset
    # target_dataset_path = "./target_dataset.csv"  # Update path
    # trainer_obj = TimeSeriesTrainer(
    #     target_dataset_path,
    #     timestamp_column,
    #     forecast_columns,
    #     context_length,
    #     forecast_horizon,
    #     batch_size,
    #     num_workers,
    # )
    # trainer_obj.prepare_datasets()
    # trainer_obj.preprocess_data()

    # finetune_output_dir = "./checkpoint/patchtst/finetune/"
    # finetuner = trainer_obj.finetune_model(pretrain_output_dir, finetune_output_dir)
    # trainer_obj.evaluate_model(finetuner)
