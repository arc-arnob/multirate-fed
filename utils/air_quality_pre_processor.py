import joblib
from sklearn.discriminant_analysis import StandardScaler
import pandas as pd


class AirQualityProcessor:
    def __init__(self, data):
        self.data = data
        self.forecast_columns = None

    def preprocess_airquality_data(self):
        """Preprocess the Air Quality dataset."""
        self.data['date'] = pd.to_datetime(self.data[['year', 'month', 'day', 'hour']])
        self.data.set_index('date', inplace=True)
        self.data.drop(columns=['No', 'year', 'month', 'day', 'hour', 'wd', 'station'], inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.interpolate(method='linear', inplace=True, limit_direction='both')
        self.data.reset_index(inplace=True)

        # Normalize numerical columns
        numerical_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 
                             'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

        scaler = StandardScaler()
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])
        joblib.dump(scaler, 'airquality_scaler.pkl')

        self.forecast_columns = list(self.data.columns)[1:]  # Exclude date column
        return self.forecast_columns

