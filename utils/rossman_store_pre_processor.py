import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd


class RossmanDataProcessor:
    def __init__(self, data):
        """
        Initialize the Rossman Data Processor.

        Args:
            data (pd.DataFrame): The Rossman sales dataset.
        """
        self.data = data
        self.forecast_columns = None

    def preprocess_rossman_data(self):
        """
        Preprocess the Rossman Sales dataset.

        Steps:
        1. Parse and process the date column.
        2. Handle missing values.
        3. Feature engineering (e.g., extract date features, map 'StateHoliday', create 'Holiday').
        4. Normalize numerical columns.
        5. Save the scaler for future use.

        Returns:
            list: List of feature columns used for forecasting.
        """
        # Convert 'Date' to datetime and set as index
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)

        # Fill missing values
        self.data['Open'].fillna(1, inplace=True)  # Assume stores are open if no information is available
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)

        # Map 'StateHoliday' to binary values
        self.data['StateHoliday'] = self.data['StateHoliday'].map({'0': 0, 'a': 1, 'b': 1, 'c': 1, 0: 0})

        # Feature engineering: Create 'Holiday' by combining SchoolHoliday and StateHoliday
        self.data['Holiday'] = self.data.apply(
            lambda row: int(row['SchoolHoliday'] or row['StateHoliday']), axis=1
        )
        self.data.reset_index(inplace=True)

        # Normalize numerical columns
        numerical_columns = ['Sales', 'Customers', 'Promo']
        scaler = StandardScaler()
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])
        joblib.dump(scaler, 'rossman_scaler.pkl')

        # Define forecast columns
        self.forecast_columns = ['Sales']
        return self.forecast_columns
