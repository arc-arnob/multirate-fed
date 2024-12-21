import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Flatten
from utils.utils import create_time_series

class LSTMModel(tf.keras.Model):
    def __init__(self,
                 num_features,
                 time_window,
                 output_window,
                 num_labels,
                 hidden_size=16,
                 dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        
        # LSTM layers
        self.lstm1 = LSTM(hidden_size, activation='relu', return_sequences=True, 
                          input_shape=(time_window, num_features), name='lstm_1')
        self.lstm2 = LSTM(hidden_size, activation='relu', return_sequences=False, name='lstm_2')

        # Dropout layer
        self.dropout = Dropout(dropout_rate, name='dropout')

        # Fully connected layers
        self.fc1 = Dense(num_features * time_window, activation='relu', name='fc1')
        self.fc2 = Dense(num_labels * output_window, activation='linear', name='fc2')

    def call(self, inputs):
        """
        Forward pass for the LSTMModel.
        """
        if isinstance(inputs, tuple):
            inputs = inputs[0] 
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

# Generate synthetic data
def generate_data(num_samples, time_window, output_window, num_features):
    X = np.random.rand(num_samples, time_window, num_features).astype(np.float32)
    y = np.random.rand(num_samples, output_window).astype(np.float32)
    return X, y
    
# Plotting function
def plot_generated_data(X, y, num_to_plot=3):
    """
    Plots a subset of the generated time-series data.
    :param X: Generated input features (num_samples, time_window, num_features)
    :param y: Generated targets (num_samples, output_window)
    :param num_to_plot: Number of samples to plot
    """
    plt.figure(figsize=(12, num_to_plot * 4))

    for i in range(num_to_plot):
        plt.subplot(num_to_plot, 1, i + 1)
        for feature in range(X.shape[2]):  # Plot each feature in the sample
            plt.plot(X[i, :, feature], label=f'Feature {feature + 1}')
        plt.title(f'Sample {i + 1} - Input and Target')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value')
        plt.legend(loc='upper right')

        # Annotate the corresponding target
        plt.axhline(y=y[i], color='red', linestyle='--', label='Target')
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# # Training and evaluation
# if __name__ == "__main__":
#     # Parameters
#     num_features = 1
#     time_window = 32
#     output_window = 1
#     num_labels = 1
#     hidden_size = 20
#     dropout_rate = 0.2
#     epochs = 30
#     batch_size = 32

#     # Generate data
#     # num_samples = 1000
#     X, y = create_time_series(processor.data, 'PM2.5', n_past=32, n_future=1) #generate_data(num_samples, time_window, output_window, num_features) #
#     train_size = int(0.8 * processor.data.shape[0])
#     X_train, X_val = X[:train_size], X[train_size:]
#     y_train, y_val = y[:train_size], y[train_size:]

#     # Build and compile the model
#     model = LSTMModel(num_features=num_features, 
#                       time_window=time_window,
#                       output_window=output_window, 
#                       num_labels=num_labels,
#                       hidden_size=hidden_size,
#                       dropout_rate=dropout_rate)

#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#     # Train the model
#     history = model.fit(
#         [X_train], y_train,  # Wrap X_train in a list for compatibility with `call`
#         validation_data=([X_val], y_val),	
#         epochs=epochs,
#         batch_size=batch_size,
#         verbose=1
#     )
    

#     # Plot training and validation loss
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.legend()
#     plt.show()