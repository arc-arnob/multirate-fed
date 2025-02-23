import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self,
                 num_tasks,
                 num_features,
                 time_window,
                 output_window,
                 num_labels,
                 num_layers=2,
                 hidden_size=16):
        super(LSTMModel, self).__init__()

        self.output_window = output_window
        self.num_labels = num_labels
        self.dual = False

        self.intermediary = False

        self.lstm = nn.LSTM(num_features,
                            hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2)
        self.fc1 = nn.Linear(hidden_size * time_window, num_features * time_window)
        self.fc2 = nn.Linear(num_features * time_window, num_labels * output_window)

    def forward(self, x):
        B, T, D = x[0].shape

        x_, _ = self.lstm(x[0].reshape(-1, D))

        if self.intermediary:
            return [x_]

        x_ = self.fc1(x_.reshape(B, -1))
        x_ = self.fc2(x_)
        x_ = (x_.reshape(B, self.output_window, self.num_labels))

        return [x_]

    def global_state_dict(self):
        # Collect state dict of global parameters only
        global_state = {}
        for name, param in self.state_dict().items():
            if 'lstm' in name:
                global_state[name] = param
                break

        return global_state

    def output_intermediary_state(self, intermediary="default"):
        if intermediary != "default":
            raise NotImplementedError(f'Intermediary state "{intermediary} does not exist for LSTM"')

        self.intermediary = True

    def output_regular(self):
        self.intermediary = False