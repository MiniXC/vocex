import numpy as np
import torch
import torch.nn as nn

class GaussianMinMaxScaler(nn.Module):
    """
    A min-max scaler that does the following, given the number of samples in the dataset N:
    1. Apply as square root to the data
    2. Apply a min-max scaler to the data ranging from the expected minimum and maximum of a Gaussian distribution with samples N
    """

    def __init__(self, width, for_tensors=True, floor=1e-6, sqrt=True):
        if for_tensors:
            super().__init__()
            self.isnan = torch.isnan
        else:
            self.isnan = np.isnan
        self.sqrt = sqrt
        expected_max = width / 2
        _max = float("nan")
        _min = float("nan")
        _n = 0
        for_tensors = for_tensors
        floor = floor
        _scale = float("nan")
        if for_tensors:
            # register buffers
            self.register_buffer("max", torch.tensor(_max))
            self.register_buffer("min", torch.tensor(_min))
            self.register_buffer("_scale", torch.tensor(_scale))
            self.register_buffer("expected_max", torch.tensor(expected_max))
            self.register_buffer("floor", torch.tensor(floor))
            self.register_buffer("_n", torch.tensor(_n))
            self.for_tensors = for_tensors
        else:
            self.max = _max
            self.min = _min
            self._scale = _scale
            self.expected_max = expected_max
            self.floor = floor
            self._n = _n
            self.for_tensors = for_tensors

    def partial_fit(self, X):
        scale_change = False
        if self.isnan(self.max):
            self.max = X.max()
            if self.for_tensors:
                self.max = self.max.detach()
            scale_change = True
        else:
            max_candidate = X.max()
            if max_candidate > self.max:
                if self.for_tensors:
                    self.max = max_candidate.detach()
                else:
                    self.max = max_candidate
                scale_change = True
        if self.isnan(self.min):
            self.min = X.min()
            if self.for_tensors:
                self.min = self.min.detach()
            scale_change = True
        else:
            min_candidate = X.min()
            if min_candidate < self.min:
                if self.for_tensors:
                    self.min = min_candidate.detach()
                else:
                    self.min = min_candidate
                scale_change = True
        if scale_change:
            if self.for_tensors:
                if self.sqrt:
                    self._scale = torch.sqrt(self.max - self.min + self.floor)
                else:
                    self._scale = self.max - self.min + self.floor
            else:
                if self.sqrt:
                    self._scale = np.sqrt(self.max - self.min + self.floor)
                else:
                    self._scale = self.max - self.min + self.floor
        # add numpy array size to n, regardless of shape
        self._n += len(X.flatten())

    def transform(self, X):
        X = X - self.min + self.floor
        if self.for_tensors:
            X = X.clamp_(min=self.floor)
            if self.sqrt:
                X = torch.sqrt(X)
            # print if any nan values
            # if torch.isnan(X).any():
            #     print("NAN values in GaussianMinMaxScaler!")
        else:
            X = np.clip(X, a_min=self.floor, a_max=None)
            if self.sqrt:
                X = np.sqrt(X)
        X = X / self._scale
        X = X - 0.5
        X = X * self.expected_max
        return X

    def inverse_transform(self, X):
        X = X.detach()
        X = X / self.expected_max
        X = X + 0.5
        X = X * self._scale
        if self.sqrt:
            X = X ** 2
        X = X + self.min - self.floor
        return X