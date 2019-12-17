# mdp/timeSeries.py

import numpy as np
import scipy
import pandas as pd

import mdp.utils as utils
from .utils import WithProperties


class TimeSeries(WithProperties):
    """
    TODO:
        [assumption]: no np.NaN values in the series (except the potential sequence of np.NaN at the start of it)
    """

    def __init__(self, name, timestamps, data, **kwargs):
        assert len(timestamps) == len(data)

        # drop initial NaNs
        isnan = np.flatnonzero(np.isnan(data))
        start_ix = 0
        if 0 < len(isnan):
            start_ix = 1 + np.amax(isnan)

        timestamps = timestamps[start_ix:]
        data = data[start_ix:]
        p_value, is_stationary = utils.adf_stationarity_test(data)

        super().__init__(cls=__class__,
                         **{'name': name,
                            'timestamps': timestamps,
                            'data': data,
                            'is_stationary': is_stationary,
                            'p_value': p_value,
                            'mean': np.mean(data),
                            'std': np.std(data)})
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return TimeSeries(self.name, self.timestamps[item],  self.data[item])

    def adf_report(self, print_results=False):
        return utils.adf_stationarity_test(self.data, print_results=print_results)

    def asSeries(self):
        return pd.Series(self.data, self.timestamps)

    def transform(self, transforms={}):
        return TransformedTimeSeries.create(self, transforms=transforms)

    def scale(self, scaler):
        return ScaledTimeSeries.create(self, scaler=scaler)

    def split(self, timestamp, window_size, forecast_offset):
        split_ix = np.amax(np.where(self.timestamps < self.timestamps.dtype.type(timestamp)))
        return (
            self.__class__(self.name, self.timestamps[:split_ix + 1], self.data[:split_ix + 1]),
            self.__class__(self.name, self.timestamps[split_ix - window_size + 1:],
                           self.data[split_ix - window_size + 1:])
        )

    @staticmethod
    def fwdfill_nans(arr):
        ix = np.where(~np.isnan(arr), np.arange(len(arr)), 0)
        np.maximum.accumulate(ix, out=ix)
        return arr[ix]


class ScaledTimeSeries(TimeSeries):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def inverse(self):
        data_unscaled = self.scaler.inverse_transform(self.data.reshape(-1, 1)).squeeze()
        return TimeSeries(self.name, self.timestamps, data_unscaled)

    @classmethod
    def create(cls, time_series, scaler):
        data = time_series.data

        # assuming sklearn api interface
        if type(scaler) is type:
            # if passed a type (class) then fit the scaler (train)
            scaler = scaler()  # using the default constructor
            scaled_data = scaler.fit_transform(data.reshape(-1, 1)).squeeze()
        else:
            # if passed with an instance then use the fitted scaler (test)
            scaled_data = scaler.transform(data.reshape(-1, 1)).squeeze()

        return cls(**{'name': time_series.name,
                      'timestamps': time_series.timestamps,
                      'data': scaled_data,
                      'scaler': scaler})


class TransformedTimeSeries(TimeSeries):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def inverse(self):
        inverse_data = self.data
        orig_data_heads = self.orig_data_heads.copy()
        for f, a in reversed(list(self.transforms.items())):
            f_inverse = f'inverse_{f}'
            inverse_data = getattr(self, f_inverse)(head=orig_data_heads.pop(),
                                                    series=inverse_data, **a)

        return TimeSeries(self.name, np.concatenate([self.orig_ts_head, self.timestamps]), inverse_data)

    @classmethod
    def create(cls, time_series, transforms={}):

        data_transformed = time_series.data
        orig_data_heads = []
        for f, a in transforms.items():
            d = getattr(cls, f)(series=data_transformed, **a)
            orig_data_heads.append(data_transformed[:len(data_transformed) - len(d)])
            data_transformed = d

        return cls(**{'name': time_series.name,
                      'timestamps': time_series.timestamps[-len(data_transformed):],
                      'data': data_transformed,
                      'transforms': transforms,
                      'orig_data_heads': orig_data_heads,
                      'orig_ts_head': time_series.timestamps[:len(time_series.data) - len(data_transformed)]})

    @staticmethod
    def diffs(series, order=1, **kwargs):
        return np.diff(series, n=order)

    @staticmethod
    def inverse_diffs(head, series, **kwargs):
        order = len(head)
        result = np.zeros(order + len(series))
        result[order:] = series
        for i in range(order)[::-1]:
            result[i] = np.diff(head[:i + 1], i)[0]
            result[i:] = np.cumsum(result[i:])
        return result

    @staticmethod
    def move(series, value, **kwargs):
        return series + value

    @staticmethod
    def inverse_move(series, value, **kwargs):
        return series - value

    @staticmethod
    def ln(series, **kwargs):
        return np.log(series)

    @staticmethod
    def inverse_ln(series, **kwargs):
        return np.exp(series)

    @staticmethod
    def ratios(series, **kwargs):
        return series[1:] / series[:-1]

    @staticmethod
    def inverse_ratios(head, series, **kwargs):
        result = np.empty(1 + len(series))
        result[0] = head[-1]
        result[1:] = series
        return np.cumprod(result)

    @staticmethod
    def boxcox(series, lmbda, **kwargs):
        result = scipy.stats.boxcox(series, lmbda)
        return result[0] if lmbda is None else result

    @staticmethod
    def inverse_boxcox(series, lmbda, **kwargs):
        return scipy.special.inv_boxcox(series, lmbda)

# __EOF__
