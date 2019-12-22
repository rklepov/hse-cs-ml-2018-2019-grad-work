# mdp/timeSeries.py

import numpy as np
import pandas as pd
import scipy

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
        if type(item) is slice:
            indices = np.arange(*item.indices(len(self)))
        elif type(item) is list:
            indices = np.array(item)
        else:
            indices = item

        indices = np.clip(indices, 0, len(self) - 1)

        return self.slice(indices)

    def adf_report(self, print_results=False):
        return utils.adf_stationarity_test(self.data, print_results=print_results)

    def asSeries(self):
        return pd.Series(self.data, self.timestamps)

    def transform(self, transforms={}):
        return TransformedTimeSeries.create(self, transforms=transforms)

    def scale(self, scaler, **kwargs):
        return ScaledTimeSeries.create(self, scaler=scaler, **kwargs)

    def unscale(self, series):
        return series.squeeze()

    def slice(self, indices):
        return self.__class__(self.name, self.timestamps[indices], self.data[indices])

    @staticmethod
    def fwdfill_nans(arr):
        ix = np.where(~np.isnan(arr), np.arange(len(arr)), 0)
        np.maximum.accumulate(ix, out=ix)
        return arr[ix]


class TransformedTimeSeries(TimeSeries):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def invert(self):
        inverse_data = self.data
        orig_data_heads = self.orig_data_heads.copy()
        for f, a in reversed(list(self.transforms.items())):
            f_inverse = f'inverse_{f}'
            inverse_data = getattr(self, f_inverse)(head=orig_data_heads.pop(),
                                                    series=inverse_data, **a)

        return TimeSeries(self.name, np.concatenate([self.orig_ts_head, self.timestamps]), inverse_data)

    def slice(self, indices):
        if not ((np.arange(len(indices)) + indices[0]) == indices).all():
            return super().slice(indices)
        if len(indices) < len(self):
            series = self.invert()
            delta = len(series) - len(self)
            indices += delta
            indices = np.concatenate([indices[:delta] - delta, indices])
            return self.__class__.create(series[indices], self.transforms)
        else:
            return self

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


class ScaledTimeSeries(TimeSeries):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def invert(self):
        data_unscaled = self.unscale(self.data)
        return TimeSeries(self.name, self.timestamps, data_unscaled)

    def slice(self, indices):
        return self.__class__(**{'name': self.name,
                                 'timestamps': self.timestamps[indices],
                                 'data': self.data[indices],
                                 'scaler': self.scaler})

    def unscale(self, series):
        return self.scaler.inverse_transform(series.reshape(-1, 1)).squeeze()

    @classmethod
    def create(cls, time_series, scaler, **kwargs):
        data = time_series.data

        # assuming sklearn api interface
        if type(scaler) is type:
            # if passed a type (class) then fit the scaler (train)
            scaler = scaler(**kwargs)
            scaled_data = scaler.fit_transform(data.reshape(-1, 1)).squeeze()
        else:
            # if passed with an instance then use the fitted scaler (test)
            scaled_data = scaler.transform(data.reshape(-1, 1)).squeeze()

        return cls(**{'name': time_series.name,
                      'timestamps': time_series.timestamps,
                      'data': scaled_data,
                      'scaler': scaler})


def invert_log_ret(price_series, log_ret_series):
    ratios = np.exp(log_ret_series.data)
    prices = price_series.data[-(len(log_ret_series) + 1):-1] * ratios
    return prices

# __EOF__
