# mdp/MovingWindowFeatures.py

import numpy as np

from . import Utils


class MovingWindowFeatures(object):
    """ Хранилище признаков: последовательность "окон" временных рядов + массив целевых переменных
    """

    def __init__(self, **kwargs):
        Utils.set_self_attr(self, __class__, **kwargs)

    def __len__(self):
        return len(self.features)

    @property
    def market_data(self):
        return self.__market_data

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def features(self):
        return self.__features

    @property
    def target(self):
        return self.__target

    @property
    def window_size(self):
        return self.__window_size

    @property
    def forecast_offset(self):
        return self.__forecast_offset

    @property
    def start_ix(self):
        return self.__start_ix

    @classmethod
    def get_target_direction(cls, target):
        return (target > 0).astype(np.int)

    @classmethod
    def scale_features(cls, features):
        mean = np.mean(features, axis=1)
        std = np.std(features, axis=1)
        return (features - mean[:, np.newaxis, :]) / std[:, np.newaxis, :]

    @classmethod
    def create(cls, market_data, selected_features=None, window_size=30, forecast_offset=1):
        """
            TODO: Исходим из предположения, что исходные данные адекватного размера,
            поэтому крайние случаи (условия вроде window > len(market_data), batch_size > 0 и т.п.) не проверяются.
        """
        if selected_features is None:
            selected_features = market_data.feature_names
        else:
            selected_features = list({f for f in market_data.feature_names if f in selected_features})

        # признаки: движущиеся окна временных рядов цен и индикаторов
        features = np.column_stack([getattr(market_data, f) for f in selected_features])

        # целевая переменная регрессии: логарифмическая доходность
        # целевая переменная классификации: (1) - доходность >0, (0) доходность <= 0
        target = market_data.c_log_ret

        # индикаторы могут содержать некоторое количество NaN в начале рядов
        # для простоты найдём максимальный индекс позиции NaN среди всех рядов
        # и пропсутим все элементы всех рядов, предшествующие этому индексу
        if np.isnan(features).any():
            start_ix = 1 + np.amax(np.argwhere(np.isnan(features)), axis=0)[0]
        else:
            start_ix = 0

        # признаками будут последовательности "окон" для каждого временного ряда
        time_series_features = np.lib.stride_tricks.as_strided(
            features,
            shape=(features.shape[0] + 1 - window_size - forecast_offset, window_size, features.shape[1]),
            strides=(features.strides[0], features.strides[0], features.strides[1]))

        return cls(market_data=market_data,
                   feature_names=selected_features, features=time_series_features,
                   target=target, window_size=window_size,
                   forecast_offset=forecast_offset, start_ix=start_ix)

# __EOF__
