# mdp/MovingWindowFeatures.py

import numpy as np

from . import Utils


class MovingWindowFeatures(object):
    """ Хранилище признаков: последовательность "окон" временных рядов + массив целевых переменных
    """

    def __init__(self, **kwargs):
        Utils.set_self_attr(self, **kwargs)

    def __len__(self):
        return len(self.features_)

    @classmethod
    def get_target_direction(cls, target):
        return (target > 0).astype(np.int)

    @classmethod
    def scale_features(cls, features):
        mean = np.mean(features, axis=1)
        std = np.std(features, axis=1)
        return (features - mean[:, np.newaxis, :]) / std[:, np.newaxis, :]

    @classmethod
    def create(cls, marketData, window=30, forecast=1):
        """
            TODO: Исходим из предположения, что исходные данные адекватного размера,
            поэтому крайние случаи (условия вроде window > len(marketData), batch_size > 0 и т.п.) не проверяются.
        """
        # признаки: движущиеся окна временных рядов цен и индикаторов
        features = np.column_stack([getattr(marketData, f)() for f in marketData.src_feature_accessors])

        # целевая переменная регрессии: логарифмическая доходность
        # целевая переменная классификации: (1) - доходность >0, (0) доходность <= 0
        target = marketData.get_c_logret()

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
            shape=(features.shape[0] + 1 - window - forecast, window, features.shape[1]),
            strides=(features.strides[0], features.strides[0], features.strides[1]))

        return cls(features=time_series_features, target=target, window=window,
                   forecast_offset=forecast, start_ix=start_ix)

# __EOF__
