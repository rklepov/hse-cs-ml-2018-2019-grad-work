# mdp/movingWindowFeatures.py

import numpy as np

from .utils import WithProperties


class MovingWindowFeatures(WithProperties):
    """ Хранилище признаков: последовательность "окон" временных рядов + массив целевых переменных
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.features)

    @classmethod
    def get_target_direction(cls, target):
        return (target > 0).astype(np.int)

    @classmethod
    def create(cls, target, market_data, selected_features, window_size=30, forecast_offset=1):
        """
        TODO:
            Исходим из предположения, что исходные данные адекватного размера,
            поэтому крайние случаи (условия вроде window_size > len(target) и т.п.) не проверяются.
        """

        # допущение: длины рядов могут быть разными, но таймстемпы считая с хвоста - одинаковые
        # потому выравниваем все временные ряды "по хвосту" и обрезаем по длине самого короткого
        all_features = None
        all_transformed_series = []
        for instrument_data in market_data:
            transformed_series = [getattr(instrument_data, n).transform(data_slice=instrument_data.slice, **f) for n, f
                                  in selected_features.items()]
            feature_slice = slice(-np.amin([len(f) for f in transformed_series]), None)
            instrument_features = np.column_stack([f[feature_slice] for f in transformed_series])
            if all_features is None:
                all_features = instrument_features
            else:
                n_rows = np.amin(all_features.shape[0], instrument_features.shape[0])
                all_features = np.hstack([all_features[-n_rows:, :], instrument_features[-n_rows:, :]])
            all_transformed_series.append(transformed_series)

        target = target.transform(data_slice=slice(-all_features.shape[0],None))

        # признаками будут последовательности "окон" для каждого временного ряда
        moving_window_features = np.lib.stride_tricks.as_strided(
            all_features,
            shape=(all_features.shape[0] + 1 - window_size - forecast_offset, window_size, all_features.shape[1]),
            strides=(all_features.strides[0], all_features.strides[0], all_features.strides[1]))

        return cls(market_data=market_data,
                   feature_names=list(selected_features.keys()),
                   features=moving_window_features,
                   target=target,
                   window_size=window_size,
                   forecast_offset=forecast_offset)
# __EOF__
