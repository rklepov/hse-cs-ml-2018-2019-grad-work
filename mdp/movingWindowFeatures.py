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

    def get_feature_windows(self, name):
        return self.features[:, :, self.feature_names.index(name)]

    @classmethod
    def create(cls, target, market_data, selected_features, window_size, forecast_offset=1):
        """
        TODO:
            Исходим из предположения, что исходные данные адекватного размера,
            поэтому крайние случаи (условия вроде window_size > len(target) и т.п.) не проверяются.
        """

        # допущение: длины рядов могут быть разными, но таймстемпы считая с хвоста - одинаковые
        # потому выравниваем все временные ряды "по хвосту" и обрезаем по длине самого короткого
        all_features = None
        for instrument_data in market_data:
            # все ряды одного инструмента имеют одинаковую длину
            series = [getattr(instrument_data, f) for f in selected_features if f in instrument_data.feature_names]
            instrument_features = np.column_stack([s.data for s in series])
            if all_features is None:
                all_features = instrument_features
            else:
                print('all_features', all_features.shape, 'instrument_features', instrument_features.shape)
                n_rows = np.amin([all_features.shape[0], instrument_features.shape[0]])
                all_features = np.hstack([all_features[-n_rows:, :], instrument_features[-n_rows:, :]])

        target = target[slice(-all_features.shape[0], None)].data

        # признаками будут последовательности "окон" для каждого временного ряда
        moving_window_features = np.lib.stride_tricks.as_strided(
            all_features,
            shape=(all_features.shape[0] + 1 - window_size - forecast_offset, window_size, all_features.shape[1]),
            strides=(all_features.strides[0], all_features.strides[0], all_features.strides[1]))

        return cls(market_data=market_data,
                   feature_names=selected_features,
                   features=moving_window_features,
                   target=target,
                   window_size=window_size,
                   forecast_offset=forecast_offset)
# __EOF__
