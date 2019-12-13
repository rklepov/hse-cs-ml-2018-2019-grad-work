# mdp/DataSetGenerator.py

import numpy as np
import tensorflow as tf

from . import Utils


class DatasetGenerator(tf.keras.utils.Sequence):
    """ Генератор данных для keras.Model.fit_generator
    """

    def __init__(self, moving_window, batch_size, start_ix, stop_ix, shuffle, scale):
        Utils.set_self_attr(self, __class__,
                            **{'moving_window': moving_window, 'batch_size': batch_size,
                               'start_ix': start_ix, 'stop_ix': stop_ix,
                               'shuffle': shuffle, 'scale': scale})
        if not shuffle:
            self.ix_ = np.arange(start_ix, stop_ix)
        self.on_epoch_end()

    def __len__(self):
        return (self.get_n_samples() // self.batch_size) + (0 < self.get_n_samples() % self.batch_size)

    def __getitem__(self, batch_ix):
        """ Возвращает batch_size массивов размером (window, n_features)
            (то есть куб).
        """
        batch_offs = batch_ix * self.batch_size
        batch_size = np.min([self.batch_size, self.get_n_samples() - batch_offs])
        ix = self.ix_[batch_offs:batch_offs + batch_size]
        batch = self.moving_window.features[ix]

        if self.scale:
            # TODO: !допущение!: стандартизация в пределах одного окна временного ряда
            batch = self.moving_window.scale_features(batch)

        return batch, self.get_target(self.moving_window, ix)

    @property
    def moving_window(self):
        return self.__moving_window

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def start_ix(self):
        return self.__start_ix

    @property
    def stop_ix(self):
        return self.__stop_ix

    @property
    def shuffle(self):
        return self.__shuffle
        return self.__shuffle

    @property
    def scale(self):
        return self.__scale

    def on_epoch_end(self):
        if self.shuffle:
            self.ix_ = self.start_ix + np.random.permutation(self.stop_ix - self.start_ix)

    def get_input_shape(self):
        return self.moving_window.features.shape[1:]

    def get_n_samples(self):
        return len(self.ix_)

    def get_target_value_(self, moving_window, ix):
        return self.moving_window.target[ix + self.moving_window.window_size - 1 + self.moving_window.forecast_offset]

    @classmethod
    def create_train_test_(cls, moving_window, test_split, batch_size=1, scale=True):
        """
            TODO: Исходим из предположения, что исходные данные адекватного размера,
            поэтому крайние случаи (условия вроде window > len(market_data), batch_size > 0 и т.п.) не проверяются.

            features  -  TimeSeriesFeatures
        """
        start_ix = moving_window.start_ix
        if type(test_split) in (float, np.float):
            split_ix = int(start_ix + (len(moving_window) - start_ix) * (1 - test_split)) - 1
        else:
            timestamps = moving_window.market_data.timestamps
            split_ix = np.amax(np.where(timestamps < timestamps.dtype.type(test_split)))

        return (
            cls(moving_window, batch_size, start_ix, split_ix - moving_window.window_size, True, scale),  # train
            cls(moving_window, batch_size, split_ix - moving_window.window_size,
                len(moving_window), False, scale)  # test
        )


class RegressionGenerator(DatasetGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_target(self, moving_window, ix):
        return self.get_target_value_(moving_window, ix)

    @classmethod
    def create_train_test(cls, *args, **kwargs):
        return super().create_train_test_(*args, **kwargs)


class ClassificationGenerator(DatasetGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_target(self, moving_window, ix):
        target = self.get_target_value_(moving_window, ix)
        return moving_window.get_target_direction(target).astype(target.dtype.type)

    @classmethod
    def create_train_test(cls, *args, **kwargs):
        return super().create_train_test_(*args, **kwargs)


class MultitaskGenerator(RegressionGenerator, ClassificationGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_target(self, moving_window, ix):
        return [RegressionGenerator.get_target(self, moving_window, ix),
                ClassificationGenerator.get_target(self, moving_window, ix)]

    @classmethod
    def create_train_test(cls, *args, **kwargs):
        return super().create_train_test_(*args, **kwargs)
# __EOF__
