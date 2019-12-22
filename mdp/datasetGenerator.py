# mdp/dataSetGenerator.py

import numpy as np
import tensorflow as tf

from .utils import WithProperties


class DatasetGenerator(WithProperties, tf.keras.utils.Sequence):
    """ Генератор данных для keras.Model.fit_generator
    """

    def __init__(self, moving_window, batch_size, shuffle):
        super().__init__(cls=__class__,
                         **{'moving_window': moving_window, 'batch_size': batch_size,
                            'shuffle': shuffle, 'input_shape': moving_window.features.shape[1:],
                            'n_samples': len(moving_window)})
        if not shuffle:
            self.ix_ = np.arange(len(moving_window))
        self.on_epoch_end()

    def __len__(self):
        return (self.n_samples // self.batch_size) + (0 < self.n_samples % self.batch_size)

    def __getitem__(self, batch_ix):
        """ Возвращает batch_size массивов размером (window, n_features)
            (то есть куб).
        """
        batch_offs = batch_ix * self.batch_size
        batch_size = np.min([self.batch_size, self.n_samples - batch_offs])
        ix = self.ix_[batch_offs:batch_offs + batch_size]
        batch = self.moving_window.features[ix]
        return batch, self.get_target(ix)

    def on_epoch_end(self):
        if self.shuffle:
            self.ix_ = np.random.permutation(self.n_samples)

    def get_target_value_(self, ix):
        return self.moving_window.target[ix + (self.moving_window.window_size - 1 + self.moving_window.forecast_offset)]

    @classmethod
    def create(cls, moving_window, batch_size=1, shuffle=True):
        return cls(moving_window, batch_size, shuffle)


class RegressionGenerator(DatasetGenerator):

    def get_target(self, ix):
        return self.get_target_value_(ix)

    @classmethod
    def create(cls, *args, **kwargs):
        return super().create(*args, **kwargs)


class ClassificationGenerator(DatasetGenerator):

    def get_target(self, ix):
        target = self.get_target_value_(ix)
        direction = self.get_target_direction(target).astype(target.dtype.type)
        return direction

    @classmethod
    def create(cls, *args, **kwargs):
        return super().create(*args, **kwargs)

    @staticmethod
    def get_target_direction(target):
        return (target > 0).astype(np.int)


class MultitaskGenerator(RegressionGenerator, ClassificationGenerator):

    def get_target(self, ix):
        return [RegressionGenerator.get_target(self, ix),
                ClassificationGenerator.get_target(self, ix)]

    @classmethod
    def create(cls, *args, **kwargs):
        return super().create(*args, **kwargs)

# __EOF__
