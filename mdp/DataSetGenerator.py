# mdp/DataSetGenerator.py

import numpy as np
import tensorflow as tf

from . import Utils


class DataSetGenerator(tf.keras.utils.Sequence):
    """ Генератор данных для keras.Model.fit_generator
    """

    def __init__(self, features, batch_size, start_ix, stop_ix, shuffle, scale):
        Utils.set_self_attr(self, **{'features': features, 'batch_size': batch_size,
                                     'start_ix': start_ix, 'stop_ix': stop_ix,
                                     'shuffle': shuffle, 'scale': scale})
        if not shuffle:
            self.ix_ = np.arange(start_ix, stop_ix)
        self.on_epoch_end()

    def __len__(self):
        return (self.get_n_samples() // self.get_batch_size()) + (0 < self.get_n_samples() % self.get_batch_size())

    def __getitem__(self, batch_ix):
        """ Возвращает batch_size массивов размером (window, n_features)
            (то есть куб).
        """
        batch_offs = batch_ix * self.batch_size_
        batch_size = np.min([self.batch_size_, self.get_n_samples() - batch_offs])
        ix = self.ix_[batch_offs:batch_offs + batch_size]
        batch = self.features_.get_features()[ix]
        if self.scale_:
            # допущение: стандартизация в пределах одного окна временного ряда
            batch = self.features_.scale_features(batch)
        target = self.features_.get_target()[ix + self.features_.get_window() - 1
                                             + self.features_.get_forecast_offset()]

        return batch, [target, self.features_.get_target_direction(target)]

    def on_epoch_end(self):
        if self.shuffle_:
            self.ix_ = self.start_ix_ + np.random.permutation(self.stop_ix_ - self.start_ix_)

    def get_input_shape(self):
        return self.features_.get_features().shape[1:]

    def get_n_samples(self):
        return len(self.ix_)

    def get_batch_size(self):
        return self.batch_size_

    @classmethod
    def create_train_test(cls, features, test_size=0.2, batch_size=1, scale=True):
        """
            TODO: Исходим из предположения, что исходные данные адекватного размера,
            поэтому крайние случаи (условия вроде window > len(marketData), batch_size > 0 и т.п.) не проверяются.

            features  -  TimeSeriesFeatures
        """
        start_ix = features.get_start_ix()
        split_ix = int(start_ix + (len(features) - start_ix) * (1 - test_size)) - 1

        return (
            cls(features, batch_size, start_ix, split_ix + 1, True, scale),  # train
            cls(features, batch_size, split_ix + features.get_window(), len(features), False, scale)  # test
        )

# __EOF__
