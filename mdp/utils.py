# mdp/utils.py

import functools
import glob
import os
import pickle

import pandas as pd
import statsmodels.tsa.stattools as stattools


class WithProperties(object):
    def __init__(self, **kwargs):
        self.set_props(**kwargs)

    def set_attr(self, name, value, cls=None):
        if cls is None:
            cls = self.__class__
        attr_name = f'_{cls.__name__}__{name}'
        setattr(self, attr_name, value)
        return cls, attr_name

    def set_prop(self, name, value, read_only=True, cls=None):
        cls, attr_name = self.set_attr(name, value, cls=cls)
        if name not in dir(cls):
            fget = functools.partial(lambda this, attr: getattr(this, attr), attr=attr_name)
            fset = None if read_only else functools.partial(lambda this, attr, val: setattr(this, attr, val),
                                                            attr=attr_name)
            setattr(cls, name, property(fget, fset))

    def set_props(self, read_only=True, cls=None, **kwargs):
        for name, value in kwargs.items():
            self.set_prop(name, value, read_only, cls)


def adf_stationarity_test(time_series, significance_level=0.05, print_results=False):
    """ Dickey-Fuller test

        http://www.insightsbot.com/blog/1MH61d/augmented-dickey-fuller-test-in-python
    """

    adf_test = stattools.adfuller(time_series, autolag='AIC')
    p_value = adf_test[1]
    stationary = p_value < significance_level

    if print_results:
        results = pd.Series(adf_test[0:4],
                            index=['ADF Test Statistic', 'P-Value', '# Lags Used', '# Observations Used'])
        # Add Critical Values
        for k, v in adf_test[4].items():
            results[f'Critical Value ({k})'] = v
        print('Augmented Dickey-Fuller Test Results:')
        print(results)

    return p_value, stationary

    # __EOF__


def train_model(model, save_dir, train_gen, test_gen, callbacks, epochs=32, force_train=False, verbose=0):
    hist_path = os.path.join(save_dir, 'train.history')
    if os.path.exists(hist_path) and not force_train:
        with open(hist_path, 'rb') as hist_dump:
            history = pickle.load(hist_dump)
        return history

    saved_models = glob.glob(os.path.join(save_dir, '*.hdf5'))
    if 0 < len(saved_models):
        for file in saved_models:
            os.remove(file)

    history = model.fit_generator(
        generator=train_gen,
        epochs=epochs,
        verbose=verbose,
        validation_data=test_gen,
        callbacks=callbacks,
        workers=4,
        shuffle=False)

    with open(hist_path, 'wb') as hist_dump:
        pickle.dump(history.history, hist_dump)

    return history.history
