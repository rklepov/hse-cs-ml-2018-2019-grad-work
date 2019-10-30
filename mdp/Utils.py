# mdp/Utils.py

import types
import functools
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as stattools


def plot_train_history(ax, hist, title, train, val, loc='best'):
    last_epoch = len(hist[train])
    ax.plot(1 + np.arange(last_epoch), hist[train])
    ax.plot(1 + np.arange(last_epoch), hist[val])

    ax.axvline(x=last_epoch, color='r', linestyle=':')
    bottom, top = ax.get_ylim()
    height = top - bottom
    ymid = bottom + height / 2
    ax.annotate(f'{last_epoch}', xy=(last_epoch, ymid))

    val_last = hist[val][last_epoch - 1]
    ax.axhline(y=val_last, color='g', linestyle=':')
    left, right = ax.get_xlim()
    y2 = [top - height / 3, bottom + height / 3][bool(val_last < ymid)]
    ax.annotate(f'{val_last:.4f}', xy=(last_epoch, val_last),
                xytext=(last_epoch - (last_epoch - left) / 4, y2),
                size=14,
                arrowprops=dict(arrowstyle='->',
                                connectionstyle=f'arc3,rad={.3 * [1, -1][bool(val_last < ymid)]:.1f}'))
    ax.set_title(title)
    ax.set_ylabel(train)
    ax.set_xlabel('epoch')
    ax.set_xticks(np.arange(0, last_epoch, 5))
    ax.legend(['train', 'val'], loc=loc)


def set_self_attr(self, **kwargs):
    for key, value in kwargs.items():
        attr_name = f'{key}_'
        setattr(self, attr_name, value)
        setattr(self, f'get_{key}',  # md.get_c_logret() etc.
                types.MethodType(functools.partial(lambda this, attr: getattr(this, attr), attr=attr_name), self))


def adf_stationarity_test(timeseries, significance_level=0.05, print_results=True):
    """ Dickey-Fuller test

        http://www.insightsbot.com/blog/1MH61d/augmented-dickey-fuller-test-in-python
    """
    adf_test = stattools.adfuller(timeseries, autolag='AIC')
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

    return (p_value, stationary)

# __EOF__
