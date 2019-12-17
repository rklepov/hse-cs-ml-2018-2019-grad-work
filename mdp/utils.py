# mdp/Utils.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import statsmodels.tsa.stattools as stattools
import tensorflow.keras as keras
from matplotlib.ticker import Formatter


class PlotDateGapsSkipper(Formatter):
    """ https://matplotlib.org/gallery/ticks_and_spines/date_index_formatter.html?highlight=customizing%20matplotlib
    """

    def __init__(self, dates, unit='M'):
        self.__dates = np.datetime_as_string(dates, unit=unit)

    def __call__(self, x, pos=0):
        ix = int(np.round(x))
        if ix >= len(self.__dates) or ix < 0:
            return ''
        return self.__dates[ix]


def plot_regr_predictions(market_data, pred_log_ret, figsize=(16, 10)):
    fig, ax = plt.subplots(2, 1, sharex=False, figsize=figsize)

    pred_log_ret = pred_log_ret.squeeze()

    timestamps = market_data.timestamps[-len(pred_log_ret):]
    x = np.arange(len(timestamps))
    formatter = PlotDateGapsSkipper(timestamps)

    ax[0].xaxis.set_major_formatter(formatter)
    y_true = market_data.c_log_ret[-len(pred_log_ret):]
    mae = keras.losses.MAE(y_true, pred_log_ret).numpy()
    mse = keras.losses.MSE(y_true, pred_log_ret).numpy()

    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Log return (%)')
    ax[0].set_title(f'{market_data.instrument} daily log return, MSE={mse:.6f}, MAE={mae:.4f}')
    ax[0].plot(x, y_true, label='Real', zorder=9)
    ax[0].plot(x, pred_log_ret, alpha=0.8, label='Predicted', zorder=10)
    ax[0].legend(loc='best')

    ax[1].xaxis.set_major_formatter(formatter)
    y_true = market_data.c[-len(pred_log_ret):]
    y_pred = market_data.get_close_price_from_log_ret(slice(-(len(pred_log_ret) + 1), -1), pred_log_ret)
    mae = keras.losses.MAE(y_true, y_pred).numpy()
    mse = keras.losses.MSE(y_true, y_pred).numpy()

    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Close price, USD')
    ax[1].set_title(f'{market_data.instrument} daily close price, MSE={mse:.6f}, MAE={mae:.4f}')
    ax[1].plot(x, market_data.c[-len(pred_log_ret):], label='Real', zorder=9)
    ax[1].plot(x, y_pred, alpha=0.8, label='Predicted', zorder=10)
    ax[1].legend(loc='lower right')

    ax2 = ax[1].twinx()
    ax2.set_ylabel('Price diff, USD')
    diff = y_true - y_pred
    bars = ax2.bar(x, diff, alpha=0.25, label='Difference', zorder=0)
    for bar in np.argwhere(diff > 0).squeeze():
        bars[bar].set_color('g')
    for bar in np.argwhere(diff <= 0).squeeze():
        bars[bar].set_color('r')
    ax2.grid()

    fig.tight_layout()


def plot_curve_xy(x, y):
    plt.plot(x, y, lw=2, color='b')


def plot_curve(nrc, curve, y_test, y_probs, title, xlabel, ylabel):
    plt.subplot(nrc)
    auc = curve(y_test, y_probs)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title} AUC={auc:.3f}')
    plt.grid(True)


def plot_roc_curve(y_test, y_probs):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_probs)
    plt.plot(plt.xlim(), plt.ylim(), ls='-.', c='r', lw=0.5, zorder=0)
    plot_curve_xy(fpr, tpr)
    return sklearn.metrics.auc(fpr, tpr)


def plot_pr_curve(y_test, y_probs):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, y_probs)
    plt.plot(plt.xlim(), plt.ylim()[::-1], ls='-.', c='r', lw=0.5, zorder=0)
    plot_curve_xy(recall, precision)
    return sklearn.metrics.auc(recall, precision)


def plot_roc_pr_curves(y_test, y_probs, figsize=(15, 6)):
    plt.figure(figsize=figsize)
    plot_curve(121, plot_roc_curve, y_test, y_probs, 'ROC', 'FPR', 'TPR')
    plot_curve(122, plot_pr_curve, y_test, y_probs, 'PR', 'Recall', 'Precision')
    plt.show()


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


def show_confusion_matrix(true, pred, labels, figsize=(3, 3)):
    plt.figure(figsize=figsize)

    conf_matrix = sklearn.metrics.confusion_matrix(true, pred)
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cbar=False)
    plt.title('Confusion matrix', fontsize=16)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return conf_matrix / conf_matrix.astype(np.float).sum(axis=0)


def set_self_attr(self, cls, **kwargs):
    for key, value in kwargs.items():
        attr_name = f'_{cls.__name__}__{key}'
        setattr(self, attr_name, value)  # md.c_logret_
        # setattr(self, f'get_{key}',  # md.get_c_logret() etc.
        #         types.MethodType(functools.partial(lambda this, attr: getattr(this, attr), attr=attr_name), self))


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
