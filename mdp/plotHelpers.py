# mdp/plotHelpers.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import tensorflow.keras as keras
from matplotlib.ticker import Formatter

from .timeSeries import ScaledTimeSeries
from .timeSeries import invert_log_ret


class PlotDateGapsSkipper(Formatter):
    """ https://matplotlib.org/gallery/ticks_and_spines/date_index_formatter.html?highlight=customizing%20matplotlib
    """

    def __init__(self, dates, datetime_unit, shift=0, **kwargs):
        self.__dates = np.datetime_as_string(dates, unit=datetime_unit)
        self.__shift = shift

    def __call__(self, x, pos=0):
        ix = int(np.round(x)) - self.__shift
        if ix >= len(self.__dates) or ix < 0:
            return ''
        return self.__dates[ix]


def set_xaxis_timestamps_formatter(ax, timestamps, datetime_unit='M', **kwargs):
    formatter = PlotDateGapsSkipper(timestamps, datetime_unit, **kwargs)
    ax.xaxis.set_major_formatter(formatter)


def plot_ax(ax, title, x, xlabel, y, ylabel, rotation=0, **kwargs):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=rotation)
    ax.plot(x, y, **kwargs)


def plot_timeseries(ax, instrument_name, timeseries, series_title, xlabel, ylabel, datetime_unit='M', **kwargs):
    set_xaxis_timestamps_formatter(ax, timeseries.timestamps, datetime_unit, **kwargs)
    plot_ax(ax, f'{instrument_name} {series_title}',
            np.arange(len(timeseries.timestamps)), f'{xlabel}',
            timeseries.data, f'{ylabel}', **kwargs)


def plot_transformed_timeseries(instrument_name, transformed_timeseries,
                                title_orig, xlabel_orig, ylabel_orig,
                                title_transformed, xlabel_transformed, ylabel_transformed,
                                datetime_unit='M', figsize=(16, 6), **kwargs):
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    orig_series = transformed_timeseries.invert()

    plot_timeseries(ax[0], instrument_name, orig_series,
                    f'{title_orig} | ADF p_value={orig_series.p_value:.5f}',
                    f'{xlabel_orig}', f'{ylabel_orig}',
                    datetime_unit,
                    **kwargs)
    plot_timeseries(ax[1], instrument_name, transformed_timeseries,
                    f'{title_transformed} | ADF p_value={transformed_timeseries.p_value:.5f}',
                    f'{xlabel_transformed}', f'{ylabel_transformed}',
                    datetime_unit,
                    **kwargs)

    fig.tight_layout()


def plot_train_val_test_split(instr, train, val, test, feature, window_size, title, xlabel, ylabel, colors='brg',
                              figsize=(16, 6), datetime_unit='M', **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    set_xaxis_timestamps_formatter(ax, instr.timestamps, datetime_unit)
    ax.plot(np.arange(len(train)), getattr(train, feature).data, c=colors[0], zorder=3)
    shift = len(train) - window_size
    ax.plot(shift + np.arange(len(val)), getattr(val, feature).data, c=colors[1], zorder=2)
    shift += len(val) - window_size
    plot_ax(ax, f'{instr.instrument} {title}', shift + np.arange(len(test)), xlabel, getattr(test, feature).data,
            ylabel, c=colors[2], zorder=1, **kwargs)


def plot_macd(instrument_data, last_n, figsize=(16, 10), xlabel='Date', datetime_unit='M', **kwargs):
    fig = plt.figure(figsize=figsize)

    s = slice(-last_n, None)

    timestamps = instrument_data.timestamps[s]
    x = np.arange(len(timestamps))

    ax = plt.subplot(211)
    set_xaxis_timestamps_formatter(ax, timestamps, datetime_unit, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel('Adjusted close price, USD')
    plt.title(f'{instrument_data.instrument} close price')

    plt.plot(x, instrument_data.c[s].data, '.-g', label='MACD')

    ax = plt.subplot(212)
    set_xaxis_timestamps_formatter(ax, timestamps, datetime_unit, **kwargs)

    plt.xlabel(xlabel)
    plt.ylabel('MACD')
    plt.plot(x, instrument_data.macd[s].data, label='MACD')
    plt.plot(x, instrument_data.macd_signal[s].data, linestyle='dashed', label='Signal')
    plt.title(f'{instrument_data.instrument} MACD')

    macd_hist = instrument_data.macd[s].data - instrument_data.macd_signal[s].data

    bars = plt.bar(x, macd_hist, alpha=0.9, label='MACD Hist')
    for bar in np.argwhere(macd_hist > 0).squeeze():
        bars[bar].set_color('g')
    for bar in np.argwhere(macd_hist <= 0).squeeze():
        bars[bar].set_color('r')
    plt.legend(loc='best')

    fig.tight_layout()


def plot_all_features(instrument_data, n_cols=3, figsize=(20, 20), datetime_unit='M', **kwargs):
    n_rows = (len(instrument_data.feature_names) + 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f'{instrument_data.instrument} ({len(instrument_data.feature_names)} features)')
    timestamps = instrument_data.timestamps
    x = np.arange(len(timestamps))
    formatter = PlotDateGapsSkipper(timestamps, datetime_unit)
    for i, f in enumerate(instrument_data.feature_names):
        ax = axs[i // n_cols, i % n_cols]
        ax.xaxis.set_major_formatter(formatter)
        plot_ax(ax, f, x=x, xlabel='', y=getattr(instrument_data, f).data, ylabel=f, **kwargs)
        ax.set_title(f)
    fig.tight_layout()


def plot_regr_predictions(orig_price_series, instr_scaled, pred_log_ret,
                          figsize=(16, 10), xlabel='Date', datetime_unit='M'):
    if isinstance(instr_scaled.c, ScaledTimeSeries):
        pred_log_ret = instr_scaled.c.unscale(pred_log_ret)
        true_log_ret = instr_scaled.c.invert().data
    else:
        pred_log_ret = pred_log_ret.squeeze()
        true_log_ret = instr_scaled.c.data

    fig, ax = plt.subplots(2, 1, sharex=False, figsize=figsize)

    timestamps = instr_scaled.timestamps[-len(pred_log_ret):]
    x = np.arange(len(timestamps))
    formatter = PlotDateGapsSkipper(timestamps, datetime_unit)

    ax[0].xaxis.set_major_formatter(formatter)
    y_true = true_log_ret[-len(pred_log_ret):]
    mae = keras.losses.MAE(y_true, pred_log_ret).numpy()
    mse = keras.losses.MSE(y_true, pred_log_ret).numpy()

    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel('Log return (%)')
    ax[0].set_title(f'{instr_scaled.instrument} log return, MSE={mse:.6f}, MAE={mae:.4f}')
    ax[0].plot(x, y_true, label='Real', zorder=9)
    ax[0].plot(x, pred_log_ret, alpha=0.8, label='Predicted', zorder=10)
    ax[0].legend(loc='best')

    ax[1].xaxis.set_major_formatter(formatter)
    y_true = orig_price_series[-len(pred_log_ret):].data
    y_pred = invert_log_ret(orig_price_series, pred_log_ret).data
    mae = keras.losses.MAE(y_true, y_pred).numpy()
    mse = keras.losses.MSE(y_true, y_pred).numpy()

    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Close price, USD')
    ax[1].set_title(f'{instr_scaled.instrument} close price, MSE={mse:.6f}, MAE={mae:.4f}')
    ax[1].plot(x, y_true, label='Real', zorder=9)
    ax[1].plot(x, y_pred, alpha=0.8, label='Predicted', zorder=10)
    ax[1].legend(loc='lower right')

    ax2 = ax[1].twinx()
    ax2.set_ylabel('Price diff, USD')
    diff = np.round(y_true - y_pred, 12)
    bars = ax2.bar(x, diff, alpha=0.25, label='Difference', zorder=0)
    for bar in np.argwhere(diff > 0).squeeze():
        bars[bar].set_color('g')
    for bar in np.argwhere(diff <= 0).squeeze():
        bars[bar].set_color('r')
    ax2.grid()

    fig.tight_layout()


#
# ROC / PR curves (+ AUC)
#
def plot_curve(nrc, curve, y_test, y_probs, title, xlabel, ylabel):
    plt.subplot(nrc)
    auc = curve(y_test, y_probs)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title} AUC={auc:.3f}')
    plt.grid(True)


def plot_curve_xy(x, y):
    plt.plot(x, y, lw=2, color='b')


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


#
# Confusion matrix
#
def show_confusion_matrix(true, pred, labels, figsize=(3, 3)):
    plt.figure(figsize=figsize)

    conf_matrix = sklearn.metrics.confusion_matrix(true, pred)
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cbar=False)
    plt.title('Confusion matrix', fontsize=16)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return conf_matrix / conf_matrix.astype(np.float).sum(axis=0)


#
# Train history (the result of keras.Model.fit())
#
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

# __EOF__
