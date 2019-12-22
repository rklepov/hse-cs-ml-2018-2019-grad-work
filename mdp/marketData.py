# mdp/marketData.py

import os

import numpy as np
import pandas_datareader as pdr
import talib

from .timeSeries import TimeSeries
from .utils import WithProperties


class MarketData(WithProperties):
    """ Загрузка и хранение рыночных данных и индикаторов
    """

    def __init__(self, instrument, timestamps, **kwargs):
        feature_names = [k for k in kwargs.keys()]

        # временные ряды могут иметь разную длину, потому что
        # при вычислении индикаторов часть начальных значений может быть отброшена
        tail_slice = slice(-np.amin([len(kwargs[f]) for f in feature_names]), None)

        features = {k: v[tail_slice] for k, v in kwargs.items() if k in feature_names}
        timestamps = timestamps[tail_slice]

        super().__init__(instrument=instrument, timestamps=timestamps, feature_names=feature_names,
                         **features)

    def __len__(self):
        return len(self.timestamps)

    def get_close_price_from_log_ret(self, c_slice, log_ret):
        """ Переход от предсказанных логарифмических доходностей к ценам.

            Предсказанная доходность следующего периода умножается на реальную цену предыдущего периода, тем самым
            получаем предсказанную цену следующего периода.
        """
        return self.c[c_slice] * np.exp(log_ret).squeeze()

    def select_transform(self, selected_features):
        transformed_features = {k: getattr(self, k).transform(transforms=v) for k, v in selected_features.items()}
        return self.__class__(self.instrument, self.timestamps, **transformed_features)

    def train_test_split(self, timestamp, window_size, forecast_offset=1, scaler=None, scaler_kwargs={}):
        split_ix = np.amax(np.where(self.timestamps < self.timestamps.dtype.type(timestamp)))
        train_slice = slice(None, split_ix + 1)
        test_slice = slice((split_ix + 1) - (window_size + forecast_offset - 1), None)

        train_features = {}
        test_features = {}
        for f in self.feature_names:
            series = getattr(self, f)
            train_series, test_series = series[train_slice], series[test_slice]
            if scaler is None:
                train_features[f] = train_series
                test_features[f] = test_series
            else:
                train_features[f] = train_series.scale(scaler, **scaler_kwargs)
                test_features[f] = test_series.scale(train_features[f].scaler)

        return (
            self.__class__(self.instrument, self.timestamps[train_slice], **train_features),
            self.__class__(self.instrument, self.timestamps[test_slice], **test_features)
        )

    @classmethod
    def create_(cls, instrument, timestamps, df, **kwargs):
        """ Обобщённая производящая функция.

            Ожидает на входе словарь, в котором ключи 'o', 'h', 'l', 'c', 'v' замаплены
            на соответствующие имена колонок датафрейма df.
        """
        init_kwargs = {'instrument': instrument, 'timestamps': timestamps}
        init_kwargs.update({k: TimeSeries(k, timestamps, df[kwargs[k]].values) for k in 'ohlcv'})

        # [скорректированная] цена закрытия - наш главный признак для вычисления индикаторов
        open_price, high_price, low_price, close_price, volume = [init_kwargs[k].data for k in 'ohlcv']

        # добавим индикаторы
        # TODO: динамическая настройка списка индикаторов?

        def ts_from_indi(indi, timestamps=timestamps):
            return {k: TimeSeries(name=k, timestamps=timestamps, data=v) for k, v in indi}

        # EMA14
        init_kwargs.update(ts_from_indi(cls.indi_ema(close_price, 14)))
        # EMA30
        init_kwargs.update(ts_from_indi(cls.indi_ema(close_price, 30)))
        # MACD: быстрая, медленная и сигнальная линии со "стандартными" периодами
        # RSI с периодом 14
        # Bollinger Bands
        for indi in (cls.indi_macd, cls.indi_rsi, cls.indi_bband):
            init_kwargs.update(ts_from_indi(indi(close_price)))
        # Williams % R
        init_kwargs.update(ts_from_indi(cls.indi_willr(high_price, low_price, close_price)))
        # Stochastic
        init_kwargs.update(ts_from_indi(cls.indi_stochastic(high_price, low_price, close_price)))
        # Chaikin A / D Line
        init_kwargs.update(ts_from_indi(cls.indi_adline(high_price, low_price, close_price, volume)))
        # Chaikin A / D Oscillator
        init_kwargs.update(ts_from_indi(cls.indi_adosc(high_price, low_price, close_price, volume)))

        return cls(**init_kwargs)

    @classmethod
    def create_from_tiingo(cls, instrument, *args, **kwargs):
        """ Загрузка данных через pandas_datareader is https://www.tiingo.com

            Преимущество данного источника, в частности, в том, что он предоставляет
            все 4 скорректированные (adjusted) цены (а не только цену закрытия)
        """
        df = pdr.get_data_tiingo(instrument, api_key=os.environ.get('TIINGO_API_KEY'), *args, **kwargs)
        return cls.create_(instrument, df.index.get_level_values('date').values, df,
                           **{'o': 'adjOpen', 'h': 'adjHigh', 'l': 'adjLow', 'c': 'adjClose', 'v': 'adjVolume'})

    @classmethod
    def create_from_quandl(cls, instrument, *args, **kwargs):
        """ Загрузка данных через pandas_datareader is https://www.quandl.com
        """
        df = pdr.get_data_quandl(instrument, api_key=os.environ.get('QUANDL_API_KEY'), *args, **kwargs)
        return cls.create_(instrument, df.index.values, df,
                           **{'o': 'AdjOpen', 'h': 'AdjHigh', 'l': 'AdjLow', 'c': 'AdjClose', 'v': 'AdjVolume'})

    @classmethod
    def create_from_alphavantage_intraday(cls, instrument, *args, **kwargs):
        """ Загрузка данных через pandas_datareader is https://www.alphavantage.co
            У них есть внутридневные (минутные) данные, но только за неделю назад
        """
        df = pdr.get_data_alphavantage(instrument, api_key=os.environ.get('ALPHAVANTAGE_API_KEY'),
                                       function='TIME_SERIES_INTRADAY', *args, **kwargs)
        return cls.create_(instrument, df.index.astype('datetime64[ns]').values, df,
                           **{'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})

    @staticmethod
    def indi_ema(p, timeperiod):
        """ EMA
        """
        return [(f'ema{timeperiod}', talib.EMA(p, timeperiod))]

    @staticmethod
    def indi_macd(p, fast=12, slow=26, signal=9):
        """ Moving Average Convergence Divergence
        """
        return zip(['macd', 'macd_signal'],
                   talib.MACD(p, fast, slow, signal)[:2])

    @staticmethod
    def indi_rsi(p, period=14):
        """ Relative Strength Index
        """
        return [(f'rsi{period}', talib.RSI(p, period))]

    @staticmethod
    def indi_bband(p, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
        """ Bollinger Bands
        """
        return zip([f'bband{timeperiod}_{s}' for s in ['upper', 'middle', 'lower']],
                   talib.BBANDS(p, timeperiod, nbdevup, nbdevdn, matype))

    @staticmethod
    def indi_willr(high, low, close, timeperiod=14):
        """ Williams % R
        """
        return [(f'willr{timeperiod}', talib.WILLR(high, low, close, timeperiod))]

    @staticmethod
    def indi_stochastic(high, low, close):
        """ Stochastic
        """
        return zip([f'stoch{s}' for s in 'kd'], talib.STOCH(high, low, close))

    @staticmethod
    def indi_adline(high, low, close, volume):
        """ Chaikin A / D Line
        """
        return [('adline', talib.AD(high, low, close, volume.astype(np.float)))]

    @staticmethod
    def indi_adosc(high, low, close, volume):
        """ Chaikin A / D Oscillator
        """
        return [('adosc', talib.ADOSC(high, low, close, volume.astype(np.float)))]

# __EOF__
