# mdp/MarketData.py

import os

import numpy as np
import pandas_datareader as pdr
import talib

from . import Utils


class MarketData(object):
    """ Загрузка и хранение рыночных данных и индикаторов
    """

    def __init__(self, **kwargs):
        Utils.set_self_attr(self, __class__, **kwargs)
        # в качсестве признаков используем логарифмические доходности, а не сами цены
        self.__feature_names = [k for k in kwargs.keys() if k not in ['instrument', 'timestamps'] + list('ohlc')]

    def __len__(self):
        return len(self.c)

    def get_close_price_from_log_ret(self, c_slice, log_ret):
        """ Переход от предсказанных логарифмических доходностей к ценам.

            Предсказанная доходность следующего периода умножается на реальную цену предыдущего периода, тем самым
            получаем предсказанную цену следующего периода.
        """
        return self.c[c_slice] * np.exp(log_ret).squeeze()

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def instrument(self):
        return self.__instrument

    @property
    def timestamps(self):
        return self.__timestamps

    @property
    def o(self):
        return self.__o

    @property
    def o_log_ret(self):
        return self.__o_log_ret

    @property
    def h(self):
        return self.__h

    @property
    def h_log_ret(self):
        return self.__h_log_ret

    @property
    def l(self):
        return self.__l

    @property
    def l_log_ret(self):
        return self.__l_log_ret

    @property
    def c_log_ret(self):
        return self.__c_log_ret

    @property
    def c(self):
        return self.__c

    @property
    def v(self):
        return self.__v

    @property
    def bband20_lower(self):
        return self.__bband20_lower

    @property
    def bband20_middle(self):
        return self.__bband20_middle

    @property
    def bband20_upper(self):
        return self.__bband20_upper

    @property
    def ema14(self):
        return self.__ema14

    @property
    def ema30(self):
        return self.__ema30

    @property
    def macd(self):
        return self.__macd

    @property
    def macd_signal(self):
        return self.__macd_signal

    @property
    def rsi14(self):
        return self.__rsi14

    @property
    def willr14(self):
        return self.__willr14

    @classmethod
    def create_(cls, instrument, df, timestamps, **kwargs):
        """ Обобщённая производящая функция.

            Ожидает на входе словарь, в котором ключи 'o', 'h', 'l', 'c', 'v' замаплены
            на соответствующие имена колонок датафрейма df.
        """
        # [скорректированная] цена закрытия - наш главный признак для вычисления индикаторов
        init_kwargs = {'instrument': instrument, 'timestamps': timestamps}
        init_kwargs.update({k: df[kwargs[k]].values for k in 'ohlcv'})
        open_price, high_price, low_price, close_price = [init_kwargs[k] for k in 'ohlc']

        for k in 'ohlc':
            # будем работать с логарифмической доходностью
            init_kwargs[f'{k}_log_ret'] = cls.log_returns(df[kwargs[k]]).values

        # добавим индикаторы
        # TODO: динамическая настройка списка индикаторов?
        #
        # EMA14
        # EMA30
        # MACD: быстрая, медленная и сигнальная линии со "стандартными" периодами
        # RSI с периодом 14
        # Bollinger Bands
        # Williams % R
        #
        init_kwargs.update(cls.indi_ema(close_price, 14))
        init_kwargs.update(cls.indi_ema(close_price, 30))
        for indi in (cls.indi_macd, cls.indi_rsi, cls.indi_bband):
            init_kwargs.update(indi(close_price))
        init_kwargs.update(cls.indi_willr(high_price, low_price, close_price))

        return cls(**init_kwargs)

    @classmethod
    def create_from_tiingo(cls, instrument, *args, **kwargs):
        """ Загрузка данных через pandas_datareader is https://www.tiingo.com

            Преимущество данного источника, в частности, в том, что он предоставляет
            все 4 скорректированные (adjusted) цены (а не только цену закрытия)
        """
        df = pdr.get_data_tiingo(instrument, api_key=os.environ.get('TIINGO_API_KEY'), *args, **kwargs)
        return cls.create_(instrument, df, df.index.get_level_values('date').values,
                           **{'o': 'adjOpen', 'h': 'adjHigh', 'l': 'adjLow', 'c': 'adjClose', 'v': 'adjVolume'})

    @classmethod
    def create_from_quandl(cls, instrument, *args, **kwargs):
        """ Загрузка данных через pandas_datareader is https://www.quandl.com
        """
        df = pdr.get_data_quandl(instrument, api_key=os.environ.get('QUANDL_API_KEY'), *args, **kwargs)
        return cls.create_(instrument, df, df.index.values,
                           **{'o': 'AdjOpen', 'h': 'AdjHigh', 'l': 'AdjLow', 'c': 'AdjClose', 'v': 'AdjVolume'})

    @classmethod
    def create_from_alphavantage_intraday(cls, instrument, *args, **kwargs):
        """ Загрузка данных через pandas_datareader is https://www.alphavantage.co
            У них есть внутридневные (минутные) данные, но только за неделю назад
        """
        df = pdr.get_data_alphavantage(instrument, api_key=os.environ.get('ALPHAVANTAGE_API_KEY'),
                                       function='TIME_SERIES_INTRADAY')
        return cls.create_(instrument, df, df.index.values,
                           **{'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})

    @staticmethod
    def log_returns(series):
        """ Вычисление логарифмических доходностей.
        """
        return series.rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)

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

# __EOF__
