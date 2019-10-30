# mdp/MarketData.py

import os
import numpy as np

import talib
import pandas_datareader as pdr

from . import Utils


class MarketData(object):
    """ Загрузка и хранение рыночных данных и индикаторов
    """

    def __init__(self, **kwargs):
        Utils.set_self_attr(self, **kwargs)
        self.src_feature_accessors = [f'get_{k}' for k in kwargs.keys() if k not in ('instrument', 'timestamps', 'c')]

    def __len__(self):
        return len(self.c_)

    @classmethod
    def create_(cls, instrument, df, timestamps, **kwargs):
        """ Обобщённая производящая функция.

            Ожидает на входе словарь, в котором ключи 'o', 'h', 'l', 'c', 'v' замаплены
            на соответствующие имена колонок датафрейма df.
        """
        # [скорректированная] цена закрытия - наш главный признак для вычисления индикаторов
        open_price, high_price, low_price, close_price = [df[kwargs[k]].values for k in 'ohlc']

        init_kwargs = {'instrument': instrument, 'timestamps': timestamps, 'c': close_price}

        for k, v in kwargs.items():
            if k in ['o', 'h', 'l', 'c']:
                # будем работать с логарифмической доходностью
                init_kwargs[f'{k}_logret'] = cls.log_returns(df[v]).values
            else:
                init_kwargs[k] = df[v].values

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
        return zip([f'macd_{s}' for s in [f'fast{fast}', f'slow{slow}', f'signal{signal}']],
                   talib.MACD(p, fast, slow, signal))

    @staticmethod
    def indi_rsi(p, period=14):
        """ Relative Strength Index
        """
        return [(f'rsi{period}', talib.RSI(p, period))]

    @staticmethod
    def indi_bband(p, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
        """ Bollinger Bands
        """
        return zip([f'bband{timeperiod}_{s}' for s in ['upper', 'middle', 'lowerband']],
                   talib.BBANDS(p, timeperiod, nbdevup, nbdevdn, matype))

    @staticmethod
    def indi_willr(high, low, close, timeperiod=14):
        """ Williams % R
        """
        return [(f'willr{timeperiod}', talib.WILLR(high, low, close, timeperiod))]

# __EOF__
