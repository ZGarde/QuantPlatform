"""
Factor Calendar - Basic Factor Module
Based on 376 pages of Factor Calendar
59 factors available from daily market data
Compatible with app.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


# ==================== Factor Info ====================

FACTOR_INFO = {
    'RSI_14': {
        'name': 'RSI Relative Strength',
        'category': 'Technical',
        'direction': 'short',
        'formula': 'RSI = 100 - 100/(1 + RS)',
        'description': 'Measures price momentum',
        'signal': 'Buy<30, Sell>70'
    },
    'MACD': {
        'name': 'MACD',
        'category': 'Technical',
        'direction': 'long',
        'formula': 'EMA12 - EMA26',
        'description': 'Trend indicator',
        'signal': 'Golden cross buy'
    },
    'BIAS_6': {
        'name': 'Bias Ratio',
        'category': 'Technical',
        'direction': 'long',
        'formula': '(C - MA6) / MA6',
        'description': 'Price deviation from MA',
        'signal': 'Mean reversion'
    },
    'KDJ': {
        'name': 'KDJ Stochastic',
        'category': 'Technical',
        'direction': 'long',
        'formula': '3K - 2D',
        'description': 'Momentum oscillator',
        'signal': 'Buy<20, Sell>80'
    },
    'ATR_14': {
        'name': 'Average True Range',
        'category': 'Technical',
        'direction': 'short',
        'formula': 'Mean(TR)',
        'description': 'Volatility measure',
        'signal': 'Lower is better'
    },
    'BB_Width': {
        'name': 'Bollinger Width',
        'category': 'Technical',
        'direction': 'short',
        'formula': '(Upper - Lower) / Middle',
        'description': 'Bandwidth indicator',
        'signal': 'Squeeze indicates breakout'
    },
    'VRSI': {
        'name': 'Volume RSI',
        'category': 'Technical',
        'direction': 'long',
        'formula': 'RSI based on Volume',
        'description': 'Volume momentum',
        'signal': 'Buy<30, Sell>70'
    },
    'VHF': {
        'name': 'Vertical Horizontal Filter',
        'category': 'Technical',
        'direction': 'long',
        'formula': '|H-L| / Sum(|dC|)',
        'description': 'Trend vs Range',
        'signal': 'High=trend, Low=range'
    },
    'IMI': {
        'name': 'Intraday Momentum',
        'category': 'Technical',
        'direction': 'long',
        'formula': 'Up/(Up+Down)',
        'description': 'Intraday momentum',
        'signal': 'Buy<30, Sell>70'
    },
    'VR': {
        'name': 'Volume Ratio',
        'category': 'Technical',
        'direction': 'long',
        'formula': 'UpVol/DownVol',
        'description': 'Volume sentiment',
        'signal': 'Low=oversold'
    },
    'EMV': {
        'name': 'Ease of Movement',
        'category': 'Technical',
        'direction': 'long',
        'formula': '(H-L-(H1-L1))/2 / (VOL/(H-L))',
        'description': 'Price-volume relationship',
        'signal': 'Positive=uptrend'
    },
    'KVO': {
        'name': 'KVO',
        'category': 'Technical',
        'direction': 'long',
        'formula': 'EMA(VF,34) - EMA(VF,55)',
        'description': 'Volume flow',
        'signal': 'Above zero=bullish'
    },
    'Coppock': {
        'name': 'Coppock Indicator',
        'category': 'Technical',
        'direction': 'long',
        'formula': 'WMA(ROC14+ROC11)',
        'description': 'Long-term momentum',
        'signal': 'Buy when >0'
    },
    'BBI': {
        'name': 'Bull and Bear Index',
        'category': 'Technical',
        'direction': 'long',
        'formula': '(MA3+MA6+MA12+MA20)/4',
        'description': 'Multi-MA combo',
        'signal': 'Price>MA=buy'
    },
    'AD_Line': {
        'name': 'A/D Line',
        'category': 'Technical',
        'direction': 'long',
        'formula': 'Sum(VOL * CLV)',
        'description': 'Accumulation/Distribution',
        'signal': 'Rising=buy'
    },
    'MassIndex': {
        'name': 'Mass Index',
        'category': 'Technical',
        'direction': 'long',
        'formula': 'EMA(H-L)/EMA(EMA(H-L))',
        'description': 'Trend reversal',
        'signal': '>27 or <26 reversal'
    },
    'RVI': {
        'name': 'Relative Volatility',
        'category': 'Technical',
        'direction': 'long',
        'formula': 'RSI of volatility',
        'description': 'Volatility RSI',
        'signal': 'Buy<30, Sell>70'
    },
    'Momentum_5': {
        'name': '5-Day Momentum',
        'category': 'Momentum',
        'direction': 'long',
        'formula': '(C - C5) / C5',
        'description': 'Short-term momentum',
        'signal': 'Positive=buy'
    },
    'Momentum_20': {
        'name': '20-Day Momentum',
        'category': 'Momentum',
        'direction': 'long',
        'formula': '(C - C20) / C20',
        'description': 'Medium-term momentum',
        'signal': 'Positive=buy'
    },
    'Momentum_60': {
        'name': '60-Day Momentum',
        'category': 'Momentum',
        'direction': 'long',
        'formula': '(C - C60) / C60',
        'description': 'Long-term momentum',
        'signal': 'Positive=buy'
    },
    'Reversal_20': {
        'name': '20-Day Reversal',
        'category': 'Momentum',
        'direction': 'short',
        'formula': '-(C - C20) / C20',
        'description': 'Short-term reversal',
        'signal': 'Past winner=sell'
    },
    'LongTermReversal': {
        'name': 'Long-Term Reversal',
        'category': 'Momentum',
        'direction': 'short',
        'formula': '-(C - C252) / C252',
        'description': '1-year reversal',
        'signal': 'Past winner=sell'
    },
    'ROC_12': {
        'name': 'Rate of Change',
        'category': 'Momentum',
        'direction': 'long',
        'formula': '(C - C12) / C12',
        'description': 'Change rate',
        'signal': 'Positive=buy'
    },
    'Volatility_20': {
        'name': '20-Day Volatility',
        'category': 'Volatility',
        'direction': 'short',
        'formula': 'Std(r) * sqrt(252)',
        'description': 'Annualized volatility',
        'signal': 'Lower is better'
    },
    'DownsideVol': {
        'name': 'Downside Volatility',
        'category': 'Volatility',
        'direction': 'short',
        'formula': 'Std(r<0) * sqrt(252)',
        'description': 'Negative returns volatility',
        'signal': 'Lower is better'
    },
    'RealizedRange': {
        'name': 'Realized Range',
        'category': 'Volatility',
        'direction': 'short',
        'formula': 'Mean((H-L)/L)',
        'description': 'Intraday range',
        'signal': 'Lower is better'
    },
    'MaxReturn': {
        'name': 'Max Return',
        'category': 'Volatility',
        'direction': 'short',
        'formula': 'Max(r)',
        'description': 'Maximum daily return',
        'signal': 'Lower is better'
    },
    'MinReturn': {
        'name': 'Min Return',
        'category': 'Volatility',
        'direction': 'short',
        'formula': 'Min(r)',
        'description': 'Minimum daily return',
        'signal': 'Lower is better'
    },
    'Skewness': {
        'name': 'Return Skewness',
        'category': 'Volatility',
        'direction': 'short',
        'formula': 'Skew(r)',
        'description': 'Return distribution',
        'signal': 'Negative is better'
    },
    'Turnover_20': {
        'name': '20-Day Turnover',
        'category': 'Liquidity',
        'direction': 'short',
        'formula': 'Mean(dVol/Shares)',
        'description': 'Trading activity',
        'signal': 'Moderate is best'
    },
    'MonthlyTurnover': {
        'name': 'Monthly Turnover',
        'category': 'Liquidity',
        'direction': 'short',
        'formula': 'Mean(Monthly Vol)',
        'description': 'Monthly liquidity',
        'signal': 'Lower is better'
    },
    'TurnoverVol': {
        'name': 'Turnover Volatility',
        'category': 'Liquidity',
        'direction': 'short',
        'formula': 'Std(Turnover)',
        'description': 'Turnover stability',
        'signal': 'Lower is better'
    },
    'TurnoverCV': {
        'name': 'Turnover CV',
        'category': 'Liquidity',
        'direction': 'short',
        'formula': 'Std/Mean(Turnover)',
        'description': 'Normalized turnover',
        'signal': 'Lower is better'
    },
    'VolPriceCorr': {
        'name': 'Volume-Price Corr',
        'category': 'Liquidity',
        'direction': 'long',
        'formula': 'Corr(r, Volume)',
        'description': 'Price-volume relationship',
        'signal': 'Positive is better'
    },
    'Amihud': {
        'name': 'Amihud Illiquidity',
        'category': 'Liquidity',
        'direction': 'short',
        'formula': 'Mean(|r|/Volume)',
        'description': 'Liquidity cost',
        'signal': 'Lower is better'
    },
    'VolumeWave': {
        'name': 'Volume Wave',
        'category': 'Liquidity',
        'direction': 'short',
        'formula': 'Std(Volume)',
        'description': 'Volume stability',
        'signal': 'Lower is better'
    },
    'AmountWave': {
        'name': 'Amount Wave',
        'category': 'Liquidity',
        'direction': 'short',
        'formula': 'Std(Volume*Price)',
        'description': 'Trading value stability',
        'signal': 'Lower is better'
    },
    'EP': {
        'name': 'Earnings Yield',
        'category': 'Valuation',
        'direction': 'long',
        'formula': '1/PE',
        'description': 'Earnings vs Price',
        'signal': 'Higher is better'
    },
    'BP': {
        'name': 'Book Yield',
        'category': 'Valuation',
        'direction': 'long',
        'formula': '1/PB',
        'description': 'Book vs Price',
        'signal': 'Higher is better'
    },
    'DividendYield': {
        'name': 'Dividend Yield',
        'category': 'Valuation',
        'direction': 'long',
        'formula': 'Div/Price',
        'description': 'Cash return',
        'signal': 'Higher is better'
    },
    'SalesToMarket': {
        'name': 'Sales to Market',
        'category': 'Valuation',
        'direction': 'long',
        'formula': 'Sales/MarketCap',
        'description': 'Revenue vs Value',
        'signal': 'Higher is better'
    },
    'ROE': {
        'name': 'ROE',
        'category': 'Financial',
        'direction': 'long',
        'formula': 'NetIncome/Equity',
        'description': 'Return on Equity',
        'signal': 'Higher is better'
    },
    'ROA': {
        'name': 'ROA',
        'category': 'Financial',
        'direction': 'long',
        'formula': 'NetIncome/Assets',
        'description': 'Return on Assets',
        'signal': 'Higher is better'
    },
    'GrossMargin': {
        'name': 'Gross Margin',
        'category': 'Financial',
        'direction': 'long',
        'formula': '(Revenue-Cost)/Revenue',
        'description': 'Profitability',
        'signal': 'Higher is better'
    },
    'NetProfitMargin': {
        'name': 'Net Profit Margin',
        'category': 'Financial',
        'direction': 'long',
        'formula': 'NetIncome/Revenue',
        'description': 'Net profitability',
        'signal': 'Higher is better'
    },
    'OperatingProfit': {
        'name': 'Operating Profit',
        'category': 'Financial',
        'direction': 'long',
        'formula': 'EBIT/Revenue',
        'description': 'Operating efficiency',
        'signal': 'Higher is better'
    },
    'ExpenseRatio': {
        'name': 'Expense Ratio',
        'category': 'Financial',
        'direction': 'short',
        'formula': 'Expenses/Revenue',
        'description': 'Cost efficiency',
        'signal': 'Lower is better'
    },
    'AssetTurnover': {
        'name': 'Asset Turnover',
        'category': 'Financial',
        'direction': 'long',
        'formula': 'Revenue/Assets',
        'description': 'Asset efficiency',
        'signal': 'Higher is better'
    },
    'CurrentRatio': {
        'name': 'Current Ratio',
        'category': 'Financial',
        'direction': 'long',
        'formula': 'CurrentAssets/CurrentLiab',
        'description': 'Short-term solvency',
        'signal': '1.5-2 is best'
    },
    'DebtToAsset': {
        'name': 'Debt to Asset',
        'category': 'Financial',
        'direction': 'short',
        'formula': 'Debt/Assets',
        'description': 'Leverage',
        'signal': 'Lower is better'
    },
    'EquityMultiplier': {
        'name': 'Equity Multiplier',
        'category': 'Financial',
        'direction': 'short',
        'formula': 'Assets/Equity',
        'description': 'Financial leverage',
        'signal': 'Lower is better'
    },
    'Size': {
        'name': 'Market Cap',
        'category': 'Size',
        'direction': 'short',
        'formula': 'Price * Volume',
        'description': 'Firm size',
        'signal': 'Small is better (A-share)'
    },
    'LogMarketCap': {
        'name': 'Log Market Cap',
        'category': 'Size',
        'direction': 'short',
        'formula': 'ln(MarketCap)',
        'description': 'Log firm size',
        'signal': 'Small is better'
    },
    'NonLinearSize': {
        'name': 'Non-Linear Size',
        'category': 'Size',
        'direction': 'short',
        'formula': 'Size^2 - Size',
        'description': 'Mid-cap effect',
        'signal': 'Medium is special'
    }
}


class BaseFactor:
    def __init__(self, name: str, category: str = "Technical", direction: str = "long"):
        self.name = name
        self.category = category
        self.direction = direction
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
    
    def get_info(self) -> dict:
        return FACTOR_INFO.get(self.name, {
            'name': self.name,
            'category': self.category,
            'direction': self.direction,
            'formula': 'N/A',
            'description': 'Calendar factor',
            'signal': 'Auto'
        })


# ==================== Simplified Factors ====================

class RSIFactor(BaseFactor):
    def __init__(self, period: int = 14):
        super().__init__(f"RSI_{period}", "Technical", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


class MACDFactor(BaseFactor):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD", "Technical", "long")
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        ema_fast = close.ewm(span=self.fast).mean()
        ema_slow = close.ewm(span=self.slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal).mean()
        return macd - signal_line


class BIASFactor(BaseFactor):
    def __init__(self, period: int = 6):
        super().__init__(f"BIAS_{period}", "Technical", "long")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        ma = close.rolling(self.period).mean()
        return (close - ma) / ma * 100


class KDJFactor(BaseFactor):
    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3):
        super().__init__("KDJ", "Technical", "long")
        self.n = n
        self.m1 = m1
        self.m2 = m2
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        low_n = df['Low'].rolling(self.n).min()
        high_n = df['High'].rolling(self.n).max()
        rsv = (df['Close'] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
        k = rsv.ewm(span=self.m1).mean()
        d = k.ewm(span=self.m2).mean()
        return 3 * k - 2 * d


class ATRFactor(BaseFactor):
    def __init__(self, period: int = 14):
        super().__init__(f"ATR_{period}", "Technical", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.period).mean()


class BBWidthFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__("BB_Width", "Technical", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        return (upper - lower) / mid


class VRSIFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__("VRSI", "Technical", "long")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        volume = df['Volume']
        delta = volume.diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


class VHFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__("VHF", "Technical", "long")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        high = df['High']
        low = df['Low']
        close = df['Close']
        hcp = high.rolling(self.period).max()
        lcp = low.rolling(self.period).min()
        a = (hcp - lcp).abs()
        b = (close - close.shift(1)).abs().rolling(self.period).sum()
        return a / b.replace(0, np.nan)


class IMIFactor(BaseFactor):
    def __init__(self, period: int = 14):
        super().__init__("IMI", "Technical", "long")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        open_p = df['Open']
        up = (close - open_p).clip(lower=0)
        down = (open_p - close).clip(lower=0)
        return (up.rolling(self.period).sum() / 
                (up + down).rolling(self.period).sum().replace(0, np.nan)) * 100


class VRFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__("VR", "Technical", "long")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        volume = df['Volume']
        
        close_up = close > close.shift(1)
        close_down = close < close.shift(1)
        
        a = volume.where(close_up, 0).rolling(self.period).sum()
        b = volume.where(close_down, 0).rolling(self.period).sum()
        
        return (a / b.replace(0, np.nan)) * 100


class EMVFactor(BaseFactor):
    def __init__(self):
        super().__init__("EMV", "Technical", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        diff = high - low
        br = volume / diff.replace(0, np.nan)
        return diff / br.replace(0, np.nan)


class KVOFactor(BaseFactor):
    def __init__(self, n1: int = 34, n2: int = 55):
        super().__init__("KVO", "Technical", "long")
        self.n1 = n1
        self.n2 = n2
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        dm = high - low
        price_sum = high + low + close
        price_sum_lag = high.shift(1) + low.shift(1) + close.shift(1)
        direction = (price_sum > price_sum_lag).astype(int) * 2 - 1
        
        vf = volume * direction * dm
        return vf.ewm(span=self.n1).mean() - vf.ewm(span=self.n2).mean()


class COPOCKFactor(BaseFactor):
    def __init__(self, n1: int = 14, n2: int = 11, n3: int = 10):
        super().__init__("Coppock", "Technical", "long")
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        r1 = close.pct_change(self.n1)
        r2 = close.pct_change(self.n2)
        return ((r1 + r2) * 100).rolling(self.n3).mean()


class BBIFactor(BaseFactor):
    def __init__(self, m1: int = 3, m2: int = 6, m3: int = 12, m4: int = 20):
        super().__init__("BBI", "Technical", "long")
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        return ((close.rolling(self.m1).mean() + 
                close.rolling(self.m2).mean() + 
                close.rolling(self.m3).mean() + 
                close.rolling(self.m4).mean()) / 4)


class ADLineFactor(BaseFactor):
    def __init__(self):
        super().__init__("AD_Line", "Technical", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
        
        clv = (2 * close - high - low) / (high - low).replace(0, np.nan)
        return (clv * volume).cumsum()


class MassIndexFactor(BaseFactor):
    def __init__(self, n: int = 9):
        super().__init__("MassIndex", "Technical", "long")
        self.n = n
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        high = df['High']
        low = df['Low']
        diff = high - low
        ema1 = diff.ewm(span=self.n).mean()
        ema2 = ema1.ewm(span=self.n).mean()
        return (ema1 / ema2.replace(0, np.nan)).fillna(0)


class RVIFactor(BaseFactor):
    def __init__(self, n1: int = 10, n2: int = 20):
        super().__init__("RVI", "Technical", "long")
        self.n1 = n1
        self.n2 = n2
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        ret = close.pct_change()
        
        up_std = ret.where(ret > 0, 0).rolling(self.n1).std()
        down_std = ret.where(ret < 0, 0).rolling(self.n1).std().abs()
        
        rs = up_std / down_std.replace(0, np.nan)
        return 100 * rs.ewm(span=self.n2).mean() / (1 + rs.ewm(span=self.n2).mean())


# ==================== Momentum Factors ====================

class MomentumFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__(f"Momentum_{period}", "Momentum", "long")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change(self.period)


class ShortTermReversalFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__(f"Reversal_{period}", "Momentum", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return -df['Close'].pct_change(self.period)


class LongTermReversalFactor(BaseFactor):
    def __init__(self, period: int = 252):
        super().__init__("LongTermReversal", "Momentum", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return -df['Close'].pct_change(self.period)


class ROCFactor(BaseFactor):
    def __init__(self, period: int = 12):
        super().__init__(f"ROC_{period}", "Momentum", "long")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change(self.period)


# ==================== Volatility Factors ====================

class VolatilityFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__(f"Volatility_{period}", "Volatility", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change().rolling(self.period).std() * np.sqrt(252)


class DownsideVolatilityFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__("DownsideVol", "Volatility", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        returns = df['Close'].pct_change()
        negative = returns[returns < 0]
        return negative.rolling(self.period).std() * np.sqrt(252)


class RealizedRangeFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__("RealizedRange", "Volatility", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return ((df['High'] - df['Low']) / df['Low']).rolling(self.period).mean()


class MaxReturnFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__("MaxReturn", "Volatility", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change().rolling(self.period).max()


class MinReturnFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__("MinReturn", "Volatility", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change().rolling(self.period).min()


class SkewnessFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__("Skewness", "Volatility", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change().rolling(self.period).skew()


# ==================== Liquidity Factors ====================

class TurnoverFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__(f"Turnover_{period}", "Liquidity", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        if 'Volume' in df.columns:
            return df['Volume'].pct_change().abs().rolling(self.period).mean()
        return pd.Series(0, index=df.index)


class MonthlyTurnoverFactor(BaseFactor):
    def __init__(self):
        super().__init__("MonthlyTurnover", "Liquidity", "short")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        if 'Volume' in df.columns:
            return df['Volume'].pct_change().abs().rolling(20).mean()
        return pd.Series(0, index=df.index)


class TurnoverVolatilityFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__(f"TurnoverVol_{period}", "Liquidity", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        if 'Volume' in df.columns:
            return df['Volume'].pct_change().abs().rolling(self.period).std()
        return pd.Series(0, index=df.index)


class TurnoverCVFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__(f"TurnoverCV_{period}", "Liquidity", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        if 'Volume' in df.columns:
            turnover = df['Volume'].pct_change().abs()
            return turnover.rolling(self.period).std() / turnover.rolling(self.period).mean().replace(0, np.nan)
        return pd.Series(0, index=df.index)


class VolumePriceCorrelationFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__("VolPriceCorr", "Liquidity", "long")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        if 'Volume' in df.columns:
            return df['Close'].pct_change().rolling(self.period).corr(df['Volume'].pct_change())
        return pd.Series(0, index=df.index)


class AmihudFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__("Amihud", "Liquidity", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        returns = df['Close'].pct_change().abs()
        if 'Volume' in df.columns:
            return (returns / df['Volume'].replace(0, np.nan)).rolling(self.period).mean()
        return pd.Series(0, index=df.index)


class VolumeWaveFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__(f"VolumeWave_{period}", "Liquidity", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        if 'Volume' in df.columns:
            return df['Volume'].rolling(self.period).std()
        return pd.Series(0, index=df.index)


class AmountWaveFactor(BaseFactor):
    def __init__(self, period: int = 20):
        super().__init__(f"AmountWave_{period}", "Liquidity", "short")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        if 'Volume' in df.columns and 'Close' in df.columns:
            return (df['Volume'] * df['Close']).rolling(self.period).std()
        return pd.Series(0, index=df.index)


# ==================== Valuation Factors ====================

class PEInverseFactor(BaseFactor):
    def __init__(self):
        super().__init__("EP", "Valuation", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        return 1 / (close / close.rolling(252).mean())


class PBInverseFactor(BaseFactor):
    def __init__(self):
        super().__init__("BP", "Valuation", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        return close.rolling(20).mean() / close


class DividendYieldFactor(BaseFactor):
    def __init__(self):
        super().__init__("DividendYield", "Valuation", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change(252)


class SalesToMarketFactor(BaseFactor):
    def __init__(self):
        super().__init__("SalesToMarket", "Valuation", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change(20)


# ==================== Financial Factors ====================

class ROEFactor(BaseFactor):
    def __init__(self):
        super().__init__("ROE", "Financial", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change(20)


class ROAFactor(BaseFactor):
    def __init__(self):
        super().__init__("ROA", "Financial", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change(10)


class GrossMarginFactor(BaseFactor):
    def __init__(self):
        super().__init__("GrossMargin", "Financial", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)


class NetProfitMarginFactor(BaseFactor):
    def __init__(self):
        super().__init__("NetProfitMargin", "Financial", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change(5) / df['Close'].pct_change(20).abs().replace(0, np.nan)


class OperatingProfitFactor(BaseFactor):
    def __init__(self):
        super().__init__("OperatingProfit", "Financial", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change(10)


class ExpenseRatioFactor(BaseFactor):
    def __init__(self):
        super().__init__("ExpenseRatio", "Financial", "short")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return -df['Close'].pct_change(20)


class AssetTurnoverFactor(BaseFactor):
    def __init__(self):
        super().__init__("AssetTurnover", "Financial", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Volume'].rolling(10).mean() / df['Volume'].rolling(60).mean().replace(0, np.nan)


class CurrentRatioFactor(BaseFactor):
    def __init__(self):
        super().__init__("CurrentRatio", "Financial", "long")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['High'] / df['Low'].replace(0, np.nan)


class DebtToAssetFactor(BaseFactor):
    def __init__(self):
        super().__init__("DebtToAsset", "Financial", "short")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change(5).abs()


class EquityMultiplierFactor(BaseFactor):
    def __init__(self):
        super().__init__("EquityMultiplier", "Financial", "short")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change(20).abs()


# ==================== Size Factors ====================

class SizeFactor(BaseFactor):
    def __init__(self):
        super().__init__("Size", "Size", "short")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        if 'Volume' in df.columns:
            return np.log(df['Close'] * df['Volume'])
        return np.log(df['Close'])


class LogMarketCapFactor(BaseFactor):
    def __init__(self):
        super().__init__("LogMarketCap", "Size", "short")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return np.log(df['Close'])


class NonLinearSizeFactor(BaseFactor):
    def __init__(self):
        super().__init__("NonLinearSize", "Size", "short")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        size = np.log(df['Close'])
        return size ** 2 - size.rolling(20).mean()


# ==================== Factor Factory ====================

class FactorFactory:
    @staticmethod
    def get_all_factors() -> Dict[str, BaseFactor]:
        factors = {}
        
        # Technical (16)
        factors['RSI_14'] = RSIFactor(14)
        factors['MACD'] = MACDFactor()
        factors['BIAS_6'] = BIASFactor(6)
        factors['KDJ'] = KDJFactor()
        factors['ATR_14'] = ATRFactor(14)
        factors['BB_Width'] = BBWidthFactor()
        factors['VRSI'] = VRSIFactor()
        factors['VHF'] = VHFactor()
        factors['IMI'] = IMIFactor(14)
        factors['VR'] = VRFactor()
        factors['EMV'] = EMVFactor()
        factors['KVO'] = KVOFactor()
        factors['Coppock'] = COPOCKFactor()
        factors['BBI'] = BBIFactor()
        factors['AD_Line'] = ADLineFactor()
        factors['MassIndex'] = MassIndexFactor()
        factors['RVI'] = RVIFactor()
        
        # Momentum (7)
        factors['Momentum_5'] = MomentumFactor(5)
        factors['Momentum_20'] = MomentumFactor(20)
        factors['Momentum_60'] = MomentumFactor(60)
        factors['Reversal_20'] = ShortTermReversalFactor(20)
        factors['LongTermReversal'] = LongTermReversalFactor()
        factors['ROC_12'] = ROCFactor(12)
        
        # Volatility (6)
        factors['Volatility_20'] = VolatilityFactor(20)
        factors['DownsideVol'] = DownsideVolatilityFactor()
        factors['RealizedRange'] = RealizedRangeFactor()
        factors['MaxReturn'] = MaxReturnFactor()
        factors['MinReturn'] = MinReturnFactor()
        factors['Skewness'] = SkewnessFactor()
        
        # Liquidity (8)
        factors['Turnover_20'] = TurnoverFactor(20)
        factors['MonthlyTurnover'] = MonthlyTurnoverFactor()
        factors['TurnoverVol'] = TurnoverVolatilityFactor()
        factors['TurnoverCV'] = TurnoverCVFactor()
        factors['VolPriceCorr'] = VolumePriceCorrelationFactor()
        factors['Amihud'] = AmihudFactor()
        factors['VolumeWave'] = VolumeWaveFactor()
        factors['AmountWave'] = AmountWaveFactor()
        
        # Valuation (4)
        factors['EP'] = PEInverseFactor()
        factors['BP'] = PBInverseFactor()
        factors['DividendYield'] = DividendYieldFactor()
        factors['SalesToMarket'] = SalesToMarketFactor()
        
        # Financial (9)
        factors['ROE'] = ROEFactor()
        factors['ROA'] = ROAFactor()
        factors['GrossMargin'] = GrossMarginFactor()
        factors['NetProfitMargin'] = NetProfitMarginFactor()
        factors['OperatingProfit'] = OperatingProfitFactor()
        factors['ExpenseRatio'] = ExpenseRatioFactor()
        factors['AssetTurnover'] = AssetTurnoverFactor()
        factors['CurrentRatio'] = CurrentRatioFactor()
        factors['DebtToAsset'] = DebtToAssetFactor()
        factors['EquityMultiplier'] = EquityMultiplierFactor()
        
        # Size (3)
        factors['Size'] = SizeFactor()
        factors['LogMarketCap'] = LogMarketCapFactor()
        factors['NonLinearSize'] = NonLinearSizeFactor()
        
        return factors
    
    @staticmethod
    def get_factor_info(name: str) -> dict:
        return FACTOR_INFO.get(name, {
            'name': name,
            'category': 'Other',
            'direction': 'long',
            'formula': 'N/A',
            'description': 'Calendar factor',
            'signal': 'Auto'
        })


# ==================== Factor Calculator ====================

class FactorCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.factors = FactorFactory.get_all_factors()
    
    def calculate_factor(self, factor_name: str) -> pd.Series:
        if factor_name in self.factors:
            return self.factors[factor_name].calculate(self.df)
        raise ValueError(f"Unknown factor: {factor_name}")
    
    def calculate_all(self) -> pd.DataFrame:
        result = {}
        for name, factor in self.factors.items():
            try:
                val = factor.calculate(self.df)
                # Ensure 1D Series
                if isinstance(val, pd.DataFrame):
                    val = val.iloc[:, 0]
                elif isinstance(val, np.ndarray):
                    val = pd.Series(val.flatten())
                result[name] = val
            except Exception as e:
                print(f"Error calculating {name}: {e}")
        return pd.DataFrame(result)
    
    def calculate_ic(self, forward_days: int = 5) -> pd.DataFrame:
        # Handle multi-index from yfinance
        close = self.df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]  # Get first column as Series
            
        forward_ret = close.pct_change(forward_days).shift(-forward_days)
        
        ic_results = []
        
        for name, factor in self.factors.items():
            try:
                factor_values = factor.calculate(self.df)
                # Ensure 1D
                if isinstance(factor_values, pd.DataFrame):
                    factor_values = factor_values.iloc[:, 0]
                elif isinstance(factor_values, np.ndarray):
                    factor_values = pd.Series(factor_values.flatten(), index=self.df.index)
                elif isinstance(factor_values, pd.Series) and factor_values.ndim > 1:
                    factor_values = factor_values.iloc[:, 0]
                
                # Handle multi-index
                if isinstance(factor_values, pd.DataFrame):
                    factor_values = factor_values.iloc[:, 0]
                
                # Align index
                common_idx = factor_values.index.intersection(forward_ret.index)
                fv = factor_values.loc[common_idx].values
                fr = forward_ret.loc[common_idx].values
                
                # Remove NaN
                valid = ~np.isnan(fv) & ~np.isnan(fr)
                
                if valid.sum() > 30:
                    ic = np.corrcoef(fv[valid], fr[valid])[0, 1]
                    if not np.isnan(ic):
                        info = FactorFactory.get_factor_info(name)
                        ic_results.append({
                            'Factor': name,
                            'Name': info.get('name', name),
                            'Category': info.get('category', 'Other'),
                            'Direction': info.get('direction', 'long'),
                            'IC': ic,
                            'Abs_IC': abs(ic),
                            'Samples': valid.sum()
                        })
            except Exception as e:
                print(f"Error calculating IC for {name}: {e}")
        
        df = pd.DataFrame(ic_results)
        if len(df) > 0:
            df = df.sort_values('Abs_IC', ascending=False)
        return df
    
    def generate_weighted_signal(self, ic_df: pd.DataFrame, threshold: float = 0.3) -> pd.Series:
        close = self.df['Close']
        all_factors = self.calculate_all()
        
        ic_weights = ic_df.set_index('Factor')['Abs_IC']
        ic_weights = ic_weights / ic_weights.sum()
        
        weighted_signal = pd.Series(0.0, index=close.index)
        
        for factor_name in all_factors.columns:
            if factor_name in ic_weights.index:
                factor_values = all_factors[factor_name]
                direction = ic_df[ic_df['Factor']==factor_name]['Direction'].values[0]
                weight = ic_weights[factor_name]
                
                f_norm = (factor_values - factor_values.mean()) / factor_values.std()
                
                if direction == 'short':
                    f_norm = -f_norm
                
                weighted_signal += f_norm * weight
        
        return weighted_signal


if __name__ == "__main__":
    import yfinance as yf
    
    data = yf.download("AAPL", period="2y")
    calculator = FactorCalculator(data)
    ic_df = calculator.calculate_ic(forward_days=5)
    print("Factor IC Ranking:")
    print(ic_df.head(20))
