"""
因子研究模块
支持多因子挖掘、检验、组合
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Callable
import warnings
warnings.filterwarnings('ignore')

# ==================== 因子定义 ====================

class Factor:
    """因子基类"""
    def __init__(self, name: str):
        self.name = name
        self.data = None
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
    
    def get_signal(self, factor_values: pd.Series, direction: str = "long") -> pd.Series:
        """
        根据因子值生成交易信号
        direction: "long" = 因子值越高越买, "short" = 因子值越低越买
        """
        if direction == "long":
            # 越高越买 (做多)
            return (factor_values > factor_values.quantile(0.8)).astype(int)
        else:
            # 越低越买 (做空)
            return (factor_values < factor_values.quantile(0.2)).astype(int)

# ----- 动量因子 -----
class MomentumFactor(Factor):
    """动量因子: 过去N天收益率"""
    def __init__(self, period: int = 20):
        super().__init__(f"Momentum_{period}")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change(self.period)

class ROCFactor(Factor):
    """变动率因子"""
    def __init__(self, period: int = 12):
        super().__init__(f"ROC_{period}")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return (df['Close'] - df['Close'].shift(self.period)) / df['Close'].shift(self.period)

# ----- 波动率因子 -----
class VolatilityFactor(Factor):
    """波动率因子: N天收益标准差"""
    def __init__(self, period: int = 20):
        super().__init__(f"Volatility_{period}")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        return df['Close'].pct_change().rolling(self.period).std() * np.sqrt(252)

class ATRFactor(Factor):
    """真实波动幅度均值"""
    def __init__(self, period: int = 14):
        super().__init__(f"ATR_{period}")
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

# ----- 流动性因子 -----
class TurnoverFactor(Factor):
    """换手率因子"""
    def __init__(self, period: int = 20):
        super().__init__(f"Turnover_{period}")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        if 'Volume' in df.columns:
            return df['Volume'].pct_change().rolling(self.period).mean()
        return pd.Series(0, index=df.index)

class VolumePriceFactor(Factor):
    """量价相关因子"""
    def __init__(self, period: int = 20):
        super().__init__(f"VolumePrice_{period}")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        returns = df['Close'].pct_change()
        volume = df['Volume']
        
        if volume is None or len(volume) == 0:
            return pd.Series(0, index=df.index)
        
        return returns.rolling(self.period).corr(volume)

# ----- 趋势因子 -----
class MAAlignmentFactor(Factor):
    """均线对齐因子: 价格与均线的偏离"""
    def __init__(self, ma_period: int = 20):
        super().__init__(f"MA_Alignment_{ma_period}")
        self.ma_period = ma_period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        ma = df['Close'].rolling(self.ma_period).mean()
        return (df['Close'] - ma) / ma

class MACDFactor(Factor):
    """MACD因子值"""
    def __init__(self):
        super().__init__("MACD")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        return ema12 - ema26

# ----- 价值因子 (需要财务数据) -----
class PEFactor(Factor):
    """市盈率因子 (简化版，用历史估算)"""
    def __init__(self):
        super().__init__("PE_Ratio")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        # 简化版: 用价格/历史平均收益估算
        avg_price = df['Close'].rolling(252).mean()
        return avg_price / df['Close']

# ----- RSI因子 -----
class RSIFactor(Factor):
    """RSI因子"""
    def __init__(self, period: int = 14):
        super().__init__(f"RSI_{period}")
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

# ==================== 因子研究工具 ====================

class FactorResearcher:
    """因子研究员"""
    
    def __init__(self, ticker: str, period: str = "2y"):
        self.ticker = ticker
        self.period = period
        self.df = None
        self.factors = {}
    
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        self.df = yf.download(self.ticker, period=self.period, progress=False)
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
        return self.df
    
    def add_factor(self, factor: Factor):
        """添加因子"""
        self.factors[factor.name] = factor.calculate(self.df)
    
    def get_all_factors(self) -> pd.DataFrame:
        """获取所有因子值"""
        if self.df is None:
            self.load_data()
        
        factor_values = pd.DataFrame(index=self.df.index)
        
        for name, values in self.factors.items():
            factor_values[name] = values
        
        return factor_values.dropna()
    
    def analyze_factor(self, factor_name: str, forward_days: int = 20) -> Dict:
        """
        分析单个因子
        
        Returns:
        - IC (Information Coefficient): 因子值与未来收益的相关性
        - IR (Information Ratio): IC均值/IC标准差
        - 回测收益
        """
        if factor_name not in self.factors:
            return {"error": "Factor not found"}
        
        factor_values = self.factors[factor_name]
        returns = self.df['Close'].pct_change(forward_days).shift(-forward_days)
        
        # 去除NaN
        valid_idx = factor_values.notna() & returns.notna()
        ic = factor_values[valid_idx].corr(returns[valid_idx])
        
        # 分组回测
        factor_quantiles = factor_values[valid_idx].quantile([0.2, 0.4, 0.6, 0.8])
        
        long_returns = returns[valid_idx][factor_values[valid_idx] > factor_quantiles[0.8]]
        short_returns = returns[valid_idx][factor_values[valid_idx] < factor_quantiles[0.2]]
        
        long_mean = long_returns.mean() * 252 / forward_days if len(long_returns) > 0 else 0
        short_mean = short_returns.mean() * 252 / forward_days if len(short_returns) > 0 else 0
        
        return {
            "factor": factor_name,
            "IC": ic,
            "IC_rank": abs(ic),
            "long_return": long_mean,
            "short_return": short_mean,
            "spread": long_mean - short_mean,
            "n_observations": valid_idx.sum()
        }
    
    def rank_factors(self, forward_days: int = 20) -> pd.DataFrame:
        """因子排名"""
        results = []
        
        for name in self.factors.keys():
            analysis = self.analyze_factor(name, forward_days)
            results.append(analysis)
        
        return pd.DataFrame(results).sort_values("IC_rank", ascending=False)
    
    def create_factor_portfolio(self, factor_names: List[str], weights: List[float] = None) -> pd.Series:
        """创建因子组合"""
        if weights is None:
            weights = [1.0 / len(factor_names)] * len(factor_names)
        
        if len(factor_names) != len(weights):
            raise ValueError("因子数量和权重数量不匹配")
        
        # 标准化因子
        combined = pd.Series(0, index=self.df.index)
        
        for name, weight in zip(factor_names, weights):
            if name in self.factors:
                factor = self.factors[name]
                # 标准化
                factor_norm = (factor - factor.mean()) / factor.std()
                combined += factor_norm * weight
        
        return combined

# ==================== 因子策略 ====================

class FactorStrategy:
    """因子策略"""
    
    def __init__(self, factors: List[Factor], direction: str = "long"):
        self.factors = factors
        self.direction = direction
    
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        # 计算所有因子
        factor_values = pd.DataFrame(index=df.index)
        
        for factor in self.factors:
            factor_values[factor.name] = factor.calculate(df)
        
        # 去除NaN
        factor_values = factor_values.dropna()
        
        if len(factor_values) == 0:
            return pd.Series(0, index=df.index)
        
        # 等权平均
        combined = factor_values.mean(axis=1)
        
        # 生成信号: 前20%买入，后20%卖出
        signal = pd.Series(0, index=df.index)
        
        if self.direction == "long":
            signal[combined > combined.quantile(0.8)] = 1
            signal[combined < combined.quantile(0.2)] = -1
        else:
            signal[combined < combined.quantile(0.2)] = 1
            signal[combined > combined.quantile(0.8)] = -1
        
        return signal

# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建研究员
    researcher = FactorResearcher("SPY", period="2y")
    researcher.load_data()
    
    # 添加因子
    researcher.add_factor(MomentumFactor(20))
    researcher.add_factor(MomentumFactor(60))
    researcher.add_factor(VolatilityFactor(20))
    researcher.add_factor(RSIFactor(14))
    researcher.add_factor(MAAlignmentFactor(20))
    researcher.add_factor(MACDFactor())
    researcher.add_factor(TurnoverFactor(20))
    
    # 因子排名
    print("因子排名 (按IC):")
    print(researcher.rank_factors(forward_days=20))
    
    # 创建多因子策略
    strategy = FactorStrategy([
        MomentumFactor(20),
        RSIFactor(14),
        MAAlignmentFactor(20)
    ], direction="long")
    
    print("\n多因子策略信号已生成!")
