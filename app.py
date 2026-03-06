"""
量化交易平台 - 完整版
支持多因子策略、技术指标、因子研究
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==================== 专业因子模块 ====================
"""
因子研究模块
支持多因子挖掘、检验、组合
"""

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



st.set_page_config(page_title="Quant Platform", page_icon="📈", layout="wide")

# ============ 侧边栏 ============
with st.sidebar:
    st.title("⚙️ 参数设置")
    
    # 标的
    st.subheader("📊 标的")
    ticker = st.text_input("股票代码", value="SPY")
    period = st.selectbox("周期", ["6mo", "1y", "2y", "3y"], index=2)
    
    # 资金
    st.subheader("💰 资金")
    initial_cash = st.number_input("Initial:     $", value=100000, step=10000)
    initial_cash = float(initial_cash)
    
    # 成本
    st.subheader("📉 成本")
    fee_rate = st.slider("手续费(%)", 0.0, 1.0, 0.1) / 100
    slippage = st.slider("滑点(%)", 0.0, 0.5, 0.05) / 100
    
    # 风控
    st.subheader("🛡️ 风控")
    max_position = st.slider("最大仓位(%)", 10.0, 99.99, 80.0) / 100
    stop_loss = st.slider("止损(%)", 0.0, 20.0, 5.0) / 100
    take_profit = st.slider("止盈(%)", 0.0, 50.0, 20.0) / 100
    st.markdown("---")
    st.markdown("""
    ### 参数说明
    
    | 参数 | 作用 |
    |------|------|
    | 初始资金 | 回测的起始资金 |
    | 手续费 | 每次买卖收的手续费 |
    | 滑点 | 实际成交价与预期的偏差 |
    | 最大仓位 | 最多用多少%资金买入 (如100%=全仓) |
    | 止损 | 亏损到%时强制卖出 |
    | 止盈 | 盈利到%时强制卖出 |
    """)


# ============ 主界面 ============
st.title("📈 量化交易平台")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 策略回测", "🔬 因子研究", "📚 因子说明", "🤖 AI因子挖掘"])

# ============ TAB1: 策略回测 ============
with tab1:
    # 因子定义
    class Factor:
        def __init__(self, name): self.name = name
        def calculate(self, df): return pd.Series(0, index=df.index)

    class MomentumFactor(Factor):
        def __init__(self, period=20):
            super().__init__(f"Momentum_{period}")
            self.period = period
        def calculate(self, df):
            return df['Close'].pct_change(self.period)

    class VolatilityFactor(Factor):
        def __init__(self, period=20):
            super().__init__(f"Volatility_{period}")
            self.period = period
        def calculate(self, df):
            return df['Close'].pct_change().rolling(self.period).std() * np.sqrt(252)

    class RSIFactor(Factor):
        def __init__(self, period=14):
            super().__init__(f"RSI_{period}")
            self.period = period
        def calculate(self, df):
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(self.period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
            return 100 - (100 / (1 + gain / loss))

    class MAAlignFactor(Factor):
        def __init__(self, period=20):
            super().__init__(f"MA_Align_{period}")
            self.period = period
        def calculate(self, df):
            ma = df['Close'].rolling(self.period).mean()
            return (df['Close'] - ma) / ma

    # 策略
    class Strategy:
        def __init__(self, name): self.name = name
        def generate_signal(self, df, i): return 0

    class BuyHold(Strategy):
        def __init__(self): super().__init__("Buy & Hold")
        def generate_signal(self, df, i): return 1 if i == 60 else 0

    class MACD(Strategy):
        def __init__(self):
            super().__init__("MACD金叉")
        def generate_signal(self, df, i):
            if i < 2: return 0
            if pd.isna(df['MACD'].iloc[i]): return 0
            if df['MACD'].iloc[i-1] < df['MACD_Signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
                return 1
            elif df['MACD'].iloc[i-1] > df['MACD_Signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i]:
                return -1
            return 0

    class RSI(Strategy):
        def __init__(self, o=30, b=70):
            super().__init__(f"RSI({o}/{b})")
            self.o, self.b = o, b
        def generate_signal(self, df, i):
            if pd.isna(df['RSI'].iloc[i]): return 0
            rsi = df['RSI'].iloc[i]
            if rsi < self.o: return 1
            elif rsi > self.b: return -1
            return 0

    class MA(Strategy):
        def __init__(self, f=5, s=20):
            super().__init__(f"MA{f}/{s}")
            self.f, self.s = f, s
        def generate_signal(self, df, i):
            if i < self.s: return 0
            f = df[f'MA{self.f}'].iloc[i]; s = df[f'MA{self.s}'].iloc[i]
            fp = df[f'MA{self.f}'].iloc[i-1]; sp = df[f'MA{self.s}'].iloc[i-1]
            if pd.isna(f) or pd.isna(s): return 0
            if fp < sp and f > s: return 1
            elif fp > sp and f < s: return -1
            return 0

    class Bollinger(Strategy):
        def __init__(self):
            super().__init__("布林带")
        def generate_signal(self, df, i):
            if i < 20 or pd.isna(df['BB_lower'].iloc[i]): return 0
            c = df['Close'].iloc[i]
            if c < df['BB_lower'].iloc[i]: return 1
            elif c > df['BB_upper'].iloc[i]: return -1
            return 0

    class Momentum(Strategy):
        def __init__(self):
            super().__init__("动量因子")
        def generate_signal(self, df, i):
            if i < 30: return 0
            mom = df['Momentum_20'].iloc[i]
            if pd.isna(mom): return 0
            if mom > 0.03: return 1
            elif mom < -0.03: return -1
            return 0

    class DualFactor(Strategy):
        def __init__(self):
            super().__init__("RSI+MACD")
        def generate_signal(self, df, i):
            if i < 30: return 0
            rsi = df['RSI'].iloc[i]
            macd = df['MACD'].iloc[i]
            ms = df['MACD_Signal'].iloc[i]
            if pd.isna(rsi) or pd.isna(macd): return 0
            if i >= 2:
                mp = df['MACD'].iloc[i-1]; msp = df['MACD_Signal'].iloc[i-1]
                if rsi < 35 and pd.notna(mp) and mp < msp and macd > ms: return 1
                elif rsi > 65 and pd.notna(mp) and mp > msp and macd < ms: return -1
            return 0

    class Trend(Strategy):
        def __init__(self):
            super().__init__("趋势跟踪")
        def generate_signal(self, df, i):
            if i < 60: return 0
            ma60 = df['MA60'].iloc[i]; p = df['Close'].iloc[i]
            if pd.isna(ma60): return 0
            if df['MA60'].iloc[i-5] < ma60 and p > df['MA20'].iloc[i]: return 1
            elif df['MA60'].iloc[i-5] > ma60 or p < df['MA20'].iloc[i] * 0.95: return -1
            return 0

    class MultiFactor(Strategy):
        """多因子组合策略"""
        def __init__(self):
            super().__init__("多因子组合")
        def generate_signal(self, df, i):
            if i < 60: return 0
            
            # 综合多个因子信号
            signals = 0
            count = 0
            
            # 动量因子
            if 'Momentum_20' in df.columns:
                mom = df['Momentum_20'].iloc[i]
                if pd.notna(mom):
                    if mom > 0.03: signals += 1
                    elif mom < -0.03: signals -= 1
                    count += 1
            
            # RSI因子
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[i]
                if pd.notna(rsi):
                    if rsi < 35: signals += 1
                    elif rsi > 65: signals -= 1
                    count += 1
            
            # MACD因子
            if 'MACD' in df.columns:
                macd = df['MACD'].iloc[i]
                ms = df['MACD_Signal'].iloc[i]
                if pd.notna(macd) and pd.notna(ms) and i >= 2:
                    mp = df['MACD'].iloc[i-1]
                    msp = df['MACD_Signal'].iloc[i-1]
                    if pd.notna(mp) and pd.notna(msp):
                        if mp < msp and macd > ms: signals += 1
                        elif mp > msp and macd < ms: signals -= 1
                    count += 1
            
            # 均线对齐
            if 'MA20' in df.columns:
                ma20 = df['MA20'].iloc[i]
                price = df['Close'].iloc[i]
                if pd.notna(ma20):
                    dev = (price - ma20) / ma20
                    if dev > 0.05: signals -= 1
                    elif dev < -0.05: signals += 1
                    count += 1
            
            if count == 0:
                return 0
            
            avg_signal = signals / count
            
            if avg_signal > 0.3: return 1
            elif avg_signal < -0.3: return -1
            return 0

    # 多因子策略类
    class Momentum20(Strategy):
        def __init__(self):
            super().__init__("动量因子(20日)")
        def generate_signal(self, df, i):
            if i < 30: return 0
            if 'Momentum_20' not in df.columns: return 0
            mom = df['Momentum_20'].iloc[i]
            if pd.isna(mom): return 0
            if mom > 0.03: return 1
            elif mom < -0.03: return -1
            return 0

    class Momentum60(Strategy):
        def __init__(self):
            super().__init__("动量因子(60日)")
        def generate_signal(self, df, i):
            if i < 70: return 0
            mom = df['Close'].pct_change(60).iloc[i]
            if pd.isna(mom): return 0
            if mom > 0.10: return 1
            elif mom < -0.10: return -1
            return 0

    class RSIFactor(Strategy):
        def __init__(self):
            super().__init__("RSI因子")
        def generate_signal(self, df, i):
            if i < 30: return 0
            if 'RSI' not in df.columns: return 0
            rsi = df['RSI'].iloc[i]
            if pd.isna(rsi): return 0
            if rsi < 30: return 1
            elif rsi > 70: return -1
            return 0

    class MAAlignFactor(Strategy):
        def __init__(self):
            super().__init__("均线对齐")
        def generate_signal(self, df, i):
            if i < 30: return 0
            if 'MA20' not in df.columns: return 0
            ma20 = df['MA20'].iloc[i]
            price = df['Close'].iloc[i]
            if pd.isna(ma20): return 0
            dev = (price - ma20) / ma20
            if dev < -0.05: return 1
            elif dev > 0.05: return -1
            return 0

    class VolatilityFactor(Strategy):
        def __init__(self):
            super().__init__("Volatility因子")
        def generate_signal(self, df, i):
            if i < 30: return 0
            returns = df['Close'].pct_change().iloc[max(0,i-20):i]
            if len(returns) < 10: return 0
            vol = returns.std() * np.sqrt(252)
            if pd.isna(vol): return 0
            # 低波动买入 (反转效应)
            if vol < 0.15: return 1
            elif vol > 0.30: return -1
            return 0

    # 策略选择
    st.subheader("🎯 选择策略")
    
    strategies = {
        "技术指标": {
            "Buy & Hold": BuyHold,
            "MACD金叉": MACD,
            "RSI超卖": lambda: RSI(30, 70),
            "均线交叉(5/20)": lambda: MA(5, 20),
            "布林带突破": Bollinger,
        },
        "多因子": {
            "动量因子(20日)": Momentum20,
            "动量因子(60日)": Momentum60,
            "RSI因子": RSIFactor,
            "均线对齐": MAAlignFactor,
            "Volatility": VolatilityFactor,
        },
        "复合策略": {
            "RSI+MACD双因子": DualFactor,
            "趋势跟踪": Trend,
            "多因子组合": MultiFactor,
        }
    }
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        strategy_type = st.selectbox("策略类型", list(strategies.keys()))
        options = list(strategies[strategy_type].keys())
        selected = st.selectbox("选择策略", options)
        strategy_class = strategies[strategy_type][selected]
        strategy = strategy_class()
    
    # 解释
    explanations = {
        "Buy & Hold": "买入后长期持有。适合牛市。",
        "MACD金叉": "MACD金叉买入，死叉卖出。",
        "RSI超卖": "RSI<30超卖买入，RSI>70超买卖出。",
        "均线交叉(5/20)": "MA5上穿MA20买入，下穿卖出。",
        "布林带突破": "价格突破下轨买入，突破上轨卖出。",
        "动量因子(20日)": "20日涨幅>5%买入，<-5%卖出。",
        "动量因子(60日)": "60日涨幅>10%买入，<-10%卖出。",
        "RSI因子": "RSI<30买入，RSI>70卖出。",
        "均线对齐": "偏离MA20>5%卖出，<-5%买入。",
        "Volatility": "波动率<15%买入，>30%卖出。",
        "RSI+MACD双因子": "RSI超卖+MACD金叉买入。",
        "趋势跟踪": "MA60向上+价格站上MA20买入。",
        "多因子组合": "综合多个因子信号决策。"
    }
    
    with c2:
        st.info(f"**{selected}**: {explanations[selected]}")
    
    # 因子计算说明
    with st.expander("📖 因子/策略计算说明"):
        factor_info = {
            "MACD金叉": """
            **计算方法:**
            - MACD = EMA12 - EMA26
            - Signal = MACD的9日EMA
            - 买入: MACD上穿Signal(金叉)
            - 卖出: MACD下穿Signal(死叉)
            """,
            "RSI超卖": """
            **计算方法:**
            - RSI = 100 - 100/(1+RS)
            - RS = 平均涨幅 / 平均跌幅 (14日)
            - 买入: RSI < 30 (超卖)
            - 卖出: RSI > 70 (超买)
            """,
            "均线交叉(5/20)": """
            **计算方法:**
            - MA5 = 5日简单移动平均
            - MA20 = 20日简单移动平均
            - 买入: MA5上穿MA20
            - 卖出: MA5下穿MA20
            """,
            "布林带突破": """
            **计算方法:**
            - BB中轨 = 20日MA
            - BB标准差 = 20日收益标准差
            - BB上轨 = 中轨 + 2*标准差
            - BB下轨 = 中轨 - 2*标准差
            - 买入: 价格突破下轨
            - 卖出: 价格突破上轨
            """,
            "动量因子(20日)": """
            **计算方法:**
            - Momentum = (收盘价 - 20日前收盘价) / 20日前收盘价
            - 买入: Momentum > 5%
            - 卖出: Momentum < -5%
            """,
            "动量因子(60日)": """
            **计算方法:**
            - Momentum = (收盘价 - 60日前收盘价) / 60日前收盘价
            - 买入: Momentum > 10%
            - 卖出: Momentum < -10%
            """,
            "RSI因子": """
            **计算方法:**
            - 与RSI超卖相同
            - 买入: RSI < 30
            - 卖出: RSI > 70
            """,
            "均线对齐": """
            **计算方法:**
            - 偏离度 = (收盘价 - MA20) / MA20
            - 买入: 偏离度 < -5%
            - 卖出: 偏离度 > +5%
            """,
            "Volatility因子": """
            **计算方法:**
            - Vol = 20日收益标准差 * sqrt(252)
            - 买入: Vol < 15% (低波动)
            - 卖出: Vol > 30% (高波动)
            """,
            "RSI+MACD双因子": """
            **计算方法:**
            - 同时满足两个条件:
              - RSI < 35 且 MACD金叉 → 买入
              - RSI > 65 且 MACD死叉 → 卖出
            """,
            "趋势跟踪": """
            **计算方法:**
            - MA60 = 60日移动平均
            - 买入: MA60向上 且 价格 > MA20
            - 卖出: MA60向下 或 价格 < MA20*0.95
            """,
            "多因子组合": """
            **计算方法:**
            - 综合4个因子信号:
              1. 动量因子(20日)
              2. RSI因子
              3. MACD
              4. 均线对齐
            - 每个因子: +1/-1/0
            - 平均信号 > 0.3 → 买入
            - 平均信号 < -0.3 → 卖出
            """
        }
        
        if selected in factor_info:
            st.markdown(factor_info[selected])
        else:
            st.markdown("**Buy & Hold**: 买入后一直持有到回测结束")
    
    # 数据获取
    @st.cache_data
    def get_data(ticker, period):
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    
    def calc_indicators(df):
        close = df['Close']
        for w in [5, 10, 20, 60]:
            df[f'MA{w}'] = close.rolling(w).mean()
        df['MACD'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))
        df['BB_mid'] = close.rolling(20).mean()
        df['BB_std'] = close.rolling(20).std()
        df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
        df['Momentum_20'] = close.pct_change(20)
        return df
    
    # 运行
    if st.button("🚀 运行回测", type="primary"):
        with st.spinner("下载数据..."):
            df = get_data(ticker, period)
        
        if df is None or len(df) == 0:
            st.error("无法获取数据")
        else:
            with st.spinner("计算..."):
                df = calc_indicators(df)
            
            # 回测
            cash = initial_cash
            position = 0
            portfolio = [initial_cash]
            trades = []
            start = 60
            
            for i in range(start, len(df)):
                price = df['Close'].iloc[i]
                
                if position > 0:
                    entry = trades[-1][2] if trades and trades[-1][1] == 'BUY' else price
                    pnl = (price - entry) / entry
                    if pnl <= -stop_loss or pnl >= take_profit:
                        cash += position * price * (1 - fee_rate - slippage)
                        trades.append((df.index[i], 'SELL', price, position, 'SL/TP'))
                        position = 0
                
                if position == 0:
                    sig = strategy.generate_signal(df, i)
                    if sig == 1:
                        shares = int(cash * max_position / price) + 1 + 1
                        cost = shares * price * (1 + fee_rate + slippage)
                        if cost <= cash and shares > 0:
                            cash -= cost
                            position = shares
                            trades.append((df.index[i], 'BUY', price, shares, '策略信号触发'))
                
                portfolio.append(cash + position * price)
            
            if position > 0:
                cash += position * df['Close'].iloc[-1] * (1 - fee_rate - slippage)
            
            final = cash
            ret = (final - initial_cash) / initial_cash
            
            # 指标
            rets = np.diff(portfolio) / portfolio[:-1]
            n_years = len(portfolio) / 252
            annual = (1 + ret) ** (1/n_years) - 1 if n_years > 0 else 0
            sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
            
            peak = np.maximum.accumulate(portfolio)
            max_dd = np.max((peak - portfolio) / peak) if len(portfolio) > 0 else 0
            
            # 结果
            st.markdown("---")
            st.subheader(f"📊 {ticker} 回测结果")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Return", f"{ret*100:+.1f}%")
            c2.metric("Annual", f"{annual*100:+.1f}%")
            c3.metric("Sharpe", f"{sharpe:.2f}")
            c4.metric("Max DD", f"{max_dd*100:.1f}%")
            
            # 图
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(portfolio, label='策略')
            bench = df['Close'].iloc[start:].values / df['Close'].iloc[start] * initial_cash
            ax.plot(bench, alpha=0.5, label='Buy & Hold')
            ax.set_title('Equity Curve')
            ax.legend(['Strategy', 'Buy & Hold'], loc='upper left')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # 交易记录
            # Remove this - will be added in proper place
            # Explain each trade
            st.markdown("### 每笔交易原因")
            
            # Get strategy explanation
            strategy_explanations = {
                "MACD金叉": "MACD线从下方穿过Signal线，形成金叉，预示上涨趋势",
                "RSI超卖": "RSI低于30，市场超卖可能反弹",
                "均线交叉(5/20)": "短期均线向上穿过长期均线，形成多头排列",
                "布林带突破": "价格突破布林带下轨，可能反弹",
                "动量因子(20日)": "过去20天涨幅超过5%，动量强劲",
                "动量因子(60日)": "过去60天涨幅超过10%，强动量",
                "RSI因子": "RSI指标显示超卖/超买状态",
                "均线对齐": "价格偏离均线过大，可能回归",
                "波动率因子": "波动率处于历史低位/高位",
                "RSI+MACD双因子": "同时满足RSI超卖和MACD金叉两个条件",
                "趋势跟踪": "长期均线向上且价格站上中期均线",
                "多因子组合": "多个因子信号综合判断"
            }
            
            for i, trade in enumerate(trades):
                date = trade[0]
                action = trade[1]
                price = trade[2]
                qty = trade[3] if len(trade) > 3 else 0
                reason = trade[4] if len(trade) > 4 else "信号触发"
                
                if action == "BUY":
                    st.markdown(f"**{i+1}. 买入** @ ${price:.2f}")
                    st.markdown(f"   - 日期: {date.date()}")
                    st.markdown(f"   - 数量: {qty}股")
                    st.markdown(f"   - 原因: {reason}")
                else:
                    st.markdown(f"**{i+1}. 卖出** @ ${price:.2f}")
                    st.markdown(f"   - 日期: {date.date()}")
                    st.markdown(f"   - 数量: {qty}股")
                    st.markdown(f"   - 原因: {reason}")
                st.markdown("---")
            
                    # 交易记录
            with st.expander("📋 交易记录"):
                if trades:
                    td = pd.DataFrame(trades, columns=['日期', '操作', '价格', '数量', '原因'])
                    st.dataframe(td, use_container_width=True)

# ============ TAB2: 因子研究 ============
with tab2:
    st.subheader("🔬 因子研究")
    
    # 因子类
    class Factor:
        def __init__(self, name): self.name = name
        def calculate(self, df): return pd.Series(0, index=df.index)
    
    class Momentum20(Factor):
        def __init__(self): super().__init__("Momentum_20")
        def calculate(self, df): return df['Close'].pct_change(20)
    
    class Momentum60(Factor):
        def __init__(self): super().__init__("Momentum_60")
        def calculate(self, df): return df['Close'].pct_change(60)
    
    class Volatility(Factor):
        def __init__(self): super().__init__("Volatility_20")
        def calculate(self, df): return df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    class RSI(Factor):
        def __init__(self): super().__init__("RSI_14")
        def calculate(self, df):
            d = df['Close'].diff()
            g = d.where(d > 0, 0).rolling(14).mean()
            l = (-d.where(d < 0, 0)).rolling(14).mean()
            return 100 - (100 / (1 + g / l))
    
    class MAAlign(Factor):
        def __init__(self): super().__init__("MA_Align_20")
        def calculate(self, df):
            ma = df['Close'].rolling(20).mean()
            return (df['Close'] - ma) / ma
    
    class MACD(Factor):
        def __init__(self): super().__init__("MACD")
        def calculate(self, df):
            return df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    
    # 因子列表
    factors = [Momentum20(), Momentum60(), Volatility(), RSI(), MAAlign(), MACD()]
    
    if st.button("🔬 分析因子", type="primary"):
        with st.spinner("分析中..."):
            df = get_data(ticker, period)
            df = calc_indicators(df)
            
            # 计算因子IC
            results = []
            for factor in factors:
                fv = factor.calculate(df)
                forward_ret = df['Close'].pct_change(20).shift(-20)
                valid = fv.notna() & forward_ret.notna()
                ic = fv[valid].corr(forward_ret[valid])
                results.append({"因子": factor.name, "IC": ic, "|IC|": abs(ic)})
            
            results_df = pd.DataFrame(results).sort_values("|IC|", ascending=False)
            
            st.subheader("📊 因子IC排名")
            st.dataframe(results_df, use_container_width=True)
            
            # 可视化
            st.subheader("📈 因子相关性")
            factor_values = pd.DataFrame({f.name: f.calculate(df) for f in factors})
            corr = factor_values.corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
            plt.colorbar(im, ax=ax)
            ax.set_title('因子相关性矩阵')
            st.pyplot(fig)
            
            st.info("💡 IC (Information Coefficient) 越高，因子预测能力越强！")

# ============ TAB3: 因子说明 ============
with tab3:
    st.subheader("📚 因子策略说明")
    
    with st.expander("什么是因子？"):
        st.markdown("""
        **因子** 是影响股票收益的特征。
        
        | 因子类型 | 说明 | 例子 |
        |---------|------|------|
        | 动量 | 过去涨得好将来也涨 | 20日涨幅 |
        | 价值 | 被低估的股票 | PE, PB |
        | 质量 | 好公司 | ROE, 毛利率 |
        | 规模 | 小盘股收益高 | 市值 |
        | Volatility | 低波动策略 | 20日Volatility |
        """)
    
    with st.expander("动量因子"):
        st.markdown("""
        **原理**: 趋势会延续
        
        **计算**: `Return = (Close[t] - Close[t-N]) / Close[t-N]`
        
        **使用方法**:
        - 20日动量 > 5% → 买入
        - 60日动量 > 10% → 买入
        """)
    
    with st.expander("Volatility因子"):
        st.markdown("""
        **原理**: 低波动策略更稳定
        
        **计算**: `Volatility = StdDev(returns) * sqrt(252)`
        
        **使用方法**:
        - Volatility最低的20%股票买入
        - 反转效应
        """)
    
    with st.expander("RSI因子"):
        st.markdown("""
        **原理**: 超卖反弹，超买反转
        
        **计算**: RSI = 100 - 100/(1+RS), RS = 平均涨幅/平均跌幅
        
        **使用方法**:
        - RSI < 30 → 超卖，可能反弹
        - RSI > 70 → 超买，可能反转
        """)
    
    with st.expander("均线对齐因子"):
        st.markdown("""
        **原理**: 价格偏离均线太远会回归
        
        **计算**: `(Close - MA20) / MA20`
        
        **使用方法**:
        - 偏离度 < -5% → 买入
        - 偏离度 > +5% → 卖出
        """)


# ============ TAB4: AI因子挖掘 ============
with tab4:
    st.subheader("🤖 AI因子挖掘")
    
    st.markdown("""
    ### 自动发现最有效的因子组合
    
    使用机器学习自动寻找最优因子！
    """)
    
    # Use professional factors from the module
    st.subheader("📊 因子池 (专业模块)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_momentum20 = st.checkbox("动量因子(20日)", value=True)
        use_momentum60 = st.checkbox("动量因子(60日)", value=True)
        use_rsi = st.checkbox("RSI因子(14日)", value=True)
        use_volatility = st.checkbox("波动率因子", value=True)
    
    with col2:
        use_macd = st.checkbox("MACD因子", value=True)
        use_ma = st.checkbox("均线对齐因子", value=True)
        use_atr = st.checkbox("ATR因子", value=True)
        use_pe = st.checkbox("PE因子(简化)", value=False)
    
    # ML settings
    st.subheader("⚙️ ML设置")
    
    col1, col2 = st.columns(2)
    with col1:
        forward_days = st.slider("预测天数", 1, 60, 5)
        test_period = st.selectbox("测试周期", ["6mo", "1y", "2y", "3y"], index=2)
    
    with col2:
        top_n = st.slider("选择Top N因子", 2, 10, 3)
    
    if st.button("🔍 开始因子挖掘", type="primary"):
        with st.spinner("挖掘中..."):
            # Get data
            df_data = get_data(ticker, test_period)
            
            if df_data is not None and len(df_data) > 150:
                close = df_data['Close']
                
                # Use professional factors already in app.py
                
                factors = {}
                
                # Create factor instances and calculate
                if use_momentum20:
                    f = MomentumFactor(20)
                    factors[f.name] = f.calculate(df_data)
                
                if use_momentum60:
                    f = MomentumFactor(60)
                    factors[f.name] = f.calculate(df_data)
                
                if use_rsi:
                    f = RSIFactor(14)
                    factors[f.name] = f.calculate(df_data)
                
                if use_volatility:
                    f = VolatilityFactor(20)
                    factors[f.name] = f.calculate(df_data)
                
                if use_macd:
                    f = MACDFactor()
                    factors[f.name] = f.calculate(df_data)
                
                if use_ma:
                    f = MAAlignmentFactor(20)
                    factors[f.name] = f.calculate(df_data)
                
                if use_atr:
                    f = ATRFactor(14)
                    factors[f.name] = f.calculate(df_data)
                
                if use_pe:
                    f = PEFactor()
                    factors[f.name] = f.calculate(df_data)
                
                # Calculate IC for each factor
                forward_ret = close.pct_change(forward_days).shift(-forward_days)
                
                ic_results = []
                
                for name, factor in factors.items():
                    valid = factor.notna() & forward_ret.notna()
                    if valid.sum() > 50:
                        ic = factor[valid].corr(forward_ret[valid])
                        ic_results.append({"因子": name, "IC": ic, "|IC|": abs(ic), "样本数": valid.sum()})
                
                if ic_results:
                    ic_df = pd.DataFrame(ic_results).sort_values("|IC|", ascending=False)
                    
                    st.subheader("📈 因子IC排名")
                    st.dataframe(ic_df, use_container_width=True)
                    
                    # Top N factors
                    top_factors = ic_df.head(top_n)["因子"].tolist()
                    
                    st.subheader(f"🏆 Top {top_n} 因子组合")
                    st.write("选中的因子:", ", ".join(top_factors))
                    
                    # Create combined signal
                    combined_signal = pd.Series(0, index=close.index)
                    for fac in top_factors:
                        if fac in factors:
                            # Normalize
                            f_norm = (factors[fac] - factors[fac].mean()) / factors[fac].std()
                            combined_signal += f_norm
                    
                    combined_signal = combined_signal / len(top_factors)
                    
                    # Backtest the combined strategy
                    st.subheader("📊 组合策略回测")
                    
                    cash = initial_cash
                    position = 0
                    portfolio = [initial_cash]
                    trades = []
                    
                    for i in range(60, len(close)):
                        price = close.iloc[i]
                        signal = combined_signal.iloc[i]
                        
                        if pd.isna(signal):
                            continue
                        
                        if signal > 0.2 and position == 0:
                            shares = int(cash * max_position / price) + 1 + 1
                            if shares > 0:
                                cash -= shares * price * (1 + fee_rate + slippage)
                                position = shares
                                trades.append((df_data.index[i], 'BUY', price))
                        
                        elif signal < -0.2 and position > 0:
                            cash += position * price * (1 - fee_rate - slippage)
                            trades.append((df_data.index[i], 'SELL', price))
                            position = 0
                        
                        portfolio.append(cash + position * price)
                    
                    if position > 0:
                        cash += position * close.iloc[-1]
                    
                    final = cash
                    ret = (final - initial_cash) / initial_cash
                    
                    # Benchmark
                    bh_ret = (close.iloc[-1] - close.iloc[60]) / close.iloc[60]
                    
                    # Results
                    c1, c2, c3 = st.columns(3)
                    c1.metric("策略收益", f"{ret*100:+.1f}%")
                    c2.metric("买入持有", f"{bh_ret*100:+.1f}%")
                    c3.metric("超额收益", f"{(ret-bh_ret)*100:+.1f}%")
                    
                    # Chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(portfolio, label='Strategy')
                    bench_norm = close.iloc[60:].values / close.iloc[60] * initial_cash
                    ax.plot(bench_norm, alpha=0.5, label='Buy & Hold')
                    ax.set_title('AI Factor Portfolio')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Trades
                    if trades:
                        st.write(f"交易次数: {len(trades)}")
                    # 显示交易详情
                    if trades:
                        st.subheader("📋 交易明细")
                        
                        trade_data = []
                        for t in trades:
                            trade_data.append({
                                "日期": t[0].date() if hasattr(t[0], 'date') else str(t[0])[:10],
                                "操作": t[1],
                                "价格": f"${t[2]:.2f}",
                                "原因": "因子信号触发" if t[1] == "BUY" else "因子信号反转"
                            })
                        
                        if trade_data:
                            trade_df = pd.DataFrame(trade_data)
                            st.dataframe(trade_df, use_container_width=True)
                            
                            st.info(f"📌 因子组合策略: 综合{len(top_factors)}个因子信号，平均值>0.2买入，<-0.2卖出")


                else:
                    st.warning("无法计算因子IC，请检查数据")
            else:
                st.error("数据不足")



    with st.expander("📖 因子挖掘原理说明"):
        st.markdown("""
        ### 什么是因子挖掘？
        
        自动找出哪些因子对未来收益有预测能力！
        
        ### 关键参数
        
        | 参数 | 说明 |
        |------|------|
        | 预测天数 | 用今天的数据预测N天后的收益 |
        | IC值 | 因子与未来收益的相关性，越高越好 |
        | Top N | 选择IC最高的N个因子组合 |
        
        ### 流程
        
        1. 计算所有因子的IC值
        2. 选出Top N因子
        3. 等权组合这些因子
        4. 信号 > 0.3 买入，< -0.3 卖出
        
        ### IC解读
        
        - IC > 0.03: 有效因子
        - IC > 0.05: 强因子
        - IC > 0.10: 非常强的因子
        """)


    with st.expander("📖 IC指标说明"):
        st.markdown("""
        ### 什么是IC？
        
        **IC (Information Coefficient)** = 因子值与未来收益的相关系数
        
        - IC > 0: 因子值越高，未来收益越高
        - IC < 0: 因子值越高，未来收益越低
        
        ### IC多强算有效？
        
        | IC范围 | 效果 |
        |--------|------|
        | < 0.01 | 效果很弱 |
        | 0.01-0.03 | 有效 |
        | 0.03-0.05 | 较有效 |
        | > 0.05 | 非常有效 |
        
        ### 使用方法
        
        1. 计算每个因子的IC
        2. 选IC高的因子
        3. 用这些因子做组合策略
        """)


    with st.expander("📖 因子详解"):
        st.markdown("""
        ### 因子详解
        
        | 因子 | 计算公式 | 说明 |
        |------|----------|------|
        | **动量因子** | (收盘价-N日前)/N日前 | 过去涨得好将来也涨得好 |
        | **RSI** | 100-100/(1+RS) | 超卖反弹，超买反转 |
        | **波动率** | StdDev*sqrt(252) | 低波动更稳定 |
        | **均线比** | MA5/MA20 | 短期vs长期趋势 |
        | **MACD** | EMA12-EMA26 | 趋势动量 |
        | **成交量比** | Volume/MA20 | 放量缩量 |
        | **布林带宽度** | (上轨-下轨)/中轨 | 市场波动程度 |
        
        ### 预测天数
        - 预测5天 = 用今天因子预测5天后收益
        - 短预测 = 短线交易
        - 长预测 = 长线投资
        """)

st.caption("📈 Quant Platform | 数据: Yahoo Finance")
