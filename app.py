"""
量化交易平台 - 完整版
支持多因子策略、技术指标、因子研究
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Fix matplotlib font for English charts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 导入因子日历模块
import sys
import os

# 添加因子模块路径
module_dir = os.path.dirname(os.path.abspath(__file__))
factors_path = os.path.join(module_dir, 'factors', 'calendar')
if factors_path not in sys.path:
    sys.path.insert(0, factors_path)

from factors_calendar import FactorFactory, FactorCalculator

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
    class 因子:
        def __init__(self, name): self.name = name
        def calculate(self, df): return pd.Series(0, index=df.index)

    class Momentum因子(因子):
        def __init__(self, period=20):
            super().__init__(f"Momentum_{period}")
            self.period = period
        def calculate(self, df):
            return df['Close'].pct_change(self.period)

    class Volatility因子(因子):
        def __init__(self, period=20):
            super().__init__(f"Volatility_{period}")
            self.period = period
        def calculate(self, df):
            return df['Close'].pct_change().rolling(self.period).std() * np.sqrt(252)

    class RSI因子(因子):
        def __init__(self, period=14):
            super().__init__(f"RSI_{period}")
            self.period = period
        def calculate(self, df):
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(self.period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
            return 100 - (100 / (1 + gain / loss))

    class MAAlign因子(因子):
        def __init__(self, period=20):
            super().__init__(f"MA_Align_{period}")
            self.period = period
        def calculate(self, df):
            ma = df['Close'].rolling(self.period).mean()
            return (df['Close'] - ma) / ma

    # 策略
    class 策略:
        def __init__(self, name): self.name = name
        def generate_signal(self, df, i): return 0

    class BuyHold(策略):
        def __init__(self): super().__init__("买入持有")
        def generate_signal(self, df, i): return 1 if i == 60 else 0

    class MACD(策略):
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

    class RSI(策略):
        def __init__(self, o=30, b=70):
            super().__init__(f"RSI({o}/{b})")
            self.o, self.b = o, b
        def generate_signal(self, df, i):
            if pd.isna(df['RSI'].iloc[i]): return 0
            rsi = df['RSI'].iloc[i]
            if rsi < self.o: return 1
            elif rsi > self.b: return -1
            return 0

    class MA(策略):
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

    class Bollinger(策略):
        def __init__(self):
            super().__init__("布林带")
        def generate_signal(self, df, i):
            if i < 20 or pd.isna(df['BB_lower'].iloc[i]): return 0
            c = df['Close'].iloc[i]
            if c < df['BB_lower'].iloc[i]: return 1
            elif c > df['BB_upper'].iloc[i]: return -1
            return 0

    class Momentum(策略):
        def __init__(self):
            super().__init__("动量因子")
        def generate_signal(self, df, i):
            if i < 30: return 0
            mom = df['Momentum_20'].iloc[i]
            if pd.isna(mom): return 0
            if mom > 0.03: return 1
            elif mom < -0.03: return -1
            return 0

    class Dual因子(策略):
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

    class Trend(策略):
        def __init__(self):
            super().__init__("趋势跟踪")
        def generate_signal(self, df, i):
            if i < 60: return 0
            ma60 = df['MA60'].iloc[i]; p = df['Close'].iloc[i]
            if pd.isna(ma60): return 0
            if df['MA60'].iloc[i-5] < ma60 and p > df['MA20'].iloc[i]: return 1
            elif df['MA60'].iloc[i-5] > ma60 or p < df['MA20'].iloc[i] * 0.95: return -1
            return 0

    class Multi因子(策略):
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
    class Momentum20(策略):
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

    class Momentum60(策略):
        def __init__(self):
            super().__init__("动量因子(60日)")
        def generate_signal(self, df, i):
            if i < 70: return 0
            mom = df['Close'].pct_change(60).iloc[i]
            if pd.isna(mom): return 0
            if mom > 0.10: return 1
            elif mom < -0.10: return -1
            return 0

    class RSI因子(策略):
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

    class MAAlign因子(策略):
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

    class Volatility因子(策略):
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
            "买入持有": BuyHold,
            "MACD金叉": MACD,
            "RSI超卖": lambda: RSI(30, 70),
            "均线交叉(5/20)": lambda: MA(5, 20),
            "布林带突破": Bollinger,
        },
        "多因子": {
            "动量因子(20日)": Momentum20,
            "动量因子(60日)": Momentum60,
            "RSI因子": RSI因子,
            "均线对齐": MAAlign因子,
            "Volatility": Volatility因子,
        },
        "复合策略": {
            "RSI+MACD双因子": Dual因子,
            "趋势跟踪": Trend,
            "多因子组合": Multi因子,
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
        "买入持有": "买入后长期持有。适合牛市。",
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
            st.markdown("**买入持有**: 买入后一直持有到回测结束")
    
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
            ax.plot(portfolio, label='Strategy')
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
    class 因子:
        def __init__(self, name): self.name = name
        def calculate(self, df): return pd.Series(0, index=df.index)
    
    class Momentum20(因子):
        def __init__(self): super().__init__("Momentum_20")
        def calculate(self, df): return df['Close'].pct_change(20)
    
    class Momentum60(因子):
        def __init__(self): super().__init__("Momentum_60")
        def calculate(self, df): return df['Close'].pct_change(60)
    
    class Volatility(因子):
        def __init__(self): super().__init__("Volatility_20")
        def calculate(self, df): return df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    class RSI(因子):
        def __init__(self): super().__init__("RSI_14")
        def calculate(self, df):
            d = df['Close'].diff()
            g = d.where(d > 0, 0).rolling(14).mean()
            l = (-d.where(d < 0, 0)).rolling(14).mean()
            return 100 - (100 / (1 + g / l))
    
    class MAAlign(因子):
        def __init__(self): super().__init__("MA_Align_20")
        def calculate(self, df):
            ma = df['Close'].rolling(20).mean()
            return (df['Close'] - ma) / ma
    
    class MACD(因子):
        def __init__(self): super().__init__("MACD")
        def calculate(self, df):
            return df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    
    # 因子选项
    use_calendar_in_research = st.checkbox("📚 包含因子日历因子 (54个)", value=True, 
                                           help="在因子研究中包含因子日历的54个基础因子")
    
    if use_calendar_in_research:
        # 使用因子日历的因子
        factors = list(FactorFactory.get_all_factors().values())
    else:
        # 使用原有因子
        factors = [Momentum20(), Momentum60(), Volatility(), RSI(), MAAlign(), MACD()]
    
    if st.button("🔬 分析因子", type="primary"):
        with st.spinner("分析中..."):
            df = get_data(ticker, period)
            df = calc_indicators(df)
            
            # 计算因子IC
            results = []
            for factor in factors:
                if use_calendar_in_research:
                    # Calendar因子是类，需要实例化
                    fv = factor.calculate(df)
                    f_name = factor.name
                else:
                    # 原有因子
                    fv = factor.calculate(df)
                    f_name = factor.name
                
                forward_ret = df['Close'].pct_change(20).shift(-20)
                valid = fv.notna() & forward_ret.notna()
                if valid.sum() > 30:
                    ic = fv[valid].corr(forward_ret[valid])
                    info = FactorFactory.get_factor_info(f_name)
                    results.append({
                        "因子": f_name, 
                        "名称": info.get('name', f_name),
                        "类别": info.get('category', '其他'),
                        "IC": ic, 
                        "|IC|": abs(ic)
                    })
            
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
            ax.set_title('因子 Correlation Matrix')
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
    
    # 因子 pool
    st.subheader("📊 因子 Pool")
    
    use_calendar_factors = st.checkbox("📚 Calendar 因子s (54)", value=True, 
                                        help="Use factors from 因子 Calendar (technical/momentum/volatility/liquidity/valuation/financial/size)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_momentum = st.checkbox("动量因子", value=True, disabled=use_calendar_factors)
        use_rsi = st.checkbox("RSI因子", value=True, disabled=use_calendar_factors)
        use_volatility = st.checkbox("波动率因子", value=True, disabled=use_calendar_factors)
        use_ma = st.checkbox("均线因子", value=True, disabled=use_calendar_factors)
    
    with col2:
        use_macd = st.checkbox("MACD因子", value=True, disabled=use_calendar_factors)
        use_volume = st.checkbox("成交量因子", value=True, disabled=use_calendar_factors)
        use_bb = st.checkbox("布林带因子", value=True, disabled=use_calendar_factors)
        use_momentum_60 = st.checkbox("动量因子(60日)", value=False, disabled=use_calendar_factors)
    
    # 信号阈值设置
    st.subheader("⚙️ 信号设置")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        signal_threshold = st.slider("买入信号阈值", 0.1, 1.0, 0.3, help="IC加权信号超过此值买入")
    with col_s2:
        sell_threshold = st.slider("卖出信号阈值", -1.0, -0.1, -0.2, help="IC加权信号低于此值卖出")
    
    # ML settings
    st.subheader("⚙️ ML设置")
    
    col1, col2 = st.columns(2)
    with col1:
        forward_days = st.slider("预测天数", 1, 60, 5, help="预测未来N天的收益")
        test_period = st.selectbox("测试周期", ["6mo", "1y", "2y"], index=1)
    
    with col2:
        top_n = st.slider("选择Top N因子", 2, 15, 5, help="选择IC最高的N个因子")
    
    if st.button("🔍 开始因子挖掘", type="primary"):
        with st.spinner("挖掘中..."):
            # Get data
            df_data = get_data(ticker, test_period)
            
            if df_data is not None and len(df_data) > 100:
                close = df_data['Close']
                
                # 使用因子日历模块
                if use_calendar_factors:
                    st.info("📚 Using Calendar 因子s...")
                    
                    # 使用FactorCalculator计算所有因子
                    calculator = FactorCalculator(df_data)
                    
                    with st.spinner("计算因子IC中..."):
                        ic_df = calculator.calculate_ic(forward_days=forward_days)
                    
                    if not ic_df.empty:
                        st.subheader("📊 因子IC排名（日历）")
                        st.dataframe(ic_df, use_container_width=True)
                        
                        # 选择Top N因子
                        top_n = min(top_n, len(ic_df))
                        top_factors = ic_df.head(top_n)["Factor"].tolist()
                        
                        st.subheader(f"🏆 Top {top_n} 因子 Portfolio (IC Weighted)")
                        st.write("已选因子：", ", ".join(top_factors))
                        
                        # 显示选中因子的详细信息
                        with st.expander("📖 已选因子详情", expanded=True):
                            for fac in top_factors:
                                info = FactorFactory.get_factor_info(fac)
                                st.markdown(f"""
                                **{fac} - {info.get('name', fac)}**
                                - 类别: {info.get('category', 'N/A')}
                                - 方向: {'Long (high=buy)' if info.get('direction')=='long' else 'Short (low=buy)'}
                                - 公式： `{info.get('formula', 'N/A')}`
                                - 描述： {info.get('description', 'N/A')}
                                - 信号： {info.get('signal', 'N/A')}
                                """)
                        
                        # 使用IC权重生成加权信号 (改进版)
                        all_factors = calculator.calculate_all()
                        weighted_signal = calculator.generate_weighted_signal(ic_df, threshold=0.3)
                        
                        # 显示信号分布
                        st.subheader("📊 信号分布")
                        
                        # 解释信号含义
                        with st.expander("📖 信号分布说明"):
                            st.markdown("""
                            **什么是信号分布？**
                            - X轴：IC加权因子得分（综合评分）
                            - Y轴：出现次数
                            
                            **如何解读？**
                            - **红色线(买入阈值)**: 得分高于此值 → 买入信号
                            - **绿色线(卖出阈值)**: 得分低于此值 → 卖出信号
                            
                            **得分分布**:
                            - 大部分得分在0附近（中性）
                            - 得分越高 → 因子组合越看好
                            - 得分越低 → 因子组合越看跌
                            """)
                        
                        fig_sig, ax_sig = plt.subplots(figsize=(10, 3))
                        ax_sig.hist(weighted_signal.dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
                        ax_sig.axvline(x=signal_threshold, color='r', linestyle='--', linewidth=2, label=f'Buy: {signal_threshold}')
                        ax_sig.axvline(x=sell_threshold, color='g', linestyle='--', linewidth=2, label=f'Sell: {sell_threshold}')
                        ax_sig.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Neutral: 0')
                        ax_sig.set_xlabel('Composite Factor Score')
                        ax_sig.set_ylabel('Frequency')
                        ax_sig.set_title('IC-Weighted Signal Distribution')
                        ax_sig.legend(loc='upper right')
                        ax_sig.grid(True, alpha=0.3)
                        st.pyplot(fig_sig)
                        
                        # Backtest with weighted signal
                        st.subheader("📊 组合回测（IC加权）")
                        cash = initial_cash
                        position = 0
                        portfolio = [initial_cash]
                        trades = []
                        
                        for i in range(60, len(close)):
                            price = close.iloc[i]
                            signal = weighted_signal.iloc[i]
                            
                            if pd.isna(signal):
                                portfolio.append(portfolio[-1])
                                continue
                            
                            # 买入信号
                            if signal > signal_threshold and position == 0:
                                shares = int(cash * max_position / price)
                                if shares > 0:
                                    cost = shares * price * (1 + fee_rate + slippage)
                                    if cost <= cash:
                                        cash -= cost
                                        position = shares
                                        trades.append((df_data.index[i], 'BUY', price, round(signal, 3)))
                            
                            # 卖出信号
                            elif signal < sell_threshold and position > 0:
                                proceeds = position * price * (1 - fee_rate - slippage)
                                cash += proceeds
                                trades.append((df_data.index[i], 'SELL', price, round(signal, 3)))
                                position = 0
                            
                            portfolio.append(cash + position * price)
                        
                        if position > 0:
                            cash += position * close.iloc[-1]
                        
                        final = cash
                        ret = (final - initial_cash) / initial_cash
                        bh_ret = (close.iloc[-1] - close.iloc[60]) / close.iloc[60]
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("策略收益", f"{ret*100:+.1f}%")
                        c2.metric("买入持有", f"{bh_ret*100:+.1f}%")
                        c3.metric("超额收益", f"{(ret-bh_ret)*100:+.1f}%")
                        
                        # 计算波动率
                        import numpy as np
                        
                        # 策略收益序列
                        portfolio_arr = np.array(portfolio[1:])  # 去掉第一个
                        strategy_returns = np.diff(portfolio_arr) / portfolio_arr[:-1]
                        strategy_returns = strategy_returns[~np.isnan(strategy_returns) & ~np.isinf(strategy_returns)]
                        strategy_vol = np.std(strategy_returns, ddof=1) * np.sqrt(252) if len(strategy_returns) > 0 else 0
                        
                        # Buy & Hold 收益序列
                        bench_arr = bench_norm[1:]
                        bh_returns = np.diff(bench_arr) / bench_arr[:-1]
                        bh_returns = bh_returns[~np.isnan(bh_returns) & ~np.isinf(bh_returns)]
                        bh_vol = np.std(bh_returns, ddof=1) * np.sqrt(252) if len(bh_returns) > 0 else 0
                        
                        # 显示波动率
                        c4, c5, c6 = st.columns(3)
                        c4.metric("策略波动率", f"{strategy_vol*100:.2f}%")
                        c5.metric("买入持有波动率", f"{bh_vol*100:.2f}%")
                        c6.metric("波动率差异", f"{(strategy_vol - bh_vol)*100:+.2f}%", 
                                 delta_color="inverse" if strategy_vol < bh_vol else "normal")
                        
                        # Chart
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(portfolio, label='Strategy', linewidth=2)
                        bench_norm = close.iloc[60:].values / close.iloc[60] * initial_cash
                        ax.plot(bench_norm, alpha=0.5, label='Buy & Hold', linestyle='--')
                        ax.set_title('Calendar Factor Portfolio')
                        ax.legend(['Strategy', 'Buy & Hold'])
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # 交易记录
                        if trades:
                            st.write(f"📋 Trades: {len(trades)}")
                            trades_df = pd.DataFrame(trades, columns=['日期', '操作', '价格', '信号'])
                            st.dataframe(trades_df, use_container_width=True)
                        
                        # 显示因子类别分布
                        st.subheader("📊 因子类别分布")
                        category_counts = ic_df.head(top_n)["Category"].value_counts()
                        st.bar_chart(category_counts)
                        
                        # 显示因子方向分布
                        st.subheader("📊 因子方向分布")
                        direction_counts = ic_df.head(top_n)["Direction"].value_counts()
                        st.bar_chart(direction_counts)
                        
                        # ==================== 策略决策解释模块 ====================
                        with st.expander("📖 策略决策解释", expanded=True):
                            st.markdown("""
                            ## 1. 数学模型
                            **IC加权多因子模型** (Information Coefficient Weighted Factor Model)
                            
                            公式：
                            $$Score_i = \sum_{k=1}^{N} (IC_k \times Factor_{i,k})$$
                            
                            其中：
                            - $i$ = 股票
                            - $k$ = 因子编号
                            - $IC_k$ = 因子k的IC值（预测能力）
                            - $Factor_{i,k}$ = 股票i的标准化因子值
                            
                            最终选择Top N股票构建组合。
                            """)
                            
                            # 2. 因子列表
                            st.markdown("### 2. 因子列表")
                            for fac in top_factors[:5]:
                                info = FactorFactory.get_factor_info(fac)
                                st.markdown(f"""
                                **{fac}** - {info.get('name', fac)}
                                - 类别: {info.get('category', 'N/A')}
                                - 方向: {'高值买入' if info.get('direction')=='long' else '低值买入'}
                                - 金融含义: {info.get('description', 'N/A')}
                                """)
                            
                            # 3. 信号生成逻辑
                            st.markdown("""
                            ### 3. 信号生成逻辑
                            
                            Step 1: 计算所有股票的因子值
                            Step 2: 标准化因子 (z-score)
                            Step 3: 根据IC加权计算综合评分
                            Step 4: 按评分排序
                            Step 5: 选择Top股票买入
                            Step 6: 周期性调仓
                            """)
                            
                            # 4. 交易触发原因
                            if len(trades) > 0:
                                st.markdown("### 4. 交易触发原因")
                                for idx, trade in enumerate(trades[:5]):
                                    date = trade[0]
                                    action = trade[1]
                                    price = trade[2]
                                    st.markdown(f"""
                                    **Trade {idx+1}**
                                    - 日期: {date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date}
                                    - 操作: {action}
                                    - 价格: ${price:.2f}
                                    """)
                            
                            # 5. 因子贡献
                            st.markdown("### 5. 因子贡献")
                            contribution_data = []
                            for fac in top_factors:
                                ic_val = ic_df[ic_df['Factor']==fac]['IC'].values[0] if len(ic_df[ic_df['Factor']==fac]) > 0 else 0
                                contribution_data.append({'因子': fac, 'IC贡献': abs(ic_val)})
                            
                            contrib_df = pd.DataFrame(contribution_data)
                            if len(contrib_df) > 0:
                                # Draw contribution chart
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.barh(contrib_df['因子'], contrib_df['IC贡献'], color='steelblue')
                                ax.set_xlabel('IC Contribution')
                                ax.set_title('Factor Contribution to Composite Score')
                                ax.invert_yaxis()
                                st.pyplot(fig)
                            
                            st.success("✅ 策略决策解释完成！")
                        
                        st.success("✅ 因子 mining completed!")
                        st.stop()
                        
                        # Backtest
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
                                shares = int(cash * max_position / price) + 1
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
                        bh_ret = (close.iloc[-1] - close.iloc[60]) / close.iloc[60]
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("策略收益", f"{ret*100:+.1f}%")
                        c2.metric("买入持有", f"{bh_ret*100:+.1f}%")
                        c3.metric("超额收益", f"{(ret-bh_ret)*100:+.1f}%")
                        
                        # 计算波动率
                        import numpy as np
                        
                        # 策略收益序列
                        portfolio_arr = np.array(portfolio[1:])
                        strategy_returns = np.diff(portfolio_arr) / portfolio_arr[:-1]
                        strategy_returns = strategy_returns[~np.isnan(strategy_returns) & ~np.isinf(strategy_returns)]
                        strategy_vol = np.std(strategy_returns, ddof=1) * np.sqrt(252) if len(strategy_returns) > 0 else 0
                        
                        # Buy & Hold 收益序列
                        bench_arr = bench_norm[1:]
                        bh_returns = np.diff(bench_arr) / bench_arr[:-1]
                        bh_returns = bh_returns[~np.isnan(bh_returns) & ~np.isinf(bh_returns)]
                        bh_vol = np.std(bh_returns, ddof=1) * np.sqrt(252) if len(bh_returns) > 0 else 0
                        
                        # 显示波动率
                        c4, c5, c6 = st.columns(3)
                        c4.metric("策略波动率", f"{strategy_vol*100:.2f}%")
                        c5.metric("买入持有波动率", f"{bh_vol*100:.2f}%")
                        c6.metric("波动率差异", f"{(strategy_vol - bh_vol)*100:+.2f}%",
                                 delta_color="inverse" if strategy_vol < bh_vol else "normal")
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(portfolio, label='Strategy')
                        bench_norm = close.iloc[60:].values / close.iloc[60] * initial_cash
                        ax.plot(bench_norm, alpha=0.5, label='Buy & Hold')
                        ax.set_title('Calendar Factor Portfolio')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        if trades:
                            st.write(f"Trades: {len(trades)}")
                        
                        # 显示因子类别分布
                        st.subheader("📊 因子类别分布")
                        category_counts = ic_df.head(top_n)["Category"].value_counts()
                        st.bar_chart(category_counts)
                        
                        st.success("✅ 因子 mining completed!")
                        st.stop()
                
                # 原有的因子计算逻辑
                factors = {}
                
                if use_momentum:
                    factors['Momentum_5'] = close.pct_change(5)
                    factors['Momentum_10'] = close.pct_change(10)
                    factors['Momentum_20'] = close.pct_change(20)
                
                if use_rsi:
                    delta = close.diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    factors['RSI'] = 100 - (100 / (1 + gain / loss))
                
                if use_volatility:
                    factors['Volatility_10'] = close.pct_change().rolling(10).std() * np.sqrt(252)
                    factors['Volatility_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
                
                if use_ma:
                    factors['MA5_MA20_Ratio'] = close.rolling(5).mean() / close.rolling(20).mean()
                    factors['MA20_MA60_Ratio'] = close.rolling(20).mean() / close.rolling(60).mean()
                
                if use_macd:
                    ema12 = close.ewm(span=12).mean()
                    ema26 = close.ewm(span=26).mean()
                    factors['MACD'] = ema12 - ema26
                    factors['MACD_Signal'] = factors['MACD'].ewm(span=9).mean()
                
                if use_volume:
                    if 'Volume' in df_data.columns:
                        factors['Volume_MA20'] = df_data['Volume'] / df_data['Volume'].rolling(20).mean()
                
                if use_bb:
                    bb_mid = close.rolling(20).mean()
                    bb_std = close.rolling(20).std()
                    factors['BB_Width'] = (bb_mid + 2*bb_std - (bb_mid - 2*bb_std)) / bb_mid
                
                if use_momentum_60:
                    factors['Momentum_60'] = close.pct_change(60)
                
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
                    
                    # 选择Top N因子
                    top_factors = ic_df.head(top_n)["Factor"].tolist()
                    
                    st.subheader(f"🏆 Top {top_n} 因子 Portfolio")
                    st.write("已选因子：", ", ".join(top_factors))
                    
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
                    ax.set_title('AI 因子 Portfolio')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Trades
                    if trades:
                        st.write(f"Trades: {len(trades)}")
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
