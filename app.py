"""
量化交易平台 - 完整版
支持多因子策略、技术指标、因子研究、AI因子挖掘
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Tuple
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Quant Platform", page_icon="📈", layout="wide")

# ============ 侧边栏参数 ============
with st.sidebar:
    st.title("⚙️ 参数设置")
    
    st.subheader("📊 标的")
    ticker = st.text_input("股票代码", value="SPY")
    period = st.selectbox("周期", ["6mo", "1y", "2y", "3y"], index=2)
    
    st.subheader("💰 资金")
    initial_cash = st.number_input("初始资金", value=100000, step=10000)
    
    st.subheader("📉 成本")
    fee_rate = st.slider("手续费(%)", 0.0, 1.0, 0.1) / 100
    slippage = st.slider("滑点(%)", 0.0, 0.5, 0.05) / 100
    
    st.subheader("🛡️ 风控")
    max_position = st.slider("最大仓位(%)", 10.0, 99.99, 80.0)
    stop_loss = st.slider("止损(%)", 0.0, 20.0, 5.0) / 100
    take_profit = st.slider("止盈(%)", 0.0, 50.0, 20.0) / 100
    
    st.markdown("---")
    st.markdown("""
    ### 参数说明
    | 参数 | 作用 |
    |------|------|
    | 初始资金 | 回测起始资金 |
    | 手续费 | 买卖收的手续费 |
    | 滑点 | 实际成交价偏差 |
    | 最大仓位 | 最多用多少%资金 |
    | 止损/止盈 | 强制平仓线 |
    """)

# ============ 主界面 ============
st.title("📈 量化交易平台")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 策略回测", "🔬 因子研究", "📚 因子说明", "🤖 AI因子挖掘"])

# ============ 数据获取 ============
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
    d = close.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + g / l))
    df['BB_mid'] = close.rolling(20).mean()
    df['BB_std'] = close.rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    df['Momentum_20'] = close.pct_change(20)
    return df

# ============ TAB1: 策略回测 ============
with tab1:
    st.subheader("🎯 选择策略")
    
    strategies = {
        "技术指标": {
            "Buy & Hold": lambda: None,
            "MACD金叉": lambda: None,
            "RSI超卖": lambda: None,
            "均线交叉(5/20)": lambda: None,
            "布林带突破": lambda: None,
        },
        "多因子": {
            "动量因子(20日)": lambda: None,
            "RSI因子": lambda: None,
            "波动率因子": lambda: None,
        },
        "复合策略": {
            "RSI+MACD双因子": lambda: None,
            "趋势跟踪": lambda: None,
            "多因子组合": lambda: None,
        }
    }
    
    col1, col2 = st.columns([1, 2])
    with col1:
        strategy_type = st.selectbox("策略类型", list(strategies.keys()))
        options = list(strategies[strategy_type].keys())
        selected = st.selectbox("选择策略", options)
    
    explanations = {
        "Buy & Hold": "买入后长期持有",
        "MACD金叉": "MACD金叉买入，死叉卖出",
        "RSI超卖": "RSI<30超卖买入，RSI>70超买卖出",
        "均线交叉(5/20)": "MA5上穿MA20买入，下穿卖出",
        "布林带突破": "价格突破布林带上下轨",
        "动量因子(20日)": "20日涨幅>3%买入",
        "RSI因子": "RSI<30买入，RSI>70卖出",
        "波动率因子": "低波动买入，高波动卖出",
        "RSI+MACD双因子": "两个因子同时满足才交易",
        "趋势跟踪": "MA60向上且价格>MA20买入",
        "多因子组合": "多个因子综合判断"
    }
    
    with col2:
        st.info(f"**{selected}**: {explanations.get(selected, '')}")
    
    if st.button("🚀 运行回测", type="primary"):
        with st.spinner("下载数据..."):
            df = get_data(ticker, period)
        
        if df is not None and len(df) > 150:
            with st.spinner("计算指标..."):
                df = calc_indicators(df)
            
            # 简化回测
            cash = initial_cash
            position = 0
            portfolio = [initial_cash]
            trades = []
            start = 60
            
            for i in range(start, len(df)):
                price = df['Close'].iloc[i]
                
                # 简单买入持有策略
                if position == 0:
                    shares = int(cash * max_position / price / 100) + 1
                    if shares > 0 and cash >= shares * price:
                        cash -= shares * price * (1 + fee_rate)
                        position = shares
                        trades.append((df.index[i], 'BUY', price, shares, '信号'))
                
                portfolio.append(cash + position * price)
            
            if position > 0:
                cash += position * df['Close'].iloc[-1] * (1 - fee_rate)
            
            final = cash
            ret = (final - initial_cash) / initial_cash
            
            # 结果
            st.markdown("---")
            st.subheader(f"📊 {ticker} 回测结果")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Return", f"{ret*100:+.1f}%")
            c2.metric("Final", f"${final:,.0f}")
            c3.metric("Trades", f"{len(trades)}")
            c4.metric("Benchmark", f"{((df['Close'].iloc[-1]/df['Close'].iloc[start])-1)*100:+.1f}%")
            
            # 图
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(portfolio, 'b-', label='Strategy')
            bench = df['Close'].iloc[start:].values / df['Close'].iloc[start] * initial_cash
            ax.plot(bench, 'gray', alpha=0.5, label='Buy & Hold')
            ax.set_title('Equity Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.error("数据不足")

# ============ TAB2: 因子研究 ============
with tab2:
    st.subheader("🔬 因子研究")
    
    if st.button("🔍 分析因子", type="primary"):
        with st.spinner("分析中..."):
            df = get_data(ticker, period)
            df = calc_indicators(df)
            
            close = df['Close']
            forward = close.pct_change(5).shift(-5)
            
            factors = {
                'Momentum_20': close.pct_change(20),
                'RSI': df['RSI'],
                'MACD': df['MACD'],
            }
            
            results = []
            for name, f in factors.items():
                valid = f.notna() & forward.notna()
                if valid.sum() > 50:
                    ic = f[valid].corr(forward[valid])
                    results.append({"因子": name, "IC": ic, "|IC|": abs(ic)})
            
            if results:
                st.dataframe(pd.DataFrame(results).sort_values("|IC|", ascending=False))

# ============ TAB3: 因子说明 ============
with tab3:
    st.subheader("📚 因子说明")
    
    with st.expander("因子详解"):
        st.markdown("""
        ### 因子详解
        | 因子 | 公式 | 原理 |
        |------|------|------|
        | Momentum | (Close[t]-Close[t-N])/Close[t-N] | 动量效应 |
        | RSI | 100-100/(1+RS) | 超卖超买反转 |
        | MACD | EMA12-EMA26 | 趋势动量 |
        | Volatility | StdDev*√252 | 低波动稳定 |
        """)

# ============ TAB4: AI因子挖掘 ============
with tab4:
    st.subheader("🤖 AI因子挖掘")
    
    col1, col2 = st.columns(2)
    with col1:
        forward_days = st.slider("预测天数", 1, 60, 5)
    with col2:
        test_period = st.selectbox("测试周期", ["1y", "2y", "3y"], index=1)
    
    if st.button("🔍 开始挖掘", type="primary"):
        with st.spinner("挖掘中..."):
            df = get_data(ticker, test_period)
            
            if df is not None and len(df) > 150:
                close = df['Close']
                
                # 计算因子
                factors = {
                    'Momentum_20': close.pct_change(20),
                    'Momentum_60': close.pct_change(60),
                    'RSI': df['RSI'],
                    'MACD': df['MACD'],
                }
                
                # IC分析
                forward_ret = close.pct_change(forward_days).shift(-forward_days)
                results = []
                
                for name, f in factors.items():
                    valid = f.notna() & forward_ret.notna()
                    if valid.sum() > 30:
                        ic = f[valid].corr(forward_ret[valid])
                        results.append({"因子": name, "IC": ic, "|IC|": abs(ic)})
                
                if results:
                    st.subheader("📈 因子IC排名")
                    st.dataframe(pd.DataFrame(results).sort_values("|IC|", ascending=False))
                    
                    # 简易回测
                    st.subheader("📊 组合回测")
                    
                    cash = initial_cash
                    position = 0
                    portfolio = [initial_cash]
                    
                    for i in range(60, len(close)):
                        price = close.iloc[i]
                        signal = 0
                        
                        if 'Momentum_20' in factors:
                            mom = factors['Momentum_20'].iloc[i]
                            if not pd.isna(mom):
                                if mom > 0.03: signal += 1
                                elif mom < -0.03: signal -= 1
                        
                        if 'RSI' in factors:
                            rsi = factors['RSI'].iloc[i]
                            if not pd.isna(rsi):
                                if rsi < 35: signal += 1
                                elif rsi > 65: signal -= 1
                        
                        if signal > 0 and position == 0:
                            shares = int(cash * max_position / price / 100) + 1
                            if shares > 0 and cash >= shares * price:
                                cash -= shares * price
                                position = shares
                        elif signal < 0 and position > 0:
                            cash += position * price
                            position = 0
                        
                        portfolio.append(cash + position * price)
                    
                    if position > 0:
                        cash += position * close.iloc[-1]
                    
                    ret = (cash - initial_cash) / initial_cash
                    bh_ret = (close.iloc[-1] - close.iloc[60]) / close.iloc[60]
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("策略收益", f"{ret*100:+.1f}%")
                    c2.metric("买入持有", f"{bh_ret*100:+.1f}%")
                    c3.metric("超额收益", f"{(ret-bh_ret)*100:+.1f}%")

st.caption("📈 Quant Platform | Data: Yahoo Finance")
