# Quant Trading Platform

A powerful quantitative trading platform with:

- **Strategy Backtesting**: Test multiple trading strategies (MACD, RSI, Moving Averages, Bollinger Bands, etc.)
- **Factor Research**: Analyze and rank factors using IC metrics
- **AI Factor Mining**: Automatically discover optimal factor combinations using machine learning
- **Multi-Factor Strategies**: Combine multiple factors for better returns

## Features

### Strategy Types
- Technical Indicators: MACD, RSI, MA Crossover, Bollinger Bands
- Multi-Factor: Momentum, Volatility, RSI Factor, MA Alignment
- Composite: RSI+MACD Dual Factor, Trend Following, Multi-Factor Portfolio

### Factor Research
- IC (Information Coefficient) analysis
- Factor correlation matrix
- Automatic factor ranking

### AI Factor Mining
- Select from 8+ factor types
- ML-based factor combination
- Automated backtesting

## Installation

```bash
pip install streamlit yfinance pandas numpy matplotlib
```

## Run

```bash
cd quant_platform
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub
4. Deploy!

## License

MIT
