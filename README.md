# End-to-End Neural Network for Direct Portfolio Weight Optimization

---

## Objective

Traditional portfolio construction follows a two-step process: predict expected returns first, then optimize weights. This creates a compounding error problem which causes inaccurate return estimates flow directly into the allocation stage, and linear assumptions may miss nonlinear market behavior.

This project asks: **can end-to-end machine learning models learn portfolio allocation weights directly from historical market signals and outperform a Mean-Variance Optimization benchmark in risk-adjusted performance and portfolio stability?**

Instead of predicting returns as an intermediate step, MLP and LSTM networks are trained to output portfolio weights directly, optimized on a Sharpe-based portfolio-level loss function. Performance is evaluated against a classical MVO benchmark across a 10-ETF diversified universe.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Yahoo Finance](https://finance.yahoo.com) via `yfinance` |
| Period | January 2015 – December 2025 |
| Frequency | Daily adjusted closing prices |
| Assets | 10 ETFs |
| Total trading days | 2,765 |

**ETF Universe:**

| Ticker | Name | Asset Class |
|---|---|---|
| SPY | SPDR S&P 500 ETF Trust | US Large Cap Equity |
| QQQ | Invesco Nasdaq-100 ETF | US Tech Equity |
| IWM | iShares Russell 2000 ETF | US Small Cap Equity |
| EFA | iShares MSCI EAFE ETF | International Developed Equity |
| EEM | iShares MSCI Emerging Markets ETF | Emerging Market Equity |
| TLT | iShares 20+ Year Treasury Bond ETF | Long-term US Bonds |
| LQD | iShares Investment Grade Corporate Bond ETF | Corporate Bonds |
| GLD | SPDR Gold Shares ETF | Commodities / Gold |
| VNQ | Vanguard Real Estate ETF | Real Estate |
| XLE | Energy Select Sector SPDR ETF | Energy |

**Features engineered (110 total):** daily returns, lagged returns (1, 3, 5 days), rolling volatility (5, 21 days), moving average ratios (10, 21, 63 days), momentum (21, 63 days).

**Data split (chronological — no lookahead):**

| Split | Period | Days |
|---|---|---|
| Train | 2015-04-06 → 2022-10-05 | 1,891 |
| Validation | 2022-10-06 → 2024-05-16 | 405 |
| Test | 2024-05-17 → 2025-12-30 | 406 |

---

## Models

### 1. Mean-Variance Optimization (MVO) — Benchmark

A classical Markowitz portfolio that maximizes the Sharpe ratio using mean returns and the covariance matrix estimated from the training period. Each asset is capped at a maximum weight of 20% to prevent concentration. Weights are fixed after training and applied statically to the test period.

**Final allocation:** 20% each in GLD, QQQ, SPY, TLT — 15.51% in LQD — 4.49% in XLE — 0% in remaining ETFs.

---

### 2. Multi-Layer Perceptron (MLP)

Learns nonlinear relationships across flattened engineered features without explicitly modeling time-sequence behavior. A 21-day rolling window of 110 features is flattened into a 2,310-dimensional input vector. The output layer applies softmax over 10 neurons to produce valid long-only portfolio weights.

**Architecture:** `Linear → ReLU → Dropout → Linear → Softmax`

Five configurations were tested. Best configuration (Experiment 4):

| Hyperparameter | Value |
|---|---|
| Hidden size | 128 |
| Hidden layers | 2 |
| Dropout | 0.4 |
| Learning rate | 0.0003 |
| Epochs | 200 |
| λ_turn / λ_dd | 0.5 / 0.5 |
| Trainable parameters | 148,554 |

---

### 3. Long Short-Term Memory (LSTM)

Preserves the time dimension of the input and learns sequential dependencies across the 21-day window. Input is reduced to 60 features per timestep (returns, lagged returns, momentum). Orthogonal initialization is used for recurrent weights and gradient clipping is applied during training for stability.

**Architecture:** `LSTM → LayerNorm → Dropout → FC → ReLU → FC → Softmax`

Five configurations were tested. Best configuration (Experiment 3):

| Hyperparameter | Value |
|---|---|
| Hidden size | 256 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| Learning rate | 0.0003 |
| Epochs | 250 |
| λ_turn / λ_dd | 0.3 / 0.5 |
| Trainable parameters | 238,538 |

---

### Loss Function (MLP and LSTM)

All neural network models are trained end-to-end on a portfolio-level objective:

```
Loss = −Sharpe Ratio + λ₁ × Turnover + λ₂ × Drawdown
```

- **−Sharpe Ratio** — directly optimizes for return per unit of risk
- **Turnover penalty** — L1 weight change between periods, discourages excessive rebalancing
- **Drawdown penalty** — penalizes decline from peak cumulative portfolio value

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Cumulative Return | Total portfolio growth over the test period |
| Annual Return | Annualized average daily return (× 252) |
| Annual Volatility | Annualized standard deviation of daily returns |
| Sharpe Ratio | Annual return divided by annual volatility (risk-free rate = 0) |
| Max Drawdown | Largest peak-to-trough loss over the test period |
| Mean Daily Turnover | Average L1 weight change per day (MLP and LSTM only) |

---

## Key Results

**Test period: May 2024 – December 2025 (406 trading days)**

| Metric | MVO | MLP | LSTM |
|---|---:|---:|---:|
| Cumulative Return | 31.86% | **87.76%** | 44.18% |
| Annual Return | 17.72% | **40.99%** | 23.67% |
| Annual Volatility | **10.52%** | 19.46% | 13.79% |
| Sharpe Ratio | 1.6842 | **2.1061** | 1.7167 |
| Max Drawdown | **−9.17%** | −10.16% | −14.69% |
| Mean Daily Turnover | N/A | 1.071 | 0.799 |

**MLP** produced the highest cumulative return (87.76%), annual return (40.99%), and Sharpe ratio (2.11) by concentrating heavily in QQQ (~42%) and TLT/XLE (~22% each). Despite its higher volatility, its maximum drawdown remained close to the MVO benchmark (−10.16% vs −9.17%).

**LSTM** improved on MVO in both return and Sharpe ratio while maintaining lower turnover (0.80/day), but experienced the largest drawdown (−14.69%). Its more diversified allocation — led by GLD (~28%), LQD and IWM (~17% each) — reflects the sequential model learning smoother rebalancing patterns.

**MVO** remained the most conservative strategy, with the lowest volatility (10.52%) and drawdown (−9.17%), but significantly underperformed on return generation.

> At the experiment level, **LSTM Experiment 2** (Sharpe 1.9631, drawdown −10.40%) is worth noting as an alternative configuration that outperformed the validation-selected LSTM on test-period risk-adjusted metrics.

---

## How to Run

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook** and upload `Neural_Network_for_Portfolio_Optimization_-_Code.ipynb`
3. Click **Runtime → Run all** (`Ctrl + F9`)

All dependencies are pre-installed in Colab. No additional setup required.

### Option 2 — Local Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install torch yfinance scikit-learn scipy pandas numpy matplotlib

# Launch the notebook
jupyter notebook Neural_Network_for_Portfolio_Optimization_-_Code.ipynb
```

> Python 3.8+ recommended. A GPU is not required but will speed up LSTM training.

---

## Repository Structure

```
.
├── Project_Code.ipynb           # Full pipeline: data → features → models → evaluation
├── Project_Report.pdf           # IEEE-format research report
├── Project_Output.pdf           # Notebook output with all code, logs, and charts
├── Project_Presentation.pdf
├── Project_Plots_Tables
└── README.md

```

**Notebook sections:**

| Section | Contents |
|---|---|
| 1. Libraries & ETF Universe | Imports and asset definitions |
| 2. Data & Features | Price download, return calculation, feature engineering |
| 3. Benchmark Portfolio | MVO construction and backtest |
| 4. Dataset Preparation | Normalization, rolling windows, train/val/test tensors |
| 5. Loss Function | Custom Sharpe + turnover + drawdown objective |
| 6. ML Models | MLP and LSTM architecture definitions and training loops |
| 7. Overall Comparison | Backtests, visualizations, hyperparameter sensitivity heatmap |

---

## References

1. H. Markowitz, "Portfolio Selection," *The Journal of Finance*, vol. 7, no. 1, pp. 77–91, 1952.
2. S. Gu, B. Kelly, and D. Xiu, "Empirical Asset Pricing via Machine Learning," *The Review of Financial Studies*, vol. 33, no. 5, pp. 2223–2273, 2020.
3. J. B. Heaton, N. G. Polson, and J. H. Witte, "Deep learning for finance: deep portfolios," *Applied Stochastic Models in Business and Industry*, vol. 33, no. 1, pp. 3–12, 2017.
4. Z. Zhang, S. Zohren, and S. Roberts, "Deep Learning for Portfolio Optimisation," *The Journal of Financial Data Science*, vol. 2, no. 4, pp. 8–20, 2020.
5. R. Aroussi, "yfinance," GitHub, 2026. Available: https://github.com/ranaroussi/yfinance
6. W. F. Sharpe, "Mutual Fund Performance," *The Journal of Business*, vol. 39, no. 1, pp. 119–138, 1966.
7. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
