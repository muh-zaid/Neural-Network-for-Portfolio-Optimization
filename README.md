# End-to-End Neural Network for Direct Portfolio Weight Optimization

Traditional portfolio management predicts returns first, then optimizes, propagating error at each step. This project trains MLP and LSTM networks to learn portfolio weights directly from financial market data, optimized end-to-end on a Sharpe-based loss function and benchmarked against Mean-Variance Optimization.

---

## Access Code — Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook** and upload `ML_Project_V1.ipynb`
3. Upload any supporting files if prompted
4. Click **Runtime → Run all** (`Ctrl + F9`)

> All dependencies (`yfinance`, `torch`, `scikit-learn`, `scipy`, `matplotlib`) are pre-installed in Colab. No additional setup required.

---

## Methodology

```
Portfolio Optimization
│
├── 1. Data
│   ├── Source: yfinance — 10 ETFs, 2015–2025 (2,765 trading days)
│   └── Assets: SPY, QQQ, IWM, EFA, EEM, TLT, LQD, GLD, VNQ, XLE
│       (US equity, international, bonds, commodities, real estate, energy)
│
├── 2. Feature Engineering (110 features)
│   ├── Daily returns
│   ├── Lagged returns         — 1, 3, 5 days
│   ├── Rolling volatility     — 5, 21 days
│   ├── Moving average ratios  — 10, 21, 63 days
│   └── Momentum               — 21, 63 days
│
├── 3. Data Split (70 / 15 / 15)
│   ├── Train : 2015-04-06 → 2022-10-05  (1,891 days)
│   ├── Val   : 2022-10-06 → 2024-05-16  (405 days)
│   └── Test  : 2024-05-17 → 2025-12-30  (406 days)
│
├── 4. Benchmark — Mean-Variance Optimization (MVO)
│   ├── Maximizes Sharpe Ratio on train returns
│   ├── Per-asset weight cap: 20%
│   └── Static weights applied to test period
│
├── 5. ML Models
│   ├── Input: rolling 21-day windows, normalized via StandardScaler
│   │
│   ├── MLP (Multilayer Perceptron)
│   │   ├── Input: flattened window → (21 days × 110 features = 2,310)
│   │   ├── Architecture: Linear → ReLU → Dropout → Linear → Softmax
│   │   └── Experiments: 3 configs (hidden size, depth, dropout, LR)
│   │
│   └── LSTM (Long Short-Term Memory)
│       ├── Input: sequential window → (21 timesteps × 60 features)
│       ├── Architecture: LSTM → LayerNorm → Dropout → FC → ReLU → FC → Softmax
│       ├── Enhancements: orthogonal weight init, gradient clipping
│       └── Experiments: 3 configs (hidden size, depth, dropout, LR)
│
├── 6. Loss Function
│   ├── − Annualized Sharpe Ratio
│   ├── + λ_turn × L1 Turnover Penalty
│   └── + λ_dd   × Max Drawdown Penalty
│
└── 7. Evaluation
    ├── Cumulative & Annual Return
    ├── Annual Volatility
    ├── Sharpe Ratio
    ├── Max Drawdown
    └── Daily Turnover (MLP and LSTM only)
```

---

## Results (Test Period: May 2024 – Dec 2025)

| Metric              |    MVO |    MLP |   LSTM |
|---------------------|-------:|-------:|-------:|
| Cumulative Return   | 31.86% | 65.25% | 46.07% |
| Annual Return       | 17.72% | 33.19% | 24.53% |
| Annual Volatility   | 10.52% | 20.10% | 14.16% |
| Sharpe Ratio        | 1.6842 | 1.6514 | **1.7324** |
| Max Drawdown        |  -9.17% | -16.12% | -11.52% |
| Mean Daily Turnover |    N/A |  1.104 |  0.682 |

---

## Key Insights

**MLP generated the highest raw return** (65.25% cumulative) by concentrating heavily in QQQ (39.6%) and TLT (24.3%) ; a high-conviction, momentum-driven strategy that paid off in a bull market but carried higher drawdown (-16.12%) and turnover (1.10/day).

**LSTM achieved the best risk-adjusted performance** (Sharpe 1.73), distributing weights more evenly across GLD, IWM, EEM, and LQD. Its lower turnover (0.68/day) suggests the sequential architecture learned smoother, more stable allocation shifts.

**MVO remained the most capital-efficient**, with the lowest drawdown (-9.17%) and volatility (10.52%). Its static weights; concentrated in GLD, QQQ, SPY, TLT — held up well across the test window but left return potential unrealized.

---

## Tech Stack

`Python` · `PyTorch` · `yfinance` · `scikit-learn` · `SciPy` · `pandas` · `NumPy` · `Matplotlib`

---

## References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Yahoo Finance historical data via `yfinance`
