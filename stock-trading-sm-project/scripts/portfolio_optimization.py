"""
Portfolio optimisation utilities.

Implements portfolio construction from model predictions using scipy:
  - equal weight
  - inverse-volatility (risk parity-ish)
  - mean-variance (max Sharpe via SLSQP)

Also provides a simple backtest of a rebalanced optimised portfolio.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class PortfolioOptimizer:
    """Construct portfolios from a returns matrix."""

    def __init__(self, returns: pd.DataFrame, risk_free: float = 0.0):
        # returns: DataFrame index=date, columns=symbol
        self.returns = returns.dropna(how='all').fillna(0.0)
        self.risk_free = risk_free
        self.symbols = self.returns.columns.tolist()
        self.n = len(self.symbols)


    def _portfolio_stats(self, weights: np.ndarray) -> (float, float):
        mean_ret = np.sum(self.returns.mean().values * weights) * TRADING_DAYS
        cov = self.returns.cov().values * TRADING_DAYS
        vol = np.sqrt(weights.T @ cov @ weights)
        return mean_ret, vol

    def equal_weight(self) -> pd.Series:
        w = np.repeat(1.0 / self.n, self.n)
        return pd.Series(w, index=self.symbols)

    def inverse_volatility(self) -> pd.Series:
        vol = self.returns.std()
        vol = vol.replace(0, np.nan).fillna(vol[vol > 0].mean())
        inv = 1.0 / vol
        w = inv / inv.sum()
        return w

    def max_sharpe(self, max_weight: float = 1.0) -> pd.Series:
        """Maximise Sharpe ratio via SLSQP."""
        mean_ret, _ = self._portfolio_stats(np.repeat(1.0 / self.n, self.n))
        if self.n == 1:
            return pd.Series([1.0], index=self.symbols)

        def neg_sharpe(weights):
            r, v = self._portfolio_stats(weights)
            return -(r - self.risk_free) / v if v > 0 else 1e9

        bounds = tuple((0.0, max_weight) for _ in range(self.n))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        x0 = np.repeat(1.0 / self.n, self.n)

        result = minimize(neg_sharpe, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        if not result.success:
            logger.warning(f"Max-Sharpe optimisation failed: {result.message}; using equal weight")
            return self.equal_weight()
        w = np.clip(result.x, 0, None)
        w = w / w.sum()
        return pd.Series(w, index=self.symbols)

    def min_volatility(self) -> pd.Series:
        cov = self.returns.cov().values * TRADING_DAYS
        if self.n == 1:
            return pd.Series([1.0], index=self.symbols)

        def vol(weights):
            return np.sqrt(weights.T @ cov @ weights)

        bounds = tuple((0.0, 1.0) for _ in range(self.n))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        x0 = np.repeat(1.0 / self.n, self.n)
        result = minimize(vol, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            return self.equal_weight()
        w = np.clip(result.x, 0, None)
        w = w / w.sum()
        return pd.Series(w, index=self.symbols)

    def backtest(self, weights: pd.Series) -> Dict[str, float]:
        w = weights.reindex(self.returns.columns).fillna(0.0).values
        port_ret = (self.returns.values @ w)
        port_ret = pd.Series(port_ret, index=self.returns.index)
        equity = (1 + port_ret).cumprod()
        sharpe = 0.0
        if port_ret.std() > 0:
            sharpe = float(np.sqrt(TRADING_DAYS) * port_ret.mean() / port_ret.std())
        roll_max = equity.cummax()
        mdd = float((equity / roll_max - 1.0).min())
        return {
            'total_return': float(equity.iloc[-1] - 1.0),
            'sharpe': sharpe,
            'max_drawdown': mdd,
            'volatility': float(port_ret.std() * np.sqrt(TRADING_DAYS)),
        }


if __name__ == "__main__":
    from config import PROCESSED_DIR

    preds_path = PROCESSED_DIR / "predictions.parquet"
    out_dir = PROCESSED_DIR

    print("Portfolio Optimisation")
    print("=" * 50)

    if not preds_path.exists():
        raise SystemExit(f"Predictions not found: {preds_path}")

    preds = pd.read_parquet(preds_path)
    preds['date'] = pd.to_datetime(preds['date'])
    preds = preds.sort_values('date')

    # Weights are fitted on the train period only, then evaluated on the
    # held-out period, so there is no look-ahead in the reported metrics.
    split_date = preds['date'].quantile(0.8)
    train = preds[preds['date'] < split_date]
    oos = preds[preds['date'] >= split_date]

    def make_returns(frame):
        px = frame.pivot_table(index='date', columns='symbol', values='Close')
        return px.pct_change().dropna(how='all').fillna(0.0)

    train_returns = make_returns(train)
    oos_returns = make_returns(oos)

    common = train_returns.columns.intersection(oos_returns.columns)
    train_returns = train_returns[common]
    oos_returns = oos_returns[common]

    fit = PortfolioOptimizer(train_returns)
    test = PortfolioOptimizer(oos_returns)

    strategies = {
        'equal_weight': fit.equal_weight(),
        'inverse_volatility': fit.inverse_volatility(),
        'max_sharpe': fit.max_sharpe(),
        'min_volatility': fit.min_volatility(),
    }

    summary = {}
    for name, w in strategies.items():
        stats = test.backtest(w)
        stats['top_holdings'] = ', '.join(w.sort_values(ascending=False).head(5).index.tolist())
        summary[name] = stats
        print(f"\n{name}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'portfolio_optimization.json', 'w') as f:
        json.dump({k: {kk: (vv if not isinstance(vv, float) else float(vv))
                       for kk, vv in v.items()} for k, v in summary.items()}, f, indent=2)

    # Save weights of the best strategy by out-of-sample Sharpe
    best = max(summary.items(), key=lambda kv: kv[1]['sharpe'])[0]
    strategies[best].to_csv(out_dir / 'optimized_weights.csv')
    print(f"\nBest portfolio strategy by OOS Sharpe: {best}")
    print(f"Saved results to {out_dir}")
