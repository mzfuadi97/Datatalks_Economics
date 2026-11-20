"""
Trading strategy simulation.

Consumes the predictions produced by train_ml_models.py
(columns: symbol, date, Close, target, pred_class, prob_up, prob_down)
and simulates several strategies, comparing against an equal-weight
buy-and-hold benchmark.

Strategies:
  sim1 - vectorized long-only  : go long when prob_up > threshold
  sim2 - vectorized long/short : long when prob_up high, short when prob_down high
  sim3 - iterative long-only   : reinvest capital across top-N signals each day

All metrics are computed on daily portfolio returns.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class TradingSimulator:
    """Simulate trading strategies from model predictions."""

    def __init__(self, df: pd.DataFrame, initial_capital: float = 100000.0,
                 benchmark: Optional[pd.Series] = None):
        self.initial_capital = float(initial_capital)
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.benchmark = benchmark

        # Pivot helpers
        self.prices = self.df.pivot_table(index='date', columns='symbol', values='Close')
        self.returns = self.prices.pct_change().fillna(0.0)
        self.prob_up = self.df.pivot_table(index='date', columns='symbol', values='prob_up')
        if 'prob_down' in self.df.columns:
            self.prob_down = self.df.pivot_table(index='date', columns='symbol', values='prob_down')
        else:
            self.prob_down = 1.0 - self.prob_up


    # Metrics

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
        r = returns.dropna()
        if r.std() == 0 or len(r) < 2:
            return 0.0
        excess = r - risk_free / TRADING_DAYS
        return float(np.sqrt(TRADING_DAYS) * excess.mean() / r.std())

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series) -> float:
        r = returns.dropna()
        downside = r[r < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return float(np.sqrt(TRADING_DAYS) * r.mean() / downside.std())

    @staticmethod
    def calculate_max_drawdown(equity: pd.Series) -> float:
        roll_max = equity.cummax()
        drawdown = equity / roll_max - 1.0
        return float(drawdown.min())

    @staticmethod
    def calculate_cagr(equity: pd.Series) -> float:
        if len(equity) < 2:
            return 0.0
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        if years <= 0:
            return 0.0
        total_return = equity.iloc[-1] / equity.iloc[0]
        if total_return <= 0:
            return -1.0
        return float(total_return ** (1.0 / years) - 1.0)

    def calculate_performance_metrics(self, returns: pd.Series,
                                      equity: pd.Series,
                                      benchmark_equity: Optional[pd.Series] = None) -> Dict[str, float]:
        metrics = {
            'total_return': float(equity.iloc[-1] / equity.iloc[0] - 1.0),
            'cagr': self.calculate_cagr(equity),
            'sharpe': self.calculate_sharpe_ratio(returns),
            'sortino': self.calculate_sortino_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(equity),
            'volatility': float(returns.std() * np.sqrt(TRADING_DAYS)),
            'win_rate': float((returns > 0).mean()),
        }
        if benchmark_equity is not None and len(benchmark_equity) == len(equity):
            bm_total = benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1.0
            metrics['benchmark_total_return'] = float(bm_total)
            metrics['benchmark_cagr'] = self.calculate_cagr(benchmark_equity)
            metrics['excess_return'] = metrics['total_return'] - float(bm_total)
        return metrics


    # Strategies

    def _benchmark_equity(self) -> pd.Series:
        if self.benchmark is not None:
            b = self.benchmark.reindex(self.prices.index).ffill().fillna(0.0)
            b = b / b.iloc[0] * self.initial_capital
            return b
        # Equal-weight buy & hold across the universe
        bm_rets = self.returns.mean(axis=1)
        return (1 + bm_rets).cumprod() * self.initial_capital

    def simulate_vectorized_strategy(
        self,
        signal_column: str = 'prob_up',
        entry_threshold: float = 0.5,
        long_short: bool = False,
        short_threshold: float = 0.5,
        top_n: Optional[int] = None,
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """Vectorized portfolio simulation, equal-weight among selected names."""
        logger.info(f"Vectorized strategy: {signal_column}, threshold={entry_threshold}, long_short={long_short}")

        up = self.prob_up
        signals = (up > entry_threshold).astype(float)

        if long_short:
            down = self.prob_down
            signals = signals - (down > short_threshold).astype(float)

        # Trade on NEXT day -> shift signals by 1 to avoid look-ahead
        positions = signals.shift(1).fillna(0.0)

        if top_n is not None:
            # keep only the top_n strongest signals (by prob_up) each day
            ranked = up.rank(axis=1, ascending=False)
            keep = ranked <= top_n
            positions = positions.where(keep, 0.0)

        gross = positions.abs().sum(axis=1)
        weights = positions.div(gross.replace(0, np.nan), axis=0).fillna(0.0)

        strat_ret = (weights * self.returns).sum(axis=1)
        equity = (1 + strat_ret).cumprod() * self.initial_capital
        benchmark_equity = self._benchmark_equity()

        daily = pd.DataFrame({
            'strategy_return': strat_ret,
            'equity': equity,
            'benchmark_return': benchmark_equity.pct_change().fillna(0.0),
            'benchmark_equity': benchmark_equity,
            'n_positions': gross,
        })
        metrics = self.calculate_performance_metrics(strat_ret, equity, benchmark_equity)

        return {'daily': daily, 'metrics': metrics}

    def simulate_iterative_strategy(
        self,
        entry_threshold: float = 0.5,
        top_n: int = 5,
        commission: float = 0.001,
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Iterative simulation (row-by-row) with capital reinvestment.
        Each day, allocate all available cash equally across up to top_n
        symbols with strong 'up' signals, holding until the signal fades.
        """
        logger.info(f"Iterative strategy: top_n={top_n}, threshold={entry_threshold}")

        dates = self.prices.index
        cash = self.initial_capital
        holdings = {}  # symbol -> shares
        equity_curve = []
        prev_equity = self.initial_capital

        up = self.prob_up
        prices = self.prices

        for i, date in enumerate(dates):
            # mark-to-market portfolio value at current prices
            value_holdings = sum(
                shares * prices.loc[date, sym]
                for sym, shares in holdings.items()
                if not np.isnan(prices.loc[date, sym])
            )
            equity = cash + value_holdings

            if i < len(dates) - 1:
                next_date = dates[i + 1]
            else:
                next_date = date

            if i < len(dates) - 1:
                # Determine desired holdings based on TODAY's signal
                row = up.loc[date].dropna()
                selected = row[row > entry_threshold].sort_values(ascending=False).head(top_n).index

                # Sell holdings no longer selected
                for sym in list(holdings.keys()):
                    if sym not in selected:
                        px = prices.loc[next_date, sym]
                        if not np.isnan(px):
                            cash += holdings[sym] * px * (1 - commission)
                        del holdings[sym]

                # Buy/rebalance selected equally with available cash
                if len(selected) > 0:
                    current_value = sum(
                        holdings.get(sym, 0) * prices.loc[next_date, sym]
                        for sym in selected
                        if not np.isnan(prices.loc[next_date, sym])
                    )
                    total_target = cash + current_value
                    target_per = total_target / len(selected)
                    for sym in selected:
                        px = prices.loc[next_date, sym]
                        if np.isnan(px) or px <= 0:
                            continue
                        cur_shares = holdings.get(sym, 0)
                        cur_val = cur_shares * px
                        delta_val = target_per - cur_val
                        if delta_val > 0 and cash >= delta_val:
                            buy_shares = delta_val / px
                            holdings[sym] = cur_shares + buy_shares
                            cash -= buy_shares * px * (1 + commission)
                        elif delta_val < 0:
                            sell_shares = min(cur_shares, -delta_val / px)
                            holdings[sym] = cur_shares - sell_shares
                            cash += sell_shares * px * (1 - commission)

            # recompute equity after trades
            value_holdings = sum(
                shares * prices.loc[date, sym]
                for sym, shares in holdings.items()
                if not np.isnan(prices.loc[date, sym])
            )
            equity = cash + value_holdings
            daily_ret = (equity / prev_equity - 1.0) if prev_equity else 0.0
            prev_equity = equity
            equity_curve.append({'date': date, 'equity': equity, 'strategy_return': daily_ret,
                                 'n_positions': len(holdings)})

        daily = pd.DataFrame(equity_curve).set_index('date')
        benchmark_equity = self._benchmark_equity()
        daily['benchmark_return'] = benchmark_equity.pct_change().fillna(0.0)
        daily['benchmark_equity'] = benchmark_equity
        metrics = self.calculate_performance_metrics(
            daily['strategy_return'], daily['equity'], benchmark_equity)

        return {'daily': daily, 'metrics': metrics}


    def export_results(self, results: Dict[str, Dict], output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary = {}
        all_daily = []
        for name, res in results.items():
            daily = res['daily'].copy()
            daily['strategy'] = name
            all_daily.append(daily.reset_index())
            summary[name] = res['metrics']

        pd.concat(all_daily, ignore_index=True).to_parquet(out / 'simulation_daily.parquet')
        pd.concat(all_daily, ignore_index=True).to_csv(out / 'simulation_daily.csv', index=False)

        with open(out / 'simulation_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)

        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(out / 'simulation_summary.csv')
        logger.info(f"Exported simulation results to {out}")
        return summary_df


if __name__ == "__main__":
    from config import PROCESSED_DIR

    preds_path = PROCESSED_DIR / "predictions.parquet"
    out_dir = PROCESSED_DIR

    print("Trading Simulation")
    print("=" * 50)

    if not preds_path.exists():
        raise SystemExit(f"Predictions not found: {preds_path}. Run train_ml_models.py first.")

    preds = pd.read_parquet(preds_path)
    print(f"Loaded predictions: {preds.shape}")

    # Use the out-of-sample portion (last 20%) to avoid fitting on training data
    preds = preds.sort_values('date')
    split_date = preds['date'].quantile(0.8)
    preds_oos = preds[preds['date'] >= split_date].copy()
    print(f"Out-of-sample predictions: {preds_oos.shape} "
          f"({preds_oos['date'].min().date()} -> {preds_oos['date'].max().date()})")

    # Benchmark: S&P 500 (SPY) if available, otherwise equal-weight buy & hold
    benchmark = None
    if 'BENCHMARK_returns' in preds_oos.columns:
        bm = preds_oos.groupby('date')['BENCHMARK_returns'].first().fillna(0.0)
        if bm.abs().sum() > 0:
            benchmark = (1 + bm).cumprod() * 100000
            print("Benchmark: S&P 500 (SPY)")
    if benchmark is None:
        print("Benchmark: equal-weight buy & hold")

    sim = TradingSimulator(preds_oos, initial_capital=100000, benchmark=benchmark)

    results = {}
    results['sim1_long_only'] = sim.simulate_vectorized_strategy(
        entry_threshold=0.5, long_short=False, top_n=5)
    results['sim2_long_short'] = sim.simulate_vectorized_strategy(
        entry_threshold=0.5, long_short=True, short_threshold=0.5, top_n=5)
    results['sim3_iterative'] = sim.simulate_iterative_strategy(
        entry_threshold=0.5, top_n=5, commission=0.001)

    summary = sim.export_results(results, str(out_dir))

    print("\nStrategy Performance Summary:")
    cols = [c for c in ['total_return', 'cagr', 'sharpe', 'max_drawdown',
                        'benchmark_total_return', 'excess_return'] if c in summary.columns]
    print(summary[cols].to_string())
