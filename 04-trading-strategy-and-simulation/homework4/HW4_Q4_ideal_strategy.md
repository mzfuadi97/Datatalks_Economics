# HW4 — Question 4: My Ideal Trading Strategy

**Idea: a confidence-ranked, capital-capped ensemble strategy.**

1. **Signal (ensemble).** For every ticker and day, average the class-1 probabilities of the Random Forest (`rf_best`) and the Decision Tree (`clf_best`). A stock is a candidate only if the Random Forest probability is ≥ 0.60 *and* the Decision Tree probability is ≥ 0.84. Two models trained differently must agree, which cuts false positives.
2. **Selection (fewer trades).** From the candidates, take at most the **top 3 by average probability** each day, and skip any ticker I already hold (no stacking the same stock inside its 30-day holding window).
3. **Position sizing.** Split the capital into 30 daily "slots" (because each position is held ~30 days). Each day uses only that day's slot, divided among the selected stocks in proportion to their probability. If there is no signal, the slot stays in cash (or a T-bill ETF), so capital is never over-committed.
4. **Risk management.**
   - Market regime filter: open no new longs when the S&P 500 is below its 200-day moving average (`growth_snp500_365d < 1` works as a proxy).
   - Per-position stop-loss of −8% and take-profit of +15%. Otherwise exit on day 30.
   - Diversification: at most 1 new position per `ticker_type` per day, and no single stock above 10% of the portfolio.
5. **Implementation.** Add `proba_rf` and `proba_dt` columns to `new_df`. Build a `pred_ensemble_top3` column with `groupby('Date').rank()` on the average probability, and simulate it with the same `sim1_` loop, but use the slot-based capital (`capital / 30` per day) instead of the Q75 estimate. Tune the two thresholds and the "top N" on the **validation** set only, then report CAGR, max drawdown and trades/day on the test set.

The goal is a CAGR similar to or higher than the best predictor (~1.15), with **≤ 3 trades per day** instead of ~14.
