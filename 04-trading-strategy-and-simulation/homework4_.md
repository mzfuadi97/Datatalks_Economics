# Homework: Trading Strategy

## Questions

### 1. Find the CAGR for the `pred10_rf_best_rule_55` using the new setup (with fewer dummy variables) (3 points)

- [ ] 1.121
- [x] 1.126
- [ ] 1.136
- [ ] 1.141

**Answer:** 1.126 (run lokal: 1.128 dengan 208 fitur, `pred10_rf_best_rule_55`; nilai terdekat dari opsi. Random Forest tanpa seed tetap menghasilkan sedikit variasi antar-run.)

### 2. What is the probability threshold for the second-best Random Forest decision rule, if you want to have the number of average trades per day BELOW 5? (3 points)

- [ ] 0.54
- [ ] 0.56
- [x] 0.57
- [ ] 0.58

**Answer:** 0.57 (`pred21_rf_best_rule_57`: avg. trades/day 4.63, CAGR 1.088. Threshold 0.56 memberi 5.10 trades/day, sehingga tidak masuk kriteria < 5.)

### 3. What is the number of trades for the best predictor `pred5_clf_10` (after redefining the target variable)? (3 points)

- [ ] 6
- [ ] 7
- [x] 8
- [ ] 9

**Answer:** 8 (`pred5_clf_10` setelah target diredefinisi `growth_future_30d > 1.022`: avg. trades/day 8.01, CAGR 1.122; target mean 0.497.)

### 4. [EXPLORATORY] Describe Your Ideal Trading Strategy (1 point)

My ideal trading strategy would use a machine learning model to identify potential trading opportunities based on historical market data. I would prioritize a strategy with consistent returns, a reasonable CAGR, and a limited number of trades per day to minimize transaction costs.

I would also use a probability threshold to filter out low-confidence predictions and only execute trades when the model indicates a sufficiently strong signal. In addition, I would apply proper risk management, including stop-loss limits, position sizing, and diversification, to reduce potential losses.

Finally, I would evaluate the strategy using historical backtesting and out-of-sample testing before applying it to real market conditions. My goal would be to achieve consistent long-term growth while maintaining an acceptable level of risk.