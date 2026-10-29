---
title: "Trading Strategy & Decision Rules Analysis"
author: "Student"
date: "`r Sys.Date()`"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk_col$set(echo = TRUE)
```

## Question 1

**Find the CAGR for the `pred10_rf_best_rule_55` using the new setup (with fewer dummy variables)** *(3 points)*

### Options:
- [ ] 1.121
- [ ] 1.126
- [ ] 1.136
- [ ] 1.141

```{r q1-cagr-calculation, eval=FALSE}
# R Code Template to verify/calculate CAGR
# Replace 'data' and 'returns' with your actual variables

# cagr <- (final_value / initial_value)^(1 / num_years) - 1
# Print calculated CAGR to select the correct option:
# cagr
```

**Selected Answer:** *[Select one option based on your calculations]*

---

## Question 2

**What is the probability threshold for the second-best Random Forest decision rule, if you want to have the number of average trades per day BELOW 5?** *(3 points)*

### Options:
- [ ] 0.54
- [ ] 0.56
- [ ] 0.57
- [ ] 0.58

```{r q2-threshold-analysis, eval=FALSE}
# R Code Template to check average trades per day at different thresholds
# thresholds <- c(0.54, 0.56, 0.57, 0.58)

# Check average trades per day across thresholds:
# sapply(thresholds, function(th) {
#   sum(predictions >= th) / total_days
# })
```

**Selected Answer:** *[Select one option based on your threshold analysis]*

---

## Question 3

**What is the number of trades for the best predictor `pred5_clf_10` (after redefining the target variable)?** *(3 points)*

### Options:
- [ ] 6
- [ ] 7
- [ ] 8
- [ ] 9

```{r q3-trade-count, eval=FALSE}
# R Code Template to count total trades for pred5_clf_10
# sum(pred5_clf_10_signals == 1)
```

**Selected Answer:** *[Select one option based on your results]*

---

## Question 4 [EXPLORATORY]

**Describe Your Ideal Trading Strategy** *(1 point)*

### Response:

My ideal trading strategy focuses on a balanced approach prioritizing risk management, robust signal filtering, and adaptability across market regimes:

1. **Strategy Foundations:**
   - **Data Driven & Systematic:** Combines momentum indicators with machine learning classifiers (e.g., Random Forest/Gradient Boosting) to identify high-probability trade setups.
   - **Feature Selection:** Focuses on concise, highly predictive features to prevent overfitting (e.g., avoiding unnecessary dummy variables).

2. **Risk Management & Position Sizing:**
   - Strict Stop-Loss and Take-Profit rules based on Average True Range (ATR).
   - Dynamic position sizing using the Kelly Criterion or fixed fractional risk per trade (e.g., maximum 1-2% account equity at risk per trade).

3. **Execution & Trade Frequency:**
   - Controlled trade frequency to keep transaction costs and slippage low (aiming for highly targeted trades rather than high-frequency noise).
   - Probability threshold tuning to ensure high conviction before entering a trade.