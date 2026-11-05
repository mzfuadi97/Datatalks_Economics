## Module 5 Homework (2026 Cohort)

In this homework, you will propose **three distinct stock research ideas** that could become a capstone project or a future investment strategy.

The previous homeworks focused primarily on working with data, building models, and evaluating strategies. This assignment focuses on deciding **what is worth investigating and why**.

All three questions are exploratory and require free-text answers. No code, trained model, or completed backtest is required.

### Why this matters

As AI makes software development and machine learning more accessible, standard trading infrastructure becomes easier to reproduce. Someone can copy our shared code with a single `git clone` command.

Your contribution comes from the research: asking useful questions, understanding a market or industry, finding relevant data, and developing a testable investment thesis. The ability to generate, evaluate, and refine ideas quickly matters—but speed is useful only when paired with thoughtful reasoning.

A strong idea connects:

- **An opportunity:** a market, sector, structural change, or recurring behavior worth investigating.
- **A thesis:** why that opportunity might produce a predictable investment outcome.
- **Data:** observable evidence that could support or challenge the thesis.
- **A test:** a practical way to determine whether the idea adds value.

### What to include in each answer

Use the same structure for all three ideas:

1. **Market and scope.** Which country, exchange, sector, or group of stocks would you study? What is your intended investment horizon?
2. **Investment thesis.** What do you expect to happen, and why? Identify the relevant macroeconomic trend, industry development, or market mechanism.
3. **Potential edge.** Why might the opportunity be overlooked or slow to appear in prices? What knowledge, data, or interpretation could make your approach useful?
4. **Data and sources.** Identify specific datasets, indicators, or sources you would investigate. Explain what they measure and how they relate to your thesis. Distinguish sources you have checked from sources you still need to verify.
5. **Research approach.** Describe the signal, rule, or model you would test and the investment decision it would inform. Explain why this approach fits the idea.
6. **Validation and risks.** What simple benchmark would you compare against? What evidence would make you reject the idea? Consider data availability, publication delays, transaction costs, liquidity, and other relevant limitations.

Be specific. “Use AI to predict stock prices” does not explain an investment thesis. Describe **what information might predict which outcome, over what horizon, and why**.

### Examples of directions to explore

These are starting points for research, not established profitable strategies:

- **Local markets or smaller exchanges:** Investigate whether local-language disclosures, economic releases, or industry statistics provide useful signals for companies receiving limited analyst coverage. Explain why local knowledge could help and whether trading liquidity would support the strategy.
- **Pairs trading:** Study two economically related companies whose relative prices sometimes diverge. Explain why the relationship should persist, how you would identify a temporary divergence, and what would indicate that the relationship has broken down.
- **Mean reversion:** Investigate whether sharp price moves following a particular type of event tend to reverse. Explain how you would distinguish a temporary overreaction from a lasting change in the business.
- **Bioinformatics:** Explore whether research funding, public procurement, or adoption of genomic analysis tools could help explain future demand for listed companies. Identify measurable indicators and the connection to company revenue.
- **Advertising:** Study whether changes in advertising activity, customer demand, or platform usage could help anticipate revenue changes at advertising businesses.
- **Gaming:** Investigate whether player activity, reviews, release schedules, or audience engagement could help anticipate company performance—and whether those signals contain information beyond what investors already expect.

A growing industry does not automatically imply attractive stock returns. Explain **what your proposed signal could reveal beyond the growth story itself**.

### Make your ideas meaningfully different

Your three submissions should explore different hypotheses. Changing only the ticker, technical indicator, or machine learning algorithm is not enough.

Aim for differences in the underlying economic mechanism, market, data source, or investment horizon. For example, you might combine a local-market information idea, a relative-value strategy, and a sector-specific demand signal.

You may use AI to brainstorm or challenge your reasoning. Your final answers should reflect your own judgment: verify factual claims, explain assumptions, and avoid generic lists of fashionable themes.

---

### Question 1: [Exploratory] Research Idea Number One

**Propose one well-reasoned stock research idea using the structure above.**

**Answer:** 
1. **Market and scope.** Cryptocurrencies (specifically BTC) vs. Traditional Assets (S&P 500, Gold). Investment horizon: Short-to-medium term (days to weeks).
2. **Investment thesis.** Cryptocurrencies often behave as high-beta assets that correlate strongly with tech-heavy indices (like the S&P 500) during bullish cycles but show divergent, often reactive, behavior during macroeconomic shocks.
3. **Potential edge.** Many market participants treat cryptocurrencies as isolated assets; exploiting cross-asset linkages during market stress can reveal predictive behavior that is ignored by crypto-exclusive models.
4. **Data and sources.** Historical OHLCV data for BTC, S&P 500 (SPY), and Gold (GLD) via `yfinance`.
5. **Research approach.** Train a Random Forest classifier to predict the direction of BTC price movements using lagged growth indicators of SPY and GLD.
6. **Validation and risks.** Benchmark against a simple trend-following strategy for BTC. Risks: Crypto-specific events (e.g., regulatory changes, exchange failures) may decouple the correlation unpredictably.

---

### Question 2: [Exploratory] Research Idea Number Two

**Propose one well-reasoned stock research idea using the structure above.**

**Answer:** 
1. **Market and scope.** Energy Sector (XLE) within the S&P 500. Investment horizon: Short-term (7-30 days).
2. **Investment thesis.** Stocks within the same sector exhibit momentum clustering due to thematic capital flows (e.g., energy price trends), which creates short-term price inefficiencies.
3. **Potential edge.** Traditional models often focus on broad market indices; focusing on sector-specific momentum is more reactive to sector-specific catalysts.
4. **Data and sources.** OHLCV data for sector-specific ETFs (XLE) and their constituent stocks via `yfinance`.
5. **Research approach.** Develop a machine learning classifier to predict whether a stock will outperform its sector index over the next 30 days based on recent relative growth and volume.
6. **Validation and risks.** Compare performance against a simple buy-and-hold strategy for the sector ETF. Risks: Over-concentration in a highly volatile sector.

---

### Question 3: [Exploratory] Research Idea Number Three

**Propose one well-reasoned stock research idea using the structure above.**

**Answer:** 
1. **Market and scope.** Energy-intensive industrial stocks vs. Oil (WTI). Investment horizon: Medium-term (30-90 days).
2. **Investment thesis.** Companies with high energy intensity in their production costs exhibit predictable, lagged reactions to sustained changes in oil prices.
3. **Potential edge.** Market reaction to commodity price changes is often inefficient, creating a delayed trade opportunity that can be captured by models looking for lagged correlations.
4. **Data and sources.** Historical oil prices (WTI) and price data for energy-intensive industrial stocks via `yfinance`.
5. **Research approach.** Use lagged oil price growth (30d, 90d) as a feature in a Random Forest model to predict the probability of positive future growth for industrial stocks.
6. **Validation and risks.** Compare performance against broad industrial sector performance. Risks: Changes in corporate hedging practices could neutralize the impact of oil price changes on firm growth.

---

**Before submitting:** Check that each idea has a clear thesis, identifiable data, a feasible first test, and a reason it might fail. Your goal is to propose three worthwhile investigations; you do not need to claim that they will be profitable.

## Submitting the solutions

Form for submitting: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw05

---

## Leaderboard

Leaderboard link: https://courses.datatalks.club/sma-zoomcamp-2026/leaderboard

---