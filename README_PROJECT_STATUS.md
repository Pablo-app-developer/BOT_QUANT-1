
# Quant Bot Project Status: CLOSED (NO EDGE FOUND)

## Current Phase: Project Audit Completed

### Final Verdict: MARKET EFFICIENT
After rigorous testing in Phase 4 (Redux) and Phase 6 (Audit), the project has concluded that there is **NO EXPLOITABLE RETAIL EDGE** in the EURUSD M1-H4 market (2016-2024).

### Key Findings
1.  **Mean Reversion (M1):** The +4.86 bps edge was an artifact of corrupt 2025 data. In normal years (2016-2024), it loses money consistently.
2.  **Trend Following (H1/H4):** Moving Average and Donchian strategies failed to produce a Sharpe Ratio > 0.0 (mostly negative).
3.  **Volatility Breakout (M15):** Failed due to false breakouts and high intraday noise.
4.  **Regime Analysis:** Filtering by Volatility (Low/High) did not improve performance enough to overcome spread costs.

### Validated Strategy: NONE
- **SnapBack M5:** DEPRECATED (Data Artifact).
- **H4 Momentum:** FAILED.

### Recommendation
**DO NOT DEPLOY THIS BOT.**
The codebase serves as a pristine example of a **Negative Result Audit**. Use the `research/` scripts (`vbt_runner.py`, `vbt_strat_*.py`) as a template for testing future assets (Crypto, Stocks) but **do not trade EURUSD with this logic.**

### Repository Structure
- `backtest/`: Execution Engine (Valid but unused).
- `strategies/`: Deprecated Logic.
- `research/`: **CORE VALUE.** Contains VectorBT scripts for rapid hypothesis testing.
- `data/`: Processed market data (Warning: 2025 is corrupt).
