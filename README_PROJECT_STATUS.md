
# Quant Bot Project Status: CLOSED (TRANSITION TO QUANT-ML)

## Current Phase: Project Audit Completed -> **PIVOT CONFIRMED**

### Final Verdict: LINEAR LOGIC FAILED
After rigorous testing:
1.  **Phase 4 (Redux):** Proved simple edges (MA Cross, Breakout) fail on EURUSD/BTC/NQ data (2016-2026).
2.  **Phase 7 (Audit):** Exposed H4 Trend as a "Bull Market Illusion" (failed in 2022).

### Strategic Decision: QUANT ML (Same Science, New Hypothesis)
We are closing `BOT_QUANT-1` to launch `BOT_QUANT-ML`.
**The Scientific Method Remains:**
- **Hypothesis Generator:** Changed from "Human Indicators" to "Machine Learning Models" (XGBoost/Transformers).
- **Validation (Gatekeeper):** The Phase 6 Framework (Walk-Forward, Regime Filter, Cost Stress) **REMAINS THE LAW.**
- **Goal:** Find *Non-Linear* patterns invisible to simple indicators.

### Validated Strategy: NONE (Linear)
- **SnapBack M5:** DEPRECATED (Data Artifact).
- **H4 Momentum:** FAILED (Correlation Trap).

### Recommendation
**ARCHIVE THIS REPO.** It contains the "Negative Proof".
Use the `research/` validation scripts as the **AUDITOR** for the new ML models.
**DO NOT TRADELIST** any strategy from this codebase.

### Repository Structure
- `backtest/`: Execution Engine (Valid but unused).
- `strategies/`: Deprecated Logic.
- `research/`: **CORE VALUE.** Contains VectorBT scripts for rapid hypothesis testing.
- `data/`: Processed market data (Warning: 2025 is corrupt).
