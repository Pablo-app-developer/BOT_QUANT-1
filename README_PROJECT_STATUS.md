
# Quant Bot Project Status

## Current Phase: Phase 6 Completed (Strategy Simulation)

### Validated Strategy: "SnapBack M5" (Mean Reversion)
*   **Logic:** Fade extreme volatility moves (>3.5 Sigma) aligned with M5 Trend.
*   **Edge:** +4.86 bps (Gross).
*   **Challenge:** Execution Gap (1.7 pips) requires sub-second execution.

### Next Steps
1.  **Refine Execution:** Move `SnapBack M5` into a live trading environment.
2.  **Deploy:** Set up a live forward test on a Demo account to verify fills at 3.5 Sigma.
3.  **Monitor:** Check slippage vs theoretical edge.

### How to Run Simulation
```bash
cd quant_bot
python main_research.py --phase 6
```

### Key Files
*   `research/phase5_validation.py`: Statistical validation logic.
*   `strategies/snapback_m5.py`: Signal generation logic.
*   `backtest/engine.py`: Bar-by-bar simulation engine.
*   `workflows/final_strategy_report.md`: Detailed findings.

### Notes
The system is statistically robust but sensitive to execution latency. Recommend using Limit Orders or Algo Execution at Close.
