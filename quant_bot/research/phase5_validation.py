
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from research.statistical_tools import bootstrap_mean, ttest_1samp, rolling_stability

logger = logging.getLogger(__name__)

def validate_mean_reversion_conditions(data: pd.DataFrame) -> List[Dict]:
    """
    Fase 5: Validación profunda del Edge de Mean Reversion.
    
    Preguntas a responder:
    1. ¿Funciona mejor la reversión en alta o baja volatilidad?
    2. ¿Mejora si filtramos a favor de la tendencia M5? (Fade dips in uptrend)
    """
    results = []
    close = data['close']
    
    # ── Parametros Base ──
    LOOKBACK = 15   # 15 min impulso
    HOLD = 15       # 15 min holding
    
    # Pre-cálculos
    rets_m1 = close.pct_change().dropna()
    rolling_std = rets_m1.rolling(60).std()  # Volatilidad horaria
    
    # Definir tendencia M5 (simple moving average alignment)
    sma_fast = close.rolling(5 * 5).mean() 
    sma_slow = close.rolling(20 * 5).mean()
    trend_m5 = np.sign(sma_fast - sma_slow)
    
    # Definir Impulso (Signal)
    past_ret = close.pct_change(LOOKBACK)
    future_ret = close.pct_change(HOLD).shift(-HOLD)
    
    # Alineación
    df = pd.DataFrame({
        'past': past_ret,
        'future': future_ret,
        'vol': rolling_std,
        'trend_m5': trend_m5,
        'timestamp': close.index
    }).dropna()
    
    std_window = df['past'].std()
    
    # Clasificación de Volatilidad
    q66 = df['vol'].quantile(0.66)
    mask_high_vol = df['vol'] > q66
    
    # ── Iterate over thresholds ──
    for sigma in [2.0, 2.5, 3.0, 3.5]:
        threshold = sigma * std_window
        
        signal_long = df['past'] < -threshold
        signal_short = df['past'] > threshold
        
        # 1. Base Strategy
        rets_base = pd.Series(np.nan, index=df.index)
        rets_base[signal_long] = df.loc[signal_long, 'future']
        rets_base[signal_short] = -df.loc[signal_short, 'future']
        
        results.append(_test_variant(
            f"MeanRev_{sigma}s_Base", 
            rets_base.dropna(), 
            f"Reversión > {sigma}std"
        ))
        
        # 2. Trend + High Vol (The "Combo")
        # Valid Longs: Signal Long AND Trend Up AND High Vol
        valid_longs = signal_long & (df['trend_m5'] > 0) & mask_high_vol
        valid_shorts = signal_short & (df['trend_m5'] < 0) & mask_high_vol
        
        rets_combo = pd.Series(np.nan, index=df.index)
        rets_combo[valid_longs] = df.loc[valid_longs, 'future']
        rets_combo[valid_shorts] = -df.loc[valid_shorts, 'future']
        
        results.append(_test_variant(
            f"MeanRev_{sigma}s_Combo",
            rets_combo.dropna(),
            f"Trend M5 + High Vol + {sigma}std"
        ))

    return results

def _test_variant(name: str, returns: pd.Series, desc: str) -> Dict:
    """Helper para correr tests estadísticos sobre una variante."""
    if len(returns) < 100:
        return {
            'name': name,
            'description': desc,
            'valid': False,
            'reason': 'Insuficientes trades'
        }
        
    vals = returns.values
    tt = ttest_1samp(vals)
    bs = bootstrap_mean(vals)
    stab = rolling_stability(vals, returns.index)
    
    # Sharpe aproximado (anualizado asumiendo holding period efectivo)
    # Es solo indicativo.
    mean = np.mean(vals)
    std = np.std(vals)
    sharpe = (mean / std) * np.sqrt(252 * 1440 / 15) if std > 0 else 0
    
    return {
        'name': name,
        'description': desc,
        'valid': True,
        'n_trades': len(returns),
        'mean_bps': mean * 10000,
        'sharpe_proxy': sharpe,
        'win_rate': np.mean(vals > 0),
        'p_value': tt['p_value'],
        'ci_lower': bs['ci_low'] * 10000,
        'ci_upper': bs['ci_high'] * 10000,
        'stability': stab['stability']
    }

def print_validation_report(results: List[Dict]):
    logger.info("=" * 80)
    logger.info("  FASE 5: VALIDACIÓN DE ESTRATEGIAS CANDIDATAS")
    logger.info("=" * 80)
    
    headers = f"{'Nombre':<22} | {'Trades':<6} | {'Efct(bps)':<9} | {'Win%':<5} | {'Stab':<4} | {'p-val':<7} | {'Descripción'}"
    logger.info(headers)
    logger.info("-" * len(headers))
    
    # Ordenar por Sharpe/Effect
    valid_results = [r for r in results if r.get('valid', False)]
    valid_results.sort(key=lambda x: x['mean_bps'], reverse=True)
    
    for r in valid_results:
        logger.info(
            f"{r['name']:<22} | {r['n_trades']:<6} | "
            f"{r['mean_bps']:>9.2f} | {r['win_rate']:.2f} | {r['stability']:.2f} | "
            f"{r['p_value']:.4f}  | {r['description']}"
        )
    
    if not valid_results:
        logger.warning("No se encontraron variantes válidas.")
