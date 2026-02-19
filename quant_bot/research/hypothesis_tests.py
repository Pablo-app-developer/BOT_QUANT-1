"""
hypothesis_tests.py — Tests de hipótesis estructurados para descubrir edges.

Cada test sigue el PROTOCOLO CIENTÍFICO:
  1. HIPÓTESIS  → qué creemos que ocurre
  2. MEDICIÓN   → cómo lo medimos (retornos condicionales)
  3. SIGNIFICANCIA → p-value + bootstrap CI
  4. ESTABILIDAD   → rolling windows (¿se mantiene en el tiempo?)
  5. VEREDICTO     → ¿es viable o no?

REGLA FUNDAMENTAL: Si el efecto no es ESTABLE temporalmente,
no importa qué tan significativo sea — es ruido o régimen pasado.

Stability > Significance > Effect Size  (nuestro ranking)
"""

import logging
import numpy as np
import pandas as pd

from research.statistical_tools import (
    bootstrap_mean,
    ttest_1samp,
    rolling_stability,
    ljung_box,
    regime_volatility,
)

logger = logging.getLogger(__name__)


def _result(name: str, hypothesis: str, effect: dict, ttest: dict,
            bootstrap: dict, stability: dict, extra: dict = None) -> dict:
    """Construye resultado estandarizado."""
    r = {
        'name': name,
        'hypothesis': hypothesis,
        'effect_size': effect.get('mean', 0.0),
        'effect_size_bps': effect.get('mean', 0.0) * 10_000,  # basis points
        # Significancia
        'p_value': ttest['p_value'],
        'significant': ttest['significant_005'],
        't_stat': ttest['t_stat'],
        # Bootstrap
        'ci_low': bootstrap['ci_low'],
        'ci_high': bootstrap['ci_high'],
        'ci_excludes_zero': bootstrap['ci_excludes_zero'],
        # Estabilidad
        'stability': stability['stability'],
        'n_windows': stability['n_windows'],
    }
    if extra:
        r.update(extra)
    return r


# ═══════════════════════════════════════════════
# TEST 1: MOMENTUM CORTO
# ═══════════════════════════════════════════════

def test_momentum(data: pd.DataFrame, lookbacks: list[int] = None) -> list[dict]:
    """
    HIPÓTESIS: Si el precio subió en los últimos N minutos,
    tiende a seguir subiendo en los próximos M minutos.

    Base: Jegadeesh & Titman (1993) en equities.
    En FX M1 probablemente es débil o inexistente — eso es lo que medimos.

    Medición: retorno futuro condicional a retorno pasado positivo vs negativo.
    """
    if lookbacks is None:
        lookbacks = [5, 15, 30, 60]

    results = []
    close = data['close']
    hold = 15  # Período de holding fijo: 15 min

    for lb in lookbacks:
        past_ret = close.pct_change(lb)
        future_ret = close.pct_change(hold).shift(-hold)

        # Alinear y limpiar
        df = pd.DataFrame({'past': past_ret, 'future': future_ret}).dropna()

        # Retorno futuro cuando pasado > 0 (momentum = long)
        long_cond = df[df['past'] > 0]['future'].values
        short_cond = df[df['past'] < 0]['future'].values

        # El "edge" de momentum = retorno condicional a señal
        # Si momentum funciona: long_returns > 0, short_returns < 0
        # Medimos la diferencia: long - short (spread)
        spread = np.mean(long_cond) - np.mean(short_cond)
        # Pero para estadística, usamos los retornos condicionados directamente:
        # "¿el retorno medio tras past>0 es significativamente >0?"
        momentum_rets = np.concatenate([long_cond, -short_cond])

        tt = ttest_1samp(momentum_rets)
        bs = bootstrap_mean(momentum_rets)
        stab = rolling_stability(momentum_rets,
                                 df.index[:len(momentum_rets)])

        r = _result(
            name=f"momentum_lb{lb}",
            hypothesis=f"Momentum {lb}-min predice retorno {hold}-min",
            effect={'mean': float(np.mean(momentum_rets))},
            ttest=tt, bootstrap=bs, stability=stab,
            extra={'lookback': lb, 'hold': hold,
                   'spread_bps': spread * 10_000,
                   'n_long': len(long_cond), 'n_short': len(short_cond)},
        )
        results.append(r)
        logger.info(f"  Momentum lb={lb}: effect={r['effect_size_bps']:.2f}bps "
                    f"p={r['p_value']:.4f} stab={r['stability']:.2f}")

    return results


# ═══════════════════════════════════════════════
# TEST 2: MEAN REVERSION
# ═══════════════════════════════════════════════

def test_mean_reversion(data: pd.DataFrame, lookbacks: list[int] = None) -> list[dict]:
    """
    HIPÓTESIS: Tras un movimiento fuerte (>1σ), el precio tiende
    a revertir parcialmente.

    Base: Lo & MacKinlay (1988). Mean reversion documentada en FX.
    Ernest Chan argumenta que es más fuerte en FX que en equities.

    Medición: retorno futuro cuando pasado excede 1 std dev.
    Si mean-reversion funciona: retorno futuro tiene signo OPUESTO al pasado.
    """
    if lookbacks is None:
        lookbacks = [15, 30, 60]

    results = []
    close = data['close']
    hold = 15

    for lb in lookbacks:
        past_ret = close.pct_change(lb)
        future_ret = close.pct_change(hold).shift(-hold)

        df = pd.DataFrame({'past': past_ret, 'future': future_ret}).dropna()

        # Solo miramos movimientos "extremos" (>1σ)
        std = df['past'].std()
        big_up = df[df['past'] > std]
        big_down = df[df['past'] < -std]

        # Mean reversion = después de big_up, short; después de big_down, long
        # Retorno de reversion: -future después de big_up, +future después de big_down
        rev_up = -big_up['future'].values   # Short después de subida fuerte
        rev_down = big_down['future'].values  # Long después de caída fuerte
        rev_rets = np.concatenate([rev_up, rev_down])

        if len(rev_rets) < 100:
            logger.warning(f"  MeanRev lb={lb}: insuficientes trades ({len(rev_rets)})")
            continue

        tt = ttest_1samp(rev_rets)
        bs = bootstrap_mean(rev_rets)
        stab = rolling_stability(rev_rets,
                                 pd.DatetimeIndex(
                                     list(big_up.index) + list(big_down.index)
                                 ).sort_values()[:len(rev_rets)])

        r = _result(
            name=f"mean_rev_lb{lb}",
            hypothesis=f"Mean reversion tras movimiento >{lb}-min >1σ",
            effect={'mean': float(np.mean(rev_rets))},
            ttest=tt, bootstrap=bs, stability=stab,
            extra={'lookback': lb, 'hold': hold,
                   'n_up': len(rev_up), 'n_down': len(rev_down),
                   'threshold_std': 1.0},
        )
        results.append(r)
        logger.info(f"  MeanRev lb={lb}: effect={r['effect_size_bps']:.2f}bps "
                    f"p={r['p_value']:.4f} stab={r['stability']:.2f}")

    return results


# ═══════════════════════════════════════════════
# TEST 3: EXPANSIÓN TRAS COMPRESIÓN
# ═══════════════════════════════════════════════

def test_compression_expansion(data: pd.DataFrame) -> list[dict]:
    """
    HIPÓTESIS: Períodos de baja volatilidad (compresión) preceden
    movimientos de alta volatilidad (expansión).

    Base: Mandelbrot (1963), Bollinger. Volatility clustering.
    No predice DIRECCIÓN, solo MAGNITUD → necesitamos combinarlo
    con otra señal para explotar.

    Medición: ¿la volatilidad futura es mayor cuando la volatilidad
    actual es baja?
    """
    results = []
    close = data['close']
    rets = close.pct_change().dropna()

    for window in [30, 60, 120]:
        vol = rets.rolling(window).std()
        future_vol = rets.rolling(window).std().shift(-window)

        df = pd.DataFrame({'vol': vol, 'fvol': future_vol}).dropna()

        # Clasificar régimen actual
        q25 = df['vol'].quantile(0.25)
        q75 = df['vol'].quantile(0.75)

        compressed = df[df['vol'] <= q25]
        expanded = df[df['vol'] >= q75]

        # ¿Qué pasa después de compresión?
        vol_after_compress = compressed['fvol'].values
        vol_after_expand = expanded['fvol'].values

        # Ratio: vol futura / vol actual (en compresión)
        ratio = compressed['fvol'].values / (compressed['vol'].values + 1e-10)

        # Test: ¿el ratio es significativamente > 1?
        ratio_centered = ratio - 1.0  # Si >0 → expansión post-compresión
        
        if len(ratio_centered) < 100:
            continue

        tt = ttest_1samp(ratio_centered)
        bs = bootstrap_mean(ratio_centered)
        stab = rolling_stability(ratio_centered,
                                 compressed.index[:len(ratio_centered)])

        r = _result(
            name=f"compress_expand_w{window}",
            hypothesis=f"Volatilidad se expande tras compresión ({window}-min)",
            effect={'mean': float(np.mean(ratio_centered))},
            ttest=tt, bootstrap=bs, stability=stab,
            extra={'window': window,
                   'mean_ratio': float(np.mean(ratio)),
                   'n_compressed': len(compressed),
                   'n_expanded': len(expanded)},
        )
        results.append(r)
        logger.info(f"  Compress→Expand w={window}: ratio={np.mean(ratio):.2f} "
                    f"p={r['p_value']:.4f} stab={r['stability']:.2f}")

    return results


# ═══════════════════════════════════════════════
# TEST 4: EFECTO HORA DEL DÍA
# ═══════════════════════════════════════════════

def test_time_of_day(data: pd.DataFrame) -> list[dict]:
    """
    HIPÓTESIS: Hay horas del día con sesgo direccional en EURUSD.

    Base: Andersen & Bollerslev (1997), patrones intradía en FX.
    Los opens de sesiones (London 08:00, NY 13:00 UTC) suelen
    tener más actividad y potencialmente sesgo.

    Medición: retorno medio por hora del día.
    """
    results = []
    close = data['close']
    rets = close.pct_change().dropna()
    hours = rets.index.hour

    for h in range(24):
        mask = hours == h
        hour_rets = rets[mask].values

        if len(hour_rets) < 500:
            continue

        tt = ttest_1samp(hour_rets)
        bs = bootstrap_mean(hour_rets)
        stab = rolling_stability(hour_rets, rets.index[mask])

        r = _result(
            name=f"hour_{h:02d}",
            hypothesis=f"Sesgo direccional a las {h:02d}:00 UTC",
            effect={'mean': float(np.mean(hour_rets))},
            ttest=tt, bootstrap=bs, stability=stab,
            extra={'hour': h, 'n_obs': len(hour_rets),
                   'vol': float(np.std(hour_rets))},
        )
        results.append(r)

    # Reportar las horas más significativas
    sig_hours = [r for r in results if r['significant']]
    logger.info(f"  TimeOfDay: {len(sig_hours)}/{len(results)} horas "
                f"con p<0.05")
    for r in sorted(sig_hours, key=lambda x: x['p_value']):
        logger.info(f"    {r['name']}: {r['effect_size_bps']:.2f}bps "
                    f"p={r['p_value']:.4f} stab={r['stability']:.2f}")

    return results


# ═══════════════════════════════════════════════
# TEST 5: EFECTO SESIÓN
# ═══════════════════════════════════════════════

def test_session(data: pd.DataFrame) -> list[dict]:
    """
    HIPÓTESIS: Las sesiones de trading (Asia, London, NYC) tienen
    comportamientos direccionales distintos.

    Sesiones (UTC aprox):
      Asia:   00:00 - 08:00
      London: 08:00 - 16:00
      NYC:    13:00 - 21:00
    """
    results = []
    close = data['close']
    rets = close.pct_change().dropna()
    hours = rets.index.hour

    sessions = {
        'asia':   (0, 8),
        'london': (8, 16),
        'nyc':    (13, 21),
    }

    for name, (h_start, h_end) in sessions.items():
        mask = (hours >= h_start) & (hours < h_end)
        session_rets = rets[mask].values

        if len(session_rets) < 500:
            continue

        tt = ttest_1samp(session_rets)
        bs = bootstrap_mean(session_rets)
        stab = rolling_stability(session_rets, rets.index[mask])

        r = _result(
            name=f"session_{name}",
            hypothesis=f"Sesgo direccional en sesión {name.upper()}",
            effect={'mean': float(np.mean(session_rets))},
            ttest=tt, bootstrap=bs, stability=stab,
            extra={'session': name, 'hours': f"{h_start}-{h_end}",
                   'n_obs': len(session_rets),
                   'vol': float(np.std(session_rets))},
        )
        results.append(r)
        logger.info(f"  Session {name}: {r['effect_size_bps']:.2f}bps "
                    f"p={r['p_value']:.4f} stab={r['stability']:.2f}")

    return results


# ═══════════════════════════════════════════════
# TEST 6: AUTOCORRELACIÓN DE RETORNOS
# ═══════════════════════════════════════════════

def test_autocorrelation(data: pd.DataFrame) -> list[dict]:
    """
    HIPÓTESIS: Los retornos tienen dependencia serial
    (autocorrelación positiva = momentum, negativa = mean reversion).

    Base: Test Ljung-Box. Si rechazamos H0 → hay estructura explotable.
    """
    results = []
    close = data['close']

    for period in [1, 5, 15, 30]:
        rets = close.pct_change(period).dropna().values

        lb = ljung_box(rets, lags=10)

        # Calcular autocorrelación directa en lag 1
        acf_1 = float(np.corrcoef(rets[:-1], rets[1:])[0, 1])

        r = {
            'name': f"autocorr_{period}min",
            'hypothesis': f"Autocorrelación en retornos {period}-min",
            'effect_size': acf_1,
            'effect_size_bps': acf_1 * 10_000,
            'p_value': lb['min_p_value'],
            'significant': lb['has_autocorrelation'],
            't_stat': 0.0,
            'ci_low': 0.0, 'ci_high': 0.0,
            'ci_excludes_zero': False,
            'stability': 0.0,
            'n_windows': 0,
            'acf_lag1': acf_1,
            'ljung_box_pvalues': lb['lag_p_values'],
            'period': period,
        }

        # Calcular estabilidad del signo de ACF
        # Usamos chunks no-solapados de 5000 barras (rápido vs rolling)
        if acf_1 != 0:
            chunk_size = 5000
            n_chunks = len(rets) // chunk_size
            chunk_signs = []
            for c in range(n_chunks):
                chunk = rets[c * chunk_size:(c + 1) * chunk_size]
                if len(chunk) > 2:
                    acf_chunk = float(np.corrcoef(chunk[:-1], chunk[1:])[0, 1])
                    chunk_signs.append(np.sign(acf_chunk))
            if chunk_signs:
                r['stability'] = float(
                    np.mean(np.array(chunk_signs) == np.sign(acf_1))
                )
                r['n_windows'] = len(chunk_signs)

        results.append(r)
        logger.info(f"  Autocorr {period}min: ACF1={acf_1:.5f} "
                    f"LB_p={lb['min_p_value']:.4f} "
                    f"stab={r['stability']:.2f}")

    return results


# ═══════════════════════════════════════════════
# TEST 7: VOLATILIDAD VS DIRECCIÓN
# ═══════════════════════════════════════════════

def test_vol_direction(data: pd.DataFrame) -> list[dict]:
    """
    HIPÓTESIS: El nivel de volatilidad tiene correlación
    con la dirección del retorno futuro.

    Si volatilidad alta → sesgo bajista (risk-off) o alcista (breakout).
    """
    close = data['close']
    rets = close.pct_change().dropna()
    results = []

    rv = regime_volatility(rets)

    for regime, st in rv['regime_stats'].items():
        if st['count'] < 100:
            continue

        # Para cada régimen, ¿la media del retorno es ≠ 0?
        mask = None
        vol = rets.rolling(60).std().dropna()
        rets_al = rets.loc[vol.index]

        if regime == 'low':
            mask = vol <= vol.quantile(0.25)
        elif regime == 'high':
            mask = vol >= vol.quantile(0.75)
        else:
            mask = (vol > vol.quantile(0.25)) & (vol <= vol.quantile(0.75))

        regime_rets = rets_al[mask].values

        tt = ttest_1samp(regime_rets)
        bs = bootstrap_mean(regime_rets)
        stab = rolling_stability(regime_rets, rets_al.index[mask])

        r = _result(
            name=f"vol_dir_{regime}",
            hypothesis=f"Sesgo direccional en régimen de volatilidad {regime}",
            effect={'mean': st['mean']},
            ttest=tt, bootstrap=bs, stability=stab,
            extra={'regime': regime, 'vol_corr': rv['vol_direction_corr'],
                   'n_obs': st['count']},
        )
        results.append(r)
        logger.info(f"  VolDir {regime}: {r['effect_size_bps']:.2f}bps "
                    f"p={r['p_value']:.4f} stab={r['stability']:.2f}")

    return results


# ═══════════════════════════════════════════════
# TEST 8: CONTINUACIÓN TRAS VELA IMPULSIVA
# ═══════════════════════════════════════════════

def test_impulse_continuation(data: pd.DataFrame) -> list[dict]:
    """
    HIPÓTESIS: Tras una vela con cuerpo grande (>2σ), el precio
    continúa en la misma dirección (impulso sigue impulso).

    Contraria a mean-reversion. Medimos ambos para ver cuál domina.
    """
    results = []
    close = data['close']
    rets = close.pct_change().dropna()

    for hold in [5, 15, 30]:
        future = close.pct_change(hold).shift(-hold)
        df = pd.DataFrame({'ret': rets, 'future': future}).dropna()

        std = df['ret'].std()
        impulse_up = df[df['ret'] > 2 * std]
        impulse_down = df[df['ret'] < -2 * std]

        # Continuación = futuro tiene mismo signo que impulso
        cont_up = impulse_up['future'].values       # Esperamos >0
        cont_down = -impulse_down['future'].values   # Esperamos >0 (shorted)
        cont_rets = np.concatenate([cont_up, cont_down])

        if len(cont_rets) < 100:
            continue

        tt = ttest_1samp(cont_rets)
        bs = bootstrap_mean(cont_rets)
        idx = pd.DatetimeIndex(
            list(impulse_up.index) + list(impulse_down.index)
        ).sort_values()[:len(cont_rets)]
        stab = rolling_stability(cont_rets, idx)

        r = _result(
            name=f"impulse_cont_h{hold}",
            hypothesis=f"Continuación tras impulso >2σ, hold {hold}min",
            effect={'mean': float(np.mean(cont_rets))},
            ttest=tt, bootstrap=bs, stability=stab,
            extra={'hold': hold, 'threshold_std': 2.0,
                   'n_up': len(cont_up), 'n_down': len(cont_down)},
        )
        results.append(r)
        logger.info(f"  Impulse cont h={hold}: {r['effect_size_bps']:.2f}bps "
                    f"p={r['p_value']:.4f} stab={r['stability']:.2f}")

    return results


# ═══════════════════════════════════════════════
# ORQUESTADOR
# ═══════════════════════════════════════════════

def run_all(data: pd.DataFrame) -> list[dict]:
    """
    Ejecuta TODOS los tests de hipótesis.
    Retorna lista plana de resultados.
    """
    logger.info("=" * 55)
    logger.info("  EJECUTANDO TESTS DE HIPÓTESIS")
    logger.info("=" * 55)

    all_results = []

    logger.info("\n[1/8] Momentum...")
    all_results.extend(test_momentum(data))

    logger.info("\n[2/8] Mean Reversion...")
    all_results.extend(test_mean_reversion(data))

    logger.info("\n[3/8] Compresión → Expansión...")
    all_results.extend(test_compression_expansion(data))

    logger.info("\n[4/8] Efecto Hora del Día...")
    all_results.extend(test_time_of_day(data))

    logger.info("\n[5/8] Efecto Sesión...")
    all_results.extend(test_session(data))

    logger.info("\n[6/8] Autocorrelación...")
    all_results.extend(test_autocorrelation(data))

    logger.info("\n[7/8] Volatilidad vs Dirección...")
    all_results.extend(test_vol_direction(data))

    logger.info("\n[8/8] Continuación tras Impulso...")
    all_results.extend(test_impulse_continuation(data))

    logger.info(f"\nTotal tests ejecutados: {len(all_results)}")
    return all_results
