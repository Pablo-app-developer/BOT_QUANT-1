"""
statistical_tools.py — Herramientas estadísticas para investigación de edges.

Cada función es una herramienta de MEDICIÓN, no una estrategia.
Sirven para responder: "¿este efecto es real o es ruido?"

Herramientas:
  1. bootstrap_mean()    — Intervalo de confianza para la media (no paramétrico)
  2. ttest_1samp()       — t-test: ¿la media es significativamente ≠ 0?
  3. rolling_stability() — ¿el efecto se mantiene en el tiempo?
  4. ljung_box()         — ¿hay autocorrelación en una serie?
  5. regime_volatility() — Detectar regímenes de alta/baja volatilidad
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

logger = logging.getLogger(__name__)


def bootstrap_mean(
    x: np.ndarray,
    n_boot: int = 5_000,
    ci: float = 0.95,
    seed: int = 42,
    max_sample: int = 50_000,
) -> dict:
    """
    Bootstrap no-paramétrico para intervalo de confianza de la media.

    ¿Por qué bootstrap y no paramétrico?
    Los retornos financieros tienen fat tails → t-test asume normalidad.
    Bootstrap no asume distribución, es más honesto.

    Optimización: para arrays > max_sample, submuestreamos.
    Con N > 50K, el CLT garantiza convergencia — no perdemos poder.

    Returns: {'mean', 'ci_low', 'ci_high', 'ci_excludes_zero'}
    """
    rng = np.random.default_rng(seed)

    # Submuestrear si el array es muy grande (para velocidad)
    if len(x) > max_sample:
        idx = rng.choice(len(x), size=max_sample, replace=False)
        x_boot = x[idx]
    else:
        x_boot = x

    n = len(x_boot)
    means = np.empty(n_boot)

    for i in range(n_boot):
        sample = x_boot[rng.integers(0, n, size=n)]
        means[i] = sample.mean()

    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    m = float(np.mean(x))  # Media sobre datos COMPLETOS

    return {
        'mean': m,
        'ci_low': float(lo),
        'ci_high': float(hi),
        'ci_excludes_zero': bool(lo > 0 or hi < 0),
    }


def ttest_1samp(x: np.ndarray) -> dict:
    """
    One-sample t-test: ¿la media de x es significativamente ≠ 0?

    Útil como primera aproximación rápida. Pero cuidado:
    asume normalidad (que los retornos NO tienen).
    Por eso SIEMPRE se complementa con bootstrap.

    Returns: {'t_stat', 'p_value', 'significant_005', 'mean', 'std'}
    """
    t, p = stats.ttest_1samp(x, 0)
    return {
        't_stat': float(t),
        'p_value': float(p),
        'significant_005': bool(p < 0.05),
        'mean': float(np.mean(x)),
        'std': float(np.std(x, ddof=1)),
    }


def rolling_stability(
    x: np.ndarray,
    timestamps: pd.DatetimeIndex,
    window_months: int = 3,
) -> dict:
    """
    ¿El efecto se mantiene a lo largo del tiempo?

    Divide la serie en ventanas de N meses y calcula la media en cada una.
    Stabilidad = fracción de ventanas donde la media tiene el mismo signo
    que la media global.

    ¿Por qué importa?
    Un edge que solo funcionó en 2020 no es edge, es coincidencia.
    Queremos efectos que aparezcan CONSISTENTEMENTE en distintos períodos.

    Returns: {'stability', 'n_windows', 'window_means', 'window_labels'}
    """
    s = pd.Series(x, index=timestamps)
    global_sign = 1 if s.mean() > 0 else -1

    # Agrupar por períodos de N meses
    groups = s.groupby(pd.Grouper(freq=f'{window_months}MS'))
    window_means = []
    window_labels = []

    for label, group in groups:
        if len(group) < 100:  # Mínimo para estadística significativa
            continue
        window_means.append(float(group.mean()))
        window_labels.append(str(label))

    if not window_means:
        return {'stability': 0.0, 'n_windows': 0,
                'window_means': [], 'window_labels': []}

    wm = np.array(window_means)
    same_sign = np.sum(np.sign(wm) == global_sign)
    stability = same_sign / len(wm)

    return {
        'stability': float(stability),
        'n_windows': len(wm),
        'window_means': window_means,
        'window_labels': window_labels,
    }


def ljung_box(x: np.ndarray, lags: int = 10, max_sample: int = 200_000) -> dict:
    """
    Test Ljung-Box para autocorrelación.

    H0: no hay autocorrelación hasta lag K.
    Si p < 0.05 → hay dependencia serial → potencial edge.

    ¿Por qué es importante?
    Si los retornos son i.i.d., no hay nada que explotar.
    Autocorrelación = predictibilidad = posible edge.

    Nota: para arrays > max_sample, usamos subsample.
    Con 200K obs, Ljung-Box tiene poder estadístico más que suficiente.

    Returns: {'has_autocorrelation', 'min_p_value', 'lag_p_values'}
    """
    # Subsample para velocidad — estadísticamente válido
    if len(x) > max_sample:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(x), size=max_sample, replace=False))
        x_sub = x[idx]
    else:
        x_sub = x

    result = acorr_ljungbox(x_sub, lags=lags, return_df=True)
    p_values = result['lb_pvalue'].values

    return {
        'has_autocorrelation': bool(np.any(p_values < 0.05)),
        'min_p_value': float(np.min(p_values)),
        'lag_p_values': {int(i + 1): float(p) for i, p in enumerate(p_values)},
    }


def regime_volatility(
    returns: pd.Series,
    quantile_low: float = 0.25,
    quantile_high: float = 0.75,
) -> dict:
    """
    Detecta regímenes de volatilidad usando rolling std.

    Divide el mercado en 3 regímenes: baja, media, alta volatilidad.
    Calcula retorno medio en cada régimen.

    ¿Por qué?
    Si la volatilidad alta/baja tiene sesgo direccional,
    eso podría ser un edge explotable.

    Returns: {'regime_stats': {low/med/high: {mean, std, count}},
              'vol_direction_corr'}
    """
    # Rolling std 60 barras (1 hora)
    vol = returns.rolling(60, min_periods=30).std()
    vol = vol.dropna()
    rets_aligned = returns.loc[vol.index]

    q_lo = vol.quantile(quantile_low)
    q_hi = vol.quantile(quantile_high)

    low_mask = vol <= q_lo
    mid_mask = (vol > q_lo) & (vol <= q_hi)
    high_mask = vol > q_hi

    def _stats(mask):
        r = rets_aligned[mask]
        if len(r) < 100:
            return {'mean': 0.0, 'std': 0.0, 'count': 0}
        return {
            'mean': float(r.mean()),
            'std': float(r.std()),
            'count': int(len(r)),
        }

    # Correlación entre nivel de volatilidad y dirección
    corr = float(vol.corr(rets_aligned))

    return {
        'regime_stats': {
            'low': _stats(low_mask),
            'med': _stats(mid_mask),
            'high': _stats(high_mask),
        },
        'vol_direction_corr': corr,
    }
