"""
nq_signal_monitor.py — Generador de Señales Diario para Paper Trading H3v2

PROPÓSITO:
  Script que se ejecuta CADA DÍA al cierre de la sesión NY (~20:05 UTC)
  y determina si mañana hay señal H3v2 activa.

  Reglas H3v2:
    CONDICIÓN 1: El día de HOY cerró en negativo respecto a apertura (>0.1%)
    CONDICIÓN 2: Mañana, si la primera hora (13:30-14:30 UTC) retorna >|0.3%|,
                 seguir la dirección de esa primera hora hasta el cierre (20:00 UTC)

  OUTPUT DIARIO:
    - Estado del filtro para mañana (¿día previo bajista?)
    - Señal provisional (solo se confirma después de la primera hora de mañana)
    - Estadísticas de trades activos en OOS

USO:
  python3 nq_signal_monitor.py                  # análisis completo del día actual
  python3 nq_signal_monitor.py --signal-only     # solo muestra la señal de mañana
  python3 nq_signal_monitor.py --history         # historial de todos los trades OOS

PAPER TRADING:
  1. Ejecutar este script al cierre NY (~20:10 UTC)
  2. Si FILTER_ACTIVE=True → poner alarma a las 14:25 UTC del día siguiente
  3. A las 14:29 UTC: calcular retorno 1H (precio actual vs 13:30 open)
  4. Si |ret_1h| > 0.3%: entrar en la dirección
  5. Exit: cierre de sesión NY (~20:00 UTC)
  6. Registrar resultado en este script
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone, date, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "quant_bot" / "research" / "artifacts" / "nq"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRADES_LOG = ARTIFACTS_DIR / "paper_trades_h3v2.json"
SIGNAL_LOG  = ARTIFACTS_DIR / "daily_signals.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SignalMonitor")

DUKASCOPY_SCALE = 82.0
COST_TARGET_PTS = 2.0   # pts NQ en Interactive Brokers (futuros MNQ)
THR_PRIOR_DAY   = -0.001  # día previo debe haber bajado > 0.1%
THR_FIRST_HOUR  = 0.003   # primera hora debe moverse > 0.3%


# ════════════════════════════════════════════════════
# CARGA DE DATOS Y SEÑAL DEL DÍA ACTUAL
# ════════════════════════════════════════════════════

def load_latest_data() -> pd.DataFrame:
    """Carga el dataset M1 y calcula retornos diarios recientes."""
    parquet = PROJECT_ROOT / "quant_bot" / "data" / "processed" / "USATECHIDXUSD_M1.parquet"
    df = pd.read_parquet(parquet, engine='pyarrow')

    if 'session' not in df.columns:
        from quant_bot.data.nq_loader import add_session_labels
        df = add_session_labels(df)

    return df


def compute_daily_summary(df: pd.DataFrame, n_days: int = 60) -> pd.DataFrame:
    """
    Construye resumen diario de los últimos N días:
      - open_price (apertura NY)
      - close_price (cierre NY)
      - day_return  (retorno de apertura a cierre)
      - first_hour_ret (retorno primera hora)
    """
    ny = df[df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])].copy()
    # Solo últimos N días
    cutoff = ny.index[-1] - pd.Timedelta(days=n_days)
    ny = ny[ny.index >= cutoff]

    records = []
    for date_key, group in ny.groupby(ny.index.date):
        oh   = group[group['session'] == 'OPEN_HOUR']
        post = group[group['session'].isin(['MIDDAY', 'CLOSE_HOUR'])]
        if len(oh) < 30 or len(post) < 10:
            continue

        open_ny   = oh['open'].iloc[0]
        close_1h  = oh['close'].iloc[-1]
        close_eod = post['close'].iloc[-1]

        if open_ny <= 0:
            continue

        first_hour_ret = (close_1h - open_ny) / open_ny
        day_return     = (close_eod - open_ny) / open_ny

        oh_atr = (oh['high'].max() - oh['low'].min()) / open_ny
        directionality = abs(first_hour_ret) / oh_atr if oh_atr > 0 else 0

        records.append({
            'date':           pd.Timestamp(date_key, tz='UTC'),
            'open_ny':        float(open_ny),
            'close_1h':       float(close_1h),
            'close_eod':      float(close_eod),
            'first_hour_ret': float(first_hour_ret),
            'day_return':     float(day_return),
            'oh_atr':         float(oh_atr),
            'directionality': float(directionality),
        })

    df_daily = pd.DataFrame(records).set_index('date').sort_index()
    df_daily['prior_day_ret'] = df_daily['day_return'].shift(1)
    return df_daily


def evaluate_today_filter(df_daily: pd.DataFrame) -> dict:
    """
    Evalúa si HOY activa el filtro para mañana.
    Retorna el estado del filtro y la señal esperada.
    """
    if df_daily.empty:
        return {'filter_active': False, 'reason': 'Sin datos'}

    today = df_daily.iloc[-1]
    today_date = df_daily.index[-1].date()

    today_return = float(today['day_return'])
    today_open   = float(today['open_ny'])
    today_close  = float(today['close_eod'])
    today_1h_ret = float(today['first_hour_ret'])

    # ¿Activó el filtro para MAÑANA?
    filter_active = today_return < THR_PRIOR_DAY

    return {
        'evaluation_date':  str(datetime.now(timezone.utc).date()),
        'last_trading_day': str(today_date),
        'today_open':       today_open,
        'today_close':      today_close,
        'today_return':     today_return,
        'today_return_pct': round(today_return * 100, 3),
        'filter_active':    bool(filter_active),
        'filter_condition': f"Día previo retorno: {today_return*100:.3f}%  (umbral: <{THR_PRIOR_DAY*100:.1f}%)",
        'signal_for_tomorrow': {
            'status':      'ALERT_SET' if filter_active else 'NO_TRADE',
            'action':      'Poner alarma a las 14:25 UTC. Si |1H ret| > 0.3%, entrar en esa dirección.' if filter_active else 'Sin trade mañana.',
            'entry_time':  '14:29 UTC (cierre OPEN_HOUR)' if filter_active else 'N/A',
            'exit_time':   '20:00 UTC (cierre sesión NY)' if filter_active else 'N/A',
            'cost_rt_pts': COST_TARGET_PTS,
            'threshold_1h': THR_FIRST_HOUR * 100,
        }
    }


# ════════════════════════════════════════════════════
# GESTOR DE PAPER TRADES
# ════════════════════════════════════════════════════

def load_paper_trades() -> list:
    """Carga el historial de trades de paper trading."""
    if TRADES_LOG.exists():
        with open(TRADES_LOG) as f:
            return json.load(f)
    return []


def save_paper_trades(trades: list) -> None:
    with open(TRADES_LOG, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


def add_paper_trade(date_str: str, direction: str, entry: float,
                    exit_: float, first_hour_ret: float) -> dict:
    """
    Registra un nuevo trade de paper trading.
    direction: 'LONG' o 'SHORT'
    """
    sign = 1 if direction == 'LONG' else -1
    ret_gross = sign * (exit_ - entry) / entry
    cost_pct  = (COST_TARGET_PTS / DUKASCOPY_SCALE) / entry
    ret_net   = ret_gross - cost_pct

    trade = {
        'date':           date_str,
        'direction':      direction,
        'entry_price':    float(entry),
        'exit_price':     float(exit_),
        'first_hour_ret': float(first_hour_ret),
        'ret_gross':      float(ret_gross),
        'cost_pct':       float(cost_pct),
        'ret_net':        float(ret_net),
        'win':            bool(ret_net > 0),
        'logged_at':      datetime.now(timezone.utc).isoformat(),
    }

    trades = load_paper_trades()
    trades.append(trade)
    save_paper_trades(trades)
    logger.info(f"  Trade registrado: {date_str} {direction} Ret={ret_net*100:.3f}%")
    return trade


def show_paper_history() -> None:
    """Muestra estadísticas del paper trading OOS."""
    trades = load_paper_trades()

    if not trades:
        logger.info("  Sin trades registrados todavía.")
        return

    rets = np.array([t['ret_net'] for t in trades])
    wins = np.array([t['win'] for t in trades])

    logger.info("\n" + "═"*60)
    logger.info("  HISTORIAL PAPER TRADING H3v2")
    logger.info("═"*60)
    logger.info(f"  Total trades:     {len(trades)}")
    logger.info(f"  Win Rate:         {wins.mean()*100:.1f}%")
    logger.info(f"  Retorno total:    {rets.sum()*100:.2f}%")
    logger.info(f"  Retorno medio:    {rets.mean()*100:.4f}% por trade")

    if rets.std() > 0:
        sh = (rets.mean() / rets.std()) * np.sqrt(252)
        logger.info(f"  Sharpe anual.:    {sh:.3f}")

    eq = np.cumprod(1 + rets)
    pk = np.maximum.accumulate(eq)
    dd = ((eq - pk) / pk).min()
    logger.info(f"  Max Drawdown:     {dd*100:.2f}%")
    logger.info(f"  Equity actual:    {(eq[-1]-1)*100:.2f}%")
    logger.info(f"  Último trade:     {trades[-1]['date']} {trades[-1]['direction']}"
                f" → {'✅ WIN' if trades[-1]['win'] else '❌ LOSS'}")

    # Últimos 5 trades
    logger.info("\n  Últimos trades:")
    for t in trades[-5:]:
        icon = "✅" if t['win'] else "❌"
        logger.info(f"  {icon} {t['date']}  {t['direction']:5s}  "
                    f"entry={t['entry_price']:.2f}  exit={t['exit_price']:.2f}  "
                    f"ret={t['ret_net']*100:.3f}%")


# ════════════════════════════════════════════════════
# VALIDACIÓN OOS INCREMENTAL (Backtest de datos en período OOS)
# ════════════════════════════════════════════════════

def oos_incremental_backtest(df_daily: pd.DataFrame,
                              oos_start: str = '2025-01-01') -> dict:
    """
    Simula cómo habrían evolucionado los trades H3v2 en el período OOS,
    asumiendo que habríamos seguido las señales diariamente.
    Muestra el equity OOS actualizado con datos reales.
    """
    from scipy import stats as scipy_stats

    oos_start_ts = pd.Timestamp(oos_start, tz='UTC')
    df_oos = df_daily[df_daily.index >= oos_start_ts].copy()
    df_oos['prior_day_ret'] = df_oos['day_return'].shift(1)

    records = []
    for i, (date_key, row) in enumerate(df_oos.iterrows()):
        if pd.isna(row['prior_day_ret']):
            continue
        prior_bearish = row['prior_day_ret'] < THR_PRIOR_DAY
        if not prior_bearish:
            records.append({'date': date_key, 'traded': False, 'ret': 0.0})
            continue

        first_hour_ret = row['first_hour_ret']
        if abs(first_hour_ret) < THR_FIRST_HOUR:
            records.append({'date': date_key, 'traded': False, 'ret': 0.0})
            continue

        # Señal activa
        signal = np.sign(first_hour_ret)
        # Retorno: desde close_1h (14:29) hasta close_eod (20:00)
        entry_p = row['close_1h']
        exit_p  = row['close_eod']
        if entry_p <= 0:
            continue
        ret_gross = signal * (exit_p - entry_p) / entry_p
        cost_pct  = (COST_TARGET_PTS / DUKASCOPY_SCALE) / entry_p
        ret_net   = ret_gross - cost_pct

        records.append({
            'date': date_key, 'traded': True,
            'signal': float(signal), 'entry': float(entry_p),
            'exit': float(exit_p), 'ret': float(ret_net),
            'first_hour_ret': float(first_hour_ret),
            'prior_day_ret': float(row['prior_day_ret']),
        })

    df_rec = pd.DataFrame(records).set_index('date')
    traded = df_rec[df_rec['traded']]

    if traded.empty:
        return {}

    rets = traded['ret'].values
    eq   = np.cumprod(1 + rets)
    pk   = np.maximum.accumulate(eq)
    dd   = ((eq - pk) / pk).min()
    wr   = (rets > 0).mean()
    ann  = eq[-1] ** (252 / len(rets)) - 1
    sh   = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    _, p = scipy_stats.ttest_1samp(rets, 0)

    return {
        'n': int(len(traded)),
        'total_days_oos': int(len(df_rec)),
        'trade_rate': float(len(traded) / len(df_rec)),
        'wr': float(wr),
        'sharpe': float(sh),
        'annual': float(ann),
        'max_dd': float(dd),
        'pvalue': float(p),
        'equity': eq.tolist(),
        'dates':  [str(d.date()) for d in traded.index],
        'rets':   rets.tolist(),
    }


# ════════════════════════════════════════════════════
# VISUALIZACIÓN DEL MONITOR
# ════════════════════════════════════════════════════

def plot_monitor(oos_stats: dict, df_daily: pd.DataFrame, signal_today: dict) -> None:
    """Dashboard visual del estado del paper trading."""
    fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    GOLD='#FFD700'; GREEN='#00FF88'; RED='#FF4444'; BLUE='#4488FF'; GRAY='#888888'; BG='#161b22'

    def ax_style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=GOLD, fontsize=10, fontweight='bold', pad=8)
        ax.tick_params(colors=GRAY)
        ax.spines[:].set_color('#333333')
        for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_color(GRAY)

    # 1. Equity OOS
    ax1 = fig.add_subplot(gs[0, :2])
    if oos_stats.get('equity'):
        eq = np.array(oos_stats['equity'])
        col = GREEN if eq[-1] > 1.0 else RED
        ax1.plot(eq, color=col, lw=2.5)
        ax1.axhline(1.0, color=GRAY, lw=0.8, ls='--')
        ax1.fill_between(range(len(eq)), 1.0, eq, alpha=0.2, color=col)

        # Paper trades encima
        paper = load_paper_trades()
        if paper:
            ax1.axvline(len(eq) - len(paper), color=GOLD, lw=1.5, ls=':',
                        label='Inicio paper')
            ax1.legend(facecolor=BG, labelcolor='white', fontsize=9)

    sh  = oos_stats.get('sharpe', 0)
    ann = oos_stats.get('annual', 0)
    n   = oos_stats.get('n', 0)
    ax_style(ax1, f"OOS H3v2 — Equity  (n={n}  Sharpe={sh:.2f}  Ann={ann*100:.1f}%)")
    ax1.set_ylabel("Equity (base=1.0)", color=GRAY)

    # 2. Señal de mañana (panel grande)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(BG)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.axis('off')
    filter_active = signal_today.get('filter_active', False)
    color_box = GREEN if filter_active else '#444444'
    ax2.add_patch(plt.Rectangle((0.05, 0.35), 0.9, 0.55, color=color_box, alpha=0.4))

    status = "🔔 FILTRO ACTIVO" if filter_active else "⏸️  SIN TRADE"
    ax2.text(0.5, 0.82, status, ha='center', va='center', color=GOLD if filter_active else GRAY,
             fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.68, f"Día previo: {signal_today.get('today_return_pct', 0):+.2f}%",
             ha='center', va='center', color='white', fontsize=11)

    if filter_active:
        ax2.text(0.5, 0.54, "MAÑANA:", ha='center', color=GOLD, fontsize=10, fontweight='bold')
        ax2.text(0.5, 0.44, "14:29 UTC → Mide retorno 1H", ha='center', color='white', fontsize=9)
        ax2.text(0.5, 0.36, "|ret| > 0.3% → Entrar en esa dirección", ha='center', color=GREEN, fontsize=9)

    ax2.text(0.5, 0.2, f"Último día de datos:", ha='center', color=GRAY, fontsize=8)
    ax2.text(0.5, 0.12, signal_today.get('last_trading_day', ''), ha='center', color='white', fontsize=9)
    ax2.set_title("SEÑAL PRÓXIMO DÍA", color=GOLD, fontsize=10, fontweight='bold', pad=8)

    # 3. Retornos diarios OOS (barras)
    ax3 = fig.add_subplot(gs[1, :2])
    if oos_stats.get('rets') and oos_stats.get('dates'):
        rets  = np.array(oos_stats['rets'])
        dates = oos_stats['dates']
        colors3 = [GREEN if r > 0 else RED for r in rets]
        ax3.bar(range(len(rets)), rets*100, color=colors3, width=0.8, alpha=0.85)
        ax3.axhline(0, color=GRAY, lw=0.8)
        # Labels cada 5
        if len(dates) > 0:
            step = max(1, len(dates) // 10)
            ax3.set_xticks(range(0, len(dates), step))
            ax3.set_xticklabels([dates[i][:7] for i in range(0, len(dates), step)],
                                 rotation=35, fontsize=7)
        cumulative = f"Acumulado: {rets.sum()*100:.2f}%  WR: {(rets>0).mean()*100:.0f}%"
        ax3.set_title(f"Retornos por Trade OOS — {cumulative}",
                      color=GOLD, fontsize=10, fontweight='bold', pad=8)
    ax3.set_facecolor(BG); ax3.tick_params(colors=GRAY)
    ax3.spines[:].set_color('#333333')
    for l in ax3.get_xticklabels() + ax3.get_yticklabels(): l.set_color(GRAY)
    ax3.set_ylabel("Ret. neto (%)", color=GRAY)

    # 4. Stats panel
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(BG); ax4.axis('off')
    stats_text = [
        ("ESTADÍSTICAS OOS", GOLD, 14),
        ("", GRAY, 8),
        (f"N trades:     {oos_stats.get('n', 0)}", 'white', 10),
        (f"Win Rate:     {oos_stats.get('wr', 0)*100:.1f}%", 'white', 10),
        (f"Sharpe:       {oos_stats.get('sharpe', 0):.3f}", 'white', 10),
        (f"Annual Ret:   {oos_stats.get('annual', 0)*100:.1f}%", 'white', 10),
        (f"Max DD:       {oos_stats.get('max_dd', 0)*100:.1f}%", 'white', 10),
        (f"p-value:      {oos_stats.get('pvalue', 1):.4f}", 'white', 10),
        ("", GRAY, 8),
        (f"Trade Rate:   {oos_stats.get('trade_rate', 0)*100:.1f}% de días", GRAY, 9),
        (f"Costo RT:     {COST_TARGET_PTS} pts NQ", GRAY, 9),
        ("", GRAY, 8),
        ("OBJETIVO:", GOLD, 10),
        ("n≥116 para p<0.05 (50% poder)", GRAY, 8),
        ("n≥265 para p<0.05 (80% poder)", GRAY, 8),
    ]
    y = 0.95
    for text, color, size in stats_text:
        ax4.text(0.05, y, text, transform=ax4.transAxes, color=color,
                 fontsize=size, va='top')
        y -= 0.067

    fig.suptitle(
        f"H3v2 MONITOR — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Señal mañana: {'🔔 ACTIVA' if filter_active else '⏸️ INACTIVA'}",
        color='white', fontsize=13, fontweight='bold', y=0.99
    )

    out = ARTIFACTS_DIR / "nq_h3v2_monitor.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"\n  ✅ Dashboard: {out}")


# ════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Monitor H3v2 Paper Trading')
    parser.add_argument('--signal-only', action='store_true', help='Solo señal de mañana')
    parser.add_argument('--history', action='store_true', help='Ver historial de trades')
    parser.add_argument('--add-trade', nargs=5,
                        metavar=('DATE', 'DIRECTION', 'ENTRY', 'EXIT', 'FIRST_HOUR_RET'),
                        help='Registrar nuevo trade. Ej: --add-trade 2025-01-15 LONG 21100 21350 0.005')
    args = parser.parse_args()

    # Registrar trade
    if args.add_trade:
        date_str, direction, entry, exit_, fhr = args.add_trade
        trade = add_paper_trade(date_str, direction.upper(),
                                float(entry), float(exit_), float(fhr))
        logger.info(f"\n  Trade registrado: {trade}")
        return

    # Solo historial
    if args.history:
        show_paper_history()
        return

    logger.info("╔" + "═"*68 + "╗")
    logger.info("║   H3v2 SIGNAL MONITOR — Paper Trading NQ100                   ║")
    logger.info(f"║   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'):<60} ║")
    logger.info("╚" + "═"*68 + "╝")

    # Cargar datos
    logger.info("\n  Cargando datos recientes...")
    df = load_latest_data()
    df_daily = compute_daily_summary(df, n_days=90)
    logger.info(f"  Último día en datos: {df_daily.index[-1].date()}")

    # Señal de mañana
    signal_today = evaluate_today_filter(df_daily)

    logger.info("\n" + "═"*68)
    logger.info("  SEÑAL PARA EL PRÓXIMO DÍA DE TRADING")
    logger.info("═"*68)
    logger.info(f"\n  Último día de datos:  {signal_today['last_trading_day']}")
    logger.info(f"  Retorno del día:      {signal_today['today_return_pct']:+.3f}%")
    logger.info(f"  Filtro activo:        {'🔔 SÍ — DÍA PREVIO BAJISTA' if signal_today['filter_active'] else '⏸️  NO — Sin trade mañana'}")

    if signal_today['filter_active']:
        logger.info("\n  ┌─────────────────────────────────────────┐")
        logger.info("  │          INSTRUCCIONES PARA MAÑANA       │")
        logger.info("  ├─────────────────────────────────────────┤")
        logger.info("  │  14:25 UTC → PONER ALARMA                │")
        logger.info("  │  13:30-14:29 UTC → Observar 1H           │")
        logger.info("  │  14:29 UTC → Calcular ret_1H              │")
        logger.info(f"  │  Si |ret_1H| > {THR_FIRST_HOUR*100:.1f}% → ENTRAR            │")
        logger.info("  │    Positivo → LONG hasta 20:00 UTC       │")
        logger.info("  │    Negativo → SHORT hasta 20:00 UTC      │")
        logger.info(f"  │  Costo RT asumido: {COST_TARGET_PTS} pts NQ             │")
        logger.info("  └─────────────────────────────────────────┘")
    else:
        logger.info("\n  Sin trade mañana. Revisar mañana al cierre NY.")

    if args.signal_only:
        # Guardar señal
        with open(SIGNAL_LOG, 'w') as f:
            json.dump(signal_today, f, indent=2, default=str)
        return

    # Backtest OOS incremental
    logger.info("\n  Calculando estadísticas OOS...")
    oos_stats = oos_incremental_backtest(df_daily, oos_start='2025-01-01')

    if oos_stats:
        logger.info(f"\n  OOS stats (2025-hoy):")
        logger.info(f"    N trades: {oos_stats['n']}")
        logger.info(f"    Sharpe:   {oos_stats['sharpe']:.3f}")
        logger.info(f"    Ann. Ret: {oos_stats['annual']*100:.1f}%")
        logger.info(f"    WR:       {oos_stats['wr']*100:.1f}%")
        logger.info(f"    p-value:  {oos_stats['pvalue']:.4f}")

    # Historial paper trades
    show_paper_history()

    # Dashboard
    logger.info("\n  Generando dashboard visual...")
    plot_monitor(oos_stats or {}, df_daily, signal_today)

    # Guardar señal del día
    out_signal = {'timestamp': datetime.now(timezone.utc).isoformat(), **signal_today}
    with open(SIGNAL_LOG, 'w') as f:
        json.dump(out_signal, f, indent=2, default=str)

    logger.info(f"\n  Señal guardada: {SIGNAL_LOG}")
    logger.info("  ✅ Monitor completado")


if __name__ == "__main__":
    main()
