"""
nq_h3_mae_mfe.py — Análisis Estructural del Trade Intra-barra (MAE / MFE)

OBJETIVO:
  El edge H3v2 mantiene la posición desde las 14:30 UTC hasta las 20:00 UTC (EOD).
  Para cumplir reglas FTMO, DEBEMOS poner un Stop Loss (SL) físico.
  
  ¿Dónde poner el SL sin cortar prematuramente demasiados trades ganadores?
  Calcularemos el Maximum Adverse Excursion (MAE) y Maximum Favorable Excursion (MFE)
  para cada trade H3v2 durante su vida (14:30 a 20:00).

MÉTRICAS:
  - MAE (Max Adverse Excursion): la peor pérdida experimentada intra-trade antes del cierre.
  - MFE (Max Favorable Excursion): la máxima ganancia experimentada.
  
  Se testearán múltiples umbrales de SL y TP (Take Profit):
  - SL y TP fijos (en % o puntos NQ)
  - SL basados en el ATR de la Primera Hora (oh_atr)
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("H3_MAE_MFE")

DUKASCOPY_SCALE = 82.0
COST_PTS = 2.0
THR_PRIOR_DAY = -0.001
THR_FIRST_HOUR = 0.003
OOS_YEAR = 2025

def calculate_mae_mfe(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae cada trade y calcula su MAE, MFE y resolución con SL/TP."""
    ny = df[df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])].copy()
    records = []
    
    for date_key, group in ny.groupby(ny.index.date):
        oh = group[group['session'] == 'OPEN_HOUR']
        post = group[group['session'].isin(['MIDDAY', 'CLOSE_HOUR'])]
        
        if len(oh) < 30 or len(post) < 10:
            continue
            
        oh_open = oh['open'].iloc[0]
        oh_close = oh['close'].iloc[-1]
        
        first_hour_ret = (oh_close - oh_open) / oh_open
        oh_atr = (oh['high'].max() - oh['low'].min()) / oh_open
        day_return = (post['close'].iloc[-1] - oh_open) / oh_open
        
        # Filtro base
        if np.abs(first_hour_ret) <= THR_FIRST_HOUR:
            continue
            
        entry_price = oh_close
        direction = np.sign(first_hour_ret)
        
        # Simular trayectoria intra-trade (Highs y Lows del M1 en post-session)
        # Si somos LONG:
        #   Favorable = Highs, Adverso = Lows
        # Si somos SHORT:
        #   Favorable = Lows, Adverso = Highs
        if direction == 1:
            max_pt = post['high'].max()
            min_pt = post['low'].min()
            mfe = (max_pt - entry_price) / entry_price
            mae = (min_pt - entry_price) / entry_price
        else:
            min_pt = post['low'].min()
            max_pt = post['high'].max()
            mfe = (entry_price - min_pt) / entry_price  # > 0
            mae = (entry_price - max_pt) / entry_price  # < 0 (adverso)
            
        exit_price = post['close'].iloc[-1]
        ret_eod = direction * (exit_price - entry_price) / entry_price
        
        records.append({
            'date': pd.Timestamp(date_key, tz='UTC'),
            'year': date_key.year,
            'direction': direction,
            'entry_price': float(entry_price),
            'first_hour_ret': float(first_hour_ret),
            'oh_atr': float(oh_atr),
            'day_return': float(day_return),
            'mfe': float(mfe),
            'mae': float(mae),
            'ret_eod': float(ret_eod),
            'post_highs': post['high'].values,
            'post_lows': post['low'].values,
            'post_closes': post['close'].values
        })
        
    df_out = pd.DataFrame(records).set_index('date')
    if df_out.empty: return df_out
    df_out['prior_day_ret'] = df_out['day_return'].shift(1)
    df_out['prior_bearish'] = df_out['prior_day_ret'] < THR_PRIOR_DAY
    
    # Filtro definitivo
    return df_out[df_out['prior_bearish']].dropna(subset=['prior_day_ret'])

def analyze_stops(df_trades: pd.DataFrame) -> pd.DataFrame:
    """Evalúa el impacto de múltiples niveles de SL basados en ATR(1H)."""
    cost_pct = (COST_PTS / DUKASCOPY_SCALE) / df_trades['entry_price'].mean()
    
    # SL multipliers (0.5x, 1x, 1.5x, 2x, Infinite)
    sl_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 999.0]
    
    rows = []
    base_rets = df_trades['ret_eod'].values - cost_pct
    base_sh = (base_rets.mean() / base_rets.std()) * np.sqrt(252) if base_rets.std() > 0 else 0

    for mult in sl_mults:
        sim_rets = []
        for _, row in df_trades.iterrows():
            atr = row['oh_atr']
            sl_pct = atr * mult
            target_mae = -sl_pct  # mae es negativo
            
            # Evaluamos tick a tick (a nivel de barra M1) cuándo toca el SL
            rets_hist = []
            hit_sl = False
            for h, l, c in zip(row['post_highs'], row['post_lows'], row['post_closes']):
                if row['direction'] == 1:
                    adv = (l - row['entry_price']) / row['entry_price']
                else:
                    adv = (row['entry_price'] - h) / row['entry_price']
                    
                if adv <= target_mae:
                    # Golpeamos SL
                    sim_rets.append(target_mae - cost_pct)
                    hit_sl = True
                    break
                    
            if not hit_sl:
                # No tocó SL, exit EOD
                sim_rets.append(row['ret_eod'] - cost_pct)
                
        sim_rets = np.array(sim_rets)
        win_rate = (sim_rets > 0).mean()
        sharpe = (sim_rets.mean() / sim_rets.std()) * np.sqrt(252) if sim_rets.std() > 0 else 0
        ann = np.cumprod(1 + sim_rets)[-1] ** (252 / len(sim_rets)) - 1
        
        label = f"SL {mult}x ATR" if mult < 100 else "Sin SL Físico (EOD)"
        rows.append({
            'SL Level': label,
            'Mult': mult,
            'Sharpe': sharpe,
            'Ann Ret': ann * 100,
            'Win Rate': win_rate * 100,
            'N Trades': len(sim_rets)
        })
        logger.info(f"  {label:20s}: Sharpe={sharpe:.3f}  WR={win_rate*100:.1f}%")
        
    return pd.DataFrame(rows)

def plot_mae_mfe(df: pd.DataFrame, sl_res: pd.DataFrame):
     fig = plt.figure(figsize=(16, 12), facecolor='#0d1117')
     gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
     GOLD='#FFD700'; GREEN='#00FF88'; RED='#FF4444'; BLUE='#4488FF'; GRAY='#888888'; BG='#161b22'

     def ax_style(ax, title):
         ax.set_facecolor(BG)
         ax.set_title(title, color=GOLD, fontsize=10, fontweight='bold', pad=8)
         ax.tick_params(colors=GRAY)
         ax.spines[:].set_color('#333333')
         for l in ax.get_xticklabels() + ax.get_yticklabels(): l.set_color(GRAY)
         
     # 1. MAE vs PnL
     ax1 = fig.add_subplot(gs[0, 0])
     colors = [GREEN if r > 0 else RED for r in df['ret_eod']]
     ax1.scatter(df['mae']*100, df['ret_eod']*100, c=colors, alpha=0.6, s=15)
     ax1.axhline(0, color=GRAY, lw=0.8, ls=':')
     ax1.axvline(0, color=GRAY, lw=0.8, ls=':')
     ax_style(ax1, "Max Adverse Excursion (MAE) vs Retorno Final")
     ax1.set_xlabel("MAE (%)", color=GRAY)
     ax1.set_ylabel("Retorno EOD (%)", color=GRAY)
     ax1.invert_xaxis() # MAE negativo a la derecha
     
     # 2. MFE vs PnL
     ax2 = fig.add_subplot(gs[0, 1])
     ax2.scatter(df['mfe']*100, df['ret_eod']*100, c=colors, alpha=0.6, s=15)
     ax2.axhline(0, color=GRAY, lw=0.8, ls=':')
     ax_style(ax2, "Max Favorable Excursion (MFE) vs Retorno Final")
     ax2.set_xlabel("MFE (%)", color=GRAY)
     
     # 3. Impacto del SL
     ax3 = fig.add_subplot(gs[1, :])
     x_labels = sl_res['SL Level']
     ax3.plot(x_labels, sl_res['Sharpe'], marker='o', color=BLUE, lw=2, label="Sharpe")
     ax3.axhline(sl_res['Sharpe'].iloc[-1], color=GOLD, ls='--', label="Sharpe sin SL")
     ax3.legend(facecolor=BG, labelcolor='white')
     ax_style(ax3, "Sensibilidad de Sharpe a Nivel de Stop Loss Físico")
     ax3.set_ylabel("Sharpe Ratio", color=GRAY)
     
     out = ARTIFACTS_DIR / "nq_h3_mae_mfe.png"
     plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
     plt.close()
     logger.info(f"✅ Gráfico MAE/MFE: {out}")

def main():
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║   H3v2: Análisis de MAE/MFE intra-trade (Risk Engine FTMO)   ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    
    parquet = PROJECT_ROOT / "quant_bot" / "data" / "processed" / "USATECHIDXUSD_M1.parquet"
    if not parquet.exists():
        logger.info("Dataset no disponible. Ejecutar nq_edge_discovery.py --rebuild primero.")
        return
        
    df = pd.read_parquet(parquet, engine='pyarrow')
    if 'session' not in df.columns:
        from quant_bot.data.nq_loader import add_session_labels
        df = add_session_labels(df)
        
    df_trades = calculate_mae_mfe(df)
    logger.info(f"Trades extraídos con métricas intra-barra: {len(df_trades)}")
    
    # Evaluar en In-Sample
    df_is = df_trades[df_trades['year'] < OOS_YEAR]
    logger.info(f"Trades IS (2021-2024): {len(df_is)}")
    
    sl_impact = analyze_stops(df_is)
    plot_mae_mfe(df_is, sl_impact)

if __name__ == "__main__":
    main()
