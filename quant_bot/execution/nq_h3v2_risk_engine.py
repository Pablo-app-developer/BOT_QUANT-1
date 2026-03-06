"""
nq_h3v2_risk_engine.py — Motor de Riesgo Completo para H3v2

PRINCIPIOS INAMOVIBLES:
  1. SL físico en el milisegundo del fill — SIEMPRE
  2. Riesgo máximo por trade: 0.5% del balance
  3. Pérdida diaria máxima: 3% (FTMO permite 5%, usamos 3% por seguridad)
  4. Drawdown máximo: 8% (FTMO permite 10%, buffer de 2%)
  5. Parada automática tras 3 pérdidas consecutivas → revisar manualmente
  6. Sin override: el engine bloquea el trade si alguna regla falla

INTEGRACIÓN:
  - Lee señal de nq_signal_monitor.py (FILTER_ACTIVE)
  - A las 14:29 UTC evalúa r_1h
  - Si señal activa → calcula size → envía orden → coloca SL inmediatamente
  - A las 19:59 UTC → cierre automático si no se cerró antes
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RISK_DIR = PROJECT_ROOT / "quant_bot" / "execution" / "risk_data"
RISK_DIR.mkdir(parents=True, exist_ok=True)

TRADES_FILE   = RISK_DIR / "live_trades.json"
EQUITY_FILE   = RISK_DIR / "equity_curve.json"
JOURNAL_FILE  = RISK_DIR / "risk_journal.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RiskEngine - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(JOURNAL_FILE),
    ]
)
logger = logging.getLogger("RiskEngine")


# ═══════════════════════════════════════════════════
# PARÁMETROS FTMO / PROP-FIRM (AJUSTABLES)
# ═══════════════════════════════════════════════════

@dataclass
class RiskParams:
    # Riesgo por trade
    risk_per_trade_pct: float = 0.005      # 0.5% del balance
    
    # SL basado en ATR de la primera hora (de nq_h3_mae_mfe.py: 1.5x es óptimo)
    sl_atr_mult: float = 1.5

    # Límites diarios (conservadores vs FTMO)
    max_daily_loss_pct: float = 0.03       # 3% (FTMO = 5%)
    max_drawdown_pct: float = 0.08         # 8% (FTMO = 10%)

    # Reglas de calidad
    max_consecutive_losses: int = 3        # pausa tras 3 pérdidas
    min_first_hour_threshold: float = 0.003  # |r_1h| > 0.3%
    prior_day_threshold: float = -0.001    # día previo < -0.1%

    # Instrument (ajustar por broker)
    instrument: str = "USATECHIDXUSD"
    dollar_per_point: float = 1.0          # $1/punto para CFD NQ (ajustar)
    min_lot_size: float = 0.01
    lot_step: float = 0.01
    
    # Comisión estimada (round-trip) para Micro lotes/MNQ
    commission_usd: float = 0.50


# ═══════════════════════════════════════════════════
# REGISTRO DE TRADE
# ═══════════════════════════════════════════════════

@dataclass
class Trade:
    trade_date: str
    direction: str               # 'LONG' | 'SHORT'
    entry_price: float
    sl_price: float
    tp_price: Optional[float]    # None = salida a las 19:59 UTC
    position_size: float         # lotes
    risk_usd: float              # en USD

    exit_price: float = 0.0
    exit_reason: str = ''        # 'EOD' | 'SL' | 'MANUAL'
    pnl_usd: float = 0.0
    pnl_pct_account: float = 0.0
    r_first_hour: float = 0.0
    oh_atr: float = 0.0
    balance_before: float = 0.0
    balance_after: float = 0.0
    status: str = 'OPEN'         # 'OPEN' | 'CLOSED' | 'BLOCKED'
    notes: str = ''
    timestamp_open: str = ''
    timestamp_close: str = ''

    def to_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════
# MOTOR DE RIESGO
# ═══════════════════════════════════════════════════

class RiskEngine:

    def __init__(self, initial_balance: float, params: Optional[RiskParams] = None):
        self.initial_balance = initial_balance
        self.peak_balance    = initial_balance
        self.current_balance = initial_balance
        self.params          = params or RiskParams()
        self.trades          = self._load_trades()
        self._update_equity_stats()
        logger.info(f"RiskEngine iniciado: balance=${initial_balance:,.2f}")

    # ─── PERSISTENCIA ──────────────────────────────

    def _load_trades(self) -> list:
        if TRADES_FILE.exists():
            with open(TRADES_FILE) as f:
                return json.load(f)
        return []

    def _save_trades(self):
        with open(TRADES_FILE, 'w') as f:
            json.dump(self.trades, f, indent=2, default=str)

    def _update_equity_stats(self):
        if self.trades:
            closed = [t for t in self.trades if t.get('status') == 'CLOSED']
            if closed:
                self.current_balance = closed[-1]['balance_after']
                self.peak_balance    = max(t['balance_after'] for t in closed)

    # ─── VERIFICACIONES DE SEGURIDAD ───────────────

    def check_daily_loss(self) -> tuple[bool, str]:
        """¿La pérdida del día de hoy ya alcanzó el límite diario?"""
        today = str(date.today())
        today_trades = [t for t in self.trades
                        if t.get('trade_date') == today and t.get('status') == 'CLOSED']
        
        daily_pnl_usd = sum(t.get('pnl_usd', 0) for t in today_trades)
        daily_pnl_pct = daily_pnl_usd / self.current_balance
        
        if daily_pnl_pct <= -self.params.max_daily_loss_pct:
            return False, (f"LÍMITE DIARIO ALCANZADO: {daily_pnl_pct*100:.2f}% "
                          f"(max: -{self.params.max_daily_loss_pct*100:.0f}%)")
        return True, f"Daily OK: {daily_pnl_pct*100:.2f}%"

    def check_drawdown(self) -> tuple[bool, str]:
        """¿El drawdown actual supera el límite máximo?"""
        if self.peak_balance <= 0:
            return True, "Peak balance no calculado aún"
        
        dd = (self.current_balance - self.peak_balance) / self.peak_balance
        if dd <= -self.params.max_drawdown_pct:
            return False, (f"DRAWDOWN MÁXIMO ALCANZADO: {dd*100:.2f}% "
                          f"(límite: -{self.params.max_drawdown_pct*100:.0f}%)")
        return True, f"DD OK: {dd*100:.2f}% (peak=${self.peak_balance:,.2f})"

    def check_consecutive_losses(self) -> tuple[bool, str]:
        """¿Hay demasiadas pérdidas consecutivas?"""
        closed = [t for t in self.trades if t.get('status') == 'CLOSED']
        if not closed:
            return True, "Sin historial"
        
        consecutive = 0
        for t in reversed(closed):
            if t.get('pnl_usd', 0) < 0:
                consecutive += 1
            else:
                break
        
        if consecutive >= self.params.max_consecutive_losses:
            return False, (f"PÉRDIDAS CONSECUTIVAS: {consecutive} "
                          f"(máx: {self.params.max_consecutive_losses}). "
                          f"Revisión manual requerida.")
        return True, f"Racha: {consecutive} pérdidas consecutivas"

    def check_today_traded(self) -> tuple[bool, str]:
        """¿Ya operamos hoy?"""
        today = str(date.today())
        today_open = [t for t in self.trades
                      if t.get('trade_date') == today and t.get('status') in ('OPEN', 'CLOSED')]
        if today_open:
            return False, f"Ya hay {len(today_open)} trade(s) hoy para esta estrategia"
        return True, "Sin trades hoy aún"

    def all_risk_checks(self) -> tuple[bool, list]:
        """Ejecuta TODOS los checks. Si alguno falla, bloquea el trade."""
        checks = [
            self.check_daily_loss(),
            self.check_drawdown(),
            self.check_consecutive_losses(),
            self.check_today_traded(),
        ]
        messages = []
        all_ok   = True
        for ok, msg in checks:
            icon = "✅" if ok else "❌"
            logger.info(f"  Risk Check: {icon} {msg}")
            messages.append(f"{icon} {msg}")
            if not ok:
                all_ok = False
        return all_ok, messages

    # ─── CÁLCULO DE SEÑAL H3v2 ─────────────────────

    def evaluate_filter(self, prior_day_return: float) -> bool:
        """
        CONDICIÓN 1 (verificar al CIERRE NY del día anterior):
        ¿El día previo cerró con -0.1% o más?
        """
        active = prior_day_return < self.params.prior_day_threshold
        logger.info(f"  Filtro previo: {prior_day_return*100:.3f}% → {'ACTIVO' if active else 'INACTIVO'}")
        return active

    def evaluate_signal(self, first_hour_return: float) -> Optional[str]:
        """
        CONDICIÓN 2 (verificar a las 14:29 UTC):
        ¿La primera hora (13:30-14:29 UTC) movió > 0.3%?
        Retorna 'LONG', 'SHORT', o None si no hay señal.
        """
        if abs(first_hour_return) > self.params.min_first_hour_threshold:
            direction = 'LONG' if first_hour_return > 0 else 'SHORT'
            logger.info(f"  Señal 1H: {first_hour_return*100:.3f}% → {direction}")
            return direction
        logger.info(f"  Señal 1H: {first_hour_return*100:.3f}% → Sin señal (< 0.3%)")
        return None

    # ─── SIZING DE POSICIÓN ────────────────────────

    def compute_position_size(
        self,
        entry_price: float,
        oh_atr_pct: float,          # oh_atr (normalizado, p.ej. 0.003 = 0.3%)
        balance: Optional[float] = None
    ) -> dict:
        """
        Calcula el tamaño de posición dado el riesgo por trade y el SL.
        
        SL = 1.5 × ATR_1H en puntos absolutos
        Size = (risk_usd) / (sl_pts × dollar_per_point)
        """
        balance  = balance or self.current_balance
        risk_usd = balance * self.params.risk_per_trade_pct

        # SL en puntos absolutos de precio
        oh_atr_pts = oh_atr_pct * entry_price  # ATR en puntos de precio
        sl_pts     = oh_atr_pts * self.params.sl_atr_mult

        # Evitar SL absurdamente pequeño
        sl_pts = max(sl_pts, 5.0)

        # Tamaño de lotes
        raw_lots = risk_usd / (sl_pts * self.params.dollar_per_point)

        # Redondear al lot step
        lots = max(
            self.params.min_lot_size,
            round(raw_lots / self.params.lot_step) * self.params.lot_step
        )

        result = {
            'balance':    balance,
            'risk_usd':   risk_usd,
            'oh_atr_pts': oh_atr_pts,
            'sl_pts':     sl_pts,
            'lots':       lots,
            'risk_actual_usd': lots * sl_pts * self.params.dollar_per_point,
            'risk_actual_pct': (lots * sl_pts * self.params.dollar_per_point) / balance,
        }

        logger.info(f"  Sizing: balance=${balance:,.0f}  risk=${risk_usd:.0f}  "
                    f"ATR={oh_atr_pts:.1f}pts  SL={sl_pts:.1f}pts  "
                    f"→ {lots:.2f} lotes  (riesgo real: {result['risk_actual_pct']*100:.2f}%)")
        return result

    def compute_sl_price(self, entry_price: float, direction: str,
                          oh_atr_pct: float) -> float:
        """Precio exacto del Stop Loss físico."""
        sl_pts = oh_atr_pct * entry_price * self.params.sl_atr_mult
        sl_pts = max(sl_pts, 5.0)
        if direction == 'LONG':
            return round(entry_price - sl_pts, 2)
        else:
            return round(entry_price + sl_pts, 2)

    # ─── APERTURA DE TRADE ─────────────────────────

    def open_trade(
        self,
        direction: str,
        entry_price: float,
        oh_atr_pct: float,
        r_first_hour: float,
        balance: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Crea el trade con todas las reglas aplicadas.
        Si algún check falla, retorna None y NO opera.
        """
        logger.info("\n" + "─"*60)
        logger.info(f"  INTENTO DE APERTURA: {direction} @ {entry_price:.2f}")

        # ── Verificaciones de riesgo ──────────────
        all_ok, msgs = self.all_risk_checks()
        if not all_ok:
            logger.warning("  ❌ TRADE BLOQUEADO por risk engine")
            return None

        # ── Cálculo de sizing ─────────────────────
        sizing = self.compute_position_size(entry_price, oh_atr_pct, balance)
        sl_price = self.compute_sl_price(entry_price, direction, oh_atr_pct)

        trade = Trade(
            trade_date    = str(date.today()),
            direction     = direction,
            entry_price   = entry_price,
            sl_price      = sl_price,
            tp_price      = None,
            position_size = sizing['lots'],
            risk_usd      = sizing['risk_usd'],
            r_first_hour  = r_first_hour,
            oh_atr        = oh_atr_pct,
            balance_before= self.current_balance,
            status        = 'OPEN',
            timestamp_open= datetime.now(timezone.utc).isoformat(),
        )

        self.trades.append(trade.to_dict())
        self._save_trades()

        logger.info(f"  ✅ TRADE ABIERTO:")
        logger.info(f"     Dirección:  {direction}")
        logger.info(f"     Entrada:    {entry_price:.2f}")
        logger.info(f"     SL físico:  {sl_price:.2f}  ({abs(entry_price-sl_price):.1f} pts)")
        logger.info(f"     Tamaño:     {sizing['lots']:.2f} lotes")
        logger.info(f"     Riesgo USD: ${sizing['risk_actual_usd']:.2f} "
                    f"({sizing['risk_actual_pct']*100:.2f}%)")
        logger.info(f"     EXIT EOD:   19:59 UTC")

        return trade

    # ─── CIERRE DE TRADE ───────────────────────────

    def close_trade(
        self,
        trade: dict,
        exit_price: float,
        exit_reason: str = 'EOD'
    ) -> dict:
        """Cierra el trade y actualiza el balance."""
        direction = 1 if trade['direction'] == 'LONG' else -1
        pts       = direction * (exit_price - trade['entry_price'])
        pnl_gross = pts * trade['position_size'] * self.params.dollar_per_point
        pnl_net   = pnl_gross - self.params.commission_usd

        trade['exit_price']       = exit_price
        trade['exit_reason']      = exit_reason
        trade['pnl_usd']          = round(pnl_net, 2)
        trade['pnl_pct_account']  = round(pnl_net / trade['balance_before'], 6)
        trade['balance_after']    = round(trade['balance_before'] + pnl_net, 2)
        trade['status']           = 'CLOSED'
        trade['timestamp_close']  = datetime.now(timezone.utc).isoformat()

        self.current_balance = trade['balance_after']
        self.peak_balance    = max(self.peak_balance, self.current_balance)

        icon = "🟢" if pnl_net > 0 else "🔴"
        logger.info(f"\n  {icon} TRADE CERRADO ({exit_reason}):")
        logger.info(f"     Exit: {exit_price:.2f}  PnL: ${pnl_net:.2f} "
                    f"({trade['pnl_pct_account']*100:.3f}%)")
        logger.info(f"     Balance: ${self.current_balance:,.2f}")

        self._save_trades()
        self._save_equity()
        return trade

    def _save_equity(self):
        closed = [t for t in self.trades if t.get('status') == 'CLOSED']
        eq_data = {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'peak_balance':    self.peak_balance,
            'total_trades':    len(closed),
            'winning_trades':  sum(1 for t in closed if t.get('pnl_usd', 0) > 0),
            'total_pnl_usd':   sum(t.get('pnl_usd', 0) for t in closed),
            'drawdown_current': (self.current_balance - self.peak_balance) / self.peak_balance,
            'last_updated':    datetime.now(timezone.utc).isoformat(),
        }
        with open(EQUITY_FILE, 'w') as f:
            json.dump(eq_data, f, indent=2)

    # ─── REPORTES ──────────────────────────────────

    def print_status(self):
        """Dashboard rápido del estado actual."""
        closed = [t for t in self.trades if t.get('status') == 'CLOSED']
        open_t = [t for t in self.trades if t.get('status') == 'OPEN']

        logger.info("\n" + "═"*60)
        logger.info("  ESTADO DEL RISK ENGINE")
        logger.info("═"*60)
        logger.info(f"  Balance actual:   ${self.current_balance:,.2f}")
        logger.info(f"  Balance inicial:  ${self.initial_balance:,.2f}")
        logger.info(f"  Balance pico:     ${self.peak_balance:,.2f}")

        if self.peak_balance > 0:
            dd = (self.current_balance - self.peak_balance) / self.peak_balance
            logger.info(f"  Drawdown actual:  {dd*100:.2f}%")

        logger.info(f"\n  Trades cerrados:  {len(closed)}")
        logger.info(f"  Trades abiertos:  {len(open_t)}")

        if closed:
            wins  = sum(1 for t in closed if t.get('pnl_usd', 0) > 0)
            total_pnl = sum(t.get('pnl_usd', 0) for t in closed)
            wr    = wins / len(closed)
            logger.info(f"  Win Rate:         {wr*100:.1f}%")
            logger.info(f"  PnL Total:        ${total_pnl:.2f}")
            logger.info(f"  PnL / Trade:      ${total_pnl/len(closed):.2f}")

            # Racha actual
            consec = 0
            for t in reversed(closed):
                if t.get('pnl_usd', 0) < 0:
                    consec += 1
                else:
                    break
            logger.info(f"  Racha pérdidas:   {consec}")

        # Checks de riesgo
        logger.info("\n  CHECKS DE RIESGO:")
        self.all_risk_checks()

    def simulate_expected_returns(self, n_months: int = 12) -> dict:
        """
        Simula el rendimiento esperado basado en estadísticas OOS 2024-25.
        Parámetros de entrada basados en los resultados del backtesting.
        """
        # Del análisis OOS 2024-2025 con retorno post-señal correcto:
        wr_oos        = 0.614
        avg_win_pct   = 0.0017   # ~0.17% por trade ganador (del OOS)
        avg_loss_pct  = 0.0008   # ~0.08% por trade perdedor promedio
        trades_month  = 3.7      # trades/mes (del análisis OOS)
        sl_hit_rate   = 0.20     # ~20% de perdedores tocan el SL

        total_trades = n_months * trades_month
        wins         = total_trades * wr_oos
        losses       = total_trades * (1 - wr_oos)
        sl_hits      = losses * sl_hit_rate

        risk_usd    = self.current_balance * self.params.risk_per_trade_pct
        commission  = self.params.commission_usd

        # Ingresos
        income_wins   = wins  * avg_win_pct * self.current_balance
        # Costos: pérdidas EOD pequeñas + SL hits + comisiones
        cost_eod_loss = (losses - sl_hits) * avg_loss_pct * self.current_balance
        cost_sl_hits  = sl_hits * risk_usd
        cost_commiss  = total_trades * commission

        net_pnl  = income_wins - cost_eod_loss - cost_sl_hits - cost_commiss
        net_pct  = net_pnl / self.current_balance

        logger.info(f"\n  PROYECCIÓN {n_months} MESES (basada en OOS 2024-25):")
        logger.info(f"  Trades esperados:  {total_trades:.0f}")
        logger.info(f"  Ingresos (wins):   ${income_wins:.0f}")
        logger.info(f"  Costo (SL hits):   ${cost_sl_hits:.0f}")
        logger.info(f"  Costo (EOD loss):  ${cost_eod_loss:.0f}")
        logger.info(f"  Comisiones:        ${cost_commiss:.0f}")
        logger.info(f"  PnL neto:          ${net_pnl:.0f} ({net_pct*100:.1f}%)")

        return {
            'months': n_months,
            'expected_trades': total_trades,
            'net_pnl_usd': net_pnl,
            'net_pct': net_pct,
            'ann_pct': net_pct * (12 / n_months),
        }


# ═══════════════════════════════════════════════════
# DEMO / VERIFICACIÓN
# ═══════════════════════════════════════════════════

def demo():
    logger.info("╔" + "═"*60 + "╗")
    logger.info("║   H3v2 RISK ENGINE — Demo de Verificación                 ║")
    logger.info("╚" + "═"*60 + "╝")

    # Cuenta demo de $10,000 (FTMO Challenge inicial)
    engine = RiskEngine(initial_balance=10_000.0)
    engine.print_status()

    # Simular un día con señal válida
    logger.info("\n" + "─"*60)
    logger.info("  SIMULACIÓN DE TRADE DE EJEMPLO:")

    prior_day_return = -0.008   # -0.8% (bajista, filtro activo)
    first_hour_ret   = +0.006   # +0.6% (señal alcista, > 0.3%)
    entry_price      = 21_000.0 # precio NQ aproximado
    oh_atr_pct       = 0.004    # 0.4% ATR de la primera hora (~84 pts)

    # Evaluación del filtro
    filter_active = engine.evaluate_filter(prior_day_return)
    direction     = engine.evaluate_signal(first_hour_ret) if filter_active else None

    if direction:
        trade = engine.open_trade(
            direction    = direction,
            entry_price  = entry_price,
            oh_atr_pct   = oh_atr_pct,
            r_first_hour = first_hour_ret,
        )

        if trade:
            # Simular cierre EOD con retorno positivo de +0.2%
            exit_price = entry_price * (1 + 0.002)
            engine.close_trade(trade.to_dict(), exit_price, exit_reason='EOD')

    # Proyección
    engine.simulate_expected_returns(n_months=12)
    engine.simulate_expected_returns(n_months=3)


if __name__ == "__main__":
    demo()
