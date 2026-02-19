"""
settings.py — Configuración centralizada del proyecto.

Todas las rutas, constantes y umbrales de validación están aquí.
Ningún módulo hardcodea paths ni valores mágicos.
"""
from pathlib import Path

# ── Rutas ──
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
EXTRACTED_DIR = RAW_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"

# ── Asset ──
PAIR      = "EURUSD"
TIMEFRAME = "M1"

# ── Columnas esperadas después de parseo ──
OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# ── Validación ──
MIN_ROWS = 10_000  # Mínimo de barras para considerar dataset válido
