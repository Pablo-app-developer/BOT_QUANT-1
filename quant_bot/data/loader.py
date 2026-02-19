"""
loader.py — Pipeline de datos para EURUSD M1.

Responsabilidades:
  1. extract_zips()         → Extraer _all.csv de cada ZIP anual
  2. load_csv(path)         → Parsear un CSV (auto-detecta formato A/B)
  3. load_and_combine_all() → Cargar + concatenar todos los CSVs
  4. clean(df)              → Ordenar, dedup, NaN handling
  5. validate(df)           → Integridad OHLC, gaps, reporte
  6. save_processed(df)     → Guardar Parquet
  7. load_processed()       → Leer Parquet

Formatos soportados:
  A (EURUSD1.csv):   2025-08-05 18:28\\t1.15758 1.15765 1.15757 1.15760 43
  B (yearly _all):   2016.01.04,00:00,1.08561,1.08729,1.08561,1.08708,1
"""

import zipfile
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. EXTRACCIÓN DE ZIPS
# ─────────────────────────────────────────────

def extract_zips(force: bool = False) -> list[Path]:
    """
    Extrae el archivo *_all.csv de cada ZIP anual a EXTRACTED_DIR.

    Problema conocido: EURUSD2018.zip y EURUSD2019.zip tienen ambos
    'EURUSD_all.csv' (sin año). Lo renombramos al extraer.

    Returns: lista de paths de los _all.csv extraídos.
    """
    settings.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    zips = sorted(settings.RAW_DIR.glob("EURUSD20*.zip"))
    extracted = []

    for zp in zips:
        # Derivar año del nombre: EURUSD2016.zip → 2016
        year = zp.stem.replace("EURUSD", "")
        target = settings.EXTRACTED_DIR / f"EURUSD_{year}_all.csv"

        if target.exists() and not force:
            logger.debug(f"  Ya existe {target.name}, skip.")
            extracted.append(target)
            continue

        with zipfile.ZipFile(zp, 'r') as z:
            # Buscar el archivo _all.csv dentro del zip
            all_csvs = [n for n in z.namelist() if n.endswith("_all.csv")]
            if not all_csvs:
                logger.warning(f"  {zp.name}: no se encontró _all.csv, skip.")
                continue

            # Leer el contenido y escribirlo con nombre normalizado
            with z.open(all_csvs[0]) as src, open(target, 'wb') as dst:
                dst.write(src.read())

            logger.info(f"  {zp.name} → {target.name}")
            extracted.append(target)

    logger.info(f"Extracción completa: {len(extracted)} archivos.")
    return extracted


# ─────────────────────────────────────────────
# 2. CARGA DE UN CSV
# ─────────────────────────────────────────────

def _detect_format(path: Path) -> str:
    """Lee la primera línea para determinar el formato."""
    with open(path, 'r') as f:
        line = f.readline().strip()
    # Formato B: fecha con puntos y comas → 2016.01.04,00:00,...
    if ',' in line and '.' in line.split(',')[0]:
        return 'B'
    return 'A'


def load_csv(path: Path) -> pd.DataFrame:
    """
    Carga UN archivo CSV y retorna DataFrame con DatetimeIndex y columnas OHLCV.

    Formato A (whitespace):  date time \\t O H L C V
    Formato B (comma):       date,time,O,H,L,C,V
    """
    path = Path(path)
    fmt = _detect_format(path)
    cols = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']

    if fmt == 'B':
        df = pd.read_csv(path, header=None, names=cols)
        # Fecha: 2016.01.04 → 2016-01-04
        df['datetime'] = pd.to_datetime(
            df['date'].str.replace('.', '-', regex=False) + ' ' + df['time'],
            format='%Y-%m-%d %H:%M',
        )
    else:
        df = pd.read_csv(path, header=None, names=cols, sep=r'\s+')
        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%Y-%m-%d %H:%M',
        )

    df.set_index('datetime', inplace=True)
    df = df[settings.OHLCV_COLS].copy()

    # Forzar tipos numéricos
    for c in settings.OHLCV_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    logger.info(f"  {path.name}: {len(df):>8,} barras  "
                f"| {df.index[0]} → {df.index[-1]}")
    return df


# ─────────────────────────────────────────────
# 3. CARGA Y COMBINACIÓN
# ─────────────────────────────────────────────

def load_and_combine_all() -> pd.DataFrame:
    """
    Carga todos los _all.csv extraídos + EURUSD1.csv.
    Los concatena en orden cronológico y elimina duplicados.

    El orden de carga importa para dedup: si hay solapamiento entre
    el EURUSD1.csv (datos recientes) y los yearly, se prefiere el yearly
    porque es la fuente primaria (se cargan primero).
    """
    frames: list[pd.DataFrame] = []

    # a) Yearly extracts (fuente primaria, 2016-2025)
    yearly = sorted(settings.EXTRACTED_DIR.glob("EURUSD_*_all.csv"))
    for fp in yearly:
        try:
            frames.append(load_csv(fp))
        except Exception as e:
            logger.warning(f"  Error cargando {fp.name}: {e}")

    # b) EURUSD1.csv (datos más recientes, puede solapar)
    recent = settings.RAW_DIR / "EURUSD1.csv"
    if recent.exists():
        try:
            frames.append(load_csv(recent))
        except Exception as e:
            logger.warning(f"  Error cargando EURUSD1.csv: {e}")

    if not frames:
        raise FileNotFoundError(
            f"No se encontraron datos en {settings.RAW_DIR} ni {settings.EXTRACTED_DIR}"
        )

    combined = pd.concat(frames, axis=0)
    combined.sort_index(inplace=True)

    # Dedup: keep='first' → prefiere el dato cargado primero (yearly)
    n_before = len(combined)
    combined = combined[~combined.index.duplicated(keep='first')]
    n_dupes = n_before - len(combined)

    logger.info(f"Combinado: {len(combined):,} barras "
                f"({n_dupes:,} duplicados eliminados)")
    logger.info(f"Rango: {combined.index[0]} → {combined.index[-1]}")

    return combined


# ─────────────────────────────────────────────
# 4. LIMPIEZA
# ─────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza del dataset:
      1. Ordenar por índice (defensivo).
      2. Eliminar duplicados de timestamp.
      3. Eliminar filas con OHLC completamente NaN.
      4. Forward-fill NaN sparse (raro pero posible).
      5. Asegurar tipos float64.
    """
    df = df.copy()
    df.sort_index(inplace=True)

    # Duplicados
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        logger.info(f"  Eliminando {n_dup} timestamps duplicados.")
        df = df[~df.index.duplicated(keep='first')]

    # Filas completamente vacías
    ohlc = ['open', 'high', 'low', 'close']
    all_nan = df[ohlc].isna().all(axis=1)
    if all_nan.any():
        n_drop = all_nan.sum()
        logger.warning(f"  Eliminando {n_drop} filas con OHLC=NaN.")
        df = df[~all_nan]

    # Forward-fill sparse NaN
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        logger.warning(f"  Forward-filling {n_nan} NaN values.")
        df = df.ffill()

    # Verificar que no quedan NaN
    remaining = df.isna().sum().sum()
    if remaining > 0:
        # Si aún quedan (inicio del dataset), backfill
        logger.warning(f"  Backfilling {remaining} NaN restantes (inicio dataset).")
        df = df.bfill()

    return df


# ─────────────────────────────────────────────
# 5. VALIDACIÓN
# ─────────────────────────────────────────────

def validate(df: pd.DataFrame) -> dict:
    """
    Validación de integridad del dataset.

    Retorna dict con todas las métricas. NO lanza excepciones
    (solo logea), para que el usuario decida qué hacer.
    Excepción: si el dataset tiene < MIN_ROWS, sí lanza ValueError.

    Checks:
      - Tamaño mínimo
      - Índice monótono creciente
      - High >= Low (integridad OHLC)
      - High >= max(Open, Close), Low <= min(Open, Close)
      - Volume >= 0
      - Gaps temporales (separando weekends de intraday)
      - Estadísticas de precio
    """
    logger.info("=" * 55)
    logger.info("  VALIDACIÓN DEL DATASET")
    logger.info("=" * 55)

    r: dict = {}

    # ── Tamaño ──
    r['total_bars'] = len(df)
    r['start'] = str(df.index[0])
    r['end'] = str(df.index[-1])
    r['calendar_days'] = (df.index[-1] - df.index[0]).days
    logger.info(f"  Barras:    {r['total_bars']:>12,}")
    logger.info(f"  Período:   {r['start']} → {r['end']}")
    logger.info(f"  Días cal:  {r['calendar_days']:>12,}")

    if r['total_bars'] < settings.MIN_ROWS:
        raise ValueError(
            f"Dataset demasiado pequeño: {r['total_bars']} "
            f"(mínimo: {settings.MIN_ROWS})"
        )

    # ── Índice monótono ──
    r['monotonic'] = bool(df.index.is_monotonic_increasing)
    icon = "✅" if r['monotonic'] else "❌"
    logger.info(f"  {icon} Índice monótono creciente: {r['monotonic']}")

    # ── OHLC integridad ──
    hl = (df['high'] < df['low']).sum()
    ho = (df['high'] < df['open']).sum()
    hc = (df['high'] < df['close']).sum()
    lo = (df['low'] > df['open']).sum()
    lc = (df['low'] > df['close']).sum()
    r['high_lt_low'] = int(hl)
    r['high_lt_open'] = int(ho)
    r['high_lt_close'] = int(hc)
    r['low_gt_open'] = int(lo)
    r['low_gt_close'] = int(lc)

    if hl == 0:
        logger.info("  ✅ OHLC: High ≥ Low en todas las barras")
    else:
        logger.error(f"  ❌ {hl} barras con High < Low")

    ohlc_warnings = ho + hc + lo + lc
    if ohlc_warnings > 0:
        logger.warning(f"  ⚠️  {ohlc_warnings} advertencias OHLC menores "
                       f"(H<O:{ho}, H<C:{hc}, L>O:{lo}, L>C:{lc})")

    # ── Volumen ──
    neg_vol = (df['volume'] < 0).sum()
    zero_vol = (df['volume'] == 0).sum()
    r['neg_volume'] = int(neg_vol)
    r['zero_volume'] = int(zero_vol)
    r['zero_vol_pct'] = round(zero_vol / len(df) * 100, 2)

    if neg_vol == 0:
        logger.info("  ✅ Volumen: sin valores negativos")
    else:
        logger.error(f"  ❌ {neg_vol} barras con volumen negativo")

    if zero_vol > 0:
        logger.info(f"  ℹ️  {zero_vol} barras con volumen 0 "
                     f"({r['zero_vol_pct']}%) — normal en horas inactivas")

    # ── Gaps temporales ──
    td = df.index.to_series().diff()
    regular = (td == pd.Timedelta(minutes=1)).sum()
    r['pct_1min'] = round(regular / (len(df) - 1) * 100, 2)

    # Separar weekends (>24h) de gaps intraday (>5min, <24h)
    gaps_all   = td[td > pd.Timedelta(minutes=5)]
    gaps_big   = td[td > pd.Timedelta(hours=24)]
    gaps_intra = gaps_all[~gaps_all.index.isin(gaps_big.index)]

    r['gaps_total'] = len(gaps_all)
    r['gaps_weekend']  = len(gaps_big)
    r['gaps_intraday'] = len(gaps_intra)
    r['largest_gap']   = str(td.max())

    logger.info(f"  ℹ️  Intervalos exactos 1-min: {r['pct_1min']}%")
    logger.info(f"  ℹ️  Gaps > 5 min: {r['gaps_total']} "
                f"(weekends: {r['gaps_weekend']}, intraday: {r['gaps_intraday']})")
    logger.info(f"  ℹ️  Gap más grande: {r['largest_gap']}")

    # ── Precio ──
    r['price_min'] = round(float(df['close'].min()), 5)
    r['price_max'] = round(float(df['close'].max()), 5)
    logger.info(f"  ℹ️  Rango precio: {r['price_min']} → {r['price_max']}")

    logger.info("=" * 55)
    return r


# ─────────────────────────────────────────────
# 6. PERSISTENCIA (PARQUET)
# ─────────────────────────────────────────────

def save_processed(
    df: pd.DataFrame,
    filename: str = "EURUSD_M1.parquet",
) -> Path:
    """Guarda DataFrame limpio como Parquet."""
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.PROCESSED_DIR / filename
    df.to_parquet(path, engine='pyarrow')
    mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"  Guardado: {path}  ({mb:.1f} MB)")
    return path


def load_processed(
    filename: str = "EURUSD_M1.parquet",
) -> pd.DataFrame:
    """Carga Parquet procesado."""
    path = settings.PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    df = pd.read_parquet(path, engine='pyarrow')
    logger.info(f"  Cargado: {len(df):,} barras desde {path.name}")
    return df
