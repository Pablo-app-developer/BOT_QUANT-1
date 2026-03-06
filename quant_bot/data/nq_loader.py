"""
nq_loader.py — Loader nativo para datos Dukascopy .bi5 del USATECHIDXUSD.

Formato .bi5 (Dukascopy tick data):
  - Compresión LZMA
  - Registros binarios de 20 bytes cada uno:
      [0:4]   uint32 BE — milliseconds from start of hour
      [4:8]   uint32 BE — Ask (price * 100000)
      [8:12]  uint32 BE — Bid (price * 100000)
      [12:16] float  BE — Ask volume
      [16:20] float  BE — Bid volume

Flujo:
  1. parse_bi5_file()       → DataFrame de ticks con timestamp UTC
  2. ticks_to_ohlcv_m1()   → Agrupación en barras de 1 minuto
  3. load_nq_day()          → Carga todos los archivos de un día
  4. load_nq_month()        → Carga todos los días de un mes
  5. load_nq_year()         → Carga todos los meses de un año
  6. load_nq_m1()           → Función principal (multi-año)
  7. save_nq_parquet()      → Persistir resultado
  8. load_nq_parquet()      → Leer desde cache
"""

import lzma
import struct
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
NQ_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "USATECHIDXUSD"
NQ_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

# Factor de escala del precio en BI5: precio * 100000 → valor entero
BI5_PRICE_FACTOR = 100_000.0

# Cada tick ocupa 20 bytes en el formato Dukascopy
BI5_TICK_BYTES = 20


# ──────────────────────────────────────────────
# 1. PARSEO DE UN ARCHIVO .BI5
# ──────────────────────────────────────────────

def parse_bi5_file(filepath: Path, hour_utc: datetime) -> pd.DataFrame:
    """
    Parsea un archivo .bi5 y retorna DataFrame de ticks.

    Args:
        filepath: Ruta al archivo .bi5
        hour_utc: datetime UTC del inicio de la hora correspondiente al archivo

    Returns:
        DataFrame con columnas [ask, bid, ask_vol, bid_vol] e index DatetimeIndex UTC.
        Retorna DataFrame vacío si el archivo no existe o es inválido.
    """
    if not filepath.exists():
        return pd.DataFrame()

    try:
        with lzma.open(filepath, 'rb') as f:
            raw = f.read()
    except (lzma.LZMAError, EOFError, OSError) as e:
        logger.debug(f"  Archivo inválido {filepath.name}: {e}")
        return pd.DataFrame()

    n_ticks = len(raw) // BI5_TICK_BYTES
    if n_ticks == 0:
        return pd.DataFrame()

    # Parsear todos los ticks de una vez con numpy (muy rápido)
    # '>u4' = Big-Endian uint32 (formato Dukascopy)
    data = np.frombuffer(raw[:n_ticks * BI5_TICK_BYTES], dtype='>u4').reshape(n_ticks, 5)

    # Convertir a native endian (Little-Endian en x86) para compatibilidad con pandas
    # np.array() con dtype nativo hace la conversión automáticamente
    ms_offsets = np.array(data[:, 0], dtype=np.int64)      # ms desde inicio de hora
    ask_raw    = np.array(data[:, 1], dtype=np.float64)
    bid_raw    = np.array(data[:, 2], dtype=np.float64)

    # Volúmenes float32 BE — parsear e inmediatamente convertir a native float32
    vol_full = np.frombuffer(raw[:n_ticks * BI5_TICK_BYTES], dtype='>f4')
    ask_vol  = np.array(vol_full[3::5], dtype=np.float32)
    bid_vol  = np.array(vol_full[4::5], dtype=np.float32)

    # Timestamps: inicio de la hora + offset en ms
    base_ns = int(hour_utc.timestamp() * 1e9)
    ts_ns   = base_ns + ms_offsets * 1_000_000   # ms → ns

    # Mid price (promedio bid/ask) escalado
    ask_price = ask_raw / BI5_PRICE_FACTOR
    bid_price = bid_raw / BI5_PRICE_FACTOR

    index = pd.to_datetime(ts_ns, unit='ns', utc=True)

    df = pd.DataFrame({
        'ask': ask_price,
        'bid': bid_price,
        'mid': (ask_price + bid_price) / 2.0,
        'ask_vol': ask_vol,
        'bid_vol': bid_vol,
        'spread': ask_price - bid_price,
    }, index=index)

    return df


# ──────────────────────────────────────────────
# 2. TICKS → OHLCV M1
# ──────────────────────────────────────────────

def ticks_to_ohlcv_m1(ticks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega ticks en barras OHLCV de 1 minuto usando el precio Mid.

    Volume = suma de (ask_vol + bid_vol) por barra.
    Spread = spread promedio por barra.
    """
    if ticks_df.empty:
        return pd.DataFrame()

    agg = {
        'mid':  ['first', 'max', 'min', 'last'],
        'ask_vol': 'sum',
        'bid_vol': 'sum',
        'spread': 'mean',
    }

    ohlcv = ticks_df.resample('1min').agg(agg)
    ohlcv.columns = ['open', 'high', 'low', 'close', 'ask_vol', 'bid_vol', 'spread_avg']
    ohlcv['volume'] = ohlcv['ask_vol'] + ohlcv['bid_vol']

    # Eliminar barras sin ticks (NaN en open = no hubo ticks ese minuto)
    ohlcv = ohlcv.dropna(subset=['open'])
    ohlcv = ohlcv[['open', 'high', 'low', 'close', 'volume', 'spread_avg']]

    return ohlcv


# ──────────────────────────────────────────────
# 3. CARGA DE UN DÍA
# ──────────────────────────────────────────────

def load_nq_day(year: int, month: int, day: int) -> pd.DataFrame:
    """
    Carga todos los archivos hora de un día específico.
    Los archivos van de 00h a 23h (24 archivos por día).
    """
    day_dir = NQ_DATA_ROOT / f"{year:04d}" / f"{month:02d}" / f"{day:02d}"

    if not day_dir.exists():
        return pd.DataFrame()

    tick_frames = []

    for hour in range(24):
        filepath = day_dir / f"{hour:02d}h_ticks.bi5"
        hour_utc = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)
        ticks = parse_bi5_file(filepath, hour_utc)
        if not ticks.empty:
            tick_frames.append(ticks)

    if not tick_frames:
        return pd.DataFrame()

    all_ticks = pd.concat(tick_frames, axis=0)
    all_ticks.sort_index(inplace=True)

    return ticks_to_ohlcv_m1(all_ticks)


# ──────────────────────────────────────────────
# 4. CARGA DE UN MES
# ──────────────────────────────────────────────

def load_nq_month(year: int, folder_month: int) -> pd.DataFrame:
    """
    Carga todos los días de un mes completo.

    NOTA: Dukascopy usa meses 0-indexed en carpetas (00=Jan, 11=Dec).
    folder_month es el número de carpeta (0-11).
    calendar_month = folder_month + 1  (para datetime).
    """
    calendar_month = folder_month + 1
    month_dir = NQ_DATA_ROOT / f"{year:04d}" / f"{folder_month:02d}"

    if not month_dir.exists():
        logger.debug(f"  Mes no encontrado: {year}/{folder_month:02d}")
        return pd.DataFrame()

    day_dirs = sorted(month_dir.iterdir())
    frames = []

    for day_dir in day_dirs:
        if not day_dir.is_dir():
            continue
        try:
            day = int(day_dir.name)
        except ValueError:
            continue

        # Los días SÍ son 1-indexed en Dukascopy (01, 02, ... 31)
        day_df = load_nq_day(year, calendar_month, day)
        if not day_df.empty:
            frames.append(day_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=0).sort_index()


# ──────────────────────────────────────────────
# 5. CARGA DE UN AÑO
# ──────────────────────────────────────────────

def load_nq_year(year: int) -> pd.DataFrame:
    """
    Carga todos los meses de un año completo.

    NOTA: Dukascopy carpetas de mes son 0-indexed: 00=Jan, 01=Feb ... 11=Dec.
    Solo se procesan carpetas con nombre numérico entre 00 y 11.
    """
    year_dir = NQ_DATA_ROOT / f"{year:04d}"

    if not year_dir.exists():
        logger.warning(f"  Año no encontrado: {year}")
        return pd.DataFrame()

    month_dirs = sorted(year_dir.iterdir())
    frames = []

    for month_dir in month_dirs:
        if not month_dir.is_dir():
            continue
        try:
            folder_month = int(month_dir.name)
        except ValueError:
            continue

        # Solo meses válidos 0-11 (Dukascopy 0-indexed)
        if folder_month < 0 or folder_month > 11:
            continue

        cal_month = folder_month + 1
        logger.info(f"  Cargando {year}/{folder_month:02d} (mes calendario {cal_month:02d})...")
        mdf = load_nq_month(year, folder_month)
        if not mdf.empty:
            logger.info(f"    → {len(mdf):,} barras M1")
            frames.append(mdf)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0).sort_index()

    # Eliminar duplicados
    n_before = len(combined)
    combined = combined[~combined.index.duplicated(keep='first')]
    n_dupes = n_before - len(combined)
    if n_dupes > 0:
        logger.warning(f"  {n_dupes} duplicados eliminados en {year}")

    return combined


# ──────────────────────────────────────────────
# 6. FUNCIÓN PRINCIPAL — MULTI-AÑO
# ──────────────────────────────────────────────

def load_nq_m1(
    years: list[int] | None = None,
    use_cache: bool = True,
    cache_name: str = "USATECHIDXUSD_M1.parquet"
) -> pd.DataFrame:
    """
    Carga datos M1 del USATECHIDXUSD para los años especificados.

    Args:
        years: Lista de años a cargar. Si es None, carga todos los disponibles.
        use_cache: Si existe parquet cacheado, usar ese en lugar de re-parsear.
        cache_name: Nombre del archivo parquet de caché.

    Returns:
        DataFrame con columnas [open, high, low, close, volume, spread_avg]
        e index DatetimeIndex UTC.
    """
    cache_path = NQ_PROCESSED / cache_name

    # Intentar cargar desde caché
    if use_cache and cache_path.exists():
        logger.info(f"  Cargando desde caché: {cache_path}")
        df = pd.read_parquet(cache_path, engine='pyarrow')
        logger.info(f"  → {len(df):,} barras M1 cargadas desde cache")
        return df

    # Detectar años disponibles si no se especificaron
    if years is None:
        available = sorted([
            int(d.name) for d in NQ_DATA_ROOT.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])
        years = available
        logger.info(f"  Años disponibles: {years}")

    logger.info("=" * 55)
    logger.info("  NQ LOADER — USATECHIDXUSD M1")
    logger.info("=" * 55)
    logger.info(f"  Años a cargar: {years}")

    all_frames = []
    for year in years:
        logger.info(f"\n{'─'*40}")
        logger.info(f"  AÑO {year}")
        df_year = load_nq_year(year)
        if not df_year.empty:
            logger.info(f"  Total año {year}: {len(df_year):,} barras")
            all_frames.append(df_year)
        else:
            logger.warning(f"  Año {year}: sin datos")

    if not all_frames:
        raise FileNotFoundError(f"No se encontraron datos en {NQ_DATA_ROOT}")

    combined = pd.concat(all_frames, axis=0).sort_index()

    # Dedup final
    n_before = len(combined)
    combined = combined[~combined.index.duplicated(keep='first')]
    logger.info(f"\n  TOTAL: {len(combined):,} barras M1")
    logger.info(f"  Período: {combined.index[0]} → {combined.index[-1]}")
    logger.info(f"  Duplicados eliminados: {n_before - len(combined)}")

    # Validación básica de integridad OHLC
    invalid = (combined['high'] < combined['low']).sum()
    if invalid > 0:
        logger.error(f"  ⚠️  {invalid} barras con High < Low detectadas!")

    return combined


# ──────────────────────────────────────────────
# 7. PERSISTENCIA
# ──────────────────────────────────────────────

def save_nq_parquet(
    df: pd.DataFrame,
    name: str = "USATECHIDXUSD_M1.parquet"
) -> Path:
    """Guarda DataFrame M1 como Parquet en el directorio processed."""
    NQ_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = NQ_PROCESSED / name
    df.to_parquet(path, engine='pyarrow')
    mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"  Guardado: {path}  ({mb:.1f} MB)")
    return path


def load_nq_parquet(
    name: str = "USATECHIDXUSD_M1.parquet"
) -> pd.DataFrame:
    """Carga DataFrame M1 desde Parquet en processed."""
    path = NQ_PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta load_nq_m1() primero para generar el cache."
        )
    df = pd.read_parquet(path, engine='pyarrow')
    logger.info(f"  Cargado: {len(df):,} barras M1 desde {path.name}")
    return df


# ──────────────────────────────────────────────
# 8. UTILIDADES DE SESIÓN
# ──────────────────────────────────────────────

def add_session_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas de sesión y metadatos temporales al DataFrame M1.

    Columnas añadidas:
      - session:    OVERNIGHT / PRE_OPEN / OPEN_HOUR / MIDDAY / CLOSE_HOUR / AFTER_HOURS
      - is_ny_session: True si está en horario principal de NY (13:30-20:00 UTC)
      - day_of_week: 0=Mon, 4=Fri
      - hour_utc:   Hora UTC
    """
    df = df.copy()
    t = df.index.hour * 60 + df.index.minute  # minutos desde medianoche UTC

    df['session'] = 'OVERNIGHT'
    df.loc[(t >= 12*60) & (t < 13*60+30), 'session'] = 'PRE_OPEN'
    df.loc[(t >= 13*60+30) & (t < 14*60+30), 'session'] = 'OPEN_HOUR'
    df.loc[(t >= 14*60+30) & (t < 18*60), 'session'] = 'MIDDAY'
    df.loc[(t >= 18*60) & (t < 20*60), 'session'] = 'CLOSE_HOUR'
    df.loc[(t >= 20*60) & (t < 21*60), 'session'] = 'AFTER_HOURS'

    df['is_ny_session'] = df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])
    df['day_of_week'] = df.index.dayofweek  # 0=Mon, 4=Fri
    df['hour_utc'] = df.index.hour

    return df


def get_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula resumen diario del DataFrame M1:
    - open diario (primer bar de NY session = 13:30 UTC)
    - high/low diario
    - close diario (último bar de NY session)
    - retorno total del día
    - retorno primera hora
    - overnight return
    """
    # Usar solo sesión NY principal
    ny = df[df['session'].isin(['OPEN_HOUR', 'MIDDAY', 'CLOSE_HOUR'])]

    daily = ny.resample('D').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna(subset=['open'])

    daily['day_return'] = daily['close'].pct_change()

    # Primera hora: retorno desde 13:30 hasta 14:30 UTC
    open_hour = df[df['session'] == 'OPEN_HOUR'].resample('D').agg({
        'open':  'first',
        'close': 'last',
    }).dropna(subset=['open'])
    open_hour.columns = ['oh_open', 'oh_close']

    daily = daily.join(open_hour, how='left')
    daily['first_hour_return'] = (daily['oh_close'] - daily['oh_open']) / daily['oh_open']

    # Overnight return (open hoy - close ayer) / close ayer
    daily['overnight_return'] = (daily['open'] - daily['close'].shift(1)) / daily['close'].shift(1)

    return daily


# ──────────────────────────────────────────────
# MAIN — CLI para construir el cache
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("================================================")
    logger.info("  NQ DATA PIPELINE — BI5 → PARQUET M1")
    logger.info("================================================")

    # Años a cargar (puede ser overrideado por argumento)
    years_arg = None
    if len(sys.argv) > 1:
        years_arg = [int(y) for y in sys.argv[1:]]

    df = load_nq_m1(years=years_arg, use_cache=False)

    logger.info("\n  Añadiendo etiquetas de sesión...")
    df = add_session_labels(df)

    logger.info("\n  Guardando Parquet con sesiones...")
    path = save_nq_parquet(df)

    logger.info(f"\n  ✅ Pipeline completado.")
    logger.info(f"  Barras totales: {len(df):,}")
    logger.info(f"  Archivo: {path}")
    logger.info(f"  Período: {df.index[0]} → {df.index[-1]}")

    # Resumen por sesión
    session_counts = df['session'].value_counts()
    logger.info("\n  Barras por sesión:")
    for sess, cnt in session_counts.items():
        logger.info(f"    {sess:15s}: {cnt:>10,}")
