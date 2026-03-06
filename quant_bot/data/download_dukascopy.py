"""
download_dukascopy.py — Descarga histórica paralela de Dukascopy (.bi5)
"""

import sys
import os
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
import concurrent.futures
import threading

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DukascopyDownloader")

def download_file(url, target_path):
    if target_path.exists() and target_path.stat().st_size > 0:
        return True, "EXISTS"
        
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Intento de descarga
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    with open(target_path, 'wb') as f:
                        f.write(response.read())
                    return True, "DOWNLOADED"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, "404"  # Fin de semana / festivo / no hay datos
        except Exception:
            pass
            
    return False, "ERROR"

def get_hours_in_year(year):
    start = datetime(year, 1, 1)
    # Queremos hasta el 31 de Diciembre
    end = datetime(year + 1, 1, 1)
    
    curr = start
    hours = []
    while curr < end:
        # Dukascopy indexa meses 00 a 11
        # Y dias 01 a 31
        month_str = f"{curr.month - 1:02d}"
        day_str = f"{curr.day:02d}"
        hour_str = f"{curr.hour:02d}h_ticks.bi5"
        
        url_path = f"{year}/{month_str}/{day_str}/{hour_str}"
        hours.append((curr, url_path))
        curr += timedelta(hours=1)
        
    return hours

def download_year(instrument, year, base_dir, max_workers=20):
    hours = get_hours_in_year(year)
    base_url = f"http://datafeed.dukascopy.com/datafeed/{instrument}/"
    inst_dir = base_dir / instrument
    
    logger.info(f"Iniciando descarga de {instrument} para el año {year} ({len(hours)} horas posibles)...")
    
    success_count = 0
    missing_count = 0
    
    lock = threading.Lock()
    done = 0
    
    def process_hour(item):
        nonlocal success_count, missing_count, done
        dt, url_suffix = item
        url = base_url + url_suffix
        target_path = inst_dir / url_suffix
        
        ok, status = download_file(url, target_path)
        
        with lock:
            done += 1
            if ok:
                success_count += 1
            else:
                missing_count += 1
                
            if done % 1000 == 0 or done == len(hours):
                print(f"Progreso {year}: {done}/{len(hours)} | OK: {success_count} | 404: {missing_count}", end='\r', flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_hour, hours)
        
    print() # newline
    logger.info(f"Finalizado {year}: {success_count} descargas exitosas, {missing_count} ausentes (fin de semana/festivo).")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instrument', required=True, help="Ej: USA500IDXUSD, USA30IDXUSD")
    parser.add_argument('--years', required=True, help="Rango. Ej: 2021-2025")
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    
    if "-" in args.years:
        curr, end = map(int, args.years.split("-"))
        years = list(range(curr, end + 1))
    else:
        years = [int(args.years)]
        
    for y in years:
        download_year(args.instrument, y, base_dir, max_workers=args.workers)

if __name__ == "__main__":
    main()
