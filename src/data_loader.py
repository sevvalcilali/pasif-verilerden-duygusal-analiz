import pandas as pd
from src.config import PATHS

def load_sensing() -> pd.DataFrame:
    """Ana sensör tablosunu yükle (sensing.csv)."""
    print("  [+] sensing.csv yükleniyor...")
    df = pd.read_csv(PATHS["sensing"], low_memory=False)
    df["day"] = pd.to_datetime(df["day"], format="%Y%m%d", errors="coerce")
    print(f"      {df.shape[0]:,} satır, {df.shape[1]} sütun, {df['uid'].nunique()} kullanıcı")
    return df

def load_steps() -> pd.DataFrame:
    """Adım sayısı verisini yükle (steps.csv)."""
    print("  [+] steps.csv yükleniyor...")
    df = pd.read_csv(PATHS["steps"], low_memory=False)
    df["day"] = pd.to_datetime(df["day"], format="%Y%m%d", errors="coerce")
    print(f"      {df.shape[0]:,} satır, {df['uid'].nunique()} kullanıcı")
    return df

def load_all() -> dict[str, pd.DataFrame]:
    """Sensing ve steps veri setlerini yükleyip sözlük olarak döndür."""
    print("\n=== VERİ YÜKLENİYOR ===")
    datasets = {
        "sensing": load_sensing(),
        "steps":   load_steps(),
    }
    print("=== YÜKLENİŞ TAMAMLANDI ===\n")
    return datasets
