# auto_train.py
# Basit otomasyon: çağrıldığında tüm modeli yeniden eğitir ve raporlar
import json
from datetime import datetime
from model_trainer import train_all

def main():
    summary = train_all(raw_path="data/raw", top_scores_k=20)
    print("=== Predicta ML v2 Güncelleme ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
