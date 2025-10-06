import os

def create_required_folders():
    """Gerekli tüm klasörleri oluştur"""
    folders = [
        "data/ai_models_v2",
        "data/historical",
        "data/predictions",
        "data/team_stats"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ {folder} klasörü oluşturuldu")
    
    print("📍 Tüm klasörler hazır!")

if __name__ == "__main__":
    create_required_folders()
