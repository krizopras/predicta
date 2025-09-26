# create_models_folder.py
import os

def create_models_folder():
    """ai_models_v2 klasörünü manuel oluştur"""
    
    folder_path = "data/ai_models_v2"
    
    try:
        # data/ klasörünü oluştur
        os.makedirs("data", exist_ok=True)
        print("✅ data/ klasörü oluşturuldu")
        
        # ai_models_v2/ klasörünü oluştur
        os.makedirs(folder_path, exist_ok=True)
        print(f"✅ {folder_path} klasörü oluşturuldu")
        
        # Boş bir .gitkeep dosyası oluştur (opsiyonel)
        with open(os.path.join(folder_path, ".gitkeep"), "w") as f:
            f.write("# AI models directory")
        print("✅ .gitkeep dosyası oluşturuldu")
        
        print(f"📍 Klasör konumu: {os.path.abspath(folder_path)}")
        
    except Exception as e:
        print(f"❌ Klasör oluşturma hatası: {e}")

if __name__ == "__main__":
    create_models_folder()