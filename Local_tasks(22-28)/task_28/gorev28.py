from ultralytics import YOLO
import shutil
import os

# --- AYARLAR ---
hedef_klasor = r"C:\Users\hamdi\OneDrive\Masaüstü\Aware Robotics\gorev28test"
model_adi = "yolov8n.onnx"

# 1. Modeli Dönüştür (Export) - DÜZELTME BURADA
model = YOLO('yolov8n.pt')

# 'opset=17' parametresini ekledik. 
# Bu sayede onnxruntime hatası almayacaksınız.
success = model.export(format='onnx', opset=17) 

print("Dönüşüm tamamlandı, dosya taşınıyor...")

# 2. Hedef Klasörü Kontrol Et
if not os.path.exists(hedef_klasor):
    os.makedirs(hedef_klasor)

# 3. Dosyayı Taşı (Eskisinin üzerine yazar)
kaynak_dosya = model_adi 
yeni_konum = os.path.join(hedef_klasor, model_adi)

try:
    # Eğer eski hatalı dosya varsa sil, yenisini koy
    if os.path.exists(yeni_konum):
        os.remove(yeni_konum)
        
    shutil.move(kaynak_dosya, yeni_konum)
    print(f"✅ BAŞARILI! Uyumlu model kaydedildi:\n{yeni_konum}")
except Exception as e:
    print(f"❌ Dosya taşınırken hata oldu: {e}")