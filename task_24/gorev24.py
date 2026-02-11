import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QMessageBox, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class DerinlikUygulamasi(QWidget):
    def __init__(self):
        super().__init__()
        self.pencere_ayarlari()
        self.degiskenleri_baslat()
        self.arayuz_olustur()
        
        # Uygulama açılınca modeli indirip hazırlar
        self.modeli_yukle()

    def pencere_ayarlari(self):
        self.setWindowTitle('PyQt5 Derinlik Tahmini (Depth Estimation V2)')
        self.setGeometry(100, 100, 1100, 600)
        self.setStyleSheet("background-color: #2c3e50; color: white; font-size: 14px;")

    def degiskenleri_baslat(self):
        self.secilen_resim_path = None
        self.model = None
        self.transform = None
        # GPU varsa kullan, yoksa CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Kullanılan cihaz: {self.device}")

    def modeli_yukle(self):
        try:
            print("MiDaS (Depth Estimation) modeli yükleniyor... Lütfen bekleyin.")
            # Torch Hub üzerinden MiDaS Small modelini çekiyoruz (Hızlı ve etkili)
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.to(self.device)
            self.model.eval()

            # MiDaS için gerekli özel transform işlemleri
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform

            print("Model ve Transformlar hazır!")
        except Exception as e:
            QMessageBox.critical(self, "Model Hatası", f"Model indirilemedi:\n{e}")

    def arayuz_olustur(self):
        ana_duzen = QVBoxLayout()

        # --- BAŞLIK ---
        lbl_baslik = QLabel("Yapay Zeka Destekli Derinlik Analizi")
        lbl_baslik.setAlignment(Qt.AlignCenter)
        lbl_baslik.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px; color: #ecf0f1;")
        ana_duzen.addWidget(lbl_baslik)

        # --- BUTONLAR ---
        buton_paneli = QHBoxLayout()
        
        self.btn_resim_yukle = QPushButton("🖼️ Resim Yükle")
        self.btn_resim_yukle.clicked.connect(self.resim_yukle)
        self.btn_resim_yukle.setStyleSheet("background-color: #2980b9; padding: 10px; border-radius: 5px;")

        self.btn_calistir = QPushButton("🚀 Derinlik Hesapla")
        self.btn_calistir.clicked.connect(self.derinlik_hesapla)
        self.btn_calistir.setStyleSheet("background-color: #e67e22; padding: 10px; border-radius: 5px;")

        buton_paneli.addWidget(self.btn_resim_yukle)
        buton_paneli.addWidget(self.btn_calistir)
        
        # --- GÖRÜNTÜ ALANLARI ---
        resim_paneli = QHBoxLayout()

        # Sol: Orijinal
        self.lbl_orijinal = QLabel("Orijinal Görüntü")
        self.lbl_orijinal.setAlignment(Qt.AlignCenter)
        self.lbl_orijinal.setFrameShape(QFrame.Box)
        self.lbl_orijinal.setMinimumSize(450, 400)
        self.lbl_orijinal.setStyleSheet("border: 2px solid #95a5a6; background-color: #34495e;")

        # Sağ: Derinlik Haritası
        self.lbl_derinlik = QLabel("Derinlik Çıktısı (Heatmap)")
        self.lbl_derinlik.setAlignment(Qt.AlignCenter)
        self.lbl_derinlik.setFrameShape(QFrame.Box)
        self.lbl_derinlik.setMinimumSize(450, 400)
        self.lbl_derinlik.setStyleSheet("border: 2px solid #95a5a6; background-color: #34495e;")

        resim_paneli.addWidget(self.lbl_orijinal)
        resim_paneli.addWidget(self.lbl_derinlik)

        # --- DÜZENİ BİRLEŞTİR ---
        ana_duzen.addLayout(buton_paneli)
        ana_duzen.addLayout(resim_paneli)
        self.setLayout(ana_duzen)

    def resim_yukle(self):
        dosya, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resim Dosyaları (*.jpg *.png *.jpeg)")
        if dosya:
            self.secilen_resim_path = dosya
            pixmap = QPixmap(dosya)
            self.lbl_orijinal.setPixmap(pixmap.scaled(450, 400, Qt.KeepAspectRatio))
            self.lbl_derinlik.setText("Analiz bekleniyor...")

    def derinlik_hesapla(self):
        if not self.secilen_resim_path:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin!")
            return

        if self.model is None:
            QMessageBox.warning(self, "Hata", "Model henüz hazır değil.")
            return

        # 1. GÜVENLİ RESİM OKUMA (Türkçe karakter destekli)
        try:
            with open(self.secilen_resim_path, "rb") as f:
                bytes_data = bytearray(f.read())
                numpy_array = np.asarray(bytes_data, dtype=np.uint8)
                img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Resim okunamadı: {e}")
            return

        # 2. MODEL İÇİN HAZIRLIK (Transform)
        input_batch = self.transform(img).to(self.device)

        # 3. TAHMİN (Inference)
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Çıktıyı orijinal resim boyutuna geri büyüt
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # 4. GÖRSELLEŞTİRME (Normalizasyon + Renklendirme)
        # Derinlik verisi ham float gelir, bunu 0-255 arasına sıkıştırıp resme çevirmeliyiz.
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        # Min-Max Normalizasyon
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        normalized_depth = (normalized_depth * 255).astype(np.uint8)

        # Renkli Harita (Heatmap) Uygulama (MAGMA veya INFERNO güzel görünür)
        depth_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_MAGMA)
        
        # OpenCV BGR döndürür, PyQt RGB ister
        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        # 5. EKRANA BASMA
        height, width, channel = depth_colored_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(depth_colored_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.lbl_derinlik.setPixmap(QPixmap.fromImage(q_img).scaled(450, 400, Qt.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pencere = DerinlikUygulamasi()
    pencere.show()
    sys.exit(app.exec_())