import sys
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QMessageBox, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class SegmentasyonUygulamasi(QWidget):
    def __init__(self):
        super().__init__()
        self.pencere_ayarlari()
        self.degiskenleri_baslat()
        self.arayuz_olustur()
        
        # Varsayılan hazır modeli arka planda indirelim (İnternet gerekir)
        self.varsayilan_modeli_yukle()

    def pencere_ayarlari(self):
        self.setWindowTitle('PyQt5 Semantik Segmentasyon Analizi')
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("background-color: #f0f0f0; font-size: 14px;")

    def degiskenleri_baslat(self):
        self.secilen_resim_path = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def varsayilan_modeli_yukle(self):
        # Başlangıçta torchvision'dan hazır eğitilmiş bir model yükleyelim
        try:
            print("Varsayılan model yükleniyor (FCN ResNet50)...")
            self.model = models.segmentation.fcn_resnet50(pretrained=True)
            self.model.eval()
            self.model.to(self.device)
            print("Model hazır!")
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")

    def arayuz_olustur(self):
        ana_duzen = QVBoxLayout()

        # --- ÜST PANEL (BUTONLAR) ---
        buton_paneli = QHBoxLayout()
        
        self.btn_model_sec = QPushButton("📂 Model Dosyası Seç (.pth)")
        self.btn_model_sec.clicked.connect(self.model_dosyasi_sec)
        self.btn_model_sec.setStyleSheet("background-color: #3498db; color: white; padding: 10px;")

        self.btn_resim_yukle = QPushButton("🖼️ Resim Yükle")
        self.btn_resim_yukle.clicked.connect(self.resim_yukle)
        self.btn_resim_yukle.setStyleSheet("background-color: #2ecc71; color: white; padding: 10px;")

        self.btn_calistir = QPushButton("🚀 Analizi Başlat")
        self.btn_calistir.clicked.connect(self.segmentasyon_yap)
        self.btn_calistir.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px;")

        buton_paneli.addWidget(self.btn_model_sec)
        buton_paneli.addWidget(self.btn_resim_yukle)
        buton_paneli.addWidget(self.btn_calistir)
        
        # --- ORTA PANEL (RESİMLER) ---
        resim_paneli = QHBoxLayout()

        # Orijinal Resim Alanı
        self.lbl_orijinal = QLabel("Orijinal Resim")
        self.lbl_orijinal.setAlignment(Qt.AlignCenter)
        self.lbl_orijinal.setFrameShape(QFrame.Box)
        self.lbl_orijinal.setMinimumSize(400, 400)
        self.lbl_orijinal.setStyleSheet("background-color: white; border: 2px dashed #aaa;")

        # İşlenmiş Resim Alanı
        self.lbl_islenmis = QLabel("Segmentasyon Çıktısı")
        self.lbl_islenmis.setAlignment(Qt.AlignCenter)
        self.lbl_islenmis.setFrameShape(QFrame.Box)
        self.lbl_islenmis.setMinimumSize(400, 400)
        self.lbl_islenmis.setStyleSheet("background-color: white; border: 2px solid #aaa;")

        resim_paneli.addWidget(self.lbl_orijinal)
        resim_paneli.addWidget(self.lbl_islenmis)

        # --- DÜZENİ BİRLEŞTİR ---
        ana_duzen.addLayout(buton_paneli)
        ana_duzen.addLayout(resim_paneli)
        self.setLayout(ana_duzen)

    def model_dosyasi_sec(self):
        dosya, _ = QFileDialog.getOpenFileName(self, "Model Seç", "", "Model Dosyaları (*.pth *.pt)")
        if dosya:
            try:
                # Kullanıcı kendi eğittiği bir modeli yüklemek isterse
                # Burada mimariyi bilmek gerekir, örnek olarak state_dict yüklüyoruz
                state_dict = torch.load(dosya, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                QMessageBox.information(self, "Başarılı", f"Model yüklendi:\n{dosya}")
            except Exception as e:
                QMessageBox.warning(self, "Hata", f"Model yüklenemedi. Varsayılan model kullanılacak.\nHata: {str(e)}")

    def resim_yukle(self):
        dosya, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resim Dosyaları (*.jpg *.png *.jpeg)")
        if dosya:
            self.secilen_resim_path = dosya
            pixmap = QPixmap(dosya)
            # Pencereye sığdır
            self.lbl_orijinal.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            self.lbl_islenmis.setText("Analiz bekleniyor...")

    def segmentasyon_yap(self):
        if not self.secilen_resim_path:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin!")
            return

        # 1. Resmi OpenCV ile oku ve hazırla
        img = cv2.imread(self.secilen_resim_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. PyTorch için ön işleme (Transform)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((400, 400)), # Hız için küçültüyoruz
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(img_rgb).unsqueeze(0).to(self.device)

        # 3. Model Tahmini (Inference)
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # 4. Maskeyi Renklendir
        # Her sınıf için rastgele bir renk atayan basit bir fonksiyon
        # COCO veri seti için 21 sınıf vardır (0: arka plan, 15: insan vb.)
        maske_renkli = self.maskeyi_renklendir(output_predictions)

        # 5. Sonucu Ekrana Bas (OpenCV -> QPixmap)
        height, width, channel = maske_renkli.shape
        bytes_per_line = 3 * width
        q_img = QImage(maske_renkli.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.lbl_islenmis.setPixmap(QPixmap.fromImage(q_img))

    def maskeyi_renklendir(self, prediction_mask):
        # 21 Sınıf için renk paleti (VOC/COCO standardı)
        VOC_COLORMAP = [
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]
        ]
        
        r = np.zeros_like(prediction_mask).astype(np.uint8)
        g = np.zeros_like(prediction_mask).astype(np.uint8)
        b = np.zeros_like(prediction_mask).astype(np.uint8)
        
        for l in range(0, 21):
            idx = prediction_mask == l
            r[idx] = VOC_COLORMAP[l][0]
            g[idx] = VOC_COLORMAP[l][1]
            b[idx] = VOC_COLORMAP[l][2]
            
        rgb = np.stack([r, g, b], axis=2)
        return rgb

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pencere = SegmentasyonUygulamasi()
    pencere.show()
    sys.exit(app.exec_())