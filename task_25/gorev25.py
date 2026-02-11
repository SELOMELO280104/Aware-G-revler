import sys
import cv2
import numpy as np
import torch
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QMessageBox, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class Derinlik3DMeshUygulamasi(QWidget):
    def __init__(self):
        super().__init__()
        self.pencere_ayarlari()
        self.degiskenleri_baslat()
        self.arayuz_olustur()
        
        # Model açılışta yüklenir
        self.modeli_yukle()

    def pencere_ayarlari(self):
        self.setWindowTitle('PyQt5 Derinlik ve 3D Mesh Oluşturucu')
        self.setGeometry(100, 100, 1100, 650)
        self.setStyleSheet("background-color: #2c3e50; color: white; font-size: 14px;")

    def degiskenleri_baslat(self):
        self.secilen_resim_path = None
        self.model = None
        self.transform = None
        self.ham_derinlik_verisi = None 
        self.orijinal_resim_rgb = None
        # GPU varsa kullan, yoksa CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"İşlem Birimi: {self.device}")

    def modeli_yukle(self):
        try:
            print("MiDaS (Depth Estimation) modeli indiriliyor/yükleniyor...")
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.to(self.device)
            self.model.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
            print("Model Başarıyla Hazırlandı!")
        except Exception as e:
            QMessageBox.critical(self, "Model Hatası", f"Model yüklenemedi:\n{e}")

    def arayuz_olustur(self):
        ana_duzen = QVBoxLayout()

        # --- BAŞLIK ---
        lbl_baslik = QLabel("Resimden Katı 3D Model (Mesh) Oluşturucu")
        lbl_baslik.setAlignment(Qt.AlignCenter)
        lbl_baslik.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px; color: #ecf0f1;")
        ana_duzen.addWidget(lbl_baslik)

        # --- BUTONLAR ---
        buton_paneli = QHBoxLayout()
        
        self.btn_resim_yukle = QPushButton("🖼️ Resim Yükle")
        self.btn_resim_yukle.clicked.connect(self.resim_yukle)
        self.btn_resim_yukle.setStyleSheet("background-color: #2980b9; padding: 12px; border-radius: 6px;")

        self.btn_calistir = QPushButton("🚀 Analiz Et")
        self.btn_calistir.clicked.connect(self.derinlik_hesapla)
        self.btn_calistir.setStyleSheet("background-color: #e67e22; padding: 12px; border-radius: 6px;")

        self.btn_3d_kaydet = QPushButton("🧊 3D Mesh Olarak Kaydet (.ply)")
        self.btn_3d_kaydet.clicked.connect(self.model_3d_olustur_ve_kaydet)
        self.btn_3d_kaydet.setEnabled(False) 
        self.btn_3d_kaydet.setStyleSheet("""
            QPushButton { background-color: #27ae60; padding: 12px; border-radius: 6px; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)

        buton_paneli.addWidget(self.btn_resim_yukle)
        buton_paneli.addWidget(self.btn_calistir)
        buton_paneli.addWidget(self.btn_3d_kaydet)
        
        # --- RESİM ÇERÇEVELERİ ---
        resim_paneli = QHBoxLayout()
        self.lbl_orijinal = self.cerceve_olustur("Orijinal Resim")
        self.lbl_derinlik = self.cerceve_olustur("Derinlik Haritası (Heatmap)")
        
        resim_paneli.addWidget(self.lbl_orijinal)
        resim_paneli.addWidget(self.lbl_derinlik)

        ana_duzen.addLayout(buton_paneli)
        ana_duzen.addLayout(resim_paneli)
        self.setLayout(ana_duzen)

    def cerceve_olustur(self, baslik):
        lbl = QLabel(baslik)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFrameShape(QFrame.Box)
        lbl.setMinimumSize(400, 350)
        lbl.setStyleSheet("border: 2px dashed #7f8c8d; background-color: #34495e;")
        return lbl

    def resim_yukle(self):
        dosya, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resim Dosyaları (*.jpg *.png *.jpeg)")
        if dosya:
            self.secilen_resim_path = dosya
            pixmap = QPixmap(dosya)
            self.lbl_orijinal.setPixmap(pixmap.scaled(400, 350, Qt.KeepAspectRatio))
            self.lbl_derinlik.setText("Analiz bekleniyor...")
            self.btn_3d_kaydet.setEnabled(False)

    def derinlik_hesapla(self):
        if not self.secilen_resim_path or self.model is None:
            return

        # 1. GÜVENLİ RESİM OKUMA (Türkçe karakter destekli)
        try:
            with open(self.secilen_resim_path, "rb") as f:
                bytes_data = bytearray(f.read())
                numpy_array = np.asarray(bytes_data, dtype=np.uint8)
                img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.orijinal_resim_rgb = img 
        except Exception as e:
            QMessageBox.critical(self, "Okuma Hatası", f"Resim okunamadı: {e}")
            return

        # 2. MODEL TAHMİNİ
        try:
            input_batch = self.transform(img).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                
                # Çıktıyı orijinal boyuta büyüt
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # 3. VERİYİ SAKLA
            depth_map = prediction.cpu().numpy()
            self.ham_derinlik_verisi = depth_map 

            # 4. GÖRSELLEŞTİR (Heatmap)
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
            normalized_depth = (normalized_depth * 255).astype(np.uint8)
            
            depth_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_MAGMA)
            depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

            h, w, c = depth_colored_rgb.shape
            q_img = QImage(depth_colored_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            self.lbl_derinlik.setPixmap(QPixmap.fromImage(q_img).scaled(400, 350, Qt.KeepAspectRatio))
            
            self.btn_3d_kaydet.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Analiz Hatası", f"Model çalışırken hata: {e}")

    def model_3d_olustur_ve_kaydet(self):
        """
        Noktaları birbirine bağlayarak 'Mesh' (Yüzey) oluşturur.
        Bu sayede Windows 3D Görüntüleyici dosyayı açabilir.
        """
        kayit_yolu, _ = QFileDialog.getSaveFileName(self, "3D Mesh Kaydet", "model_mesh.ply", "Polygon File Format (*.ply)")
        if not kayit_yolu:
            return

        try:
            QMessageBox.information(self, "İşleniyor", "Yüzeyler örülüyor (Meshing)...\nBu işlem resim boyutuna göre 5-10 saniye sürebilir.")
            QApplication.processEvents()

            # --- 1. VERİYİ KÜÇÜLTME (Optimize Etme) ---
            # Çok büyük resimler Windows Viewer'ı kilitler, o yüzden 400px genişliğe indiriyoruz.
            hedef_genislik = 400
            oran = hedef_genislik / self.orijinal_resim_rgb.shape[1]
            hedef_yukseklik = int(self.orijinal_resim_rgb.shape[0] * oran)
            
            # Resmi ve Derinliği küçült
            rgb_small = cv2.resize(self.orijinal_resim_rgb, (hedef_genislik, hedef_yukseklik))
            depth_small = cv2.resize(self.ham_derinlik_verisi, (hedef_genislik, hedef_yukseklik))

            height, width = depth_small.shape
            
            # --- 2. NOKTALARI (VERTICES) HAZIRLAMA ---
            x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
            
            x = x_grid.flatten()
            y = y_grid.flatten()
            z = depth_small.flatten() * 0.05  # Derinlik ölçek katsayısı (Kabartma miktarı)
            
            # Renkleri düzleştir
            colors = rgb_small.reshape(-1, 3)
            r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
            
            vertex_count = len(x)

            # --- 3. YÜZEYLERİ (FACES) ÖRME ---
            # Her 4 noktalı kareyi 2 üçgene bölüyoruz.
            idx_grid = np.arange(height * width).reshape(height, width)
            
            # Komşusu olan noktaları seç
            top_left = idx_grid[:-1, :-1].flatten()
            top_right = idx_grid[:-1, 1:].flatten()
            bottom_left = idx_grid[1:, :-1].flatten()
            bottom_right = idx_grid[1:, 1:].flatten()

            # Üçgenleri oluştur: (TL, BL, TR) ve (TR, BL, BR)
            triangles1 = np.column_stack((top_left, bottom_left, top_right))
            triangles2 = np.column_stack((top_right, bottom_left, bottom_right))
            
            faces = np.vstack((triangles1, triangles2))
            face_count = len(faces)

            # --- 4. DOSYAYA YAZMA ---
            with open(kayit_yolu, "w", encoding="utf-8") as f:
                # HEADER
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {vertex_count}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write(f"element face {face_count}\n") # Yüzey sayısı eklendi
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")

                # VERTICES (Noktalar)
                for i in range(vertex_count):
                    f.write(f"{x[i]:.2f} {y[i]:.2f} {z[i]:.2f} {r[i]} {g[i]} {b[i]}\n")
                
                # FACES (Yüzeyler)
                # Format: 3 <idx1> <idx2> <idx3>
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

            QMessageBox.information(self, "Başarılı", f"Mesh modeli kaydedildi!\nDosya Yolu: {kayit_yolu}\n\nArtık Windows ile açabilirsiniz.")

        except Exception as e:
            QMessageBox.critical(self, "Kaydetme Hatası", f"Dosya yazılamadı:\n{e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pencere = Derinlik3DMeshUygulamasi()
    pencere.show()
    sys.exit(app.exec_())