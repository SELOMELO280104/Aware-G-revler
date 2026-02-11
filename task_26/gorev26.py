import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO

# --- 1. KAMERA VE YAPAY ZEKA İŞLEMLERİ (ARKA PLAN) ---
class KameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        # YOLOv8 Nano modelini yükle (En hızlı modeldir)
        # İlk çalıştırıldığında model internetten otomatik iner.
        print("YOLO Modeli yükleniyor...")
        self.model = YOLO('yolov8n.pt')
        print("Model hazır!")

    def run(self):
        # Kamerayı başlat (0 varsayılan web kamerasını temsil eder)
        self.cap = cv2.VideoCapture(0)
        
        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                # --- YAPAY ZEKA KISMI (TRACKING) ---
                # persist=True: Nesne ID'lerini (Kimliklerini) hafızada tutar.
                # Bu sayede nesne hareket etse bile ID'si (Örn: #1) değişmez.
                results = self.model.track(source=frame, persist=True, verbose=False)
                
                # Sonuçları çizilmiş kareyi al (Kutular ve ID numaraları)
                annotated_frame = results[0].plot()

                # --- GÖRÜNTÜYÜ PYQT FORMATINA ÇEVİRME ---
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Görüntüyü arayüze sığacak şekilde ölçekle
                p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                
                # Sinyali gönder (Arayüzü güncelle)
                self.change_pixmap_signal.emit(p)
            else:
                break
        
        # Döngü biterse kamerayı serbest bırak
        self.cap.release()

    def stop(self):
        """Thread'i güvenli şekilde durdurur"""
        self._run_flag = False
        self.wait()


# --- 2. ARAYÜZ TASARIMI (ÖN PLAN) ---
class TakipArayuzu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 + YOLOv8 Nesne Takip ve ID Sistemi")
        self.setGeometry(100, 100, 700, 600)
        self.setStyleSheet("background-color: #f0f0f0;")
        self.thread = None

        self.arayuzu_kur()

    def arayuzu_kur(self):
        # Ana Düzen
        layout = QVBoxLayout()

        # Başlık
        lbl_baslik = QLabel("Yapay Zeka Destekli Takip Sistemi")
        lbl_baslik.setAlignment(Qt.AlignCenter)
        lbl_baslik.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #333;")
        
        # 1. Kamera Görüntü Alanı
        self.image_label = QLabel("Kamerayı Başlatmak İçin Butona Basın")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2c3e50; color: white; font-size: 16px; border: 2px solid #34495e; border-radius: 10px;")
        self.image_label.setFixedSize(640, 480)
        
        # 2. Butonlar
        btn_layout = QHBoxLayout()
        
        self.btn_baslat = QPushButton("▶ Kamerayı Başlat")
        self.btn_baslat.clicked.connect(self.kamerayi_baslat)
        self.btn_baslat.setStyleSheet("""
            QPushButton { background-color: #27ae60; color: white; font-weight: bold; padding: 12px; border-radius: 5px; font-size: 14px;}
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        
        self.btn_durdur = QPushButton("⏹ Kamerayı Durdur")
        self.btn_durdur.clicked.connect(self.kamerayi_durdur)
        self.btn_durdur.setEnabled(False) # Başlangıçta pasif
        self.btn_durdur.setStyleSheet("""
            QPushButton { background-color: #c0392b; color: white; font-weight: bold; padding: 12px; border-radius: 5px; font-size: 14px;}
            QPushButton:hover { background-color: #e74c3c; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)

        btn_layout.addWidget(self.btn_baslat)
        btn_layout.addWidget(self.btn_durdur)

        # Düzeni Birleştir
        layout.addWidget(lbl_baslik)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def update_image(self, qt_img):
        """Thread'den gelen yeni resmi ekrana basar"""
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def kamerayi_baslat(self):
        if not self.thread:
            self.image_label.setText("Yapay Zeka Başlatılıyor...\nLütfen Bekleyin.")
            self.thread = KameraThread()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()
            
            # Buton durumlarını güncelle
            self.btn_baslat.setEnabled(False)
            self.btn_durdur.setEnabled(True)

    def kamerayi_durdur(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
            
            # Buton durumlarını güncelle
            self.btn_baslat.setEnabled(True)
            self.btn_durdur.setEnabled(False)
            self.image_label.clear()
            self.image_label.setText("Kamera Durduruldu")

    def closeEvent(self, event):
        """Pencere kapatılırsa kamerayı da kapat"""
        self.kamerayi_durdur()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pencere = TakipArayuzu()
    pencere.show()
    sys.exit(app.exec_())