import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO

# --- SİZİN OLUŞTURDUĞUNUZ ONNX DOSYASININ YOLU ---
# (Windows yolları için başına 'r' koyuyoruz)
ONNX_MODEL_PATH = r"C:\Users\hamdi\OneDrive\Masaüstü\Aware Robotics\gorev28test\yolov8n.onnx"

# --- 1. KAMERA VE YAPAY ZEKA İŞLEMLERİ (ARKA PLAN) ---
class KameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        print(f"Model yükleniyor: {ONNX_MODEL_PATH}")
        try:
            # --- DEĞİŞİKLİK BURADA ---
            # .pt yerine .onnx yüklüyoruz. 
            # task='detect' parametresi, ONNX modelinin ne iş yaptığını belirtmek için gereklidir.
            self.model = YOLO(ONNX_MODEL_PATH, task='detect')
            print("ONNX Modeli Başarıyla Yüklendi!")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            self.model = None

    def run(self):
        # Model yüklenemediyse çalışmasın
        if self.model is None:
            return

        self.cap = cv2.VideoCapture(0)
        
        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                # --- YAPAY ZEKA KISMI (TRACKING) ---
                # ONNX modelleri de .track() komutunu destekler!
                # persist=True: ID numarasını (Örn: ID 1) hafızada tutar.
                try:
                    results = self.model.track(source=frame, persist=True, verbose=False, tracker="bytetrack.yaml")
                    
                    # Sonuçları çizilmiş kareyi al
                    annotated_frame = results[0].plot()

                    # --- GÖRÜNTÜYÜ PYQT FORMATINA ÇEVİRME ---
                    rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    # Görüntüyü ölçekle
                    p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                    
                    # Sinyali gönder
                    self.change_pixmap_signal.emit(p)
                except Exception as e:
                    print(f"Takip hatası: {e}")
                    break
            else:
                break
        
        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


# --- 2. ARAYÜZ TASARIMI (ÖN PLAN) ---
class TakipArayuzu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 + YOLOv8 (ONNX) Takip Sistemi")
        self.setGeometry(100, 100, 700, 600)
        self.setStyleSheet("background-color: #f0f0f0;")
        self.thread = None

        self.arayuzu_kur()

    def arayuzu_kur(self):
        layout = QVBoxLayout()

        lbl_baslik = QLabel("ONNX Destekli Takip Sistemi")
        lbl_baslik.setAlignment(Qt.AlignCenter)
        lbl_baslik.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #333;")
        
        self.image_label = QLabel("Kamerayı Başlatmak İçin Butona Basın")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2c3e50; color: white; font-size: 16px; border: 2px solid #34495e; border-radius: 10px;")
        self.image_label.setFixedSize(640, 480)
        
        btn_layout = QHBoxLayout()
        
        self.btn_baslat = QPushButton("▶ Başlat (ONNX)")
        self.btn_baslat.clicked.connect(self.kamerayi_baslat)
        self.btn_baslat.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 12px; border-radius: 5px;")
        
        self.btn_durdur = QPushButton("⏹ Durdur")
        self.btn_durdur.clicked.connect(self.kamerayi_durdur)
        self.btn_durdur.setEnabled(False)
        self.btn_durdur.setStyleSheet("""
            QPushButton { background-color: #c0392b; color: white; font-weight: bold; padding: 12px; border-radius: 5px; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)

        btn_layout.addWidget(self.btn_baslat)
        btn_layout.addWidget(self.btn_durdur)

        layout.addWidget(lbl_baslik)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def kamerayi_baslat(self):
        if not self.thread:
            self.image_label.setText("ONNX Modeli Yükleniyor...\nLütfen Bekleyin.")
            self.thread = KameraThread()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()
            self.btn_baslat.setEnabled(False)
            self.btn_durdur.setEnabled(True)

    def kamerayi_durdur(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
            self.btn_baslat.setEnabled(True)
            self.btn_durdur.setEnabled(False)
            self.image_label.clear()
            self.image_label.setText("Kamera Durduruldu")

    def closeEvent(self, event):
        self.kamerayi_durdur()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pencere = TakipArayuzu()
    pencere.show()
    sys.exit(app.exec_())