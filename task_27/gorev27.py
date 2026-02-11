import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QTextEdit, QMessageBox, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QColor
from PyQt5.QtCore import Qt
import pytesseract

# --- 1. ÖRNEK RESİM OLUŞTURUCU (Qt ile) ---
def ornek_resim_olustur():
    image = QImage(600, 250, QImage.Format_RGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    
    font = QFont("Arial", 16)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor("black"))

    metin = "TESSERACT TESTI!\nBu yazi yapay zeka tarafindan okunmaktadir.\nRakamlar: 0123456789\nKonum: C:\\Program Files\\Tesseract-OCR"
    painter.drawText(20, 40, 550, 200, Qt.AlignLeft, metin)
    
    painter.end()
    image.save("ocr_deneme.png")
    print("ocr_deneme.png oluşturuldu.")

# --- 2. ANA ARAYÜZ ---
class OCRUygulamasi(QWidget):
    def __init__(self):
        super().__init__()
        
        # --- İŞTE BURASI: Sizin verdiğiniz yolu koda tanımlıyoruz ---
        # Windows yollarındaki ters eğik çizgiler (\) sorun çıkarmasın diye başına 'r' koyuyoruz.
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        self.pencere_ayarlari()
        self.arayuz_olustur()

    def pencere_ayarlari(self):
        self.setWindowTitle('PyQt5 Optik Karakter Okuma (OCR)')
        self.setGeometry(100, 100, 950, 550)
        self.setStyleSheet("background-color: #ecf0f1; font-family: Segoe UI;")

    def arayuz_olustur(self):
        ana_duzen = QHBoxLayout()

        # SOL TARA (Resim)
        sol_layout = QVBoxLayout()
        self.lbl_resim = QLabel("Resim Bekleniyor...")
        self.lbl_resim.setAlignment(Qt.AlignCenter)
        self.lbl_resim.setFrameShape(QFrame.StyledPanel)
        self.lbl_resim.setFixedSize(450, 350)
        self.lbl_resim.setStyleSheet("background-color: white; border: 2px dashed #95a5a6; border-radius: 10px;")
        
        # Butonlar
        self.btn_ornek = QPushButton("✨ Örnek Resim Oluştur")
        self.btn_ornek.clicked.connect(self.ornek_uret)
        self.btn_ornek.setStyleSheet("background-color: #e67e22; color: white; padding: 10px; border-radius: 5px;")

        self.btn_oku = QPushButton("📂 Resim Seç ve Oku")
        self.btn_oku.clicked.connect(self.resim_sec_ve_oku)
        self.btn_oku.setStyleSheet("background-color: #2980b9; color: white; padding: 12px; font-weight: bold; border-radius: 5px;")

        sol_layout.addWidget(QLabel("Kaynak:"))
        sol_layout.addWidget(self.lbl_resim)
        sol_layout.addWidget(self.btn_ornek)
        sol_layout.addWidget(self.btn_oku)

        # SAĞ TARAF (Çıktı)
        sag_layout = QVBoxLayout()
        self.txt_cikti = QTextEdit()
        self.txt_cikti.setPlaceholderText("Sonuçlar burada görünecek...")
        self.txt_cikti.setStyleSheet("font-size: 14px; padding: 10px;")
        
        sag_layout.addWidget(QLabel("Sonuç:"))
        sag_layout.addWidget(self.txt_cikti)

        ana_duzen.addLayout(sol_layout, 40)
        ana_duzen.addLayout(sag_layout, 60)
        self.setLayout(ana_duzen)

    def ornek_uret(self):
        ornek_resim_olustur()
        self.resim_goster("ocr_deneme.png")
        self.txt_cikti.setText("Örnek resim oluşturuldu.\nŞimdi 'Resim Seç ve Oku' diyerek bu dosyayı (ocr_deneme.png) seçin.")

    def resim_goster(self, yol):
        pixmap = QPixmap(yol)
        self.lbl_resim.setPixmap(pixmap.scaled(450, 350, Qt.KeepAspectRatio))

    def resim_sec_ve_oku(self):
        dosya_yolu, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resimler (*.png *.jpg)")
        if not dosya_yolu: return

        self.resim_goster(dosya_yolu)
        self.txt_cikti.setText("Okunuyor...")
        QApplication.processEvents()

        try:
            # OCR İşlemi
            text = pytesseract.image_to_string(dosya_yolu)
            self.txt_cikti.setText(text if text.strip() else "Yazı bulunamadı.")
        except Exception as e:
            self.txt_cikti.setText(f"HATA:\n{e}\n\nLütfen Tesseract yolunu kontrol edin.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pencere = OCRUygulamasi()
    pencere.show()
    sys.exit(app.exec_())