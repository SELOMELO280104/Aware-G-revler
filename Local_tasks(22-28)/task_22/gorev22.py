import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ResimPenceresi(QWidget):
    def __init__(self):
        super().__init__()
        self.pencere_ayarlari()

    def pencere_ayarlari(self):
        self.setWindowTitle('Aware Robotics - Resim Gösterici')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()
        self.label_resim = QLabel(self)
        self.label_resim.setAlignment(Qt.AlignCenter)

        # --- İŞTE GARANTİ YÖNTEM BURASI ---
        # 1. Şu an çalışan python dosyasının (gorev22.py) klasörünü bulur:
        dosya_klasoru = os.path.dirname(os.path.abspath(__file__))
        
        # 2. O klasör yolunu resim adıyla birleştirir:
        tam_resim_yolu = os.path.join(dosya_klasoru, 'deneme.jpg')

        if os.path.exists(tam_resim_yolu):
            pixmap = QPixmap(tam_resim_yolu)
            pixmap = pixmap.scaled(550, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_resim.setPixmap(pixmap)
        else:
            self.label_resim.setText(f"HATA!\nAranan Yol: {tam_resim_yolu}\nLütfen dosya adının 'deneme.jpg' olduğundan emin olun.")
        # ----------------------------------

        layout.addWidget(self.label_resim)
        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pencere = ResimPenceresi()
    pencere.show()
    sys.exit(app.exec_())