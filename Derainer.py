import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import testing

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # current working directory
        self.cwd = os.getcwd()
        # variables for the file path
        self.rainy_image_path = ""
        # set the title
        self.setWindowTitle("Derainer")
        # setting  the geometry of window
        self.setGeometry(400, 80, 1200, 900)

        # creating a label widget
        self.image_label = QLabel(self)
        # moving position
        self.image_label.move(100, 50)
        # set fixed size
        self.image_label.setFixedSize(1000, 650)
        # set image showing
        self.image_label.setScaledContents(True)
        self.image_label.setPixmap(QPixmap('./introduction.jpg'))

        # creating a button widget
        self.choose_file_button = QPushButton(self)
        # moving position
        self.choose_file_button.move(500, 730)
        # set fixed size
        self.choose_file_button.setFixedSize(240, 50)
        # show text
        self.choose_file_button.setText("Choose a Rainy Image")
        self.choose_file_button.setFont(QFont('Arial', 12))
        self.choose_file_button.clicked.connect(self.slot_choose_file)

        # creating a button widget
        self.derain_button = QPushButton(self)
        # moving position
        self.derain_button.move(500, 810)
        # set fixed size
        self.derain_button.setFixedSize(240, 50)
        # show text
        self.derain_button.setText("Derain")
        self.derain_button.setFont(QFont('Arial', 12))
        self.derain_button.clicked.connect(self.slot_derain)

    def slot_choose_file(self):
        filepath, filetype = QFileDialog.getOpenFileName(self, "Choosing image", self.cwd,
                                                         "JPG Files (*.jpg);;PNG Files (*.png);;JPEG Files (*.jpeg)")
        self.rainy_image_path = filepath
        if filepath != "":
            self.image_label.setPixmap(QPixmap(filepath))
        else:
            self.image_label.setPixmap(QPixmap('./introduction.jpg'))

    def slot_derain(self):
        derained_file_path = testing.derain(self.rainy_image_path)
        self.image_label.setPixmap(QPixmap(derained_file_path))


if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(App.exec())