from PyQt5.QtWidgets import QApplication, QWidget

app = QApplication([])
window = QWidget()
window.setWindowTitle('Qt Application Test')
window.show()
app.exec_()
