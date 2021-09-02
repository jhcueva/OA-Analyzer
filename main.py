import os
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QFileSystemModel
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QPoint
from gui.viewer import Viewer
from gui.roi import Roi
from gui.mouseTracker import MouseTracker


class App(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('./gui/main.ui', self)
        self.__configuremenuBar()
        self.__configureFileNavigator()
        self.__configureButtons()
        self.__configureMouseTrack()

        self.imageViewer = Viewer(self.image)
        self.dragPositionL = QPoint()
        self.dragPositionM = QPoint()

        # self.systemPath = os.path.dirname(os.path.abspath(__file__))
        # print(os.path.)
        # self.testPath = os.path.join(self.systemPath, gui)

    def __configureMouseTrack(self):
        tracker = MouseTracker(self.roi)
        tracker.positionChanged.connect(self.on_positionChanged)

    def __configureButtons(self):
        self.btnRoi.clicked.connect(self.roiSelector)
        self.btnProcess.clicked.connect(self.process)

    def __configureFileNavigator(self):
        self.lstFilesList.itemClicked.connect(self.displayImage)

    def __configuremenuBar(self):
        self.mniOpen.triggered.connect(self.openBrowser)
        self.btnOpenFile.clicked.connect(self.openBrowser)

    def openBrowser(self):
        try:
            self.lstFilesList.clear()
            desktop = os.path.expanduser("~/Desktop")
            self.dir = QFileDialog.getExistingDirectory(self, "Select directory", desktop)
            for dcmFiles in os.listdir(self.dir):
                _, ext = os.path.splitext(dcmFiles)
                if ext == ".dcm":
                    self.lstFilesList.addItem(dcmFiles)
        except Exception as e:
            print("No files to add", e)

    def displayImage(self, item):
        self.fileName = item.text()
        # self.imageViewer = Viewer(self.image)
        self.dicomImg = self.imageViewer.read_dicom(os.path.join(self.dir, self.fileName))
        rx = self.imageViewer.preprocess_xray(self.dicomImg)
        self.pixmap = self.imageViewer.arrayToPixmap(rx)
        self.imageViewer.setImage(self.pixmap)

    def process(self):
        self.roiViewer.saveRoi(self.fileName, self.roiViewer.lateralSquare(), self.roiViewer.medialSquare(), self.pixmap)

    def roiSelector(self):
        self.roiViewer = Roi(self.roi, self.image, self.dicomImg, self.pixmap, self.posX, self.posY)
        self.roiViewer.setRoi(self.pixmap)

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.roiViewer.lateralSquare().contains(self.posX, self.posY):
                self.roi.setCursor(Qt.ClosedHandCursor)
                self.dragPositionL = event.pos() - self.roiViewer.lateralSquare().topLeft()
            super().mousePressEvent(event)

            if self.roiViewer.medialSquare().contains(self.posX, self.posY):
                self.roi.setCursor(Qt.ClosedHandCursor)
                self.dragPositionM = event.pos() - self.roiViewer.medialSquare().topLeft()
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.dragPositionL.isNull():
            self.roiViewer.lateralSquare().moveTopLeft(event.pos() - self.dragPositionL)
            self.displayImage(self.lstFilesList.currentItem())
            self.roiViewer.setRoi(self.pixmap)
            self.update()
        super().mouseMoveEvent(event)

        if not self.dragPositionM.isNull():
            self.roiViewer.medialSquare().moveTopLeft(event.pos() - self.dragPositionM)
            self.displayImage(self.lstFilesList.currentItem())
            self.roiViewer.setRoi(self.pixmap)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragPositionL = QPoint()
        self.dragPositionM = QPoint()
        self.roi.setCursor(Qt.ArrowCursor)
        self.update()
        super().mouseReleaseEvent(event)

    @QtCore.pyqtSlot(QtCore.QPoint)
    def on_positionChanged(self, pos):
        try:
            self.posX = (pos.x() * self.dicomImg.shape[1]) // self.roi.width()
            self.posY = (pos.y() * self.dicomImg.shape[0]) // self.roi.height()
            self.roiViewer.mousePressEvent()
        except Exception as e:
            print('No value', e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec_())
