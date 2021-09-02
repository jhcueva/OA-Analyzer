import os
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QFileSystemModel
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from gui.viewer import Viewer
from gui.roi import Roi
from gui.mouseTracker import MouseTracker

from inference.inference import Knee


class App(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('./gui/main.ui', self)
        self.__configuremenuBar()
        self.__configureFileNavigator()
        self.__configureButtons()
        self.__configureMouseTrack()

        self.imageViewer = Viewer(self.image)
        self.rectL = QRect()
        self.rectM = QRect()
        self.dragPositionL = QPoint()
        self.dragPositionM = QPoint()

        self.system_path = os.path.dirname(os.path.abspath(__file__))
        self.gui = os.path.join(self.system_path, "gui")
        self.analyzed = os.path.join(self.gui, "analyzed")

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
        self.delete_ROI()
        self.fileName = item.text()
        # self.imageViewer = Viewer(self.image)
        self.dicomImg = self.imageViewer.read_dicom(os.path.join(self.dir, self.fileName))
        rx = self.imageViewer.preprocess_xray(self.dicomImg)
        self.pixmap = self.imageViewer.arrayToPixmap(rx)
        self.imageViewer.setImage(self.pixmap)

    def process(self):
        self.delete_ROI()
        self.imageViewer.setImage(self.pixmap)
        # self.roiViewer.setRoi(self.pixmap)
        self.roiViewer.saveRoi(self.fileName, self.lateralSquare(), self.medialSquare(), self.pixmap)
        # self.saveRoi(self.fileName, self.roiViewer.lateralSquare(), self.roiViewer.medialSquare())
        prediction, id = Knee(self.gui)
        print(prediction)
        print(id)

    def roiSelector(self):
        try:
            self.imageViewer.setImage(self.pixmap)
            self.roiViewer.setRoi(self.pixmap, self.lateralSquare(), self.medialSquare())
        except Exception as e:
            print(e)

        self.roiViewer = Roi(self.roi, self.image, self.dicomImg, self.posX, self.posY)
        # self.imageViewer.setImage(self.pixmap)
        self.roiViewer.setRoi(self.pixmap, self.lateralSquare(), self.medialSquare())

    def delete_ROI(self):
        print(self.analyzed)
        for file in os.listdir(self.analyzed):
            os.remove(os.path.join(self.analyzed, file))

        try:
            self.htmpRK.clear()
            self.htmpLK.clear()
            self.htmpSingle.clear()
            self.barPredictR.clear()
            self.barPredictL.clear()
            self.barPredictSingle.clear()

        except Exception as e:
            print("No files to delete", e)

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.lateralSquare().contains(self.posX, self.posY):
                self.roi.setCursor(Qt.ClosedHandCursor)
                self.dragPositionL = event.pos() - self.lateralSquare().topLeft()
            super().mousePressEvent(event)

            if self.medialSquare().contains(self.posX, self.posY):
                self.roi.setCursor(Qt.ClosedHandCursor)
                self.dragPositionM = event.pos() - self.medialSquare().topLeft()
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.dragPositionL.isNull():
            self.lateralSquare().moveTopLeft(event.pos() - self.dragPositionL)
            self.displayImage(self.lstFilesList.currentItem())
            self.roiViewer.setRoi(self.pixmap, self.lateralSquare(), self.medialSquare())
        super().mouseMoveEvent(event)

        if not self.dragPositionM.isNull():
            self.medialSquare().moveTopLeft(event.pos() - self.dragPositionM)
            self.displayImage(self.lstFilesList.currentItem())
            self.roiViewer.setRoi(self.pixmap, self.lateralSquare(), self.medialSquare())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragPositionL = QPoint()
        self.dragPositionM = QPoint()
        self.roi.setCursor(Qt.ArrowCursor)
        # self.displayImage(self.lstFilesList.currentItem())
        # self.roiViewer.setRoi(self.pixmap)
        self.update()
        super().mouseReleaseEvent(event)

    @QtCore.pyqtSlot(QtCore.QPoint)
    def on_positionChanged(self, pos):
        try:
            self.posX = (pos.x() * self.dicomImg.shape[1]) // self.roi.width()
            self.posY = (pos.y() * self.dicomImg.shape[0]) // self.roi.height()
            # self.roiViewer.mousePressEvent()
        except Exception as e:
            print('No value', e)

    def lateralSquare(self, height= 495, width=495):
        if self.rectL.isNull():
            self.rectL = QRect(QPoint(self.roiPoints()[0], self.roiPoints()[1]), QSize(height, width))
            self.update()
        return self.rectL

    def medialSquare(self, height=495, width=495):
        if self.rectM.isNull():
            self.rectM = QRect(QPoint(self.roiPoints()[2], self.roiPoints()[3]), QSize(height, width))
            self.update()
        return self.rectM

    def roiPoints(self):
        center = self.dicomImg.shape[1] // 2
        right_x1 = (self.dicomImg.shape[1] - center) // 3
        left_x1 = ((self.dicomImg.shape[1] - center) // 3) + center
        left_y1 = self.dicomImg.shape[0] // 4
        print(right_x1)
        print(left_y1)
        print(left_y1)
        print(center)
        return right_x1, left_y1, left_x1, left_y1, center


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec_())
