import os
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import uic, QtCore
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from gui.viewer import Viewer
from gui.roi import Roi
from gui.utils import Utils
from gui.results import ResutlsViewer
from gui.mouseTracker import MouseTracker

from inference.inference import Inference


class App(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('./gui/main.ui', self)
        self.__configuremenuBar()
        self.__configureFileNavigator()
        self.__configureButtons()
        self.__configureMouseTrack()

        self.system_path = os.path.dirname(os.path.abspath(__file__))
        self.gui = os.path.join(self.system_path, "gui")
        self.analyzed = os.path.join(self.gui, "analyzed")

        self.imageViewer = Viewer(self.image)
        self.inference = Inference(self.gui)
        self.rectL = QRect()
        self.rectM = QRect()
        self.dragPositionL = QPoint()
        self.dragPositionM = QPoint()

    def __configureMouseTrack(self):
        tracker = MouseTracker(self.roi)
        tracker.positionChanged.connect(self.on_positionChanged)

    def __configureButtons(self):
        self.btnRoi.clicked.connect(self.roiSelector)
        self.btnProcess.clicked.connect(self.process)
        self.btnDelete.clicked.connect(self.delete)
        self.spinSquareValue.valueChanged.connect(self.reSize)

    def __configureFileNavigator(self):
        self.utils = Utils(self.lstFilesList, self.lneSearch, self.displayImage)
        self.lstFilesList.itemClicked.connect(self.displayImage)
        self.lstFilesList.itemSelectionChanged.connect(self.newPoints)
        self.btnRight.clicked.connect(self.utils.right)
        self.btnLeft.clicked.connect(self.utils.left)

    def __configuremenuBar(self):
        self.mniOpen.triggered.connect(self.openBrowser)
        self.btnOpenFile.clicked.connect(self.openBrowser)

    def openBrowser(self):
        try:
            self.lstFilesList.clear()
            desktop = os.path.expanduser("~/Desktop")
            self.dir = QFileDialog.getExistingDirectory(self, "Select directory", desktop)
            self.utils.dir(self.dir)
            for dcmFiles in os.listdir(self.dir):
                _, ext = os.path.splitext(dcmFiles)
                if ext == ".dcm":
                    self.lstFilesList.addItem(dcmFiles)
        except Exception as e:
            print("No files to add", e)

    def displayImage(self, item):
        self.delete_ROI()
        self.fileName = item.text()
        self.dicomImg = self.imageViewer.read_dicom(os.path.join(self.dir, self.fileName))
        self.imageViewer.patientInfo(self.dicomImg[1], self.lblSex, self.lblID, self.lblDate)
        rx = self.imageViewer.preprocess_xray(self.dicomImg[0])
        self.pixmap = self.imageViewer.arrayToPixmap(rx)
        self.imageViewer.setImage(self.pixmap)

    def process(self):
        self.delete_ROI()
        self.displayImage(self.lstFilesList.currentItem())
        self.roiViewer.saveRoi(self.fileName, self.lateralSquare(), self.medialSquare(), self.pixmap)
        self.roiViewer.setRoi(self.pixmap, self.lateralSquare(), self.medialSquare())
        prediction, id = self.inference.Knee()
        self.inference.plot_prediction(prediction, id)
        self.resultsViewer = ResutlsViewer(self.analyzed)
        if len(id) > 1:
            self.resultsViewer.setBilateralViewer(id, self.barPredictR, self.barPredictL, self.htmpR, self.htmpL)
        else:
            self.resultsViewer.setSingleViewer(id, self.barPredictS, self.htmpSingle)
            
    def roiSelector(self):
        self.roiViewer = Roi(self.roi, self.image, self.dicomImg[0])
        self.roiViewer.setRoi(self.pixmap, self.lateralSquare(), self.medialSquare())

    def delete_ROI(self):
        for file in os.listdir(self.analyzed):
            os.remove(os.path.join(self.analyzed, file))

        try:
            self.htmpR.clear()
            self.htmpL.clear()
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
        self.update()
        super().mouseReleaseEvent(event)

    @QtCore.pyqtSlot(QtCore.QPoint)
    def on_positionChanged(self, pos):
        try:
            self.posX = (pos.x() * self.dicomImg[0].shape[1]) // self.roi.width()
            self.posY = (pos.y() * self.dicomImg[0].shape[0]) // self.roi.height()
        except Exception as e:
            print('No value', e)

    def lateralSquare(self, height= 490, width=490):
        if self.rectL.isNull():
            self.rectL = QRect(QPoint(self.roiPoints()[0], self.roiPoints()[1]), QSize(height, width))
            self.update()
        return self.rectL

    def medialSquare(self, height=490, width=490):
        if self.rectM.isNull():
            self.rectM = QRect(QPoint(self.roiPoints()[2], self.roiPoints()[3]), QSize(height, width))
            self.update()
        return self.rectM

    def roiPoints(self):
        center = self.dicomImg[0].shape[1] // 2
        right_x1 = center // 3
        left_x1 = (center // 3) + center
        left_y1 = self.dicomImg[0].shape[0] // 4
        print("Uno", right_x1)
        print("Dos", left_x1)
        print("Tres", left_y1)
        print("Cuatro", left_y1)
        print(center)
        return right_x1, left_y1, left_x1, left_y1, center

    def delete(self):
        """Delete one ROI rectangle"""
        self.rectL.setWidth(0)
        self.rectL.setHeight(1)
        self.displayImage(self.lstFilesList.currentItem())
        self.roiViewer.setRoi(self.pixmap, self.lateralSquare(), self.medialSquare())

    def newPoints(self):
        self.rectL.setWidth(490)
        self.rectL.setHeight(490)
        # self.rectM.setWidth(490)
        # self.rectM.setHeight(490)

    def reSize(self):
        size = self.spinSquareValue.value()
        try:
            self.rectL.setWidth(size)
            self.rectL.setHeight(size)
            self.rectM.setWidth(size)
            self.rectM.setHeight(size)
            self.displayImage(self.lstFilesList.currentItem())
            self.roiViewer.setRoi(self.pixmap, self.lateralSquare(), self.medialSquare())
        except Exception as e:
            print(e)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec_())
