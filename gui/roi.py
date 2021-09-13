import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QAction, QApplication, QFileDialog, QMainWindow, QGraphicsPixmapItem, QWidget, QLineEdit, QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QCursor
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QRectF, QRect, QSize, QPoint, pyqtSignal


class Roi():
    def __init__(self, qlabelRoi, qlabelImg, dcm):
        self._roi = qlabelRoi
        self._img = qlabelImg
        self._dcm = dcm

        self.rectL = QRect()
        self.rectM = QRect()

        self.system_path = os.path.dirname(os.path.abspath(__file__))
        self.analyzed = os.path.join(self.system_path, "analyzed")

    def setRoi(self, Imagepixmap, lateralS, medialS):
        self._imgPixmap = Imagepixmap
        pixmap = QPixmap(self._roi.size())
        pixmap.fill(Qt.transparent)
        self.painter = QPainter(Imagepixmap)
        painterRectangle = QPen(Qt.green)
        painterRectangle.setWidth(10)
        self.painter.setPen(painterRectangle)
        self.painter.drawRect(lateralS)
        self.painter.drawRect(medialS)
        self._roi.setPixmap(pixmap)
        self._img.setPixmap(Imagepixmap)
        self.painter.end()

    def roiPoints(self):
        center = self._dcm.shape[1] // 2
        right_x1 = (self._dcm.shape[1] - center) // 3
        left_x1 = ((self._dcm.shape[1] - center) // 3) + center
        left_y1 = self._dcm.shape[0] // 4
        print(right_x1)
        print(left_y1)
        print(left_y1)
        print(center)
        return right_x1, left_y1, left_x1, left_y1, center

    def saveRoi(self, dir, rectL, rectM, Imagepixmap):
        name, ext = os.path.splitext(dir)
        if rectL.width() > 0:
            currentQrectL = rectL
            currentQrectM = rectM
            # self.setRoi(self._imgPixmap)
            cropL = self._imgPixmap.copy(currentQrectL)
            cropM = self._imgPixmap.copy(currentQrectM)
            if currentQrectL.x() < self.roiPoints()[-1]:
                nameL = name + "_R" + ".png"
                nameR = name + "_L" + ".png"
                cropL.save(os.path.join(self.analyzed, nameL), quality=0)
                cropM.save(os.path.join(self.analyzed, nameR), quality=0)

            else:
                nameL = name + "_L" + ".png"
                nameR = name + "_R" + ".png"
                cropL.save(os.path.join(self.analyzed, nameL), quality=0)
                cropM.save(os.path.join(self.analyzed, nameR), quality=0)
        if rectL.width() < 495:
            currentQrect = rectM
            crop = self._imgPixmap.copy(currentQrect)
            crop.save(os.path.join(self.analyzed, name+".png"), quality=0)
