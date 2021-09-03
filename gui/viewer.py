import numpy as np
import pydicom as dicom
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QCursor
from PyQt5.QtWidgets import QFrame, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import  Qt, pyqtSignal, QObject, QEvent, QRectF
from PyQt5 import QtCore, QtGui



class Viewer(QObject):
    def __init__(self, qlabel):
        super().__init__()
        self._image = qlabel
        self._zoom = 0

    def setImage(self, pixmap):
        self._image.setPixmap(pixmap)

    def arrayToPixmap(self, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        img = QtGui.QPixmap.fromImage(img)
        img = QtGui.QPixmap(img)
        return img

    def patientInfo(self, dcmInfo, sexlbl, idlbl, datelbl):
        name = str(dcmInfo.PatientName)
        name = name.replace("^", " ")
        date = str(dcmInfo.StudyDate)
        date = date[6:] + "/" + date[4:6] + "/" + date[:4]
        sexlbl.setText('Sex: ' + dcmInfo.PatientSex)
        idlbl.setText("Name: " + str(name))
        datelbl.setText("Date: " + date)
        

    def read_dicom(self, fileDir):
        dcm = dicom.read_file(fileDir)
        img = np.frombuffer(dcm.PixelData, dtype=np.uint16).copy()

        if dcm.PhotometricInterpretation == 'MONOCHROME1':
            img = img.max() - img
        img = img.reshape((dcm.Rows, dcm.Columns))
        return img, dcm

    def preprocess_xray(self, img, cut_min=5., cut_max=99.):
        """Preprocess the X-ray image using histogram clipping and global contrast normalization.

        Parameters
        ----------
        cut_min: int
            Lowest percentile which is used to cut the image histogram.
        cut_max: int
            Highest percentile.
        """

        img = img.astype(np.float64)

        lim1, lim2 = np.percentile(img, [cut_min, cut_max])

        img[img < lim1] = lim1
        img[img > lim2] = lim2

        img -= lim1

        img /= img.max()
        img *= 255

        return img.astype(np.uint8, casting='unsafe')
