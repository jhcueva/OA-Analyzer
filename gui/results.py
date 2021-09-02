import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class ResutlsViewer:
    def __init__(self, directory):
        self.directory = directory

    def setBilateralViewer(self, id, predictionR, predictionL, heatmapR, heatmapL):
        predictionStatsR = QPixmap(os.path.join(self.directory, "stats_" + id[1]))
        predictionStatsL = QPixmap(os.path.join(self.directory, "stats_" + id[0]))
        height, width = predictionR.size().height(), predictionR.size().width()
        pixmapStatsR = predictionStatsR.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        pixmapStatsL = predictionStatsL.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        predictionR.setPixmap(pixmapStatsR)
        predictionL.setPixmap(pixmapStatsL)

        htmapR = QPixmap(os.path.join(self.directory, "heatmap_" + id[1]))
        htmapL = QPixmap(os.path.join(self.directory, "heatmap_" + id[0]))
        height, width = heatmapR.size().height(), heatmapR.size().width()
        pixmapHtmapR = htmapR.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        pixmapHtmapL = htmapL.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        heatmapR.setPixmap(pixmapHtmapR)
        heatmapL.setPixmap(pixmapHtmapL)

    def setSingleViewer(self, id, prediction, heatmap):
        predictionStats = QPixmap(os.path.join(self.directory, "stats_" + id[0]))
        height, width = prediction.size().height(), prediction.size().width()
        pixampStats = predictionStats.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        prediction.setPixmap(pixampStats)

        htmap = QPixmap(os.path.join(self.directory, "heatmap_" + id[0]))
        height, width = heatmap.size().height(), heatmap.size().width()
        pixmapHtmap = htmap.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        heatmap.setPixmap(pixmapHtmap)
