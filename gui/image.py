import os
import cv2
import numpy as np

class Image:
    """class image for reading a file with opencv.
    :param src: filepath
    :param maxSize: tuple with max size allowed on the current image.

    Example usage::
        src = "my/image/path"
        image = Image(src)
        imageArray = image.load()

    """
    def __init__(self, src="", maxSize=(1024, 768, 3)):
        self.src = src
        self.maxSize = maxSize
        self.image = None
        self.imageFixed = None

    def getName(self):
        """It returns the current filename."""
        if self.hasImage():
            name = self.src.split(os.sep)[-1]
            return name

    def hasFixed(self):
        """It checks if an fixed image is available."""
        return self.imageFixed is not None

    def hasImage(self):
        """It checks if an image is available."""
        return self.image is not None

    def setSource(self, src=""):
        self.src = src

    def load(self):
        """It loads a image from a src."""
        if self.image is None:
            self.image = cv2.imread(self.src)

        if self.image is not None:
            if self.image.shape > self.maxSize:
                if not self.hasFixed():
                    self.imageFixed = cv2.resize(self.image, self.maxSize)
                return self.imageFixed
        return self.image

    def getImage(self):
        """It returns the current image."""
        if self.hasImage():
            return self.image

    def getFixed(self):
        """It returns the current fixed image."""
        if self.hasFixed():
            return self.imageFixed

    def getFixedShape(self):
        """it returns the current fixed image shape."""
        if self.hasFixed():
            return self.imageFixed.shape
        return self.getOriginalShape()

    def getOriginalShape(self):
        """It returns the orignal image shape."""
        if self.hasImage():
            return self.image.shape

    def clear(self):
        """It clears the last image loaded."""
        self.image = None
        self.imageFixed = None