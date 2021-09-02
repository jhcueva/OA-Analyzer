import os
import sys

from PyQt5.QtWidgets import QFileDialog

from main import App


class MenuBar(App):
    def __init__(self):
        super().__init__()
        self.ui.mni_Open.triggered.connect(self.open_browser)
        self.ui.mni_Exit.triggered.connect(self.exit)

    def open_browser(self):
        """Load DICOM files from the selected directory and add them to the list viewer"""
        try:
            self.ui.lst_FilesList.clear()
            desktop = os.path.expanduser("~/Desktop")
            self.dir = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)
            for dcm_image in os.listdir(self.dir):
                _, ext = os.path.splitext(dcm_image)
                if ext == '.dcm':
                    self.ui.lst_FilesList.addItem(dcm_image)
        except:
            print("Error opening file")

    def exit(self):
        sys.exit(0)

