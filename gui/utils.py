import os


class Utils:
    def __init__(self, qlist, qline, image):
        self.list = qlist
        self.search = qline
        self.image = image
        # self.dir = dir

        self.search.textChanged.connect(self.filter)
        # self.search.textEdited.connect(self.filter)

    def right(self):
        print("Boron derecho")
        if None != self.list.currentItem():
            if (self.list.count() - 1) == self.list.currentRow():
                self.list.setCurrentRow(-1)

            self.list.setCurrentRow(self.list.currentRow() + 1)
            self.image(self.list.currentItem())

    def left(self):
        print("Boron izquierdo")
        if None != self.list.currentItem():
            val = 0
            if self.list.currentRow() == 0:
                self.list.setCurrentRow(self.list.count() - 1)
                val = 1
            self.list.setCurrentRow(self.list.currentRow() - 1 + val)
            self.image(self.list.currentItem())

    def dir(self, dir):
        self.dir = dir

    def filter(self):
        print("Cambio")
        if self.search.text() == "":
            try:
                for dcmFiles in os.listdir(self.dir):
                    _, ext = os.path.splitext(dcmFiles)
                    if ext == ".dcm":
                        self.list.addItem(dcmFiles)
            except Exception as e:
                print("No files to add", e)

        itemsText = []
        for index in range(self.list.count()):
            itemsText.append(self.list.item(index).text())

        searchList = []
        for item in range(len(itemsText)):
            if self.search.text() in itemsText[item]:
                searchList.append(itemsText[item])

        self.list.clear()

        for item in range(len(searchList)):
            self.list.addItem(searchList[item])

