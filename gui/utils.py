class Utils:
    def __init__(self, qlist, qline,image):
        self.list = qlist
        self.image = image
        self.search = qline

    def right(self):
        if None != self.list.currentItem():
            if (self.list.count() - 1) == self.list.currentRow():
                self.list.setCurrentRow(-1)

            self.list.setCurrentRow(self.list.currentRow() + 1)
            self.image(self.list.currentItem())

    def left(self):
        if None != self.list.currentItem():
            val = 0
            if self.list.currentRow() == 0:
                self.list.setCurrentRow(self.list.count() - 1)
                val = 1
            self.list.setCurrentRow(self.list.currentRow() - 1 + val)
            self.image(self.list.currentItem())

    # def search(self, text):
    #     if None != self.list.currentItem():
    #         print(text)

