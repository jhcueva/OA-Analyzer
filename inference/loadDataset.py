import os
from PIL import Image
from torch.utils.data import Dataset


class LoadDataset(Dataset):
    def __init__(self, directory, split = 'train', transform = None):
        path = os.path.join(directory, '{}'.format(split))
        files = os.listdir(path)

        self.file = [os.path.join(path, name) for name in files]
        self.file_name = [name for name in files]
        # self.target = [int(name[0]) for name in files]
        self.transform = transform

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, item):
        image = Image.open(self.file[item])
        file = self.file[item]
        name = self.file_name[item]
        lateral, medial = self.Lateral_Medial(image)
        if self.transform:
            lateral = self.transform(lateral)
            medial = self.transform(medial)
            # image = self.transform(image)

        return lateral, medial, file, name
        # return image, self.target[item], file, name

    def Lateral_Medial(self, img):
        '''
        Generates the lateral and medial image from knee joint image
        :param img: Knee joint image
        :return: Lateral and medial image

        '''
        width, height = img.size
        pad = (width // 2) + 16
        lateral = img.crop((0, (height // 4), pad, (height // 4) + pad))
        medial = img.crop((width - pad, (height // 4), width, (height // 4) + pad))
        medial = medial.transpose(Image.FLIP_LEFT_RIGHT)
        # name, ext = os.path.splitext(self.name)
        # lateral.save(os.path.join(path, name+'_L'+ext))
        # medial.save(os.path.join(path, name+'_M'+ext))

        return lateral, medial
