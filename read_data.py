from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)  # 图片的路径列表

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 单独拼接每一个
        print(img_item_path)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)  # 返回多少张图片


root_dir = "hymenoptera_data/train"

ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)
img,label = bees_dataset[0]
img.show()
