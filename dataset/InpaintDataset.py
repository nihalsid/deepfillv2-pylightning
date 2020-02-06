from torch.utils.data import Dataset
from torchvision import transforms
import os
from util import constants
from PIL import Image
import torch
import random
from util.transforms import NormalizeRange, ToNumpyRGB256


class InpaintDataset(Dataset):

    def __init__(self, root_folder, split, image_size, bbox_shape, bbox_randomness, bbox_margin, bbox_max_num, is_overfit):
        self.root_folder = root_folder
        self.split = split
        self.image_size = image_size
        self.bbox_shape = bbox_shape
        self.bbox_margin = bbox_margin
        self.bbox_randomness = bbox_randomness
        self.bbox_max_num = bbox_max_num
        with open(os.path.join(constants.DATASET_ROOT, self.root_folder, constants.SPLITS_FOLDER, split + ".txt"),"r") as fptr:
            self.files = [x.strip() for x in fptr.readlines() if x.strip() != ""]
            if is_overfit and split == 'train':
                self.files = self.files * 10
        if split == 'train':
            self.transform_func = self.transform_train
        else:
            self.transform_func = self.transform_test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        image = Image.open(os.path.join(constants.DATASET_ROOT, self.root_folder, constants.IMAGES_FOLDER, file + ".jpg"))
        image_tensor = self.transform_func(image)
        mask = self.generate_mask()
        return {'name': file, 'image': image_tensor, 'mask': mask}

    def generate_mask(self):
        mask = torch.zeros((1, self.image_size, self.image_size)).float()
        num_bbox = random.randint(1, self.bbox_max_num)
        for i in range(num_bbox):
            bbox_height = random.randint(int(self.bbox_shape * (1 - self.bbox_randomness)), int(self.bbox_shape * (1 + self.bbox_randomness)))
            bbox_width = random.randint(int(self.bbox_shape * (1 - self.bbox_randomness)), int(self.bbox_shape * (1 + self.bbox_randomness)))
            y_min = random.randint(self.bbox_margin, self.image_size - bbox_height - self.bbox_margin - 1)
            x_min = random.randint(self.bbox_margin, self.image_size - bbox_width - self.bbox_margin - 1)
            mask[0, y_min:y_min + bbox_height, x_min:x_min + bbox_width] = 1
        return mask

    def transform_train(self, image):
        transform_list = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(self.image_size, scale=(1.0, 1.25), ratio=(1, 1), interpolation=2),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            NormalizeRange(minval=-1, maxval=1)
        ]
        composed_transform = transforms.Compose(transform_list)
        return composed_transform(image)

    def transform_test(self, image):
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            NormalizeRange(minval=-1, maxval=1)
        ]
        composed_transform = transforms.Compose(transform_list)
        return composed_transform(image)


class FixedInpaintDataset(Dataset):

    def __init__(self, root_folder, split, image_size, box_collection_idx):
        self.root_folder = root_folder
        self.image_size = image_size
        with open(os.path.join(constants.DATASET_ROOT, self.root_folder, constants.SPLITS_FOLDER, split + ".txt"), "r") as fptr:
            self.files = [x.strip() for x in fptr.readlines() if x.strip() != ""]
        self.box_collection_idx = box_collection_idx

    def __len__(self):
        return len(self.files) * len(self.get_test_masks())

    def __getitem__(self, index):
        file = self.files[index // len(self.get_test_masks())]
        image = Image.open(os.path.join(constants.DATASET_ROOT, self.root_folder, constants.IMAGES_FOLDER, file + ".jpg"))
        image_tensor = self.transform(image)
        mask_bounds = self.get_test_masks()
        mask = self.generate_mask(*mask_bounds[index % len(mask_bounds)])
        return {'name': f'{file}_{index % len(mask_bounds):02d}', 'image': image_tensor, 'mask': mask}

    def generate_mask(self, y_min, x_min, bbox_height, bbox_width):
        mask = torch.zeros((1, self.image_size, self.image_size)).float()
        mask[0, y_min:y_min + bbox_height, x_min:x_min + bbox_width] = 1
        return mask

    def get_test_masks(self):
        box_collections = [
            [(152, 41, 48, 43), (171, 114, 49, 45), (121, 91, 53, 37), (109, 62, 52, 58), (141, 150, 47, 39), (39, 73, 39, 53), (149, 167, 48, 38), (155, 46, 43, 38), (84, 64, 46, 51)],
        ]
        return box_collections[self.box_collection_idx]

    def transform(self, image):
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            NormalizeRange(minval=-1, maxval=1)
        ]
        composed_transform = transforms.Compose(transform_list)
        return composed_transform(image)


def test_inpaint_dataset():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np

    _root_folder = 'matterport'
    _split = 'train'
    _image_size = 256
    _bbox_shape = 48
    _bbox_randomness = 0.25
    _bbox_margin = 32
    _bbox_max_num = 2
    dataset = InpaintDataset(_root_folder, _split, _image_size, _bbox_shape, _bbox_randomness, _bbox_margin, _bbox_max_num, False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    torgb = ToNumpyRGB256(-1, 1)

    for i, sample in enumerate(dataloader, 0):
        for j in range(sample['image'].size()[0]):
            image = sample['image'].numpy()
            mask = sample['mask'].squeeze().numpy()[j]
            image_unnormalized = torgb(image[j])
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(image_unnormalized)
            plt.subplot(212)
            plt.imshow(mask)
            plt.show(block=True)


def test_fixed_inpaint_dataset():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np

    _root_folder = 'matterport'
    _split = 'train'
    _image_size = 256
    dataset = FixedInpaintDataset(_root_folder, "vis_0", _image_size, 0)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    torgb = ToNumpyRGB256(-1, 1)

    for i, sample in enumerate(dataloader, 0):
        for j in range(sample['image'].size()[0]):
            image = sample['image'].numpy()
            mask = sample['mask'].squeeze().numpy()[j]
            image_unnormalized = torgb(image[j])
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(image_unnormalized)
            plt.subplot(212)
            plt.imshow(mask)
            plt.show(block=True)


if __name__ == '__main__':
    test_fixed_inpaint_dataset()
