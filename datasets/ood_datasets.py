import os
import torch
import json
import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from collections import namedtuple


def read_image(path):

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    return img


class LostAndFound(Dataset):

    LostAndFoundClass = namedtuple('LostAndFoundClass', ['name', 'id', 'train_id', 'category_name',
                                                         'category_id', 'color'])

    labels = [
        LostAndFoundClass('unlabeled', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('ego vehicle', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('rectification border', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('out of roi', 0, 255, 'Miscellaneous', 0, (0, 0, 0)),
        LostAndFoundClass('background', 0, 255, 'Counter hypotheses', 1, (0, 0, 0)),
        LostAndFoundClass('free', 1, 1, 'Counter hypotheses', 1, (128, 64, 128)),
        LostAndFoundClass('Crate (black)', 2, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (black - stacked)', 3, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (black - upright)', 4, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray)', 5, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray - stacked) ', 6, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Crate (gray - upright)', 7, 2, 'Standard objects', 2, (0, 0, 142)),
        LostAndFoundClass('Bumper', 8, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Cardboard box 1', 9, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (blue)', 10, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (blue - small)', 11, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (green)', 12, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Crate (green - small)', 13, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Exhaust Pipe', 14, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Headlight', 15, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Euro Pallet', 16, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon', 17, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon (large)', 18, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Pylon (white)', 19, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Rearview mirror', 20, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Tire', 21, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Ball', 22, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bicycle', 23, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Dog (black)', 24, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Dog (white)', 25, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Kid dummy', 26, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby car (gray)', 27, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby Car (red)', 28, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Bobby Car (yellow)', 29, 2, 'Emotional hazards', 4, (0, 0, 142)),
        LostAndFoundClass('Cardboard box 2', 30, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Marker Pole (lying)', 31, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Plastic bag (bloated)', 32, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Post (red - lying)', 33, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Post Stand', 34, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Styrofoam', 35, 2, 'Random hazards', 3, (0, 0, 142)),
        LostAndFoundClass('Timber (small)', 36, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Timber (squared)', 37, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Wheel Cap', 38, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Wood (thin)', 39, 0, 'Random non-hazards', 5, (0, 0, 0)),
        LostAndFoundClass('Kid (walking)', 40, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (on a bobby car)', 41, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (small bobby)', 42, 2, 'Humans', 6, (0, 0, 142)),
        LostAndFoundClass('Kid (crawling)', 43, 2, 'Humans', 6, (0, 0, 142)),
    ]

    train_id_in = 1
    train_id_out = 2
    # cs = Cityscapes()
    # mean = cs.mean
    # std = cs.std
    num_eval_classes = 19

    def __init__(self, split='test', root="/media/nazirnayal/DATA/datasets/LostAndFound", transform=None):
        assert os.path.exists(root), "lost&found valid not exists"
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.split = split      # ['test', 'train']
        self.images = []        # list of all raw input images
        self.targets = []       # list of all ground truth TrainIds images
        self.annotations = []   # list of all ground truth LabelIds images

        for root, _, filenames in os.walk(os.path.join(root, 'leftImg8bit', self.split)):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.png':
                    filename_base = '_'.join(filename.split('_')[:-1])
                    city = '_'.join(filename.split('_')[:-3])
                    self.images.append(os.path.join(root, filename_base + '_leftImg8bit.png'))
                    target_root = os.path.join(self.root, 'gtCoarse', self.split)
                    self.targets.append(os.path.join(target_root, city, filename_base + '_gtCoarse_labelTrainIds.png'))
                    self.annotations.append(os.path.join(target_root, city, filename_base + '_gtCoarse_labelIds.png'))

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image and trainIds as PIL image or torch.Tensor"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        target = torch.from_numpy(np.array(target)).long()
        return image, target


    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'LostAndFound Split: %s\n' % self.split
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()


class Fishyscapes(Dataset):
    FishyscapesClass = namedtuple('FishyscapesClass', ['name', 'id', 'train_id', 'hasinstances',
                                                       'ignoreineval', 'color'])
    # --------------------------------------------------------------------------------
    # A list of all Lost & Found labels
    # --------------------------------------------------------------------------------
    labels = [
        FishyscapesClass('in-distribution', 0, 0, False, False, (144, 238, 144)),
        FishyscapesClass('out-distribution', 1, 1, False, False, (255, 102, 102)),
        FishyscapesClass('unlabeled', 2, 255, False, True, (0, 0, 0)),
    ]

    train_id_in = 0
    train_id_out = 1
    # cs = Cityscapes()
    # mean = cs.mean
    # std = cs.std
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, split='Static', root="", transform=None):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.split = split  # ['Static', 'LostAndFound']
        self.images = []  # list of all raw input images
        self.targets = []  # list of all ground truth TrainIds images
        filenames = os.listdir(os.path.join(root, self.split, 'original'))
        root = os.path.join(root, self.split)
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.png':
                f_name = os.path.splitext(filename)[0]
                filename_base_img = os.path.join("original", f_name)
                filename_base_labels = os.path.join("labels", f_name)

                self.images.append(os.path.join(root, filename_base_img + '.png'))
                self.targets.append(os.path.join(root, filename_base_labels + '.png'))
        self.images = sorted(self.images)
        self.targets = sorted(self.targets)

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'LostAndFound Split: %s\n' % self.split
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()



class RoadAnomaly(Dataset):
    RoadAnomaly_class = namedtuple('RoadAnomalyClass', ['name', 'id', 'train_id', 'hasinstances',
                                                        'ignoreineval', 'color'])
    # --------------------------------------------------------------------------------
    # A list of all Lost & Found labels
    # --------------------------------------------------------------------------------
    labels = [
        RoadAnomaly_class('in-distribution', 0, 0, False, False, (144, 238, 144)),
        RoadAnomaly_class('out-distribution', 1, 1, False, False, (255, 102, 102)),
    ]

    train_id_in = 0
    train_id_out = 1
    # cs = Cityscapes()
    # mean = cs.mean
    # std = cs.std
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, root="/media/nazirnayal/DATA/datasets/RoadAnomaly/RoadAnomaly_jpg/", transform=None):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.images = []  # list of all raw input images
        self.targets = []  # list of all ground truth TrainIds images
        
        with open(os.path.join(root, 'frame_list.json'), 'r') as f:
            self.images = json.load(f)

        self.num_samples = len(self.images)
       
        for i in range(self.num_samples):
            label_path = os.path.join(root, 'frames', self.images[i][:-4] + '.labels',
                                      'labels_semantic.png')
            self.targets.append(label_path)
            self.images[i] = os.path.join(root, 'frames', self.images[i])

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        
        
        if self.transform is not None:
            image = self.transform(image)

        target = torch.from_numpy(np.array(target)).long()

        target[target == 2] = 1

        return image, target

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'Road anomaly Dataset: \n'
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()



class FishyscapesLAF(Dataset):
    """
    The Dataset folder is assumed to follow the following structure. In the given root folder, there must be two
    sub-folders:
    - fishyscapes_lostandfound: contains the mask labels.
    - laf_images: contains the images taken from the Lost & Found Dataset
    """

    train_id_in = 0
    train_id_out = 1

    def __init__(self, dataset_root='/media/nazirnayal/DATA/datasets/Fishyscapes/', transforms=None):
        super().__init__()

        self.dataset_root = dataset_root
        self.transforms = transforms

        self.images = []
        self.labels = []

        labels_path = os.path.join(dataset_root, 'fishyscapes_lostandfound')
        for lbl in os.listdir(labels_path):

            self.labels.extend([os.path.join(labels_path, lbl)])
            img_name = lbl[5:-10] + 'leftImg8bit.png'
            self.images.extend([os.path.join(dataset_root, 'laf_images', img_name)])

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = read_image(self.images[index])
        label = read_image(self.labels[index])
        
    
        if self.transforms is not None:
            image = self.transforms(image)

        label = label[:, :, 0]

        label = torch.from_numpy(label).long()

        return image, label.type(torch.LongTensor)

    def __len__(self):
        return self.num_samples


class FishyscapesStatic(Dataset):
    """
    The dataset folder is assumed to follow the following structure. In the given root folder there must be two
    sub-folders:
    - fs_val_v3: contains the mask labels in .png format
    - fs_static_images: contains the images also in .png format. These images need a processing step to be created from
    cityscapes. the fs_val_v3 file contains .npz files that contain numpy arrays. According to ID of each file, the
    corresponding image from cityscapes should be loaded and then the cityscape image and the image from the .npz file
    should be summed to form the modified image, which should be stored in fs_static_images folder. The images files are
    named using the label file name as follows: img_name = label_name[:-10] + 'rgb.png'
    """
    
    train_id_in = 0
    train_id_out = 1
    
    def __init__(self, dataset_root='/media/nazirnayal/DATA/datasets/Fishyscapes/', transforms=None):
        super().__init__()

        self.dataset_root = dataset_root
        self.transforms = transforms

        labels_root = os.path.join(dataset_root, 'fs_val_v3')
        images_root = os.path.join(dataset_root, 'fs_static_images')
        files = os.listdir(labels_root)

        self.images = []
        self.labels = []
        for f in files:
            if f[-3:] != 'png':
                continue

            self.labels.extend([os.path.join(labels_root, f)])
            image_path = os.path.join(images_root, f[:-10] + 'rgb.png')
            self.images.extend([image_path])

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = read_image(self.images[index])
        label = read_image(self.labels[index])

        

        if self.transforms is not None:
            image = self.transforms(image)
        
        label = label[:, :, 0]
        label = torch.from_numpy(label).long()
        
        return image, label.type(torch.LongTensor)

    def __len__(self):
        return self.num_samples