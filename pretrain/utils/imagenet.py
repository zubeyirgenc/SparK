# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import Any, Callable, Optional, Tuple

import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            imagenet_folder: str,
            train: bool,
            transform: Callable,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        imagenet_folder = os.path.join(imagenet_folder, 'train' if train else 'val')
        super(ImageNetDataset, self).__init__(
            imagenet_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform, target_transform=None, is_valid_file=is_valid_file
        )
        
        self.samples = tuple(self.samples)
        self.targets = tuple([s[1] for s in self.samples])
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        path, target = self.samples[index]
        return self.transform(self.loader(path)), target


def build_dataset_to_pretrain(dataset_path, input_size) -> Dataset:
    """
    You may need to modify this function to fit your own dataset.
    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    
    dataset_path = os.path.abspath(dataset_path)
    for postfix in ('train', 'val'):
        if dataset_path.endswith(postfix):
            dataset_path = dataset_path[:-len(postfix)]
    
    dataset_train = ImageNetDataset(imagenet_folder=dataset_path, transform=trans_train, train=True)
    print_transform(trans_train, '[pre-train]')
    return dataset_train


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')


def build_metric_dataset_to_pretrain(args) -> Dataset:
    """
    You may need to modify this function to fit your own dataset.
    :param args: the arguments
    :return: the dataset used for pretraining
    """
    dataset_names = args.dataset_names

    transform_cfg = {
        'normalization': args.normalization,
        'crop_and_pad': False,
        'img_size': args.input_size,
        'mean_std': {
            'mean': args.mean,
            'std': args.std,
        },
        'square': args.square,
        'border_mode': args.border_mode,
    }
    print(f'Using transform: {transform_cfg}')
    # sys.path.append('/home/zubeyir/Desktop/work/r-d-metric-learning')
    from pl_multidataset import MetricDataModule

    dm = MetricDataModule(
        data_config=args.config_yaml,
        transform_cfg=transform_cfg,
        dataset_names=dataset_names,
        is_valid_file=None,
        batch_size=args.bs,
        num_workers=args.num_workers,
        collate_fn=None,
        use_sampler=False,
        distributed_sampler=False,
    )
    dm.setup(stage='fit')
    train_dls = dm.train_datasets[0]

    return train_dls
