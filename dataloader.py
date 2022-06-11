import glob
import os
import random
from typing import Callable, Optional

import numpy as np
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

__all__ = ['prepare_dataset']

EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TinyImagenet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """

    def __init__(self, root, split='train', transform: Optional[Callable] = None, target_transform=None,
                 in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing
        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            # return index, img
            return img
        else:
            # file_name = file_path.split('/')[-1]
            # return index, img, self.labels[os.path.basename(file_path)]
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path).convert('RGB')
        return self.transform(img)  # if self.transform is not None else img


class TinyImagenetPair(TinyImagenet):
    def __init__(self, root, transform, weak_aug=None):
        super().__init__(root, transform)
        self.weak_aug = weak_aug

    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = self.load_image(path)
        pos_1 = self.transform(img)
        if self.weak_aug is not None:
            pos_2 = self.weak_aug(img)
        else:
            pos_2 = self.transform(img)
        return pos_1, pos_2


class STL10Pair(datasets.STL10):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)

        self.weak_aug = weak_aug

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)

            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)
        return pos_1, pos_2


class CIFAR10Pair(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.weak_aug = weak_aug

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)

            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)

        return pos_1, pos_2


class CIFAR100Pair(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, weak_aug=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.weak_aug = weak_aug

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            pos_1 = self.transform(img)

            if self.weak_aug is not None:
                pos_2 = self.weak_aug(img)
            else:
                pos_2 = self.transform(img)
        return pos_1, pos_2


class TwoCrop:
    def __init__(self, weak, strong):
        self.weak = weak
        self.strong = strong

    def __call__(self, img):
        im_1 = self.strong(img)
        im_2 = self.weak(img)

        return im_1, im_2


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_strong_augment(dataset):
    size = 32
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform


def get_weak_augment(dataset):
    size = 32
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform


def get_linear_augment(dataset):
    size = 32
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 64
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        size = 64
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform


def get_test_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
    elif dataset == 'tinyimagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    normalize = transforms.Normalize(mean=mean, std=std)
    none_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return none_transform


def prepare_dataset(args):
    """Define train/test"""
    strong_transform = get_strong_augment(args.dataset)
    weak_transform = eval(f'get_{args.aug}')(args.dataset)
    test_transform = get_test_augment(args.dataset)
    linear_transform = get_linear_augment(args.dataset)

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_path, download=True,
                                         transform=TwoCrop(weak_transform, strong_transform))
        memory_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=test_transform)
        linear_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=linear_transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
        args.num_classes = 10
    elif args.dataset == 'stl10':
        train_dataset = datasets.STL10(root=args.data_path, download=True, split='train+unlabeled',
                                       transform=TwoCrop(weak_transform, strong_transform))
        memory_dataset = datasets.STL10(root=args.data_path, download=True, split='train', transform=test_transform)
        linear_dataset = datasets.STL10(root=args.data_path, download=True, split='train', transform=linear_transform)
        test_dataset = datasets.STL10(root=args.data_path, download=True, split='test', transform=test_transform)
        args.num_classes = 10

    elif args.dataset == 'tinyimagenet':
        train_dataset = TinyImagenet(root=args.data_path, split='train',
                                     transform=TwoCrop(weak_transform, strong_transform))
        memory_dataset = TinyImagenet(root=args.data_path, split='train', transform=test_transform)
        linear_dataset = TinyImagenet(root=args.data_path, split='train', transform=linear_transform)
        test_dataset = TinyImagenet(root=args.data_path, split='val', transform=test_transform)
        args.num_classes = 200

    else:
        train_dataset = datasets.CIFAR100(root=args.data_path, download=True,
                                          transform=TwoCrop(weak_transform, strong_transform))
        memory_dataset = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=test_transform)
        linear_dataset = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=linear_transform)
        test_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=test_transform)
        args.num_classes = 100

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True,
                              drop_last=True)
    memory_loader = DataLoader(memory_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6,
                               pin_memory=True)
    linear_loader = DataLoader(linear_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)

    return train_loader, memory_loader, linear_loader, test_loader
