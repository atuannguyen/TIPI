
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np


class CorruptedData:
    def __init__(self,args,train_source=False):
        self.corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                                'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
                                'elastic_transform', 'pixelate', 'jpeg_compression']
        
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])

        if args.dataset == 'cifar10':
            DatasetClass = CIFAR10
            path = os.path.join(args.dataset_folder,f'CIFAR-10-C/{self.corruption_types[args.test_env]}.npy')
            label_path = os.path.join(args.dataset_folder,'CIFAR-10-C/labels.npy')
        elif args.dataset == 'cifar100':
            DatasetClass = CIFAR100
            path = os.path.join(args.dataset_folder,f'CIFAR-100-C/{self.corruption_types[args.test_env]}.npy')
            label_path = os.path.join(args.dataset_folder,'CIFAR-100-C/labels.npy')
        else:
            raise NotImplementedError()
        


        self.test_set = DatasetClass(root=args.dataset_folder, train=False, download=True, transform=self.transform)
        self.test_set.data = np.load(path)[-len(self.test_set.data):]
        self.test_set.targets = np.load(label_path)[-len(self.test_set.data):]


