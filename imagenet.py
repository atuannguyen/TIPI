
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np



class CorruptedData:
    def __init__(self,args,train_source=False):
        self.corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                                'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
                                'elastic_transform', 'pixelate', 'jpeg_compression']

        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        ])

        self.test_set = ImageFolder(os.path.join(args.dataset_folder,'ImageNet-C',self.corruption_types[args.test_env],args.severity),transform)
        



