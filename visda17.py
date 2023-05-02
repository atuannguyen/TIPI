
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np



class CorruptedData:
    def __init__(self,args,train_source=False):
        transform = transforms.Compose([transforms.Resize([224,224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        ])

        self.test_set = ImageFolder(os.path.join(args.dataset_folder,'VisDA17/validation'),transform)



