import os 
import random
import sys 
import importlib
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from datetime import datetime
import subprocess
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np

from util import *
from prepare import *
if args.dataset.startswith('cifar'):
    from cifar import *
elif args.dataset == 'digits':
    from digits import *
elif args.dataset == 'visda17':
    from visda17 import *
else:
    from imagenet import *

def test(args):
    data = CorruptedData(args)
    print(len(data.test_set))
    args.num_classes = len(data.test_set.classes)

    test_loader = DataLoader(data.test_set,batch_size=args.batchsize,shuffle=True,num_workers=4)

    algo = importlib.import_module('methods.'+args.method).Model(args=args).to(device)

    algo.test(test_loader)
    print('==========================')

def test_all(args):
    for i in range(15):
        print('Env:', i)
        args.test_env = i
        test(args)

if __name__=='__main__':
    if args.test_env == -1:
        test_all(args)
    else:
        test(args)

