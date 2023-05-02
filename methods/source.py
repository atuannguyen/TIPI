import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.distributions as distributions
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from copy import deepcopy

from util import *
from methods.base import *

class Model(Base):
    def __init__(self,args):
        self.probabilistic = False
        super(Model, self).__init__(args)


    def test(self,testloader):
        acc_meter = AverageMeter()
        for x,y in tqdm(testloader,ncols=75,leave=False):
            x,y = x.to(self.device), y.to(self.device)
            preds = self.model(x)
            
            acc = (preds.argmax(1)==y).float().mean()
            acc_meter.update(acc.data,x.shape[0])
        print(f'>>> Target ACC {acc_meter.average()}')

        


