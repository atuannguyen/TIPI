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

from tipi import TIPI

class Model(Base):
    def __init__(self,args):
        self.probabilistic = False
        super(Model, self).__init__(args)


    def test(self,testloader):
        if self.dataset == 'imagenet':
            tipi_model = TIPI(self.model, lr_per_sample=0.00025/64, optim='SGD', epsilon=self.epsilon, random_init_adv=True, tent_coeff=1.0)
        elif self.dataset == 'visda17':
            tipi_model = TIPI(self.model, lr_per_sample=0.00025/64, optim='SGD', epsilon=self.epsilon, random_init_adv=True, tent_coeff=1.0)
        elif self.dataset == 'digits':
            tipi_model = TIPI(self.model, lr_per_sample=0.001/128, optim='Adam', epsilon=self.epsilon, random_init_adv=False, tent_coeff=1.0)
        elif self.dataset == 'cifar100':
            tipi_model = TIPI(self.model, lr_per_sample=0.001/200, optim='Adam', epsilon=self.epsilon, random_init_adv=False, tent_coeff=5.0, use_test_bn_with_large_batches=True)
        else:
            tipi_model = TIPI(self.model, lr_per_sample=0.001/200, optim='Adam', epsilon=self.epsilon, random_init_adv=False, tent_coeff=0.0, use_test_bn_with_large_batches=True)

        meters = defaultdict(AverageMeter)
        acc_meter = AverageMeter()
        for x,y in tqdm(testloader,ncols=75,leave=False):
            x,y = x.to(self.device), y.to(self.device)
            
            preds = tipi_model(x)
            acc = (preds.argmax(1)==y).float().mean()
            acc_meter.update(acc.data,x.shape[0])
        print(f'>>> Target ACC Ours {acc_meter.average()}')



