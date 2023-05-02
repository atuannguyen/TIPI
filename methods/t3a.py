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
        super(Model, self).__init__(args)


    def test(self,testloader):
        if hasattr(self.model,'fc'):
            classifier = self.model.fc
            self.model.fc = nn.Identity()
            featurizer = self.model
        else:
            classifier = self.model.linear
            self.model.linear = nn.Identity()
            featurizer = self.model

        model = T3A(featurizer,classifier,self.num_classes,filter_K=True)
        acc_meter = AverageMeter()
        for x,y in tqdm(testloader,ncols=75,leave=False):
            x,y = x.to(self.device), y.to(self.device)
            preds = model(x)
            
            acc = (preds.argmax(1)==y).float().mean()
            acc_meter.update(acc.data,x.shape[0])
        print(f'>>> Target ACC {acc_meter.average()}')

        

class T3A(nn.Module):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, featurizer, classifier, num_classes, filter_K):
        super(T3A,self).__init__()
        self.featurizer = featurizer
        self.classifier = classifier

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = filter_K
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, adapt=True):
        with torch.no_grad():
            z = self.featurizer(x)
            if adapt:
                # online adaptation
                p = self.classifier(z)
                yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
                ent = softmax_entropy(p)

                # prediction
                self.supports = self.supports.to(z.device)
                self.labels = self.labels.to(z.device)
                self.ent = self.ent.to(z.device)
                self.supports = torch.cat([self.supports, z])
                self.labels = torch.cat([self.labels, yhat])
                self.ent = torch.cat([self.ent, ent])
            
            supports, labels = self.select_supports()
            supports = torch.nn.functional.normalize(supports, dim=1)
            weights = (supports.T @ (labels))
            return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data




@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
