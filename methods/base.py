import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.distributions as distributions
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel
from robustbench.model_zoo.architectures.wide_resnet import WideResNet
from robustbench.model_zoo.architectures.resnet import ResNet, BasicBlock

from util import *



class Base(nn.Module):
    def __init__(self,args):
        super(Base, self).__init__()
        
        if args.dataset == 'cifar10':
            args.num_classes = 10
            if args.back_bone == 'resnet26':
                model = ResNet(BasicBlock, [2, 3, 4, 2], args.num_classes)
                model.load_state_dict(torch.load(f'models/{args.back_bone}_{args.dataset}.pt'))
            elif args.back_bone == 'resnet26GN':
                model = ResNet(BasicBlock, [2, 3, 4, 2], args.num_classes)
                replace_BNwGN(model,8)
                model.load_state_dict(torch.load(f'models/{args.back_bone}_{args.dataset}.pt'))
            elif args.back_bone == 'wideresnet28-10':
                model = load_model('Standard', dataset='cifar10', threat_model=ThreatModel.corruptions)
        elif args.dataset == 'digits':
            args.num_classes = 10
            if args.back_bone == 'resnet26':
                model = ResNet(BasicBlock, [2, 3, 4, 2], args.num_classes)
                model.load_state_dict(torch.load(f'models/{args.back_bone}_{args.dataset}_3.pt'))
            elif args.back_bone == 'wideresnet28-10':
                model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)
                model.load_state_dict(torch.load(f'models/{args.back_bone}_{args.dataset}.pt'))
        elif args.dataset == 'cifar100':
            args.num_classes = 100
            if args.back_bone == 'resnet26':
                model = ResNet(BasicBlock, [2, 3, 4, 2], args.num_classes)
                model.load_state_dict(torch.load(f'models/{args.back_bone}_{args.dataset}.pt'))
            elif args.back_bone == 'wideresnet28-10':
                model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)
                model.load_state_dict(torch.load(f'models/{args.back_bone}_{args.dataset}.pt'))
        elif args.dataset == 'imagenet':
            model = models.resnet50(pretrained=True)
        elif args.dataset == 'visda17':
            model = models.resnet50(num_classes=args.num_classes)
            model.load_state_dict(torch.load(f'models/resnet50_{args.dataset}.pt'))
        else:
            NotImplementedError

        self.model = model
        self.model.to(args.device)

        self.model.eval()

        for name in args.__dict__:
            setattr(self,name,getattr(args,name))


def replace_BNwGN(net,num_groups=8):
    for attr_str in dir(net):
        m = getattr(net, attr_str)
        if type(m) == nn.BatchNorm2d:
            new_bn = nn.GroupNorm(num_groups=num_groups,num_channels=m.num_features)
            setattr(net, attr_str, new_bn)
    for n, ch in net.named_children():
        replace_BNwGN(ch, num_groups)
