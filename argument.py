import argparse
import numpy as np
import hashlib

parser = argparse.ArgumentParser(description='Exp',conflict_handler='resolve')

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--test_env', type=int, default=1)
parser.add_argument('--severity', type=str, default='5')
parser.add_argument('--method', type=str, default='TIPI')
parser.add_argument('--back_bone', type=str, default='resnet26')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--selective_entropy', type=str, default='False',choices=['True','False'])


# hyper paramters
# can use the HparamsGen to auto-generate hyper parameters when performning random search.
# However, if a hparam is specified when running the program, it will always use that value
class HparamsGen(object):
    def __init__(self, name, default, gen_fn=None):
        self.name = name
        self.default = default
        self.gen_fn = gen_fn
    def __call__(self,hparams_gen_seed=0):
        if hparams_gen_seed == 0 or self.gen_fn is None:
            return self.default
        else:
            s = f"{hparams_gen_seed}_{self.name}"
            seed = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (2**31)
            return self.gen_fn(np.random.RandomState(seed))
            
    
parser.add_argument('--hparams_gen_seed', type=int, default=0) # if not 0, used as the seed to generate hyper parameters when appicable
parser.add_argument('--batchsize', type=int, default=200)
parser.add_argument('--epsilon', type=float, default=2/255)

# must have for distributed code
parser.add_argument('--dataset_folder', type=str, default='/path/to/data/')
parser.add_argument('--experiment_path', type=str, default='./experiment_folder/')
parser.add_argument('--distributed', type=str, default='False',choices=['True','False'])
parser.add_argument('--world_size', type=int, default=1)
# for running by torch.distributed
parser.add_argument('--rank', type=int, default=0)
# for slurm
parser.add_argument('--local_rank', type=int, default=0)

# exclude unimportant args when saving the args
unimportant_args = ['save_checkpoint','load_unfinished','saved_folder','dataset_folder','experiment_path','distributed','world_size','rank','local_rank','unimportant_args']
