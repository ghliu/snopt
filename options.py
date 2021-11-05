import numpy as np
import os
import argparse
import torch
import random

def set():

    parser = argparse.ArgumentParser()

    # general options
    parser.add_argument("--group",                      default="0",        help="name for group")
    parser.add_argument("--name",                       default="debug",    help="name for model instance")
    parser.add_argument("--seed",           type=int,   default=0,          help="fix random seed")
    parser.add_argument("--gpu",            type=int,   default=0,          help="GPU device")
    parser.add_argument("--load",           type=str,   default=None,       help="load (pre)trained model")
    parser.add_argument("--problem",        type=str,   default='toy1',     help="")
    parser.add_argument("--optimizer",      type=str,   default='sgd',      help="")
    parser.add_argument("--batch-size",     type=int,   default=32,         help="input batch size")
    parser.add_argument("--lr",             type=float, default=1e-2,       help="base learning rate")
    parser.add_argument("--epoch",          type=int,   default=500,        help="train to epoch number")
    parser.add_argument("--momentum",       type=float, default=0,          help="")
    parser.add_argument("--l2-norm",        type=float, default=0,          help="weight decay")
    parser.add_argument("--nhidden",        type=int,   default=20,         help="number of hidden dimension")
    parser.add_argument("--eval-itr",       type=int,   default=10,         help="iteration to report evaluation")
    parser.add_argument("--checkpoint-it",  type=int,   default=100000,     help="iteration to store checkpoint")
    parser.add_argument("--result-dir",     type=str,   default='result',   help="where to store the result npy")

    # time series specific options
    parser.add_argument("--lr2",            type=float, default=1e-2,       help="second learning rate for non-NeuralODE modules")
    parser.add_argument("--lr-gamma",       type=float, default=1.0,        help="learning rate decay")
    parser.add_argument("--milestones",     type=int,   default=None,       nargs='+', help="milestones to decay learning rate")

    # continuous nf specific options
    parser.add_argument('--divergence-type',type=str,   default='rademacher',choices=['rademacher', 'gaussian', 'exact'], help="how divergence is being computed in CNF")
    parser.add_argument("--dataset-ratio",  type=float, default=1.0,        help="number of data we use from dataset")


    # torchdiffeq options
    parser.add_argument("--ode-solver",     type=str,   default='dopri5',   help='ODE solver for both forward & adjoint')
    parser.add_argument("--ode-step-size",  type=float, default=0.05,       help="only for fixed grid ODE solver")
    parser.add_argument('--tol',            type=float, default=1e-3,       help="tolerance of ODE solver")
    parser.add_argument("--t1",             type=float, default=1.0,        help="integration horizon of Neural ODE")
    parser.add_argument("--seminorm",       action='store_true',            help="use semi-norm in Neural ODE")

    # snopt specific
    parser.add_argument("--snopt-freq",     type=int,   default=100,        help="frequency to refresh precondition matrices")
    parser.add_argument("--snopt-eps",      type=float, default=0.05,       help="Tikhonov regularization")
    parser.add_argument("--snopt-step-size",type=float, default=0.01,       help="the step size which we sample along the ODE solver")
    parser.add_argument("--adaptive-t1",    type=str,   default='disable',  choices=['disable','baseline','feedback'], help="type of adaptive t1 optimizer")
    parser.add_argument("--t1-lr",          type=float, default=0.1,        help="")
    parser.add_argument("--t1-reg",         type=float, default=0.01,       help="quadratic penalty on having larger integration horizon t1")
    parser.add_argument("--t1-update-freq", type=int,   default=100,        help="frequency to update integration horizon t1")

    opt = parser.parse_args()

    if opt.seed is not None:
        # https://github.com/pytorch/pytorch/issues/7068
        seed = opt.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    opt.device = "cuda:{}".format(opt.gpu) if torch.cuda.is_available() else 'cpu'
    opt.use_adaptive_t1 = (opt.adaptive_t1 != 'disable')
    opt.optimizer_config = get_optimizer_config_string(opt)

    return opt


def get_optimizer_config_string(opt):
    # share config for all optimizers
    config = '{}_lr{}_l2{}_m{}'.format(opt.optimizer, opt.lr, opt.l2_norm, opt.momentum)

    # snopt-specific config
    if opt.optimizer == 'SNOpt':
        config += '_eps{}_freq{}_adj_step{}'.format(opt.snopt_eps, opt.snopt_freq, opt.snopt_step_size)

    # t1 optimization specific config
    if opt.use_adaptive_t1:
        config += '-{}_lr{}_reg{}_freq{}'.format(opt.adaptive_t1, opt.t1_lr, opt.t1_reg, opt.t1_update_freq)

    return config
