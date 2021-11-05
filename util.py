import numpy as np
import torch
import os, time
import termcolor

# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = sec%60
    return h,m,s

def get_time_elapsed(start):
    return get_time(time.time()-start)

def restore_checkpoint(opt, model, load_name, keys):
    print(magenta("loading checkpoint {}...".format(load_name)))
    with torch.cuda.device(opt.gpu):
        checkpoint = torch.load(load_name,map_location=opt.device)
        for k in keys:
            getattr(model,k).load_state_dict(checkpoint[k])

def save_checkpoint(opt, model, keys, train_it):
    os.makedirs("checkpoint/{0}/{1}".format(opt.group,opt.name), exist_ok=True)
    checkpoint = {}
    with torch.cuda.device(opt.gpu):
        for k in keys:
            checkpoint[k] = getattr(model,k).state_dict()
        torch.save(checkpoint,"checkpoint/{0}/{1}/train_it{2}.npz".format(opt.group, opt.name, train_it))
    print(green("checkpoint saved: ({0}) {1}, train_it {2}".format(opt.group, opt.name, train_it)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_train_progress(opt, trainer, train_it, loss):
    f_nfe = trainer.f_nfe_meter.avg
    b_nfe = trainer.b_nfe_meter.avg
    lr = trainer.get_last_lr(opt)
    time_elapsed = get_time_elapsed(trainer.time_start)
    max_itr = opt.epoch*len(trainer.train_loader)

    print("[TRAIN] train_it {0}/{1} | Time {2} | NFE-F {3} | NFE-B {4} | Train Loss {5} | lr:{6}"
        .format(cyan("{}".format(train_it+1)),
                max_itr,
                green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
                yellow("{:.1f}".format(f_nfe)),
                yellow("{:.1f}".format(b_nfe)),
                red("{:.4f}".format(loss)),
                yellow("{:.2e}".format(lr)),
    ))

def print_eval_progress(opt, trainer, train_it, loss_eval, accuracy=None):
    f_nfe = trainer.f_nfe_meter.avg
    b_nfe = trainer.b_nfe_meter.avg
    time_elapsed = get_time_elapsed(trainer.time_start)
    max_itr = opt.epoch*len(trainer.train_loader)

    print("[EVAL] train_it {0}/{1} | Time {2} | NFE-F {3} | NFE-B {4} | Val Loss {5} {6}"
        .format(cyan("{}".format(train_it+1)),
                max_itr,
                green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
                yellow("{:.1f}".format(f_nfe)),
                yellow("{:.1f}".format(b_nfe)),
                red("{:.4f}".format(loss_eval)),
                "| Test Acc {}".format(red("{:.4f}".format(accuracy))) if accuracy is not None else "",
    ))

class OptimizerList:
    def __init__(self, optims):
        for optim in optims:
            assert isinstance(optim, torch.optim.Optimizer)
        self.optims = optims

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def step(self):
        for optim in self.optims:
            optim.step()

    def state_dict(self):
        d = dict()
        for idx, optim in enumerate(self.optims):
            d[str(idx)] = optim.state_dict()
        return d

    def load_state_dict(self, state_dict):
        assert len(state_dict) == len(self.optims)
        for idx, optim in enumerate(self.optims):
            optim.load_state_dict(state_dict[str(idx)])

class MultiStepLRList:
    def __init__(self, scheds, allow_none=False):
        for sched in scheds:
            assert self._check(sched, allow_none)
        self.scheds = scheds

    def _check(self, sched, allow_none):
        is_multi_step_lr = isinstance(sched, torch.optim.lr_scheduler.MultiStepLR)
        return (is_multi_step_lr or sched is None) if allow_none else is_multi_step_lr

    def step(self):
        for sched in self.scheds:
            if sched: sched.step()

    def get_last_lr(self):
        for sched in self.scheds:
            if sched: return sched.get_last_lr()

class ExponentialRunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class Collector(object):
    """docstring for Collector"""

    def __init__(self, fn):
        super(Collector, self).__init__()
        self.fn = fn if fn[-4:] == '.npy' else fn+'.npy'
        self.collections = list()

    def append(self,item):
        if isinstance(item,torch.Tensor):
            item=item.item()
        self.collections.append(item)

    def save(self):
        np.save(self.fn, np.array(self.collections))

def _rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()

def make_seminorm(state):
    if isinstance(state, tuple):
        state_size = sum([s.numel() for s in state])
    elif torch.is_tensor(state):
        state_size = state.numel()

    def seminorm(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(_rms_norm(y), _rms_norm(adj_y))

    return seminorm
