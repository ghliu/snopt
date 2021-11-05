from __future__ import print_function
import os, time
import torch
import torch.nn.functional as F

from datasets import get_img_loader
from nets import ConvODENet
from trainer import TrainerBase

import util, options
import easydict

from torch.optim import SGD, Adam
from torchdiffeq import odeint_adjoint as odesolve
from snopt import SNOpt, ODEFuncBase, ODEBlock

import colored_traceback.always
from ipdb import set_trace as debug


def build_optim_and_precond(opt, network):
    # build optimizer
    optim_dict = {"lr": opt.lr, 'weight_decay':opt.l2_norm, 'momentum':opt.momentum}
    if opt.optimizer =='Adam': optim_dict.pop('momentum', None)
    optim = {
        'SGD': SGD,
        'Adam': Adam,
        'SNOpt': SGD,
    }.get(opt.optimizer)(network.parameters(), **optim_dict)

    # build precond
    if opt.optimizer=='SNOpt':
        kwargs = dict(eps=opt.snopt_eps, update_freq=opt.snopt_freq, full_precond=True)
        precond = SNOpt(network, **kwargs)
    else:
        precond = None

    return optim, precond

class ConcatConv2d(torch.nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ConvODEfunc(ODEFuncBase):
    def __init__(self, opt, hidden):
        super(ConvODEfunc, self).__init__(opt)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(hidden, hidden, 3, 1, 1)
        self.conv2 = ConcatConv2d(hidden, hidden, 3, 1, 1)

    def F(self, t, x):
        self.nfe += 1
        out = x
        out = self.conv1(t, out)
        out = self.relu(out)
        out = self.conv2(t, out)
        return out


class Trainer(TrainerBase):
    def __init__(self, train_loader, test_loader, network, optim, loss,
            precond=None, sched=None):
        super(Trainer, self).__init__(
            train_loader, test_loader, network, optim, loss, precond, sched
        )

    def prepare_var(self, opt, batch):
        var = easydict.EasyDict()
        [var.data, var.target] = [v.to(opt.device) for v in batch]
        return var

def build_clf_neural_ode(opt, hidden=64, t0=0.0, t1=1.0):
    odefunc = ConvODEfunc(opt, hidden)
    integration_time = torch.tensor([t0, t1]).float()
    ode = ODEBlock(opt, odefunc, odesolve, integration_time, is_clf_problem=True)
    network = ConvODENet(ode, hidden, opt.input_dim[0]).to(opt.device)

    print(network)
    print(util.magenta("Number of trainable parameters: {}".format(
        util.count_parameters(network)
    )))
    return network


if __name__ == '__main__':

    # build opt and trainer
    opt = options.set()
    train_loader, test_loader = get_img_loader(opt)
    network = build_clf_neural_ode(opt, t1=opt.t1)
    optim, precond = build_optim_and_precond(opt, network)
    loss = F.cross_entropy
    trainer = Trainer(train_loader, test_loader, network, optim, loss, precond=precond)
    trainer.restore_checkpoint(opt, keys=["network","optim"])

    # save path
    os.makedirs(opt.result_dir, exist_ok=True)
    path = "{}/{}-{}_seed_{}_".format(opt.result_dir, opt.problem, opt.optimizer_config, opt.seed)

    # things we're going to collect over training
    losses       = util.Collector(path + 'train')
    eval_losses  = util.Collector(path + 'eval')
    accuracies   = util.Collector(path + 'accuracy')
    train_clocks = util.Collector(path + 'train_clock')
    eval_clocks  = util.Collector(path + 'eval_clock')
    if opt.use_adaptive_t1: t1s = util.Collector(path + 't1s')

    # strat training
    print(util.yellow("======= TRAINING START ======="))
    print(util.green(path))
    trainer.time_start()
    for ep in range(opt.epoch):
        for it, batch in enumerate(trainer.train_loader):
            train_it = ep*len(trainer.train_loader)+it

            loss = trainer.train_step(opt, train_it, batch=batch)
            # util.print_train_progress(opt, trainer, train_it, loss)

            losses.append(loss)
            train_clocks.append(trainer.clock)
            if opt.use_adaptive_t1: t1s.append(trainer.get_ode_t1())

            if (train_it+1)%opt.eval_itr==0:
                eval_loss, accuracy=trainer.evaluate(opt, ep, train_it)
                util.print_eval_progress(opt, trainer, train_it, eval_loss, accuracy=accuracy)

                eval_losses.append(eval_loss)
                accuracies.append(accuracy)
                eval_clocks.append(trainer.clock)

        losses.save()
        eval_losses.save()
        accuracies.save()

        train_clocks.save()
        eval_clocks.save()
        if opt.use_adaptive_t1: t1s.save()

    time.sleep(1)
    print(util.yellow("======= TRAINING DONE ======="))
