from __future__ import print_function
import os, time
import torch
import torch.nn.functional as F

from datasets import get_uea_loader
from nets import TimeSeriesODENet
from trainer import TrainerBase

import util, options
import easydict

from torch.optim import SGD, Adam, lr_scheduler
from torchdiffeq import odeint_adjoint as odesolve
from snopt import SNOpt, ODEFuncBase, ODEBlock

import colored_traceback.always
from ipdb import set_trace as debug


def build_optim_and_precond_and_sched(opt, network):
    assert hasattr(network, 'gru_cell')
    assert hasattr(network, 'linear')
    assert hasattr(network, 'ode')

    # build optimizer
    optim_dict_ode = {"lr": opt.lr, 'weight_decay':opt.l2_norm, 'momentum':opt.momentum}
    optim_dict_other = {"lr": opt.lr2, 'weight_decay':opt.l2_norm}
    if opt.optimizer =='Adam': optim_dict_ode.pop('momentum', None)

    optim1 = Adam(network.gru_cell.parameters(), **optim_dict_other)
    optim2 = Adam(network.linear.parameters(),   **optim_dict_other)
    optim3 = {
        'SGD':SGD, 'Adam':Adam, 'SNOpt': SGD,
    }.get(opt.optimizer)(network.ode.parameters(), **optim_dict_ode)
    optim = util.OptimizerList([optim1, optim2, optim3])

    # build precond
    if opt.optimizer=='SNOpt':
        kwargs = dict(eps=opt.snopt_eps, update_freq=opt.snopt_freq, full_precond=False)
        precond = SNOpt(network, **kwargs)
    else:
        precond = None

    # build sched
    def build_sched(opt, optim):
        if opt.lr_gamma >= 1.0: return None
        return lr_scheduler.MultiStepLR(optim, milestones=opt.milestones, gamma=opt.lr_gamma)
    sched1 = build_sched(opt, optim1)
    sched2 = build_sched(opt, optim2)
    sched3 = build_sched(opt, optim3)
    sched = util.MultiStepLRList([sched1, sched2, sched3], allow_none=True)

    return optim, precond, sched


class Trainer(TrainerBase):
    def __init__(self, train_loader, test_loader, network, optim, loss,
            precond=None, sched=None):
        super(Trainer, self).__init__(
            train_loader, test_loader, network, optim, loss, precond, sched
        )

    def prepare_var(self, opt, batch):
        batch = tuple(b.to(opt.device) for b in batch)
        *train_coeffs, train_y, lengths = batch

        var = easydict.EasyDict()
        var.data = [opt.times, train_coeffs, lengths]
        var.target = train_y
        return var

class LinearODEfunc(ODEFuncBase):
    def __init__(self, opt, hidden):
        super(LinearODEfunc, self).__init__(opt)
        self.tanh = torch.nn.Tanh()
        self.fc1 = torch.nn.Linear(hidden, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, hidden)
        self.fc4 = torch.nn.Linear(hidden, hidden)

    def F(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        return out

def build_time_series_neural_ode(opt, hidden):
    odefunc = LinearODEfunc(opt, hidden)
    integration_time = None  # true integration_time will be feeded during forward
    ode = ODEBlock(opt, odefunc, odesolve, integration_time)
    network = TimeSeriesODENet(ode, hidden, opt.input_dim, opt.output_dim).to(opt.device)

    print(network)
    print(util.magenta("Number of trainable parameters: {}".format(
        util.count_parameters(network)
    )))
    return network

if __name__ == '__main__':

    # build opt and trainer
    opt = options.set()
    train_loader, test_loader = get_uea_loader(opt)
    network = build_time_series_neural_ode(opt, opt.nhidden)
    optim, precond, sched = build_optim_and_precond_and_sched(opt, network)
    loss = F.cross_entropy
    trainer = Trainer(train_loader, test_loader, network, optim, loss, precond=precond, sched=sched)
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

    # strat training
    print(util.yellow("======= TRAINING START ======="))
    print(util.green(path))
    trainer.time_start()
    for ep in range(opt.epoch):
        for it, batch in enumerate(trainer.train_loader):
            train_it = ep*len(trainer.train_loader)+it

            loss = trainer.train_step(opt, train_it, batch=batch)
            util.print_train_progress(opt, trainer, train_it, loss)

            losses.append(loss)
            train_clocks.append(trainer.clock)

            if (train_it+1)%opt.eval_itr==0:
                eval_loss, accuracy=trainer.evaluate(opt, ep, train_it)
                util.print_eval_progress(opt, trainer, train_it, eval_loss, accuracy=accuracy)

                eval_losses.append(eval_loss)
                accuracies.append(accuracy)
                eval_clocks.append(trainer.clock)

            if (train_it+1)%opt.checkpoint_it==0:
                trainer.save_checkpoint(opt, train_it+1, keys=["network","optim"])

        losses.save()
        eval_losses.save()
        accuracies.save()

        train_clocks.save()
        eval_clocks.save()

    time.sleep(1)
    print(util.yellow("======= TRAINING DONE ======="))
