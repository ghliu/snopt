from __future__ import print_function
import os, time
import torch

from datasets import get_tabular_loader
from nets import CNFODENet
from trainer import TrainerBase

import util, options
import easydict

from torch.optim import SGD, Adam
from torchdiffeq import odeint_adjoint as odesolve
from snopt import SNOpt, CNFFuncBase, ODEBlock

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
        kwargs = dict(eps=opt.snopt_eps, update_freq=opt.snopt_freq, full_precond=False)
        precond = SNOpt(network, **kwargs)
    else:
        precond = None

    return optim, precond

class ConcatSquashLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = torch.nn.Linear(dim_in, dim_out)
        self._hyper_bias = torch.nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = torch.nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))) \
            + self._hyper_bias(t.view(1, 1))


class LinearSoftplusODEfunc(CNFFuncBase):
    def __init__(self, opt, hidden):
        super(LinearSoftplusODEfunc, self).__init__(opt)
        self.csl_0 = ConcatSquashLinear(opt.input_dim, hidden)
        self.csl_1 = ConcatSquashLinear(hidden, hidden)
        self.csl_2 = ConcatSquashLinear(hidden, opt.input_dim)

        self.activ_0 = torch.nn.Softplus()
        self.activ_1 = torch.nn.Softplus()

    def F(self, t, z):
        z = self.csl_0(t,z)
        z = self.activ_0(z)
        z = self.csl_1(t,z)
        z = self.activ_1(z)
        dz_dt = self.csl_2(t,z)
        return dz_dt

class LinearTanhODEfunc(CNFFuncBase):
    def __init__(self, opt, hidden):
        super(LinearTanhODEfunc, self).__init__(opt)
        self.csl_0 = ConcatSquashLinear(opt.input_dim, hidden)
        self.csl_1 = ConcatSquashLinear(hidden, hidden)
        self.csl_2 = ConcatSquashLinear(hidden, hidden)
        self.csl_3 = ConcatSquashLinear(hidden, opt.input_dim)

        self.activ_0 = torch.nn.Tanh()
        self.activ_1 = torch.nn.Tanh()
        self.activ_2 = torch.nn.Tanh()

    def F(self, t, z):
        z = self.csl_0(t,z)
        z = self.activ_0(z)
        z = self.csl_1(t,z)
        z = self.activ_1(z)
        z = self.csl_2(t,z)
        z = self.activ_2(z)
        dz_dt = self.csl_3(t,z)
        return dz_dt


class Trainer(TrainerBase):
    def __init__(self, train_loader, test_loader, network, optim, loss,
            precond=None, sched=None):
        super(Trainer, self).__init__(
            train_loader, test_loader, network, optim, loss, precond, sched
        )

    def prepare_var(self, opt, batch):
        var = easydict.EasyDict()
        [var.data, var.target] = batch
        return var

def build_cnf_neural_ode(opt, t0=0.0, t1=1.0):
    integration_time = torch.tensor([t1, t0]).float()
    odefunc_builder = {
        'miniboone': LinearSoftplusODEfunc,
        'gas': LinearTanhODEfunc,
    }.get(opt.problem)

    def ode_builder():
        odefunc = odefunc_builder(opt, opt.nhidden)
        return ODEBlock(opt, odefunc, odesolve, integration_time, atol=1e-8, rtol=1e-6)
    n_ode = 5 if opt.problem=='gas' else 1
    network = CNFODENet(n_ode, ode_builder).to(opt.device)

    print(network)
    print(util.magenta("Number of trainable parameters: {}".format(
        util.count_parameters(network)
    )))
    return network


def cnf_loss(out_var, samp):
    z_t0,logp_diff_t0=out_var
    p_z0=samp
    logp_x = p_z0.log_prob(z_t0).to(z_t0.device) - logp_diff_t0.view(-1)
    loss = -logp_x.mean(0)
    return loss

if __name__ == '__main__':

    # build opt and trainer
    opt = options.set()
    train_loader, test_loader = get_tabular_loader(opt)
    network = build_cnf_neural_ode(opt)
    optim, precond = build_optim_and_precond(opt, network)
    loss = cnf_loss
    trainer = Trainer(train_loader, test_loader, network, optim, loss, precond=precond)
    trainer.restore_checkpoint(opt, keys=["network","optim"])

    # save path
    os.makedirs(opt.result_dir, exist_ok=True)
    path = "{}/{}-{}_seed_{}_".format(opt.result_dir, opt.problem, opt.optimizer_config, opt.seed)

    # things we're going to collect over training
    losses       = util.Collector(path + 'train')
    eval_losses  = util.Collector(path + 'eval')
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
                eval_loss, _ = trainer.evaluate(opt, ep, train_it, compute_accu=False)
                util.print_eval_progress(opt, trainer, train_it, eval_loss)

                eval_losses.append(eval_loss)
                eval_clocks.append(trainer.clock)

        losses.save()
        eval_losses.save()

        train_clocks.save()
        eval_clocks.save()

    time.sleep(1)
    print(util.yellow("======= TRAINING DONE ======="))
