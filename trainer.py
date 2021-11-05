import time, abc
import torch
import util


class TrainerBase(metaclass=abc.ABCMeta):
    def __init__(self, train_loader, test_loader, network, optim, loss, precond, sched):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.network = network
        self.optim = optim
        self.loss = loss
        self.precond = precond
        self.sched = sched

        self.f_nfe_meter = util.ExponentialRunningAverageMeter()
        self.b_nfe_meter = util.ExponentialRunningAverageMeter()

    def time_start(self):
        self.time_start = time.time()

    @property
    def clock(self):
        return time.time() - self.time_start

    @abc.abstractmethod
    def prepare_var(self, opt, batch):
        raise NotImplementedError

    def get_ode_t1(self):
        return self.network.ode.t1.item()

    def get_last_lr(self, opt):
        if self.sched:
            lr = self.sched.get_last_lr()
            return opt.lr if lr is None else lr[0]
        else:
            return opt.lr

    def set_optimizer(self, opt, train_it):
        if self.precond: self.precond.train_itr_setup()
        if opt.use_adaptive_t1:
            self.network.ode.t1_train_itr_setup(train_it)
        self.optim.zero_grad()

    def forward_graph(self, opt, var, training=False):
        if training: self.network.train()
        self.network.ode.nfe = 0
        var.pred = self.network.forward(var.data)
        self.f_nfe_meter.update(self.network.ode.nfe)
        return var

    def run_optimizer(self, opt, train_it, var, loss):
        self.network.ode.nfe = 0
        loss.backward()
        self.b_nfe_meter.update(self.network.ode.nfe)

        if self.precond: self.precond.step()
        self.optim.step()

        if opt.use_adaptive_t1:
            self.network.ode.update_t1()

    def train_step(self, opt, train_it, batch):
        var = self.prepare_var(opt, batch)
        self.set_optimizer(opt, train_it)
        var = self.forward_graph(opt, var, training=True)

        loss = self.loss(var.pred, var.target)
        self.run_optimizer(opt, train_it, var, loss)

        return loss

    def evaluate(self, opt, ep, train_it, compute_accu=True):
        self.network.eval()
        loss_eval = 0.
        count = correct = total = 0
        with torch.no_grad():
            for it, batch in enumerate(self.test_loader):
                # compute var
                var = self.prepare_var(opt, batch)
                var = self.forward_graph(opt, var, training=False)

                # compute loss
                loss = self.loss(var.pred, var.target)
                batch_size = len(var.data)
                loss_eval += loss*batch_size
                count += batch_size

                # compute accuracy if needed
                if compute_accu:
                    _, predicted = torch.max(var.pred,1)
                    correct += (predicted==var.target).sum().item()
                    total += var.target.size(0)

        loss_eval /= count
        accuracy = correct/total if compute_accu else None
        return loss_eval, accuracy

    def save_checkpoint(self, opt, train_it, keys):
        util.save_checkpoint(opt, self, keys, train_it+1)

    def restore_checkpoint(self, opt, keys):
        if opt.load is not None:
            util.restore_checkpoint(opt, self, opt.load, keys)
        else:
            print(util.magenta("training from scratch..."))
