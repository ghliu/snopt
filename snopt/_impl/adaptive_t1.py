import torch
import easydict, termcolor

from . import misc

def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])


def _compute_t1_update_baseline(state, odefunc, debug=False):
    s = state

    delta_t1 = - s.t1_lr * s.scale2.avg
    if debug:
        delta_t1_val = - s.t1_lr * s.scale2.val
        print(green("baseline_avg: {: .5f} | baseline_val: {: .5f}".format(delta_t1, delta_t1_val)))
    return delta_t1


def _compute_t1_update_feedback(state, odefunc, debug=False):
    s = state

    #### compute open gain ####
    delta_t1 = s.update_t1_open

    #### compute delta_u ####
    new_p_flat = torch.cat([p.reshape(-1) for p in odefunc.parameters() if p.requires_grad])
    delta_u = new_p_flat.detach().clone() - s.p_flat

    #### compute feedback ####
    feedback = - s.t1_lr * (s.scale.val * s.p_grad.val * delta_u).sum() / s.QTT_t1.avg # val

    if debug:
        print(green("feedback: {: .5f}".format(feedback)))

    return delta_t1 + feedback


class IntegrationTimeOptimizer(object):
    def __init__(self, opt, odefunc, t1, debug=False):
        assert opt.adaptive_t1 in ('baseline', 'feedback')

        self.debug  = debug

        self.odefunc = odefunc
        self.update_t1_itr = False
        self.t1 = t1.detach().clone()

        self.state = easydict.EasyDict()
        self.state.t1_lr  = opt.t1_lr
        self.state.method = opt.adaptive_t1
        self.state.t1_reg = opt.t1_reg
        self.state.t1_update_freq = opt.t1_update_freq

    def train_itr_setup(self, train_it):
        self.update_t1_itr = ((train_it+1) % self.state.t1_update_freq==0)


    def register_hook(self, x, out):
        odefunc, s = self.odefunc, self.state

        s.out = out.detach().clone()
        if s.method == 'feedback' and self.update_t1_itr:
            p_flat = torch.cat([p.reshape(-1) for p in odefunc.parameters() if p.requires_grad])
            s.p_flat = p_flat.detach().clone()

        # register hook
        x.register_hook(self._adaptive_t1_x_hook)
        out.register_hook(self._adaptive_t1_out_hook)
        return x, out


    def _adaptive_t1_out_hook(self, grad):
        odefunc, t1, s = self.odefunc, self.t1, self.state
        assert odefunc.training

        batch = grad.shape[0]
        odefunc.eval()
        with torch.no_grad():
            ######### compute scale #########
            func_eval = odefunc(t1, s.out).reshape(batch,-1)
            grad = grad.reshape(batch,-1)

            scale = ((func_eval * grad).sum()/batch).item()
            QT_t1  = s.t1_reg * t1.item() + scale # quadratic cost
            QTT_t1 = s.t1_reg + scale ** 2        # quadratic cost
            scale2 = ((func_eval * (grad + s.t1_reg * t1.item() )).sum()/batch).item()

            ######### store in state #########
            if 'scale' not in s: # initialization
                s.scale = misc.ExponentialRunningAverageMeter()
                s.QT_t1 = misc.ExponentialRunningAverageMeter()
                s.QTT_t1 = misc.ExponentialRunningAverageMeter()
                s.scale2 = misc.ExponentialRunningAverageMeter()
            s.scale.update(scale)
            s.QT_t1.update(QT_t1)
            s.QTT_t1.update(QTT_t1)
            s.scale2.update(scale2)

        odefunc.train()


    def _adaptive_t1_x_hook(self, grad):
        odefunc, s = self.odefunc, self.state

        s.update_t1_open = - s.t1_lr * s.QT_t1.avg / s.QTT_t1.avg # use moving average

        if self.debug:
            print("scale_avg: {: .4f} | QT_avg: {: .4f} | dT_avg: {: .4f} | T*: {: .4f}".format(
                s.scale.avg, s.QT_t1.avg, s.update_t1_open, -s.scale.avg/s.t1_reg
            ))

        if s.method == 'feedback':
            if 'p_grad' not in s: # initialization
                s.p_grad = misc.ExponentialRunningAverageMeter()
            p_grad = torch.cat([p.grad.reshape(-1) for p in odefunc.parameters() if p.requires_grad])
            s.p_grad.update(p_grad)


    def compute_new_t1(self):
        if not self.update_t1_itr: return self.t1

        old_t1, s, odefunc = self.t1, self.state, self.odefunc

        delta_t1 = {
            'baseline': _compute_t1_update_baseline,
            'feedback': _compute_t1_update_feedback,
        }.get(s.method)(s, odefunc, debug=self.debug)

        new_t1 = old_t1 + torch.tensor(delta_t1, device=old_t1.device)
        new_t1 = max(torch.tensor(0.1), new_t1)

        self.t1 = new_t1

        print(red('update terminal time from {: .4f} to {: .4f}'.format(old_t1, new_t1)))

        return new_t1
