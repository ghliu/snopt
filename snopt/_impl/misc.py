import torch
import torch.nn.functional as F
import warnings


def get_class_name(mod):
    return mod.__class__.__name__


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


def divergence_approx(f, y, e=None):
    """Calculates the trace of the Jacobian df/dz.
       source: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py
    """
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def divergence_exact(f, z):
    """Calculates the trace of the Jacobian df/dz.
       source: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


def compute_xxt(mod, x, gathering_filter=None):
    # ==== Build correct x ====
    if get_class_name(mod) == 'Conv2d':
        assert gathering_filter is not None
        x = F.conv2d(x, gathering_filter,
                     stride=mod.stride, padding=mod.padding,
                     groups=mod.in_channels)
        x = x.data.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
    else:
        if x.dim() > 2:
            warnings.warn('TODO: need to check if x.dim()>2 is properly handled (bs may needs to be changed).')
        x = x.data.reshape(-1, x.shape[-1]).t()

    if mod.bias is not None:
        ones = torch.ones_like(x[:1])
        x = torch.cat([x, ones], dim=0)

    # ==== Compute xxt ====
    return torch.mm(x, x.t()) / float(x.shape[1])


def compute_ggt(mod, gy):
    # ==== Build correct gy ====
    if get_class_name(mod) == 'Conv2d':
        gy = gy.data.permute(1, 0, 2, 3)
        gy = gy.contiguous().view(gy.shape[0], -1) # Cout x everything else
    else:
        gy = gy.data.reshape(-1, gy.shape[-1]).t()

    # ==== Compute ggt ====
    return torch.mm(gy, gy.t()) / float(gy.shape[1])


def compute_nloc(mod, gy):
    if get_class_name(mod) == 'Conv2d':
        num_locations = gy.shape[2] * gy.shape[3]
    else:
        num_locations = 1

    return num_locations


def build_gathering_filter(mod):
    """Convolution filter that extracts input patches.
       source: https://github.com/Thrandis/EKFAC-pytorch/blob/master/ekfac.py
    """

    if not get_class_name(mod)=='Conv2d': return None

    kw, kh = mod.kernel_size
    g_filter = mod.weight.data.new(kw * kh * mod.in_channels, 1, kw, kh)
    g_filter.fill_(0)
    for i in range(mod.in_channels):
        for j in range(kw):
            for k in range(kh):
                g_filter[k + kh*j + kw*kh*i, 0, j, k] = 1
    return g_filter


def create_snopt_samp_t(samp_t, samp_step_size):
    assert samp_t.size(0) == 2
    start_t, end_t = samp_t[0], samp_t[-1]

    # Create additional time samples between the duration samp_t. these are
    # samples we'll use to construct fisher factorization
    new_samp_t = torch.arange(start_t, end_t, -samp_step_size, device=samp_t.device)

    # Make sure the end point in samp_t is included
    if (new_samp_t[-1] - end_t).abs() < samp_step_size/2:
        new_samp_t[-1] = end_t
    else:
        new_samp_t = torch.cat((new_samp_t, end_t.reshape(1)))

    return new_samp_t

class ExponentialRunningAverageMeter(object):
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

class RunningAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.N = 0

    def _avg_torch(self, val):
        if self.val is None:
            self.avg = val.detach().clone()
        else:
            self.avg.mul_(self.N/(self.N+1)).add_(val, alpha=1/(self.N+1))

    def _avg(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg*self.N/(self.N+1) + val/(self.N+1)

    def update(self, val):
        if torch.is_tensor(val):
            self._avg_torch(val)
        else:
            self._avg(val)
        self.val = val
        self.N += 1

    @property
    def sum(self):
        return self.avg * self.N
