import torch

from . import adaptive_t1 as adapt


def get_odesolve_kwargs(opt, **kwargs):
    # ===== build shared kwargs =====
    odesolve_kwargs = {
        'rtol':opt.tol,
        'atol':opt.tol,
        'method': opt.ode_solver,
    }

    # overwrite if for keys specified in kwargs
    for k,v in kwargs.items():
        if k in odesolve_kwargs:
            odesolve_kwargs[k] = v
        else:
            print('Unused keys {}={} when initializing ODE solver'.format(k,v))

    # build options
    odesolve_kwargs["options"] = dict(use_snopt=(opt.optimizer=='SNOpt'))
    if hasattr(opt,'ode_step_size'):
        odesolve_kwargs["options"]['step_size'] = opt.ode_step_size

    if hasattr(opt,'seminorm') and opt.seminorm:
        odesolve_kwargs["adjoint_options"] = odesolve_kwargs["options"].copy()
        odesolve_kwargs["adjoint_options"]['norm'] = 'seminorm'
    return odesolve_kwargs


class ODEBlock(torch.nn.Module):

    def __init__(self, opt, odefunc, odesolve, ts, is_clf_problem=False, **kwargs):
        """ Neural ODE Block.
        Args:
            opt: options config that setups the Neural ODE computation.
            odefunc (torch.nn.Module): The parametrized vector field.
            odesolve: a function call of ODE solver (e.g., torchdiffeq.odeint
                      or torchdiffeq.odeint_adjoint)
            ts (torch.tensor): integration time [t_0, t_1]
            is_clf_problem (bool): Whether the module is used in classfication.
        """
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.odesolve = odesolve
        self.odesolve_kwargs = get_odesolve_kwargs(opt, **kwargs)
        self.ts = ts
        self.tol = opt.tol
        self.opt = opt # TODO need this for snopt
        self.is_clf_problem = is_clf_problem

        if opt.use_adaptive_t1:
            self.t1_optim = adapt.IntegrationTimeOptimizer(opt, odefunc, ts[-1])

    def update_t1(self):
        assert hasattr(self, 't1_optim')
        new_t1 = self.t1_optim.compute_new_t1()
        self.ts = torch.tensor([self.ts[0], new_t1]).float().to(self.ts.device)

    def forward(self, x, ts=None):
        ts = self.ts if ts is None else ts
        ts = ts.type_as(x[0] if isinstance(x, tuple) else x)

        out = self.odesolve(self.odefunc, x, ts, **self.odesolve_kwargs)

        if self.is_clf_problem:
            out = out[-1]

        if self.training and hasattr(self, 't1_optim'):
            x, out = self.t1_optim.register_hook(x, out)

        return out

    def t1_train_itr_setup(self, train_it):
        assert hasattr(self, 't1_optim')
        self.t1_optim.train_itr_setup(train_it)

    @property
    def t1(self):
        return self.ts[-1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
