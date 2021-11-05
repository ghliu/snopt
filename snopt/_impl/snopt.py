import torch
from torch.optim.optimizer import Optimizer

from . import misc
from .base import ODEFuncBase


class SNOptAdjointCollector():

    def __init__(self, odefunc):
        """ SNOpt collector during adjoint process.
        Args:
            odefunc (ODEFuncBase): the (possibly wrapped) Neural ODE
        """

        # find the base DOE dynamics of class ODEFuncBase
        # *** this ASSUMES the syntax "base_func" in torchdiffeq ***
        while not isinstance(odefunc, ODEFuncBase):
            assert hasattr(odefunc, 'base_func')
            odefunc = odefunc.base_func

        self.odefunc = odefunc
        self.step_size = odefunc.opt.snopt_step_size
        if self.store_this_itr:
            self.odefunc.initialize_snopt_state()

    @property
    def store_this_itr(self):
        return self.odefunc.ctx.store_this_itr

    def call_invoke(self, func, t, y):
        if not self.store_this_itr: return
        self.odefunc.ctx.save_input_flag='Save'
        func(t, y)
        self.odefunc.ctx.save_input_flag='Stop'

    def check_inputs(self, adjoint_options, samp_t):
        if self.store_this_itr:
            adjoint_options['snopt_collector'] = self
            if self.step_size < (samp_t[-1] - samp_t[0]).abs():
                samp_t = misc.create_snopt_samp_t(samp_t, self.step_size)
        return adjoint_options, samp_t


class SNOpt(Optimizer):

    def __init__(self, net, eps=0.05, update_freq=100, alpha=.75, full_precond=False, debug=False):
        """ SNOpt Preconditionner for Neural ODEs.
        Computes second order updates.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter
            full_precond (bool): Preconditioning layers that aren't belong to Neural ODE.
        """

        assert hasattr(net, 'odes'), "SNOpt requires specifying net.odes"
        assert hasattr(net, 'ode_mods'), "SNOpt requires specifying net.ode_mods"
        for ode in net.odes: hasattr(ode, 'odefunc'), "module {} does not have odefunc".format(ode)

        self.net = net
        self.eps = eps
        self.update_freq = update_freq
        self.alpha = alpha
        self.full_precond = full_precond
        self.debug = debug

        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        self.state_precond=dict()

        # All linear and conv layers are candidates of SNOpt preconditioning
        mods = [mod for mod in net.modules() if misc.get_class_name(mod) in ['Linear', 'Conv2d']]

        # We need to additionally hook modules that aren't belong to Neural ODEs
        # if we're doing full preconditioning
        self.mods_need_hook = [mod for mod in mods if mod not in net.ode_mods] if full_precond else []

        # Register modules that need SNOpt precond
        mods_need_precond = mods if full_precond else [mod for mod in mods if mod in net.ode_mods]
        for mod in mods_need_precond:
            mod_class = misc.get_class_name(mod)
            self.state_precond[mod]=dict()
            params = [mod.weight]
            if mod.bias is not None:
                params.append(mod.bias)
            d = {'params': params, 'mod': mod, 'layer_type': mod_class}
            if mod_class == 'Conv2d':
                # Adding gathering filter for convolution
                d['gathering_filter'] = misc.build_gathering_filter(mod)
            self.params.append(d)
            self.state_precond[mod]=d

        super(SNOpt, self).__init__(self.params, {})

    def remove_hook(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
        self._fwd_handles.clear()
        self._bwd_handles.clear()

        for ode in self.net.odes:
            ode.odefunc.remove_hook()


    def register_hook(self):
        for mod in self.mods_need_hook:
            self._fwd_handles.append(
                mod.register_forward_pre_hook(self._save_input)
            )
            self._bwd_handles.append(
                mod.register_backward_hook(self._save_grad_output)
            )

    def train_itr_setup(self):
        # tell odefunc to ignore or store statistics at this training iteration
        # this should be set for both odeint and adjoint ode-backprop
        for ode in self.net.odes:
            assert isinstance(ode.odefunc, ODEFuncBase)
            ode.odefunc.ctx.store_this_itr = self.store_this_itr

        # register hook only when we need it at this iteration
        if self.store_this_itr:
            self.register_hook()


    def step(self, update_stats=True, update_params=True):
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None

            mod = group['mod']
            if self._iteration_counter % self.update_freq == 0:
                self._compute_kfe(mod)

            # preconditioning
            self._precond_ra(weight, bias, mod)

            # clean up moving average for this iteration
            run_avg_x, run_avg_g = self._get_store_keys(mod, ['xxt', 'ggt'])
            run_avg_x.reset()
            run_avg_g.reset()

        self._iteration_counter += 1
        self.remove_hook()

    @property
    def store_this_itr(self):
        # if it's the iteration to update SNOpt then we need to store
        # forward/backward information at this iteration
        return self._iteration_counter % self.update_freq == 0


    def _save_to_state(self, mod, key, val):
        mod_state = self.state[mod]
        if key not in mod_state:
            mod_state[key] = misc.RunningAverageMeter()
        mod_state[key].update(val)


    def _save_input(self, mod, i):

        if self.debug: print('_save_input', mod, 'snopt')

        if not mod.training: return
        if self.store_this_itr:
            gathering_filter = self.state_precond[mod]['gathering_filter'] if isinstance(mod, torch.nn.Conv2d) else None
            xxt = misc.compute_xxt(mod, i[0], gathering_filter=gathering_filter)
            self._save_to_state(mod, 'xxt', xxt)

        # store batch size
        self.state[mod]['bs'] = i[0].size(0)


    def _save_grad_output(self, mod, grad_input, grad_output):

        assert self.full_precond

        if self.debug: print('_save_grad_output', mod, 'snopt')
        if self.store_this_itr:
            gy = grad_output[0] * grad_output[0].size(0)
            ggt = misc.compute_ggt(mod, gy)
            num_locations = misc.compute_nloc(mod, gy)

            self._save_to_state(mod, 'ggt', ggt)
            self._save_to_state(mod, 'num_locations', num_locations)


    def _precond_ra(self, weight, bias, mod):
        """Applies preconditioning.
           source: https://github.com/Thrandis/EKFAC-pytorch/blob/master/ekfac.py
        """

        state=self.state[mod]
        state_precond=self.state_precond[mod]

        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        m2 = state['m2']

        # ==== Prepare ====
        g = weight.grad.data
        s = g.shape
        bs, = self._get_store_keys(mod, ['bs'])
        if state_precond['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)

        # ==== Preconditioning ====
        g_kfe = torch.mm(torch.mm(kfe_gy.t(), g), kfe_x)
        m2.mul_(self.alpha).add_(g_kfe**2, alpha=(1. - self.alpha) * bs)
        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = torch.mm(torch.mm(kfe_gy, g_nat_kfe), kfe_x.t())
        if bias is not None:
            gb = g_nat[:, -1].contiguous().view(*bias.shape)
            bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        g_nat = g_nat.contiguous().view(*s)
        weight.grad.data = g_nat


    def _get_store_keys(self, mod, keys):
        store_state = None
        if mod in self.mods_need_hook:
            assert self.full_precond
            store_state = self.state[mod]
        else: # mod must belong to ode_mods
            net = self.net
            if len(net.odes) == 1:
                store_state = net.ode.odefunc.state[mod]
            else: # TODO(Guan) may be able to better handle this
                for ode in net.odes:
                    if mod in ode.odefunc.modules():
                        store_state = ode.odefunc.state[mod]
                        break

        assert store_state is not None, 'Cannot find the store_state of module {}!'.format(mod)
        return [store_state[k] for k in keys]


    def _compute_kfe(self, mod):

        state=self.state[mod]

        # ==== Query xxt and ggt ====
        keys = ['xxt', 'ggt', 'num_locations']
        run_avg_x, run_avg_g, run_avg_nl = self._get_store_keys(mod, keys)

        xxt, nfe = run_avg_x.avg, run_avg_x.N
        ggt, nbe = run_avg_g.avg, run_avg_g.N

        num_locations = run_avg_nl.sum

        # ==== compute kfe_x, kfe_gy, m2 ====
        Ex, state['kfe_x'] = torch.symeig(xxt, eigenvectors=True)
        Eg, state['kfe_gy'] = torch.symeig(ggt, eigenvectors=True)

        state['m2'] = Eg.unsqueeze(1) * Ex.unsqueeze(0) * num_locations

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
