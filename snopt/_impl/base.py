import torch
from . import misc
import easydict


class ODEFuncBase(torch.nn.Module):
    def __init__(self, opt, debug=False):
        super(ODEFuncBase, self).__init__()
        self.opt = opt
        self.nfe = 0
        self.debug = debug

        self.ctx=easydict.EasyDict()
        self.ctx.save_input_flag='Stop'
        self.ctx.store_this_itr=False
        self.gy_scale = opt.snopt_step_size

        self._fwd_handles = []
        self._bwd_handles = []

    def initialize_snopt_state(self):
        self.state = {}
        self.remove_hook()
        for mod in self.modules():
            if misc.get_class_name(mod) in ['Linear', 'Conv2d']:
                self._fwd_handles.append(
                    mod.register_forward_pre_hook(self._save_input)
                )
                self._bwd_handles.append(
                    mod.register_backward_hook(self._save_grad_output)
                )
                self.state[mod]={
                    'gathering_filter': misc.build_gathering_filter(mod)
                }


    def remove_hook(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
        self._fwd_handles.clear()
        self._bwd_handles.clear()


    def _save_to_state(self, mod, input, key, val_generator, flag='Save', **kwargs):

        assert flag == 'Save', 'Unsupported flag {} for saving in ODEFuncHookWrapper'.format(flag)

        state = self.state[mod]

        val = val_generator(mod, input, **kwargs)
        if key not in state:
            state[key] = misc.RunningAverageMeter()
        state[key].update(val)


    def _save_grad_output(self, mod, grad_input, grad_output):
        if self.debug: print('_save_grad_output', mod, 'flag:', self.ctx.save_input_flag)


        flag = self.ctx.save_input_flag
        store_this_itr = self.ctx.store_this_itr

        # skip backward hoook if
        # (1) we're not in training mode, or
        # (2) it's disable (e.g. not the SNOpt update iteration), or
        # (3) we're asked by the ode solver to stop saving input
        if not mod.training or not store_this_itr or flag == 'Stop': return

        gy = grad_output[0] * grad_output[0].size(0) * self.gy_scale
        self._save_to_state(mod, gy, 'ggt', misc.compute_ggt)
        self._save_to_state(mod, gy, 'num_locations', misc.compute_nloc)


    def _save_input(self, mod, i):
        if self.debug: print('_save_input', mod, 'flag:', self.ctx.save_input_flag)

        flag = self.ctx.save_input_flag
        store_this_itr = self.ctx.store_this_itr

        # skip forward hoook if
        # (1) we're not in training mode, or
        # (2) it's disable (e.g. not the SNOpt update iteration), or
        # (3) we're asked by the ode solver to stop saving input
        if not mod.training or not store_this_itr or flag == 'Stop': return

        x = i[0]
        gathering_filter = self.state[mod]['gathering_filter']
        self._save_to_state(mod, x, 'xxt', misc.compute_xxt, flag=flag, gathering_filter=gathering_filter)
        self.state[mod]['bs'] = x.size(0)


    def F(self, t, x):
        raise NotImplementedError


    def forward(self, t, x):
        return self.F(t, x)

class CNFFuncBase(ODEFuncBase):
    def __init__(self, opt):
        super(CNFFuncBase, self).__init__(opt)
        self.divergence_type = opt.divergence_type

    def F(self, t, z):
        raise NotImplementedError

    def reset_sample_e(self, divergence_type=None):
        self._e = None # reset _e
        if divergence_type: self.divergence_type = divergence_type

    def forward(self, t, states):
        self.nfe += 1

        z = states[0]
        logp_z = states[1]
        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            dz_dt = self.F(t, z)

            # Special handle for snopt: disable storing grad when
            # we have backward hook (TODO better handle it)
            if self.opt.optimizer == 'SNOpt':
                orig_flag = self.ctx.save_input_flag
                self.ctx.save_input_flag='Stop'

            # Compute divergence
            if self.divergence_type == 'exact':
                divergence = misc.divergence_exact(dz_dt, z)
            else:
                sample_e_fn = {
                    'rademacher': misc.sample_rademacher_like,
                    'gaussian':   misc.sample_gaussian_like,
                }.get(self.divergence_type)
                self._e = sample_e_fn(z) if self._e is None else self._e
                divergence = misc.divergence_approx(dz_dt, z, e=self._e)

            # Special handle for snopt: restore flag back to
            # original one (TODO better handle it)
            if self.opt.optimizer == 'SNOpt':
                self.ctx.save_input_flag=orig_flag

            dlogp_z_dt = - divergence.view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)
