import torch
import torch.nn.functional as F


#############################################################

class GRUCell(torch.nn.Module):
    def __init__(self, nobservation, nhidden):
        super(GRUCell, self).__init__()
        self.nhidden = nhidden
        self.i2r = torch.nn.Linear(nobservation + nhidden, nhidden)
        self.i2z = torch.nn.Linear(nobservation + nhidden, nhidden)
        self.x2n = torch.nn.Linear(nobservation, nhidden)
        self.h2n = torch.nn.Linear(nhidden, nhidden)


    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        r = torch.sigmoid(self.i2r(combined))
        z = torch.sigmoid(self.i2z(combined))

        n = self.x2n(x) + self.h2n(h) * r
        n = torch.tanh(n)
        h = (1-z) * n + z * h
        return h

class NaturalCubicSpline:
    """Calculates the natural cubic spline approximation to the batch of controls given. Also calculates its derivative.
       source: https://github.com/patrick-kidger/NeuralCDE/blob/master/controldiffeq/interpolate.py
    """

    def __init__(self, times, coeffs, **kwargs):
        """
        Arguments:
            times: As was passed as an argument to natural_cubic_spline_coeffs.
            coeffs: As returned by natural_cubic_spline_coeffs.
        """
        super(NaturalCubicSpline, self).__init__(**kwargs)

        a, b, two_c, three_d = coeffs

        self._times = times
        self._a = a
        self._b = b
        # as we're typically computing derivatives, we store the multiples of these coefficients that are more useful
        self._two_c = two_c
        self._three_d = three_d

    def _interpret_t(self, t):
        maxlen = self._b.size(-2) - 1
        index = (t > self._times).sum() - 1
        index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
        # will never access the last element of self._times; this is correct behaviour
        fractional_part = t - self._times[index]
        return fractional_part, index

    def evaluate(self, t):
        """Evaluates the natural cubic spline interpolation at a point t, which should be a scalar tensor."""
        fractional_part, index = self._interpret_t(t)
        inner = 0.5 * self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part / 3
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part

    def derivative(self, t):
        """Evaluates the derivative of the natural cubic spline at a point t, which should be a scalar tensor."""
        fractional_part, index = self._interpret_t(t)
        inner = self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part
        deriv = self._b[..., index, :] + inner * fractional_part
        return deriv

class TimeSeriesODENet(torch.nn.Module):
    """
       source: https://github.com/patrick-kidger/NeuralCDE/blob/master/experiments/models/other.py
    """
    def __init__(self, ode, hidden, input_dim, output_dim, use_intensity=False):
        super(TimeSeriesODENet,self).__init__()

        self.input_channels = input_dim
        self.hidden_channels = hidden
        self.output_channels = output_dim
        self.use_intensity = use_intensity

        gru_channels = input_dim if use_intensity else (input_dim - 1) // 2
        self.gru_cell = GRUCell(gru_channels, hidden)
        self.linear = torch.nn.Linear(hidden, output_dim)
        self.ode = ode


    def evolve(self, h, time_diff):
        t = torch.tensor([0, time_diff.item()], dtype=time_diff.dtype, device=time_diff.device)
        out = self.ode(h,t)
        return out[1]

    def _step(self, Xi, h, dt, half_num_channels):
        observation = Xi[:, 1: 1 + half_num_channels].max(dim=1).values > 0.5
        if observation.any():
            Xi_piece = Xi if self.use_intensity else Xi[:, 1 + half_num_channels:]
            Xi_piece = Xi_piece.clone()
            Xi_piece[:, 0] += dt
            new_h = self.gru_cell(Xi_piece, h)
            h = torch.where(observation.unsqueeze(1), new_h, h)
            dt += torch.where(observation, torch.tensor(0., dtype=Xi.dtype, device=Xi.device), Xi[:, 0])
        return h, dt

    def forward(self, x, z0=None):
        times, coeffs, final_index = x
        interp = NaturalCubicSpline(times, coeffs)
        X = torch.stack([interp.evaluate(t) for t in times], dim=-2)
        half_num_channels = (self.input_channels - 1) // 2

        # change cumulative intensity into intensity i.e. was an observation made or not, which is what is typically
        # used here
        X[:, 1:, 1:1 + half_num_channels] -= X[:, :-1, 1:1 + half_num_channels]

        # change times into delta-times
        X[:, 0, 0] -= times[0]
        X[:, 1:, 0] -= times[:-1]

        batch_dims = X.shape[:-2]

        if z0 is None:
            z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=X.dtype, device=X.device)

        X_unbound = X.unbind(dim=1)
        h, dt = self._step(X_unbound[0], z0, torch.zeros(*batch_dims, dtype=X.dtype, device=X.device),
                           half_num_channels)
        hs = [h]
        time_diffs = times[1:] - times[:-1]
        for time_diff, Xi in zip(time_diffs, X_unbound[1:]):
            h = self.evolve(h, time_diff)
            h, dt = self._step(Xi, h, dt, half_num_channels)
            hs.append(h)
        out = torch.stack(hs, dim=1)

        final_index_indices = final_index.unsqueeze(-1).expand(out.size(0), out.size(2)).unsqueeze(1)
        final_out = out.gather(dim=1, index=final_index_indices).squeeze(1)

        return self.linear(final_out)

    @property
    def ode_mods(self):
        return [mod for mod in self.ode.odefunc.modules()]

    @property
    def odes(self):
        return [self.ode]

#############################################################

class CNFODENet(torch.nn.Module):
    def __init__(self, n_ode, ode_builder):
        super(CNFODENet,self).__init__()
        self.n_ode = n_ode
        for i in range(n_ode):
            ode_name = 'ode{}'.format('' if i==0 else str(i+1))
            setattr(self, ode_name, ode_builder())

    def forward(self, x):
        x,   logp_diff_t1 = x
        x.requires_grad_(True)
        logp_diff_t1.requires_grad_(True)

        for ode in self.odes:
            # Pre-forward setup for divergence computation.
            ode.odefunc.reset_sample_e()

            out = ode((x, logp_diff_t1))

            z_t,  logp_diff_t  = out
            x, logp_diff_t1 = z_t[-1], logp_diff_t[-1]

        return [x, logp_diff_t1]

    @property
    def odes(self):
        return [getattr(self,'ode{}'.format('' if i==0 else str(i+1))) for i in range(self.n_ode)]

    @property
    def ode_mods(self):
        ode_mods = []
        for ode in self.odes:
            ode_mods.extend([mod for mod in ode.odefunc.modules()])
        return ode_mods

#############################################################

class ConvODENet(torch.nn.Module):
    def __init__(self, ode, hidden, input_dim):
        super(ConvODENet,self).__init__()

        self.conv1 = torch.nn.Conv2d(input_dim, hidden, 3, stride=1, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(hidden, hidden, 4, stride=2, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(hidden, hidden, 4, stride=2, padding=1, bias=False)
        self.pooling = torch.nn.AvgPool2d((6, 6))
        self.flatten = lambda x: x.reshape(x.shape[0],-1)
        self.fc1 = torch.nn.Linear(64, 10)
        self.ode = ode


    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(self.ode(x))
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x

    @property
    def ode_mods(self):
        return [mod for mod in self.ode.odefunc.modules()]

    @property
    def odes(self):
        return [self.ode]
