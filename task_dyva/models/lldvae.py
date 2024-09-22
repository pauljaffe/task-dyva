"""Implementation of the LLDVAE model."""
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist
from torch.nn.parameter import Parameter

from ..utils import Constants, _init_dynamics_mats


class _EncoderHZ2W(nn.Module):
    # A component of the encoder/recognition model: maps information from the
    # current/future timesteps (summarized in h) and latent state z to
    # the innovation/noise variable w.
    def __init__(self, params):
        super(_EncoderHZ2W, self).__init__()
        self.latent_dim = params["model_params"]["latent_dim"]
        self.w_dim = params["model_params"]["w_dim"]
        self.encoder_combo_hidden_dim = params["model_params"][
            "encoder_combo_hidden_dim"
        ]
        self.encoder_rnn_hidden_dim = params["model_params"][
            "encoder_rnn_hidden_dim"
        ]

        self.hz2w_shared = nn.Sequential(
            nn.Linear(
                self.encoder_rnn_hidden_dim + self.latent_dim,
                self.encoder_combo_hidden_dim,
            ),
            nn.ReLU(),
        )

        self.hz2w_means = nn.Linear(self.encoder_combo_hidden_dim, self.w_dim)
        self.hz2w_vars = nn.Linear(self.encoder_combo_hidden_dim, self.w_dim)

    def forward(self, h, z):
        inputs_cat = torch.cat((z, h), -1)
        w_pre = self.hz2w_shared(inputs_cat)
        w_means = self.hz2w_means(w_pre)
        w_vars = self.hz2w_vars(w_pre)
        w_vars = F.softmax(w_vars, dim=-1) * w_vars.size(-1) + Constants.eta
        return w_means, w_vars


class _EncoderXU2H(nn.Module):
    # A component of the encoder/recognition model: maps the observed data
    # (stimuli + responses) to the internal state variable h using a RNN.
    # During inference, observed data from the current timestep and future
    # contribute to the evolution of z. This information is contained in h.
    def __init__(self, params):
        super(_EncoderXU2H, self).__init__()
        self.input_dim = params["data_params"]["input_dim"]
        self.u_dim = params["data_params"]["u_dim"]
        self.w_dim = params["model_params"]["w_dim"]
        self.encoder_rnn_hidden_dim = params["model_params"][
            "encoder_rnn_hidden_dim"
        ]
        self.encoder_mlp_hidden_dim = params["model_params"][
            "encoder_mlp_hidden_dim"
        ]
        self.encoder_rnn_dropout = params["model_params"].get(
            "encoder_rnn_dropout", 0.0
        )
        self.encoder_rnn_n_layers = params["model_params"].get(
            "encoder_rnn_n_layers", 1
        )
        self.encoder_rnn_bidir = params["model_params"].get(
            "encoder_rnn_bidir", False
        )
        self.encoder_rnn_input_dim = params["model_params"][
            "encoder_rnn_input_dim"
        ]

        self.encoder_mlp = nn.Sequential(
            nn.Linear(
                self.input_dim + self.u_dim, self.encoder_mlp_hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(self.encoder_mlp_hidden_dim, self.encoder_rnn_input_dim),
        )

        encoder_rnn = nn.LSTM(
            self.encoder_rnn_input_dim,
            self.encoder_rnn_hidden_dim,
            num_layers=self.encoder_rnn_n_layers,
            dropout=self.encoder_rnn_dropout,
            bidirectional=self.encoder_rnn_bidir,
        )

        for name, param in encoder_rnn.named_parameters():
            if "weight" in name:
                torch.nn.init.xavier_uniform_(param)
        self.encoder_rnn = encoder_rnn

    def forward(self, xu):
        xu_h = self.encoder_mlp(xu)
        h, _ = self.encoder_rnn(torch.flip(xu_h, [0]))
        h = torch.flip(h, [0])
        return h


class _Decoder(nn.Module):
    # Maps the latent state variable z to the parameters of the
    # observation model, from which x (the responses) are sampled.
    def __init__(
        self, latent_dim, decoder_hidden_dim, scale_param, input_dim=4
    ):
        super(_Decoder, self).__init__()
        self.scale_param = scale_param
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        mean_out = self.decoder(z).clamp(Constants.eta, 1 - Constants.eta)
        return mean_out, torch.tensor(self.scale_param).to(z.device)


class LLDVAE(nn.Module):
    """Implements a locally linear dynamical variational autoencoder (LLDVAE).

    Implementation note: the PyTorch distribution method rsample
    automatically reparameterizes the variables so that the sampling
    process is differentiable (i.e., there is no need to explicitly
    implement a separate reparameterization method as is commonly done).

    Notes on the variables
    ----------------------
    The main variables used in the implementation are listed below.
    For each variable/tensor, the three dimensions correspond to:
    (n_timesteps x n_samples x n_features).

    x: The task responses (n_features = 4).
    u: The task stimuli (n_features = 10).
    z: The latent state variables (n_features = latent_dim).
    w: The noise term in the latent state update equation
       (n_features = w_dim).
    h: The internal state variable summarizing current and future information
       (n_features = encoder_rnn_hidden_dim).

    Args
    ----
    params (dict): Parameters for the model, data, and training algorithm.
        See the default config file (config/model_config.yaml).

    """

    def __init__(self, params, device):
        super(LLDVAE, self).__init__()
        self.device = device
        self.input_dim = params["data_params"]["input_dim"]
        self.u_dim = params["data_params"]["u_dim"]
        self.latent_dim = params["model_params"]["latent_dim"]
        self.init_rnn_dim = params["model_params"]["init_rnn_dim"]
        self.init_hidden_dim = params["model_params"]["init_hidden_dim"]
        self.w_dim = params["model_params"]["w_dim"]
        self.dyn_mat_mult = params["model_params"].get(
            "dynamics_matrix_mult", 0.99
        )
        self.dyn_init = params["model_params"].get(
            "dynamics_init_method", "custom_rotation"
        )
        self.encoder_xu2h = _EncoderXU2H(params)
        self.encoder_hz2w = _EncoderHZ2W(params)
        self.decoder_hidden_dim = params["model_params"]["decoder_hidden_dim"]
        decoder_scale_param = params["model_params"].get(
            "likelihood_scale_param", 0.75
        )
        self.decoder = _Decoder(
            self.latent_dim, self.decoder_hidden_dim, decoder_scale_param
        )
        # trans_dim = number of transition matrices in state update equation
        self.trans_dim = params["model_params"]["trans_dim"]
        self.batch_size = params["training_params"]["batch_size"]
        self.rand_seed = params["training_params"]["rand_seed"]
        self.pw = getattr(
            dist, params["model_params"].get("prior_dist", "Normal")
        )
        self.px_w = getattr(
            dist, params["model_params"].get("likelihood_dist", "Normal")
        )
        self.qw_x = getattr(
            dist, params["model_params"].get("posterior_dist", "Normal")
        )

        prior_grad = {
            "requires_grad": params["training_params"]["learn_prior"]
        }
        self._pw_params = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(1, 1, self.latent_dim, device=self.device),
                    requires_grad=False,
                ),
                nn.Parameter(
                    torch.ones(1, 1, self.latent_dim, device=self.device),
                    **prior_grad
                ),
            ]
        )

        # Latent state initialization
        self.init_z0 = nn.Sequential(
            nn.Linear(self.w_dim, self.init_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.init_hidden_dim, self.latent_dim),
        )

        # Build the transition network: self.trans_net determines
        # the weights on the transition matrices in the state update equation.
        if self.trans_dim == 0:
            num_mats = 1
        else:
            num_mats = self.trans_dim
        self.trans_net = nn.Sequential(
            nn.Linear((self.latent_dim + self.u_dim), num_mats),
            nn.Softmax(dim=1),
        )

        # Innovation/noise initialization
        w0_means = torch.empty(self.w_dim, device=self.device)
        self.w0_means = Parameter(nn.init.normal_(w0_means))
        w0_vars = torch.empty(self.w_dim, device=self.device)
        self.w0_vars = Parameter(nn.init.normal_(w0_vars))

        # Initialize matrices for the latent dynamics
        B_mats = torch.empty(
            num_mats, self.latent_dim, self.u_dim, device=self.device
        )
        self.B_mats = Parameter(nn.init.normal_(B_mats))
        self.A_mats = Parameter(
            self.dyn_mat_mult
            * _init_dynamics_mats(
                self.latent_dim, num_mats, self.rand_seed, self.dyn_init
            )
        )
        self.C_mats = Parameter(
            self.dyn_mat_mult
            * _init_dynamics_mats(
                self.latent_dim, num_mats, self.rand_seed, self.dyn_init
            )
        )

    @property
    def pw_params(self):
        # Return the parameters of the prior distribution.
        # pw_params[0] = prior means; pw_params[1] = prior variances.
        pw_vars = F.softmax(self._pw_params[1], dim=2) * self._pw_params[
            1
        ].size(-1)
        return self._pw_params[0], pw_vars

    def forward(self, xu, generate_mode=False, clamp=False, z0_supplied=None):
        """Compute a forward pass of the model.

        Args
        ----
        xu (PyTorch tensor): The model input data consisting of
            the task responses (x) and task stimuli (u), concatenated along
            the last dimension. This tensor has dimensions
            (n_timesteps x n_samples x 14).
        generate_mode (Boolean, optional): If True, the model is run
            in generative mode: u is mapped to x. If False, the model
            is run in inference mode: x and u are mapped back to x.
        clamp (Boolean, optional): Only affects behavior if
            generate_mode == True. If True, the noise variable w is
            set to zero. If False, w is sampled from the prior.
        z0_supplied (PyTorch tensor, optional): Pass in user-defined
            initial states to initialize the model. Used when
            finding fixed points.

        Returns
        -------
        x_in (PyTorch tensor): The task response component of the model inputs.
        px_w (PyTorch distribution object): The model likelihood.
        w (PyTorch tensor): The noise variables in the latent state
            update equation.
        z (PyTorch tensor): The latent state variables.
        w_means (PyTorch tensor): The means of the prior distribution.
        w_vars (PyTorch tensor): The variances of the prior distibution.
        """

        x_in = xu[:, :, : self.input_dim]
        u = xu[:, :, self.input_dim :]
        setup_vars = self._setup_propagate(xu, generate_mode, clamp)
        if z0_supplied is None:
            w, z, w_means, w_vars = self._propagate(
                *setup_vars, u, generate_mode
            )
        else:
            w, z, w_means, w_vars = self._propagate(
                z0_supplied, *setup_vars[1:], u, generate_mode
            )
        px_w = self.px_w(*self.decoder(z))
        return x_in, px_w, w, z, w_means, w_vars

    def _init_w_z(self, batch_size):
        w0_means = self.w0_means.unsqueeze(0).expand(1, batch_size, self.w_dim)
        w0_vars = (
            F.softmax(
                self.w0_vars.unsqueeze(0).expand(1, batch_size, self.w_dim),
                dim=-1,
            )
            * self.w0_vars.size(-1)
            + Constants.eta
        )
        w0_sample = self.qw_x(w0_means, w0_vars).rsample()
        z0 = self.init_z0(w0_sample)
        return z0, w0_sample, w0_means, w0_vars

    def _gen_trans_params(self, z, u):
        z_u_cat = torch.cat((z, u), -1)
        alpha_t = self.trans_net(z_u_cat)
        return alpha_t

    def _batch_wsum(self, M, alphas):
        return (alphas[:, :, None, None] * M[None, :, :, :]).sum(dim=1)

    def _setup_propagate(self, xu, generate_mode, clamp):
        seq_length = xu.shape[0]
        n_data = xu.shape[1]
        z0, w0, all_w_means, all_w_vars = self._init_w_z(n_data)
        if generate_mode:
            # w0 will have different dims than when generate_mode=False
            w0 = (
                self.pw(*self.pw_params)
                .expand((seq_length, n_data, self.w_dim))
                .rsample()
            )
            if clamp:
                w0 = torch.zeros(
                    seq_length, n_data, self.w_dim, device=self.device
                )
            h = torch.tensor([0], device=self.device)
        else:
            h = self.encoder_xu2h(xu)
        return z0, w0, h, all_w_means, all_w_vars

    def _update_state(self, z_t, u_t, w_t):
        if self.trans_dim == 0:
            n_data = z_t.shape[0]
            alpha_t = torch.tensor([0], device=self.device)
            A_t = self.A_mats[0].unsqueeze(0).repeat(n_data, 1, 1)
            B_t = self.B_mats[0].unsqueeze(0).repeat(n_data, 1, 1)
            C_t = self.C_mats[0].unsqueeze(0).repeat(n_data, 1, 1)
        else:
            alpha_t = self._gen_trans_params(z_t, u_t)
            A_t = self._batch_wsum(self.A_mats, alpha_t)
            B_t = self._batch_wsum(self.B_mats, alpha_t)
            C_t = self._batch_wsum(self.C_mats, alpha_t)

        z_t_next = (
            torch.bmm(A_t, z_t.unsqueeze(2))
            + torch.bmm(B_t, u_t.unsqueeze(2))
            + torch.bmm(C_t, w_t.unsqueeze(2))
        )
        return z_t_next.squeeze().unsqueeze(0)

    def _propagate(self, z, w, h, w_means, w_vars, u, generate_mode):
        time_steps = u.shape[0]
        for t in range(time_steps - 1):
            z_t = z[t, :, :]
            u_t = u[t, :, :]

            if generate_mode:
                w_t = w[t, :, :]
            else:
                h_t = h[t, :, :]
                w_means_t, w_vars_t = self.encoder_hz2w(h_t, z_t)
                w_t = self.qw_x(w_means_t, w_vars_t).rsample()
                w_means = torch.cat((w_means, w_means_t.unsqueeze(0)), 0)
                w_vars = torch.cat((w_vars, w_vars_t.unsqueeze(0)), 0)
                w = torch.cat((w, w_t.unsqueeze(0)), 0)

            z_t_next = self._update_state(z_t, u_t, w_t)
            z = torch.cat((z, z_t_next), 0)
        return w, z, w_means, w_vars
