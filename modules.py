from os import stat
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from functools import reduce

import utils


class CompILE(nn.Module):
    """CompILE example implementation.

    Args:
        input_dim: Dictionary size of embeddings.
        hidden_dim: Number of hidden units.
        latent_dim: Dimensionality of latent variables (z).
        num_segments: Maximum number of segments to predict.
        temp_b: Gumbel softmax temperature for boundary variables (b).
        temp_z: Temperature for latents (z), only if latent_dist='concrete'.
        latent_dist: Whether to use Gaussian latents ('gaussian') or concrete /
            Gumbel softmax latents ('concrete').
    """
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim, num_segments,
                 temp_b=1., temp_z=1., latent_dist='gaussian', device='cpu'):
        super(CompILE, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_segments = num_segments
        self.temp_b = temp_b
        self.temp_z = temp_z
        self.latent_dist = latent_dist
        self.device = device
        self.K = latent_dim

        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lstm_cell = nn.LSTMCell(hidden_dim*2, hidden_dim)

        # LSTM output heads.
        # self.head_z_1 = nn.Linear(hidden_dim, hidden_dim)
        self.head_z_1 = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, hidden_dim + latent_dim))]
        )
        nn.init.xavier_normal_(self.head_z_1[-1])
        self.head_z_1_bias = nn.Parameter(torch.zeros(hidden_dim))

        if latent_dist == 'gaussian':
            self.head_z_2 = nn.Linear(hidden_dim, latent_dim * 2)
        elif latent_dist == 'concrete':
            self.head_z_2 = nn.ModuleList([nn.Linear(hidden_dim, 1).to(device) for _ in range(latent_dim)])
        else:
            raise ValueError('Invalid argument for `latent_dist`.')

        self.head_b_1 = nn.Linear(hidden_dim, hidden_dim)  # Boundaries (b).
        self.head_b_2 = nn.Linear(hidden_dim, 1)

        # Decoder MLP.
        self.state_embedding_decoder = nn.Sequential(
            # nn.Linear(state_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
        )
        self.subpolicies = nn.ModuleList(
            [nn.Linear(state_dim, action_dim).to(device) for _ in range(latent_dim)]
        )

        self.high_level_posterior = StickBreakingKumaraswamy(latent_dim, device)

    def embed_input(self, inputs):
        state_embedding = self.state_embedding(inputs[0])
        action_embedding = self.action_embedding(inputs[1])

        embedding = torch.cat([state_embedding, action_embedding], dim=-1)
        return embedding

    def masked_encode(self, inputs, mask):
        """Run masked RNN encoder on input sequence."""
        hidden = utils.get_lstm_initial_state(
            inputs.size(0), self.hidden_dim, device=inputs.device)
        outputs = []
        for step in range(inputs.size(1)):
            hidden = self.lstm_cell(inputs[:, step], hidden)
            hidden = (mask[:, step, None] * hidden[0],
                      mask[:, step, None] * hidden[1])  # Apply mask.
            outputs.append(hidden[0])
        return torch.stack(outputs, dim=1)

    def get_boundaries(self, encodings, segment_id, lengths):
        """Get boundaries (b) for a single segment in batch."""
        if segment_id == self.num_segments - 1:
            # Last boundary is always placed on last sequence element.
            logits_b = None
            sample_b = torch.zeros_like(encodings[:, :, 0]).scatter_(
                1, lengths.unsqueeze(1) - 1, 1)
        else:
            hidden = F.relu(self.head_b_1(encodings))
            logits_b = self.head_b_2(hidden).squeeze(-1)
            # Mask out first position with large neg. value.
            neg_inf = torch.ones(
                encodings.size(0), 1, device=encodings.device) * utils.NEG_INF
            # TODO(tkipf): Mask out padded positions with large neg. value.
            logits_b = torch.cat([neg_inf, logits_b[:, 1:]], dim=1)
            if self.training:
                sample_b = utils.gumbel_softmax_sample(
                    logits_b, temp=self.temp_b)
            else:
                sample_b_idx = torch.argmax(logits_b, dim=1)
                sample_b = utils.to_one_hot(sample_b_idx, logits_b.size(1))

        return logits_b, sample_b

    def get_latents(self, encodings, probs_b, eta):
        """Read out latents (z) form input encodings for a single segment."""
        readout_mask = probs_b[:, 1:, None]  # Offset readout by 1 to left.
        readout = (encodings[:, :-1] * readout_mask).sum(1)
        # hidden = F.relu(self.head_z_1(readout))
        readout_eta = torch.cat([readout, eta], dim=-1)
        hidden = F.relu(F.linear(readout_eta, reduce(lambda x,y: torch.cat((x,y), 1), self.head_z_1)) + self.head_z_1_bias)

        # Gaussian latents.
        if self.latent_dist == 'gaussian':
            logits_z = self.head_z_2(hidden)
            if self.training:
                mu, log_var = torch.split(logits_z, self.latent_dim, dim=1)
                sample_z = utils.gaussian_sample(mu, log_var)
            else:
                sample_z = logits_z[:, :self.latent_dim]

        # Concrete / Gumbel softmax latents.
        elif self.latent_dist == 'concrete':
            logits_z = torch.cat([layer(hidden) for layer in self.head_z_2], dim=-1)
            if self.training:
                sample_z = utils.gumbel_softmax_sample(
                    logits_z, temp=self.temp_z)
            else:
                sample_z_idx = torch.argmax(logits_z, dim=1)
                sample_z = utils.to_one_hot(sample_z_idx, logits_z.size(1))
        else:
            raise ValueError('Invalid argument for `latent_dist`.')

        return logits_z, sample_z

    def decode(self, sample_z, states):
        """Decode single time step from latents and repeat over full seq."""
        embed = self.state_embedding_decoder(states)
        subpolicies = torch.cat([F.softmax(subpolicy(embed), dim=-1).unsqueeze(-1) for subpolicy in self.subpolicies], dim=-1)
        pred = (subpolicies * sample_z.unsqueeze(1).unsqueeze(1)).sum(dim=-1)
        return pred

    def get_next_masks(self, all_b_samples):
        """Get RNN hidden state masks for next segment."""
        if len(all_b_samples) < self.num_segments:
            # Product over cumsums (via log->sum->exp).
            log_cumsums = list(
                map(lambda x: utils.log_cumsum(x, dim=1), all_b_samples))
            mask = torch.exp(sum(log_cumsums))
            return mask
        else:
            return None

    def forward(self, inputs, lengths):

        # Sample eta
        eta, pre_sb_eta = self.high_level_posterior.sample_mean(return_pre_sb=True)
        eta_repeated = eta.repeat(inputs[0].size(0), 1)

        # Embed inputs.
        embeddings = self.embed_input(inputs)

        # Create initial mask.
        mask = torch.ones(
            inputs[0].size(0), inputs[0].size(1), device=inputs[0].device)

        all_b = {'logits': [], 'samples': []}
        all_z = {'logits': [], 'samples': []}
        all_encs = []
        all_recs = []
        all_masks = []
        for seg_id in range(self.num_segments):

            # Get masked LSTM encodings of inputs.
            encodings = self.masked_encode(embeddings, mask)
            all_encs.append(encodings)

            # Get boundaries (b) for current segment.
            logits_b, sample_b = self.get_boundaries(
                encodings, seg_id, lengths)
            all_b['logits'].append(logits_b)
            all_b['samples'].append(sample_b)

            # Get latents (z) for current segment.
            logits_z, sample_z = self.get_latents(
                encodings, sample_b, eta_repeated)
            all_z['logits'].append(logits_z)
            all_z['samples'].append(sample_z)

            # Get masks for next segment.
            mask = self.get_next_masks(all_b['samples'])
            all_masks.append(mask)

            # Decode current segment from latents (z).
            reconstructions = self.decode(sample_z, inputs[0])
            all_recs.append(reconstructions)

        return all_encs, all_recs, all_masks, all_b, all_z, eta, pre_sb_eta

    def save(self, path):
        checkpoint = {'model': self.state_dict(), 'latent_dim': self.latent_dim}
        for i, subpolicy in enumerate(self.subpolicies):
            checkpoint[f"subpolicy-{i}"] = subpolicy.state_dict()
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        for _ in range(checkpoint['latent_dim']-self.latent_dim):
            self.add_option(None)
        self.load_state_dict(checkpoint['model'])
        for i, subpolicy in enumerate(self.subpolicies):
            subpolicy.load_state_dict(checkpoint[f"subpolicy-{i}"])

    def play_from_observation(self, option, obs):
        with torch.no_grad():
            state = torch.tensor(obs).unsqueeze(0).unsqueeze(0).to(self.device).float()
            o_vector = torch.zeros(1, self.latent_dim).to(self.device).float()
            o_vector[0, option] = 1
            policy = self.decode(o_vector, state).cpu().numpy()
            termination = 0.
        return np.argmax(policy), termination

    def get_policy_from_observation(self, option, obs):
        with torch.no_grad():
            state = torch.tensor(obs).unsqueeze(0).unsqueeze(0).to(self.device).float()
            o_vector = torch.zeros(1, self.latent_dim).to(self.device).float()
            o_vector[0, option] = 1
            policy = self.decode(o_vector, state).cpu().numpy()
        return policy

    def add_option(self, optimizer):
        self.subpolicies.append(nn.Linear(self.state_dim, self.action_dim).to(self.device))
        self.head_z_1.append(nn.Parameter(torch.empty(self.hidden_dim, 1, device=self.device)))
        nn.init.xavier_normal_(self.head_z_1[-1])
        self.head_z_2.append(nn.Linear(self.hidden_dim, 1).to(self.device))
        self.high_level_posterior.add_option(optimizer)
        self.latent_dim += 1
        self.K = self.latent_dim
        if optimizer is not None:
            optimizer.add_param_group({"params" : self.subpolicies[-1].parameters()})
            optimizer.add_param_group({"params" : self.head_z_2[-1].parameters()})

    def evaluate_score(self, states, actions):
        with torch.no_grad():
            o_vector = torch.zeros(1, self.latent_dim).to(self.device).float()
            o_vector[0, 0] = 1
            policy = self.decode(o_vector, states)
            policy = policy.view(-1, policy.shape[-1]).cpu().numpy()
            max_probs = np.take_along_axis(policy, actions.view((-1, 1)).cpu().numpy(), 1).reshape(-1)
            for option in range(1, self.latent_dim):
                o_vector = torch.zeros(1, self.latent_dim).to(self.device).float()
                o_vector[0, option] = 1
                policy = self.decode(o_vector, states)
                policy = policy.view(-1, policy.shape[-1]).cpu().numpy()
                prob = np.take_along_axis(policy, actions.view((-1, 1)).cpu().numpy(), 1).reshape(-1)
                max_probs = np.maximum(max_probs, prob)
        return np.mean(max_probs)


class StickBreakingKumaraswamy(nn.Module):
    """
    Contains parameters for K independent Kumaraswamy distributions and allows to sample each of them through the
    reparameterization trick. The stick breaking procedure is then applied.
    """
    def __init__(self, K, device):
        super(StickBreakingKumaraswamy, self).__init__()
        self.K = K
        self.device = device
        self.log_kuma_params = nn.ParameterList([nn.Parameter(torch.randn([K, 2]))])
        log_alpha_fixed = False
        if log_alpha_fixed:
            self.log_alpha = torch.tensor(2).float().to(device)
        else:
            self.log_alpha = nn.Parameter(torch.randn([1]))
        self.soft_plus = nn.Softplus()

    def add_option(self, optimizer):
        self.log_kuma_params.append(nn.Parameter(torch.randn(1, 2, device=self.device)))
        self.K += 1
        if optimizer is not None:
            optimizer.add_param_group({"params" : self.log_kuma_params[-1]})

    def compute_kl(self, k=None, pre_sb=None, eps=10e-6):
        # returns an approximation of the KL between the product of Kumaraswamys and a product of Betas(1, alpha)
        # uses only the first k distributions and ignores the rest

        calculate_with_taylor_expansion = False
        log_kuma_params = reduce(lambda x,y: torch.cat((x,y), 0), self.log_kuma_params)
        kuma_params = torch.exp(log_kuma_params)
        alpha_param = torch.exp(self.log_alpha)
        if pre_sb is None:
            _, pre_sb = self.sample(return_pre_sb=True)
        if k is None:
            k = self.K

        if not calculate_with_taylor_expansion:
            # calc by samplings
            clamped_pre_sb = (pre_sb[:k]-0.5)*(1-2*eps) + 0.5
            beta_log_pdf = torch.distributions.Beta(1., alpha_param).log_prob(clamped_pre_sb)

            kuma_log_pdf = torch.sum(log_kuma_params[:k], axis=1) +\
                           (kuma_params[:k, 0] - 1.) * utils.stable_log(clamped_pre_sb, eps) +\
                           (kuma_params[:k, 1] - 1.) * utils.stable_log(1. - torch.pow(clamped_pre_sb, kuma_params[:k, 0]), eps)
            return torch.sum(kuma_log_pdf - beta_log_pdf)

        else:
            # calc taylor expansion
            beta =  alpha_param
            alpha = torch.tensor(1).float()

            kl = 1. /(1. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(1./kuma_params[:k, 0], kuma_params[:k, 1])
            kl += 1. /(2. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(2./kuma_params[:k, 0], kuma_params[:k, 1])
            kl += 1. / (3. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(3. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (4. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(4. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (5. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(5. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (6. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(6. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (7. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(7. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (8. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(8. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (9. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(9. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (10. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(10. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl *= (kuma_params[:k,1]) * (beta - 1)

            psi_b_taylor_approx = utils.stable_log(kuma_params[:k, 1], eps) - 1. / (2 * kuma_params[:k, 1]) - 1. / (12 * torch.pow(kuma_params[:k, 1],2))
            kl += (kuma_params[:k, 0] - alpha) / kuma_params[:k, 0] * (-0.57721 - psi_b_taylor_approx - 1/kuma_params[:k, 1])
            kl += log_kuma_params[:k, 0] * log_kuma_params[:k, 1] + utils.stable_log(self.beta_fn(alpha, beta), eps)


            kl += -(kuma_params[:k, 1] - 1) / kuma_params[:k, 1]

            return torch.sum(kl)

    def sample(self, return_pre_sb=False):
        log_kuma_params = reduce(lambda x,y: torch.cat((x,y), 0), self.log_kuma_params)
        kuma_params = torch.exp(log_kuma_params)
        u = torch.rand(self.K).to(self.device)
        pre_sb = torch.pow(1. - torch.pow(1. - u, 1. / kuma_params[:, 1]), 1. / kuma_params[:, 0])
        if return_pre_sb:
            return utils.sb(pre_sb, self.device), pre_sb
        else:
            return utils.sb(pre_sb, self.device)

    def sample_mean(self, return_pre_sb=False, nb_samples=20):
        log_kuma_params = reduce(lambda x,y: torch.cat((x,y), 0), self.log_kuma_params)
        kuma_params = torch.exp(log_kuma_params)
        u = torch.rand((nb_samples, self.K)).to(self.device)
        pre_sb = torch.pow(1. - torch.pow(1. - u, 1. / kuma_params[:, 1]), 1. / kuma_params[:, 0])
        post_sb = utils.sb(pre_sb, self.device)
        pre_sb = torch.mean(pre_sb, dim=0)
        post_sb = torch.mean(post_sb, dim=0)
        if return_pre_sb:
            return post_sb, pre_sb
        else:
            return post_sb

    def beta_fn(self, a, b):
        return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b))