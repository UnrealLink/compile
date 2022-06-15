import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import utils


class CompILE(nn.Module):
    """CompILE example implementation.

    Args:
        input_dim: Dictionary size of embeddings.
        hidden_dim: Number of hidden units.
        latent_dim: Dimensionality of latent variables (z).
        max_num_segments: Maximum number of segments to predict.
        temp_b: Gumbel softmax temperature for boundary variables (b).
        temp_z: Temperature for latents (z), only if latent_dist='concrete'.
        latent_dist: Whether to use Gaussian latents ('gaussian') or concrete /
            Gumbel softmax latents ('concrete').
    """
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim, max_num_segments,
                 temp_b=1., temp_z=1., latent_dist='gaussian', device='cpu'):
        super(CompILE, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_num_segments = max_num_segments
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
        self.lstm_cell = nn.LSTMCell(2*hidden_dim, hidden_dim)

        # LSTM output heads.
        self.head_z_1 = nn.Linear(hidden_dim, hidden_dim)

        if latent_dist == 'gaussian':
            self.head_z_2 = nn.Linear(hidden_dim, latent_dim * 2)
        elif latent_dist == 'concrete':
            self.head_z_2 = nn.Linear(hidden_dim, latent_dim)
        else:
            raise ValueError('Invalid argument for `latent_dist`.')

        self.head_b_1 = nn.Linear(hidden_dim, hidden_dim)  # Boundaries (b).
        self.head_b_2 = nn.Linear(hidden_dim, 1)

        # Decoder MLP.
        self.state_embedding_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.subpolicies = [
            nn.Linear(hidden_dim, action_dim).to(device) for i in range(latent_dim)
        ]

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
        if segment_id == self.max_num_segments - 1:
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

    def get_latents(self, encodings, probs_b):
        """Read out latents (z) form input encodings for a single segment."""
        readout_mask = probs_b[:, 1:, None]  # Offset readout by 1 to left.
        readout = (encodings[:, :-1] * readout_mask).sum(1)
        hidden = F.relu(self.head_z_1(readout))
        logits_z = self.head_z_2(hidden)

        # Gaussian latents.
        if self.latent_dist == 'gaussian':
            if self.training:
                mu, log_var = torch.split(logits_z, self.latent_dim, dim=1)
                sample_z = utils.gaussian_sample(mu, log_var)
            else:
                sample_z = logits_z[:, :self.latent_dim]

        # Concrete / Gumbel softmax latents.
        elif self.latent_dist == 'concrete':
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
        subpolicies = torch.cat([subpolicy(embed).unsqueeze(-1) for subpolicy in self.subpolicies], dim=-1)
        pred = (subpolicies * sample_z.unsqueeze(1).unsqueeze(1)).sum(dim=-1)
        return pred

    def get_next_masks(self, all_b_samples):
        """Get RNN hidden state masks for next segment."""
        if len(all_b_samples) < self.max_num_segments:
            # Product over cumsums (via log->sum->exp).
            log_cumsums = list(
                map(lambda x: utils.log_cumsum(x, dim=1), all_b_samples))
            mask = torch.exp(sum(log_cumsums))
            return mask
        else:
            return None

    def forward(self, inputs, lengths):

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
        for seg_id in range(self.max_num_segments):

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
                encodings, sample_b)
            all_z['logits'].append(logits_z)
            all_z['samples'].append(sample_z)

            # Get masks for next segment.
            mask = self.get_next_masks(all_b['samples'])
            all_masks.append(mask)

            # Decode current segment from latents (z).
            reconstructions = self.decode(sample_z, inputs[0])
            all_recs.append(reconstructions)

        return all_encs, all_recs, all_masks, all_b, all_z

    def save(self, path):
        checkpoint = {'model': self.state_dict()}
        for i, subpolicy in enumerate(self.subpolicies):
            checkpoint[f"subpolicy-{i}"] = subpolicy.state_dict()
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
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
            policy = F.softmax(self.decode(o_vector, state), dim=-1).cpu().numpy()
        return policy
