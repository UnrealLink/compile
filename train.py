import argparse
import os
import sys
import datetime
import json
import torch
import numpy as np
import gym

import utils
import modules

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=100,
                    help='Number of training iterations.')
parser.add_argument('--learning-rate', type=float, default=1e-2,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=6,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='concrete',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=512,
                    help='Mini-batch size (for averaging gradients).')

parser.add_argument('--num-symbols', type=int, default=5,
                    help='Number of distinct symbols in data generation.')
parser.add_argument('--num-segments', type=int, default=3,
                    help='Number of segments in data generation.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--log-interval', type=int, default=5,
                    help='Logging interval.')

parser.add_argument('--nb-rooms', type=int, default=5,
                    help='number of rooms in the room environment')
parser.add_argument('--max-steps', type=int, default=100,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--save-dir', type=str, default='',
                    help='directory where model and config are saved')
parser.add_argument('--random-seed', type=int, default=0,
                    help='Used to seed random number generators')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
run_ID = f"compile_{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
if args.save_dir == '':
    run_dir = f"runs/{run_ID}"
else:
    run_dir = args.save_dir
os.makedirs(run_dir)

with open(os.path.join(run_dir, "config.json"), "w") as f:
    f.write(json.dumps(vars(args), indent=4))

device = torch.device('cuda' if args.cuda else 'cpu')
np.random.seed(args.random_seed) # there were some issue with reproducibility
torch.manual_seed(args.random_seed)

sys.path.append('../nonparametric_options')
from env.toy_env import ToyEnv

rng_env = np.random.RandomState(args.random_seed)
rng_split = np.random.RandomState(args.random_seed)
env = ToyEnv(rng=rng_env, n_rooms=args.nb_rooms, max_steps=args.max_steps)
data = env.generate_expert_trajectories(n_traj=1000, noise_level=0,
                                        max_steps=args.max_steps, action_seed=args.random_seed)

state_dim = env.state_dim
action_dim = env.action_dim
max_steps = args.max_steps

device = torch.device('cuda' if args.cuda else 'cpu')
np.random.seed(args.random_seed) # there were some issue with reproducibility
torch.manual_seed(args.random_seed)

model = modules.CompILE(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    max_num_segments=args.num_segments,
    latent_dist=args.latent_dist,
    device=device).to(device)

parameter_list = list(model.parameters()) + sum([list(subpolicy.parameters()) for subpolicy in model.subpolicies], [])

optimizer = torch.optim.Adam(parameter_list, lr=args.learning_rate)

# Split data
data_states, data_actions, _ = data
data_states = data_states[:, :-1, :]
data_actions = np.argmax(data_actions, axis=-1)
train_test_split = np.random.permutation(len(data_states))
train_test_split_ratio = 0.1

train_data_states = data_states[train_test_split[int(len(data_states)*train_test_split_ratio):]]
train_action_states = data_actions[train_test_split[int(len(data_states)*train_test_split_ratio):]]

test_data_states = data_states[train_test_split[:int(len(data_states)*train_test_split_ratio)]]
test_action_states = data_actions[train_test_split[:int(len(data_states)*train_test_split_ratio)]]

test_lengths = torch.tensor([max_steps-1] * len(test_data_states)).to(device)
test_inputs = (torch.tensor(test_data_states).to(device), torch.tensor(test_action_states).to(device))

perm = utils.PermManager(len(train_data_states), args.batch_size)

beta_entropy = 1.
beta_entropy_ratio = 1

# Train model.
print('Training model...')
for step in range(args.iterations):
    rec = None
    batch_loss = 0
    batch_acc = 0
    optimizer.zero_grad()

    # Generate data.
    batch = perm.get_indices()
    batch_states, batch_actions = train_data_states[batch], train_action_states[batch]
    lengths = torch.tensor([max_steps-1] * args.batch_size).to(device)
    inputs = (torch.tensor(batch_states).to(device), torch.tensor(batch_actions).to(device))

    # Run forward pass.
    model.train()
    outputs = model.forward(inputs, lengths)
    loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args, beta_entropy=beta_entropy)

    loss.backward()
    optimizer.step()

    if step % args.log_interval == 0:
        # Run evaluation.
        model.eval()
        outputs = model.forward(test_inputs, test_lengths)
        acc, rec = utils.get_reconstruction_accuracy(test_inputs, outputs, args)

        # Accumulate metrics.
        batch_acc += acc.item()
        batch_loss += nll.item()
        print('step: {}, nll_train: {:.6f}, rec_acc_eval: {:.3f}'.format(
            step, batch_loss, batch_acc))
        # print('input sample: {}'.format(test_inputs[1][-1, :test_lengths[-1] - 1]))
        # print('reconstruction: {}'.format(rec[-1]))

    beta_entropy *= beta_entropy_ratio

model.eval()
room_score = [0.]*args.nb_rooms
obs = [0, 4, -1]
for option in range(model.latent_dim):
    policy = model.get_policy_from_observation(option, obs).reshape(-1)
    for room in range(args.nb_rooms):
        room_score[room] = max(room_score[room], policy[room+1])
with open('results.txt', 'a') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')
    f.write(str(np.mean(room_score)))
    f.write('\n')

# model.save(os.path.join(run_dir, 'checkpoint.pth'))

