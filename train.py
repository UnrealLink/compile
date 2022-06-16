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
parser.add_argument('--batch-size', type=int, default=128,
                    help='Mini-batch size (for averaging gradients).')

parser.add_argument('--num-symbols', type=int, default=5,
                    help='Number of distinct symbols in data generation.')
parser.add_argument('--num-segments', type=int, default=3,
                    help='Number of segments in data generation.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--log-interval', type=int, default=5,
                    help='Logging interval.')

parser.add_argument('--demo-file', type=str, default='',
                    help='path to the expert trajectories file')
parser.add_argument('--max-steps', type=int, default=100,
                    help='maximum number of steps in an expert trajectory')
parser.add_argument('--atari-env-name', type=str, default='',
                    help='name of the atari env')
parser.add_argument('--save-dir', type=str, default='',
                    help='directory where model and config are saved')
parser.add_argument('--random-seed', type=int, default=0,
                    help='Used to seed random number generators')
parser.add_argument('--add-option-interval', type=int, default=10,
                    help='Interval between checks if a new option needs to be added.')
parser.add_argument('--fixed-options', action="store_true", default=False,
                    help='prevent any option from being added')
parser.add_argument('--results-file', type=str, default='results.txt',
                    help='file where results are saved')
parser.add_argument('--beta-entropy', type=float, default=0.,
                    help='Number of training iterations.')
parser.add_argument('--beta-entropy-ratio', type=float, default=0.99,
                    help='Learning rate.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
run_ID = f"compile_{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
if args.save_dir == '':
    run_dir = f"runs/{run_ID}"
else:
    run_dir = args.save_dir
os.makedirs(run_dir, exist_ok=True)

with open(os.path.join(run_dir, "config.json"), "w") as f:
    f.write(json.dumps(vars(args), indent=4))

data_path = args.demo_file
if args.atari_env_name == '':
    env_name = args.demo_file.split('/')[-2]
    env_name = env_name[0].upper() + env_name[1:]
else:
    env_name = args.atari_env_name
env = gym.make(f'{env_name}NoFrameskip-v4')
state_dim = 1024
action_dim = env.action_space.n
max_steps = args.max_steps

device = torch.device('cuda' if args.cuda else 'cpu')
np.random.seed(args.random_seed) # there were some issue with reproducibility
torch.manual_seed(args.random_seed)

model = modules.CompILE(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    num_segments=args.num_segments,
    latent_dist=args.latent_dist,
    device=device).to(device)

parameter_list = list(model.parameters())

optimizer = torch.optim.Adam(parameter_list, lr=args.learning_rate)

# Load data
data = np.load(data_path, allow_pickle=True)
data_states = np.zeros([len(data), max_steps, state_dim], dtype='float32')
data_actions = np.zeros([len(data), max_steps], dtype='long')
for i, trajectory in enumerate(data):
    for j in range(max_steps):
        data_states[i, j] = np.unpackbits(trajectory[j][0])
        data_actions[i, j] = int(trajectory[j][1])
train_test_split = np.random.permutation(len(data))
train_test_split_ratio = 0.01

train_states = data_states[train_test_split[int(len(data_states)*train_test_split_ratio):]]
train_action = data_actions[train_test_split[int(len(data_states)*train_test_split_ratio):]]

test_states = data_states[train_test_split[:int(len(data_states)*train_test_split_ratio)]]
test_actions = data_actions[train_test_split[:int(len(data_states)*train_test_split_ratio)]]

test_lengths = torch.tensor([max_steps-1] * len(test_states)).to(device)
test_inputs = (torch.tensor(test_states).to(device), torch.tensor(test_actions).to(device))

perm = utils.PermManager(len(train_states), args.batch_size)


beta_entropy = args.beta_entropy
beta_entropy_ratio = args.beta_entropy_ratio

# Train model.
print('Training model...')
step = 0
rec = None
batch_loss = 0
batch_acc = 0
while perm.epoch < args.iterations:
    optimizer.zero_grad()

    # Generate data.
    batch = perm.get_indices()
    batch_states, batch_actions = train_states[batch], train_action[batch]
    lengths = torch.tensor([max_steps-1] * args.batch_size).to(device)
    inputs = (torch.tensor(batch_states).to(device), torch.tensor(batch_actions).to(device))

    # Run forward pass.
    model.train()
    outputs = model.forward(inputs, lengths)
    loss, nll, kl_z, kl_b, kl_eta = utils.get_losses(inputs, outputs, model, beta_entropy=beta_entropy)

    loss.backward()
    optimizer.step()

    if perm.epoch % args.add_option_interval == 0 and not(args.fixed_options):
        all_encs, all_recs, all_masks, all_b, all_z, eta, pre_sb_eta = outputs
        opt_policies = []
        batch_actions = np.array(batch_actions)
        batch_actions_one_hot = np.zeros((batch_actions.size, batch_actions.max()+1))
        batch_actions_one_hot[np.arange(batch_actions.size), batch_actions.reshape(-1)] = 1
        batch_actions_one_hot = batch_actions_one_hot.reshape(batch_actions.shape[0], batch_actions.shape[1], -1)
        for option in range(model.latent_dim):
            opt_vector = torch.zeros_like(all_z['samples'][0])
            opt_vector[:, option] = 1.
            opt_policy = model.decode(opt_vector, inputs[0])
            opt_policy = torch.sum(torch.tensor(batch_actions_one_hot).to(device) * opt_policy, axis=2, keepdim=True)
            opt_policies.append(opt_policy)
        opt_policies_cat = torch.cat(opt_policies, dim=-1)
        opt_policies_cat_amax = torch.argmax(opt_policies_cat, axis=-1)
        option_usage = torch.bincount(opt_policies_cat_amax.view(-1), minlength=model.latent_dim)/(opt_policies_cat_amax.shape[0]*opt_policies_cat_amax.shape[1])
        if min(option_usage) > 0.5/model.latent_dim:
            model.add_option(optimizer)


    if step % args.log_interval == 0:
        # Run evaluation.
        model.eval()
        outputs = model.forward(test_inputs, test_lengths)
        acc, rec = utils.get_reconstruction_accuracy(test_inputs, outputs, args)

        # Accumulate metrics.
        batch_acc = acc.item()
        batch_loss = nll.item()
        print('step: {}, nll_train: {:.6f}, rec_acc_eval: {:.3f}, K: {}'.format(
            step, batch_loss, batch_acc, model.latent_dim))
        # print('eta: {}'.format(outputs[-2]))
        # print('z: {}'.format(outputs[-3]['logits'][0]))
        # print('input sample: {}'.format(test_inputs[1][-1, :test_lengths[-1] - 1]))
        # print('reconstruction: {}'.format(rec[-1]))

    beta_entropy *= beta_entropy_ratio
    step += 1

model.save(os.path.join(run_dir, 'checkpoint.pth'))

score = model.evaluate_score(test_inputs[0], test_inputs[1])
print(f"Achieved a score of {score:.3f}.")

if args.results_file:
    with open(args.results_file, 'a') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
        f.write(str(score))
        f.write(' ')
        f.write(str(batch_acc))
        f.write(' ')
        f.write(str(model.K))
        f.write('\n')

