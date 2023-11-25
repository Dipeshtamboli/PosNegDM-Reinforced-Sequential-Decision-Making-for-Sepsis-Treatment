import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os 
import argparse
import random, datetime
from utils import read_dst, discount_cumsum

from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
# from tensorboardX import SummaryWriter

seed_val = 1
torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# gpu_devs = [0, 1]
# gpu_devs = [0, 1, 2, 3]
# device = torch.device("cuda:2,3" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
# device = variant.get('device', 'cuda')


def experiment(variant):

	exp_name = "1_08_1"
	description_string = f"Ablation_alpha_beta_gamma"

	log_txt = f"logs_text_mse/{exp_name}.txt"
	def write_to_file(string_to_write, print_it=True):
		write_f = open(f"{log_txt}", "a")
		write_f.write(f"{string_to_write}\n")
		write_f.close()	
		if print_it:
			print(string_to_write)

	def INFO(string_to_write):
		write_to_file(f"|| INFO || {string_to_write}")

	write_to_file(f"###################"*4)
	INFO(f"{datetime.datetime.now()}")
	INFO(description_string)

	model_type = variant['model_type']
	pos_only = variant['pos_only']
	unique_token = f"{model_type}_{datetime.datetime.now()}"

	# env relevant
	state_dim = 47
	act_dim = 25 # TODO: discrete
	max_ep_len = 100 # TODO
	scale = 1.0 # TODO
	state_obs_wt = variant['state_obs_wt']

	log_dir = 'logs' + f'/feed_' + unique_token
	# log_dir = 'logs' + f'/try_' + unique_token
	# load dataset
	dst = variant.get('dst')
	dataset_path = f'data/{dst}'
	trajectories = read_dst(dataset_path)
	# print("before: ", len(trajectories))
	test_trajs = trajectories[:int(0.3*len(trajectories))]
	trajectories = trajectories[int(0.3*len(trajectories)):]
	# print("after: ", len(trajectories), len(test_trajs))

	# split the test trajectories according to its return
	pos_idxes = []
	neg_idxes = []
	pos_length, neg_length = 0, 0
	for idx, t_path in enumerate(test_trajs):
		ret = t_path['rewards'].sum()
		if ret > 0:
			pos_idxes.append(idx)
			pos_length += len(t_path['rewards'])
		else:
			neg_idxes.append(idx)
			neg_length += len(t_path['rewards'])


	# print(len(pos_idxes), pos_length, len(neg_idxes), neg_length)
	print(f"pos: {len(pos_idxes)}, neg: {len(neg_idxes)}")
	print(f"pos: {pos_length}, neg: {neg_length}")

	if pos_only:
		temp_trajectories = []
		for path in trajectories:
			ret = path['rewards'].sum()
			if ret > 0:
				temp_trajectories.append(path)
		trajectories = temp_trajectories

	# save all path information into separate lists
	mode = variant.get('mode', 'normal')
	states, traj_lens, returns = [], [], []
	for path in trajectories:
		if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
			path['rewards'][-1] = path['rewards'].sum()
			path['rewards'][:-1] = 0.
		states.append(path['observations'])
		traj_lens.append(len(path['observations']))
		returns.append(path['rewards'].sum())

	traj_lens, returns = np.array(traj_lens), np.array(returns)
	num_timesteps = sum(traj_lens)
	# print("return list: ", returns[:100])
	print("0: ", len(states), states[0].shape)
	## used for input normalization
	states = np.concatenate(states, axis=0)
	state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
	print("1: ", traj_lens.shape, returns.shape, states.shape, state_mean.shape, state_std.shape)
	# print("2: ", states)

	print('=' * 50)
	print(f'Starting new experiment')
	print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
	print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
	print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
	print('=' * 50)

	K = variant['K']
	if model_type == 'bc':
		K = 3
	batch_size = variant['batch_size']
	pct_traj = variant.get('pct_traj', 1.)

	# only train on top pct_traj trajectories (for %BC experiment)
	num_timesteps = max(int(pct_traj * num_timesteps), 1)
	sorted_inds = np.argsort(returns)  # lowest to highest

	num_trajectories = 1
	timesteps = traj_lens[sorted_inds[-1]]

	ind = len(trajectories) - 2

	while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
		timesteps += traj_lens[sorted_inds[ind]]
		num_trajectories += 1
		ind -= 1
	sorted_inds = sorted_inds[-num_trajectories:]

	print("3: ", num_trajectories)

	# used to reweight sampling so we sample according to timesteps instead of trajectories
	p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

	def get_batch(batch_size=256, max_len=K):
		batch_inds = np.random.choice(
			np.arange(num_trajectories),
			size=batch_size,
			replace=True,
			p=p_sample,  # reweights so we sample according to timesteps, proportional to the trajectory length
		)

		s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
		for i in range(batch_size):
			traj = trajectories[int(sorted_inds[batch_inds[i]])]

			# si = random.randint(0, traj['rewards'].shape[0] - 1)
			# get last K states
			si = max(0, traj['rewards'].shape[0] - max_len)

			# get sequences from dataset
			s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
			acts = traj['actions'][si:si + max_len]
			onehot_acts = np.eye(act_dim)[acts]
			a.append(onehot_acts.reshape(1, -1, act_dim))
			r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
			# print(r)

			if 'terminals' in traj:
				d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
			else:
				d.append(traj['dones'][si:si + max_len].reshape(1, -1))

			timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
			timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
			# print(timesteps)

			rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
			if rtg[-1].shape[1] <= s[-1].shape[1]: # why would this happen?
				rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

			# padding and state + reward normalization
			tlen = s[-1].shape[1]
			s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
			s[-1] = (s[-1] - state_mean) / state_std
			a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
			r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
			d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
			rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
			timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
			mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

		s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
		a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
		r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
		d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
		rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
		timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
		mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

		return s, a, r, d, rtg, timesteps, mask

	def get_test_batch(batch_size=563, max_len=K, mode=0):
		if mode == 'god_mode':
			batch_inds = np.random.choice(
				np.arange(len(test_trajs)),
				size=(len(test_trajs)),
				replace=False)
		elif mode == 0:
			batch_inds = np.random.choice(
				np.arange(len(test_trajs)),
				# size=len(test_trajs),
				size=batch_size,
				replace=False)
		elif mode == 1:
			batch_inds = np.random.choice(
				pos_idxes,
				# size=len(pos_idxes),
				size=batch_size,
				replace=False)
		else:
			assert mode == -1
			batch_inds = np.random.choice(
				neg_idxes,
				size=len(neg_idxes),
				replace=False)

		s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
		for i in range(len(batch_inds)):
			traj = test_trajs[int(batch_inds[i])]

			# si = 0 # different from training
			si = max(0, traj['rewards'].shape[0] - max_len)
			# get sequences from dataset
			s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
			acts = traj['actions'][si:si + max_len]
			onehot_acts = np.eye(act_dim)[acts]
			a.append(onehot_acts.reshape(1, -1, act_dim))
			r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
			# print(r)
			if 'terminals' in traj:
				d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
			else:
				d.append(traj['dones'][si:si + max_len].reshape(1, -1))

			timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
			timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
			# print(timesteps)
			rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
			if rtg[-1].shape[1] <= s[-1].shape[1]:  # why would this happen?
				rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
				# padding and state + reward normalization
			tlen = s[-1].shape[1]
			s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
			s[-1] = (s[-1] - state_mean) / state_std
			a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
			r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
			d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
			rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
			timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
			mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

		s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
		a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
		r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
		d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
		rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
		timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
		mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

		return s, a, r, d, rtg, timesteps, mask



	if model_type == 'dt':
		model = DecisionTransformer(
			state_dim=state_dim,
			act_dim=act_dim,
			max_length=K,
			max_ep_len=max_ep_len,
			hidden_size=variant['embed_dim'],
			n_layer=variant['n_layer'],
			n_head=variant['n_head'],
			n_inner=4 * variant['embed_dim'],
			activation_function=variant['activation_function'],
			n_positions=1024,
			resid_pdrop=variant['dropout'],
			attn_pdrop=variant['dropout'],
		)
	elif model_type == 'bc':
		model = MLPBCModel(
			state_dim=state_dim,
			act_dim=act_dim,
			max_length=K,
			hidden_size=variant['embed_dim'],
			n_layer=variant['n_layer'],
		)
	else:
		raise NotImplementedError

	model = model.to(device=device)
	# model= nn.DataParallel(model,device_ids = gpu_devs)
	# model.to(f'cuda:{model.device_ids[0]}')

	warmup_steps = variant['warmup_steps']
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=variant['learning_rate'],
		weight_decay=variant['weight_decay'],
	)
	scheduler = torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		lambda steps: min((steps + 1) / warmup_steps, 1)
	)

	if model_type == 'dt':
		trainer = SequenceTrainer(
			model=model,
			optimizer=optimizer,
			batch_size=batch_size,
			get_batch=get_batch,
			write_to_file=write_to_file,
			get_test_batch=get_test_batch,
			scheduler=scheduler,
			log_dir = log_dir,
			loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2)
		)
	elif model_type == 'bc':
		trainer = ActTrainer(
			model=model,
			optimizer=optimizer,
			batch_size=batch_size,
			get_batch=get_batch,
			get_test_batch=get_test_batch,
			scheduler=scheduler,
			log_dir = log_dir,
			loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
			loss_wt = state_obs_wt,
		)


	for iter in tqdm(range(variant['max_iters'])):
		trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter + 1, print_logs=True)

	torch.save(model.state_dict(), os.path.join(log_dir, 'model.pt'))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--dst', type=str, default='sepsis_final_data_withTimes.csv')
	parser.add_argument('--pos_only', type=bool, default=False)
	parser.add_argument('--mode', type=str, default='delayed')  # normal for standard setting, delayed for sparse
	parser.add_argument('--K', type=int, default=10) # TODO
	parser.add_argument('--pct_traj', type=float, default=1.)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
	parser.add_argument('--embed_dim', type=int, default=128)
	parser.add_argument('--n_layer', type=int, default=3)
	parser.add_argument('--n_head', type=int, default=1)
	parser.add_argument('--activation_function', type=str, default='relu')
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
	parser.add_argument('--warmup_steps', type=int, default=10000)
	parser.add_argument('--max_iters', type=int, default=20)
	parser.add_argument('--num_steps_per_iter', type=int, default=10000)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--state_obs_wt', '-sow', type=float, default=1.0)

	args = parser.parse_args()


	experiment(variant=vars(args))
