import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import write_summary
import time
from tqdm import tqdm
import tensorboard as tb
import sys
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

class Mortality(nn.Module):
    def __init__(self, input_dim):
        super(Mortality, self).__init__()
        self.layer1 = nn.Linear(input_dim, 250)
        self.bn1 = nn.BatchNorm1d(250)
        self.dp1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(250, 150)
        self.bn2 = nn.BatchNorm1d(150)
        self.dp2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(150, 75)
        self.bn3 = nn.BatchNorm1d(75)
        self.dp3 = nn.Dropout(0.2)
        self.layer4 = nn.Linear(75,25)
        self.bn4 = nn.BatchNorm1d(25)
        self.dp4 = nn.Dropout(0.2)
        self.layer5 = nn.Linear(25,2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dp1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dp2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dp3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dp4(x)
        x = self.layer5(x)
        x = self.softmax(x)
        return x

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, write_to_file, get_test_batch, loss_fn, scheduler=None, eval_fns=None, log_dir=None, loss_wt = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.get_test_batch = get_test_batch
        self.write_to_file = write_to_file
        # self.loss_fn = loss_fn
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn_mse = torch.nn.MSELoss()
        self.loss_wt = loss_wt
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.writer = SummaryWriter(log_dir)
        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        # load/nn_model/NNmodel500.pt this pytorch model
        # model defination is in nn.py
        nn_model = Mortality(47)
        # path = 'model.load_state_dict(torch.load('/home/amridul/mimic/RL-for-Sepsis/feedback_classifier_DT/nnclsfier/NNMortalityborder.pth'))'
        # nn_model.load_state_dict(torch.load('nn_model/NNmodel500.pt'))
        model_path = "/home/dipesh/feedback_classifier_DT/nnclsfier/NNfinal.pth"
        nn_model.load_state_dict(torch.load(model_path))
        nn_model.cuda()
        nn_model.eval()
        # nn_model = torch.load('nn_model/NNmodel500.pt')
        # self.model.train()
        for i in range(num_steps):
            train_loss, accuracy, error_rate_rmspe, error_rate_max_div, state_loss = self.train_step(nn_model)
            # adv_train_loss, _ = self.train_step_complete_traj(nn_model)
            write_summary(self.writer, info={'training loss': train_loss, 'accuracy': accuracy}, step=i+(iter_num-1)*num_steps)
            train_losses.append(train_loss)
            if i % 2000 == 0:

                self.write_to_file(f"### ###\nStep: {i}/{num_steps}", print_it=True)
                self.write_to_file(f"Training loss: {train_loss:.3f} Accuracy: {accuracy:.3f}", print_it=True)
                self.write_to_file(f"State loss: {state_loss:.3f}", print_it=True)
                self.write_to_file(f"Error rate Max Div (state_loss/max(states_gt)): {error_rate_max_div:.3f} x100%", print_it=True)
                self.write_to_file(f"Error rate RMSPE: {error_rate_rmspe:.3f} x100%", print_it=True)

                test_loss, test_accuracy,obs, state_preds, rtg, _ = self.test_step(mode=0)
                write_summary(self.writer, info={'test loss': test_loss, 'test accuracy': test_accuracy},
                               step=i + (iter_num - 1) * num_steps)
                self.write_to_file(f"Test loss: {test_loss:.3f} Test accuracy: {test_accuracy:.3f}", print_it=True)

                neg_test_loss, neg_test_accuracy, obs, state_preds, rew, all_state_preds = self.test_step(mode=-1)
                write_summary(self.writer, info={'neg test loss': neg_test_loss, 'neg test accuracy': neg_test_accuracy},
                              step=i + (iter_num - 1) * num_steps)
                self.write_to_file(f"Neg Test loss: {neg_test_loss:.3f} Neg Test accuracy: {neg_test_accuracy:.3f}", print_it=True)
                dying_patient_count = torch.argmax(nn_model(state_preds), axis  = 1).sum()
                dying_patient_die = torch.argmax(nn_model(state_preds), axis  = 1).sum()/len(state_preds)
                total_dying_patient = len(state_preds)

                all_state_mort = nn_model(all_state_preds.reshape(-1,all_state_preds.shape[2])).reshape(-1,all_state_preds.shape[1],2)
                all_state_dying_mort_count = torch.argmax(all_state_mort, axis  = 2).max(axis = 1)[0].sum()
                all_state_dying_mort = all_state_dying_mort_count/len(all_state_preds)

                pos_test_loss, pos_test_accuracy,obs, state_preds, rew, all_state_preds = self.test_step(mode=1)
                write_summary(self.writer, info={'pos test loss': pos_test_loss, 'pos test accuracy': pos_test_accuracy},
                              step=i + (iter_num - 1) * num_steps)
                self.write_to_file(f"Pos Test loss: {pos_test_loss:.3f} Pos Test accuracy: {pos_test_accuracy:.3f}", print_it=True)
                good_patient_count = torch.argmax(nn_model(state_preds), axis  = 1).sum()
                good_patient_die = torch.argmax(nn_model(state_preds), axis  = 1).sum()/len(state_preds)
                total_good_patient = len(state_preds)

                all_state_mort = nn_model(all_state_preds.reshape(-1,all_state_preds.shape[2])).reshape(-1,all_state_preds.shape[1],2)
                all_state_good_mort_count = torch.argmax(all_state_mort, axis  = 2).max(axis = 1)[0].sum()
                all_state_good_mort = all_state_good_mort_count/len(all_state_preds)

                # count_survive = self.test_complete_traj(mode = 0)

                self.write_to_file(f"=> Patients dying in the last state (originally surviving) : {good_patient_die:.3f}, count: {good_patient_count}/{total_good_patient}", print_it=True)
                self.write_to_file(f"=> Patients dying in the last state (originally dying) : {dying_patient_die:.3f}, count: {dying_patient_count}/{total_dying_patient}", print_it=True)

                self.write_to_file(f"=> Patients dying in any state (originally surviving) : {all_state_good_mort:.3f}, count: {all_state_good_mort_count}/{total_good_patient}", print_it=True)
                self.write_to_file(f"=> Patients dying in any state (originally dying) : {all_state_dying_mort:.3f}, count: {all_state_dying_mort_count}/{total_dying_patient}", print_it=True)
                # pdb.set_trace()

                ############################################################################################################
                # Getting complete trajectory
                ############################################################################################################

                # test_loss, test_accuracy,obs, state_preds, rtg, _ = self.test_complete_traj(mode=0)
                # write_summary(self.writer, info={'test loss': test_loss, 'test accuracy': test_accuracy},
                #                step=i + (iter_num - 1) * num_steps)
                # self.write_to_file(f"Test loss: {test_loss:.3f} Test accuracy: {test_accuracy:.3f}", print_it=True)

                neg_test_loss, neg_test_accuracy, obs, state_preds, rew, all_state_preds = self.test_complete_traj(mode=-1)
                # write_summary(self.writer, info={'neg test loss': neg_test_loss, 'neg test accuracy': neg_test_accuracy},
                #               step=i + (iter_num - 1) * num_steps)
                # self.write_to_file(f"Neg Test loss: {neg_test_loss:.3f} Neg Test accuracy: {neg_test_accuracy:.3f}", print_it=True)
                dying_patient_count = torch.argmax(nn_model(state_preds), axis  = 1).sum()
                dying_patient_die = torch.argmax(nn_model(state_preds), axis  = 1).sum()/len(state_preds)
                total_dying_patient = len(state_preds)

                all_state_mort = nn_model(all_state_preds.reshape(-1,all_state_preds.shape[2])).reshape(-1,all_state_preds.shape[1],2)
                all_state_dying_mort_count = torch.argmax(all_state_mort, axis  = 2).max(axis = 1)[0].sum()
                all_state_dying_mort = all_state_dying_mort_count/len(all_state_preds)

                pos_test_loss, pos_test_accuracy,obs, state_preds, rew, all_state_preds = self.test_complete_traj(mode=1)
                # write_summary(self.writer, info={'pos test loss': pos_test_loss, 'pos test accuracy': pos_test_accuracy},
                            #   step=i + (iter_num - 1) * num_steps)
                # self.write_to_file(f"Pos Test loss: {pos_test_loss:.3f} Pos Test accuracy: {pos_test_accuracy:.3f}", print_it=True)
                good_patient_count = torch.argmax(nn_model(state_preds), axis  = 1).sum()
                good_patient_die = torch.argmax(nn_model(state_preds), axis  = 1).sum()/len(state_preds)
                total_good_patient = len(state_preds)

                all_state_mort = nn_model(all_state_preds.reshape(-1,all_state_preds.shape[2])).reshape(-1,all_state_preds.shape[1],2)
                all_state_good_mort_count = torch.argmax(all_state_mort, axis  = 2).max(axis = 1)[0].sum()
                all_state_good_mort = all_state_good_mort_count/len(all_state_preds)

                count_survive = self.test_complete_traj(mode = 0)

                self.write_to_file(f"Complete trajectory results", print_it=True)
                self.write_to_file(f"=> Patients dying in the last state (originally surviving) : {good_patient_die:.3f}, count: {good_patient_count}/{total_good_patient}", print_it=True)
                self.write_to_file(f"=> Patients dying in the last state (originally dying) : {dying_patient_die:.3f}, count: {dying_patient_count}/{total_dying_patient}", print_it=True)

                self.write_to_file(f"=> Patients dying in any state (originally surviving) : {all_state_good_mort:.3f}, count: {all_state_good_mort_count}/{total_good_patient}", print_it=True)
                self.write_to_file(f"=> Patients dying in any state (originally dying) : {all_state_dying_mort:.3f}, count: {all_state_dying_mort_count}/{total_dying_patient}", print_it=True)


            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        # logs['training/train_loss_mean'] = np.mean(train_losses)
        # logs['training/train_loss_std'] = np.std(train_losses)

        # for k in self.diagnostics:
        #     logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs,obs

    def train_step(self):
        print(f"Not Using this")
        exit()
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def test_step(self, mode):
        raise NotImplementedError
