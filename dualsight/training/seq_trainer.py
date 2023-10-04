import numpy as np
import torch
from utils import get_accuracy
import pdb
from decision_transformer.training.trainer import Trainer
import json

class SequenceTrainer(Trainer):


    states_gt = []
    reward_gt = []
    states_pred = []
    reward_pred = []


    def train_step(self, nn_model):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        #print(states.size())
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )


        final_pred_states = state_preds[:,9,:]
        all_state_preds = state_preds.reshape(-1,state_preds.shape[2])
        nn_model_preds = nn_model(final_pred_states)
        nn_model_all_preds = nn_model(all_state_preds)
        one_tens = torch.tensor([1, 0]).repeat(64, 1)
        one_all_tens = torch.tensor([1, 0]).repeat(nn_model_all_preds.shape[0], 1)
        # pdb.set_trace()
        # argmax of one_tens
        # one_tens = torch.argmax(one_tens, dim=1).long()

        # adversarial_loss = self.loss_fn(nn_model_preds, torch.argmax(one_tens, dim=1).long().cuda())
        adversarial_loss_all = self.loss_fn(nn_model_all_preds, torch.argmax(one_all_tens, dim=1).long().cuda())

        # create y of shape nn_model_preds for cross entropy loss where all the classes are zero
        act_dim = action_preds.shape[2]

        pos_ids = rewards[:, -1, 0] == 1
        pos_action_preds = action_preds[pos_ids]
        pos_action_target = action_target[pos_ids]
        pos_traj_states_preds = state_preds[pos_ids]
        pos_traj_states_target = states[pos_ids]

        pos_action_preds = pos_action_preds.reshape(-1, act_dim)[attention_mask[pos_ids].reshape(-1) > 0] # predict one-hot
        pos_action_target = pos_action_target.reshape(-1, act_dim)[attention_mask[pos_ids].reshape(-1) > 0] # real value
        pos_action_target = torch.argmax(pos_action_target, dim=1).long()

        # pdb.set_trace()
        # print(state_preds.size())
        # print("before: ", action_preds[-1][-1], action_preds.shape, action_preds[-1][-1].sum())
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0] # predict one-hot
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0] # real value
        action_target = torch.argmax(action_target, dim=1).long()

        # loss = self.loss_fn(action_preds, action_target)
        # loss = self.loss_fn(action_preds, action_target) + self.loss_wt * self.loss_fn_mse(state_preds, states) + adversarial_loss * 1.0
        action_all_loss = self.loss_fn(action_preds, action_target)
        action_pos_loss = self.loss_fn(pos_action_preds, pos_action_target)
        states_pos_loss =  self.loss_fn_mse(pos_traj_states_preds, pos_traj_states_target)
        states_all_loss = self.loss_fn_mse(state_preds, states)

        # measure relative error states_all_loss with respect to ground truth states

        # divide states_all_loss

        # error_rate = np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))
        # write above error rate in torch
        error_rate_rmspe = torch.sqrt(torch.mean(torch.square(((states - state_preds) / states))))
        error_rate_max_div = states_all_loss / torch.max(states)

        # loss = action_pos_loss + states_all_loss + adversarial_loss_all
        alpha = 1
        beta = 0.8
        gamma = 1
        loss = alpha * action_pos_loss + beta * states_all_loss + gamma * adversarial_loss_all

        # loss = states_all_loss + adversarial_loss_all
        # pdb.set_trace()
        # loss = self.loss_wt * self.loss_fn_mse(state_preds, states) + adversarial_loss
        # loss = self.loss_fn(action_preds, action_target) + self.loss_wt * self.loss_fn_mse(state_preds, states) 
        # loss = self.loss_fn(action_preds, action_target) +  adversarial_loss * 1.0
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        accuracy = get_accuracy(action_preds.cpu().detach().clone().numpy(), action_target.cpu().detach().clone().numpy())
        # with torch.no_grad():
        #     self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(), accuracy, error_rate_rmspe.detach().cpu().item(), error_rate_max_div.detach().cpu().item(), states_all_loss.detach().cpu().item()

    def train_step_complete_traj(self, nn_model):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        #print(states.size())
        attention_mask = torch.zeros_like(attention_mask)

        init_state = states[:,:1,:]
        init_rewards = rewards[:,:1,:]
        init_actions = actions[:,:1,:]
        init_rtg = rtg[:,:1,:]
        init_timesteps = timesteps[:,:1]
        init_attention_mask = torch.ones_like(attention_mask[:,:1])
        loss_in_step = 0
        for i in range(9):

            state_preds, action_preds, reward_preds = self.model.forward(
                init_state, init_actions, init_rewards, init_rtg[:,:i+1,:], init_timesteps, attention_mask=init_attention_mask)

            last_pred_states = state_preds[:,-1,:].reshape(-1,state_preds.shape[2])
            nn_model_last_state_pred = nn_model(last_pred_states)
            ones_gt = torch.tensor([1, 0]).repeat(nn_model_last_state_pred.shape[0], 1)
            loss_in_step += self.loss_fn(nn_model_last_state_pred, torch.argmax(ones_gt, dim=1).long().cuda())

            init_state = torch.cat((init_state, state_preds[:,-1:,:]), dim=1)
            init_actions = torch.cat((init_actions, action_preds[:,-1:,:]), dim=1)
            init_rewards = torch.cat((init_rewards, reward_preds[:,-1:,:]), dim=1)
            init_rtg = torch.cat((init_rtg, rtg[:,i+1:i+2,:]), dim=1)
            init_timesteps = torch.cat((init_timesteps, timesteps[:,i+1:i+2]), dim=1)
            init_attention_mask = torch.cat((init_attention_mask, attention_mask[:,i+1:i+2]), dim=1)

        # pdb.set_trace()
        state_preds = init_state

        final_pred_states = state_preds[:,9,:]
        all_state_preds = state_preds.reshape(-1,state_preds.shape[2])
        nn_model_preds = nn_model(final_pred_states)
        nn_model_all_preds = nn_model(all_state_preds)
        one_tens = torch.tensor([1, 0]).repeat(64, 1)
        one_all_tens = torch.tensor([1, 0]).repeat(nn_model_all_preds.shape[0], 1)
        # pdb.set_trace()
        # argmax of one_tens
        # one_tens = torch.argmax(one_tens, dim=1).long()

        # adversarial_loss = self.loss_fn(nn_model_preds, torch.argmax(one_tens, dim=1).long().cuda())
        adversarial_loss_all = self.loss_fn(nn_model_all_preds, torch.argmax(one_all_tens, dim=1).long().cuda())
        loss = loss_in_step + 10 * adversarial_loss_all

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item(), 0

    def test_step(self, mode):
        
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_test_batch(mode=mode)
        action_target = torch.clone(actions)

        SequenceTrainer.states_gt.extend(states[:,9,:].cpu().tolist())
        SequenceTrainer.reward_gt.extend((rewards.sum(dim=1)).cpu().tolist())

        # print(states.size())
        self.model.eval()
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask)
        SequenceTrainer.states_pred.extend(state_preds[:,9,:].cpu().tolist())
        # pdb.set_trace()
        obs=[SequenceTrainer.states_gt,SequenceTrainer.reward_gt,SequenceTrainer.states_pred]
       
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0] # predict one-hot
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0] # real value
        action_target = torch.argmax(action_target, dim=1).long()

        loss = self.loss_fn(action_preds, action_target)
        accuracy = get_accuracy(action_preds.cpu().detach().clone().numpy(), action_target.cpu().detach().clone().numpy())

        self.model.train()

        return loss.detach().cpu().item(), accuracy, obs, state_preds[:,9,:], rewards, state_preds

    def test_complete_traj(self, mode):
        
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_test_batch(mode=mode)
        action_target = torch.clone(actions)

        # SequenceTrainer.states_gt.extend(states[:,9,:].cpu().tolist())
        # SequenceTrainer.reward_gt.extend((rewards.sum(dim=1)).cpu().tolist())

        # print(states.size())
        self.model.eval()
        attention_mask = torch.zeros_like(attention_mask)

        init_state = states[:,:1,:]
        init_rewards = rewards[:,:1,:]
        init_actions = actions[:,:1,:]
        init_rtg = rtg[:,:1,:]
        init_timesteps = timesteps[:,:1]
        init_attention_mask = torch.ones_like(attention_mask[:,:1])
        for i in range(10):

            state_preds, action_preds, reward_preds = self.model.forward(
                init_state, init_actions, init_rewards, init_rtg[:,:i+1,:], init_timesteps, attention_mask=init_attention_mask)

            init_state = torch.cat((init_state, state_preds[:,-1:,:]), dim=1)
            init_actions = torch.cat((init_actions, action_preds[:,-1:,:]), dim=1)
            init_rewards = torch.cat((init_rewards, reward_preds[:,-1:,:]), dim=1)
            init_rtg = torch.cat((init_rtg, rtg[:,i+1:i+2,:]), dim=1)
            init_timesteps = torch.cat((init_timesteps, timesteps[:,i+1:i+2]), dim=1)
            init_attention_mask = torch.cat((init_attention_mask, attention_mask[:,i+1:i+2]), dim=1)

        state_preds = init_state[:, 1:, :]

        # new_attention_mask = torch.zeros_like(attention_mask)
        # new_state_preds, new_action_preds, new_reward_preds = self.model.forward(
        #     states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=new_attention_mask)

        SequenceTrainer.states_pred.extend(state_preds[:,-1,:].cpu().tolist())
        # pdb.set_trace()
        obs=[SequenceTrainer.states_gt,SequenceTrainer.reward_gt,SequenceTrainer.states_pred]
       
        # act_dim = action_preds.shape[2]
        # action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0] # predict one-hot
        # action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0] # real value
        # action_target = torch.argmax(action_target, dim=1).long()

        # loss = self.loss_fn(action_preds, action_target)
        # accuracy = get_accuracy(action_preds.cpu().detach().clone().numpy(), action_target.cpu().detach().clone().numpy())

        self.model.train()

        return 0, 0, obs, state_preds[:,9,:], rewards, state_preds    
    