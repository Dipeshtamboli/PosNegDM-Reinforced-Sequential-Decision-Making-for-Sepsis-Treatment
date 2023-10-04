import numpy as np
import torch
from utils import get_accuracy

from decision_transformer.training.trainer import Trainer


class ActTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, _, attention_mask = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:,0],
        )

        act_dim = action_preds.shape[2]
        # print("before: ", action_preds.shape, action_target.shape)
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:,-1].reshape(-1, act_dim)
        action_target = torch.argmax(action_target, dim=1).long()
        # print("after: ", action_preds.shape, action_target.shape)

        loss = self.loss_fn(action_preds, action_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        accuracy = get_accuracy(action_preds.cpu().detach().clone().numpy(),
                                action_target.cpu().detach().clone().numpy())

        return loss.detach().cpu().item(), accuracy

    def test_step(self, mode):
        states, actions, rewards, dones, rtg, _, attention_mask = self.get_test_batch(mode=mode)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        self.model.eval()
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:,0])

        act_dim = action_preds.shape[2]
        # print("before: ", action_preds.shape, action_target.shape)
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:,-1].reshape(-1, act_dim)
        action_target = torch.argmax(action_target, dim=1).long()
        # print("after: ", action_preds.shape, action_target.shape)

        loss = self.loss_fn(action_preds, action_target)
        accuracy = get_accuracy(action_preds.cpu().detach().clone().numpy(),
                                action_target.cpu().detach().clone().numpy())

        return loss.detach().cpu().item(), accuracy
