import gym, os
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0").unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0001
epsilon = 0.2

class SharedLinear(nn.Module):
    def __init__(self, state_size, feature_size):
        super(SharedLinear, self).__init__()
        self.state_size = state_size
        self.feature_size = feature_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.feature_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output



class Actor_A(nn.Module):
    def __init__(self, state_size, feature_size, action_size):
        super(Actor_A, self).__init__()
        self.state_size = state_size
        self.feature_size = feature_size
        self.action_size = action_size
        self.shared_linear = SharedLinear(self.state_size, self.feature_size)
        self.add_module('shared_linear', self.shared_linear)  # add shared_linear as a submodule
        self.linear1 = nn.Linear(self.feature_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        feature = self.shared_linear(state)
        output = F.relu(self.linear1(feature))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Actor_B(nn.Module):
    def __init__(self, state_size, feature_size, action_size):
        super(Actor_B, self).__init__()
        self.state_size = state_size
        self.feature_size = feature_size
        self.action_size = action_size
        self.shared_linear = SharedLinear(self.state_size, self.feature_size)
        self.add_module('shared_linear', self.shared_linear)  # add shared_linear as a submodule
        self.linear1 = nn.Linear(self.feature_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        feature = self.shared_linear(state)
        output = F.relu(self.linear1(feature))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Meta_net(nn.Module):
    def __init__(self, state_size, feature_size, task_num):
        super(Meta_net, self).__init__()
        self.state_size = state_size
        self.feature_size = feature_size
        self.task_num = task_num
        # self.shared_linear = SharedLinear(self.state_size, self.feature_size)
        # self.add_module('shared_linear', self.shared_linear)  # add shared_linear as a submodule
        # self.linear1 = nn.Linear(self.feature_size, 128)
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear2_1 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, self.task_num)

    def forward(self, state):
        # feature = self.shared_linear(state)
        # output = F.relu(self.linear1(feature))
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear2_1(output))
        output = self.linear3(output)
        output = F.softmax(output)
        return output


class Critic_A(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic_A, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        # self.linear2_2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        # output = F.relu(self.linear2_2(output))
        value = self.linear3(output)
        return value

class Critic_B(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic_B, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.98):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def clip(ratio, epsilon):
    return torch.clamp(ratio,1-epsilon,1+epsilon)


def trainIters(shared_linear, meta, actor_a, actor_b, critic_a, critic_b, n_iters):
    clip_epsilon = 0.3
    optimizerA = optim.Adam(list(actor_a.parameters()) + list(actor_b.parameters()) + list(shared_linear.parameters()), lr=0.0017)
    optimizerC = optim.Adam(list(critic_a.parameters()) + list(critic_b.parameters()), lr=0.0017)
    optimizerMeta = optim.Adam(list(meta.parameters()), lr=0.00017)
    schedulerA = lr_scheduler.StepLR(optimizerA, step_size=80, gamma=0.01)
    schedulerC = lr_scheduler.StepLR(optimizerC, step_size=80, gamma=0.01)
    schedulerMeta = lr_scheduler.StepLR(optimizerMeta, step_size=80, gamma=0.01)
    loss_record = []
    for iter in range(n_iters):
        state = env.reset()
        states = []
        actions_a = []
        actions_b = []
        meta_parameters = []
        log_probs = []
        log_probs_a = []
        log_probs_b = []
        values = []
        values_a = []
        values_b = []
        rewards = []
        rewards_a = []
        rewards_b = []
        masks = []
        entropy_a = 0
        entropy_b = 0
        env.reset()
        index_record = []

        for i in count():
            env.render()
            state = torch.FloatTensor(state).to(device)
            states.append(state)
            # feature = shared_linear(state)
            dist_a, dist_b, value_a, value_b, meta_parameter = actor_a(state), actor_b(state), critic_a(state), critic_b(state), meta(state)

            action_a = dist_a.sample()
            action_b = dist_b.sample()
            actions_a.append(action_a)
            actions_b.append(action_b)

            meta_parameters.append(meta_parameter)


            # Test the effect of a single model
            # meta_parameter = [1, 0]  # model b: [0, 1]
            print("meta_parameter", meta_parameter)


            log_prob_a = dist_a.log_prob(action_a).unsqueeze(0)
            entropy_a += dist_a.entropy().mean()
            log_prob_b = dist_b.log_prob(action_b).unsqueeze(0)
            entropy_b += dist_b.entropy().mean()

            log_prob_a_no_grad = log_prob_a.detach()
            log_prob_b_no_grad = log_prob_b.detach()
            log_prob = meta_parameter[0] * log_prob_a_no_grad + meta_parameter[1] * log_prob_b_no_grad

            action = meta_parameter[0] * action_a + meta_parameter[1] * action_b
            env_action = int(action.detach().numpy())
            next_state, reward, done, _ = env.step(env_action)

            # r_a opposite
            # r_b same
            if action == 1 and np.sign(state[3]) == 1:
                r_a = 0.0
                r_b = 1.0
            elif action == 0 and np.sign(state[3]) == -1:
                r_a = 0.0
                r_b = 1.0
            else:
                r_a = 1.0
                r_b = 0.0
            # print(r_a, r_b)

            log_probs.append(log_prob)
            log_probs_a.append(log_prob_a)
            log_probs_b.append(log_prob_b)

            values_a.append(value_a)
            values_b.append(value_b)
            value = meta_parameter[0] * value_a + meta_parameter[1] * value_b
            values.append(torch.tensor(value, dtype=torch.float, device=device))

            # reward = meta_parameter[0] * r_a + meta_parameter[1] * r_b
            rewards_a.append(torch.tensor([r_a], dtype=torch.float, device=device))
            rewards_b.append(torch.tensor([r_b], dtype=torch.float, device=device))
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                index_record = i
                # if i > 100:
                #     rewards[-1] = 100
                break
            elif i > 8000:
                print('Iteration: {}, Score: {}'.format(iter, i))
                index_record = 8000
                break

        next_state = torch.FloatTensor(next_state).to(device)
        next_value_a = critic_a(next_state)
        next_value_b = critic_b(next_state)
        next_meta_parameter = meta(next_state)

        next_value_a_no_grad = next_value_a.detach()
        next_value_b_no_grad = next_value_b.detach()
        next_value = next_meta_parameter[0] * next_value_a_no_grad + next_meta_parameter[1] * next_value_b_no_grad
        returns = compute_returns(next_value, rewards, masks)
        returns_a = compute_returns(next_value_a, rewards_a, masks)
        returns_b = compute_returns(next_value_b, rewards_b, masks)

        #Actor_A######################################

        log_probs_a = torch.cat(log_probs_a)
        returns_a = torch.cat(returns_a).detach()
        values_a = torch.cat(values_a)

        advantage_a = returns_a - values_a

        actor_loss_a = -(log_probs_a * advantage_a.detach()).mean()
        critic_loss_a = advantage_a.pow(2).mean()

        # Actor_B######################################

        log_probs_b = torch.cat(log_probs_b)
        returns_b = torch.cat(returns_b).detach()
        values_b = torch.cat(values_b)

        advantage_b = returns_b - values_b


        actor_loss_b = -(log_probs_b * advantage_b.detach()).mean()
        critic_loss_b = advantage_b.pow(2).mean()


        #####################################################3######
        states = torch.stack(states).to(device)
        action_dists_a = actor_a(states)
        action_dists_b = actor_b(states)
        new_log_probs_a = action_dists_a.log_prob(torch.stack(actions_a))
        new_log_probs_b = action_dists_b.log_prob(torch.stack(actions_b))
        meta_parameters = torch.stack(meta_parameters).to(device)
        # print(new_log_probs_a.shape, meta_parameters[:][0].shape)
        new_log_probs = new_log_probs_a * meta_parameters[:,0] + new_log_probs_b * meta_parameters[:,1]

        old_log_probs = torch.cat(log_probs)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

        # log_probs = torch.cat(log_probs)
        # log_probs = np.concatenate(log_probs, axis=0)
        # log_probs = torch.from_numpy(log_probs)
        returns = torch.cat(returns)
        values = torch.cat(values)

        advantages = returns - values


        critic_loss = advantages.pow(2).mean()
        # actor_loss = -(log_probs * advantage.detach()).mean()
        actor_loss = -torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach()).mean()
        loss_record.append(actor_loss.item())
        # print(loss_record)



        # optimizerA.zero_grad()
        # optimizerC.zero_grad()
        # actor_loss_a.backward()
        # critic_loss_a.backward()
        # actor_loss_b.backward()
        # critic_loss_b.backward()
        # optimizerA.step()
        # optimizerC.step()
        # schedulerA.step()
        # schedulerC.step()

        # optimizerMeta.zero_grad()
        # actor_loss.backward()
        # critic_loss.backward()
        # optimizerMeta.step()
        # schedulerMeta.step()
    # torch.save(actor_a, 'model_2/actor_a.pkl')
    # torch.save(actor_b, 'model_2/actor_b.pkl')
    # torch.save(critic_a, 'model_2/critic_a.pkl')
    # torch.save(critic_b, 'model_2/critic_b.pkl')
    # torch.save(meta, 'model_2/meta.pkl')
    # torch.save(shared_linear, 'model_2/shared_linear.pkl')
    env.close()
    # print(loss_record)
    # plt.figure()
    # plt.plot(loss_record)
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()


if __name__ == '__main__':
    # if os.path.exists('model/actor.pkl'):
    #     actor = torch.load('model/actor.pkl')
    #     print('Actor Model loaded')
    # else:
    #     actor = Actor(state_size, action_size).to(device)
    # if os.path.exists('model/critic.pkl'):
    #     critic = torch.load('model/critic.pkl')
    #     print('Critic Model loaded')
    # else:
    #     critic = Critic(state_size, action_size).to(device)
    model_name = 'model'
    task_num = 2
    feature_size = 256
    shared_linear = SharedLinear(state_size, feature_size).to(device)
    if os.path.exists(model_name + '/meta.pkl'):
        meta = torch.load(model_name + '/meta.pkl')
    else:
        meta = Meta_net(state_size, feature_size, task_num).to(device)
    if os.path.exists(model_name + '/shared_linear.pkl'):
        shared_linear = torch.load(model_name + '/shared_linear.pkl')
    else:
        shared_linear = SharedLinear(state_size, feature_size).to(device)
    if os.path.exists(model_name + '/actor_a.pkl'):
        actor_a = torch.load(model_name + '/actor_a.pkl')
    else:
        actor_a = Actor_A(state_size, feature_size, action_size).to(device)
    if os.path.exists(model_name + '/actor_b.pkl'):
        actor_b = torch.load(model_name + '/actor_b.pkl')
    else:
        actor_b = Actor_B(state_size, feature_size, action_size).to(device)
    if os.path.exists(model_name + '/critic_a.pkl'):
        critic_a = torch.load(model_name + '/critic_a.pkl')
    else:
        critic_a = Critic_A(state_size, action_size).to(device)
    if os.path.exists(model_name + '/critic_b.pkl'):
        critic_b = torch.load(model_name + '/critic_b.pkl')
    else:
        critic_b = Critic_B(state_size, action_size).to(device)
    trainIters(shared_linear, meta, actor_a, actor_b, critic_a, critic_b, n_iters=1000)