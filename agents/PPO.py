import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Actor_Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, act_dim):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.std_layer = nn.Parameter(torch.ones(act_dim) * -1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_layer(x)
        log_std = torch.clamp(self.std_layer, -2, 2)
        std = torch.exp(log_std).expand_as(mu)
        return mu, std


class Critic_Net(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPO_Agent:
    def __init__(self, state_dim, hidden_dim, action_dim, eps, gamma=0.99):
        self.gamma = gamma
        self.eps = eps
        self.actor_net = Actor_Net(state_dim, hidden_dim, action_dim).to(device)
        self.critic_net = Critic_Net(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=2e-5)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-5)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        mu, std = self.actor_net(state)

        actions = []
        log_probs = []

        for i in range(mu.shape[1]):
            if i in [0, 1]:
                dist = torch.distributions.Bernoulli(logits=mu[:, i])
                a = dist.sample()
                log_prob = dist.log_prob(a)
            else:
                dist = Normal(mu[:, i], std[:, i])
                raw_a = dist.rsample()
                a = torch.tanh(raw_a)
                log_prob = dist.log_prob(raw_a) - torch.log(1 - a ** 2 + 1e-6)

            actions.append(a)
            log_probs.append(log_prob)

        actions = torch.stack(actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)

        return actions.squeeze(0).detach().cpu().numpy(), log_probs.squeeze(0).detach()

    def compute_advatage(self, values, next_values, reward, done, lamda=0.95):
        adv = torch.zeros_like(reward)
        length = reward.shape[0]
        res = 0

        for i in range(length - 1, -1, -1):
            delta = reward[i] + (1 - done[i]) * self.gamma * next_values[i] - values[i]
            res = self.gamma * lamda * res * (1 - done[i]) + delta
            adv[i] = res

        return adv

    def update(self, states, actions, rewards, next_states, dones, old_log_probs):
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        actions = np.array(actions)

        state = torch.tensor(states, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_states, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        old_log_probs = torch.stack(old_log_probs).detach().to(device)

        with torch.no_grad():
            next_values = self.critic_net(next_state)
            td_target = rewards + self.gamma * next_values * (1 - dones)

        for _ in range(6):
            values = self.critic_net(state)
            critic_loss = F.mse_loss(values, td_target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(),0.5)
            self.critic_optimizer.step()

        with torch.no_grad():
            values = self.critic_net(state)
            next_values = self.critic_net(next_state)
            adv = self.compute_advatage(values, next_values, rewards, dones)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(4):
            mu, std = self.actor_net(state)
            new_log_probs = []
            entropy_coef = 0.01
            entropy = 0

            for i in range(actions.shape[1]):
                if i in [0, 1]:
                    dist = torch.distributions.Bernoulli(logits=mu[:,i])
                    log_prob = dist.log_prob(actions[:, i])
                else:
                    dist = Normal(mu[:, i], std[:, i])
                    raw_action = torch.atanh(actions[:, i].clamp(-0.999, 0.999))
                    log_prob = dist.log_prob(raw_action) - torch.log(1 - actions[:, i] ** 2 + 1e-6)

                new_log_probs.append(log_prob)
                entropy += dist.entropy().mean()

            new_log_probs = torch.stack(new_log_probs, dim=1)
            logp_new = new_log_probs.sum(dim=1, keepdim=True)
            logp_old = old_log_probs.sum(dim=1, keepdim=True)

            ratio = torch.exp(logp_new - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(),0.5)
            self.actor_optimizer.step()




