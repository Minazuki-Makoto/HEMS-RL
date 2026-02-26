from Env import env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

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
        log_std=torch.clamp(self.std_layer,-2,2)
        std=torch.exp(self.std_layer).expand_as(mu)
        return  mu, std

class Critic_Net(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Critic_Net,self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPO_Agent():
    def __init__(self,state_dim,hidden_dim,action_dim,eps,gamma=0.99):
        self.gamma = gamma
        self.eps = eps
        self.actor_net=Actor_Net(state_dim,hidden_dim,action_dim)
        self.critic_net=Critic_Net(state_dim,hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(),lr=1e-5)
        self.critic_optimizer=optim.Adam(self.critic_net.parameters(),lr=2e-5)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mu, std = self.actor_net(state)

        actions = []
        log_probs = []

        for i in range(mu.shape[1]):
            # ===== Bernoulli (MW, DIS) =====
            if i in [0, 1]:
                p = torch.sigmoid(mu[:, i])
                dist = torch.distributions.Bernoulli(p)
                a = dist.sample()
                log_prob = dist.log_prob(a)
            # ===== Normal (continuous) =====
            else:
                dist = Normal(mu[:, i], std[:, i])
                raw_a = dist.rsample()
                a = torch.tanh(raw_a)
                log_prob = dist.log_prob(raw_a) - torch.log(1 - a ** 2 + 1e-6)

            actions.append(a)
            log_probs.append(log_prob)

        actions = torch.stack(actions, dim=1)  # shape [1, act_dim]
        log_probs = torch.stack(log_probs, dim=1)  # shape [1, act_dim]

        return actions.squeeze(0).detach().cpu().numpy(), log_probs.squeeze(0).detach()


    def compute_advatage(self,values,next_values,reward,done,lamda=0.95):
        adv=torch.zeros_like(reward)
        lenth=reward.shape[0]
        res=0
        for i in range(lenth-1,-1,-1):
            delta=reward[i]+(1-done[i])*self.gamma*next_values[i]-values[i]
            res=self.gamma*lamda*res+delta
            adv[i]=res
        return adv

    def update(self, states, actions, rewards, next_states, dones, old_log_probs):
        states=np.array(states)
        next_states=np.array(next_states)
        rewards=np.array(rewards)
        dones=np.array(dones)
        actions=np.array(actions)
        state = torch.tensor(states, dtype=torch.float32)
        next_state = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_log_probs = torch.stack(old_log_probs).detach()  # shape [T, act_dim]

        for _ in range(10):
            # ===== critic =====
            values = self.critic_net(state)
            next_values = self.critic_net(next_state)
            td_target = rewards + self.gamma * next_values * (1 - dones)
            critic_loss = F.mse_loss(values, td_target.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ===== advantage =====
            adv = self.compute_advatage(values,next_values,rewards,dones)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv=adv.detach()
            # ===== actor =====
            mu, std = self.actor_net(state)
            new_log_probs = []

            for i in range(actions.shape[1]):
                # Bernoulli
                if i in [0, 1]:
                    p = torch.sigmoid(mu[:, i])
                    dist = torch.distributions.Bernoulli(p)
                    log_prob = dist.log_prob(actions[:,i])
                # Normal
                else:
                    dist = Normal(mu[:, i], std[:, i])
                    raw_action = torch.atanh(actions[:, i].clamp(-0.999, 0.999))
                    log_prob = dist.log_prob(raw_action) - torch.log(1 - actions[:, i] ** 2 + 1e-6)

                new_log_probs.append(log_prob)

            new_log_probs = torch.stack(new_log_probs, dim=1)  # shape [T, act_dim]
            logp_new=new_log_probs.sum(dim=1, keepdim=True)
            logp_old=old_log_probs.sum(dim=1, keepdim=True)
            # ⭐ PPO ratio：逐维计算，而不是 sum
            ratio = torch.exp(logp_new-logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()




