import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli,Normal
import random
from collections import deque
import copy

class ActNet(nn.Module):
    def __init__(self, state_dim,hidden_dim,action_dim):
        super(ActNet, self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc_mu=nn.Linear(hidden_dim,action_dim)
        self.fc_std = nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        mu=self.fc_mu(x)
        std=F.softplus(self.fc_std(x))
        return mu,std


class QNet(nn.Module):
    def __init__(self, state_dim,hidden_dim,action_dim):
        super(QNet, self).__init__()
        self.fc1=nn.Linear(state_dim+action_dim,hidden_dim)
        self.fc=nn.Linear(hidden_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,1)

    def forward(self,s,a):
        in_put=torch.cat((s,a),dim=-1)
        x=F.relu(self.fc1(in_put))
        x=F.relu(self.fc(x))
        return self.fc2(x)

class Buffer:
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)

    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def get_sample(self,batch_size):
        samples=random.sample(self.buffer,batch_size)
        states,actions,reward,next_states,dones=zip(*samples)
        return states,actions,reward,next_states,dones

    def __len__(self):
        return len(self.buffer)


class SAC_Agent:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, tau, alpha, buffer_size, batch_size):
        self.actor_net = ActNet(state_dim, hidden_dim, action_dim)

        self.Qnet1 = QNet(state_dim, hidden_dim, action_dim)
        self.Qnet2 = QNet(state_dim, hidden_dim, action_dim)

        self.Qtarget_net1 = copy.deepcopy(self.Qnet1)
        self.Qtarget_net2 = copy.deepcopy(self.Qnet2)

        self.tau = tau
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = Buffer(buffer_size)

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-4)
        self.Qnet1_optimizer = optim.Adam(self.Qnet1.parameters(), lr=2e-4)
        self.Qnet2_optimizer = optim.Adam(self.Qnet2.parameters(), lr=2e-4)

        continuous_dim = action_dim - 2
        discrete_dim = 2

        self.target_alpha = -(continuous_dim + 0.5 * discrete_dim)
        self.log_alpha=torch.tensor(np.log(alpha),dtype=torch.float32,requires_grad=True)
        self.log_alpha_optimizer=optim.Adam([self.log_alpha],lr=1e-4)

    def choose_action(self,state,flag=False):
        if not torch.is_tensor(state):
            state=torch.tensor(state,dtype=torch.float32)
        else:
            state=state
        if flag == False:
            state=state.unsqueeze(0)
        mu,std=self.actor_net(state)
        actions=[]
        log_probs=[]
        for i in range(mu.shape[1]):
            if i in [0,1]:
                p=torch.sigmoid(mu[:,i])
                dist=Bernoulli(p)
                action=dist.sample()
                log_prob=dist.log_prob(action)
            else:
                dist=Normal(mu[:,i],std[:,i])
                raw_action=dist.rsample()
                action=torch.tanh(raw_action)
                log_prob=dist.log_prob(raw_action)-torch.log(1-action.pow(2)+1e-6)
            actions.append(action)
            log_probs.append(log_prob)
        actions=torch.stack(actions,dim=1)
        log_probs=torch.stack(log_probs,dim=1)
        if flag==False:
            actions=actions.squeeze(0).detach().cpu().numpy()
            log_probs=log_probs.sum(dim=-1)
        else:
            actions=actions.cpu()
            log_probs=log_probs.sum(dim=-1,keepdim=True)
        return actions,log_probs

    def soft_update(self):
        for local_pama,target_pama in zip(self.Qnet1.parameters(),self.Qtarget_net1.parameters()):
            target_pama.data.copy_(local_pama*self.tau+target_pama*(1-self.tau))
        for local2_pama,target2_pama in zip(self.Qnet2.parameters(),self.Qtarget_net2.parameters()):
            target2_pama.data.copy_(local2_pama*self.tau+target2_pama*(1-self.tau))

    def update(self):
        if len(self.buffer)>4000:
            states,actions,reward,next_states,dones=self.buffer.get_sample(self.batch_size)
        else:
            return
        states=torch.tensor(np.array(states),dtype=torch.float32)
        actions=torch.tensor(np.array(actions),dtype=torch.float32)
        rewards=torch.tensor(np.array(reward),dtype=torch.float32).view(-1,1)
        next_states=torch.tensor(np.array(next_states),dtype=torch.float32)
        dones=torch.tensor(np.array(dones),dtype=torch.float32).view(-1,1)


        for _ in range(5):
            value1=self.Qnet1(states,actions)
            value2=self.Qnet2(states,actions)
            alpha = self.log_alpha.exp().detach()
            with torch.no_grad():
                next_actions,log_probs=self.choose_action(next_states,True)
                next_value1=self.Qtarget_net1(next_states,next_actions)
                next_value2=self.Qtarget_net2(next_states,next_actions)
                next_value=torch.min(next_value1,next_value2)
                td_target=rewards+self.gamma*(1-dones)*(next_value-alpha*log_probs)
            Q_loss1=F.mse_loss(value1,td_target)
            Q_loss2=F.mse_loss(value2,td_target)

            self.Qnet1_optimizer.zero_grad()
            Q_loss1.backward()
            self.Qnet1_optimizer.step()

            self.Qnet2_optimizer.zero_grad()
            Q_loss2.backward()
            self.Qnet2_optimizer.step()

        new_action,new_probs=self.choose_action(states,True)
        q1=self.Qnet1(states,new_action)
        q2=self.Qnet2(states,new_action)

        q=torch.min(q1,q2)

        alpha=self.log_alpha.exp().detach()
        loss=(alpha*new_probs-q).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        a_loss=(self.log_alpha*(-new_probs.detach()-self.target_alpha)).mean()
        self.log_alpha_optimizer.zero_grad()
        a_loss.backward()
        self.log_alpha_optimizer.step()




